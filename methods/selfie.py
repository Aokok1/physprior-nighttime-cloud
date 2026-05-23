"""SELFIE: SElf-Label FIltering with Evidence — teacher-student purification.

Three-phase framework:
  Phase 1 (Teacher): Train MT-UNet on all 2089 CLDMSK-labeled samples
                     with radar center loss as anchor. This is essentially
                     the current best MT-UNet model.
  Phase 2 (Purify):  Teacher predicts on ALL training samples.
                     For each pixel: if Teacher is high-confidence AND
                     disagrees with CLDMSK → replace CLDMSK label with
                     Teacher prediction.
  Phase 3 (Student): Train new model on the purified dataset.
                     Student should outperform Teacher because it trained
                     on corrected labels across ALL pixels.

Optional EDL mode: If Teacher uses Evidential Deep Learning head, we can
use prediction uncertainty (not just confidence) to decide which pixels
to relabel. High confidence + low uncertainty → safe to replace.
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from config import train_cfg, model_cfg, noise_cfg, CHECKPOINT_DIR, LOG_DIR, MTUNET_DATASET
from models.mt_unet import MT_UNet, MT_Loss
from data.dataset import CloudDataset
from methods.baseline import EarlyStopping, compute_binary_metrics


class SELFIE:
    """Teacher-Student label purification framework.

    Usage:
        selfie = SELFIE(train_loader, val_loader, mtunet_dataset_root)
        selfie.run_phase1()        # Train teacher
        purified = selfie.phase2()  # Purify labels
        student = selfie.phase3()   # Train student on purified labels
    """

    def __init__(self, train_loader, val_loader, dataset_root=MTUNET_DATASET,
                 device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.dataset_root = dataset_root
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.teacher = None
        self.teacher_b = None    # second teacher for dual mode
        self.student = None
        self.purified_labels = {}  # filename → corrected mask
        self.stats = {}

    # ── Phase 1: Train Teacher ──────────────────────────────────────
    def run_phase1(self, pretrained_checkpoint: str = None):
        """Train teacher model on CLDMSK + radar center loss.

        If pretrained_checkpoint is provided, load it instead of training.
        (e.g., the current MT-UNet best model)
        """
        if pretrained_checkpoint:
            print(f"\n[Phase 1] Loading pre-trained teacher from {pretrained_checkpoint}")
            self.teacher = MT_UNet(
                backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
                mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
                use_evidential=noise_cfg.selfie_teacher_mode == "edl",
            ).to(self.device)
            state = torch.load(pretrained_checkpoint, map_location=self.device,
                               weights_only=True)
            # Handle potentially mismatched keys (EDL head vs standard head)
            self.teacher.load_state_dict(state, strict=False)
            self.teacher.eval()
            print(f"  Teacher loaded. {sum(p.numel() for p in self.teacher.parameters()):,} params")
            return self.teacher

        print(f"\n[Phase 1] Training teacher (CLDMSK + radar center loss)")
        self.teacher = MT_UNet(
            backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
            use_evidential=False,
        ).to(self.device)

        criterion = MT_Loss(
            weight_center=train_cfg.weight_center,
            dice_weight=train_cfg.dice_weight,
        ).to(self.device)
        optimizer = AdamW(self.teacher.parameters(), lr=train_cfg.learning_rate,
                          weight_decay=train_cfg.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=train_cfg.T_0, T_mult=train_cfg.T_mult, eta_min=1e-6,
        )
        early_stop = EarlyStopping(patience=train_cfg.patience)

        for epoch in range(1, train_cfg.epochs + 1):
            self.teacher.train()
            train_loss = 0.0
            for batch in self.train_loader:
                x = batch["image"].to(self.device)
                y_mask = batch["mask_noisy"].to(self.device)
                y_height = batch["height"].to(self.device)
                moon = batch["moon_phase"].to(self.device)
                sza = batch["solar_zenith"].to(self.device)
                radar_loc = batch["radar_loc"].to(self.device)
                center = batch["center_label"].to(self.device)

                optimizer.zero_grad()
                pred_m, pred_h = self.teacher(x, moon_phase=moon, solar_zenith=sza)
                loss, _, _, _ = criterion(
                    pred_m, pred_h, y_mask, y_height, radar_loc, center
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.teacher.parameters(), train_cfg.grad_clip)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step(epoch - 1)
            avg_loss = train_loss / len(self.train_loader)

            if epoch % 10 == 0:
                print(f"  Teacher epoch {epoch:3d}  loss={avg_loss:.4f}")

            if early_stop.step(avg_loss):
                print(f"  Teacher early stop at epoch {epoch}")
                break

        torch.save(self.teacher.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "selfie_teacher.pth"))
        print(f"  Teacher training complete. Best loss={early_stop.best:.4f}")
        return self.teacher

    # ── Phase 2: Purify Labels ──────────────────────────────────────
    @torch.no_grad()
    def phase2(self, replace_ratio: float = None):
        """Use teacher to identify and correct CLDMSK errors.

        For each pixel in each training sample:
          - Teacher predicts class probabilities
          - If max probability > confidence_threshold AND
               teacher class != CLDMSK class → replace with teacher class
          - Limit replacements to replace_ratio * total_pixels per sample
        """
        if self.teacher is None:
            raise RuntimeError("Must run phase1() first")

        replace_ratio = replace_ratio or noise_cfg.selfie_replace_ratio
        conf_thresh = noise_cfg.selfie_confidence_threshold
        self.teacher.eval()

        print(f"\n[Phase 2] Purifying CLDMSK labels...")
        print(f"  Confidence threshold: {conf_thresh}")
        print(f"  Max replace ratio: {replace_ratio}")

        total_pixels = 0
        total_replaced = 0
        samples_modified = 0

        for mode in ["Train", "Test"]:
            ds = CloudDataset(self.dataset_root, mode, augment=False)
            for i in range(len(ds)):
                sample = ds[i]
                fname = sample["filename"]
                x = sample["image"].unsqueeze(0).to(self.device)  # (1, 3, 128, 128)
                cldmsk = sample["mask_noisy"].numpy()             # (128, 128)

                # Teacher prediction
                moon = torch.tensor([sample["moon_phase"]]).to(self.device)
                sza = torch.tensor([sample["solar_zenith"]]).to(self.device)
                pred, _ = self.teacher(x, moon_phase=moon, solar_zenith=sza)
                probs = F.softmax(pred, dim=1).squeeze(0).cpu().numpy()  # (4, 128, 128)
                teacher_cls = probs.argmax(axis=0)                       # (128, 128)
                teacher_conf = probs.max(axis=0)                         # (128, 128)

                # Identify pixels where teacher is confident AND disagrees
                disagree = (teacher_cls != cldmsk)
                high_conf = (teacher_conf > conf_thresh)
                fixable = disagree & high_conf

                # Cap replacements per sample
                n_fixable = fixable.sum()
                max_replace = int(replace_ratio * cldmsk.size)
                if n_fixable > max_replace:
                    # Keep only the highest-confidence disagreements
                    fix_scores = teacher_conf[fixable]
                    threshold_idx = int(max_replace)
                    if threshold_idx > 0:
                        score_thresh = np.partition(fix_scores, -threshold_idx)[-threshold_idx]
                        fixable = fixable & (teacher_conf >= score_thresh).astype(bool)

                # Apply corrections
                corrected = cldmsk.copy()
                corrected[fixable] = teacher_cls[fixable]

                self.purified_labels[fname] = corrected
                total_pixels += cldmsk.size
                total_replaced += fixable.sum()
                if fixable.sum() > 0:
                    samples_modified += 1

        self.stats["total_pixels"] = int(total_pixels)
        self.stats["total_replaced"] = int(total_replaced)
        self.stats["replace_fraction"] = total_replaced / max(total_pixels, 1)
        self.stats["samples_modified"] = samples_modified

        print(f"  Replaced {total_replaced:,} / {total_pixels:,} pixels "
              f"({self.stats['replace_fraction']:.2%})")
        print(f"  Modified {samples_modified} samples")
        return self.purified_labels

    # ── Phase 3: Train Student ──────────────────────────────────────
    def phase3(self):
        """Train student model on purified labels."""
        if not self.purified_labels:
            raise RuntimeError("Must run phase2() first")

        print(f"\n[Phase 3] Training student on purified labels")

        self.student = MT_UNet(
            backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
            use_evidential=False,
        ).to(self.device)

        criterion = MT_Loss(
            weight_center=train_cfg.weight_center,
            dice_weight=train_cfg.dice_weight,
        ).to(self.device)
        optimizer = AdamW(self.student.parameters(), lr=train_cfg.learning_rate,
                          weight_decay=train_cfg.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=train_cfg.T_0, T_mult=train_cfg.T_mult, eta_min=1e-6,
        )
        early_stop = EarlyStopping(patience=train_cfg.patience)

        best_loss = float("inf")

        for epoch in range(1, train_cfg.epochs + 1):
            self.student.train()
            train_loss = 0.0
            for batch in self.train_loader:
                x = batch["image"].to(self.device)
                y_height = batch["height"].to(self.device)
                moon = batch["moon_phase"].to(self.device)
                sza = batch["solar_zenith"].to(self.device)
                radar_loc = batch["radar_loc"].to(self.device)
                center = batch["center_label"].to(self.device)

                # Use purified labels instead of noisy CLDMSK
                filenames = batch["filename"]
                y_mask_purified = torch.stack([
                    torch.from_numpy(
                        self.purified_labels.get(fn,
                            batch["mask_noisy"][j].numpy())
                    )
                    for j, fn in enumerate(filenames)
                ]).to(self.device)

                optimizer.zero_grad()
                pred_m, pred_h = self.student(x, moon_phase=moon, solar_zenith=sza)
                loss, _, _, _ = criterion(
                    pred_m, pred_h, y_mask_purified, y_height, radar_loc, center
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self.student.parameters(), train_cfg.grad_clip)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step(epoch - 1)
            avg_loss = train_loss / len(self.train_loader)

            if epoch % 10 == 0:
                print(f"  Student epoch {epoch:3d}  loss={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.student.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "selfie_student_best.pth"))

            if early_stop.step(avg_loss):
                print(f"  Student early stop at epoch {epoch}")
                break

        print(f"  Student training complete. Best loss={best_loss:.4f}")
        return self.student

    @torch.no_grad()
    def evaluate(self, model=None, loader=None):
        """Evaluate any model on val loader (after phase3 for student)."""
        model = model or self.student
        loader = loader or self.val_loader
        if model is None:
            raise RuntimeError("No model to evaluate")

        model.eval()
        correct = total_radar = 0
        tp_all = fp_all = fn_all = tn_all = 0

        for batch in loader:
            x = batch["image"].to(self.device)
            y_mask = batch["mask_noisy"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)

            pred_logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
            pred_cls = torch.argmax(pred_logits, dim=1)

            tp, fp, fn, tn = compute_binary_metrics(pred_cls, y_mask)
            tp_all += tp; fp_all += fp; fn_all += fn; tn_all += tn

            for b in range(x.size(0)):
                cl = center[b].item()
                if cl != cl:
                    continue
                ry, rx = radar_loc[b, 0].item(), radar_loc[b, 1].item()
                y0, y1 = max(0, ry - 1), min(128, ry + 2)
                x0, x1 = max(0, rx - 1), min(128, rx + 2)
                win = pred_cls[b, y0:y1, x0:x1]
                bin_win = (win >= 2).int()
                pred_bin = 1 if bin_win.sum() > 0 else 0 if bin_win.float().mean() <= 0.5 else 1
                if pred_bin == int(cl):
                    correct += 1
                total_radar += 1

        iou = tp_all / (tp_all + fp_all + fn_all + 1e-6)
        f1 = 2 * tp_all / (2 * tp_all + fp_all + fn_all + 1e-6)
        radar_acc = correct / (total_radar + 1e-8)

        return {
            "iou": iou, "f1": f1,
            "radar_acc": radar_acc, "correct": correct, "total_radar": total_radar,
        }

    def run_full(self, teacher_ckpt: str = None):
        """Run all three phases and return comparison."""
        self.run_phase1(pretrained_checkpoint=teacher_ckpt)

        teacher_metrics = self.evaluate(self.teacher)
        print(f"\n  Teacher:  IoU={teacher_metrics['iou']:.4f}  "
              f"F1={teacher_metrics['f1']:.4f}  "
              f"Radar={teacher_metrics['radar_acc']:.2%}")

        self.phase2()
        self.phase3()

        student_metrics = self.evaluate(self.student)
        print(f"  Student:  IoU={student_metrics['iou']:.4f}  "
              f"F1={student_metrics['f1']:.4f}  "
              f"Radar={student_metrics['radar_acc']:.2%}")

        return {
            "teacher": teacher_metrics,
            "student": student_metrics,
            "purify_stats": self.stats,
        }

    # ── Dual-Teacher Mode (P0-2 fix: break circular reasoning) ──────
    def run_phase1_dual(self):
        """Train TWO teachers on different halves of the training data.

        Each teacher sees different CLDMSK noise patterns because the data
        halves cover different time periods. When both teachers agree with
        high confidence, it's more likely true signal than CLDMSK bias.
        """
        # Split train samples into two halves by index
        all_files = self.train_loader.dataset.file_list
        n = len(all_files)
        mid = n // 2

        # Create subset dataloaders
        from torch.utils.data import DataLoader, Subset

        indices_a = list(range(0, mid))
        indices_b = list(range(mid, n))

        print(f"\n[Phase 1-Dual] Training Teacher A on {len(indices_a)} samples")
        self.teacher = self._train_one_teacher(indices_a, "selfie_teacher_a.pth")

        print(f"\n[Phase 1-Dual] Training Teacher B on {len(indices_b)} samples")
        self.teacher_b = self._train_one_teacher(indices_b, "selfie_teacher_b.pth")

        print(f"  Dual teachers trained: A={sum(p.numel() for p in self.teacher.parameters()):,} params, "
              f"B={sum(p.numel() for p in self.teacher_b.parameters()):,} params")
        return self.teacher, self.teacher_b

    def _train_one_teacher(self, indices, save_name):
        """Train a single teacher on a subset of the training data."""
        from torch.utils.data import Subset

        sub_ds = Subset(self.train_loader.dataset, indices)
        sub_loader = DataLoader(sub_ds, batch_size=self.train_loader.batch_size,
                                shuffle=True, num_workers=0, pin_memory=True)

        model = MT_UNet(
            backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
            use_evidential=False,
        ).to(self.device)

        criterion = MT_Loss(
            weight_center=train_cfg.weight_center,
            dice_weight=train_cfg.dice_weight,
        ).to(self.device)
        optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                          weight_decay=train_cfg.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=train_cfg.T_0, T_mult=train_cfg.T_mult, eta_min=1e-6,
        )
        early_stop = EarlyStopping(patience=train_cfg.patience)

        for epoch in range(1, train_cfg.epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in sub_loader:
                x = batch["image"].to(self.device)
                y_mask = batch["mask_noisy"].to(self.device)
                y_height = batch["height"].to(self.device)
                moon = batch["moon_phase"].to(self.device)
                sza = batch["solar_zenith"].to(self.device)
                radar_loc = batch["radar_loc"].to(self.device)
                center = batch["center_label"].to(self.device)

                optimizer.zero_grad()
                pred_m, pred_h = model(x, moon_phase=moon, solar_zenith=sza)
                loss, _, _, _ = criterion(
                    pred_m, pred_h, y_mask, y_height, radar_loc, center
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step(epoch - 1)

            if early_stop.step(train_loss / len(sub_loader)):
                break

        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, save_name))
        model.eval()
        return model

    @torch.no_grad()
    def phase2_dual(self, replace_ratio: float = None,
                    dual_conf_thresh: float = 0.9):
        """Purify labels using DUAL-teacher consensus.

        Only replace a CLDMSK pixel when BOTH teachers:
          1. Predict the SAME class (consensus)
          2. Are high-confidence (>dual_conf_thresh)
          3. Disagree with CLDMSK (there's something to correct)

        This is more conservative than single-teacher — fewer replacements
        but much lower false-correction rate.
        """
        if self.teacher is None or self.teacher_b is None:
            raise RuntimeError("Must run phase1_dual() first")

        replace_ratio = replace_ratio or noise_cfg.selfie_replace_ratio
        self.teacher.eval()
        self.teacher_b.eval()

        print(f"\n[Phase 2-Dual] Purifying labels with dual-teacher consensus")
        print(f"  Consensus confidence threshold: {dual_conf_thresh}")
        print(f"  Max replace ratio: {replace_ratio}")

        # Build a combined dataset for full-scan (no augmentation, no shuffle)
        ds = CloudDataset(self.dataset_root, "Train", augment=False)
        ds_test = CloudDataset(self.dataset_root, "Test", augment=False)

        total_pixels = 0
        total_replaced = 0
        single_agree = 0  # one teacher would have replaced, but other vetoed
        stats = {"agree_replace": 0, "agree_keep": 0, "disagree_clr": 0}

        for dataset in [ds, ds_test]:
            for i in range(len(dataset)):
                sample = dataset[i]
                fname = sample["filename"]
                x = sample["image"].unsqueeze(0).to(self.device)
                cldmsk = sample["mask_noisy"].numpy()
                moon = torch.tensor([sample["moon_phase"]]).to(self.device)
                sza = torch.tensor([sample["solar_zenith"]]).to(self.device)

                # Both teachers predict
                pred_a, _ = self.teacher(x, moon_phase=moon, solar_zenith=sza)
                pred_b, _ = self.teacher_b(x, moon_phase=moon, solar_zenith=sza)

                probs_a = F.softmax(pred_a, dim=1).squeeze(0).cpu().numpy()
                probs_b = F.softmax(pred_b, dim=1).squeeze(0).cpu().numpy()

                cls_a = probs_a.argmax(axis=0)
                cls_b = probs_b.argmax(axis=0)
                conf_a = probs_a.max(axis=0)
                conf_b = probs_b.max(axis=0)

                # Consensus: same prediction + both high confidence
                consensus = (cls_a == cls_b)
                both_high = (conf_a > dual_conf_thresh) & (conf_b > dual_conf_thresh)
                disagree_cldmsk = (cls_a != cldmsk)

                fixable = consensus & both_high & disagree_cldmsk

                # Count single-teacher false positives (one would fix, other vetoes)
                single_fixable_a = (~consensus) & ((cls_a != cldmsk) & (conf_a > dual_conf_thresh))
                single_fixable_b = (~consensus) & ((cls_b != cldmsk) & (conf_b > dual_conf_thresh))
                single_agree += (single_fixable_a | single_fixable_b).sum()

                # Cap replacements
                n_fixable = fixable.sum()
                max_replace = int(replace_ratio * cldmsk.size)
                if n_fixable > max_replace:
                    fix_scores = np.minimum(conf_a, conf_b)  # use min conf for ranking
                    threshold_idx = int(max_replace)
                    if threshold_idx > 0:
                        score_thresh = np.partition(
                            fix_scores[fixable], -threshold_idx
                        )[-threshold_idx]
                        fixable = fixable & (np.minimum(conf_a, conf_b) >= score_thresh)

                # Apply corrections
                corrected = cldmsk.copy()
                corrected[fixable] = cls_a[fixable]  # both agree, so cls_a == cls_b
                self.purified_labels[fname] = corrected

                total_pixels += cldmsk.size
                total_replaced += fixable.sum()
                stats["agree_replace"] += fixable.sum()
                stats["agree_keep"] += (consensus & both_high & ~disagree_cldmsk).sum()
                stats["disagree_clr"] += (~consensus).sum()

        self.stats = {
            "total_pixels": int(total_pixels),
            "total_replaced": int(total_replaced),
            "replace_fraction": total_replaced / max(total_pixels, 1),
            "single_teacher_would_replace": int(single_agree),
            "vetoed_by_consensus": int(single_agree),
            "dual_purify_breakdown": stats,
        }

        print(f"  Replaced: {total_replaced:,} / {total_pixels:,} "
              f"({self.stats['replace_fraction']:.2%})")
        print(f"  Single-teacher would replace (vetoed by consensus): "
              f"{single_agree:,} pixels")
        print(f"  Consensus: {stats['agree_replace']:,} replace + "
              f"{stats['agree_keep']:,} keep | "
              f"Disagree: {stats['disagree_clr']:,}")

        # Warn if consensus replacing nothing
        if total_replaced == 0:
            print("  [WARNING] Zero pixels passed dual-consensus filter. "
                  "Consider lowering dual_conf_thresh (current={dual_conf_thresh}).")

        return self.purified_labels

    def run_full_dual(self):
        """Run dual-teacher SELFIE: Phase1-2-3 with consensus purification."""
        self.run_phase1_dual()

        teacher_a_metrics = self.evaluate(self.teacher)
        teacher_b_metrics = self.evaluate(self.teacher_b)
        print(f"\n  Teacher A: Radar={teacher_a_metrics['radar_acc']:.2%}")
        print(f"  Teacher B: Radar={teacher_b_metrics['radar_acc']:.2%}")

        self.phase2_dual()
        self.phase3()

        student_metrics = self.evaluate(self.student)
        print(f"\n  Student:  Radar={student_metrics['radar_acc']:.2%}")

        return {
            "teacher_a": teacher_a_metrics,
            "teacher_b": teacher_b_metrics,
            "student": student_metrics,
            "purify_stats": self.stats,
        }

    # ── PhysPrior-Guided Teacher (breaking circular reasoning) ──────
    def run_full_phys_teacher(self, phys_preset="moderate"):
        """SELFIE with PhysPrior-corrected labels for teacher training.

        Phase 0: Pre-correct CLDMSK labels using physical prior (M15+std).
                 This removes the systematic cold-surface bias BEFORE the
                 teacher sees it, breaking the circular reasoning.
        Phase 1: Train teacher on physically-corrected labels.
        Phase 2: Teacher purifies the remaining (non-physical) errors.
        Phase 3: Student trained on full purified labels.
        """
        print(f"\n{'='*50}")
        print(f"  SELFIE with PhysPrior Teacher ({phys_preset})")
        print(f"{'='*50}")

        # Phase 0: Physically correct labels
        from scipy.ndimage import uniform_filter
        from config import noise_cfg

        presets = {
            "conservative": {"m15_min": 266.0, "std_max": 1.0},
            "moderate":     {"m15_min": 264.0, "std_max": 1.5},
            "aggressive":   {"m15_min": 262.0, "std_max": 2.0},
        }
        p = presets[phys_preset]
        m15_min, std_max = p["m15_min"], p["std_max"]

        phys_labels = {}  # filename → physically-corrected mask
        total_flipped = 0
        total_pixels = 0

        print(f"\n[Phase 0] Applying physical correction (M15>={m15_min}K, "
              f"std<={std_max}K)...")

        ds = CloudDataset(self.dataset_root, "Train", augment=False)
        for i in range(len(ds)):
            sample = ds[i]
            fname = sample["filename"]
            cldmsk = sample["mask_noisy"].numpy().astype(np.int32)

            npz_path = os.path.join(self.dataset_root, "Train", fname)
            data = np.load(npz_path)
            m15_raw = data.get("X_m15", data.get("X_mod", None))
            if m15_raw is None:
                phys_labels[fname] = cldmsk
                continue

            m15_filled = np.where(np.isnan(m15_raw), np.nanmedian(m15_raw),
                                  m15_raw)
            sqr_mean = uniform_filter(m15_filled.astype(np.float64), 5)
            mean_sqr = uniform_filter(m15_filled.astype(np.float64) ** 2, 5)
            local_std = np.sqrt(np.maximum(mean_sqr - sqr_mean ** 2, 0))
            local_std = local_std.astype(np.float32)

            is_warm = m15_filled >= m15_min
            is_uniform = local_std <= std_max
            is_cloud = np.isin(cldmsk, [2, 3])

            corrected = cldmsk.copy()
            flip_mask = is_warm & is_uniform & is_cloud
            corrected[flip_mask] = 0  # Flip to True Clear
            phys_labels[fname] = corrected
            total_flipped += int(flip_mask.sum())
            total_pixels += cldmsk.size

        print(f"  Flipped {total_flipped:,}/{total_pixels:,} pixels "
              f"({total_flipped/total_pixels:.1%})")

        # Phase 1: Train teacher on physically-corrected labels
        print(f"\n[Phase 1-Phys] Training teacher on physically-corrected labels")
        self.teacher = MT_UNet(
            backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
            use_evidential=False,
        ).to(self.device)

        criterion = MT_Loss(
            weight_center=train_cfg.weight_center,
            dice_weight=train_cfg.dice_weight,
        ).to(self.device)
        optimizer = AdamW(self.teacher.parameters(), lr=train_cfg.learning_rate,
                          weight_decay=train_cfg.weight_decay)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=train_cfg.T_0, T_mult=train_cfg.T_mult,
            eta_min=1e-6,
        )
        early_stop = EarlyStopping(patience=train_cfg.patience)

        for epoch in range(1, train_cfg.epochs + 1):
            self.teacher.train()
            train_loss = 0.0
            for batch in self.train_loader:
                x = batch["image"].to(self.device)
                y_height = batch["height"].to(self.device)
                moon = batch["moon_phase"].to(self.device)
                sza = batch["solar_zenith"].to(self.device)
                radar_loc = batch["radar_loc"].to(self.device)
                center = batch["center_label"].to(self.device)
                filenames = batch["filename"]

                # Use physically-corrected labels
                y_mask_phys = torch.stack([
                    torch.from_numpy(phys_labels.get(
                        fn, batch["mask_noisy"][j].numpy()).astype(np.int64)
                    ).long()
                    for j, fn in enumerate(filenames)
                ]).to(self.device)

                optimizer.zero_grad()
                pred_m, pred_h = self.teacher(x, moon_phase=moon,
                                              solar_zenith=sza)
                loss, _, _, _ = criterion(
                    pred_m, pred_h, y_mask_phys, y_height, radar_loc, center)
                loss.backward()
                nn.utils.clip_grad_norm_(self.teacher.parameters(),
                                         train_cfg.grad_clip)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step(epoch - 1)
            avg_loss = train_loss / len(self.train_loader)

            if epoch % 10 == 0:
                print(f"  Teacher epoch {epoch:3d}  loss={avg_loss:.4f}")

            if early_stop.step(avg_loss):
                print(f"  Teacher early stop at epoch {epoch}")
                break

        torch.save(self.teacher.state_dict(),
                   os.path.join(CHECKPOINT_DIR, "selfie_phys_teacher.pth"))
        print(f"  PhysPrior-teacher training complete.")

        # Teacher evaluation
        teacher_metrics = self.evaluate(self.teacher)
        print(f"  PhysPrior-Teacher Radar={teacher_metrics['radar_acc']:.2%}")

        # Phase 2: Teacher purifies further
        self.phase2()

        # Phase 3: Student
        self.phase3()
        student_metrics = self.evaluate(self.student)
        print(f"  Student Radar={student_metrics['radar_acc']:.2%}")

        return {
            "teacher": teacher_metrics,
            "student": student_metrics,
            "purify_stats": self.stats,
        }
