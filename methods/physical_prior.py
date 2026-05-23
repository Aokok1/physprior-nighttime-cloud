"""Physical Prior Correction — ablation experiment.

Based on three-expert analysis: CLDMSK false positives are concentrated in
cold (M15 BT elevated), spatially-uniform regions that physically correspond
to radiatively-cooled land surfaces, not clouds.

Three ablated variants:
  conservative:  M15>=266K & 5x5_std<=1.0K → flip cloud→clear
                 (captures ~42% FP, sacrifices ~23% TP)
  moderate:      M15>=264K & 5x5_std<=1.5K → flip cloud→clear
                 (captures ~53% FP, sacrifices ~33% TP)
  aggressive:    M15>=262K & 5x5_std<=2.0K → flip cloud→clear
                 (captures ~63% FP, sacrifices ~41% TP)

Each variant can run as:
  - Standalone: phys_only — train on physically-corrected labels
  - Hybrid:     phys + baseline/loss_correction/coteaching/selfie
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.ndimage import uniform_filter

from config import train_cfg, model_cfg, CHECKPOINT_DIR, LOG_DIR, MTUNET_DATASET
from models.mt_unet import MT_UNet, MT_Loss
from data.dataset import CloudDataset
from methods.baseline import EarlyStopping, compute_binary_metrics

# ── Physical Correction Presets ────────────────────────────────────────
PRESETS = {
    "conservative": {"m15_min": 266.0, "std_max": 1.0,
                     "desc": "M15>=266K & std<=1.0K (42% FP, 23% TP lost)"},
    "moderate":     {"m15_min": 264.0, "std_max": 1.5,
                     "desc": "M15>=264K & std<=1.5K (53% FP, 33% TP lost)"},
    "aggressive":   {"m15_min": 262.0, "std_max": 2.0,
                     "desc": "M15>=262K & std<=2.0K (63% FP, 41% TP lost)"},
}


def compute_local_std(m15: np.ndarray, window: int = 5) -> np.ndarray:
    """Compute per-pixel local standard deviation of M15 BT.

    Uses uniform_filter for speed (O(N) instead of O(N*w²)).
    """
    sqr_mean = uniform_filter(m15.astype(np.float64), window)
    mean_sqr = uniform_filter(m15.astype(np.float64) ** 2, window)
    var = mean_sqr - sqr_mean ** 2
    var = np.maximum(var, 0)
    return np.sqrt(var).astype(np.float32)


def apply_physical_correction(
    cldmsk: np.ndarray,
    m15: np.ndarray,
    m15_min: float = 266.0,
    std_max: float = 1.0,
    target_classes: tuple = (2, 3),  # CLDMSK classes to correct (prob_cloud, True Cloud)
    target_value: int = 0,             # flip to True Clear
) -> tuple:
    """Apply physics-driven label correction to a CLDMSK mask.

    For pixels where CLDMSK says cloud but M15 is warm-ish AND spatially
    uniform → likely cold surface misclassification → flip to clear.

    Returns (corrected_mask, n_corrected, correction_mask).
    """
    if np.isnan(m15).all():
        return cldmsk, 0, np.zeros_like(cldmsk, dtype=bool)

    # Fill NaN in M15 with median for std computation
    m15_filled = np.where(np.isnan(m15), np.nanmedian(m15), m15)

    local_std = compute_local_std(m15_filled, window=5)
    is_warm = m15_filled >= m15_min
    is_uniform = local_std <= std_max
    is_cloud_class = np.isin(cldmsk, target_classes)

    correction_mask = is_warm & is_uniform & is_cloud_class
    corrected = cldmsk.copy()
    corrected[correction_mask] = target_value

    return corrected, int(correction_mask.sum()), correction_mask


class PhysicalPriorCorrector:
    """Pre-process labels with physical constraints, then train normally.

    This is a LABEL PREPROCESSING step, not a training method.
    After correction, training uses standard MT_Loss (same as Baseline).
    """

    def __init__(self, train_loader, val_loader, preset="conservative",
                 device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.preset = preset
        params = PRESETS[preset]
        self.m15_min = params["m15_min"]
        self.std_max = params["std_max"]
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.best_val_loss = float("inf")
        self.correction_stats = {}

    def correct_dataset(self) -> dict:
        """Scan dataset, apply physical correction, store corrected labels.

        The corrected labels are saved as dict[filename → corrected_mask].
        During training, the dataloader's __getitem__ is monkey-patched to
        return corrected labels instead of raw CLDMSK.

        Returns correction statistics.
        """
        print(f"\n[Physical Prior] Preset: {self.preset}")
        print(f"  {PRESETS[self.preset]['desc']}")

        total_pixels = 0
        total_corrected = 0
        class_corrected = {c: 0 for c in range(4)}
        class_total = {c: 0 for c in range(4)}

        for mode in ["Train", "Test"]:
            ds = CloudDataset(MTUNET_DATASET, mode, augment=False)
            for i in range(len(ds)):
                sample = ds[i]
                fname = sample["filename"]
                cldmsk = sample["mask_noisy"].numpy().astype(np.int32)

                # Load M15 directly from npz (not through dataset normalization)
                npz_path = os.path.join(MTUNET_DATASET, mode, fname)
                data = np.load(npz_path)
                m15 = data.get("X_m15", data.get("X_mod", None))
                if m15 is None:
                    continue

                corrected, n_corr, corr_mask = apply_physical_correction(
                    cldmsk, m15,
                    m15_min=self.m15_min,
                    std_max=self.std_max,
                )

                # Track stats
                for c in range(4):
                    cls_mask = cldmsk == c
                    class_total[c] += int(cls_mask.sum())
                    class_corrected[c] += int((cls_mask & corr_mask).sum())

                total_pixels += cldmsk.size
                total_corrected += n_corr

        self.correction_stats = {
            "preset": self.preset,
            "m15_min": self.m15_min,
            "std_max": self.std_max,
            "total_pixels": total_pixels,
            "total_corrected": total_corrected,
            "corrected_fraction": total_corrected / max(total_pixels, 1),
            "per_class_corrected": class_corrected,
            "per_class_total": class_total,
        }

        print(f"  Corrected: {total_corrected:,} / {total_pixels:,} pixels "
              f"({self.correction_stats['corrected_fraction']:.2%})")
        for c, name in enumerate(["True Clear", "prob_clear", "prob_cloud", "True Cloud"]):
            if class_total[c] > 0:
                print(f"    {name}: {class_corrected[c]:,}/{class_total[c]:,} "
                      f"({class_corrected[c]/class_total[c]:.1%})")

        return self.correction_stats

    def setup(self):
        self.model = MT_UNet(
            backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
            use_evidential=False,
        ).to(self.device)

        self.criterion = MT_Loss(
            weight_center=train_cfg.weight_center,
            dice_weight=train_cfg.dice_weight,
        ).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(), lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=train_cfg.T_0, T_mult=train_cfg.T_mult,
            eta_min=1e-6,
        )

    def train_epoch(self, epoch):
        self.model.train()
        sum_loss = 0.0
        for batch in self.train_loader:
            x = batch["image"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)
            y_height = batch["height"].to(self.device)

            # Apply physical correction on-the-fly to this batch's labels
            y_mask_orig = batch["mask_noisy"].numpy()
            filenames = batch["filename"]
            corrected_batch = []
            for j, fn in enumerate(filenames):
                npz_path = os.path.join(MTUNET_DATASET, "Train", fn)
                data = np.load(npz_path)
                m15 = data.get("X_m15", data.get("X_mod", None))
                if m15 is not None:
                    corr, _, _ = apply_physical_correction(
                        y_mask_orig[j].astype(np.int32), m15,
                        m15_min=self.m15_min, std_max=self.std_max,
                    )
                else:
                    corr = y_mask_orig[j]
                corrected_batch.append(torch.from_numpy(corr).long())
            y_mask = torch.stack(corrected_batch).to(self.device)

            self.optimizer.zero_grad()
            pred_mask, pred_hgt = self.model(x, moon_phase=moon, solar_zenith=sza)
            loss, _, _, _ = self.criterion(
                pred_mask, pred_hgt, y_mask, y_height, radar_loc, center
            )
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.grad_clip)
            self.optimizer.step()
            sum_loss += loss.item()

        self.scheduler.step(epoch - 1)
        return sum_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        v_loss = 0.0
        correct = total_radar = 0
        tp_all = fp_all = fn_all = tn_all = 0

        for batch in self.val_loader:
            x = batch["image"].to(self.device)
            y_height = batch["height"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)

            # Use raw CLDMSK for validation (unbiased comparison)
            y_mask = batch["mask_noisy"].to(self.device)

            pred_logits, pred_hgt = self.model(x, moon_phase=moon, solar_zenith=sza)
            loss, _, _, _ = self.criterion(
                pred_logits, pred_hgt, y_mask, y_height, radar_loc, center
            )
            v_loss += loss.item()

            pred_cls = torch.argmax(pred_logits, dim=1)
            tp, fp, fn, tn = compute_binary_metrics(pred_cls, y_mask)
            tp_all += tp; fp_all += fp; fn_all += fn; tn_all += tn

            for b in range(x.size(0)):
                cl = center[b].item()
                if cl != cl: continue
                ry, rx = radar_loc[b, 0].item(), radar_loc[b, 1].item()
                y0, y1 = max(0, ry - 1), min(128, ry + 2)
                x0, x1 = max(0, rx - 1), min(128, rx + 2)
                win = pred_cls[b, y0:y1, x0:x1]
                bin_win = (win >= 2).int()
                pred_bin = 1 if bin_win.sum() > 0 else (0 if bin_win.float().mean() <= 0.5 else 1)
                if pred_bin == int(cl): correct += 1
                total_radar += 1

        n = len(self.val_loader)
        iou = tp_all / (tp_all + fp_all + fn_all + 1e-6)
        f1 = 2 * tp_all / (2 * tp_all + fp_all + fn_all + 1e-6)
        radar_acc = correct / (total_radar + 1e-8)
        return {"loss": v_loss / n, "iou": iou, "f1": f1,
                "radar_acc": radar_acc, "correct": correct, "total_radar": total_radar}

    def run(self):
        self.correct_dataset()
        self.setup()

        early_stop = EarlyStopping(patience=train_cfg.patience)
        log_path = os.path.join(LOG_DIR, f"physprior_{self.preset}_train_log.txt")

        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,iou,f1,radar_acc,lr,time\n")

        print(f"\n{'='*50}")
        print(f"  Physical Prior ({self.preset}) — Training")
        print(f"{'='*50}\n")

        for epoch in range(1, train_cfg.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            metrics = self.validate()
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            elapsed_str = f"{elapsed:.0f}s"
            print(f"[{epoch:03d}] t={elapsed_str} LR={lr:.1e} "
                  f"Train={train_loss:.4f} Val={metrics['loss']:.4f} "
                  f"IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f} "
                  f"Radar={metrics['radar_acc']:.2%}")

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{metrics['loss']:.6f},"
                        f"{metrics['iou']:.6f},{metrics['f1']:.6f},"
                        f"{metrics['radar_acc']:.6f},{lr:.8f},{elapsed:.1f}\n")

            if metrics["loss"] < self.best_val_loss:
                self.best_val_loss = metrics["loss"]

            # Save checkpoint based on radar accuracy (primary metric)
            if not hasattr(self, 'best_radar_acc'):
                self.best_radar_acc = 0.0
            if metrics["radar_acc"] > self.best_radar_acc:
                self.best_radar_acc = metrics["radar_acc"]
                torch.save(self.model.state_dict(),
                           os.path.join(CHECKPOINT_DIR,
                                        f"physprior_{self.preset}_best.pth"))

            if early_stop.step(metrics["loss"]):
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"\nDone. Best val_loss={self.best_val_loss:.4f}")
        return self.model, self.best_val_loss, self.correction_stats
