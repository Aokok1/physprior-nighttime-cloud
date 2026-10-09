"""Co-teaching: two models train simultaneously, each filtering samples for the other.

Core idea (Han et al. 2018):
  - Two models with different learning trajectories
  - Each model selects "clean" samples (lowest loss) for the other
  - Only update on samples the peer model deemed clean
  - The forget rate controls how many "noisy" samples are dropped

Applied to cloud detection:
  - Two MT-UNet models, same architecture, different random initializations
  - For each batch, each model computes per-sample loss
  - The R% of samples with lowest loss are "clean" → used to train the OTHER model
  - R = 1 - forget_rate, forget_rate ramps up from 0 to target over warmup epochs
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import train_cfg, model_cfg, CHECKPOINT_DIR, LOG_DIR
from models.mt_unet import MT_UNet, MT_Loss
from methods.baseline import EarlyStopping, compute_binary_metrics


class CoTeachingTrainer:
    """Two-model co-teaching with per-sample loss filtering."""

    def __init__(self, train_loader, val_loader, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_a = None
        self.model_b = None
        self.optimizer_a = None
        self.optimizer_b = None
        self.criterion = None
        self.best_val_loss = float("inf")

        # Forget rate schedule
        self.forget_rate = train_cfg.coteach_forget_rate
        self.num_gradual = train_cfg.coteach_num_gradual

    def setup(self):
        self.model_a = MT_UNet(
            backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
        ).to(self.device)
        self.model_b = MT_UNet(
            backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
        ).to(self.device)

        self.criterion = MT_Loss(
            weight_center=train_cfg.weight_center,
            dice_weight=train_cfg.dice_weight,
        ).to(self.device)

        self.optimizer_a = AdamW(
            self.model_a.parameters(), lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.optimizer_b = AdamW(
            self.model_b.parameters(), lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler_a = CosineAnnealingWarmRestarts(
            self.optimizer_a, T_0=train_cfg.T_0, T_mult=train_cfg.T_mult,
            eta_min=1e-6,
        )
        self.scheduler_b = CosineAnnealingWarmRestarts(
            self.optimizer_b, T_0=train_cfg.T_0, T_mult=train_cfg.T_mult,
            eta_min=1e-6,
        )

    def _rate_schedule(self, epoch):
        """Ramp forget rate from 0 to target over num_gradual epochs."""
        if epoch < self.num_gradual:
            return self.forget_rate * epoch / self.num_gradual
        return self.forget_rate

    def _per_sample_loss(self, model, batch):
        """Compute per-sample mask loss (used for sample selection)."""
        x = batch["image"].to(self.device)
        y_mask = batch["mask_noisy"].to(self.device)
        y_height = batch["height"].to(self.device)
        moon = batch["moon_phase"].to(self.device)
        sza = batch["solar_zenith"].to(self.device)
        radar_loc = batch["radar_loc"].to(self.device)
        center = batch["center_label"].to(self.device)

        pred_mask, pred_hgt = model(x, moon_phase=moon, solar_zenith=sza)
        loss, _, _, _ = self.criterion(
            pred_mask, pred_hgt, y_mask, y_height, radar_loc, center
        )

        # For per-sample filtering, compute CE per sample
        ce = nn.functional.cross_entropy(
            pred_mask, y_mask, reduction="none"
        ).mean(dim=[1, 2])  # (B,)

        return loss, ce

    def train_epoch(self, epoch):
        self.model_a.train()
        self.model_b.train()
        rate = self._rate_schedule(epoch)
        sum_loss_a = 0.0
        sum_loss_b = 0.0

        for batch in self.train_loader:
            x = batch["image"].to(self.device)
            y_mask = batch["mask_noisy"].to(self.device)
            y_height = batch["height"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)
            B, H, W = y_mask.shape

            # ── Pixel-level CE for both models (no grad for selection) ──
            with torch.no_grad():
                pred_a, _ = self.model_a(x, moon_phase=moon, solar_zenith=sza)
                pred_b, _ = self.model_b(x, moon_phase=moon, solar_zenith=sza)

                # Per-pixel CE: (B, H, W)
                ce_a = F.cross_entropy(pred_a, y_mask, reduction="none")
                ce_b = F.cross_entropy(pred_b, y_mask, reduction="none")

                n_total = B * H * W
                n_keep = max(1, int(n_total * (1 - rate)))

                # Stratified selection: keep top-k pixels PER CLDMSK CLASS
                # Prevents discarding all boundary pixels (naturally higher CE)
                mask_a = torch.zeros(B, H, W, device=x.device)
                mask_b = torch.zeros(B, H, W, device=x.device)

                for c in range(4):
                    class_c = (y_mask == c)
                    n_class_c = class_c.sum().item()
                    if n_class_c == 0:
                        continue

                    # Per-class keep count (proportional OR equal)
                    n_keep_c = max(1, int(n_class_c * (1 - rate)))

                    # Model A selects lowest-CE pixels in class c for Model B
                    ce_a_c = ce_a[class_c]
                    _, idx_a_c = torch.topk(ce_a_c, min(n_keep_c, len(ce_a_c)),
                                            largest=False)
                    # Model B selects for Model A
                    ce_b_c = ce_b[class_c]
                    _, idx_b_c = torch.topk(ce_b_c, min(n_keep_c, len(ce_b_c)),
                                            largest=False)

                    # Get flat indices within class_c mask
                    class_indices = class_c.nonzero(as_tuple=False)  # (n_class_c, 3)
                    selected_a = class_indices[idx_a_c]  # indices B selected for A
                    selected_b = class_indices[idx_b_c]  # indices A selected for B

                    mask_a[selected_b[:, 0], selected_b[:, 1], selected_b[:, 2]] = 1.0
                    mask_b[selected_a[:, 0], selected_a[:, 1], selected_a[:, 2]] = 1.0

            # ── Train A on B-selected clean pixels ──
            self.optimizer_a.zero_grad()
            pred_a, pred_hgt_a = self.model_a(x, moon_phase=moon, solar_zenith=sza)
            # Masked CE: only compute loss on B-selected pixels
            ce_full = F.cross_entropy(pred_a, y_mask, reduction="none")
            loss_ce_masked = (ce_full * mask_a).sum() / max(mask_a.sum(), 1)

            # Radar center loss (on all valid radar pixels)
            loss_center = torch.tensor(0.0, device=x.device)
            valid_idx = ~torch.isnan(center)
            if valid_idx.sum() > 0:
                probs = F.softmax(pred_a, dim=1)
                p_cloud = (probs[:, 2, :, :] + probs[:, 3, :, :]).clamp(1e-6, 1 - 1e-6)
                v_locs = radar_loc[valid_idx]
                v_labels = center[valid_idx].float()
                loss_sum = 0.0
                for i in range(len(v_locs)):
                    ry, rx = v_locs[i, 0].item(), v_locs[i, 1].item()
                    y0, y1 = max(0, ry - 1), min(H, ry + 2)
                    x0, x1 = max(0, rx - 1), min(W, rx + 2)
                    win = p_cloud[valid_idx][i, y0:y1, x0:x1]
                    val = win.max() if v_labels[i] == 1.0 else win.mean()
                    loss_sum += F.binary_cross_entropy(val.unsqueeze(0), v_labels[i].unsqueeze(0))
                loss_center = loss_sum / len(v_locs)

            loss_a = loss_ce_masked + train_cfg.weight_center * loss_center
            loss_a.backward()
            nn.utils.clip_grad_norm_(self.model_a.parameters(), train_cfg.grad_clip)
            self.optimizer_a.step()
            sum_loss_a += loss_a.item()

            # ── Train B on A-selected clean pixels ──
            self.optimizer_b.zero_grad()
            pred_b, pred_hgt_b = self.model_b(x, moon_phase=moon, solar_zenith=sza)
            ce_full_b = F.cross_entropy(pred_b, y_mask, reduction="none")
            loss_ce_masked_b = (ce_full_b * mask_b).sum() / max(mask_b.sum(), 1)

            # Radar center loss for B
            loss_center_b = torch.tensor(0.0, device=x.device)
            if valid_idx.sum() > 0:
                probs_b = F.softmax(pred_b, dim=1)
                p_cloud_b = (probs_b[:, 2, :, :] + probs_b[:, 3, :, :]).clamp(1e-6, 1 - 1e-6)
                loss_sum_b = 0.0
                for i in range(len(v_locs)):
                    ry, rx = v_locs[i, 0].item(), v_locs[i, 1].item()
                    y0, y1 = max(0, ry - 1), min(H, ry + 2)
                    x0, x1 = max(0, rx - 1), min(W, rx + 2)
                    win = p_cloud_b[valid_idx][i, y0:y1, x0:x1]
                    val = win.max() if v_labels[i] == 1.0 else win.mean()
                    loss_sum_b += F.binary_cross_entropy(val.unsqueeze(0), v_labels[i].unsqueeze(0))
                loss_center_b = loss_sum_b / len(v_locs)

            loss_b = loss_ce_masked_b + train_cfg.weight_center * loss_center_b
            loss_b.backward()
            nn.utils.clip_grad_norm_(self.model_b.parameters(), train_cfg.grad_clip)
            self.optimizer_b.step()
            sum_loss_b += loss_b.item()

        self.scheduler_a.step(epoch - 1)
        self.scheduler_b.step(epoch - 1)
        n = len(self.train_loader)
        return sum_loss_a / n, sum_loss_b / n

    @torch.no_grad()
    def validate(self):
        """Validate using ensemble of both models (average logits)."""
        self.model_a.eval()
        self.model_b.eval()
        v_loss = 0.0
        correct = total_radar = 0
        tp_all = fp_all = fn_all = tn_all = 0

        for batch in self.val_loader:
            x = batch["image"].to(self.device)
            y_mask = batch["mask_noisy"].to(self.device)
            y_height = batch["height"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)

            pred_a, hgt_a = self.model_a(x, moon_phase=moon, solar_zenith=sza)
            pred_b, hgt_b = self.model_b(x, moon_phase=moon, solar_zenith=sza)
            pred_logits = (pred_a + pred_b) / 2.0  # ensemble
            pred_hgt = (hgt_a + hgt_b) / 2.0  # ensemble height

            # Loss on ensemble prediction
            loss, _, _, _ = self.criterion(
                pred_logits, pred_hgt, y_mask, y_height, radar_loc, center
            )
            v_loss += loss.item()

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

        n = len(self.val_loader)
        iou = tp_all / (tp_all + fp_all + fn_all + 1e-6)
        f1 = 2 * tp_all / (2 * tp_all + fp_all + fn_all + 1e-6)
        radar_acc = correct / (total_radar + 1e-8)

        return {
            "loss": v_loss / n, "iou": iou, "f1": f1,
            "radar_acc": radar_acc, "correct": correct, "total_radar": total_radar,
        }

    def run(self):
        if self.model_a is None:
            self.setup()

        early_stop = EarlyStopping(patience=train_cfg.patience)
        log_path = os.path.join(LOG_DIR, "coteaching_train_log.txt")

        with open(log_path, "w") as f:
            f.write("epoch,train_loss_a,train_loss_b,val_loss,iou,f1,radar_acc,forget_rate,time\n")

        print(f"\n{'='*50}")
        print(f"  Co-Teaching Training (forget_rate={self.forget_rate})")
        print(f"{'='*50}\n")

        for epoch in range(1, train_cfg.epochs + 1):
            t0 = time.time()
            rate = self._rate_schedule(epoch)
            loss_a, loss_b = self.train_epoch(epoch)
            metrics = self.validate()
            elapsed = time.time() - t0
            lr = self.optimizer_a.param_groups[0]["lr"]

            print(
                f"[{epoch:03d}] t={elapsed:.0f}s forget={rate:.2f} LR={lr:.1e} "
                f"LA={loss_a:.4f} LB={loss_b:.4f} Val={metrics['loss']:.4f} "
                f"IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f} "
                f"Radar={metrics['radar_acc']:.2%}"
            )

            with open(log_path, "a") as f:
                f.write(f"{epoch},{loss_a:.6f},{loss_b:.6f},{metrics['loss']:.6f},"
                        f"{metrics['iou']:.6f},{metrics['f1']:.6f},"
                        f"{metrics['radar_acc']:.6f},{rate:.4f},{elapsed:.1f}\n")

            if metrics["loss"] < self.best_val_loss:
                self.best_val_loss = metrics["loss"]
                torch.save(self.model_a.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "coteaching_a_best.pth"))
                torch.save(self.model_b.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "coteaching_b_best.pth"))

            if early_stop.step(metrics["loss"]):
                break

        print(f"\nDone. Best val_loss={self.best_val_loss:.4f}")
        return (self.model_a, self.model_b), self.best_val_loss
