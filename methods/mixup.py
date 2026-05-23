"""Mixup baseline: noise-robust regularization via sample interpolation.

Mixup (Zhang et al. ICLR 2018) trains on convex combinations of sample pairs:
  x_mix = λ * x_i + (1-λ) * x_j
  y_mix = λ * y_i + (1-λ) * y_j

For noisy labels, Mixup acts as implicit label smoothing — when a noisy sample
is mixed with a clean one, the interpolated label reduces the noise impact.

Key advantage: no noise rate estimation needed, zero hyperparameter overhead.
"""
import os
import time
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import train_cfg, model_cfg, CHECKPOINT_DIR, LOG_DIR
from models.mt_unet import MT_UNet, MT_Loss
from methods.baseline import EarlyStopping, compute_binary_metrics


class MixupTrainer:
    """MT-UNet training with Mixup augmentation for noise robustness."""

    def __init__(self, train_loader, val_loader, alpha=0.4, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.alpha = alpha  # Beta distribution parameter
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.best_val_loss = float("inf")

    def setup(self):
        self.model = MT_UNet(
            backbone=model_cfg.backbone,
            in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes,
            use_film=model_cfg.use_film,
        ).to(self.device)

        self.criterion = MT_Loss(
            weight_center=train_cfg.weight_center,
            dice_weight=train_cfg.dice_weight,
        ).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=train_cfg.T_0, T_mult=train_cfg.T_mult,
            eta_min=1e-6,
        )

    def _mixup_data(self, x, y_mask, y_height):
        """Apply Mixup to image, mask, and height tensors."""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        # Random permutation for pairing
        index = torch.randperm(batch_size, device=x.device)

        x_mix = lam * x + (1 - lam) * x[index]
        # For segmentation masks: use soft labels (probability maps)
        y_mask_onehot = torch.zeros(
            batch_size, model_cfg.mask_classes, *y_mask.shape[1:],
            device=y_mask.device, dtype=torch.float32
        ).scatter_(1, y_mask.unsqueeze(1), 1.0)
        y_mask_soft = lam * y_mask_onehot + (1 - lam) * y_mask_onehot[index]

        # Height: linear interpolation
        y_height_mix = lam * y_height + (1 - lam) * y_height[index]

        return x_mix, y_mask_soft, y_height_mix, lam, index

    def train_epoch(self, epoch):
        self.model.train()
        sum_loss = 0.0
        for batch in self.train_loader:
            x = batch["image"].to(self.device)
            y_mask = batch["mask_noisy"].to(self.device)
            y_height = batch["height"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)

            # Apply Mixup
            x_mix, y_mask_soft, y_height_mix, lam, idx = self._mixup_data(
                x, y_mask, y_height
            )

            # Mix radar_loc and center: use original (not mixed) for radar loss
            # Radar positions don't make physical sense when mixed

            self.optimizer.zero_grad()
            pred_mask, pred_hgt = self.model(x_mix, moon_phase=moon, solar_zenith=sza)

            # Soft-label CE loss (cross-entropy with probability targets)
            log_probs = torch.log_softmax(pred_mask, dim=1)
            loss_ce = -(y_mask_soft * log_probs).sum(dim=1).mean()

            # Dice loss on hard labels (use original, not mixed)
            from models.mt_unet import DiceLoss
            dice_fn = DiceLoss()
            loss_dice = dice_fn(pred_mask, y_mask)

            # Height loss
            valid = (y_height_mix != -1.0)
            if valid.sum() > 0:
                loss_height = nn.functional.mse_loss(
                    pred_hgt[valid], y_height_mix[valid]
                )
            else:
                loss_height = torch.tensor(0.0, device=x.device)

            # Radar center loss (on original samples only)
            loss_center = torch.tensor(0.0, device=x.device)
            valid_idx = ~torch.isnan(center)
            if valid_idx.sum() > 0:
                probs = torch.softmax(pred_mask[:x.size(0)], dim=1)
                p_cloud = (probs[:, 2, :, :] + probs[:, 3, :, :]).clamp(1e-6, 1 - 1e-6)
                v_locs = radar_loc[valid_idx]
                v_labels = center[valid_idx].float()
                H, W = y_mask.shape[1], y_mask.shape[2]
                loss_sum = 0.0
                for i in range(len(v_locs)):
                    ry, rx = v_locs[i, 0].item(), v_locs[i, 1].item()
                    y0, y1 = max(0, ry - 1), min(H, ry + 2)
                    x0, x1 = max(0, rx - 1), min(W, rx + 2)
                    win = p_cloud[valid_idx][i, y0:y1, x0:x1]
                    val = win.max() if v_labels[i] == 1.0 else win.mean()
                    loss_sum += nn.functional.binary_cross_entropy(
                        val.unsqueeze(0), v_labels[i].unsqueeze(0)
                    )
                loss_center = loss_sum / len(v_locs)

            dice_w = train_cfg.dice_weight
            loss = (1 - dice_w) * loss_ce + dice_w * loss_dice + \
                   loss_height + train_cfg.weight_center * loss_center

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
            y_mask = batch["mask_noisy"].to(self.device)
            y_height = batch["height"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)

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
        if self.model is None:
            self.setup()

        early_stop = EarlyStopping(patience=train_cfg.patience)
        log_path = os.path.join(LOG_DIR, "mixup_train_log.txt")

        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,iou,f1,radar_acc,lr,time\n")

        print(f"\n{'='*50}")
        print(f"  Mixup Training (alpha={self.alpha})")
        print(f"{'='*50}\n")

        for epoch in range(1, train_cfg.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            metrics = self.validate()
            elapsed = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"[{epoch:03d}] t={elapsed:.0f}s LR={lr:.1e} "
                f"Train={train_loss:.4f} Val={metrics['loss']:.4f} "
                f"IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f} "
                f"Radar={metrics['radar_acc']:.2%} "
                f"({metrics['correct']}/{metrics['total_radar']})"
            )

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{metrics['loss']:.6f},"
                        f"{metrics['iou']:.6f},{metrics['f1']:.6f},"
                        f"{metrics['radar_acc']:.6f},{lr:.8f},{elapsed:.1f}\n")

            if metrics["loss"] < self.best_val_loss:
                self.best_val_loss = metrics["loss"]
                torch.save(self.model.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "mixup_best.pth"))
                print(f"  [*] Best model saved (val_loss={self.best_val_loss:.4f})")

            if early_stop.step(metrics["loss"]):
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"\nDone. Best val_loss={self.best_val_loss:.4f}")
        return self.model, self.best_val_loss
