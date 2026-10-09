"""Baseline trainer: standard training with CLDMSK as ground truth.

Reproduces the current MT-UNet training approach. Serves as the
reference point against which all noise-label methods are compared.
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import train_cfg, model_cfg, CHECKPOINT_DIR, LOG_DIR
from models.mt_unet import MT_UNet, MT_Loss


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def compute_binary_metrics(pred_classes, y_mask):
    """IoU/F1 components for binary cloud (classes 2,3 = cloud)."""
    pred_cloud = (pred_classes >= 2)
    true_cloud = (y_mask >= 2)
    tp = int((pred_cloud & true_cloud).sum())
    fp = int((pred_cloud & ~true_cloud).sum())
    fn = int((~pred_cloud & true_cloud).sum())
    tn = int((~pred_cloud & ~true_cloud).sum())
    return tp, fp, fn, tn


class BaselineTrainer:
    """Standard MT-UNet training — CLDMSK as ground truth + radar center loss."""

    def __init__(self, train_loader, val_loader, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.best_val_loss = float("inf")
        self.best_radar_acc = 0.0

    def setup(self):
        self.model = MT_UNet(
            backbone=model_cfg.backbone,
            in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes,
            use_film=model_cfg.use_film,
            use_evidential=False,
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

            # Radar accuracy (3×3 tolerance)
            for b in range(x.size(0)):
                cl = center[b].item()
                if cl != cl:  # NaN check
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
        """Full training loop."""
        if self.model is None:
            self.setup()

        early_stop = EarlyStopping(patience=train_cfg.patience)
        log_path = os.path.join(LOG_DIR, "baseline_train_log.txt")

        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,iou,f1,radar_acc,lr,time\n")

        print(f"\n{'='*50}")
        print(f"  Baseline Training — CLDMSK as Ground Truth")
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

            # Save checkpoint based on radar accuracy (primary metric)
            if not hasattr(self, 'best_radar_acc'):
                self.best_radar_acc = 0.0
            if metrics["radar_acc"] > self.best_radar_acc:
                self.best_radar_acc = metrics["radar_acc"]
                torch.save(self.model.state_dict(),
                           os.path.join(CHECKPOINT_DIR, "baseline_best.pth"))
                print(f"  [*] Best model saved (radar_acc={self.best_radar_acc:.2%})")

            if early_stop.step(metrics["loss"]):
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"\nDone. Best val_loss={self.best_val_loss:.4f}")
        return self.model, self.best_val_loss
