"""GCE (Generalized Cross Entropy) baseline — noise-robust loss function.

GCE interpolates between CE (q→0, noise-sensitive but fast convergence)
and MAE (q→1, noise-robust but slow convergence). q=0.7 is recommended.

Reference: Zhang & Sabuncu (NeurIPS 2018) "Generalized Cross Entropy Loss
for Training Deep Neural Networks with Noisy Labels"
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import train_cfg, model_cfg, CHECKPOINT_DIR, LOG_DIR
from models.mt_unet import MT_UNet, DiceLoss
from methods.baseline import EarlyStopping, compute_binary_metrics


class GCELoss(nn.Module):
    """L_gce = (1 - p_y^q) / q. q=0 → CE, q=1 → MAE."""

    def __init__(self, num_classes=4, q=0.7):
        super().__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, self.num_classes)
        targets_oh = targets_oh.permute(0, 3, 1, 2).float()
        p_y = (probs * targets_oh).sum(dim=1)  # (B, H, W)
        p_y = p_y.clamp(1e-6, 1.0)
        loss = (1.0 - p_y ** self.q) / self.q
        return loss.mean()


class GCEBaseline:
    """Standard MT-UNet training with GCE loss instead of CE+Dice."""

    def __init__(self, train_loader, val_loader, q=0.7, device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.q = q
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.best_val_loss = float("inf")

    def setup(self):
        self.model = MT_UNet(
            backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
            use_evidential=False,
        ).to(self.device)

        self.criterion_gce = GCELoss(num_classes=4, q=self.q)
        self.criterion_height = nn.MSELoss(reduction="none")
        self.criterion_center = nn.BCELoss()

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
            y_mask = batch["mask_noisy"].to(self.device)
            y_height = batch["height"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)

            self.optimizer.zero_grad()
            pred_mask, pred_hgt = self.model(x, moon_phase=moon, solar_zenith=sza)

            # GCE on mask
            loss_mask = self.criterion_gce(pred_mask, y_mask)

            # Height regression (unchanged)
            valid = (y_height != -1.0)
            loss_hgt = self.criterion_height(pred_hgt, y_height)[valid].mean() \
                if valid.sum() > 0 else torch.tensor(0.0, device=x.device)

            # Radar center loss
            loss_center = torch.tensor(0.0, device=x.device)
            valid_idx = ~torch.isnan(center)
            if valid_idx.sum() > 0:
                probs = F.softmax(pred_mask, dim=1)
                p_cloud = (probs[:, 2, :, :] + probs[:, 3, :, :]).clamp(1e-6, 1 - 1e-6)
                v_locs = radar_loc[valid_idx]
                v_labels = center[valid_idx].float()
                loss_sum = 0.0
                for i in range(len(v_locs)):
                    ry, rx = v_locs[i, 0].item(), v_locs[i, 1].item()
                    y0, y1 = max(0, ry - 1), min(128, ry + 2)
                    x0, x1 = max(0, rx - 1), min(128, rx + 2)
                    win = p_cloud[valid_idx][i, y0:y1, x0:x1]
                    val = win.max() if v_labels[i] == 1.0 else win.mean()
                    loss_sum += self.criterion_center(
                        val.unsqueeze(0), v_labels[i].unsqueeze(0)
                    )
                loss_center = loss_sum / len(v_locs)

            loss = loss_mask + loss_hgt + train_cfg.weight_center * loss_center
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
            loss_mask = self.criterion_gce(pred_logits, y_mask)
            v_loss += loss_mask.item()

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

        return {"loss": v_loss / n, "iou": iou, "f1": f1,
                "radar_acc": radar_acc, "correct": correct, "total_radar": total_radar}

    def run(self):
        if self.model is None:
            self.setup()

        early_stop = EarlyStopping(patience=train_cfg.patience)
        log_path = os.path.join(LOG_DIR, f"gce_q{self.q}_train_log.txt")

        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,iou,f1,radar_acc,lr,time\n")

        print(f"\n{'='*50}")
        print(f"  GCE Baseline (q={self.q})")
        print(f"{'='*50}\n")

        for epoch in range(1, train_cfg.epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(epoch)
            metrics = self.validate()
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            print(f"[{epoch:03d}] t={elapsed:.0f}s LR={lr:.1e} "
                  f"Train={train_loss:.4f} Val={metrics['loss']:.4f} "
                  f"IoU={metrics['iou']:.4f} F1={metrics['f1']:.4f} "
                  f"Radar={metrics['radar_acc']:.2%}")

            with open(log_path, "a") as f:
                f.write(f"{epoch},{train_loss:.6f},{metrics['loss']:.6f},"
                        f"{metrics['iou']:.6f},{metrics['f1']:.6f},"
                        f"{metrics['radar_acc']:.6f},{lr:.8f},{elapsed:.1f}\n")

            if metrics["loss"] < self.best_val_loss:
                self.best_val_loss = metrics["loss"]
                ckpt_name = f"gce_q{self.q}_best.pth"
                torch.save(self.model.state_dict(),
                           os.path.join(CHECKPOINT_DIR, ckpt_name))

            if early_stop.step(metrics["loss"]):
                break

        print(f"\nDone. Best val_loss={self.best_val_loss:.4f}")
        return self.model, self.best_val_loss
