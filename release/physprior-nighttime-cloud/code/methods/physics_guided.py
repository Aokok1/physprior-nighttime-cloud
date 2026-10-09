"""Physics-Guided Loss trainer — soft, differentiable physical constraint.

Replaces hard label flipping (PhysicalPriorCorrector) with continuous
physics prior mask that penalizes cloud predictions in warm+uniform regions.
"""
import os, time, glob
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.ndimage import uniform_filter

from config import train_cfg, model_cfg, CHECKPOINT_DIR, LOG_DIR, MTUNET_DATASET, OUTPUT_DIR
from models.mt_unet import MT_UNet, PhysicsGuidedLoss
from methods.baseline import EarlyStopping, compute_binary_metrics


def compute_local_std(m15, window=5):
    sqr_mean = uniform_filter(m15.astype(np.float64), window)
    mean_sqr = uniform_filter(m15.astype(np.float64) ** 2, window)
    var = mean_sqr - sqr_mean ** 2
    return np.sqrt(np.maximum(var, 0)).astype(np.float32)


class PhysicsGuidedTrainer:
    def __init__(self, train_loader, val_loader, preset="moderate", device=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Preset parameters matching PhysicalPriorCorrector
        presets = {
            "conservative": {"T_min": 266.0, "S_max": 1.0, "T0": 2.0, "S0": 0.2},
            "moderate":     {"T_min": 264.0, "S_max": 1.5, "T0": 2.0, "S0": 0.2},
            "aggressive":   {"T_min": 262.0, "S_max": 2.0, "T0": 2.0, "S0": 0.2},
        }
        p = presets[preset]
        self.T_min, self.S_max, self.T0, self.S0 = \
            p["T_min"], p["S_max"], p["T0"], p["S0"]
        self.preset = preset
        self.model = None
        self.best_val_loss = float("inf")

    def setup(self):
        self.model = MT_UNet(
            backbone=model_cfg.backbone, in_channels=model_cfg.in_channels,
            mask_classes=model_cfg.mask_classes, use_film=model_cfg.use_film,
            use_evidential=False,
        ).to(self.device)

        self.criterion = PhysicsGuidedLoss(
            T_min=self.T_min, S_max=self.S_max, T0=self.T0, S0=self.S0,
            lambda_phys=1.0, weight_center=train_cfg.weight_center,
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
        sum_loss = sum_ce = sum_phys = 0.0

        for batch in self.train_loader:
            x = batch["image"].to(self.device)
            y_mask = batch["mask_noisy"].to(self.device)
            y_height = batch["height"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)
            filenames = batch["filename"]

            # Load M15 BT from .npz files and compute local std
            B, H, W = y_mask.shape
            m15_bt = np.zeros((B, H, W), dtype=np.float32)
            for j, fn in enumerate(filenames):
                # Determine mode from filename (Longmen prefix → Train)
                for mode in ["Train", "Test"]:
                    npz_path = os.path.join(MTUNET_DATASET, mode, fn)
                    if os.path.exists(npz_path):
                        break
                data = np.load(npz_path)
                m15_raw = data.get("X_m15", data.get("X_mod", None))
                if m15_raw is not None:
                    m15_bt[j] = np.where(np.isnan(m15_raw),
                                         np.nanmedian(m15_raw), m15_raw)
            m15_t = torch.from_numpy(m15_bt).float().to(self.device)

            # Compute local std on CPU (scipy) then to GPU
            local_std = np.zeros((B, H, W), dtype=np.float32)
            for j in range(B):
                local_std[j] = compute_local_std(m15_bt[j], window=5)
            std_t = torch.from_numpy(local_std).float().to(self.device)

            self.optimizer.zero_grad()
            pred_mask, pred_hgt = self.model(x, moon_phase=moon, solar_zenith=sza)
            loss, l_ce, l_phys, _ = self.criterion(
                pred_mask, y_mask, m15_t, std_t, radar_loc, center)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.grad_clip)
            self.optimizer.step()

            sum_loss += loss.item()
            sum_ce += l_ce.item()
            sum_phys += l_phys.item()

        self.scheduler.step(epoch - 1)
        n = len(self.train_loader)
        return sum_loss / n

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        v_loss = 0.0
        correct = total_radar = 0
        tp_all = fp_all = fn_all = tn_all = 0

        for batch in self.val_loader:
            x = batch["image"].to(self.device)
            y_mask = batch["mask_noisy"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)

            pred_logits, pred_hgt = self.model(x, moon_phase=moon, solar_zenith=sza)
            pred_cls = torch.argmax(pred_logits, dim=1)

            # Simple CE for validation (consistent comparison)
            v_loss += nn.functional.cross_entropy(pred_logits, y_mask).item()

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
        self.setup()
        early_stop = EarlyStopping(patience=train_cfg.patience)
        log_path = os.path.join(LOG_DIR, f"physguide_{self.preset}_log.txt")

        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,iou,f1,radar_acc,lr,time\n")

        print(f"\n{'='*50}")
        print(f"  Physics-Guided Loss ({self.preset})")
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
                torch.save(self.model.state_dict(),
                           os.path.join(CHECKPOINT_DIR, f"physguide_{self.preset}_best.pth"))

            if early_stop.step(metrics["loss"]):
                print(f"  Early stopping at epoch {epoch}")
                break

        print(f"\nDone. Best val_loss={self.best_val_loss:.4f}")
        return self.model, self.best_val_loss
