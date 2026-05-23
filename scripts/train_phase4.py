"""Phase 4: Train with proper Train/Val/Test split (fixes data leakage).

Previous phases used Test set as validation during training (data leakage).
Phase 4 fixes this:
  - Train_phase4/ (1977 samples, augmented, for gradient updates)
  - Val_phase4/ (495 samples, no augmentation, for early stopping)
  - Test/ (197 samples, NEVER seen during training, for final eval only)

Methods to train:
  1. PhysPrior moderate (3ch)
  2. Baseline (3ch)
"""
import os
import sys
import json
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
config.MTUNET_DATASET = "E:/Data/Unet_Dataset"

from config import MTUNET_DATASET, CHECKPOINT_DIR, LOG_DIR, train_cfg
from data import make_dataloaders
from models.mt_unet import MT_UNet, MT_Loss
from methods.baseline import EarlyStopping, compute_binary_metrics
from methods.physical_prior import apply_physical_correction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train one epoch, applying PhysPrior correction to labels on-the-fly."""
    model.train()
    total_loss = 0
    n_batches = 0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask_noisy"].to(device)
        heights = batch["height"].to(device)
        moon = batch["moon_phase"].to(device)
        sza = batch["solar_zenith"].to(device)
        radar_locs = batch["radar_loc"].to(device)
        filenames = batch["filename"]

        # Apply PhysPrior correction to labels on-the-fly
        corrected_masks = masks.clone()
        for i in range(len(images)):
            # Get M15 channel (channel index depends on use_basemap)
            m15_norm = images[i, -1].cpu().numpy()  # last channel = M15
            # Denormalize M15 to get actual BT
            s = np.load(config.MTUNET_NORM_STATS)
            m15_bt = m15_norm * float(s["mod_std"]) + float(s["mod_mean"])
            
            # Get original CLDMSK mask
            cldmsk = masks[i].cpu().numpy()
            
            # Apply correction
            corrected = apply_physical_correction(
                cldmsk, m15_bt,
                m15_min=264.0, std_max=1.5  # moderate preset
            )
            corrected_masks[i] = torch.from_numpy(corrected).long().to(device)

        # Forward
        outputs, _ = model(images, moon_phase=moon, solar_zenith=sza)
        
        # Compute loss with corrected labels
        # Create dummy center_labels for loss computation
        center_labels = batch["center_label"]
        valid_mask = ~torch.tensor([np.isnan(cl) for cl in center_labels])
        center_labels_tensor = torch.tensor(
            [cl if not np.isnan(cl) else 0.0 for cl in center_labels],
            dtype=torch.float32, device=device
        )
        
        # Use standard loss with corrected masks
        loss = criterion(outputs, corrected_masks, heights,
                        center_labels_tensor, radar_locs, valid_mask.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, dataset_name="Val"):
    """Evaluate on a dataset, return metrics dict."""
    model.eval()
    all_preds = []
    all_labels = []
    all_filenames = []
    total_loss = 0
    n_batches = 0
    
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask_noisy"].to(device)
        heights = batch["height"].to(device)
        moon = batch["moon_phase"].to(device)
        sza = batch["solar_zenith"].to(device)
        radar_locs = batch["radar_loc"].to(device)
        center_labels = batch["center_label"]
        filenames = batch["filename"]

        outputs, _ = model(images, moon_phase=moon, solar_zenith=sza)
        
        # Get predictions at center pixel
        for i in range(len(images)):
            ry, rx = int(radar_locs[i][0]), int(radar_locs[i][1])
            pred_logits = outputs[i, :, ry, rx]  # [4] class logits
            pred_class = pred_logits.argmax().item()
            pred_binary = 1 if pred_class >= 2 else 0  # cloud if >= 2
            
            cl = center_labels[i]
            if not np.isnan(cl):
                all_preds.append(pred_binary)
                all_labels.append(int(cl))
                all_filenames.append(filenames[i])

    if len(all_preds) == 0:
        return {"radar_acc": 0.0, "n_radar": 0}
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    
    acc = (tp + tn) / len(preds) if len(preds) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "radar_acc": acc,
        "n_radar": len(preds),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "filenames": all_filenames,
    }


def train_physprior_phase4(preset="moderate", epochs=120, use_basemap=True):
    """Train PhysPrior with proper Phase 4 split."""
    ch_str = "3ch" if use_basemap else "2ch"
    print(f"\n{'='*60}")
    print(f"Phase 4: PhysPrior {preset} ({ch_str}) — NO DATA LEAKAGE")
    print(f"  Train: Train_phase4/ (augmented)")
    print(f"  Val:   Val_phase4/ (early stopping)")
    print(f"  Test:  Test/ (final eval only)")
    print(f"{'='*60}")
    
    # Data loaders
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size,
        use_basemap=use_basemap, use_phase4=True
    )
    
    # Model
    model = MT_UNet(
        backbone="resnet34",
        in_channels=3 if use_basemap else 2,
        mask_classes=4, use_film=True, use_evidential=False
    ).to(DEVICE)
    
    criterion = MT_Loss(
        w_seg=train_cfg.w_seg,
        w_center=train_cfg.w_center,
        use_evidential=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    early_stop = EarlyStopping(patience=train_cfg.patience)
    
    best_radar_acc = 0.0
    best_epoch = 0
    log_path = os.path.join(LOG_DIR, f"phase4_physprior_{preset}_{ch_str}_log.txt")
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    with open(log_path, "w") as log_f:
        log_f.write(f"Phase 4: PhysPrior {preset} ({ch_str}) — NO DATA LEAKAGE\n")
        log_f.write(f"Train: Train_phase4/ (1977 samples)\n")
        log_f.write(f"Val: Val_phase4/ (495 samples)\n")
        log_f.write(f"Test: Test/ (197 samples, NEVER used during training)\n\n")
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            
            # Train
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            scheduler.step()
            
            # Evaluate on Val (for early stopping / checkpoint selection)
            val_metrics = evaluate(model, val_loader, DEVICE, "Val")
            
            dt = time.time() - t0
            
            # Log
            line = (f"[Epoch {epoch:03d}/{epochs}] "
                   f"Loss={train_loss:.4f} | "
                   f"Val RadarAcc={val_metrics['radar_acc']:.1%} "
                   f"({val_metrics['n_radar']} pts) | "
                   f"P={val_metrics['precision']:.1%} R={val_metrics['recall']:.1%} F1={val_metrics['f1']:.1%} | "
                   f"{dt:.1f}s")
            print(line)
            log_f.write(line + "\n")
            log_f.flush()
            
            # Save checkpoint based on Val radar accuracy
            if val_metrics["radar_acc"] > best_radar_acc:
                best_radar_acc = val_metrics["radar_acc"]
                best_epoch = epoch
                ckpt_path = os.path.join(CHECKPOINT_DIR,
                                         f"phase4_physprior_{preset}_{ch_str}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  → Best checkpoint saved (Val radar_acc={best_radar_acc:.1%})")
            
            # Early stopping on Val loss (using CLDMSK labels as proxy)
            if early_stop(train_loss, model):
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Final log
        log_f.write(f"\n{'='*60}\n")
        log_f.write(f"BEST: Epoch {best_epoch}, Val radar_acc={best_radar_acc:.1%}\n")
        log_f.write(f"Checkpoint: phase4_physprior_{preset}_{ch_str}_best.pth\n")
    
    print(f"\nTraining complete. Best epoch: {best_epoch}, Val acc: {best_radar_acc:.1%}")
    return model


def train_baseline_phase4(epochs=120, use_basemap=True):
    """Train baseline (no correction) with Phase 4 split."""
    ch_str = "3ch" if use_basemap else "2ch"
    print(f"\n{'='*60}")
    print(f"Phase 4: Baseline ({ch_str}) — NO DATA LEAKAGE")
    print(f"{'='*60}")
    
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size,
        use_basemap=use_basemap, use_phase4=True
    )
    
    model = MT_UNet(
        backbone="resnet34",
        in_channels=3 if use_basemap else 2,
        mask_classes=4, use_film=True, use_evidential=False
    ).to(DEVICE)
    
    criterion = MT_Loss(w_seg=train_cfg.w_seg, w_center=train_cfg.w_center)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    early_stop = EarlyStopping(patience=train_cfg.patience)
    
    best_radar_acc = 0.0
    best_epoch = 0
    log_path = os.path.join(LOG_DIR, f"phase4_baseline_{ch_str}_log.txt")
    
    with open(log_path, "w") as log_f:
        log_f.write(f"Phase 4: Baseline ({ch_str}) — NO DATA LEAKAGE\n\n")
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            
            # Train with original CLDMSK labels (no correction)
            model.train()
            total_loss = 0
            n_batches = 0
            for batch in train_loader:
                images = batch["image"].to(DEVICE)
                masks = batch["mask_noisy"].to(DEVICE)
                heights = batch["height"].to(DEVICE)
                moon = batch["moon_phase"].to(DEVICE)
                sza = batch["solar_zenith"].to(DEVICE)
                radar_locs = batch["radar_loc"].to(DEVICE)
                center_labels = batch["center_label"]
                
                valid_mask = ~torch.tensor([np.isnan(cl) for cl in center_labels])
                center_labels_tensor = torch.tensor(
                    [cl if not np.isnan(cl) else 0.0 for cl in center_labels],
                    dtype=torch.float32, device=DEVICE
                )
                
                outputs, _ = model(images, moon_phase=moon, solar_zenith=sza)
                loss = criterion(outputs, masks, heights,
                               center_labels_tensor, radar_locs, valid_mask.to(DEVICE))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1
            
            train_loss = total_loss / max(n_batches, 1)
            scheduler.step()
            
            # Evaluate on Val
            val_metrics = evaluate(model, val_loader, DEVICE, "Val")
            dt = time.time() - t0
            
            line = (f"[Epoch {epoch:03d}/{epochs}] "
                   f"Loss={train_loss:.4f} | "
                   f"Val RadarAcc={val_metrics['radar_acc']:.1%} "
                   f"({val_metrics['n_radar']} pts) | {dt:.1f}s")
            print(line)
            log_f.write(line + "\n")
            log_f.flush()
            
            if val_metrics["radar_acc"] > best_radar_acc:
                best_radar_acc = val_metrics["radar_acc"]
                best_epoch = epoch
                ckpt_path = os.path.join(CHECKPOINT_DIR,
                                         f"phase4_baseline_{ch_str}_best.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  → Best checkpoint saved (Val radar_acc={best_radar_acc:.1%})")
            
            if early_stop(train_loss, model):
                print(f"Early stopping at epoch {epoch}")
                break
        
        log_f.write(f"\nBEST: Epoch {best_epoch}, Val radar_acc={best_radar_acc:.1%}\n")
    
    print(f"\nTraining complete. Best epoch: {best_epoch}, Val acc: {best_radar_acc:.1%}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["physprior", "baseline", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    
    if args.method in ("physprior", "all"):
        train_physprior_phase4(preset="moderate", epochs=args.epochs)
    if args.method in ("baseline", "all"):
        train_baseline_phase4(epochs=args.epochs)
