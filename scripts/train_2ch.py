"""Phase 2: Retrain core methods with 2-channel (DNB+M15) configuration.

Based on ablation finding: Basemap is harmful, DNB+M15 gives best performance.

Methods to retrain:
  1. Baseline (expected ~69%)
  2. Co-Teaching (expected improvement from 69.6%)
  3. PhysPrior moderate (expected 80-85%)
  4. Physics-Guided Loss moderate (retest after Basemap removal)
"""
import os
import sys
import json
import time
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET, CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR, train_cfg
from data import make_dataloaders
from models.mt_unet import MT_UNet, MT_Loss
from methods.baseline import EarlyStopping, compute_binary_metrics


def train_2ch_baseline(epochs=120, device=None):
    """Train baseline with DNB+M15 (no Basemap)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2-channel model, no FiLM (matches ablation setup)
    model = MT_UNet(in_channels=2, mask_classes=4, use_film=False).to(device)
    
    criterion = MT_Loss(
        weight_center=train_cfg.weight_center,
        dice_weight=train_cfg.dice_weight,
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                      weight_decay=train_cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg.T_0,
                                             T_mult=train_cfg.T_mult, eta_min=1e-6)
    
    # 2-channel dataloaders
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, 
                                                 batch_size=train_cfg.batch_size,
                                                 use_basemap=False)
    
    early_stop = EarlyStopping(patience=train_cfg.patience)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "2ch_baseline_best.pth")
    
    print("="*60)
    print("Training: 2-channel Baseline (DNB+M15)")
    print("="*60)
    
    best_acc = 0.0
    log_path = os.path.join(LOG_DIR, "2ch_baseline_train_log.txt")
    
    with open(log_path, "w") as log_f:
        log_f.write("epoch,train_loss,radar_acc,time\n")
        
        for epoch in range(epochs):
            # Train
            model.train()
            total_loss = 0
            for batch in train_loader:
                x = batch["image"].to(device)
                y_mask = batch["mask_noisy"].to(device)
                y_height = batch["height"].to(device)
                moon = batch["moon_phase"].to(device)
                sza = batch["solar_zenith"].to(device)
                radar_loc = batch["radar_loc"].to(device)
                center = batch["center_label"].to(device)
                
                optimizer.zero_grad()
                pred_mask, pred_hgt = model(x, moon_phase=moon, solar_zenith=sza)
                loss, _, _, _ = criterion(pred_mask, pred_hgt, y_mask, y_height,
                                          radar_loc, center)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            train_loss = total_loss / len(train_loader)
            
            # Validate every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch["image"].to(device)
                        center = batch["center_label"]
                        radar_loc = batch["radar_loc"]
                        moon = batch["moon_phase"].to(device)
                        sza = batch["solar_zenith"].to(device)
                        
                        pred_logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
                        pred_cls = torch.argmax(pred_logits, dim=1)
                        
                        for b in range(x.size(0)):
                            cl = center[b].item()
                            if cl != cl: continue
                            ry, rx = radar_loc[b, 0].item(), radar_loc[b, 1].item()
                            y0, y1 = max(0, ry-1), min(128, ry+2)
                            x0, x1 = max(0, rx-1), min(128, rx+2)
                            win = pred_cls[b, y0:y1, x0:x1]
                            pred_bin = 1 if (win >= 2).any() else 0
                            if pred_bin == int(cl):
                                correct += 1
                            total += 1
                
                acc = correct / max(total, 1)
                elapsed = time.time()
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                      f"Radar Acc: {acc:.1%}")
                log_f.write(f"{epoch+1},{train_loss:.4f},{acc:.4f},{elapsed:.0f}\n")
                
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), checkpoint_path)
                    early_stop.counter = 0
                else:
                    early_stop.step(train_loss)
                
                if early_stop.counter >= early_stop.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
    
    print(f"  Best accuracy: {best_acc:.1%}")
    return checkpoint_path, best_acc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline"],
                        help="Method to train")
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    
    if args.method == "baseline":
        train_2ch_baseline(epochs=args.epochs)


if __name__ == "__main__":
    main()
