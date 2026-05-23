"""Phase 4: Train PhysPrior + Baseline with proper Train/Val/Test split.

Fixes critical data leakage: Test set is NEVER used during training.
  - Train: Train_phase4/ (1977 samples, augmented)
  - Val: Val_phase4/ (495 samples, for early stopping)
  - Test: Test/ (197 samples, for final eval only)
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
config.MTUNET_DATASET = "E:/Data/Unet_Dataset"
from config import MTUNET_DATASET, CHECKPOINT_DIR, LOG_DIR, train_cfg
from models.mt_unet import MT_UNet, MT_Loss
from methods.physical_prior import apply_physical_correction, compute_local_std

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Simple Dataset (no config dependency) ─────────────────────────
import glob, random
from torch.utils.data import Dataset, DataLoader

class SimpleCloudDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        self.augment = augment
        self.data_dir = data_dir
        
    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx):
        d = np.load(self.files[idx])
        dnb = d["X_dnb"].astype(np.float32)
        bm = d["X_basemap"].astype(np.float32)
        m15 = d.get("X_m15", d.get("X_mod", np.zeros_like(dnb))).astype(np.float32)
        y_mask = np.squeeze(d["Y_mask"]).copy().astype(np.int64)
        
        # Simple normalization (log10 for DNB, BT for M15)
        dnb_norm = np.log10(np.clip(dnb, 1e-10, None))
        dnb_norm = (dnb_norm - (-8.85)) / 0.54  # approximate stats
        bm_norm = np.log10(np.clip(bm, 1e-10, None))
        bm_norm = (bm_norm - (-8.85)) / 0.54
        m15_norm = (m15 - 274.0) / 16.7
        
        # Stack as 3ch [DNB, Basemap, M15]
        image = np.stack([dnb_norm, bm_norm, m15_norm], axis=0)
        
        # Augment (train only)
        if self.augment:
            if random.random() > 0.5:
                image = np.flip(image, axis=2).copy()
                y_mask = np.flip(y_mask, axis=1).copy()
            if random.random() > 0.5:
                image = np.flip(image, axis=1).copy()
                y_mask = np.flip(y_mask, axis=0).copy()
            if random.random() > 0.5:
                k = random.randint(0, 3)
                image = np.rot90(image, k, axes=(1, 2)).copy()
                y_mask = np.rot90(y_mask, k).copy()
        
        center_label = float(d.get("Center_Label", float("nan")))
        moon = float(d.get("Moon_Phase", 180.0))
        sza = float(d.get("Solar_Zenith", 100.0))
        
        return {
            "image": torch.from_numpy(image),
            "mask_noisy": torch.from_numpy(y_mask),
            "m15_raw": m15,  # for PhysPrior correction
            "center_label": center_label,
            "moon_phase": moon,
            "solar_zenith": sza,
            "filename": os.path.basename(self.files[idx]),
        }

# ── Training ───────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, apply_physprior=False):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        x = batch["image"].to(DEVICE)
        y_mask = batch["mask_noisy"].numpy()
        m15_raw = batch["m15_raw"].numpy()
        moon = torch.tensor(batch["moon_phase"], dtype=torch.float32).to(DEVICE)
        sza = torch.tensor(batch["solar_zenith"], dtype=torch.float32).to(DEVICE)
        
        # Apply PhysPrior correction if needed
        if apply_physprior:
            corrected = []
            for j in range(len(y_mask)):
                corr, _, _ = apply_physical_correction(
                    y_mask[j].astype(np.int32), m15_raw[j],
                    m15_min=264.0, std_max=1.5
                )
                corrected.append(corr)
            y_mask = np.stack(corrected)
        
        y_mask_t = torch.from_numpy(y_mask.astype(np.int64)).to(DEVICE)
        
        pred_mask, pred_hgt = model(x, moon_phase=moon, solar_zenith=sza)
        
        # Simplified loss: CE + Dice on mask
        # Flatten
        B, C, H, W = pred_mask.shape
        pred_flat = pred_mask.permute(0, 2, 3, 1).reshape(-1, C)
        target_flat = y_mask_t.reshape(-1)
        
        ce_loss = nn.CrossEntropyLoss()(pred_flat, target_flat)
        
        # Dice (binary cloud vs clear)
        pred_prob = torch.softmax(pred_mask, dim=1)
        cloud_prob = pred_prob[:, 2] + pred_prob[:, 3]  # prob_cloud + cloud
        cloud_target = (y_mask_t >= 2).float()
        intersection = (cloud_prob * cloud_target).sum()
        dice = 1 - (2 * intersection + 1) / (cloud_prob.sum() + cloud_target.sum() + 1)
        
        loss = ce_loss + 0.5 * dice
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_radar(model, loader):
    """Evaluate accuracy against radar labels."""
    model.eval()
    correct = total = 0
    tp = fp = fn = tn = 0
    
    for batch in loader:
        x = batch["image"].to(DEVICE)
        moon = torch.tensor(batch["moon_phase"], dtype=torch.float32).to(DEVICE)
        sza = torch.tensor(batch["solar_zenith"], dtype=torch.float32).to(DEVICE)
        center_labels = batch["center_label"]
        filenames = batch["filename"]
        
        pred_mask, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = torch.argmax(pred_mask, dim=1)  # [B, H, W]
        
        for j in range(len(center_labels)):
            cl = center_labels[j]
            if np.isnan(cl): continue
            
            # Center pixel prediction (binary: >=2 = cloud)
            pred_binary = 1 if pred_cls[j, 64, 64].item() >= 2 else 0
            label = int(cl)
            
            if pred_binary == label: correct += 1
            total += 1
            
            if label == 1 and pred_binary == 1: tp += 1
            elif label == 0 and pred_binary == 1: fp += 1
            elif label == 1 and pred_binary == 0: fn += 1
            else: tn += 1
    
    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    
    return {
        "radar_acc": acc, "n_radar": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1
    }


def run_experiment(method, epochs=120):
    """Run one training experiment."""
    print(f"\n{'='*60}")
    print(f"Phase 4: {method} — NO DATA LEAKAGE")
    print(f"{'='*60}")
    
    # Data
    train_dir = os.path.join(MTUNET_DATASET, "Train_phase4")
    val_dir = os.path.join(MTUNET_DATASET, "Val_phase4")
    test_dir = os.path.join(MTUNET_DATASET, "Test")
    
    train_ds = SimpleCloudDataset(train_dir, augment=True)
    val_ds = SimpleCloudDataset(val_dir, augment=False)
    test_ds = SimpleCloudDataset(test_dir, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Model
    model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                    use_film=True, use_evidential=False).to(DEVICE)
    criterion = MT_Loss(weight_center=5.0, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-6)
    
    apply_physprior = ("physprior" in method.lower())
    
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 20
    
    ckpt_name = f"phase4_{method}_best.pth"
    log_name = f"phase4_{method}_log.txt"
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    with open(os.path.join(LOG_DIR, log_name), "w") as log_f:
        log_f.write(f"Phase 4: {method} — NO DATA LEAKAGE\n")
        log_f.write(f"Train: Train_phase4/ ({len(train_ds)})\n")
        log_f.write(f"Val: Val_phase4/ ({len(val_ds)})\n")
        log_f.write(f"Test: Test/ ({len(test_ds)}, NEVER used during training)\n\n")
        
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                    apply_physprior=apply_physprior)
            scheduler.step()
            
            # Validate on Val (for checkpoint selection)
            val_metrics = evaluate_radar(model, val_loader)
            dt = time.time() - t0
            
            line = (f"[{epoch:03d}/{epochs}] "
                   f"Loss={train_loss:.4f} | "
                   f"Val Acc={val_metrics['radar_acc']:.1%} ({val_metrics['n_radar']}pts) | "
                   f"P={val_metrics['precision']:.1%} R={val_metrics['recall']:.1%} | "
                   f"{dt:.1f}s")
            print(line)
            log_f.write(line + "\n")
            log_f.flush()
            
            if val_metrics["radar_acc"] > best_val_acc:
                best_val_acc = val_metrics["radar_acc"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(),
                          os.path.join(CHECKPOINT_DIR, ckpt_name))
                print(f"  → Best (Val={best_val_acc:.1%})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stop at epoch {epoch}")
                    break
        
        log_f.write(f"\nBEST: Epoch {best_epoch}, Val acc={best_val_acc:.1%}\n")
    
    # Final evaluation on TEST set (never seen during training)
    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION ON TEST SET (never seen during training)")
    print(f"{'='*60}")
    
    model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, ckpt_name), map_location=DEVICE, weights_only=True))
    
    test_metrics = evaluate_radar(model, test_loader)
    print(f"Test Accuracy: {test_metrics['radar_acc']:.1%} ({test_metrics['n_radar']} samples)")
    print(f"  TP={test_metrics['tp']} FP={test_metrics['fp']} "
          f"TN={test_metrics['tn']} FN={test_metrics['fn']}")
    print(f"  Precision={test_metrics['precision']:.1%} "
          f"Recall={test_metrics['recall']:.1%} F1={test_metrics['f1']:.1%}")
    
    # Bootstrap CI
    from scipy.stats import binomtest
    ci = binomtest(test_metrics['tp'] + test_metrics['tn'],
                   test_metrics['n_radar']).proportion_ci()
    print(f"  95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]")
    
    # Save results
    results = {
        "method": method,
        "best_epoch": best_epoch,
        "val_acc": best_val_acc,
        "test_acc": test_metrics["radar_acc"],
        "test_ci": [float(ci[0]), float(ci[1])],
        "confusion_matrix": {
            "tp": test_metrics["tp"], "fp": test_metrics["fp"],
            "tn": test_metrics["tn"], "fn": test_metrics["fn"]
        },
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "data_leakage": False,
    }
    
    results_path = os.path.join(PROJECT_ROOT, "output", f"phase4_{method}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="all",
                       choices=["physprior", "baseline", "all"])
    parser.add_argument("--epochs", type=int, default=120)
    args = parser.parse_args()
    
    results = {}
    if args.method in ("physprior", "all"):
        results["physprior"] = run_experiment("physprior_moderate", args.epochs)
    if args.method in ("baseline", "all"):
        results["baseline"] = run_experiment("baseline", args.epochs)
    
    if len(results) == 2:
        print(f"\n{'='*60}")
        print(f"COMPARISON (Phase 4, no data leakage)")
        print(f"{'='*60}")
        for name, r in results.items():
            print(f"  {name:20s}: {r['test_acc']:.1%} [{r['test_ci'][0]:.1%}, {r['test_ci'][1]:.1%}]")
