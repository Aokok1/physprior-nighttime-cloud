"""Evaluate Phase 4 PhysPrior model per-station on Test set."""
import os, sys, json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
config.MTUNET_DATASET = "E:/Data/Unet_Dataset"
from config import MTUNET_DATASET, CHECKPOINT_DIR
from models.mt_unet import MT_UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
ckpt_path = os.path.join(CHECKPOINT_DIR, "phase4_physprior_moderate_best.pth")
print(f"Loading: {ckpt_path}")
model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                use_film=True, use_evidential=False).to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()

# Load test data
test_dir = os.path.join(MTUNET_DATASET, "Test")
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".npz")])

results = {"Changsha": [], "Longmen": []}

for fname in files:
    site = "Changsha" if "Changsha" in fname else "Longmen"
    d = np.load(os.path.join(test_dir, fname))
    
    dnb = d["X_dnb"].astype(np.float32)
    bm = d["X_basemap"].astype(np.float32)
    m15 = d.get("X_m15", d.get("X_mod", np.zeros_like(dnb))).astype(np.float32)
    
    # Normalize
    dnb_norm = np.log10(np.clip(dnb, 1e-10, None))
    dnb_norm = (dnb_norm - (-8.85)) / 0.54
    bm_norm = np.log10(np.clip(bm, 1e-10, None))
    bm_norm = (bm_norm - (-8.85)) / 0.54
    m15_norm = (m15 - 274.0) / 16.7
    
    image = np.stack([dnb_norm, bm_norm, m15_norm], axis=0)
    image_t = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    cl = float(d.get("Center_Label", float("nan")))
    moon = float(d.get("Moon_Phase", 180.0))
    sza = float(d.get("Solar_Zenith", 100.0))
    
    with torch.no_grad():
        pred_mask, _ = model(image_t, 
                           moon_phase=torch.tensor([moon], dtype=torch.float32).to(DEVICE),
                           solar_zenith=torch.tensor([sza], dtype=torch.float32).to(DEVICE))
        pred_cls = torch.argmax(pred_mask, dim=1)[0, 64, 64].item()
    
    pred_binary = 1 if pred_cls >= 2 else 0
    
    if not np.isnan(cl):
        results[site].append({
            "filename": fname,
            "radar": int(cl),
            "pred": pred_binary,
        })

# Compute per-station metrics
print("\n" + "=" * 60)
print(" Phase 4 PhysPrior — Per-Station Results")
print("=" * 60)

for site in ["Changsha", "Longmen"]:
    recs = results[site]
    if not recs:
        continue
    
    preds = np.array([r["pred"] for r in recs])
    labels = np.array([r["radar"] for r in recs])
    
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    
    n = len(recs)
    acc = (tp + tn) / n
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    spec = tn / max(tn + fp, 1)
    
    print(f"\n{site} (n={n}):")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Accuracy:    {acc:.1%}")
    print(f"  Precision:   {prec:.1%}")
    print(f"  Recall:      {rec:.1%}")
    print(f"  Specificity: {spec:.1%}")
    print(f"  F1:          {f1:.1%}")

# Combined
all_preds = []
all_labels = []
for site in ["Changsha", "Longmen"]:
    for r in results[site]:
        all_preds.append(r["pred"])
        all_labels.append(r["radar"])

preds = np.array(all_preds)
labels = np.array(all_labels)
tp = int(((preds == 1) & (labels == 1)).sum())
fp = int(((preds == 1) & (labels == 0)).sum())
tn = int(((preds == 0) & (labels == 0)).sum())
fn = int(((preds == 0) & (labels == 1)).sum())
n = len(preds)
acc = (tp + tn) / n
prec = tp / max(tp + fp, 1)
rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-8)
spec = tn / max(tn + fp, 1)

print(f"\n{'='*60}")
print(f"Combined (n={n}):")
print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"  Accuracy:    {acc:.1%}")
print(f"  Precision:   {prec:.1%}")
print(f"  Recall:      {rec:.1%}")
print(f"  Specificity: {spec:.1%}")
print(f"  F1:          {f1:.1%}")
