"""Phase 4 complete verification: compute ALL numbers for the paper."""
import os, sys, json, math
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
config.MTUNET_DATASET = "E:/Data/Unet_Dataset"
from config import MTUNET_DATASET, CHECKPOINT_DIR
from models.mt_unet import MT_UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Phase 4 model
ckpt_path = os.path.join(CHECKPOINT_DIR, "phase4_physprior_moderate_best.pth")
model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                use_film=True, use_evidential=False).to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()

# Load test data
test_dir = os.path.join(MTUNET_DATASET, "Test")
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".npz")])

records = []

for fname in files:
    site = "Changsha" if "Changsha" in fname else "Longmen"
    d = np.load(os.path.join(test_dir, fname))
    
    dnb = d["X_dnb"].astype(np.float32)
    bm = d["X_basemap"].astype(np.float32)
    m15 = d.get("X_m15", d.get("X_mod", np.zeros_like(dnb))).astype(np.float32)
    
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
    
    # Extract season from julian day
    parts = fname.split("_A")[1] if "_A" in fname else ""
    julian = int(parts[:3]) if len(parts) >= 3 else 0
    month = min((julian - 1) // 30 + 1, 12)
    if month in [12, 1, 2]: season = "winter"
    elif month in [3, 4, 5]: season = "spring"
    elif month in [6, 7, 8]: season = "summer"
    else: season = "autumn"
    
    with torch.no_grad():
        pred_mask, _ = model(image_t,
                           moon_phase=torch.tensor([moon], dtype=torch.float32).to(DEVICE),
                           solar_zenith=torch.tensor([sza], dtype=torch.float32).to(DEVICE))
        pred_cls = torch.argmax(pred_mask, dim=1)[0, 64, 64].item()
    
    pred_binary = 1 if pred_cls >= 2 else 0
    
    if not np.isnan(cl):
        records.append({
            "filename": fname,
            "site": site,
            "season": season,
            "radar": int(cl),
            "pred": pred_binary,
        })

# ── Compute metrics ──
def compute_metrics(recs):
    if not recs: return {"n": 0}
    preds = np.array([r["pred"] for r in recs])
    labels = np.array([r["radar"] for r in recs])
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    n = len(recs)
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    spec = tn / max(tn + fp, 1)
    balanced_acc = (rec + spec) / 2
    # MCC
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn - fp*fn) / max(denom, 1e-8)
    # Binomial CI
    from scipy.stats import binomtest
    ci = binomtest(tp+tn, n).proportion_ci()
    return {
        "n": n, "acc": acc, "ci": [ci[0], ci[1]],
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "prec": prec, "rec": rec, "f1": f1, "spec": spec,
        "balanced_acc": balanced_acc, "mcc": mcc,
    }

# Overall
all_metrics = compute_metrics(records)
print("=" * 70)
print("PHASE 4 — COMPLETE NUMBERS FOR PAPER")
print("=" * 70)

print(f"\n【Overall】n={all_metrics['n']}")
print(f"  Accuracy: {all_metrics['acc']:.1%} [{all_metrics['ci'][0]:.1%}, {all_metrics['ci'][1]:.1%}]")
print(f"  Confusion: TP={all_metrics['tp']} FP={all_metrics['fp']} TN={all_metrics['tn']} FN={all_metrics['fn']}")
print(f"  Precision={all_metrics['prec']:.1%}  Recall={all_metrics['rec']:.1%}  F1={all_metrics['f1']:.1%}")
print(f"  Specificity={all_metrics['spec']:.1%}  Balanced Acc={all_metrics['balanced_acc']:.1%}  MCC={all_metrics['mcc']:.3f}")

# Per-station
print(f"\n【Per-Station】")
for site in ["Changsha", "Longmen"]:
    m = compute_metrics([r for r in records if r["site"] == site])
    print(f"  {site} (n={m['n']}): Acc={m['acc']:.1%} [{m['ci'][0]:.1%}, {m['ci'][1]:.1%}]")
    print(f"    TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"    P={m['prec']:.1%} R={m['rec']:.1%} F1={m['f1']:.1%} Spec={m['spec']:.1%}")

# Per-season
print(f"\n【Per-Season】")
for season in ["spring", "summer", "autumn", "winter"]:
    m = compute_metrics([r for r in records if r["season"] == season])
    print(f"  {season:8s} (n={m['n']:3d}): Acc={m['acc']:.1%} [{m['ci'][0]:.1%}, {m['ci'][1]:.1%}]")

# CLDMSK raw (no model)
print(f"\n【CLDMSK Raw (no model)】")
cldmsk_records = []
for fname in files:
    site = "Changsha" if "Changsha" in fname else "Longmen"
    d = np.load(os.path.join(test_dir, fname))
    cl = float(d.get("Center_Label", float("nan")))
    y_mask = int(d["Y_mask"].squeeze()[64, 64])
    cldmsk_cloud = 1 if y_mask >= 2 else 0
    if not np.isnan(cl):
        cldmsk_records.append({"radar": int(cl), "pred": cldmsk_cloud, "site": site})

cldmsk_m = compute_metrics(cldmsk_records)
print(f"  Overall (n={cldmsk_m['n']}): Acc={cldmsk_m['acc']:.1%}")
print(f"  TP={cldmsk_m['tp']} FP={cldmsk_m['fp']} TN={cldmsk_m['tn']} FN={cldmsk_m['fn']}")
print(f"  FP rate = {cldmsk_m['fp']/(cldmsk_m['fp']+cldmsk_m['tn']):.1%}")
print(f"  FN rate = {cldmsk_m['fn']/(cldmsk_m['fn']+cldmsk_m['tp']):.1%}")

for site in ["Changsha", "Longmen"]:
    m = compute_metrics([r for r in cldmsk_records if r["site"] == site])
    print(f"  {site}: Acc={m['acc']:.1%} (n={m['n']})")

# Day-night gap
print(f"\n【Day-Night Gap】")
print(f"  Night CLDMSK: {cldmsk_m['acc']:.1%}")
print(f"  Day CLDMSK:   78.7%")
print(f"  Gap: {0.787 - cldmsk_m['acc']:.1%}pp = {(0.787 - cldmsk_m['acc'])*100:.1f}pp")

# PhysPrior improvement
improvement = all_metrics['acc'] - cldmsk_m['acc']
print(f"\n【PhysPrior Improvement】")
print(f"  PhysPrior: {all_metrics['acc']:.1%}")
print(f"  CLDMSK:    {cldmsk_m['acc']:.1%}")
print(f"  Δ:        {improvement:.1%} = {improvement*100:.1f}pp")
