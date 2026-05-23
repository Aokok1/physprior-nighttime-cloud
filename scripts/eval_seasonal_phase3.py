"""Evaluate PhysPrior moderate on Phase 3 test set with SEASONAL breakdown.
Outputs per-season accuracy + bootstrap CI + count, ensuring sum matches overall.
"""
import os, sys, json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
config.MTUNET_DATASET = "E:/Data/Unet_Dataset"

from config import MTUNET_DATASET, CHECKPOINT_DIR
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet
from eval import _collect_radar_predictions

# Julian day to season mapping (subtropical China)
def julian_to_season(jd):
    if 60 <= jd < 152:   return "spring"   # Mar-May
    if 152 <= jd < 244:  return "summer"   # Jun-Aug
    if 244 <= jd < 335:  return "autumn"   # Sep-Nov
    return "winter"                          # Dec-Feb

# Extract Julian day from filename
def extract_julian_day(filename):
    # Filename pattern: ..._2020123_... or ..._20220601_...
    import re
    # Try YYYYDDD format first
    m = re.search(r'(\d{4})(\d{3})', filename)
    if m:
        return int(m.group(2))
    # Try YYYYMMDD format
    m = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if m:
        from datetime import datetime
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return dt.timetuple().tm_yday
    return None

# Load test data directly (Test set, Phase 3)
from data.dataset import CloudDataset
from torch.utils.data import DataLoader
test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False, use_phase3=True)
val_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
ckpt_path = os.path.join(CHECKPOINT_DIR, "physprior_moderate_best.pth")
print(f"Loading: {ckpt_path}")
model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                use_film=True, use_evidential=False).to(device)
state = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()

# Collect predictions
records = _collect_radar_predictions(model, val_loader, device)
print(f"Total records: {len(records)}")

# Assign season to each record
for r in records:
    jd = extract_julian_day(r["filename"])
    r["julian_day"] = jd
    r["season"] = julian_to_season(jd) if jd else "unknown"

# Per-season analysis
seasons = ["spring", "summer", "autumn", "winter"]
results = {}
total_correct = 0
total_n = 0

for season in seasons:
    sr = [r for r in records if r["season"] == season]
    n = len(sr)
    if n == 0:
        results[season] = {"count": 0, "correct": 0, "accuracy": 0, "ci": [0, 0]}
        continue
    
    correct = sum(1 for r in sr 
                  if (r["radar_truth"]==1 and r["pred_binary"]==1) or
                     (r["radar_truth"]==0 and r["pred_binary"]==0))
    acc = correct / n
    
    # Bootstrap CI
    corrects = np.array([
        1 if (r["radar_truth"]==1 and r["pred_binary"]==1) or
             (r["radar_truth"]==0 and r["pred_binary"]==0) else 0
        for r in sr
    ])
    boot_means = []
    for _ in range(2000):
        idx = np.random.choice(n, n, replace=True)
        boot_means.append(corrects[idx].mean())
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    
    results[season] = {
        "count": n, "correct": correct,
        "accuracy": round(acc, 4),
        "ci_lower": round(ci_lo, 4),
        "ci_upper": round(ci_hi, 4)
    }
    total_correct += correct
    total_n += n
    
    print(f"  {season:8s}: {acc:.1%} [{ci_lo:.1%}, {ci_hi:.1%}]  "
          f"({correct}/{n})")

print(f"\n  Overall:  {total_correct}/{total_n} = {total_correct/total_n:.1%}")
print(f"  Cross-check sum: {'PASS' if total_n == len(records) else 'FAIL'}")

# Save
out_path = os.path.join(PROJECT_ROOT, "output", "seasonal_phase3.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")
