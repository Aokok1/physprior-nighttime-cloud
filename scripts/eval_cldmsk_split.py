"""Nighttime CLDMSK vs radar - Test set only, plus daytime eval (optimized)."""
import numpy as np
import os, glob, sys
from scipy.stats import binomtest
from datetime import datetime, timedelta

# ====== Part 1: Nighttime (Test set only) ======
print("="*60)
print("NIGHTTIME CLDMSK vs Radar (Test set)")
print("="*60)

TEST_DIR = r"H:\code\server_package\data\Unet_Dataset\Test"
results_night_test = []
for fp in sorted(glob.glob(os.path.join(TEST_DIR, "*.npz"))):
    d = np.load(fp, allow_pickle=True)
    radar_label = float(d["Center_Label"])
    if np.isnan(radar_label):
        continue
    mask = d["Y_mask"]
    cldmsk_cloud = mask[64, 64] >= 2
    radar_cloud = radar_label == 1
    fname = os.path.basename(fp)
    site = "Changsha" if "Changsha" in fname else "Longmen"
    results_night_test.append({
        "file": fname, "site": site,
        "cldmsk_cloud": cldmsk_cloud, "radar_cloud": radar_cloud,
        "correct": cldmsk_cloud == radar_cloud,
    })

print(f"Files with radar labels: {len(results_night_test)}")
if results_night_test:
    correct = sum(r["correct"] for r in results_night_test)
    acc = correct / len(results_night_test)
    ci = binomtest(correct, len(results_night_test)).proportion_ci()
    print(f"\nTest Combined: {acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]  (n={len(results_night_test)})")
    for site in ["Changsha", "Longmen"]:
        sr = [r for r in results_night_test if r["site"] == site]
        if sr:
            sc = sum(r["correct"] for r in sr)
            sci = binomtest(sc, len(sr)).proportion_ci()
            print(f"  {site}: {sc/len(sr):.1%} [{sci[0]:.1%}, {sci[1]:.1%}]  (n={len(sr)})")
    
    for label_name, label_val in [("True Clear", False), ("True Cloud", True)]:
        subset = [r for r in results_night_test if r["radar_cloud"] == label_val]
        if subset:
            sc = sum(r["correct"] for r in subset)
            print(f"  {label_name} recall: {sc/len(subset):.1%} ({sc}/{len(subset)})")
    
    tp = sum(1 for r in results_night_test if r["radar_cloud"] and r["cldmsk_cloud"])
    fn = sum(1 for r in results_night_test if r["radar_cloud"] and not r["cldmsk_cloud"])
    fp = sum(1 for r in results_night_test if not r["radar_cloud"] and r["cldmsk_cloud"])
    tn = sum(1 for r in results_night_test if not r["radar_cloud"] and not r["cldmsk_cloud"])
    print(f"  CM: TP={tp} FN={fn} FP={fp} TN={tn}")

# ====== Part 2: Nighttime (Val set) ======
print("\n" + "="*60)
print("NIGHTTIME CLDMSK vs Radar (Val set)")
print("="*60)

VAL_DIR = r"H:\code\server_package\data\Unet_Dataset\Val"
results_night_val = []
for fp in sorted(glob.glob(os.path.join(VAL_DIR, "*.npz"))):
    d = np.load(fp, allow_pickle=True)
    radar_label = float(d["Center_Label"])
    if np.isnan(radar_label):
        continue
    mask = d["Y_mask"]
    cldmsk_cloud = mask[64, 64] >= 2
    fname = os.path.basename(fp)
    site = "Changsha" if "Changsha" in fname else "Longmen"
    results_night_val.append({
        "file": fname, "site": site,
        "cldmsk_cloud": cldmsk_cloud, "radar_cloud": radar_label == 1,
        "correct": cldmsk_cloud == (radar_label == 1),
    })

if results_night_val:
    correct = sum(r["correct"] for r in results_night_val)
    acc = correct / len(results_night_val)
    ci = binomtest(correct, len(results_night_val)).proportion_ci()
    print(f"Val Combined: {acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]  (n={len(results_night_val)})")
    for site in ["Changsha", "Longmen"]:
        sr = [r for r in results_night_val if r["site"] == site]
        if sr:
            sc = sum(r["correct"] for r in sr)
            sci = binomtest(sc, len(sr)).proportion_ci()
            print(f"  {site}: {sc/len(sr):.1%} [{sci[0]:.1%}, {sci[1]:.1%}]  (n={len(sr)})")
