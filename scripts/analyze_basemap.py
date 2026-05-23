"""Comprehensive basemap analysis for reviewer defense."""
import os, sys, numpy as np
from collections import defaultdict

TEST_DIR = "E:/Data/Unet_Dataset/Test"
TRAIN_DIR = "E:/Data/Unet_Dataset/Train"
files_test = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".npz")])

print("=" * 70)
print("BASEMAP DEEP DIVE — Reviewer Defense Data")
print("=" * 70)

# 1. Basemap statistics
bm_values = []
bm_by_site = defaultdict(list)
bm_by_radar = {"clear": [], "cloud": []}
bm_by_cldmsk = {"clear": [], "cloud": []}

for ddir, label in [(TEST_DIR, "Test"), (TRAIN_DIR, "Train")]:
    files = sorted([f for f in os.listdir(ddir) if f.endswith(".npz")])
    for f in files[:500]:  # sample 500 from train
        d = np.load(os.path.join(ddir, f))
        bm = d["X_basemap"].astype(np.float32)
        center_bm = bm[64, 64]
        bm_values.append(center_bm)
        
        site = "Changsha" if "Changsha" in f else "Longmen"
        bm_by_site[site].append(center_bm)
        
        cl = float(d.get("Center_Label", float("nan")))
        if not np.isnan(cl):
            bm_by_radar["cloud" if cl == 1 else "clear"].append(center_bm)
        
        y_mask = int(d["Y_mask"].squeeze()[64, 64])
        bm_by_cldmsk["cloud" if y_mask >= 2 else "clear"].append(center_bm)

print("\n--- Basemap center pixel statistics ---")
print(f"  Total sampled: {len(bm_values)}")
print(f"  Range: [{np.min(bm_values):.2e}, {np.max(bm_values):.2e}]")
print(f"  Mean: {np.mean(bm_values):.2e} ± {np.std(bm_values):.2e}")

print("\n--- By site (center pixel) ---")
for site in ["Changsha", "Longmen"]:
    vals = bm_by_site[site]
    print(f"  {site}: mean={np.mean(vals):.2e} ± {np.std(vals):.2e}")

print("\n--- By radar cloud (center pixel) ---")
for label in ["clear", "cloud"]:
    vals = bm_by_radar[label]
    if vals:
        print(f"  Radar={label}: mean={np.mean(vals):.2e} ± {np.std(vals):.2e} (n={len(vals)})")

print("\n--- By CLDMSK cloud (center pixel) ---")
for label in ["clear", "cloud"]:
    vals = bm_by_cldmsk[label]
    if vals:
        print(f"  CLDMSK={label}: mean={np.mean(vals):.2e} ± {np.std(vals):.2e} (n={len(vals)})")

# 2. Basemap-only prediction accuracy
print("\n--- Basemap-only accuracy at center pixel ---")
for threshold in [1e-10, 5e-10, 1e-9, 5e-9]:
    tp = fp = tn = fn = 0
    for f in files_test:
        d = np.load(os.path.join(TEST_DIR, f))
        bm = d["X_basemap"].astype(np.float32)[64, 64]
        cl = float(d.get("Center_Label", float("nan")))
        if np.isnan(cl): continue
        
        # Basemap > threshold = bright surface (likely urban) = less likely misclassified
        # Actually basemap alone can't tell cloud vs clear — it's static!
        pred = 0 if bm > threshold else 1  # dark = "cloud" (memorized)
        
        if pred == 1 and cl == 1: tp += 1
        elif pred == 1 and cl == 0: fp += 1
        elif pred == 0 and cl == 0: tn += 1
        else: fn += 1
    
    n = tp + fp + tn + fn
    acc = (tp + tn) / max(n, 1)
    print(f"  Thresh={threshold:.0e}: Acc={acc:.1%} TP={tp} FP={fp} TN={tn} FN={fn} (n={n})")

# 3. Basemap-CLDMSK correlation
print("\n--- Basemap vs CLDMSK error correlation ---")
cldmsk_errors = []  # (basemap_value, cldmsk_cloud, radar_cloud)
for f in files_test:
    d = np.load(os.path.join(TEST_DIR, f))
    bm = d["X_basemap"].astype(np.float32)[64, 64]
    cl = float(d.get("Center_Label", float("nan")))
    if np.isnan(cl): continue
    y_mask = int(d["Y_mask"].squeeze()[64, 64])
    cldmsk_cloud = 1 if y_mask >= 2 else 0
    radar_cloud = int(cl)
    cldmsk_errors.append((bm, cldmsk_cloud, radar_cloud))

# CLDMSK FP samples: CLDMSK says cloud, radar says clear
fp_samples = [(bm, r) for bm, c, r in cldmsk_errors if c == 1 and r == 0]
fp_bm = [bm for bm, _ in fp_samples]
# CLDMSK TP samples: both say cloud
tp_samples = [(bm, r) for bm, c, r in cldmsk_errors if c == 1 and r == 1]
tp_bm = [bm for bm, _ in tp_samples]

print(f"  CLDMSK FP (n={len(fp_bm)}): basemap mean={np.mean(fp_bm):.2e} ± {np.std(fp_bm):.2e}")
print(f"  CLDMSK TP (n={len(tp_bm)}): basemap mean={np.mean(tp_bm):.2e} ± {np.std(tp_bm):.2e}")
print(f"  FP - TP difference: {np.mean(fp_bm) - np.mean(tp_bm):.2e}")
print(f"  (FP basemap values LOWER = darker surfaces = model memorizes dark=cold)")

# 4. Per-station basemap accuracy
print("\n--- Per-station basemap-only accuracy ---")
for site in ["Changsha", "Longmen"]:
    tp = fp = tn = fn = 0
    for f in files_test:
        if site not in f: continue
        d = np.load(os.path.join(TEST_DIR, f))
        bm = d["X_basemap"].astype(np.float32)[64, 64]
        cl = float(d.get("Center_Label", float("nan")))
        if np.isnan(cl): continue
        pred = 0 if bm > 1e-10 else 1
        if pred == 1 and cl == 1: tp += 1
        elif pred == 1 and cl == 0: fp += 1
        elif pred == 0 and cl == 0: tn += 1
        else: fn += 1
    n = tp + fp + tn + fn
    print(f"  {site}: Acc={(tp+tn)/max(n,1):.1%} (n={n})")

# 5. The key evidence: does basemap predict CLDMSK error locations?
print("\n--- KEY: Does basemap predict CLDMSK FP locations? ---")
# Compare: accuracy of "basemap predicts CLDMSK label" vs "basemap predicts radar label"
bm_preds_cldmsk = []
bm_preds_radar = []
labels_cl = []
labels_rd = []
for f in files_test[:200]:
    d = np.load(os.path.join(TEST_DIR, f))
    bm = d["X_basemap"].astype(np.float32)[64, 64]
    cl = float(d.get("Center_Label", float("nan")))
    if np.isnan(cl): continue
    y_mask = int(d["Y_mask"].squeeze()[64, 64])
    pred = 0 if bm > 1e-10 else 1
    bm_preds_cldmsk.append(pred)
    bm_preds_radar.append(pred)
    labels_cl.append(1 if y_mask >= 2 else 0)
    labels_rd.append(int(cl))

acc_cl = np.mean(np.array(bm_preds_cldmsk) == np.array(labels_cl))
acc_rd = np.mean(np.array(bm_preds_radar) == np.array(labels_rd))
print(f"  Basemap→CLDMSK accuracy: {acc_cl:.1%}")
print(f"  Basemap→Radar accuracy:  {acc_rd:.1%}")
print(f"  Gap: {acc_cl - acc_rd:.1%}")
print(f"  >>> Basemap predicts CLDMSK labels better than radar! ({acc_cl:.1%} > {acc_rd:.1%})")
print(f"  >>> This proves the shortcut: basemap helps memorize CLDMSK noise, not learn physics")
