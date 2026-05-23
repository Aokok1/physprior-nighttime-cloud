"""Basemap analysis — simpler, more robust version."""
import os, sys, numpy as np

TEST_DIR = "E:/Data/Unet_Dataset/Test"

files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".npz")])

# Collect all data
data = []
for f in files:
    d = np.load(os.path.join(TEST_DIR, f))
    bm = float(d["X_basemap"].astype(np.float32)[64,64])
    cl = float(d.get("Center_Label", float("nan")))
    y_mask = int(d["Y_mask"].squeeze()[64,64])
    site = "Changsha" if "Changsha" in f else "Longmen"
    if not np.isnan(cl):
        data.append({
            "bm": bm, "radar": int(cl), "cldmsk": 1 if y_mask >= 2 else 0,
            "site": site, "fname": f
        })

print("=" * 70)
print("BASEMAP DEEP DIVE — Test Set (n={})".format(len(data)))
print("=" * 70)

# 1. Basic stats
bms = [d["bm"] for d in data]
print(f"\nBasemap value range: [{min(bms):.2e}, {max(bms):.2e}]")
print(f"Mean: {np.mean(bms):.2e} ± {np.std(bms):.2e}")

# 2. By site
for site in ["Changsha", "Longmen"]:
    sv = [d["bm"] for d in data if d["site"] == site]
    print(f"  {site}: mean={np.mean(sv):.2e} ± {np.std(sv):.2e} (n={len(sv)})")

# 3. By CLDMSK class
for label, cond in [("Clear", lambda d: d["cldmsk"]==0), ("Cloud", lambda d: d["cldmsk"]==1)]:
    sv = [d["bm"] for d in data if cond(d)]
    print(f"  CLDMSK={label}: mean={np.mean(sv):.2e} ± {np.std(sv):.2e} (n={len(sv)})")

# 4. By radar class
for label, cond in [("Clear", lambda d: d["radar"]==0), ("Cloud", lambda d: d["radar"]==1)]:
    sv = [d["bm"] for d in data if cond(d)]
    print(f"  Radar={label}: mean={np.mean(sv):.2e} ± {np.std(sv):.2e} (n={len(sv)})")

# 5. CLDMSK FP vs TP basemap values
fp_bm = [d["bm"] for d in data if d["cldmsk"]==1 and d["radar"]==0]
tp_bm = [d["bm"] for d in data if d["cldmsk"]==1 and d["radar"]==1]
fp_mean = np.mean(fp_bm) if fp_bm else 0
tp_mean = np.mean(tp_bm) if tp_bm else 0
print(f"\nCLDMSK FP (n={len(fp_bm)}): basemap={fp_mean:.2e}")
print(f"CLDMSK TP (n={len(tp_bm)}): basemap={tp_mean:.2e}")
print(f"FP vs TP ratio: {fp_mean/tp_mean:.2f}x" if tp_mean > 0 else "FP/TP: N/A")
if fp_mean < tp_mean:
    print("  >>> FP surfaces are DARKER = more likely misclassified as cloud")
    print("  >>> Model learns: dark basemap → predict cloud")

# 6. Key evidence: can basemap alone distinguish CLDMSK errors?
print(f"\n--- KEY EVIDENCE ---")
# Comput baseline: always predict majority class
from collections import Counter
label_counts = Counter(d["radar"] for d in data)
majority = max(label_counts, key=label_counts.get)
majority_acc = label_counts[majority] / len(data)
print(f"  Majority class baseline (always predict {majority}): {majority_acc:.1%}")

# Basemap prediction at different thresholds
best_acc = 0
best_thresh = 0
best_tp = best_fp = best_tn = best_fn = 0
for t in [1e-10, 3e-10, 5e-10, 1e-9, 2e-9, 5e-9, 1e-8]:
    tp = fp = tn = fn = 0
    for d in data:
        pred = 0 if d["bm"] > t else 1
        if pred == 1 and d["radar"] == 1: tp += 1
        elif pred == 1 and d["radar"] == 0: fp += 1
        elif pred == 0 and d["radar"] == 0: tn += 1
        else: fn += 1
    acc = (tp+tn)/len(data)
    if acc > best_acc:
        best_acc = acc
        best_thresh = t
        best_tp, best_fp, best_tn, best_fn = tp, fp, tn, fn
    print(f"  Thresh={t:.1e}: Acc={acc:.1%} TP={tp} FP={fp} TN={tn} FN={fn}")

print(f"\n  Best basemap-only: Acc={best_acc:.1%} at T={best_thresh:.1e}")
print(f"  TP={best_tp} FP={best_fp} TN={best_tn} FN={best_fn}")

# 7. The smoking gun
print(f"\n--- SMOKING GUN ---")
# Does basemap agree more with CLDMSK or with radar?
cldmsk_agree = sum(1 for d in data if (d["bm"] > 1e-10) == (d["cldmsk"] == 0))
radar_agree = sum(1 for d in data if (d["bm"] > 1e-10) == (d["radar"] == 0))
print(f"  Basemap ↔ CLDMSK agreement: {cldmsk_agree/len(data):.1%}")
print(f"  Basemap ↔ Radar agreement:  {radar_agree/len(data):.1%}")
if cldmsk_agree > radar_agree:
    print(f"  >>> Basemap matches CLDMSK better than radar!")
    print(f"  >>> Gap: {cldmsk_agree/len(data) - radar_agree/len(data):.1%} — model learns noise, not physics")
