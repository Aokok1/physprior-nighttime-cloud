"""Quick nighttime CLDMSK vs radar evaluation + radar profiling plots."""
import numpy as np
import os, glob, sys
from scipy.stats import binomtest

NPZ_DIRS = [
    r"H:\code\server_package\data\Unet_Dataset\Val",
    r"H:\code\server_package\data\Unet_Dataset\Test",
]

night_results = []
for ddir in NPZ_DIRS:
    if not os.path.exists(ddir):
        continue
    files = sorted(glob.glob(os.path.join(ddir, "*.npz")))
    print(f"{os.path.basename(ddir)}: {len(files)} files")
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        mask = d["Y_mask"]
        radar_label = float(d["Center_Label"])
        if np.isnan(radar_label):
            continue
        cldmsk_cloud = mask[64, 64] >= 2
        radar_cloud = radar_label == 1
        fname = os.path.basename(fp)
        site = "Changsha" if "Changsha" in fname else "Longmen"
        night_results.append({
            "file": fname, "site": site,
            "cldmsk_val": int(mask[64,64]),
            "radar_label": int(radar_label),
            "correct": cldmsk_cloud == radar_cloud,
        })

print(f"\n{'='*60}")
print("NIGHTTIME CLDMSK vs Radar")
print(f"{'='*60}")
if night_results:
    correct = sum(r["correct"] for r in night_results)
    acc = correct / len(night_results)
    ci = binomtest(correct, len(night_results)).proportion_ci()
    print(f"Combined: {acc:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]  (n={len(night_results)})")
    for site in ["Changsha", "Longmen"]:
        sr = [r for r in night_results if r["site"] == site]
        if sr:
            sc = sum(r["correct"] for r in sr)
            sci = binomtest(sc, len(sr)).proportion_ci()
            print(f"  {site}: {sc/len(sr):.1%} [{sci[0]:.1%}, {sci[1]:.1%}]  (n={len(sr)})")
    
    # Clear / Cloud recall
    for label_name, label_val in [("True Clear", 0), ("True Cloud", 1)]:
        subset = [r for r in night_results if r["radar_label"] == label_val]
        if subset:
            sc = sum(r["correct"] for r in subset)
            print(f"  {label_name} recall: {sc/len(subset):.1%} ({sc}/{len(subset)})")
    
    # Confusion matrix
    tp = sum(1 for r in night_results if r["radar_label"]==1 and r["cldmsk_val"]>=2)
    fn = sum(1 for r in night_results if r["radar_label"]==1 and r["cldmsk_val"]<2)
    fp = sum(1 for r in night_results if r["radar_label"]==0 and r["cldmsk_val"]>=2)
    tn = sum(1 for r in night_results if r["radar_label"]==0 and r["cldmsk_val"]<2)
    print(f"  Confusion: TP={tp} FN={fn} FP={fp} TN={tn}")
else:
    print("NO DATA")
