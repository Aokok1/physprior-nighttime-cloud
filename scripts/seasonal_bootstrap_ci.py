"""Seasonal Bootstrap 95% CI for PhysPrior moderate (Phase 4, no-leakage).
对每个 season, 在该 season 内所有 (test 样本) 做 2000-resample bootstrap。
"""
import sys, os, json, glob
import numpy as np
import torch
from datetime import date, timedelta

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = r"E:/Claude code/project/noise-label-cloud/output/checkpoints/phase4_physprior_moderate_best.pth"

# Load model
model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                use_film=True, use_evidential=False).to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(sd, strict=False)
model.eval()

# Predict on every test sample, store per-fname correctness
ds = CloudDataset(r"E:/Data/Unet_Dataset", "Test", augment=False, use_basemap=True)
records = []
for i in range(len(ds)):
    s = ds[i]
    cl = s["center_label"]
    if cl != cl: continue
    x = s["image"].unsqueeze(0).to(DEVICE)
    moon = torch.tensor([s["moon_phase"]], device=DEVICE)
    sza  = torch.tensor([s["solar_zenith"]], device=DEVICE)
    with torch.no_grad():
        logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = logits.argmax(1)[0].cpu().numpy()
    ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
    y0, y1 = max(0, ry - 1), min(128, ry + 2)
    x0, x1 = max(0, rx - 1), min(128, rx + 2)
    win = pred_cls[y0:y1, x0:x1]
    bin_win = (win >= 2).astype(int)
    pbin = 1 if bin_win.sum() > 0 else (0 if bin_win.mean() <= 0.5 else 1)
    fname = s["filename"]
    records.append({
        "fname": fname,
        "pred": int(pbin),
        "truth": int(cl),
        "correct": int(pbin == int(cl)),
    })

# Season from DOY (calendar-accurate)
def _season(fname):
    try:
        doy = int(fname.split("_A")[1].split(".")[0])
        year = doy // 1000
        ddd  = doy % 1000
        d = date(year, 1, 1) + timedelta(days=ddd - 1)
        m = d.month
    except Exception:
        return "Unknown"
    if m in (3, 4, 5): return "Spring"
    if m in (6, 7, 8): return "Summer"
    if m in (9, 10, 11): return "Autumn"
    return "Winter"

# Bucket by season
season_records = {"Spring": [], "Summer": [], "Autumn": [], "Winter": []}
for r in records:
    s = _season(r["fname"])
    season_records[s].append(r)

# Block bootstrap within each season (preserving sample size, 2000 resamples)
print("=" * 72)
print("Seasonal Bootstrap 95% CI — PhysPrior moderate (Phase 4, no-leakage)")
print("=" * 72)

np.random.seed(42)
ci_results = {}
for sname in ("Spring", "Summer", "Autumn", "Winter"):
    rs = season_records[sname]
    n = len(rs)
    corrects = np.array([r["correct"] for r in rs], dtype=float)
    if n == 0:
        continue
    means = np.empty(2000)
    for b in range(2000):
        idx = np.random.choice(n, n, replace=True)
        means[b] = corrects[idx].mean()
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    mean = float(corrects.mean())
    ci_results[sname] = {"n": n, "mean": mean, "lo": lo, "hi": hi, "raw_corrects": corrects.tolist()}
    width = hi - lo
    print(f"  {sname:>6}: n={n:>3}  Acc={mean*100:5.1f}%  CI[{lo*100:5.1f}%, {hi*100:5.1f}%]  (width = {width*100:.1f}pp)")

# Cross-table for paper
print("\n=== FOR PAPER Table V ===")
print(f"{'Season':<8} {'n':>4} {'Acc.':>6} {'95% CI':>20}")
print("-" * 42)
for sname in ("Spring", "Summer", "Autumn", "Winter"):
    r = ci_results[sname]
    print(f"  {sname:<6} {r['n']:>4} {r['mean']*100:5.1f}%  [{r['lo']*100:5.1f}%, {r['hi']*100:5.1f}%]")

# Also do block-bootstrap (within month) for i.i.d. relaxation test
print("\n=== Block Bootstrap (by month) — to address R1's weather-autocorrelation concern ===")
from collections import defaultdict

def _month(fname):
    try:
        doy = int(fname.split("_A")[1].split(".")[0])
        year = doy // 1000
        ddd  = doy % 1000
        d = date(year, 1, 1) + timedelta(days=ddd - 1)
        return f"{d.year}-{d.month:02d}"
    except Exception:
        return "Unknown"

month_season = {}
month_records = defaultdict(list)
for r in records:
    m = _month(r["fname"])
    month_records[m].append(r)
    month_season[m] = _season(r["fname"])

np.random.seed(42)
ci_block = {}
for sname in ("Spring", "Summer", "Autumn", "Winter"):
    # Find months in this season
    months = [m for m, s in month_season.items() if s == sname]
    if not months:
        continue
    # Block bootstrap: sample MONTHS with replacement, all samples within
    pool_corrects = []
    pool_per_month = {m: np.array([r["correct"] for r in month_records[m]], dtype=float) for m in months}
    mean_overall = np.mean(np.concatenate([pool_per_month[m] for m in months]))

    means = np.empty(2000)
    for b in range(2000):
        # Sample months with replacement until total sample >= original season size
        picked_corrects = []
        target_n = sum(len(month_records[m]) for m in months)
        while sum(len(picked_corrects) for _ in [0]) < target_n:
            m_idx = np.random.randint(len(months))
            m_pick = months[m_idx]
            picked_corrects.extend(pool_per_month[m_pick].tolist())
        arr = np.array(picked_corrects[:target_n])
        means[b] = arr.mean()
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    ci_block[sname] = {"mean": mean_overall, "lo": lo, "hi": hi,
                       "n_months": len(months)}
    print(f"  {sname:>6}: mean={mean_overall*100:5.1f}%  Block-CI[{lo*100:5.1f}%, {hi*100:5.1f}%]  (months={len(months)})")

# Save full results
out_path = r"E:/Claude code/project/noise-label-cloud/output/seasonal_bootstrap_ci.json"
with open(out_path, "w") as f:
    json.dump({"iid_bootstrap": {k: {"n": v["n"], "mean": v["mean"], "lo": v["lo"], "hi": v["hi"]}
                                  for k, v in ci_results.items()},
               "block_bootstrap_by_month": ci_block}, f, indent=2)
print(f"\nSaved → {out_path}")
