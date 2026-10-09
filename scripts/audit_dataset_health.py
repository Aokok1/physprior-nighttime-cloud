"""Full content-level health check of the training/evaluation dataset.

Reads every .npz in the canonical split directories and verifies, per file:
  - it opens and exposes the keys the pipeline consumes
  - the channel/grid shapes are the expected 128 x 128
  - no NaN/Inf in the channels used for training (DNB, basemap, M15)
  - Y_mask holds only {0,1,2,3} and Center_Label is binary or NaN
  - Radar_Loc is [64, 64]
It then cross-checks the manifest against the files on disk, looks for duplicate
content by hash, and reports the overpass/temporal structure that governs the
train/val/test split.

Run:  python scripts/audit_dataset_health.py
Out:  output/audit/dataset_health.json  +  stdout summary
"""
import collections
import csv
import hashlib
import json
import os
import sys

import numpy as np

ROOT = r"E:/Claude code/project/noise-label-cloud"
DATA = r"E:/Data/Unet_Dataset"
OUT = os.path.join(ROOT, "output", "audit")
SPLIT_DIRS = ["Train_phase4", "Val_phase4", "Test"]
NEEDED = ["X_dnb", "X_basemap", "X_m15", "Y_mask", "Center_Label", "Radar_Loc"]

os.makedirs(OUT, exist_ok=True)
report = {"files": {}, "problems": [], "warnings": []}

manifest = {}
with open(os.path.join(ROOT, "output", "manifest", "sample_manifest.csv"), encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        manifest[r["fname"]] = r

print("=" * 74)
print("DATASET HEALTH CHECK")
print("=" * 74)

# ---- 1. per-file content check ----------------------------------------------
disk = {}
for sub in SPLIT_DIRS:
    d = os.path.join(DATA, sub)
    names = sorted(x for x in os.listdir(d) if x.endswith(".npz"))
    disk[sub] = names
    print(f"\n[{sub}] {len(names)} npz")

    stats = collections.defaultdict(list)
    bad = []
    for i, n in enumerate(names):
        p = os.path.join(d, n)
        try:
            z = np.load(p, allow_pickle=True)
        except Exception as e:
            bad.append((n, f"unreadable: {e}"))
            continue
        keys = set(z.files)
        miss = [k for k in NEEDED if k not in keys]
        if miss:
            bad.append((n, f"missing keys {miss}"))
            continue
        shape_problems = []
        for k in ("X_dnb", "X_basemap", "X_m15"):
            if z[k].shape != (128, 128):
                shape_problems.append(f"{k}{z[k].shape}")
        if z["Y_mask"].shape != (128, 128):
            shape_problems.append(f"Y_mask{z['Y_mask'].shape}")
        if shape_problems:
            bad.append((n, "shape " + ",".join(shape_problems)))
            continue

        dnb, bm, m15 = z["X_dnb"], z["X_basemap"], z["X_m15"]
        ym = z["Y_mask"].astype(int)
        cl = float(z["Center_Label"])
        rl = z["Radar_Loc"]

        if not np.isfinite(dnb).all():
            bad.append((n, f"X_dnb non-finite x{int((~np.isfinite(dnb)).sum())}"))
        if not np.isfinite(bm).all():
            bad.append((n, f"X_basemap non-finite x{int((~np.isfinite(bm)).sum())}"))
        n_m15_bad = int((~np.isfinite(m15)).sum())
        if n_m15_bad:
            bad.append((n, f"X_m15 non-finite x{n_m15_bad}"))
        if ym.min() < 0 or ym.max() > 3:
            bad.append((n, f"Y_mask range [{ym.min()},{ym.max()}]"))
        if not (np.isnan(cl) or cl in (0.0, 1.0)):
            bad.append((n, f"Center_Label={cl}"))
        if list(rl) != [64, 64]:
            bad.append((n, f"Radar_Loc={list(rl)}"))

        stats["dnb_min"].append(float(dnb.min()))
        stats["dnb_max"].append(float(dnb.max()))
        stats["m15_min"].append(float(np.nanmin(m15)))
        stats["m15_max"].append(float(np.nanmax(m15)))
        stats["m15_nan_px"].append(n_m15_bad)
        stats["ym3_frac"].append(float((ym == 3).mean()))
        stats["has_radar"].append(int(not np.isnan(cl)))
        stats["radar_label"].append(int(cl) if not np.isnan(cl) else -1)
        if i % 250 == 0:
            print(f"   ... {i}/{len(names)}", flush=True)

    report["files"][sub] = {
        "n": len(names),
        "bad": len(bad),
        "dnb_min": min(stats["dnb_min"]) if stats["dnb_min"] else None,
        "dnb_max": max(stats["dnb_max"]) if stats["dnb_max"] else None,
        "m15_min": min(stats["m15_min"]) if stats["m15_min"] else None,
        "m15_max": max(stats["m15_max"]) if stats["m15_max"] else None,
        "m15_nan_px_total": int(sum(stats["m15_nan_px"])),
        "files_with_m15_nan": int(sum(1 for x in stats["m15_nan_px"] if x)),
        "has_radar": int(sum(stats["has_radar"])),
        "radar_cloud": int(sum(1 for x in stats["radar_label"] if x == 1)),
        "mean_ym3_frac": float(np.mean(stats["ym3_frac"])) if stats["ym3_frac"] else None,
    }
    print(f"   bad files: {len(bad)}")
    for n, why in bad[:10]:
        print(f"     ! {n}: {why}")
    if len(bad) > 10:
        print(f"     ... and {len(bad)-10} more")
    report["problems"].extend([{"dir": sub, "file": n, "why": w} for n, w in bad])

# ---- 2. manifest vs disk ----------------------------------------------------
print("\n" + "=" * 74)
print("MANIFEST vs DISK")
print("=" * 74)
union = set()
for sub in SPLIT_DIRS:
    union |= set(disk[sub])
print(f"unique npz on disk (3 split dirs) = {len(union)}")
print(f"manifest rows                      = {len(manifest)}")
only_disk = sorted(union - set(manifest))
only_man = sorted(set(manifest) - union)
print(f"on disk but not in manifest        = {len(only_disk)} {only_disk[:5]}")
print(f"in manifest but not on disk        = {len(only_man)} {only_man[:5]}")
report["manifest_vs_disk"] = {"disk": len(union), "manifest": len(manifest),
                              "only_disk": only_disk[:20], "only_manifest": only_man[:20]}
if only_disk or only_man:
    report["problems"].append({"kind": "manifest_mismatch",
                               "only_disk": only_disk[:20], "only_manifest": only_man[:20]})

# ---- 3. duplicate content ---------------------------------------------------
print("\n" + "=" * 74)
print("DUPLICATE CONTENT (sha256 of file bytes)")
print("=" * 74)
hashes = collections.defaultdict(list)
for sub in SPLIT_DIRS:
    for n in disk[sub]:
        p = os.path.join(DATA, sub, n)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        hashes[h].append((sub, n))
dups = {h: v for h, v in hashes.items() if len(v) > 1}
print(f"duplicate content groups = {len(dups)}")
for h, v in list(dups.items())[:8]:
    print(f"   {h}: {v}")
report["duplicate_groups"] = {h: v for h, v in list(dups.items())[:20]}

# ---- 4. split structure / leakage ------------------------------------------
print("\n" + "=" * 74)
print("SPLIT STRUCTURE")
print("=" * 74)
rows = list(manifest.values())
sp = collections.Counter(r["split"] or "(unassigned)" for r in rows)
print("split counts:", dict(sp))

grp = collections.defaultdict(set)
for r in rows:
    grp[r["overpass_group"]].add(r["split"] or "unassigned")
multi = {k: v for k, v in grp.items() if len(v) > 1}
print(f"overpass groups = {len(grp)}; spanning >1 split = {len(multi)}")
tr_va = sum(1 for v in grp.values() if {"train_phase4", "val_phase4"} <= v)
tr_te = sum(1 for v in grp.values() if "train_phase4" in v and "test" in v)
va_te = sum(1 for v in grp.values() if "val_phase4" in v and "test" in v)
print(f"  train_phase4 ∩ val_phase4 groups = {tr_va}")
print(f"  train_phase4 ∩ test        groups = {tr_te}")
print(f"  val_phase4   ∩ test        groups = {va_te}")
report["leakage"] = {"train_val_groups": tr_va, "train_test_groups": tr_te, "val_test_groups": va_te}

# temporal proximity, same station
import datetime


def d2(r):
    return datetime.date(int(r["year"]), 1, 1) + datetime.timedelta(days=int(r["doy"]) - 1)


tr = [r for r in rows if r["split"] in ("train_phase4", "val_phase4")]
te = [r for r in rows if r["split"] == "test"]
for st in ("CS", "LM"):
    a = [r for r in te if r["station"] == st]
    b = [r for r in tr if r["station"] == st]
    if not a or not b:
        continue
    gaps = []
    for x in a:
        gaps.append(min(abs((d2(x) - d2(y)).days) for y in b))
    gaps.sort()
    print(f"  {st}: nearest train/val patch to a test patch, days = "
          f"min {gaps[0]}, median {gaps[len(gaps)//2]}, <=7d {sum(1 for g in gaps if g <= 7)}/{len(gaps)}")
    report.setdefault("temporal", {})[st] = {
        "min_days": gaps[0], "median_days": gaps[len(gaps) // 2],
        "n_within_7d": sum(1 for g in gaps if g <= 7), "n_test": len(gaps)}

# ---- 5. write ---------------------------------------------------------------
with open(os.path.join(OUT, "dataset_health.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=1)
print(f"\nwrote {os.path.join(OUT, 'dataset_health.json')}")
print(f"TOTAL PROBLEMS: {len(report['problems'])}")
