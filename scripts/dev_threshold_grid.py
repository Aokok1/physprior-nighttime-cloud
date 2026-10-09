"""Development-set threshold grid and preset table, with the deployed operator.

The manuscript and supplementary S1 make a specific claim about the 160
radar-collocated development pixels: that neither the moderate preset nor any cell
of an 8x6 threshold grid beats raw CLDMSK there. This script is the artifact that
claim rests on, so the numbers can be re-derived from the released data rather
than trusted.

Operator (identical to methods/physical_prior.py, and to the manifest columns):
  * M15 brightness temperature at the radar pixel itself,
  * population standard deviation (ddof=0) over the 5x5 window,
  * flip a CLDMSK cloud label to clear when M15 >= T_th AND sigma <= sigma_th.

Run:  python -X utf8 scripts/dev_threshold_grid.py
Out:  output/dev_threshold_grid_grp.json
"""
import csv
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud"
SUF = "_grp"
OUT = os.path.join(ROOT, "output", f"dev_threshold_grid{SUF}.json")

T_RANGE = [258, 260, 262, 264, 266, 268, 270, 272]
S_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
PRESETS = {"conservative": (266.0, 1.0), "moderate": (264.0, 1.5), "aggressive": (262.0, 2.0)}

rows = list(csv.DictReader(open(os.path.join(ROOT, "output", "manifest",
                                             f"sample_manifest{SUF}.csv"),
                                encoding="utf-8", newline="")))
dev = [r for r in rows
       if r["split"] in ("train_grp", "val_grp") and str(r["has_radar"]).lower() == "true"]
m15 = np.array([float(r["m15_at_radar"]) for r in dev])
sd = np.array([float(r["m15_local_std"]) for r in dev])
ycls = np.array([int(float(r["y_mask_class_at_radar"])) for r in dev])
rad = np.array([int(float(r["radar_label"])) for r in dev])
cldmsk = (ycls >= 2).astype(int)

cloud = int((cldmsk == 1).sum())
apparent_fp = int(((cldmsk == 1) & (rad == 0)).sum())
true_pos = int(((cldmsk == 1) & (rad == 1)).sum())
raw_acc = float((cldmsk == rad).mean())


def evaluate(t_th, s_th):
    pred = cldmsk.copy()
    fire = (cldmsk == 1) & (m15 >= t_th) & (sd <= s_th)
    pred[fire] = 0
    return {
        "flipped": int(fire.sum()),
        "fp_recovered": int((fire & (rad == 0)).sum()),
        "tp_lost": int((fire & (rad == 1)).sum()),
        "accuracy": float((pred == rad).mean()),
    }


presets = {}
for name, (t, s) in PRESETS.items():
    r = evaluate(t, s)
    r.update({"T_th": t, "sigma_th": s})
    presets[name] = r

matrix, best = [], {"acc": -1.0, "T": None, "sigma": None}
for t in T_RANGE:
    line = []
    for s in S_RANGE:
        acc = evaluate(float(t), float(s))["accuracy"]
        line.append(acc)
        if acc > best["acc"]:
            best = {"acc": acc, "T": t, "sigma": s}
    matrix.append(line)

payload = {
    "description": "PhysPrior threshold behaviour on the 160 radar-collocated development pixels",
    "operator": "M15 at the radar pixel; 5x5 population std (ddof=0); flip when M15>=T_th and sigma<=sigma_th",
    "n_pixels": len(dev),
    "composition": {s: sum(1 for r in dev if r["split"] == s) for s in ("train_grp", "val_grp")},
    "stations": sorted({r["station"] for r in dev}),
    "cldmsk_cloud_pixels": cloud,
    "apparent_fp": apparent_fp,
    "true_positive": true_pos,
    "raw_cldmsk_accuracy": raw_acc,
    "presets": presets,
    "grid": {"T_range": T_RANGE, "S_range": S_RANGE, "matrix": matrix},
    "grid_max": best,
    "grid_max_ties_raw": abs(best["acc"] - raw_acc) < 1e-12,
    "any_cell_beats_raw": bool(best["acc"] > raw_acc),
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(payload, open(OUT, "w", encoding="utf-8"), indent=2)

print(f"development pixels        : {len(dev)}  {payload['composition']}  station(s) {payload['stations']}")
print(f"CLDMSK-cloud / apparent FP / TP : {cloud} / {apparent_fp} / {true_pos}")
print(f"raw CLDMSK accuracy       : {raw_acc * 100:.2f}%  ({int((cldmsk == rad).sum())}/{len(dev)})")
for name, r in presets.items():
    print(f"  {name:<13} acc {r['accuracy'] * 100:6.2f}%  flipped {r['flipped']:>3}  "
          f"FP {r['fp_recovered']:>2}/{apparent_fp}  TP {r['tp_lost']:>2}/{true_pos}")
print(f"grid maximum              : {best['acc'] * 100:.2f}% at (T={best['T']}, sigma={best['sigma']})")
print(f"any cell beats raw?       : {payload['any_cell_beats_raw']}  (ties raw: {payload['grid_max_ties_raw']})")
print(f"\nwrote {OUT}")
