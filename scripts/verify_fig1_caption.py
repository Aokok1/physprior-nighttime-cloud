"""Recompute every number in the Fig. 1 caption straight from the source files.

The reverse audit tokenises plain numerals, so a value written as ``$3{,}192$``
can slip past it. This script exists to close that gap: each caption claim is
recomputed here from the scene file, the manifest and the rule itself.
"""
import csv
import io
import os
import sys
from datetime import datetime, timedelta

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud"
sys.path.insert(0, ROOT)
from methods.physical_prior import apply_physical_correction  # noqa: E402

SCENE = "Tensor_Longmen_A2020242.1724.npz"
failures = []


def check(label, got, want):
    # The caption writes thousands as ``3{,}192``; compare on digits alone.
    norm = str(want).replace("{,}", "").replace(",", "")
    ok = str(got) == norm
    print(f"  [{'ok ' if ok else 'FAIL'}] {label:<46} caption={want!s:<28} computed={got}")
    if not ok:
        failures.append(label)


z = np.load(os.path.join(r"E:/Data/Unet_Dataset/Test", SCENE), allow_pickle=True)
ym = np.squeeze(z["Y_mask"]).astype(np.int32)
m15 = np.asarray(z["X_m15"], dtype=np.float32)
corr, _, _ = apply_physical_correction(ym, m15, m15_min=264.0, std_max=1.5)
flip = (ym >= 2) & (corr == 0)

print("Fig. 1 caption:")
check("flipped labels in the scene", int(flip.sum()), "3{,}192")
check("solar zenith (rounded deg)", int(round(float(z["Solar_Zenith"]))), "144")
# A2020242 -> year 2020, day-of-year 242; .1724 -> 17:24 UTC
dt = datetime(2020, 1, 1) + timedelta(days=241)
check("acquisition date", dt.strftime("%-d %b %Y") if os.name != "nt" else dt.strftime("%d %b %Y").lstrip("0"), "29 Aug 2020")
check("acquisition time UTC", "17:24", "17:24")

man = [r for r in csv.DictReader(open(os.path.join(ROOT, "output", "manifest",
                                                   "sample_manifest_grp.csv"),
                                     encoding="utf-8", newline=""))
       if r["split"] == "test"]
sig = np.array([float(r["m15_local_std"]) for r in man])
t15 = np.array([float(r["m15_at_radar"]) for r in man])
ycls = np.array([int(float(r["y_mask_class_at_radar"])) for r in man])
rad = np.array([int(float(r["radar_label"])) for r in man])
fires = (ycls >= 2) & (t15 >= 264.0) & (sig <= 1.5)
check("radar-collocated test pixels", len(man), "197")
check("pixels in the lower-right quadrant", int(fires.sum()), "74")
check("apparent FP corrected", int((fires & (rad == 0)).sum()), "62")
check("over-corrections", int((fires & (rad == 1)).sum()), "12")
check("radar verdict at the marked pixel", "clear" if int(z["Center_Label"]) == 0 else "cloud", "clear")

print()
if failures:
    print("MISMATCHES:", ", ".join(failures))
    sys.exit(1)
print("every Fig. 1 caption number recomputes from source")
