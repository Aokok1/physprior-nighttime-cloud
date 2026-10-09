"""Rebuild the seasonal basemaps from training and validation scenes only.

Background
----------
The eight station-season composites in `base_map/` were built by
`basemap_make.py`, which selected hand-picked "TrueClear" candidate scenes.
That candidate list no longer exists on disk and the selection criterion was
never recorded, and the script searched both `Train/` and `Test/` source
directories -- so a test scene was eligible to contribute to the composite that
feeds the third input channel of every trained model. Whether any did can no
longer be determined from the artifacts.

This script replaces that construction with one that is fully reproducible from
the released split and labels:

    a scene contributes to the station-season composite if and only if
      * it belongs to the training or validation split of the group-disjoint
        split (`NL_SPLIT_TAG`, default `grp`), and
      * at least 95% of its CLDMSK pixels are clear (classes 0 or 1).

No radar label and no test scene enters. The 95% threshold is deliberately
tight: CLDMSK over-detects cloud, so a tight clear fraction can only exclude
genuinely clear scenes, never admit a cloudy one.

Output goes to a NEW directory (`base_map_trainval/`); the original composites
are left untouched so the earlier results stay comparable.

Usage:
    python -X utf8 scripts/rebuild_basemap.py
    NL_BASEMAP_DIR=... python -X utf8 scripts/rebuild_basemap.py   # override
"""
import collections
import csv
import glob
import io
import json
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"E:/Claude code/project/noise-label-cloud"
DATASET = r"E:/Data/Unet_Dataset"
SPLIT_TAG = os.environ.get("NL_SPLIT_TAG", "grp")
OUT_DIR = os.environ.get("NL_BASEMAP_DIR", os.path.join(DATASET, "base_map_trainval"))
CLEAR_FRAC_MIN = 0.95
SITES = {"CS": "Changsha", "LM": "Longmen"}


def season_from_doy(doy):
    doy = int(doy)
    if 60 <= doy <= 151:
        return "Spring"
    if 152 <= doy <= 243:
        return "Summer"
    if 244 <= doy <= 334:
        return "Autumn"
    return "Winter"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    man_path = os.path.join(ROOT, "output", "manifest",
                            f"sample_manifest_{SPLIT_TAG}.csv")
    man = list(csv.DictReader(open(man_path, encoding="utf-8", newline="")))
    print(f"manifest: {man_path}  ({len(man)} rows)")

    # one path per unique basename; the split directories are hard links
    seen = {}
    for p in glob.glob(os.path.join(DATASET, "**", "*.npz"), recursive=True):
        b = os.path.basename(p)
        if not b.startswith("Tensor_"):
            continue
        seen.setdefault(b, p)

    groups = collections.defaultdict(list)
    for r in man:
        if r["split"] not in (f"train_{SPLIT_TAG}", f"val_{SPLIT_TAG}"):
            continue
        p = seen.get(r["fname"])
        if p is None:
            continue
        groups[(SITES[r["station"]], season_from_doy(r["doy"]))].append((r["fname"], p))

    print(f"train+val scenes available: {sum(len(v) for v in groups.values())}")
    print()

    provenance = {}
    for key in sorted(groups):
        site, seas = key
        cand = groups[key]
        chosen, arrays = [], []
        for fname, p in cand:
            try:
                d = np.load(p, allow_pickle=True)
                ym = np.squeeze(d["Y_mask"])
                frac = float(((ym == 0) | (ym == 1)).mean())
                if frac >= CLEAR_FRAC_MIN:
                    chosen.append({"fname": fname, "clear_frac": round(frac, 4)})
                    arrays.append(np.asarray(d["X_dnb"], dtype=np.float64))
            except Exception as e:  # noqa: BLE001
                print(f"  skip {fname}: {e}")

        if not arrays:
            print(f"  {site}/{seas}: NO clear scene at >= {CLEAR_FRAC_MIN:.0%} - skipped")
            continue

        stack = np.array(arrays)
        comp = np.nanmedian(stack, axis=0).astype(np.float32)

        # geometry anchors, same convention as basemap_make.py
        ref = np.load(cand[0][1], allow_pickle=True)
        out = os.path.join(OUT_DIR, f"Super_Basemap_{site}_{seas}.npz")
        np.savez_compressed(out, X_dnb=comp,
                            Lat=ref["Lat"], Lon=ref["Lon"],
                            Radar_Loc=ref["Radar_Loc"],
                            Season=np.array(seas))

        provenance[f"{site}_{seas}"] = {
            "n_selected": len(chosen),
            "n_candidates": len(cand),
            "clear_frac_min": CLEAR_FRAC_MIN,
            "split_tag": SPLIT_TAG,
            "sources": chosen,
        }
        print(f"  {site:>9}/{seas:<7} selected {len(chosen):>3} of {len(cand):>3} "
              f"train+val scenes  -> {os.path.basename(out)}")

    prov_path = os.path.join(OUT_DIR, "basemap_provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump({"criterion": f"train+val scene with >= {CLEAR_FRAC_MIN:.0%} CLDMSK-clear pixels",
                   "split_tag": SPLIT_TAG, "groups": provenance}, f, indent=1)
    print(f"\nwrote {prov_path}")
    print(f"wrote {len(provenance)} composites to {OUT_DIR}")


if __name__ == "__main__":
    main()
