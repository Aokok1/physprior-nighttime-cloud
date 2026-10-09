"""Assemble the release bundle the manuscript's Data and Code Availability promises.

The manuscript now states that the per-sample prediction records behind every
row of Tables I-VI, the 357 radar-collocated reference labels, the eight
station-season basemap composites with their provenance record, the per-patch
split manifest, and all analysis code are released. This script builds exactly
that set, plus a README and a file-level manifest with sizes and hashes, so the
promise is checkable rather than aspirational.

Raw Ka-band radar volumes and VIIRS granules are deliberately NOT included.

Usage:
    python -X utf8 scripts/build_release.py [--out DIR]
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import shutil
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"E:/Claude code/project/noise-label-cloud"
DATASET = r"E:/Data/Unet_Dataset"
SUF = "_grp"

CODE_GLOBS = [
    "determinism.py", "train.py", "config.py",
    "data/*.py", "models/*.py", "methods/*.py",
    "scripts/build_split_grp.py", "scripts/rebuild_basemap.py",
    "scripts/rebuild_paper_tables.py", "scripts/derive_station_season_tables.py",
    "scripts/freeze_predictions_grp.py", "scripts/redo_loss_correction.py",
    "scripts/train_physprior_5seeds.py", "scripts/train_coteaching_5seeds.py",
    "scripts/train_2x2_ablation.py", "scripts/train_physprior_coteaching_long.py",
    "scripts/rerun_grp.ps1", "scripts/verify_manuscript_numbers.py",
    "scripts/verify_reproducibility.py", "scripts/verify_reproducibility.ps1",
    "scripts/independent_table_check.py", "scripts/audit_manuscript_numbers.py",
    "scripts/probe_determinism.py",
    "scripts/make_fig1.py", "scripts/verify_fig1_caption.py",
    "scripts/dev_threshold_grid.py",
]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "release",
                                                  "physprior-nighttime-cloud"))
    a = ap.parse_args()

    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    for sub in ("data", "code", "basemap", "predictions"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    entries = []

    def add(src, dst):
        if not os.path.exists(src):
            print(f"  MISSING {src}")
            return False
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        entries.append({"path": os.path.relpath(dst, a.out),
                        "bytes": os.path.getsize(dst),
                        "sha256": sha256(dst)})
        return True

    print("=== 1. reference labels ===")
    man_path = os.path.join(ROOT, "output", "manifest", f"sample_manifest{SUF}.csv")
    rows = list(csv.DictReader(open(man_path, encoding="utf-8", newline="")))
    lab = os.path.join(a.out, "data", "reference_labels.csv")
    EVAL_SPLITS = {f"train{SUF}", f"val{SUF}", "test"}
    with open(lab, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "station", "year", "doy", "split",
                    "radar_label", "radar_row", "radar_col"])
        n = 0
        skipped = 0
        for r in rows:
            if str(r.get("has_radar", "")).lower() not in ("true", "1"):
                continue
            if r.get("split") not in EVAL_SPLITS:
                # radar-labelled patches held out of all three splits
                skipped += 1
                continue
            rr = (r.get("radar_loc", "") or "").strip("[]").replace(" ", "")
            row, col = (rr.split(",") + ["", ""])[:2] if rr else ("", "")
            w.writerow([r["fname"], r["station"], r["year"], r["doy"], r["split"],
                        r.get("radar_label", ""), row, col])
            n += 1
    print(f"  wrote {n} radar-collocated reference labels in train/val/test"
          f" ({skipped} further labelled patches are held out of all three)")
    entries.append({"path": "data/reference_labels.csv",
                    "bytes": os.path.getsize(lab), "sha256": sha256(lab)})

    print("=== 2. split manifest ===")
    add(man_path, os.path.join(a.out, "data", f"sample_manifest{SUF}.csv"))
    add(os.path.join(ROOT, "output", "manifest", "split_grp_manifest.json"),
        os.path.join(a.out, "data", "split_grp_manifest.json"))

    print("=== 3. basemap composites + provenance ===")
    for p in sorted(glob.glob(os.path.join(DATASET, "base_map_trainval", "*.npz"))):
        add(p, os.path.join(a.out, "basemap", os.path.basename(p)))
    add(os.path.join(DATASET, "base_map_trainval", "basemap_provenance.json"),
        os.path.join(a.out, "basemap", "basemap_provenance.json"))

    print("=== 4. per-sample prediction records ===")
    for p in (os.path.join(ROOT, "output", "frozen_predictions",
                           f"all_frozen_predictions{SUF}.json"),
              os.path.join(ROOT, "output", "frozen_predictions",
                           f"all_frozen_predictions_v2{SUF}.json"),
              os.path.join(ROOT, "output", f"paper_tables{SUF}.json"),
              os.path.join(ROOT, "output", f"derived_tables{SUF}.json"),
              os.path.join(ROOT, "output", f"dev_threshold_grid{SUF}.json")):
        add(p, os.path.join(a.out, "predictions", os.path.basename(p)))

    print("=== 5. code ===")
    for g in CODE_GLOBS:
        for src in glob.glob(os.path.join(ROOT, g)):
            add(src, os.path.join(a.out, "code", os.path.relpath(src, ROOT)))

    with open(os.path.join(a.out, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump({"files": entries,
                   "n_files": len(entries),
                   "total_bytes": sum(e["bytes"] for e in entries)}, f, indent=1)

    readme = os.path.join(a.out, "README.md")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(README.format(n_files=len(entries),
                              mb=sum(e["bytes"] for e in entries) / 1e6))
    entries.append({"path": "README.md", "bytes": os.path.getsize(readme),
                    "sha256": sha256(readme)})

    total = sum(e["bytes"] for e in entries)
    print(f"\nwrote {len(entries)} files, {total/1e6:.1f} MB -> {a.out}")
    print("(raw Ka-band radar volumes and VIIRS granules are not included)")


README = """# PhysPrior release bundle

Artifacts for *Toward Improving Nighttime Cloud Detection Labels: A
Physics-Guided One-Directional Correction Approach for VIIRS CLDMSK*.

{n_files} files, {mb:.1f} MB.

## Contents

| Path | What it is |
|---|---|
| `data/reference_labels.csv` | The radar-collocated reference labels: one row per sample that has a radar label, with station, date, split and the radar row/column. Derived from Ka-band radar; the raw volumes are not redistributed. |
| `data/sample_manifest_grp.csv` | The per-patch split manifest: every one of the 2,782 samples with its station, year, day-of-year, split, overpass group and radar fields. Reconstructs train/validation/test exactly. |
| `data/split_grp_manifest.json` | Split-construction record (seed, validation fraction, exclusion window, closure counts). |
| `basemap/*.npz` | The eight station-season DNB composites used as the third input channel (2 stations x 4 seasons). |
| `basemap/basemap_provenance.json` | Which scenes entered each composite, and under what criterion. |
| `predictions/all_frozen_predictions_grp.json` | Per-sample predictions for every method row of Tables I-VI. Every reported metric and bootstrap interval is derived from this file. |
| `predictions/paper_tables_grp.json`, `predictions/derived_tables_grp.json` | The table builder's output, including confusion matrices, intervals and the rule-versus-model agreement block. |
| `code/` | Analysis code: the correction rule, the dataset loader, the split builder, the basemap builder, the training entry points, the table builders and the verification gates. |

## Reproducing the reported numbers

1. Recompute every metric and interval from the released predictions:
   `python code/scripts/rebuild_paper_tables.py` (needs `predictions/` on
   `output/`).
2. Check that the manuscript's numbers trace to those artifacts:
   `python code/scripts/verify_manuscript_numbers.py`.
3. Recompute the tables independently of the builder:
   `python code/scripts/independent_table_check.py`.

## Reproducing the corrected labels

The correction rule is one-directional: a CLDMSK "cloud" pixel becomes "clear"
only when the M15 brightness temperature is at least `T_th` and the population
standard deviation of the M15 field over a 5x5 window is at most `sigma_th`.
Presets are (266, 1.0), (264, 1.5) and (262, 2.0) K. The label field is the
product's `Integer_Cloud_Mask` with its confidence ordering reversed,
`y = 3 - Integer_Cloud_Mask`. The rule's development-set and test-set behaviour,
including its flip counts, is recorded in the supplementary material.

## Not included

Raw Ka-band radar volumes and VIIRS granules. The reference labels above are the
derived per-sample product of the radar processing, not the radar data itself;
they are available to the editor and reviewers on request.
"""


if __name__ == "__main__":
    main()
