# PhysPrior release bundle

Artifacts for *Toward Improving Nighttime Cloud Detection Labels: A
Physics-Guided One-Directional Correction Approach for VIIRS CLDMSK*.

54 files, 3.5 MB.

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
