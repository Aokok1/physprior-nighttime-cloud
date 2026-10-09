# PhysPrior — physics-guided label correction for nighttime cloud detection

Code and data release for the manuscript

> **Toward Improving Nighttime Cloud Detection Labels: A Physics-Guided
> One-Directional Correction Approach for VIIRS CLDMSK**
> Mingyu Chen, Shensen Hu, Weihua Ai, Shuo Ma
> College of Meteorology and Oceanography, National University of Defense Technology

## What PhysPrior does

NASA's VIIRS CLDMSK cloud mask over land at night confuses radiatively cooled
surfaces with cloud. PhysPrior corrects that bias **in the label field**, not by
building another detector: it changes a CLDMSK "cloud" label to "clear" only when
both conditions hold at the pixel, and it never modifies a CLDMSK "clear" label.

| | condition |
|---|---|
| 1 | M15 brightness temperature at the pixel, `T15 >= T_th` (warm) |
| 2 | population standard deviation of M15 in the local 5x5 window, `sigma5 <= sigma_th` (uniform) |

The deployed operating point is the **moderate** member of a three-preset family:
conservative `(266 K, 1.0 K)`, **moderate `(264 K, 1.5 K)`**, aggressive
`(262 K, 2.0 K)`. The family spans the M15 warm-cloud discriminator range given
by the MOD35 algorithm-theoretical-basis document. On the 197-point radar
reference the rule raises direct radar agreement from 44.2% to 69.5% while
changing 74 labels (62 corrections, 12 over-corrections).

## Released artefacts

`release/physprior-nighttime-cloud/` (54 files, 3.3 MB) is the package that
accompanies the manuscript:

| Path | What it is |
|---|---|
| `data/reference_labels.csv` | radar-collocated reference labels (station, date, split, radar row/column) |
| `data/sample_manifest_grp.csv` | per-patch split manifest: all 2,782 samples with station, year, day-of-year, split, overpass group and radar fields |
| `data/split_grp_manifest.json` | split-construction record (seed, validation fraction, exclusion window, closure counts) |
| `basemap/*.npz` | the eight station-season DNB composites used as the third input channel |
| `basemap/basemap_provenance.json` | which scenes entered each composite, under which criterion |
| `predictions/all_frozen_predictions_grp.json` | per-sample predictions behind every row of Tables I-VI |
| `predictions/paper_tables_grp.json`, `predictions/derived_tables_grp.json` | table-builder output (confusion matrices, intervals, rule-versus-model agreement) |
| `code/` | the correction rule, dataset loader, split builder, basemap builder, training entry points, table builders and verification gates |

### Reproducing the reported numbers

```bash
# 1. every metric and bootstrap interval, from the released predictions
python code/scripts/rebuild_paper_tables.py

# 2. forward gate: every value the manuscript asserts must be produced
python code/scripts/verify_manuscript_numbers.py

# 3. independent check of the tables, bypassing the builder
python code/scripts/independent_table_check.py
```

## Repository layout

| Path | What it is | Where in the paper |
|---|---|---|
| `methods/physical_prior.py` | the correction rule and its three presets | II-B |
| `methods/loss_correction.py` | Loss Correction baseline (Patrini et al., CVPR 2017); the noise-transition matrix is re-estimated from the 160 training+validation radar pixels (`scripts/redo_loss_correction.py`), never from the test set | II-E, Table I |
| `methods/coteaching.py` | Co-Teaching baseline (Han et al., NeurIPS 2018) | II-E, Table I |
| `methods/gce_baseline.py` | GCE baseline (Zhang & Sabuncu, NeurIPS 2018), `q = 0.7` | II-E, Table I |
| `methods/mixup.py` | Mixup baseline (Zhang et al., ICLR 2018), `alpha = 0.4` | II-E, Table I |
| `methods/baseline.py` | standard training on raw CLDMSK labels | Table I |
| `models/mt_unet.py` | the segmentation backbone used for every trained row | II-E |
| `scripts/build_split_grp.py` | overpass-disjoint, temporally clean split | III-A |
| `scripts/rebuild_basemap.py` | the eight station-season DNB composites (training and validation scenes only) | II-C |
| `scripts/train_physprior_5seeds.py`, `scripts/train_coteaching_5seeds.py`, `scripts/train_2x2_ablation.py` | the five-seed runs and the fixed-budget 2x2 ablation | III-F, III-G |
| `scripts/build_paper_tables.py`, `scripts/rebuild_paper_tables.py` | table generation from the per-sample records | Tables I-VI |
| `scripts/make_fig1.py`, `scripts/verify_fig1_caption.py` | Fig. 1 and the recomputation of every number in its caption | Fig. 1 |
| `scripts/verify_manuscript_numbers.py`, `scripts/audit_manuscript_numbers.py`, `scripts/independent_table_check.py` | the forward gate, the reverse audit and an independent table check | - |
| `determinism.py` | seeding of every entry point and the data loader | II-E |
| `train.py` | entry point: `python train.py --method {baseline,loss_correction,coteaching,gce,mixup,selfie,...}` | - |

`methods/selfie.py` (a teacher-student label-purification variant) is kept for
completeness but is **not** used anywhere in the manuscript.

## Data

Released: the radar-collocated reference labels, the per-patch split manifest,
the eight basemap composites with provenance, and the per-sample prediction
records behind every reported number.

Not redistributed: the raw Ka-band radar volumes (institutional restriction; the
derived reference labels are released instead) and the per-patch `.npz` arrays.
VIIRS CLDMSK L2 granules are public from the NOAA Comprehensive Large
Array-Data Stewardship System (CLASS).

## Environment

Python 3.12 with PyTorch (CUDA); see `requirements.txt`. The reported runs used a
single NVIDIA RTX 4060 Laptop GPU, batch size 16, 1,971 training patches of
128x128.

## Citation

```bibtex
@article{chen_physprior_nighttime,
  title  = {Toward Improving Nighttime Cloud Detection Labels: A Physics-Guided
            One-Directional Correction Approach for VIIRS CLDMSK},
  author = {Chen, Mingyu and Hu, Shensen and Ai, Weihua and Ma, Shuo},
  note   = {submitted to IEEE Geoscience and Remote Sensing Letters}
}
```

## Contact

Shensen Hu (corresponding author), hushensen18@nudt.edu.cn
