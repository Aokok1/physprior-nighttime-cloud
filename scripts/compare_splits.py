"""Side-by-side comparison of the published (phase4) and re-run (grp) splits.

Reads output/paper_tables.json for the original split and
output/paper_tables_grp.json for the group-disjoint split, and prints the
per-row deltas that the manuscript has to be updated with.

Run:  python scripts/compare_splits.py
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud"


def load(p):
    fp = os.path.join(ROOT, p)
    return json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else None


old = load("output/paper_tables.json")
new = load("output/paper_tables_grp.json")

ROWS = [
    ("CLDMSK_raw", "Raw CLDMSK"),
    ("PureM15_T264", "M15-only T=264 K"),
    ("PureM15_T266", "M15-only T=266 K"),
    ("PhysPriorLabel_conservative", "PhysPrior labels, conservative"),
    ("PhysPriorLabel_moderate", "PhysPrior labels, moderate"),
    ("PhysPriorLabel_aggressive", "PhysPrior labels, aggressive"),
    ("abl_raw_2ch", "Ablation: raw labels, 2ch"),
    ("abl_raw_3ch", "Ablation: raw labels, 3ch"),
    ("abl_physprior_2ch", "Ablation: PhysPrior labels, 2ch"),
    ("abl_physprior_3ch", "Ablation: PhysPrior labels, 3ch"),
    ("Standard_training", "Standard training (3ch)"),
    ("LossCorrection", "Loss Correction"),
    ("CoTeaching_A", "Co-Teaching A"),
    ("CoTeaching_B", "Co-Teaching B"),
    ("GCE_q07", "GCE q=0.7"),
    ("Mixup", "Mixup"),
    ("PhysPrior_DL_3ch", "PhysPrior + DL (3ch)"),
    ("PhysPrior_CoTeaching_ensemble", "PhysPrior + CoT ensemble"),
]

print("=" * 104)
print("TABLE I  —  published split (phase4)  vs  group-disjoint split (grp)")
print("=" * 104)
print(f"{'row':34s}{'phase4 acc':>11s}{'grp acc':>10s}{'delta':>9s}   {'grp BA':>7s}{'grp F1':>8s}{'grp MCC':>9s}")
for key, label in ROWS:
    o = (old or {}).get("table1", {}).get(key)
    n = (new or {}).get("table1", {}).get(key)
    if n is None:
        print(f"{label:34s}{(f'{o[chr(97)+chr(99)+chr(99)]*100:.2f}' if o else '--'):>11s}{'--':>10s}{'--':>9s}")
        continue
    na = n["metrics"]["acc"] * 100
    if o:
        oa = o["metrics"]["acc"] * 100
        print(f"{label:34s}{oa:11.2f}{na:10.2f}{na-oa:+9.2f}   "
              f"{n['metrics']['balanced_accuracy']*100:7.1f}{n['metrics']['f1']*100:8.1f}"
              f"{n['metrics']['mcc']:+9.3f}")
    else:
        print(f"{label:34s}{'--':>11s}{na:10.2f}{'--':>9s}   "
              f"{n['metrics']['balanced_accuracy']*100:7.1f}{n['metrics']['f1']*100:8.1f}"
              f"{n['metrics']['mcc']:+9.3f}")

print("\n" + "=" * 104)
print("2x2 ABLATION")
print("=" * 104)
for tag, src in (("phase4", old), ("grp", new)):
    if not src or "table_ablation_deltas" not in src:
        continue
    d = src["table_ablation_deltas"]
    a = src["table_ablation_2x2"]
    cells = {k: v["metrics"]["acc"] * 100 for k, v in a.items()}
    print(f"  [{tag}] raw2={cells.get('abl_raw_2ch', float('nan')):.1f} "
          f"raw3={cells.get('abl_raw_3ch', float('nan')):.1f} "
          f"pp2={cells.get('abl_physprior_2ch', float('nan')):.1f} "
          f"pp3={cells.get('abl_physprior_3ch', float('nan')):.1f}")
    print(f"         basemap(raw) {d['basemap_effect_raw_pp']:+.2f}  "
          f"basemap(pp) {d['basemap_effect_physprior_pp']:+.2f}  "
          f"PhysPrior(2ch) {d['physprior_effect_2ch_pp']:+.2f}  "
          f"PhysPrior(3ch) {d['physprior_effect_3ch_pp']:+.2f}")

print("\n" + "=" * 104)
print("FIVE-SEED SUMMARIES")
print("=" * 104)
for tag, src in (("phase4", old), ("grp", new)):
    if not src:
        continue
    for key, label in (("table_five_seed_physprior", "PhysPrior+DL"),
                       ("table_five_seed_coteaching", "Co-Teaching"),
                       ("table_five_seed", "PhysPrior+DL")):
        d = src.get(key)
        if not d:
            continue
        print(f"  [{tag}] {label:16s} acc {d['accuracy_mean']*100:.2f} +/- {d['accuracy_sd']*100:.2f}"
              f"   BA {d['balanced_accuracy_mean']*100:.2f} +/- {d['balanced_accuracy_sd']*100:.2f}"
              f"   F1 {d['f1_mean']*100:.2f} +/- {d['f1_sd']*100:.2f}")
        print(f"          per-seed acc: {[round(p['acc']*100,2) for p in d['per_seed']]}")
        break
