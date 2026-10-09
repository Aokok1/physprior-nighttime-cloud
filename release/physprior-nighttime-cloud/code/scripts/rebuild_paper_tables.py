"""Rebuild the single source of truth for the paper's tables.

Consolidates every row that appears in the manuscript into one per-sample
prediction file, then derives all metrics and bootstrap CIs with one documented
generator, so that a reader (or build_paper_tables.py) can regenerate Tables I-VI
from a named artifact.

Row -> record source
  trivial baselines      derived from the radar truth vector
  direct rows            output/frozen_predictions/all_frozen_predictions{SUF}.json
  trained networks       same file (frozen by scripts/freeze_predictions_grp.py)
  2x2 ablation cells     output/2x2_ablation{SUF}.json  test_records
  five-seed runs         output/physprior_5seed_results{SUF}.json
                         output/coteaching_5seed_results{SUF}.json

Run:  NL_CKPT_TAG=grp python scripts/rebuild_paper_tables.py
Out:  output/frozen_predictions/all_frozen_predictions_v2{SUF}.json
      output/paper_tables{SUF}.json
"""
import json
import math
import os

import numpy as np

ROOT = r"E:/Claude code/project/noise-label-cloud"
OUT = os.path.join(ROOT, "output")
FROZEN = os.path.join(OUT, "frozen_predictions")
TAG = os.environ.get("NL_CKPT_TAG", "")
SUF = f"_{TAG}" if TAG else ""
SEED = 42
N_BOOT = 10000


def boot_ci(pred, truth, seed=SEED, n_boot=N_BOOT):
    """Percentile bootstrap CI for accuracy, one documented generator."""
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    n = len(pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    accs = (pred[idx] == truth[idx]).mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def metrics(pred, truth):
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    n = tp + fp + tn + fn
    acc = (tp + tn) / n
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    lo, hi = boot_ci(pred, truth)
    return {
        "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "acc": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "specificity": float(spe),
        "balanced_accuracy": float((rec + spe) / 2), "mcc": float(mcc),
        "prevalence": float((tp + fn) / n),
        "ci": {"mean": float(acc), "lo": lo, "hi": hi},
    }


def load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pairs(records):
    return ([r["pred"] for r in records], [r["truth"] for r in records])


frozen = load(f"output/frozen_predictions/all_frozen_predictions{SUF}.json")
if frozen is None:
    raise SystemExit(f"missing frozen predictions for tag '{TAG}'")

store = {}
truth_recs = frozen["CLDMSK_raw"]
store["Always_clear"] = [{"fname": r["fname"], "pred": 0, "truth": r["truth"], "valid": True}
                         for r in truth_recs]
store["Always_cloud"] = [{"fname": r["fname"], "pred": 1, "truth": r["truth"], "valid": True}
                         for r in truth_recs]

for k, recs in frozen.items():
    store[k] = [{"fname": r["fname"], "pred": r["pred"], "truth": r["truth"], "valid": True}
                for r in recs if r.get("valid", True)]

# 2x2 ablation
abl = load(f"output/2x2_ablation{SUF}.json")
if abl:
    for cell in abl:
        key = f"abl_{cell['label_mode']}_{cell['n_channels']}ch"
        store[key] = [{"fname": r["fname"], "pred": r["pred"], "truth": r["truth"], "valid": True}
                      for r in cell["test_records"]]

# five seeds
s5 = load(f"output/physprior_5seed_results{SUF}.json")
if s5:
    for e in s5:
        store[f"PhysPrior_5seed_seed{e['seed']}"] = [
            {"fname": r["fname"], "pred": r["pred"], "truth": r["truth"], "valid": True}
            for r in e["test_records"]]
ct5 = load(f"output/coteaching_5seed_results{SUF}.json")
if ct5:
    for e in ct5:
        store[f"CoTeaching_5seed_seed{e['seed']}"] = [
            {"fname": r["fname"], "pred": r["pred"], "truth": r["truth"], "valid": True}
            for r in e["test_records"]]

with open(os.path.join(FROZEN, f"all_frozen_predictions_v2{SUF}.json"), "w", encoding="utf-8") as f:
    json.dump(store, f, indent=1)
print(f"wrote all_frozen_predictions_v2{SUF}.json with {len(store)} methods")

# ---- derive tables ----------------------------------------------------------
TABLE1 = [
    ("Always_clear", "Always-clear (trivial baseline)"),
    ("Always_cloud", "Always-cloud (trivial baseline)"),
    ("CLDMSK_raw", "Raw CLDMSK"),
    ("PureM15_T264", "M15-only, T=264 K (matched)"),
    ("PureM15_T266", "M15-only, T=266 K (post-hoc oracle)"),
    ("PhysPriorLabel_conservative", "Direct PhysPrior, conservative"),
    ("PhysPriorLabel_moderate", "Direct PhysPrior, moderate"),
    ("PhysPriorLabel_aggressive", "Direct PhysPrior, aggressive"),
    ("Standard_training", "Standard training (raw labels, 3ch)"),
    ("LossCorrection", "Loss Correction"),
    ("CoTeaching_A", "Co-Teaching A"),
    ("CoTeaching_B", "Co-Teaching B"),
    ("GCE_q07", "GCE q=0.7"),
    ("Mixup", "Mixup"),
    # The dedicated PhysPrior+DL run, trained with the same entry point as the
    # rows above. The 2x2 ablation cells are a separate fixed-budget experiment
    # and are reported only through `table_ablation_2x2`.
    ("PhysPrior_DL_3ch", "PhysPrior moderate + DL (3ch)"),
    ("PhysPrior_CoTeaching_ensemble", "PhysPrior + Co-Teaching (predef. ensemble)"),
]

t1 = {}
print(f"\n{'row':38s}{'n':>4}{'acc':>8}{'CI':>17}{'BA':>7}{'F1':>7}{'MCC':>8}")
for key, label in TABLE1:
    if key not in store:
        print(f"{key:38s}  MISSING")
        continue
    m = metrics(*pairs(store[key]))
    t1[key] = {"label": label, "metrics": m, "ci": m["ci"]}
    print(f"{key:38s}{m['n']:>4}{m['acc']*100:8.2f}"
          f"{'[' + format(m['ci']['lo']*100,'.1f') + ',' + format(m['ci']['hi']*100,'.1f') + ']':>17}"
          f"{m['balanced_accuracy']*100:7.1f}{m['f1']*100:7.1f}{m['mcc']:+8.3f}")

out = {"generator": {"bootstrap": "numpy default_rng(seed=42), percentile 2.5/97.5",
                     "n_boot": N_BOOT, "tag": TAG or "phase4",
                     "records": f"output/frozen_predictions/all_frozen_predictions_v2{SUF}.json"},
       "table1": t1}

if abl:
    t_abl = {}
    for cell in abl:
        key = f"abl_{cell['label_mode']}_{cell['n_channels']}ch"
        m = metrics(*pairs(store[key]))
        t_abl[key] = {"label": cell["label"], "metrics": m, "ci": m["ci"],
                      "best_epoch": cell.get("best_epoch"), "best_val_acc": cell.get("best_val_acc")}
    a = {k: v["metrics"]["acc"] for k, v in t_abl.items()}
    out["table_ablation_2x2"] = t_abl
    out["table_ablation_deltas"] = {
        "basemap_effect_raw_pp": (a["abl_raw_3ch"] - a["abl_raw_2ch"]) * 100,
        "basemap_effect_physprior_pp": (a["abl_physprior_3ch"] - a["abl_physprior_2ch"]) * 100,
        "physprior_effect_2ch_pp": (a["abl_physprior_2ch"] - a["abl_raw_2ch"]) * 100,
        "physprior_effect_3ch_pp": (a["abl_physprior_3ch"] - a["abl_raw_3ch"]) * 100,
    }
    print("\n2x2 ablation:")
    for k, v in a.items():
        print(f"  {k:26s} {v*100:.4f}")
    for k, v in out["table_ablation_deltas"].items():
        print(f"  {k:30s} {v:+.4f}")


def five_summary(prefix):
    keys = [k for k in store if k.startswith(prefix)]
    if not keys:
        return None
    ms = [metrics(*pairs(store[k])) for k in sorted(keys)]
    def mm(v):
        arr = np.array(v)
        return float(arr.mean()), float(arr.std(ddof=0))
    d = {"n_seeds": len(ms),
         "per_seed": [{"acc": m["acc"], "balanced_accuracy": m["balanced_accuracy"],
                       "f1": m["f1"], "recall": m["recall"], "specificity": m["specificity"]}
                      for m in ms]}
    for name, f in (("accuracy", "acc"), ("balanced_accuracy", "balanced_accuracy"),
                    ("f1", "f1"), ("recall", "recall"), ("specificity", "specificity")):
        mu, sd = mm([m[f] for m in ms])
        d[f"{name}_mean"], d[f"{name}_sd"] = mu, sd
    return d


fs = five_summary("PhysPrior_5seed_seed")
if fs:
    out["table_five_seed_physprior"] = fs
    print(f"\nfive-seed PhysPrior+DL: acc {fs['accuracy_mean']*100:.2f} +/- {fs['accuracy_sd']*100:.2f}"
          f"  BA {fs['balanced_accuracy_mean']*100:.2f} +/- {fs['balanced_accuracy_sd']*100:.2f}"
          f"  F1 {fs['f1_mean']*100:.2f} +/- {fs['f1_sd']*100:.2f}"
          f"  R {fs['recall_mean']*100:.2f} +/- {fs['recall_sd']*100:.2f}"
          f"  Sp {fs['specificity_mean']*100:.2f} +/- {fs['specificity_sd']*100:.2f}")
cs = five_summary("CoTeaching_5seed_seed")
if cs:
    out["table_five_seed_coteaching"] = cs
    print(f"five-seed Co-Teaching:  acc {cs['accuracy_mean']*100:.2f} +/- {cs['accuracy_sd']*100:.2f}")

with open(os.path.join(OUT, f"paper_tables{SUF}.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(f"\nwrote output/paper_tables{SUF}.json")
