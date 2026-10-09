"""Regenerate the per-station, per-season and rule-agreement tables on the new split.

Tables III, IV and VI of the manuscript all describe one saved single-checkpoint
PhysPrior+DL model, so they have to be recomputed whenever that checkpoint is
retrained. This reads the frozen predictions of the new checkpoint plus the direct
PhysPrior label rule and writes output/derived_tables_grp.json.
"""
import collections
import csv
import datetime
import io
import json
import math
import os
import sys

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud"
TAG = os.environ.get("NL_CKPT_TAG", "")
SUF = f"_{TAG}" if TAG else ""

man = {r["fname"]: r for r in csv.DictReader(
    open(os.path.join(ROOT, "output", "manifest",
                      "sample_manifest_grp.csv" if TAG == "grp" else "sample_manifest.csv"),
         encoding="utf-8", newline=""))}

recs = json.load(open(os.path.join(ROOT, "output", "frozen_predictions",
                                   f"all_frozen_predictions{SUF}.json"), encoding="utf-8"))
model = {r["fname"]: r for r in recs["PhysPrior_DL_3ch"]}
rule = {r["fname"]: r for r in recs["PhysPriorLabel_moderate"]}


def season(doy):
    d = datetime.date(2020, 1, 1) + datetime.timedelta(days=int(doy) - 1)
    return {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn",
            11: "Autumn"}[d.month]


def metrics(pairs):
    tp = sum(1 for p, t in pairs if p == 1 and t == 1)
    fp = sum(1 for p, t in pairs if p == 1 and t == 0)
    tn = sum(1 for p, t in pairs if p == 0 and t == 0)
    fn = sum(1 for p, t in pairs if p == 0 and t == 1)
    n = tp + fp + tn + fn
    acc = (tp + tn) / n
    rec = tp / (tp + fn) if tp + fn else 0
    spe = tn / (tn + fp) if tn + fp else 0
    prec = tp / (tp + fp) if tp + fp else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": acc,
            "precision": prec, "recall": rec, "specificity": spe, "f1": f1,
            "balanced_accuracy": (rec + spe) / 2,
            "mcc": (tp * tn - fp * fn) / den if den else 0.0}


def boot_ci(pairs, seed=42, n_boot=10000):
    p = np.array([x[0] for x in pairs])
    t = np.array([x[1] for x in pairs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))
    a = (p[idx] == t[idx]).mean(axis=1)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


fn_list = [f for f in model if f in man and man[f]["split"] == "test"]
print(f"test patches with a model record: {len(fn_list)}")

out = {}

# ---- per station ------------------------------------------------------------
st = {}
for s in ("CS", "LM"):
    pairs = [(model[f]["pred"], model[f]["truth"]) for f in fn_list if man[f]["station"] == s]
    m = metrics(pairs)
    st[s] = m
    print(f"  {s}: n={m['n']} acc={m['acc']*100:.1f}% P={m['precision']*100:.1f}% "
          f"R={m['recall']*100:.1f}% Sp={m['specificity']*100:.1f}%")
allp = [(model[f]["pred"], model[f]["truth"]) for f in fn_list]
st["combined"] = metrics(allp)
out["per_station"] = st

# ---- per season -------------------------------------------------------------
se = {}
for s in ("Spring", "Summer", "Autumn", "Winter"):
    pairs = [(model[f]["pred"], model[f]["truth"]) for f in fn_list if season(man[f]["doy"]) == s]
    m = metrics(pairs)
    lo, hi = boot_ci(pairs)
    m["ci"] = {"lo": lo, "hi": hi}
    se[s] = m
    print(f"  {s}: n={m['n']} acc={m['acc']*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]")
out["per_season"] = se

# ---- agreement between the direct rule and the DL model ---------------------
# `model_*` is the trained network, `rule_*` is the direct PhysPrior label. The
# two counter names below were previously `botonly` (incremented for the model)
# and `dlnonly` (incremented for the rule) and were exported under each other's
# names, which swapped the two rows of the manuscript's agreement table.
both = model_only = rule_only = neither = agree = 0
for f in fn_list:
    t = model[f]["truth"]
    mc = model[f]["pred"] == t
    rc = rule[f]["pred"] == t
    both += mc and rc
    model_only += mc and not rc
    rule_only += rc and not mc
    neither += (not mc) and (not rc)
    agree += model[f]["pred"] == rule[f]["pred"]
n = len(fn_list)
b = sum(1 for f in fn_list if model[f]["pred"] == 1 and rule[f]["pred"] == 0)
c = sum(1 for f in fn_list if model[f]["pred"] == 0 and rule[f]["pred"] == 1)
# exact McNemar two-sided
from math import comb
k = min(b, c)
p_val = min(1.0, 2 * sum(comb(b + c, i) for i in range(k + 1)) / (2 ** (b + c))) if (b + c) else 1.0
po = agree / n
pe = ((sum(1 for f in fn_list if model[f]["pred"] == 1) / n) *
      (sum(1 for f in fn_list if rule[f]["pred"] == 1) / n) +
      (sum(1 for f in fn_list if model[f]["pred"] == 0) / n) *
      (sum(1 for f in fn_list if rule[f]["pred"] == 0) / n))
kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0
assert both + model_only + rule_only + neither == n, "agreement block does not close"
out["agreement"] = {"n": n,
                    "model_correct": both + model_only,
                    "rule_correct": both + rule_only,
                    "both_correct": both,
                    "rule_only_correct": rule_only,
                    "dl_only_correct": model_only,
                    "both_wrong": neither, "agree": agree, "kappa": kappa,
                    "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p_val}
print(f"\n  agreement {agree}/{n} ({po*100:.2f}%)  kappa={kappa:+.3f}  "
      f"McNemar b={b} c={c} p={p_val:.3f}")
print(f"  model correct {both + model_only}  rule correct {both + rule_only}  "
      f"both {both}  model only {model_only}  rule only {rule_only}  neither {neither}")

# ---- confusion matrix of the direct rule on the test set --------------------
rr = [(rule[f]["pred"], rule[f]["truth"]) for f in fn_list]
m = metrics(rr)
out["direct_rule_confusion"] = m
print(f"\n  direct rule: TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']} "
      f"acc={m['acc']*100:.2f}% R={m['recall']*100:.1f}% Sp={m['specificity']*100:.1f}% "
      f"F1={m['f1']*100:.1f}% MCC={m['mcc']:+.3f}")

with open(os.path.join(ROOT, "output", f"derived_tables{SUF}.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(f"\nwrote output/derived_tables{SUF}.json")
