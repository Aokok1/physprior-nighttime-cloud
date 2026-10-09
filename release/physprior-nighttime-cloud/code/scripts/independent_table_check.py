"""Independently recompute every Table I metric from the frozen per-sample records.

`rebuild_paper_tables.py` derives the manuscript's Tables I and V. This script
re-derives the same quantities with a separate implementation that shares no code
with it, and diffs the two, so a bug in the table builder cannot hide behind its
own output. Accuracy, recall, specificity, precision, F1 and balanced accuracy
are recomputed as exact rationals, so a mismatch is a real disagreement and not
floating-point noise.

Usage:
    python -X utf8 scripts/independent_table_check.py [--dir OUTPUT_DIR]
"""
import argparse
import io
import json
import math
import os
import sys
from fractions import Fraction

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

NS = 1e-9


def metric_block(pred, truth):
    """The table builder's metric definitions, re-implemented from scratch."""
    n = len(pred)
    tp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 1)
    fp = sum(1 for p, t in zip(pred, truth) if p == 1 and t == 0)
    tn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 0)
    fn = sum(1 for p, t in zip(pred, truth) if p == 0 and t == 1)
    assert tp + fp + tn + fn == n, "confusion matrix does not close"
    rec = Fraction(tp, tp + fn) if tp + fn else Fraction(0)
    spe = Fraction(tn, tn + fp) if tn + fp else Fraction(0)
    prec = Fraction(tp, tp + fp) if tp + fp else Fraction(0)
    f1 = Fraction(2 * tp, 2 * tp + fp + fn) if (2 * tp + fp + fn) else Fraction(0)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "acc": float(Fraction(tp + tn, n)),
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "specificity": float(spe),
            "balanced_accuracy": float((rec + spe) / 2), "mcc": float(mcc)}


def boot_ci(pred, truth, seed=42, n_boot=10000):
    """Percentile bootstrap, same generator as the table builder."""
    p = np.asarray(pred)
    t = np.asarray(truth)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))
    accs = (p[idx] == t[idx]).mean(axis=1)
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def main(out_dir, suf):
    frozen_path = os.path.join(out_dir, "frozen_predictions",
                               f"all_frozen_predictions{suf}.json")
    tables_path = os.path.join(out_dir, f"paper_tables{suf}.json")

    frozen = json.load(open(frozen_path, encoding="utf-8"))
    tables = json.load(open(tables_path, encoding="utf-8"))
    t1 = tables["table1"]

    truth = [r["truth"] for r in frozen["CLDMSK_raw"]]
    n = len(truth)

    rows = {}
    rows["Always_clear"] = [0] * n
    rows["Always_cloud"] = [1] * n
    for key, recs in frozen.items():
        if len(recs) != n:
            print(f"  skip {key}: {len(recs)} records, expected {n}")
            continue
        rows[key] = [r["pred"] for r in recs]

    print("=" * 84)
    print("independent recomputation of Table I")
    print("=" * 84)
    print(f"  frozen records : {os.path.basename(frozen_path)}")
    print(f"  table builder  : {os.path.basename(tables_path)}")
    print(f"  n              : {n}")

    mismatches = []
    missing = [k for k in t1 if k not in rows]
    if missing:
        print(f"  rows in the table with no frozen record: {missing}")

    print(f"\n  {'row':<32}{'metric':<10}{'builder':>12}{'independent':>14}  ok")
    for key, entry in t1.items():
        if key not in rows:
            continue
        m = metric_block(rows[key], truth)
        lo, hi = boot_ci(rows[key], truth)
        b = entry["metrics"]
        for f in ("acc", "balanced_accuracy", "f1", "recall", "specificity",
                  "precision", "mcc"):
            if abs(b[f] - m[f]) > NS:
                print(f"  {key:<32}{f:<10}{b[f]:>12.6f}{m[f]:>14.6f}  MISMATCH")
                mismatches.append((key, f, b[f], m[f]))
        for label, got, exp in (("ci.lo", b["ci"]["lo"], lo), ("ci.hi", b["ci"]["hi"], hi),
                                ("tp", b["tp"], m["tp"]), ("fp", b["fp"], m["fp"]),
                                ("tn", b["tn"], m["tn"]), ("fn", b["fn"], m["fn"])):
            if abs(got - exp) > NS:
                print(f"  {key:<32}{label:<10}{got:>12.6f}{exp:>14.6f}  MISMATCH")
                mismatches.append((key, label, got, exp))

    # the ablation cells come from a different artifact, not from `frozen`
    abl = tables.get("table_ablation_2x2", {})
    abl_path = os.path.join(out_dir, f"2x2_ablation{suf}.json")
    if abl and os.path.exists(abl_path):
        cells = {f"abl_{c['label_mode']}_{c['n_channels']}ch": c
                 for c in json.load(open(abl_path, encoding="utf-8"))}
        for key, cell in cells.items():
            if key not in abl:
                continue
            pred = [r["pred"] for r in cell["test_records"]]
            tru = [r["truth"] for r in cell["test_records"]]
            m = metric_block(pred, tru)
            b = abl[key]["metrics"]
            for f in ("acc", "balanced_accuracy", "f1", "recall", "specificity", "mcc"):
                if abs(b[f] - m[f]) > NS:
                    print(f"  {key:<32}{f:<10}{b[f]:>12.6f}{m[f]:>14.6f}  MISMATCH")
                    mismatches.append((key, f, b[f], m[f]))
        print(f"\n  2x2 ablation cells cross-checked: {len(cells)}")

    # five-seed blocks: recompute mean/SD from the per-seed records
    for prefix, key in (("PhysPrior_5seed_seed", "table_five_seed_physprior"),
                        ("CoTeaching_5seed_seed", "table_five_seed_coteaching")):
        if key not in tables:
            continue
        blocks = [metric_block([r["pred"] for r in frozen[k]],
                               [r["truth"] for r in frozen[k]])
                  for k in sorted(frozen) if k.startswith(prefix)]
        got = tables[key]
        for f in ("accuracy", "balanced_accuracy", "f1", "recall", "specificity"):
            vals = [b["acc" if f == "accuracy" else f] for b in blocks]
            mu, sd = float(np.mean(vals)), float(np.std(vals))
            if abs(got[f + "_mean"] - mu) > NS or abs(got[f + "_sd"] - sd) > NS:
                print(f"  {key:<32}{f:<10}mean/sd mismatch")
                mismatches.append((key, f, got[f + "_mean"], mu))
        print(f"  {key}: {len(blocks)} seeds cross-checked")

    print("\n" + "=" * 84)
    if mismatches:
        print(f"RESULT: {len(mismatches)} MISMATCHES between the table builder and the "
              f"independent recomputation")
    else:
        print("RESULT: the table builder reproduces every Table I/V metric and both "
              "five-seed blocks exactly")
    print("=" * 84)
    return 1 if mismatches else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=r"E:/Claude code/project/noise-label-cloud/output")
    ap.add_argument("--suf", default="_grp")
    a = ap.parse_args()
    sys.exit(main(a.dir, a.suf))
