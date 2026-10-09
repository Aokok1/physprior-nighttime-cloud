"""Independently recompute the rule-vs-model agreement block from frozen records.

`derive_station_season_tables.py` builds the agreement block with two counters
named `botonly` (incremented when the MODEL is right and the rule wrong) and
`dlnonly` (incremented when the RULE is right and the model wrong), then exports
them as `rule_only_correct` and `dl_only_correct`. If that naming is wrong the
exported keys are swapped and every downstream table that reads them is wrong.

This script recomputes the block from the per-sample records without reusing any
of that code, and prints which assignment the data supports.

Usage:
    python -X utf8 scripts/check_rule_model_agreement.py [frozen.json]
"""
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PATH = (sys.argv[1] if len(sys.argv) > 1
        else r"E:/Claude code/project/noise-label-cloud/output/frozen_predictions/all_frozen_predictions_grp.json")
MODEL_KEY = "PhysPrior_DL_3ch"
RULE_KEY = "PhysPriorLabel_moderate"

recs = json.load(open(PATH, encoding="utf-8"))
model = {r["fname"]: r for r in recs[MODEL_KEY]}
rule = {r["fname"]: r for r in recs[RULE_KEY]}
fns = sorted(set(model) & set(rule))

model_correct = rule_correct = both = model_only = rule_only = neither = agree = 0
for f in fns:
    t = model[f]["truth"]
    assert rule[f]["truth"] == t, f"truth mismatch on {f}"
    mc = model[f]["pred"] == t
    rc = rule[f]["pred"] == t
    model_correct += mc
    rule_correct += rc
    both += mc and rc
    model_only += mc and not rc
    rule_only += rc and not mc
    neither += (not mc) and (not rc)
    agree += model[f]["pred"] == rule[f]["pred"]

n = len(fns)
b = sum(1 for f in fns if model[f]["pred"] == 1 and rule[f]["pred"] == 0)
c = sum(1 for f in fns if model[f]["pred"] == 0 and rule[f]["pred"] == 1)

print("=" * 70)
print("rule vs model agreement, recomputed from per-sample records")
print("=" * 70)
print(f"  model key : {MODEL_KEY}")
print(f"  rule key  : {RULE_KEY}")
print(f"  n         : {n}")
print()
print(f"  {'quantity':<34}{'count':>7}{'share':>10}")
for label, v in (("downstream DL correct", model_correct),
                 ("direct PhysPrior rule correct", rule_correct),
                 ("both correct", both),
                 ("DL only correct", model_only),
                 ("rule only correct", rule_only),
                 ("both incorrect", neither),
                 ("label == model (agreement)", agree)):
    print(f"  {label:<34}{v:>7}{v / n * 100:>9.2f}%")
print()
print(f"  McNemar b (model=cloud, rule=clear) : {b}")
print(f"  McNemar c (model=clear, rule=cloud) : {c}")
print()
print("  check: both + DL_only + rule_only + neither = "
      f"{both + model_only + rule_only + neither}  (must equal n={n})")
print("  check: DL correct  = both + DL_only   = "
      f"{both + model_only}  vs counted {model_correct}")
print("  check: rule correct = both + rule_only = "
      f"{both + rule_only}  vs counted {rule_correct}")

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
DERIVED = PATH.replace("frozen_predictions/all_frozen_predictions", "derived_tables")
DERIVED = DERIVED.replace(".json", ".json")
try:
    d = json.load(open(DERIVED, encoding="utf-8"))["agreement"]
except Exception as e:  # noqa: BLE001
    print(f"\n  (could not read {DERIVED}: {e})")
    sys.exit(0)

print()
print("=" * 70)
print("cross-check against the exported derived table")
print("=" * 70)
for k, truth in (("each_correct", model_correct),
                 ("both_correct", both),
                 ("rule_only_correct", rule_only),
                 ("dl_only_correct", model_only),
                 ("both_wrong", neither),
                 ("agree", agree)):
    got = d.get(k)
    flag = "ok" if got == truth else "MISMATCH"
    print(f"  {k:<20} exported={got!s:>6}  recomputed={truth:>6}  {flag}")
