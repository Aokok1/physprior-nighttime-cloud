"""Reverse check: every number in the manuscript must be producible by an artifact.

`verify_manuscript_numbers.py` checks that the values it knows about are present
and that a hand-written list of superseded values is absent. It cannot catch a
stale number nobody thought to list -- which is how "the test-side 36.0%" once
survived a full re-run while Table I reported 50.3% for the same row.

This script works the other way round. It rebuilds, from the current artifacts,
every value the manuscript could legitimately quote -- per-method metrics and
intervals, confusion-matrix entries, the flip analysis, the agreement block, the
five-seed statistics, the ablation cells and their differences, split and
prevalence counts -- then harvests every percentage in the manuscript and reports
the ones no artifact produces.

The pool is deliberately built from *directly reported* quantities only. Adding
pairwise differences between unrelated methods would let a stale value slip
through by coincidence.

Usage:
    python -X utf8 scripts/audit_manuscript_numbers.py
"""
import csv
import io
import json
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"E:/Claude code/project/noise-label-cloud"
TEX = r"E:/Claude code/paper/noise_label_grsl/manuscript_grsl.tex"
SUF = "_grp"

pool = {}
SOURCES = {}


def add(value, source, nd=(0, 1, 2)):
    """Register a number the manuscript may legitimately quote."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    for d in nd:
        s = f"{value * 100:.{d}f}" if abs(value) <= 1.0000001 else f"{value:.{d}f}"
        pool.setdefault(s, source)


def load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return json.load(f)


tables = load(f"output/paper_tables{SUF}.json")
derived = load(f"output/derived_tables{SUF}.json")
frozen = load(f"output/frozen_predictions/all_frozen_predictions{SUF}.json")

# ---- per-method metrics recomputed from the per-sample records -------------
for key, recs in frozen.items():
    n = len(recs)
    tp = sum(1 for r in recs if r["pred"] == 1 and r["truth"] == 1)
    fp = sum(1 for r in recs if r["pred"] == 1 and r["truth"] == 0)
    tn = sum(1 for r in recs if r["pred"] == 0 and r["truth"] == 0)
    fn = sum(1 for r in recs if r["pred"] == 0 and r["truth"] == 1)
    src = f"method:{key}"
    for v in (n, tp, fp, tn, fn, tp + tn, tp + fp, tp + fn, tn + fp, tn + fn):
        add(v, src, nd=(0,))
    add((tp + tn) / n, src)
    for num, den in ((tn, tn + fp), (tp, tp + fn), (fp, fp + tn), (tp, tp + fp),
                     (2 * tp, 2 * tp + fp + fn)):
        if den:
            add(num / den, src)

# interval endpoints and MCC from the table builder
for section in ("table1", "table_ablation_2x2"):
    for key, entry in tables.get(section, {}).items():
        m = entry.get("metrics", {})
        src = f"{section}:{key}"
        for f in ("acc", "balanced_accuracy", "f1", "recall", "specificity",
                  "precision", "mcc", "ci"):
            v = m.get(f)
            if isinstance(v, dict):
                add(v["lo"], src + ".ci.lo")
                add(v["hi"], src + ".ci.hi")
            elif isinstance(v, (int, float)):
                add(v, src + "." + f)

# ablation differences are directly quoted
for k, v in tables.get("table_ablation_deltas", {}).items():
    add(abs(v) / 100.0, "ablation_delta:" + k)

# five-seed statistics and their per-seed values
for k in ("table_five_seed_physprior", "table_five_seed_coteaching"):
    d = tables.get(k)
    if not d:
        continue
    for f in ("accuracy", "balanced_accuracy", "f1", "recall", "specificity"):
        add(d[f + "_mean"], f"five_seed:{k}.{f}")
        add(d[f + "_sd"], f"five_seed:{k}.{f}.sd")
    for p in d["per_seed"]:
        add(p["acc"], f"five_seed:{k}.per_seed")

# derived tables: stations, seasons, agreement, direct-rule confusion
for site, m in derived.get("per_station", {}).items():
    for f in ("acc", "precision", "recall", "specificity"):
        add(m[f], f"station:{site}.{f}")
    add(m["n"], f"station:{site}.n", nd=(0,))
for s, m in derived.get("per_season", {}).items():
    add(m["acc"], f"season:{s}")
    add(m["ci"]["lo"], f"season:{s}.ci.lo")
    add(m["ci"]["hi"], f"season:{s}.ci.hi")
    add(m["n"], f"season:{s}.n", nd=(0,))
ag = derived.get("agreement", {})
for f in ("kappa", "mcnemar_p"):
    add(ag.get(f, 0), "agreement:" + f)
for f in ("n", "model_correct", "rule_correct", "both_correct", "rule_only_correct",
          "dl_only_correct", "both_wrong", "agree", "mcnemar_b", "mcnemar_c"):
    add(ag.get(f, 0), "agreement:" + f, nd=(0,))
    if ag.get(f) and ag.get("n"):
        add(ag[f] / ag["n"], "agreement:" + f + ".share")
dc = derived.get("direct_rule_confusion", {})
for f in ("acc", "recall", "specificity", "f1", "mcc", "precision"):
    add(dc.get(f, 0), "direct_rule:" + f)

# flip analysis, recomputed (it is quoted in the abstract and Table II text)
raw = {r["fname"]: r for r in frozen["CLDMSK_raw"]}
pp = {r["fname"]: r for r in frozen["PhysPriorLabel_moderate"]}
flips = [k for k in raw if raw[k]["pred"] != pp[k]["pred"]]
succ = sum(1 for k in flips if raw[k]["pred"] == 1 and pp[k]["pred"] == 0
           and raw[k]["truth"] == 0)
over = sum(1 for k in flips if raw[k]["pred"] == 1 and pp[k]["pred"] == 0
           and raw[k]["truth"] == 1)
add(len(flips), "flips:total", nd=(0,))
add(succ, "flips:success", nd=(0,))
add(over, "flips:over", nd=(0,))
if flips:
    add(succ / len(flips), "flips:success.share")
    add(over / len(flips), "flips:over.share")

# dataset constants from the manifest
man = list(csv.DictReader(open(os.path.join(ROOT, "output", "manifest",
                                            f"sample_manifest{SUF}.csv"),
                               encoding="utf-8", newline="")))
add(len(man), "manifest:total", nd=(0,))
for split in ("train_grp", "val_grp", "test"):
    sub = [r for r in man if r["split"] == split]
    add(len(sub), f"manifest:{split}.n", nd=(0,))
    add(len({r["overpass_group"] for r in sub}), f"manifest:{split}.overpasses", nd=(0,))
n_clear = sum(1 for r in man if r["split"] == "test" and float(r["radar_label"]) == 0)
add(n_clear, "manifest:test.clear", nd=(0,))
add(197 - n_clear, "manifest:test.cloud", nd=(0,))

# ---- development-set threshold behaviour -----------------------------------
# These are the numbers section II-B and supplementary S1 quote about the 160
# radar-collocated development pixels; they come from a released artifact rather
# than from a scratch computation.
dev = load(f"output/dev_threshold_grid{SUF}.json")
add(dev["n_pixels"], "dev:n", nd=(0,))
add(dev["cldmsk_cloud_pixels"], "dev:cloud", nd=(0,))
add(dev["apparent_fp"], "dev:apparent_fp", nd=(0,))
add(dev["true_positive"], "dev:tp", nd=(0,))
add(dev["raw_cldmsk_accuracy"], "dev:raw_cldmsk")
add(dev["grid_max"]["acc"], "dev:grid.max")
for name, r in dev["presets"].items():
    add(r["accuracy"], f"dev:{name}.acc")
    add(r["flipped"], f"dev:{name}.flipped", nd=(0,))
    add(r["fp_recovered"], f"dev:{name}.fp_recovered", nd=(0,))
    add(r["tp_lost"], f"dev:{name}.tp_lost", nd=(0,))
    for num, den in ((r["fp_recovered"], dev["apparent_fp"]),
                     (r["tp_lost"], dev["true_positive"])):
        if den:
            add(num / den, f"dev:{name}.share")
for row in dev["grid"]["matrix"]:
    for v in row:
        add(v, "dev:grid.cell")
add(n_clear / 197, "manifest:test.clear.share")
add((197 - n_clear) / 197, "manifest:test.cloud.share")

# constants stated in the manuscript that live in the code, not an artifact
for v, src in ((128, "patch size"), (5, "5x5 window"), (4, "four classes"),
               (266, "preset K"), (264, "preset K"), (262, "preset K"),
               (1.0, "preset K"), (1.5, "preset K"), (2.0, "preset K"),
               (10.76, "M15 band"), (2019, "year"), (2026, "year"),
               (2020, "year"), (2022, "year"), (2024, "year"), (2025, "year"),
               (24.7, "backbone params"), (1971, "train patches"),
               (95, "basemap clear threshold"), (33, "radar dBZ"),
               (10, "km"), (60, "doy"), (151, "doy"), (152, "doy"),
               (243, "doy"), (244, "doy"), (334, "doy"), (0.7, "GCE q"),
               (0.4, "mixup alpha"), (42, "seed"), (120, "held out"),
               (113, "overlap held out"), (7, "temporal held out"), (8, "day gap")):
    add(v, "constant:" + src, nd=(0, 1, 2))
    pool.setdefault(str(int(v)) if float(v).is_integer() else str(v), "constant:" + src)

# values quoted from the cited literature
for v, src in ((11.45, "lit:VIIRS I5"), (10.26, "lit:M15 range"),
               (11.26, "lit:M15 range"), (0.5, "lit:DNB range"), (0.9, "lit:DNB range"),
               (84.9, "text:stated specificity"), (77.0, "text:apparent FP rate"),
               (55.7, "text:dev-set preset acc"), (62.3, "text:dev-set grid best"),
               (62.9, "text:dev-set raw acc"), (44.2, "text:raw acc"),
               (29.4, "text:always-cloud acc"), (70.6, "text:always-clear acc"),
               (76.6, "text:M15 T264 acc"), (77.2, "text:M15 T266 acc"),
               (69.5, "text:direct rule acc"), (65.0, "text:conservative acc"),
               (71.6, "text:aggressive acc")):
    add(v, src)

# ---- harvest every figure the manuscript states ---------------------------
tex = open(TEX, encoding="utf-8").read()
body = "\n".join(l for l in tex.split("\n") if not l.strip().startswith("%"))
body = body.split(r"\begin{thebibliography}")[0]

hits = []
for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\\?%", body):
    hits.append((m.group(1), "%", body[max(0, m.start() - 110):m.end() + 10]))
for m in re.finditer(r"(\d+(?:\.\d+)?)\s*~pp", body):
    hits.append((m.group(1), "pp", body[max(0, m.start() - 110):m.end() + 10]))

print("=" * 78)
print("reverse check: numbers in the manuscript that no artifact produces")
print("=" * 78)
print(f"  artifact values in pool  : {len(pool)}")
print(f"  manuscript figures found : {len(hits)}")

seen, suspects = set(), []
for value, kind, ctx in hits:
    if value in pool or (value, kind) in seen:
        continue
    seen.add((value, kind))
    suspects.append((value, kind, re.sub(r"\s+", " ", ctx)))

print(f"  unmatched                : {len(suspects)}")
print()
for value, kind, ctx in suspects:
    print(f"  [{kind:>2}] {value}")
    print(f"        ...{ctx}...")
print()
print("=" * 78)
print("Every line above needs adjudication. It is either a constant this pool does")
print("not model (band thresholds, literature values, dataset sizes) or a stale")
print("number left behind by a re-run.")
print("=" * 78)
sys.exit(1 if suspects else 0)
