"""Compare the pre-determinism artifacts against the deterministic re-run.

The 2026-09-24 re-run produced every number that was in the manuscript, but
stages 01-06 and 10 ran with no RNG seeding at all and stages 07-09 ran with
cuDNN autotuning enabled. `scripts/backup_predet.ps1` archived those artifacts
under `output/_predet_backup_20260924/`. This script aligns the archived tables
with the freshly rebuilt ones and prints the per-row movement, so the run report
can state exactly which reported numbers changed once the pipeline was pinned.

Usage:
    python -X utf8 scripts/compare_predet.py
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = r"E:/Claude code/project/noise-label-cloud/output"
OLD = os.path.join(ROOT, "_predet_backup_20260924")
NEW = ROOT

# table key -> (source filename, dict-of-rows key or None, metric key)
PAIRS = [
    ("paper_tables", "paper_tables_grp.json", "table_methods", "accuracy"),
    ("ablation_2x2", "2x2_ablation_grp.json", None, "accuracy"),
    ("five_seed_physprior", "physprior_5seed_summary_grp.json", None, "accuracy_mean"),
    ("five_seed_coteaching", "coteaching_5seed_summary_grp.json", None, "accuracy_mean"),
    ("derived_tables", "derived_tables_grp.json", None, None),
]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten(d, prefix=""):
    """Yield (dotted-key, float) for every numeric leaf."""
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from flatten(v, f"{prefix}{i}.")
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        yield prefix.rstrip("."), float(d)


print("=" * 78)
print("pre-determinism run  vs  deterministic re-run")
print("=" * 78)

for label, fname, rows_key, _metric in PAIRS:
    old = load(os.path.join(OLD, fname))
    new = load(os.path.join(NEW, fname))
    print(f"\n### {label}   ({fname})")
    if old is None:
        print("   archived copy missing - nothing to compare")
        continue
    if new is None:
        print("   new artifact missing - re-run has not reached this stage yet")
        continue

    o = {k: v for k, v in flatten(old)}
    n = {k: v for k, v in flatten(new)}
    keys = [k for k in o if k in n]
    moved = [(k, o[k], n[k]) for k in keys if abs(o[k] - n[k]) > 1e-9]

    print(f"   numeric fields compared : {len(keys)}")
    print(f"   unchanged               : {len(keys) - len(moved)}")
    print(f"   moved                   : {len(moved)}")
    if moved:
        moved.sort(key=lambda t: -abs(t[2] - t[1]))
        shown = [m for m in moved if abs(m[2] - m[1]) > 1e-6]
        print(f"   {'field':<52} {'old':>10} {'new':>10} {'delta':>10}")
        for k, a, b in shown[:40]:
            print(f"   {k[:52]:<52} {a:>10.6f} {b:>10.6f} {b - a:>+10.6f}")
        if len(shown) > 40:
            print(f"   ... and {len(shown) - 40} more with |delta| > 1e-6")

print("\n" + "=" * 78)
print("note: 'moved' counts only fields present in both artifacts; a stage that")
print("      crashed pre-determinism has no archived counterpart and is skipped.")
