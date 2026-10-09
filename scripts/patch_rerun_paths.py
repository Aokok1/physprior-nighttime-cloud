"""One-off patcher: make the re-run scripts tag-aware and fix two defects.

Applies, idempotently:
  * output JSON paths gain a tag suffix when NL_CKPT_TAG is set, so a re-run on
    an alternative split does not overwrite the published artifacts
  * train_coteaching_5seeds.py no longer writes the TEST accuracy into the field
    named best_val_acc
  * redo_loss_correction.py reads the active split instead of hard-coding
    train_phase4/val_phase4

Run:  python scripts/patch_rerun_paths.py
"""
import io
import os
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud"

TAG_HELPER = '''
def _out_tag():
    """Suffix for derived artifacts when running on an alternative split."""
    t = os.environ.get("NL_CKPT_TAG", "")
    return f"_{t}" if t else ""
'''

PATCHES = [
    # (file, old, new)
    ("scripts/train_2x2_ablation.py",
     'out_path = r"E:/Claude code/project/noise-label-cloud/output/2x2_ablation.json"',
     'out_path = rf"E:/Claude code/project/noise-label-cloud/output/2x2_ablation{_out_tag()}.json"'),
    ("scripts/train_physprior_5seeds.py",
     'out_path = r"E:/Claude code/project/noise-label-cloud/output/physprior_5seed_results.json"',
     'out_path = rf"E:/Claude code/project/noise-label-cloud/output/physprior_5seed_results{_out_tag()}.json"'),
    ("scripts/train_coteaching_5seeds.py",
     'out_path = r"E:/Claude code/project/noise-label-cloud/output/coteaching_5seed_results.json"',
     'out_path = rf"E:/Claude code/project/noise-label-cloud/output/coteaching_5seed_results{_out_tag()}.json"'),
    ("scripts/train_physprior_coteaching_long.py",
     'out_path = f"E:/Claude code/project/noise-label-cloud/output/physprior_coteaching_long_{label}.json"',
     'out_path = f"E:/Claude code/project/noise-label-cloud/output/physprior_coteaching_long_{label}{_out_tag()}.json"'),
]

reports = []
for rel, old, new in PATCHES:
    p = os.path.join(ROOT, rel)
    s = open(p, encoding="utf-8").read()
    if old not in s:
        reports.append((rel, "already patched or pattern absent"))
        continue
    s = s.replace(old, new)
    if "_out_tag" in s and "def _out_tag" not in s:
        # insert the helper after the last top-level import block
        lines = s.split("\n")
        idx = 0
        for i, ln in enumerate(lines[:60]):
            if ln.startswith("import ") or ln.startswith("from "):
                idx = i
        lines.insert(idx + 1, TAG_HELPER)
        s = "\n".join(lines)
    open(p, "w", encoding="utf-8").write(s)
    reports.append((rel, "patched"))

# --- coteaching 5-seed: best_val_acc must be the validation value ------------
p = os.path.join(ROOT, "scripts/train_coteaching_5seeds.py")
s = open(p, encoding="utf-8").read()
old = ('json.dump({"seed": seed, "history": r.get("history", {}), '
       '"best_val_acc": r.get("test", {}).get("acc", 0.0), '
       '"best_epoch": r.get("best_epoch", 0)}, f, indent=2, default=float)')
new = ('_va = (r.get("history", {}) or {}).get("val_acc", [])\n'
       '            json.dump({"seed": seed, "history": r.get("history", {}),\n'
       '                       "best_val_acc": max(_va) if _va else 0.0,\n'
       '                       "test_acc": r.get("test", {}).get("acc", 0.0),\n'
       '                       "best_epoch": r.get("best_epoch", 0)}, f, indent=2, default=float)')
if old in s:
    s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    reports.append(("scripts/train_coteaching_5seeds.py", "best_val_acc mislabel fixed"))
else:
    reports.append(("scripts/train_coteaching_5seeds.py", "best_val_acc fix: pattern absent"))

# --- redo_loss_correction: read the active split -----------------------------
p = os.path.join(ROOT, "scripts/redo_loss_correction.py")
s = open(p, encoding="utf-8").read()
old = "radar_non_test = df[(df.has_radar == True) & (df.split.isin(['train_phase4', 'val_phase4']))]"
new = ("_tag = os.environ.get('NL_SPLIT_TAG', 'phase4')\n"
       "_tr, _va = f'train_{_tag}', f'val_{_tag}'\n"
       "radar_non_test = df[(df.has_radar == True) & (df.split.isin([_tr, _va]))]")
if old in s:
    s = s.replace(old, new)
    old2 = "    for sub in ['Train_phase4', 'Val_phase4']:"
    new2 = ("    for sub in [f'Train_{_tag}', f'Val_{_tag}', 'Train_phase4', 'Val_phase4']:")
    s = s.replace(old2, new2)
    old3 = 'with open(r"E:/Claude code/project/noise-label-cloud/output/loss_correction_redo.json", \'w\') as f:'
    new3 = ('_suf = f"_{_tag}" if _tag != "phase4" else ""\n'
            'with open(rf"E:/Claude code/project/noise-label-cloud/output/loss_correction_redo{_suf}.json", \'w\') as f:')
    s = s.replace(old3, new3)
    open(p, "w", encoding="utf-8").write(s)
    reports.append(("scripts/redo_loss_correction.py", "split-aware"))
else:
    reports.append(("scripts/redo_loss_correction.py", "already patched or pattern absent"))

for r in reports:
    print(f"  {r[0]:46s} {r[1]}")
