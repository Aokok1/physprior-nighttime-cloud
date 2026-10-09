"""Second-pass patcher: tag the remaining hard-coded output paths."""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud/"

FIXES = [
    ("scripts/train_physprior_5seeds.py",
     'sum_path = r"E:/Claude code/project/noise-label-cloud/output/physprior_5seed_summary.json"',
     'sum_path = rf"E:/Claude code/project/noise-label-cloud/output/physprior_5seed_summary{_out_tag()}.json"'),
    ("scripts/train_coteaching_5seeds.py",
     'with open(r"E:/Claude code/project/noise-label-cloud/output/coteaching_5seed_summary.json", "w") as f:',
     'with open(rf"E:/Claude code/project/noise-label-cloud/output/coteaching_5seed_summary{_out_tag()}.json", "w") as f:'),
    ("scripts/redo_loss_correction.py",
     'out_path = r"E:/Claude code/project/noise-label-cloud/output/noise_matrix_from_train_val.json"',
     'out_path = rf"E:/Claude code/project/noise-label-cloud/output/noise_matrix_from_train_val{_suf}.json"'),
]

for rel, old, new in FIXES:
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    if old in s:
        open(p, "w", encoding="utf-8").write(s.replace(old, new))
        print("patched  " + rel)
    else:
        print("absent   " + rel)

# move the _suf definition above its first use
p = ROOT + "scripts/redo_loss_correction.py"
s = open(p, encoding="utf-8").read()
line = '_suf = f"_{_tag}" if _tag != "phase4" else ""\n'
if line in s:
    s = s.replace(line, "")
s = s.replace("_tr, _va = f'train_{_tag}', f'val_{_tag}'",
              "_tr, _va = f'train_{_tag}', f'val_{_tag}'\n_suf = f'_{_tag}' if _tag != 'phase4' else ''")
open(p, "w", encoding="utf-8").write(s)
print("reordered _suf in redo_loss_correction.py")
