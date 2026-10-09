"""Make the on-the-fly PhysPrior label loader follow the active split.

`apply_physprior_to_batch` re-reads the raw .npz for each sample to obtain the
uncorrected CLDMSK labels and the raw M15 patch. Four scripts hard-coded
"Train_phase4" there, so a run on an alternative split aborted on the first batch
even though the dataloader itself had already been redirected.

The replacement resolves the directory with the same NL_SPLIT_TAG convention used
by data/dataset.py, so no call site needs to change.
"""
import io
import re
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud/"

HELPER = '''

def _split_dir(root, mode="Train"):
    """Resolve the active split directory (mirrors data/dataset.py)."""
    tag = os.environ.get("NL_SPLIT_TAG", "phase4")
    return os.path.join(root, f"{mode}_{tag}")
'''

TARGETS = [
    "scripts/train_2x2_ablation.py",
    "scripts/train_physprior_5seeds.py",
    "scripts/train_coteaching_5seeds.py",
]

OLD = 'os.path.join(MTUNET_DATASET, "Train_phase4", fn)'
NEW = 'os.path.join(_split_dir(MTUNET_DATASET, "Train"), fn)'

for rel in TARGETS:
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    if "_split_dir" in s:
        print("already patched  " + rel)
        continue
    if OLD not in s:
        print("pattern absent   " + rel)
        continue
    s = s.replace(OLD, NEW)
    # insert the helper after the last import line near the top
    lines = s.split("\n")
    idx = 0
    for i, ln in enumerate(lines[:70]):
        if ln.startswith("import ") or ln.startswith("from "):
            idx = i
    lines.insert(idx + 1, HELPER)
    s = "\n".join(lines)
    open(p, "w", encoding="utf-8").write(s)
    print("patched          " + rel)

# the long co-teaching script passes the directory name explicitly
p = ROOT + "scripts/train_physprior_coteaching_long.py"
s = open(p, encoding="utf-8").read()
OLD2 = 'apply_physprior_to_batch(MTUNET_DATASET, "Train_phase4", batch)'
NEW2 = 'apply_physprior_to_batch(MTUNET_DATASET, _split_dir_name("Train"), batch)'
if "_split_dir_name" in s:
    print("already patched  scripts/train_physprior_coteaching_long.py")
elif OLD2 in s:
    s = s.replace(OLD2, NEW2)
    s = s.replace("def apply_physprior_to_batch(",
                  'def _split_dir_name(mode="Train"):\n'
                  '    """Active split directory name (mirrors data/dataset.py)."""\n'
                  '    return f"{mode}_{os.environ.get(\'NL_SPLIT_TAG\', \'phase4\')}"\n\n\n'
                  'def apply_physprior_to_batch(', 1)
    open(p, "w", encoding="utf-8").write(s)
    print("patched          scripts/train_physprior_coteaching_long.py")
else:
    print("pattern absent   scripts/train_physprior_coteaching_long.py")
