"""Finish the determinism sweep: seed the entry points that had no seeding at all.

`patch_determinism.py` covered the two multi-seed scripts and the 2x2 ablation.
The audit then found that the remaining entry points never seeded anything:

  * train.py                              (stages 02-06: baseline, co-teaching,
                                           GCE, mixup, PhysPrior+DL)
  * scripts/redo_loss_correction.py       (stage 01)
  * scripts/train_physprior_coteaching_long.py (stage 10)

and that the train DataLoader shuffled off the global torch RNG, which makes the
shuffle order depend on how much randomness the model consumed first.

This patch routes all three through `seed_everything` (placed immediately after
the sys.path insert so CUBLAS_WORKSPACE_CONFIG is set before the first CUDA
handle) and gives the train loader its own generator.
"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud/"

SEED_CALL = ('from determinism import seed_everything\n'
             'seed_everything(int(os.environ.get("NL_SEED", "42")))  '
             '# fixed seed: the pipeline must be bit-reproducible\n')

report = []

# ── 1. module-level seeding for the three entry points ───────────────────────
ENTRY_POINTS = [
    ("train.py",
     'sys.path.insert(0, PROJECT_ROOT)'),
    ("scripts/redo_loss_correction.py",
     'sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")'),
    ("scripts/train_physprior_coteaching_long.py",
     'sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")'),
]

for rel, anchor in ENTRY_POINTS:
    p = ROOT + rel
    s = open(p, encoding="utf-8").read()
    if "seed_everything" in s:
        report.append((rel, "already seeded"))
        continue
    if anchor not in s:
        report.append((rel, "ANCHOR MISSING - not patched"))
        continue
    if "\nimport os" not in s and not s.startswith("import os") and ", os" not in s:
        report.append((rel, "os not imported - not patched"))
        continue
    s = s.replace(anchor, anchor + "\n" + SEED_CALL, 1)
    open(p, "w", encoding="utf-8").write(s)
    report.append((rel, "seeded"))

# ── 2. give the train DataLoader its own generator ───────────────────────────
p = ROOT + "data/dataset.py"
s = open(p, encoding="utf-8").read()
if "loader_generator" in s:
    report.append(("data/dataset.py", "already has a loader generator"))
else:
    old = ("    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,\n"
           "                              num_workers=num_workers, pin_memory=True)")
    new = ("    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,\n"
           "                              num_workers=num_workers, pin_memory=True,\n"
           "                              generator=loader_generator(\n"
           "                                  int(os.environ.get(\"NL_SEED\", \"42\"))))")
    if old not in s:
        report.append(("data/dataset.py", "DataLoader line not found - not patched"))
    else:
        s = s.replace(old, new, 1)
        if "from determinism import loader_generator" not in s:
            s = s.replace("from config import data_cfg, MTUNET_NORM_STATS",
                          "from config import data_cfg, MTUNET_NORM_STATS\n"
                          "from determinism import loader_generator", 1)
        open(p, "w", encoding="utf-8").write(s)
        report.append(("data/dataset.py", "train loader given a seeded generator"))

for rel, what in report:
    print(f"{what:<45} {rel}")
