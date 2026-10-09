"""Compact progress report for the group-disjoint re-run."""
import glob
import io
import json
import os
import re
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = r"E:/Claude code/project/noise-label-cloud"
LOGD = os.path.join(ROOT, "output", "logs_grp")
CKPT = os.path.join(ROOT, "output", "checkpoints_grp")

STAGES = ["01_loss_correction", "02_standard", "03_coteaching", "04_gce", "05_mixup",
          "06_physprior_dl", "07_ablation_2x2", "08_physprior_5seed",
          "09_coteaching_5seed", "10_cot_ensemble"]

print("=" * 70)
print("RE-RUN PROGRESS  (group-disjoint split)")
print("=" * 70)
for s in STAGES:
    p = os.path.join(LOGD, s + ".log")
    if not os.path.exists(p):
        print(f"  {s:20s}  --  not started")
        continue
    txt = open(p, encoding="utf-8", errors="replace").read()
    epochs = re.findall(r"\[(\d{3})\]", txt)
    age = (time.time() - os.path.getmtime(p)) / 60.0
    last = [ln for ln in txt.strip().split("\n") if ln.strip()][-1][:88]
    state = "RUNNING" if age < 3 else "done/idle"
    print(f"  {s:20s}  {state:10s} epochs_seen={len(epochs):3d}  log_age={age:5.1f} min")
    print(f"      last: {last}")

print("\n--- checkpoints ---")
for f in sorted(glob.glob(os.path.join(CKPT, "*.pth"))):
    print(f"  {os.path.basename(f):48s} {os.path.getsize(f)/1e6:6.1f} MB")

print("\n--- tagged result json ---")
for f in sorted(glob.glob(os.path.join(ROOT, "output", "*_grp.json"))):
    print(f"  {os.path.basename(f):48s} {os.path.getsize(f)/1e3:8.1f} kB")

p = os.path.join(LOGD, "rerun_stage_summary.json")
if os.path.exists(p):
    print("\n--- stage summary ---")
    for r in json.load(open(p, encoding="utf-8")):
        print(f"  {r['stage']:20s} exit={r['exit']}  {r['minutes']} min")
