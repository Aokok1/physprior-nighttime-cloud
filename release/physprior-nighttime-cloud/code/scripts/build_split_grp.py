"""Build a group-disjoint, temporally-clean train/val split.

Fixes the two split defects found in the audit:
  1. train_phase4 and val_phase4 shared 49 overpass groups (98 patches), so the
     validation metric used for early stopping was not independent.
  2. Eight of the 29 Changsha test patches lay within seven days of a Changsha
     training/validation patch, the smallest gap being one day.

Construction
  - Test stays exactly as it is: all 197 radar-collocated patches.
  - The train/val pool (Train_phase4 + Val_phase4 = 2472) loses every patch that
    shares a station and lies within EXCLUDE_DAYS days of any test patch.
  - The remainder is split by overpass_group, so no group spans train and val.
  - Val fraction 0.2, seed 42, groups shuffled.

Output
  - E:/Data/Unet_Dataset/Train_grp/ and Val_grp/  (hardlinks; no extra disk use)
  - output/manifest/split_grp_manifest.json

Run:  python scripts/build_split_grp.py
"""
import collections
import csv
import datetime
import json
import os
import random
import shutil
import sys

ROOT = r"E:/Claude code/project/noise-label-cloud"
DATA = r"E:/Data/Unet_Dataset"
EXCLUDE_DAYS = 7
VAL_FRAC = 0.2
SEED = 42

rows = list(csv.DictReader(open(os.path.join(ROOT, "output", "manifest", "sample_manifest.csv"),
                                encoding="utf-8", newline="")))


def date_of(r):
    return datetime.date(int(r["year"]), 1, 1) + datetime.timedelta(days=int(r["doy"]) - 1)


test = [r for r in rows if r["split"] == "test"]
pool = [r for r in rows if r["split"] in ("train_phase4", "val_phase4")]
print(f"test = {len(test)}, train/val pool = {len(pool)}")

# ---- 1. temporal exclusion ---------------------------------------------------
test_by_station = collections.defaultdict(list)
for r in test:
    test_by_station[r["station"]].append(date_of(r))

excluded = []
kept = []
for r in pool:
    d = date_of(r)
    near = any(abs((d - td).days) <= EXCLUDE_DAYS for td in test_by_station.get(r["station"], []))
    (excluded if near else kept).append(r)

print(f"\ntemporal exclusion (same station, |Δdays| <= {EXCLUDE_DAYS} from a test patch):")
print(f"  excluded {len(excluded)}  kept {len(kept)}")
by_st = collections.Counter(r["station"] for r in excluded)
print(f"  excluded by station: {dict(by_st)}")
if excluded:
    gaps = sorted(min(abs((date_of(r) - td).days) for td in test_by_station[r["station"]]) for r in excluded)
    print(f"  excluded gap days: min {gaps[0]} median {gaps[len(gaps)//2]} max {gaps[-1]}")

# ---- 2. group-disjoint split -------------------------------------------------
groups = collections.defaultdict(list)
for r in kept:
    groups[r["overpass_group"]].append(r)
gkeys = sorted(groups)
rng = random.Random(SEED)
rng.shuffle(gkeys)
n_val_groups = int(round(len(gkeys) * VAL_FRAC))
val_groups = set(gkeys[:n_val_groups])
train_groups = set(gkeys[n_val_groups:])

train = [r for g in train_groups for r in groups[g]]
val = [r for g in val_groups for r in groups[g]]

# ---- 3. verify ---------------------------------------------------------------
gt = {r["overpass_group"] for r in train}
gv = {r["overpass_group"] for r in val}
gte = {r["overpass_group"] for r in test}
print(f"\ngroup-disjoint split (seed {SEED}, val_frac {VAL_FRAC}):")
print(f"  groups: train {len(train_groups)}  val {len(val_groups)}  test {len(gte)}")
print(f"  patches: train {len(train)}  val {len(val)}  test {len(test)}")
print(f"  train ∩ val groups = {len(gt & gv)}   (must be 0)")
print(f"  (train ∪ val) ∩ test groups = {len((gt | gv) & gte)}   (must be 0)")

min_gap = 10 ** 6
for r in train + val:
    for td in test_by_station.get(r["station"], []):
        min_gap = min(min_gap, abs((date_of(r) - td).days))
print(f"  minimum |Δdays| between any remaining train/val patch and a test patch = {min_gap}")

by_st_tr = collections.Counter(r["station"] for r in train)
by_st_va = collections.Counter(r["station"] for r in val)
print(f"  train stations {dict(by_st_tr)}   val stations {dict(by_st_va)}")
print(f"  train radar-labelled = {sum(1 for r in train if r['has_radar']=='True')}"
      f"   val radar-labelled = {sum(1 for r in val if r['has_radar']=='True')}")

assert len(gt & gv) == 0 and len((gt | gv) & gte) == 0

# ---- 4. materialise (hardlink) ----------------------------------------------
SRC_DIRS = ["Train_phase4", "Val_phase4"]
src_of = {}
for r in rows:
    for sub in SRC_DIRS:
        p = os.path.join(DATA, sub, r["fname"])
        if os.path.exists(p):
            src_of[r["fname"]] = p
            break

for name, members in (("Train_grp", train), ("Val_grp", val)):
    out = os.path.join(DATA, name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    linked = 0
    for r in members:
        src = src_of.get(r["fname"])
        if not src:
            continue
        dst = os.path.join(out, r["fname"])
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        linked += 1
    print(f"  materialised {name}: {linked} files")

manifest = {
    "seed": SEED, "val_frac": VAL_FRAC, "exclude_days": EXCLUDE_DAYS,
    "train_count": len(train), "val_count": len(val), "test_count": len(test),
    "excluded_temporal": len(excluded),
    "train_files": [r["fname"] for r in train],
    "val_files": [r["fname"] for r in val],
    "excluded_files": [r["fname"] for r in excluded],
}
os.makedirs(os.path.join(ROOT, "output", "manifest"), exist_ok=True)
p = os.path.join(ROOT, "output", "manifest", "split_grp_manifest.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=1)
print(f"\nwrote {p}")

# ---- 5. tagged sample manifest ---------------------------------------------
# Scripts that reason about splits by reading sample_manifest.csv need a version
# whose `split` column matches the new assignment. Drop the temporally excluded
# patches to the unassigned bucket so downstream counts stay consistent.
new_split = {}
for r in train:
    new_split[r["fname"]] = "train_grp"
for r in val:
    new_split[r["fname"]] = "val_grp"
for r in test:
    new_split[r["fname"]] = "test"
for r in excluded:
    new_split[r["fname"]] = ""

cols = list(rows[0].keys())
out_csv = os.path.join(ROOT, "output", "manifest", "sample_manifest_grp.csv")
with open(out_csv, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        r = dict(r)
        if r["fname"] in new_split:
            r["split"] = new_split[r["fname"]]
        w.writerow(r)
print(f"wrote {out_csv}")
print("split counts in tagged manifest:",
      dict(collections.Counter(new_split.get(r['fname'], r['split']) or '(unassigned)' for r in rows)))

