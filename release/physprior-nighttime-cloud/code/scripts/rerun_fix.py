"""Full re-run of every deep-learning row under the corrected data path.

Mirrors scripts/rerun_grp.ps1 but with NL_CKPT_TAG=fix, so no published artifact
is overwritten. Stages are skipped when their artifact already exists, so the
chain can be resumed after an interruption.

Why this exists: two defects in the training-side data path were found and fixed
in data/dataset.py and methods/physical_prior.py —

  1. the physical prior was evaluated on X_m15 read from disk (untransformed)
     against an already flipped/rotated label mask;
  2. CloudDataset._augment advanced ry/rx with the clockwise rot90 index map
     while np.rot90 rotates counter-clockwise, so for rotation k = 1 and k = 3
     (half of all training views) radar_loc addressed the wrong pixel and the
     centre-pixel supervision (loss weight 5.0) plus the height target were
     mis-anchored.

Neither can touch the evaluation protocol (Test and Val load with
augment=False), so every number produced here is scored on the same 197
radar-reference pixels with the same 3x3-OR rule as the published table.

Clobber guard: scripts/redo_loss_correction.py names its artifacts after
NL_SPLIT_TAG (not NL_CKPT_TAG), so it would overwrite the published
*_grp.json. The published copies are restored from the backup directory after
that stage and the fresh ones are renamed to *_fix.json.

Usage:  python scripts/rerun_fix.py            # run everything pending
        python scripts/rerun_fix.py --status   # show what is done
"""
import json, os, shutil, subprocess, sys, time

ROOT = r"E:/Claude code/project/noise-label-cloud"
BACKUP = rf"{ROOT}/output/_backup_pre_alignment_fix_20261010"
LOGDIR = rf"{ROOT}/output/logs_fix"
os.makedirs(LOGDIR, exist_ok=True)

ENV = dict(os.environ)
ENV.update({
    "NL_SPLIT_TAG": "grp", "NL_CKPT_TAG": "fix", "NL_SEED": "42",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NL_BASEMAP_DIR": r"E:\Data\Unet_Dataset\base_map_trainval",
    "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
})

# stage id -> (argv, artifact that proves completion, post-step)
STAGES = [
    ("01_paired_5seed_label_sources",
     ["python", "-X", "utf8", "scripts/train_m15_label_control.py", "--paired",
      "--seeds", "0,1,2,3,4", "--modes", "raw,physprior,m15thresh",
      "--channels", "2,3"],
     rf"{ROOT}/output/label_source_paired_5seed_fix.json", None),
    ("02_standard", ["python", "-X", "utf8", "train.py", "--method", "baseline"],
     rf"{ROOT}/output/checkpoints_fix/baseline_best.pth", None),
    ("03_coteaching", ["python", "-X", "utf8", "train.py", "--method", "coteaching"],
     rf"{ROOT}/output/checkpoints_fix/coteaching_a_best.pth", None),
    ("04_gce", ["python", "-X", "utf8", "train.py", "--method", "gce", "--gce-q", "0.7"],
     rf"{ROOT}/output/checkpoints_fix/gce_q0.7_best.pth", None),
    ("05_mixup", ["python", "-X", "utf8", "train.py", "--method", "mixup"],
     rf"{ROOT}/output/checkpoints_fix/mixup_best.pth", None),
    ("06_physprior_dl", ["python", "-X", "utf8", "train.py", "--method", "phys-mod"],
     rf"{ROOT}/output/checkpoints_fix/physprior_moderate_best.pth",
     [("move", rf"{ROOT}/output/physprior_moderate_stats.json",
       rf"{ROOT}/output/physprior_moderate_stats_fix.json")]),
    ("07_physprior_5seed",
     ["python", "-X", "utf8", "scripts/train_physprior_5seeds.py"],
     rf"{ROOT}/output/physprior_5seed_results_fix.json", None),
    ("08_coteaching_5seed",
     ["python", "-X", "utf8", "scripts/train_coteaching_5seeds.py"],
     rf"{ROOT}/output/coteaching_5seed_results_fix.json", None),
    ("09_coteaching_ensemble",
     ["python", "-X", "utf8", "scripts/train_physprior_coteaching_long.py"],
     rf"{ROOT}/output/checkpoints_fix/physprior_coteaching_long_a_best.pth", None),
    ("10_loss_correction",
     ["python", "-X", "utf8", "scripts/redo_loss_correction.py"],
     rf"{ROOT}/output/checkpoints_fix/loss_correction_best.pth",
     [("clobber_fix", None, None)]),
    ("11_freeze", ["python", "-X", "utf8", "scripts/freeze_predictions_grp.py"],
     rf"{ROOT}/output/frozen_predictions/all_frozen_predictions_fix.json", None),
    ("12_tables", ["python", "-X", "utf8", "scripts/rebuild_paper_tables.py"],
     rf"{ROOT}/output/paper_tables_fix.json", None),
    ("13_station_season",
     ["python", "-X", "utf8", "scripts/derive_station_season_tables.py"],
     rf"{ROOT}/output/derived_tables_fix.json", None),
]

LOSS_ARTIFACTS = ["loss_correction_redo_grp.json", "noise_matrix_4x4_grp.json",
                   "noise_matrix_from_train_val_grp.json"]


def status():
    for name, argv, art, post in STAGES:
        done = os.path.exists(art)
        if done and art.endswith("5seed_fix.json"):
            try:
                n = len(json.load(open(art)))
                done, extra = n == 30, f" ({n}/30 runs)"
            except Exception:
                extra = ""
        else:
            extra = ""
        print(f"  {'DONE ' if done else 'TODO '} {name}{extra}")


def run_stage(name, argv, artifact, post):
    log = os.path.join(LOGDIR, f"{name}.log")
    print(f"\n=== {name}: {' '.join(argv)}   (log: {log})", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n----- started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.flush()
        p = subprocess.Popen(argv, cwd=ROOT, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
        rc = p.wait()
        fh.write(f"----- exit {rc} after {time.time()-t0:.0f}s\n")
    print(f"    exit={rc}  {time.time()-t0:.0f}s", flush=True)
    if rc != 0:
        return False
    for action, src, dst in (post or []):
        if action == "move" and os.path.exists(src):
            shutil.move(src, dst)
        elif action == "clobber_fix":
            # redo_loss_correction.py names files after NL_SPLIT_TAG, so it just
            # wrote over the published *_grp.json. Stash the new ones as *_fix,
            # then put the published copies back.
            for f in LOSS_ARTIFACTS:
                now = os.path.join(ROOT, "output", f)
                fixed = os.path.join(ROOT, "output", f.replace("_grp", "_fix"))
                if os.path.exists(now):
                    shutil.move(now, fixed)
                bk = os.path.join(BACKUP, f)
                if os.path.exists(bk):
                    shutil.copy2(bk, now)
                    print(f"    restored published {f} from backup")
    return True


def _py(argv):
    """Resolve 'python' to this interpreter: Git Bash's PATH lookup inside a
    detached subprocess picks a Python without torch."""
    return [sys.executable if a == "python" else a for a in argv]


def stage_done(name, artifact):
    """A resumable stage is only done when its artifact is complete."""
    if not os.path.exists(artifact):
        return False
    if name == "01_paired_5seed_label_sources":
        try:
            return len(json.load(open(artifact))) >= 30
        except Exception:
            return False
    return True


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
        sys.exit(0)
    print(f"rerun_fix started {time.strftime('%Y-%m-%d %H:%M:%S')}  tag=fix "
          f"interpreter={sys.executable}", flush=True)
    failed = []
    for name, argv, artifact, post in STAGES:
        if stage_done(name, artifact):
            print(f"\n=== {name}: complete, skipping", flush=True)
            continue
        if not run_stage(name, _py(argv), artifact, post):
            print(f"    !! {name} FAILED — stopping the chain", flush=True)
            failed.append(name)
            break
    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETE" if not failed else f"CHAIN STOPPED AT {failed}")
    print("=" * 60, flush=True)
