"""Aggregate the per-seed training traces into the table used by Supplementary S2.

Reads output/physprior_5seed_history_seed<N>{SUF}.json, which
scripts/train_physprior_5seeds.py writes once per seed, and emits the markdown
table plus output/supp_S2_table{SUF}.json.

Run:  NL_CKPT_TAG=grp python scripts/aggregate_5seed_history.py
"""
import glob
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HIST_DIR = r"E:/Claude code/project/noise-label-cloud/output"
TAG = os.environ.get("NL_CKPT_TAG", "grp")
SUF = f"_{TAG}" if TAG else ""

files = sorted(glob.glob(os.path.join(HIST_DIR, f"physprior_5seed_history_seed*{SUF}.json")))
if not files:
    raise SystemExit(f"no history files matching physprior_5seed_history_seed*{SUF}.json")

rows = []
for fp in files:
    with open(fp, encoding="utf-8") as f:
        d = json.load(f)
    h = d["history"]
    val_accs = h["val_acc"]
    best_idx = val_accs.index(max(val_accs))
    rows.append({
        "seed": d["seed"],
        "train_loss_at_best": round(h["train_loss"][best_idx], 4),
        "val_acc_at_best": round(val_accs[best_idx] * 100, 2),
        "best_epoch": best_idx + 1,
        "epochs_run": len(h["train_loss"]),
        "train_loss_at_last": round(h["train_loss"][-1], 4),
        "val_acc_at_last": round(val_accs[-1] * 100, 2),
        "test_acc": round(d.get("test_acc", 0.0) * 100, 2),
    })

rows.sort(key=lambda r: r["seed"])

print("| Seed | best epoch | epochs run | train loss @ best | val acc @ best | "
      "train loss @ last | val acc @ last | test acc |")
print("|------|-----------:|-----------:|------------------:|---------------:|"
      "------------------:|---------------:|---------:|")
for r in rows:
    print(f"| {r['seed']} | {r['best_epoch']} | {r['epochs_run']} | "
          f"{r['train_loss_at_best']:.3f} | {r['val_acc_at_best']:.2f}% | "
          f"{r['train_loss_at_last']:.3f} | {r['val_acc_at_last']:.2f}% | "
          f"{r['test_acc']:.2f}% |")

accs = [r["test_acc"] for r in rows]
print(f"\ntest accuracy: {sum(accs) / len(accs):.2f}% mean over {len(accs)} seeds, "
      f"range {min(accs):.2f}-{max(accs):.2f}")

out = os.path.join(HIST_DIR, f"supp_S2_table{SUF}.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2)
print(f"saved {out}")
