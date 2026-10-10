"""M15-threshold-as-label control experiment for the GRSL letter.

Question this answers: PhysPrior's Table I shows a pure M15 threshold (264 K)
reaching 76.6% on the 197-pixel radar reference, above PhysPrior's own 69.5%.
A fair reader will therefore ask why the corrected CLDMSK label field should be
preferred as a TRAINING signal over the threshold mask itself. No existing row
answers that, because the threshold has never been used as a label source.

Cells run here (same architecture, same group-disjoint split, same seed 42,
same fixed 20-epoch budget with no early stopping, same 3x3-OR test protocol as
the published 2x2 channel-label ablation):

  m15thresh  : labels replaced wholesale by  M15 >= 264 K -> clear, else cloudy
  physprior  : labels corrected by the deployed one-directional rule
  raw        : unmodified CLDMSK  (sanity reproduction of the published cell)

Difference from the published ablation driver: label fields are derived from the
M15 map **in the same augmented frame as the network input** (recovered from the
batch's own normalised M15 channel), so every cell sees spatially consistent
labels. The published driver reads M15 from disk un-augmented while the mask has
been flipped/rotated, which misaligns the flip set (measured flip-set IoU 0.54).
A `physprior_disk` cell reproduces the published path for direct comparison.

Outputs go to output/m15_label_control.json only. No published artifact,
checkpoint or manifest is touched.
"""
import os, sys, json, time, gc

ROOT = r"E:/Claude code/project/noise-label-cloud"
sys.path.insert(0, ROOT)
os.environ.setdefault("NL_SPLIT_TAG", "grp")
os.environ.setdefault("NL_CKPT_TAG", "ctl")
os.environ.setdefault("NL_SEED", "42")
os.environ.setdefault("NL_BASEMAP_DIR", r"E:\Data\Unet_Dataset\base_map_trainval")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import MTUNET_DATASET, MTUNET_NORM_STATS, train_cfg
from data import make_dataloaders
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet, MT_Loss
from methods.baseline import EarlyStopping
from methods.physical_prior import (apply_physical_correction, correct_batch_labels,
                                    threshold_label_field)
from determinism import seed_everything

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_TAG = os.environ.get("NL_CKPT_TAG", "")
OUT = rf"{ROOT}/output/m15_label_control{'_' + _TAG if _TAG else ''}.json"
M15_T = 264.0          # matched to the primary PhysPrior preset
STD_MAX = 1.5
_s = np.load(MTUNET_NORM_STATS)
MOD_MEAN, MOD_STD = float(_s["mod_mean"]), float(_s["mod_std"])
CELLS = [("raw", 2), ("raw", 3), ("physprior_disk", 2), ("physprior", 2),
         ("m15thresh", 2), ("physprior", 3), ("m15thresh", 3)]


def make_labels(label_mode, batch):
    masks = batch["mask_noisy"]
    if label_mode == "raw":
        return masks
    if label_mode == "m15thresh":
        return threshold_label_field(batch, M15_T)
    if label_mode == "physprior":
        return correct_batch_labels(batch, m15_min=M15_T, std_max=STD_MAX)
    if label_mode == "physprior_disk":
        # The published training path: M15 read from disk, untransformed,
        # against an already augmented mask. Kept to quantify the bug.
        corrected = []
        for j, fn in enumerate(batch["filename"]):
            npz = np.load(os.path.join(MTUNET_DATASET, "Train_grp", fn))
            m15 = npz.get("X_m15", npz.get("X_mod", None))
            c, _, _ = apply_physical_correction(
                masks[j].cpu().numpy().astype(np.int32), m15,
                m15_min=M15_T, std_max=STD_MAX)
            corrected.append(torch.from_numpy(c).long())
        return torch.stack(corrected)
    raise ValueError(label_mode)


def train_one(label_mode, n_channels, max_epochs=20, seed=42, epochs_only=False):
    seed_everything(seed)
    use_basemap = (n_channels == 3)
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size, use_basemap=use_basemap,
        use_phase4=True)
    model = MT_UNet(backbone="resnet34", in_channels=n_channels, mask_classes=4,
                    use_film=True, use_evidential=False).to(DEVICE)
    criterion = MT_Loss(weight_center=train_cfg.weight_center,
                        dice_weight=train_cfg.dice_weight).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                      weight_decay=train_cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg.T_0,
                                            T_mult=train_cfg.T_mult, eta_min=1e-6)
    early_stop = EarlyStopping(patience=10 ** 6)      # fixed budget, no early stop
    epoch_times = []
    stop = 1 if epochs_only else max_epochs
    for epoch in range(1, stop + 1):
        t0 = time.time()
        model.train()
        tot = 0.0
        for batch in train_loader:
            x = batch["image"].to(DEVICE)
            y_mask = make_labels(label_mode, batch).to(DEVICE)
            optimizer.zero_grad()
            pred_m, pred_h = model(x, moon_phase=batch["moon_phase"].to(DEVICE),
                                   solar_zenith=batch["solar_zenith"].to(DEVICE))
            loss, _, _, _ = criterion(pred_m, pred_h, y_mask,
                                      batch["height"].to(DEVICE),
                                      batch["radar_loc"].to(DEVICE),
                                      batch["center_label"].to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            tot += loss.item()
        scheduler.step(epoch - 1)
        dt = time.time() - t0
        epoch_times.append(dt)
        print(f"  [{label_mode} {n_channels}ch ep={epoch:02d}] "
              f"loss={tot/max(len(train_loader),1):.4f} ({dt:.0f}s)", flush=True)

    model.eval()
    test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False,
                           use_basemap=use_basemap)
    recs = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            s = test_ds[i]
            cl = s["center_label"]
            if cl != cl:
                continue
            logits, _ = model(s["image"].unsqueeze(0).to(DEVICE),
                              moon_phase=torch.tensor([s["moon_phase"]], device=DEVICE),
                              solar_zenith=torch.tensor([s["solar_zenith"]], device=DEVICE))
            pc = logits.argmax(1)[0].cpu().numpy()
            ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
            win = pc[max(0, ry - 1):min(128, ry + 2), max(0, rx - 1):min(128, rx + 2)]
            bw = (win >= 2).astype(int)
            pbin = 1 if bw.sum() > 0 else (0 if bw.mean() <= 0.5 else 1)
            recs.append({"fname": s["filename"], "pred": int(pbin),
                         "truth": int(cl)})
    tp = sum(1 for r in recs if r["pred"] == 1 and r["truth"] == 1)
    fp = sum(1 for r in recs if r["pred"] == 1 and r["truth"] == 0)
    fn = sum(1 for r in recs if r["pred"] == 0 and r["truth"] == 1)
    tn = sum(1 for r in recs if r["pred"] == 0 and r["truth"] == 0)
    n = tp + fp + fn + tn
    rec = {"label_mode": label_mode, "n_channels": n_channels, "seed": seed,
           "epochs": len(epoch_times), "sec_per_epoch": float(np.mean(epoch_times)),
           "protocol": "fixed 20-epoch budget, final-epoch weights, 3x3-OR test",
           "test": {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "acc": (tp + tn) / max(n, 1) * 100,
                    "recall": tp / max(tp + fn, 1) * 100,
                    "specificity": tn / max(tn + fp, 1) * 100,
                    "balanced_accuracy": (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)) / 2 * 100,
                    "f1": 2 * tp / max(2 * tp + fp + fn, 1) * 100},
           "test_records": recs}
    torch.cuda.empty_cache(); gc.collect()
    return rec


def _argv(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == "__main__":
    paired = "--paired" in sys.argv
    quick = "--quick" in sys.argv
    if paired:
        # Paired-seed label-source comparison: one fixed budget for all three
        # label sources, so the seed is the only thing that varies. This is the
        # comparison the manuscript lists as still-required.
        seeds = [int(s) for s in (_argv("--seeds", "0,1,2,3,4")).split(",")]
        modes = (_argv("--modes", "raw,physprior,m15thresh")).split(",")
        chans = [int(c) for c in (_argv("--channels", "2,3")).split(",")]
        OUT = rf"{ROOT}/output/label_source_paired_5seed{'_' + _TAG if _TAG else ''}.json"
        results = json.load(open(OUT)) if os.path.exists(OUT) else []
        done = {(r["label_mode"], r["n_channels"], r["seed"]) for r in results}
        plan = [(m, c, s) for m in modes for c in chans for s in seeds
                if (m, c, s) not in done]
        print(f"=== paired-seed label-source block: {len(plan)} runs of "
              f"{len(modes)*len(chans)*len(seeds)} (resumed, {len(done)} done) ===",
              flush=True)
        for lm, nc, sd in plan:
            print(f"\n----- {lm} / {nc}ch / seed {sd} -----", flush=True)
            r = train_one(lm, nc, max_epochs=20, seed=sd)
            results.append(r)
            json.dump(results, open(OUT, "w"), indent=2, default=float)
            t = r["test"]
            print(f"  acc {t['acc']:.2f}%  BA {t['balanced_accuracy']:.2f}  "
                  f"recall {t['recall']:.1f}%  spec {t['specificity']:.1f}%  "
                  f"tp/fp/fn/tn {t['tp']}/{t['fp']}/{t['fn']}/{t['tn']}  "
                  f"[{len(results)}/{len(plan)+len(done)}]", flush=True)
        print(f"\nwrote {OUT}", flush=True)
        sys.exit(0)

    which = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "all"
    cells = CELLS if which == "all" else [c for c in CELLS if c[0] == which or str(c[1]) == which]
    if quick:
        cells = cells[:1]
    print(f"=== M15-threshold label control on {DEVICE} | cells {cells} ===", flush=True)
    results = []
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    for lm, nc in cells:
        print(f"\n----- {lm} / {nc}ch -----", flush=True)
        r = train_one(lm, nc, max_epochs=2 if quick else 20, epochs_only=quick)
        results = [x for x in results if not (x["label_mode"] == lm and x["n_channels"] == nc)]
        results.append(r)
        if not quick:
            json.dump(results, open(OUT, "w"), indent=2, default=float)
        t = r["test"]
        print(f"  acc {t['acc']:.2f}%  BA {t['balanced_accuracy']:.2f}  "
              f"recall {t['recall']:.1f}%  spec {t['specificity']:.1f}%  "
              f"tp/fp/fn/tn {t['tp']}/{t['fp']}/{t['fn']}/{t['tn']}", flush=True)
        if quick:
            print(f"  (quick mode: {r['sec_per_epoch']:.0f}s/epoch -> "
                  f"20 epochs ~= {r['sec_per_epoch']*20/60:.1f} min)", flush=True)
