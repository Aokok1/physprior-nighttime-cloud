"""Step 2: 2x2 basemap ablation.
训练 4 种配置 (label × channels):
  A: Raw CLDMSK labels + Standard training + 2ch (DNB+M15)
  B: Raw CLDMSK labels + Standard training + 3ch (DNB+Basemap+M15)
  C: PhysPrior labels + DL training + 2ch
  D: PhysPrior labels + DL training + 3ch

Phase 4 no-leakage split, seed=42, 20 epochs (early stop).
评估所有 4 个在 n=197 radar test 上。
"""
import os, sys, time, json, gc
sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
from determinism import seed_everything
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import MTUNET_DATASET, train_cfg, CHECKPOINT_DIR
from data import make_dataloaders
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet, MT_Loss
from methods.baseline import EarlyStopping, compute_binary_metrics
from methods.physical_prior import apply_physical_correction, correct_batch_labels


def _split_dir(root, mode="Train"):
    """Resolve the active split directory (mirrors data/dataset.py)."""
    tag = os.environ.get("NL_SPLIT_TAG", "phase4")
    return os.path.join(root, f"{mode}_{tag}")


def _out_tag():
    """Suffix for derived artifacts when running on an alternative split."""
    t = os.environ.get("NL_CKPT_TAG", "")
    return f"_{t}" if t else ""


DEVICE = "cuda"

def apply_physprior_to_batch(batch):
    """Correct this batch's labels with the deployed moderate preset.

    Evaluated on batch["m15_raw"], the brightness temperature of the same scene
    after the same geometric augmentation as batch["mask_noisy"]. It used to
    read X_m15 from disk, which is untransformed: the flip set then matched the
    corrected field with IoU 0.54 and changed 3.6% of the decisions at the
    radar-collocated pixel. See methods.physical_prior.correct_batch_labels.
    """
    return correct_batch_labels(batch, m15_min=264.0, std_max=1.5)


def train_one(label_mode, n_channels, max_epochs=20, patience=None):
    seed_everything(42)
    # With final-epoch selection the budget must be identical for all four cells,
    # so early stopping is disabled: every cell runs the full max_epochs.
    selection = os.environ.get("NL_ABL_SELECT", "final")
    if patience is None:
        patience = 10 ** 6 if selection == "final" else 6

    use_basemap = (n_channels == 3)
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size,
        use_basemap=use_basemap, use_phase4=True)
    model = MT_UNet(backbone="resnet34", in_channels=n_channels, mask_classes=4,
                     use_film=True, use_evidential=False).to(DEVICE)
    criterion = MT_Loss(weight_center=train_cfg.weight_center,
                          dice_weight=train_cfg.dice_weight).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                        weight_decay=train_cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg.T_0,
                                             T_mult=train_cfg.T_mult, eta_min=1e-6)
    early_stop = EarlyStopping(patience=patience)
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        model.train()
        sum_loss = 0
        for batch in train_loader:
            x = batch["image"].to(DEVICE)
            moon = batch["moon_phase"].to(DEVICE)
            sza = batch["solar_zenith"].to(DEVICE)
            radar = batch["radar_loc"].to(DEVICE)
            center = batch["center_label"].to(DEVICE)
            y_h = batch["height"].to(DEVICE)
            if label_mode == "raw":
                y_mask = batch["mask_noisy"].to(DEVICE)
            elif label_mode == "physprior":
                y_mask = apply_physprior_to_batch(batch).to(DEVICE)
            optimizer.zero_grad()
            pred_m, pred_h = model(x, moon_phase=moon, solar_zenith=sza)
            loss, _, _, _ = criterion(pred_m, pred_h, y_mask, y_h, radar, center)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            sum_loss += loss.item()
        scheduler.step(epoch - 1)
        # Val
        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(DEVICE)
                moon = batch["moon_phase"].to(DEVICE)
                sza = batch["solar_zenith"].to(DEVICE)
                logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
                pred_cls = logits.argmax(1)
                for i in range(x.size(0)):
                    cl = batch["center_label"][i].item()
                    if cl != cl: continue
                    ry, rx = int(batch["radar_loc"][i, 0]), int(batch["radar_loc"][i, 1])
                    y0, y1 = max(0, ry - 1), min(128, ry + 2)
                    x0, x1 = max(0, rx - 1), min(128, rx + 2)
                    win = pred_cls[i, y0:y1, x0:x1]
                    bw = (win >= 2).int()
                    pbin = 1 if bw.sum() > 0 else (0 if bw.float().mean() <= 0.5 else 1)
                    if pbin == int(cl): v_correct += 1
                    v_total += 1
            val_acc = v_correct / max(v_total, 1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
            best_epoch = epoch
        dt = time.time() - t0
        print(f"  [{label_mode} {n_channels}ch ep={epoch:02d}] train_loss={sum_loss/max(len(train_loader),1):.4f} "
              f"val_acc={val_acc:.1%} ({dt:.0f}s)", flush=True)
        if early_stop.step(val_acc):
            print(f"  [{label_mode} {n_channels}ch] early stop at epoch {epoch}", flush=True)
            break

    # Test.
    #
    # Default: evaluate the FINAL-epoch weights. The alternative — picking the
    # epoch with the best validation radar agreement — is not usable for this
    # ablation: that agreement is measured on the 28 radar-labelled validation
    # patches, where an always-cloud prediction already scores 17/28, and the
    # criterion peaks at epoch 1-2 for three of the four cells. Selecting on it
    # compares early, near-trivial checkpoints against a trained one.
    # Set NL_ABL_SELECT=bestval to reproduce the earlier selection rule.
    # Set NL_ABL_SELECT=bestval to reproduce the earlier selection rule.
    if selection == "bestval":
        model.load_state_dict(best_state)
    model.eval()
    test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False,
                           use_basemap=use_basemap)
    tp = fp = tn = fn = 0
    test_records = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            s = test_ds[i]
            cl = s["center_label"]
            if cl != cl: continue
            x = s["image"].unsqueeze(0).to(DEVICE)
            moon = torch.tensor([s["moon_phase"]], device=DEVICE)
            sza = torch.tensor([s["solar_zenith"]], device=DEVICE)
            logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
            pred_cls = logits.argmax(1)[0].cpu().numpy()
            ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
            y0, y1 = max(0, ry - 1), min(128, ry + 2)
            x0, x1 = max(0, rx - 1), min(128, rx + 2)
            win = pred_cls[y0:y1, x0:x1]
            bw = (win >= 2).astype(int)
            pbin = 1 if bw.sum() > 0 else (0 if bw.mean() <= 0.5 else 1)
            truth = int(cl)
            test_records.append({"fname": s["filename"], "pred": int(pbin),
                                  "truth": truth})
            if pbin == 1 and truth == 1: tp += 1
            elif pbin == 1 and truth == 0: fp += 1
            elif pbin == 0 and truth == 0: tn += 1
            else: fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); sp = tn/max(tn+fp,1)
    f1 = 2*p*r/max(p+r,1e-9)
    ba = (r + sp) / 2
    denom = np.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1))
    mcc = (tp*tn - fp*fn)/denom if denom > 0 else 0
    return {
        "label_mode": label_mode, "n_channels": n_channels,
        "best_val_acc": best_val_acc, "best_epoch": best_epoch,
        "epochs_run": max_epochs, "selection": selection,
        "test": {"n": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                  "acc": float(acc), "precision": float(p), "recall": float(r),
                  "specificity": float(sp), "f1": float(f1),
                  "balanced_accuracy": float(ba), "mcc": float(mcc)},
        "test_records": test_records,
    }


if __name__ == "__main__":
    import sys
    print("=== 2x2 basemap ablation (197-test, group-disjoint split) ===", flush=True)
    print(f"    checkpoint selection: {os.environ.get('NL_ABL_SELECT', 'final')}", flush=True)
    sys.stdout.flush()
    out_path = rf"E:/Claude code/project/noise-label-cloud/output/2x2_ablation{_out_tag()}.json"
    all_results = []
    for label_mode, n_ch, label in [
        ("raw",          2, "A: Raw CLDMSK + 2ch (DNB+M15)"),
        ("raw",          3, "B: Raw CLDMSK + 3ch (DNB+Basemap+M15)"),
        ("physprior",    2, "C: PhysPrior labels + 2ch (DNB+M15)"),
        ("physprior",    3, "D: PhysPrior labels + 3ch (DNB+Basemap+M15)"),
    ]:
        print(f"\n----- {label} -----", flush=True)
        r = train_one(label_mode, n_ch)
        r["label"] = label
        all_results.append(r)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"  Saved → {out_path}  | acc={r['test']['acc']*100:.2f}%", flush=True)
        sys.stdout.flush()
        # clear CUDA cache between runs
        torch.cuda.empty_cache(); gc.collect()

    # Print 2x2 table
    print("\n\n=== 2x2 ABLATION TABLE ===")
    print(f"                           2ch (DNB+M15)    3ch (DNB+Basemap+M15)")
    for lm in ["raw", "physprior"]:
        a_2ch = next((r for r in all_results if r["label_mode"]==lm and r["n_channels"]==2), None)
        a_3ch = next((r for r in all_results if r["label_mode"]==lm and r["n_channels"]==3), None)
        label = "Raw CLDMSK" if lm == "raw" else "PhysPrior"
        print(f"  {label:<22}  {a_2ch['test']['acc']*100:6.1f}%            {a_3ch['test']['acc']*100:6.1f}%")

    # Basemap effect
    raw_effect = next(r["test"]["acc"] for r in all_results if r["label_mode"]=="raw" and r["n_channels"]==3) - \
                 next(r["test"]["acc"] for r in all_results if r["label_mode"]=="raw" and r["n_channels"]==2)
    pp_effect = next(r["test"]["acc"] for r in all_results if r["label_mode"]=="physprior" and r["n_channels"]==3) - \
                next(r["test"]["acc"] for r in all_results if r["label_mode"]=="physprior" and r["n_channels"]==2)
    label_effect = next(r["test"]["acc"] for r in all_results if r["label_mode"]=="physprior" and r["n_channels"]==2) - \
                  next(r["test"]["acc"] for r in all_results if r["label_mode"]=="raw" and r["n_channels"]==2)

    print(f"\nBasemap effect (3ch - 2ch):")
    print(f"  With raw CLDMSK    : {raw_effect*100:+.2f}pp")
    print(f"  With PhysPrior     : {pp_effect*100:+.2f}pp")
    print(f"\nLabel-correction effect (PhysPrior - raw, fixed channels):")
    print(f"  2ch  : {label_effect*100:+.2f}pp")
    print(f"  3ch  : {(next(r['test']['acc'] for r in all_results if r['label_mode']=='physprior' and r['n_channels']==3) - next(r['test']['acc'] for r in all_results if r['label_mode']=='raw' and r['n_channels']==3))*100:+.2f}pp")
