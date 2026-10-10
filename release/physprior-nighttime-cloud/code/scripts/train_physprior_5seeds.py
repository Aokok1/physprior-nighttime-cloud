"""Step 4: 5-seed PhysPrior moderate training using Phase 4 split.
20-25 epochs (from phase4_physprior_log.txt observation: epoch 22 triggers early-stop on val).
Saves per-seed test predictions so that the final result is the mean ± SD of 5 frozen prediction files.
"""
import os, sys, time, json
sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
from determinism import seed_everything
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from config import MTUNET_DATASET, CHECKPOINT_DIR, LOG_DIR, train_cfg
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 25
PATIENCE = 8


def apply_physprior_to_batch(batch):
    """Correct the batch's labels with the moderate preset in its own frame.

    Previously read X_m15 from disk while the mask had already been flipped and
    rotated by the loader; see methods.physical_prior.correct_batch_labels.
    """
    return correct_batch_labels(batch, m15_min=264.0, std_max=1.5)


def train_one_seed(seed):
    seed_everything(seed)

    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size,
        use_basemap=True, use_phase4=True)

    model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                     use_film=True, use_evidential=False).to(DEVICE)
    criterion = MT_Loss(weight_center=train_cfg.weight_center,
                          dice_weight=train_cfg.dice_weight, return_per_sample=True).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                        weight_decay=train_cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg.T_0,
                                             T_mult=train_cfg.T_mult, eta_min=1e-6)
    early_stop = EarlyStopping(patience=PATIENCE)
    history = {"train_loss": [], "val_acc": [], "val_p": [], "val_r": [], "val_f1": []}
    best_val_acc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
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
            y_mask = apply_physprior_to_batch(batch).to(DEVICE)

            optimizer.zero_grad()
            pred_m, pred_h = model(x, moon_phase=moon, solar_zenith=sza)
            loss, _, _, _, _ = criterion(pred_m, pred_h, y_mask, y_h, radar, center)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            sum_loss += loss.item()
        scheduler.step(epoch - 1)
        avg_train_loss = sum_loss / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            v_correct = v_total = 0
            v_tp = v_fp = v_tn = v_fn = 0
            for batch in val_loader:
                x = batch["image"].to(DEVICE)
                moon = batch["moon_phase"].to(DEVICE)
                sza = batch["solar_zenith"].to(DEVICE)
                logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
                pc = logits.argmax(1)
                tp, fp, fn, tn = compute_binary_metrics(pc, batch["mask_noisy"].to(DEVICE))
                v_tp += tp; v_fp += fp; v_fn += fn; v_tn += tn
                for i in range(x.size(0)):
                    cl = batch["center_label"][i].item()
                    if cl != cl: continue
                    ry, rx = int(batch["radar_loc"][i, 0]), int(batch["radar_loc"][i, 1])
                    y0, y1 = max(0, ry - 1), min(128, ry + 2)
                    x0, x1 = max(0, rx - 1), min(128, rx + 2)
                    win = pc[i, y0:y1, x0:x1]
                    bw = (win >= 2).int()
                    pbin = 1 if bw.sum() > 0 else (0 if bw.float().mean() <= 0.5 else 1)
                    if pbin == int(cl): v_correct += 1
                    v_total += 1
            val_acc = v_correct / max(v_total, 1)
            vp = v_tp / max(v_tp + v_fp, 1); vr = v_tp / max(v_tp + v_fn, 1)
            vf1 = 2 * vp * vr / max(vp + vr, 1e-9)

        history["train_loss"].append(avg_train_loss)
        history["val_acc"].append(val_acc)
        history["val_p"].append(vp); history["val_r"].append(vr); history["val_f1"].append(vf1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}

        dt = time.time() - t0
        print(f"  [seed={seed} ep={epoch:02d}] train_loss={avg_train_loss:.4f} "
              f"val_acc={val_acc:.1%} P={vp:.1%} R={vr:.1%} F1={vf1:.1%} ({dt:.0f}s)",
              flush=True)
        if early_stop.step(val_acc):
            print(f"  [seed={seed}] early stop at epoch {epoch}", flush=True)
            break

    # Load best state for test
    model.load_state_dict(best_state)
    model.eval()
    test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False)
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
    n_correct = tp + tn
    acc = (tp + tn) / max(total, 1)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    sp = tn / max(tn + fp, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    ba = (r + sp) / 2
    return {
        "seed": seed, "history": history, "best_val_acc": best_val_acc,
        "test": {"acc": acc, "precision": p, "recall": r, "f1": f1,
                  "balanced_accuracy": ba, "specificity": sp,
                  "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                  "n": total, "n_correct": int(n_correct)},
        "test_records": test_records,
        "best_epoch": best_epoch,
    }


if __name__ == "__main__":
    print(f"\n===== PhysPrior moderate — 5-seed repeated training =====")
    all_results = []
    out_path = rf"E:/Claude code/project/noise-label-cloud/output/physprior_5seed_results{_out_tag()}.json"
    for seed in SEEDS:
        print(f"\n----- Seed {seed} -----", flush=True)
        r = train_one_seed(seed)
        all_results.append(r)
        # Save after each seed for crash safety
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        # Save per-seed training history (train/val loss per epoch + best epoch)
        hist_path = out_path.replace("physprior_5seed_results", f"physprior_5seed_history_seed{seed}")
        with open(hist_path, "w") as f:
            hist_to_save = r.get("history", {"train_loss": [], "val_acc": [], "val_p": [], "val_r": [], "val_f1": []})
            # best_val_acc must be the VALIDATION agreement (the quantity the
            # checkpoint was selected on), not the test agreement. Writing the
            # test value here made the history files look as if the checkpoint
            # had been chosen on the test set.
            val_trace = hist_to_save.get("val_acc", [])
            json.dump({"seed": seed, "history": hist_to_save,
                       "best_val_acc": max(val_trace) if val_trace else 0.0,
                       "test_acc": r["test"].get("acc", 0.0),
                       "best_epoch": r.get("best_epoch", 0)}, f, indent=2, default=float)
        print(f"  [seed={seed}] acc={r['test']['acc']*100:.2f}% "
              f"F1={r['test']['f1']*100:.2f}%  saved.", flush=True)

    # Aggregate
    accs = [r["test"]["acc"] for r in all_results]
    f1s = [r["test"]["f1"] for r in all_results]
    sps = [r["test"]["specificity"] for r in all_results]
    rs = [r["test"]["recall"] for r in all_results]
    bas = [r["test"]["balanced_accuracy"] for r in all_results]
    summary = {
        "n_seeds": len(all_results),
        "accuracy_mean": float(np.mean(accs)), "accuracy_sd": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)), "f1_sd": float(np.std(f1s)),
        "specificity_mean": float(np.mean(sps)), "specificity_sd": float(np.std(sps)),
        "recall_mean": float(np.mean(rs)), "recall_sd": float(np.std(rs)),
        "balanced_accuracy_mean": float(np.mean(bas)),
        "balanced_accuracy_sd": float(np.std(bas)),
        "per_seed": [{"seed": r["seed"], **{k: r["test"][k] for k in
                       ("acc", "f1", "specificity", "recall", "balanced_accuracy")}}
                       for r in all_results],
    }
    print("\n===== 5-seed summary =====")
    print(f"  Accuracy          : {summary['accuracy_mean']*100:.2f}% ± {summary['accuracy_sd']*100:.2f}pp")
    print(f"  F1                : {summary['f1_mean']*100:.2f}% ± {summary['f1_sd']*100:.2f}pp")
    print(f"  Specificity       : {summary['specificity_mean']*100:.2f}% ± {summary['specificity_sd']*100:.2f}pp")
    print(f"  Recall            : {summary['recall_mean']*100:.2f}% ± {summary['recall_sd']*100:.2f}pp")
    print(f"  Balanced accuracy : {summary['balanced_accuracy_mean']*100:.2f}% ± {summary['balanced_accuracy_sd']*100:.2f}pp")

    sum_path = rf"E:/Claude code/project/noise-label-cloud/output/physprior_5seed_summary{_out_tag()}.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved → {sum_path}")
