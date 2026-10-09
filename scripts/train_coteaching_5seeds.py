"""5-seed Co-Teaching training on Phase 4 split with PhysPrior labels.
Two networks filtering peer samples; same seed-budget as PhysPrior 5-seed.
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
from methods.baseline import EarlyStopping
from methods.physical_prior import apply_physical_correction


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
FORGET_RATE = 0.3
NUM_CLASSES = 4


def apply_physprior_to_batch(batch):
    fns = batch["filename"]
    masks = batch["mask_noisy"]
    corrected = []
    for j, fn in enumerate(fns):
        npz = np.load(os.path.join(_split_dir(MTUNET_DATASET, "Train"), fn))
        m15 = npz.get("X_m15")
        if m15 is None: corrected.append(masks[j]); continue
        c, _, _ = apply_physical_correction(
            masks[j].cpu().numpy().astype(np.int32), m15,
            m15_min=264.0, std_max=1.5)
        corrected.append(torch.from_numpy(c).long())
    return torch.stack(corrected)


def get_loss(loss_module, pred_m, pred_h, y_mask, y_h, radar, center, ce_only=False):
    loss, _, _, _ = loss_module(pred_m, pred_h, y_mask, y_h, radar, center)
    return loss


@torch.no_grad()
def eval_val(model, val_loader):
    model.eval()
    correct = 0; total = 0
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
            if pbin == int(cl): correct += 1
            total += 1
    return correct / max(total, 1)


def co_teaching_loss(loss_a, loss_b, y_mask, forget_rate, class_weights=None):
    """Co-Teaching: select low-loss samples for peer.
    For each model, keep the (1 - forget_rate) fraction of lowest-loss samples
    on the OTHER model's predictions.
    Inputs: loss_a, loss_b are per-sample loss tensors of shape [B].
    """
    if loss_a.dim() > 1:
        losses_a = loss_a.reshape(loss_a.size(0), -1).mean(1)
    else:
        losses_a = loss_a
    if loss_b.dim() > 1:
        losses_b = loss_b.reshape(loss_b.size(0), -1).mean(1)
    else:
        losses_b = loss_b
    n_keep = max(1, int((1.0 - forget_rate) * float(losses_a.size(0))))
    # Sort losses from low to high
    _, idx_a = losses_a.sort()
    _, idx_b = losses_b.sort()
    keep_a = idx_a[:n_keep]
    keep_b = idx_b[:n_keep]
    # Each model trains on samples it itself ranked as low-loss
    loss_a_sel = loss_a[keep_a]
    loss_b_sel = loss_b[keep_b]
    return loss_a_sel.mean(), loss_b_sel.mean()


def train_one_seed(seed):
    seed_everything(seed)
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size,
        use_basemap=True, use_phase4=True)

    history = {"train_loss_a": [], "train_loss_b": [], "val_a": [], "val_b": [], "val_ens": []}
    model_a = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                       use_film=True, use_evidential=False).to(DEVICE)
    model_b = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                       use_film=True, use_evidential=False).to(DEVICE)
    loss_fn = MT_Loss(weight_center=train_cfg.weight_center,
                       dice_weight=train_cfg.dice_weight, return_per_sample=True).to(DEVICE)
    opt_a = AdamW(model_a.parameters(), lr=train_cfg.learning_rate,
                   weight_decay=train_cfg.weight_decay)
    opt_b = AdamW(model_b.parameters(), lr=train_cfg.learning_rate,
                   weight_decay=train_cfg.weight_decay)
    sched_a = CosineAnnealingWarmRestarts(opt_a, T_0=train_cfg.T_0,
                                            T_mult=train_cfg.T_mult, eta_min=1e-6)
    sched_b = CosineAnnealingWarmRestarts(opt_b, T_0=train_cfg.T_0,
                                            T_mult=train_cfg.T_mult, eta_min=1e-6)
    early = EarlyStopping(patience=PATIENCE)
    best_val_a = 0.0
    best_epoch = 0
    best_state_a = None
    best_state_b = None

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model_a.train(); model_b.train()
        for batch in train_loader:
            x = batch["image"].to(DEVICE)
            moon = batch["moon_phase"].to(DEVICE)
            sza = batch["solar_zenith"].to(DEVICE)
            radar = batch["radar_loc"].to(DEVICE)
            center = batch["center_label"].to(DEVICE)
            y_h = batch["height"].to(DEVICE)
            y_mask_pp = apply_physprior_to_batch(batch).to(DEVICE)
            opt_a.zero_grad(); opt_b.zero_grad()
            logits_a, h_a = model_a(x, moon_phase=moon, solar_zenith=sza)
            logits_b, h_b = model_b(x, moon_phase=moon, solar_zenith=sza)
            # Unpack 5-tuple: (total, loss_mask, loss_height, loss_center, per_sample)
            total_a, _, _, _, per_a = loss_fn(logits_a, h_a, y_mask_pp, y_h, radar, center)
            total_b, _, _, _, per_b = loss_fn(logits_b, h_b, y_mask_pp, y_h, radar, center)
            if y_mask_pp.shape[0] != per_a.shape[0]:
                print(f"  BATCH MISMATCH: y_mask_pp={y_mask_pp.shape}, per_a={per_a.shape}")
                continue
            # Co-Teaching: each model trains on its own low-loss samples (peer filter)
            # per_a, per_b are shape (B,)
            loss_a_sel, loss_b_sel = co_teaching_loss(
                per_a, per_b, y_mask_pp, FORGET_RATE)
            loss_a_sel.backward()
            loss_b_sel.backward()
            nn.utils.clip_grad_norm_(model_a.parameters(), train_cfg.grad_clip)
            nn.utils.clip_grad_norm_(model_b.parameters(), train_cfg.grad_clip)
            opt_a.step(); opt_b.step()
        sched_a.step(epoch - 1); sched_b.step(epoch - 1)
        # Eval with model_a as representative
        val_a = eval_val(model_a, val_loader)
        val_b = eval_val(model_b, val_loader)
        val_ens = max(val_a, val_b)
        history["val_a"].append(val_a)
        history["val_b"].append(val_b)
        history["val_ens"].append(val_ens)
        if val_ens > best_val_a:
            best_val_a = val_ens
            best_epoch = epoch
            best_epoch = epoch
            best_state_a = {k: v.detach().cpu().clone() for k, v in model_a.state_dict().items()}
            best_state_b = {k: v.detach().cpu().clone() for k, v in model_b.state_dict().items()}
        dt = time.time() - t0
        print(f"  [seed={seed} ep={epoch}] val_a={val_a:.1%} val_b={val_b:.1%} ({dt:.0f}s)", flush=True)
        if early.step(val_ens):
            break

    # Test ensemble
    model_a.load_state_dict(best_state_a)
    model_b.load_state_dict(best_state_b)
    model_a.eval(); model_b.eval()
    test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False, use_basemap=True)
    records = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            s = test_ds[i]
            cl = s["center_label"]
            if cl != cl: continue
            x = s["image"].unsqueeze(0).to(DEVICE)
            moon = torch.tensor([s["moon_phase"]], device=DEVICE)
            sza = torch.tensor([s["solar_zenith"]], device=DEVICE)
            la, _ = model_a(x, moon_phase=moon, solar_zenith=sza)
            lb, _ = model_b(x, moon_phase=moon, solar_zenith=sza)
            pa = la.argmax(1)[0].cpu().numpy()
            pb = lb.argmax(1)[0].cpu().numpy()
            ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
            y0, y1 = max(0, ry - 1), min(128, ry + 2)
            x0, x1 = max(0, rx - 1), min(128, rx + 2)
            wa = (pa[y0:y1, x0:x1] >= 2).astype(int)
            wb = (pb[y0:y1, x0:x1] >= 2).astype(int)
            pe = (wa + wb) // 2  # average
            pbin = 1 if pe.sum() > 0 else 0
            truth = int(cl)
            records.append({"fname": s["filename"], "pred": int(pbin), "truth": truth})
    arr = np.array([(r["pred"], r["truth"]) for r in records])
    tp = int(((arr[:,0]==1)&(arr[:,1]==1)).sum())
    fp = int(((arr[:,0]==1)&(arr[:,1]==0)).sum())
    tn = int(((arr[:,0]==0)&(arr[:,1]==0)).sum())
    fn = int(((arr[:,0]==0)&(arr[:,1]==1)).sum())
    n = tp+fp+tn+fn
    acc = (tp+tn)/max(n,1)
    return {"seed": seed, "best_val_acc": best_val_a,
            "test": {"n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": float(acc)},
            "test_records": records}


if __name__ == "__main__":
    out_path = rf"E:/Claude code/project/noise-label-cloud/output/coteaching_5seed_results{_out_tag()}.json"
    results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        r = train_one_seed(seed)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        hist_path = out_path.replace("coteaching_5seed_results", f"coteaching_5seed_history_seed{seed}")
        with open(hist_path, "w") as f:
            _va = (r.get("history", {}) or {}).get("val_ens", [])
            json.dump({"seed": seed, "history": r.get("history", {}),
                       "best_val_acc": max(_va) if _va else 0.0,
                       "test_acc": r.get("test", {}).get("acc", 0.0),
                       "best_epoch": r.get("best_epoch", 0)}, f, indent=2, default=float)
        print(f"  saved acc={r['test']['acc']*100:.2f}%", flush=True)
        torch.cuda.empty_cache()
    # Summary
    accs = [r["test"]["acc"] for r in results]
    f1s = []
    for r in results:
        tp, fp, tn, fn = r["test"]["tp"], r["test"]["fp"], r["test"]["tn"], r["test"]["fn"]
        f1 = 2*tp / (2*tp + fp + fn + 1e-8)
        f1s.append(f1)
    print(f"\n=== 5-seed summary ===")
    print(f"Accuracy: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}pp")
    print(f"F1      : {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}pp")
    summary = {
        "n_seeds": len(results),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_sd": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_sd": float(np.std(f1s)),
        "per_seed": [{"seed": r["seed"], "acc": r["test"]["acc"],
                       "n": r["test"]["n"],
                       "tp": r["test"]["tp"], "fp": r["test"]["fp"],
                       "tn": r["test"]["tn"], "fn": r["test"]["fn"]}
                      for r in results]
    }
    with open(rf"E:/Claude code/project/noise-label-cloud/output/coteaching_5seed_summary{_out_tag()}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {out_path}")
