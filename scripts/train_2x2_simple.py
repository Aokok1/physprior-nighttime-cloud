"""Minimal 2x2 ablation: 4 runs, 5 epoch each, raw output."""
import os, sys, time, json
sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
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
from methods.physical_prior import apply_physical_correction

DEVICE = "cuda"

def apply_pp(batch):
    fns = batch["filename"]
    masks = batch["mask_noisy"]
    corrected = []
    for j, fn in enumerate(fns):
        npz = np.load(os.path.join(MTUNET_DATASET, "Train_phase4", fn))
        m15 = npz.get("X_m15")
        if m15 is None: corrected.append(masks[j]); continue
        c, _, _ = apply_physical_correction(
            masks[j].cpu().numpy().astype(np.int32), m15,
            m15_min=264.0, std_max=1.5)
        corrected.append(torch.from_numpy(c).long())
    return torch.stack(corrected)


def run(label_mode, n_ch, max_epochs=8):
    torch.manual_seed(42)
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size,
        use_basemap=(n_ch == 3), use_phase4=True)
    model = MT_UNet(backbone="resnet34", in_channels=n_ch, mask_classes=4,
                     use_film=True, use_evidential=False).to(DEVICE)
    criterion = MT_Loss(weight_center=train_cfg.weight_center,
                          dice_weight=train_cfg.dice_weight).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=train_cfg.learning_rate,
                        weight_decay=train_cfg.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=train_cfg.T_0,
                                             T_mult=train_cfg.T_mult, eta_min=1e-6)
    early_stop = EarlyStopping(patience=6)
    best_val = 0.0
    best_state = None
    history = []

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
            else:
                y_mask = apply_pp(batch).to(DEVICE)
            optimizer.zero_grad()
            pred_m, pred_h = model(x, moon_phase=moon, solar_zenith=sza)
            loss, _, _, _ = criterion(pred_m, pred_h, y_mask, y_h, radar, center)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            sum_loss += loss.item()
        scheduler.step(epoch - 1)
        # Quick val
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
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        dt = time.time() - t0
        print(f"  ep={epoch} train_loss={sum_loss/len(train_loader):.4f} val_acc={val_acc:.1%} ({dt:.0f}s)", flush=True)
        if early_stop.step(val_acc):
            break

    # Test
    model.load_state_dict(best_state)
    model.eval()
    test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False,
                           use_basemap=(n_ch == 3))
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
            test_records.append({"fname": s["filename"], "pred": int(pbin), "truth": truth})
            if pbin == 1 and truth == 1: tp += 1
            elif pbin == 1 and truth == 0: fp += 1
            elif pbin == 0 and truth == 0: tn += 1
            else: fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    return {"label_mode": label_mode, "n_channels": n_ch,
            "best_val_acc": best_val,
            "test": {"n": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                      "acc": float(acc)},
            "test_records": test_records}


if __name__ == "__main__":
    out_path = r"E:/Claude code/project/noise-label-cloud/output/2x2_ablation.json"
    all_results = []
    print("=== 2x2 basemap ablation ===", flush=True)
    for label_mode, n_ch, name in [
        ("raw",          2, "A_raw_2ch"),
        ("raw",          3, "B_raw_3ch"),
        ("physprior",    2, "C_pp_2ch"),
        ("physprior",    3, "D_pp_3ch"),
    ]:
        print(f"\n--- {name} (label={label_mode}, channels={n_ch}) ---", flush=True)
        r = run(label_mode, n_ch, max_epochs=8)
        all_results.append({"name": name, **r})
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"  saved. test acc={r['test']['acc']*100:.2f}%", flush=True)
        torch.cuda.empty_cache()

    print("\n=== 2x2 RESULTS ===")
    raw_2ch = next(r for r in all_results if r["name"] == "A_raw_2ch")
    raw_3ch = next(r for r in all_results if r["name"] == "B_raw_3ch")
    pp_2ch = next(r for r in all_results if r["name"] == "C_pp_2ch")
    pp_3ch = next(r for r in all_results if r["name"] == "D_pp_3ch")
    print(f"                    2ch (DNB+M15)    3ch (DNB+Basemap+M15)")
    print(f"  Raw CLDMSK        {raw_2ch['test']['acc']*100:6.1f}%          {raw_3ch['test']['acc']*100:6.1f}%")
    print(f"  PhysPrior         {pp_2ch['test']['acc']*100:6.1f}%          {pp_3ch['test']['acc']*100:6.1f}%")
    print(f"  Basemap effect (3ch - 2ch):")
    print(f"    Raw CLDMSK: {(raw_3ch['test']['acc'] - raw_2ch['test']['acc'])*100:+.2f}pp")
    print(f"    PhysPrior : {(pp_3ch['test']['acc'] - pp_2ch['test']['acc'])*100:+.2f}pp")
