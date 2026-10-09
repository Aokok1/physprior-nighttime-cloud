"""P1 论文任务(3位专家一致要求):PhysPrior + Co-Teaching 组合。

论文里只有 PhysPrior 和 Co-Teaching 各自 69.5%/69.6% 的报数,且论文明确写道
"the two approaches are complementary"——但没有实际跑过组合实验。
本次实现在 Phase 4 数据分割上用 PhysPrior-moderate 校正后的标签训练
双模型 Co-Teaching。

训练时长:<15 分钟 (RTX 4060, 40 epochs)。
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.ndimage import uniform_filter

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
from config import MTUNET_DATASET, train_cfg, CHECKPOINT_DIR, LOG_DIR
from data import make_dataloaders
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet, MT_Loss
from methods.baseline import EarlyStopping, compute_binary_metrics
from methods.physical_prior import apply_physical_correction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def apply_physprior_to_batch(ds_root, mode, batch):
    """读取本 batch 对应的 npz,拿原始 M15,调用 moderate PhysPrior校正,然后返回
    校正后的整 batch 标签张量(batch x H x W long tensor)。
    """
    fns = batch["filename"]
    masks = batch["mask_noisy"]  # (B,H,W)
    corrected = []
    for j, fn in enumerate(fns):
        npz = np.load(os.path.join(ds_root, mode, fn))
        m15 = npz.get("X_m15", npz.get("X_mod", None))
        if m15 is None:
            corrected.append(masks[j])
            continue
        c, _, _ = apply_physical_correction(
            masks[j].cpu().numpy().astype(np.int32), m15,
            m15_min=264.0, std_max=1.5,
        )
        corrected.append(torch.from_numpy(c).long())
    return torch.stack(corrected)


def per_sample_ce(pred_logits, y_mask):
    """返回 (B,) 每个样本的 mean CE。"""
    return F.cross_entropy(pred_logits, y_mask, reduction="none").mean(dim=[1, 2])


def train():
    print(f"\n{'='*60}")
    print("  PhysPrior + Co-Teaching (Phase 4 split, no leakage)")
    print(f"{'='*60}\n")
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size,
        use_basemap=True, use_phase4=True,
    )
    n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)
    print(f"Train: {n_train}  Val: {n_val}\n")

    # 双模型
    model_a = MT_UNet(
        backbone="resnet34", in_channels=3, mask_classes=4,
        use_film=True, use_evidential=False,
    ).to(DEVICE)
    model_b = MT_UNet(
        backbone="resnet34", in_channels=3, mask_classes=4,
        use_film=True, use_evidential=False,
    ).to(DEVICE)

    criterion = MT_Loss(
        weight_center=train_cfg.weight_center,
        dice_weight=train_cfg.dice_weight,
    ).to(DEVICE)

    optim_a = AdamW(model_a.parameters(), lr=train_cfg.learning_rate,
                    weight_decay=train_cfg.weight_decay)
    optim_b = AdamW(model_b.parameters(), lr=train_cfg.learning_rate,
                    weight_decay=train_cfg.weight_decay)
    sched_a = CosineAnnealingWarmRestarts(optim_a, T_0=train_cfg.T_0, T_mult=2, eta_min=1e-6)
    sched_b = CosineAnnealingWarmRestarts(optim_b, T_0=train_cfg.T_0, T_mult=2, eta_min=1e-6)

    # Co-Teaching settings
    forget_rate = 0.3   # CLDMSK 噪声 ~50% — 但 PhysPrior 已经去掉了系统性 FP,因此 forget_rate 调小
    n_warmup   = 5
    epochs     = 40    # 快速验证,co-teaching 收敛快
    early_stop = EarlyStopping(patience=8)

    log_path = os.path.join(LOG_DIR, "physprior_coteaching_train_log.txt")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path, "w") as f:
        f.write("epoch,loss_a,loss_b,val_loss,val_acc,val_p,val_r,val_f1,forget_rate,time\n")

    best_val_acc = 0.0
    best_epoch = 0
    t_overall = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        rate = forget_rate * min(1.0, (epoch - 1) / max(n_warmup, 1)) if epoch > 1 else 0.0
        model_a.train(); model_b.train()
        sum_la = sum_lb = 0; n_batch = 0

        for batch in train_loader:
            x = batch["image"].to(DEVICE)
            moon = batch["moon_phase"].to(DEVICE)
            sza  = batch["solar_zenith"].to(DEVICE)
            radar = batch["radar_loc"].to(DEVICE)
            center = batch["center_label"].to(DEVICE)
            y_h    = batch["height"].to(DEVICE)

            # PhysPrior 校正标签(整个 batch)
            y_corrected = apply_physprior_to_batch(MTUNET_DATASET, "Train_phase4", batch).to(DEVICE)

            # ── 互筛 ──────────────────────────────────────
            with torch.no_grad():
                pa, _ = model_a(x, moon_phase=moon, solar_zenith=sza)
                pb, _ = model_b(x, moon_phase=moon, solar_zenith=sza)
                ce_a = F.cross_entropy(pa, y_corrected, reduction="none")  # (B,H,W)
                ce_b = F.cross_entropy(pb, y_corrected, reduction="none")
                if rate > 0 and rate < 1.0:
                    flat_a = ce_a.flatten()
                    flat_b = ce_b.flatten()
                    n_a = flat_a.numel()
                    k = max(1, int((1 - rate) * n_a))
                    k = min(k, n_a)
                    thresh_a = flat_a.kthvalue(k).values
                    thresh_b = flat_b.kthvalue(k).values
                    mask_a = ce_b <= thresh_a  # A 训练用 B 认可的像素
                    mask_b = ce_a <= thresh_b
                else:
                    mask_a = torch.ones_like(ce_a, dtype=torch.bool)
                    mask_b = torch.ones_like(ce_b, dtype=torch.bool)

            # ── Train A ──
            optim_a.zero_grad()
            pa, hgt_a = model_a(x, moon_phase=moon, solar_zenith=sza)
            ce_full_a = F.cross_entropy(pa, y_corrected, reduction="none")
            loss_a = (ce_full_a * mask_a).sum() / max(mask_a.sum().item(), 1.0)
            # center loss on raw labels
            cl = torch.nan_to_num(center, nan=0.0).float()
            valid = ~torch.isnan(center)
            if valid.any():
                probs = F.softmax(pa, dim=1)
                p_cloud = (probs[:, 2] + probs[:, 3]).clamp(1e-6, 1 - 1e-6)
                sum_bce = 0.0
                for i in range(p_cloud.size(0)):
                    if not bool(valid[i].item()): continue
                    ry, rx = int(radar[i, 0]), int(radar[i, 1])
                    y0, y1 = max(0, ry - 1), min(128, ry + 2)
                    x0, x1 = max(0, rx - 1), min(128, rx + 2)
                    win = p_cloud[i, y0:y1, x0:x1]
                    v = win.max() if float(center[i]) == 1.0 else win.mean()
                    sum_bce = sum_bce + F.binary_cross_entropy(v.unsqueeze(0), cl[i].unsqueeze(0))
                bce = sum_bce / max(valid.sum().item(), 1)
            else:
                bce = torch.tensor(0.0, device=DEVICE)
            loss_a = loss_a + train_cfg.weight_center * bce
            loss_a.backward()
            nn.utils.clip_grad_norm_(model_a.parameters(), train_cfg.grad_clip)
            optim_a.step()
            sum_la += loss_a.item()

            # ── Train B ──
            optim_b.zero_grad()
            pb, hgt_b = model_b(x, moon_phase=moon, solar_zenith=sza)
            ce_full_b = F.cross_entropy(pb, y_corrected, reduction="none")
            loss_b = (ce_full_b * mask_b).sum() / max(mask_b.sum().item(), 1.0)
            if valid.any():
                probs = F.softmax(pb, dim=1)
                p_cloud = (probs[:, 2] + probs[:, 3]).clamp(1e-6, 1 - 1e-6)
                sum_bce = 0.0
                for i in range(p_cloud.size(0)):
                    if not bool(valid[i].item()): continue
                    ry, rx = int(radar[i, 0]), int(radar[i, 1])
                    y0, y1 = max(0, ry - 1), min(128, ry + 2)
                    x0, x1 = max(0, rx - 1), min(128, rx + 2)
                    win = p_cloud[i, y0:y1, x0:x1]
                    v = win.max() if float(center[i]) == 1.0 else win.mean()
                    sum_bce = sum_bce + F.binary_cross_entropy(v.unsqueeze(0), cl[i].unsqueeze(0))
                bce_b = sum_bce / max(valid.sum().item(), 1)
            else:
                bce_b = torch.tensor(0.0, device=DEVICE)
            loss_b = loss_b + train_cfg.weight_center * bce_b
            loss_b.backward()
            nn.utils.clip_grad_norm_(model_b.parameters(), train_cfg.grad_clip)
            optim_b.step()
            sum_lb += loss_b.item()
            n_batch += 1

        sched_a.step(epoch - 1)
        sched_b.step(epoch - 1)

        # ── Val ──
        model_a.eval(); model_b.eval()
        with torch.no_grad():
            v_correct = 0; v_total = 0
            v_tp = v_fp = v_tn = v_fn = 0
            for batch in val_loader:
                x = batch["image"].to(DEVICE)
                moon = batch["moon_phase"].to(DEVICE)
                sza  = batch["solar_zenith"].to(DEVICE)
                pa, _ = model_a(x, moon_phase=moon, solar_zenith=sza)
                pb, _ = model_b(x, moon_phase=moon, solar_zenith=sza)
                pred = (pa + pb) / 2
                pred_cls = pred.argmax(1)
                tp, fp, fn, tn = compute_binary_metrics(pred_cls, batch["mask_noisy"].to(DEVICE))
                v_tp += tp; v_fp += fp; v_fn += fn; v_tn += tn
                for i in range(x.size(0)):
                    cl = batch["center_label"][i].item()
                    if cl != cl: continue
                    ry, rx = int(batch["radar_loc"][i, 0]), int(batch["radar_loc"][i, 1])
                    y0, y1 = max(0, ry - 1), min(128, ry + 2)
                    x0, x1 = max(0, rx - 1), min(128, rx + 2)
                    win = pred_cls[i, y0:y1, x0:x1]
                    bin_win = (win >= 2).int()
                    pbin = 1 if bin_win.sum() > 0 else (0 if bin_win.float().mean() <= 0.5 else 1)
                    if pbin == int(cl): v_correct += 1
                    v_total += 1
            val_acc = v_correct / max(v_total, 1)
            v_p = v_tp / max(v_tp + v_fp, 1)
            v_r = v_tp / max(v_tp + v_fn, 1)
            v_f1 = 2 * v_p * v_r / max(v_p + v_r, 1e-9)

        avg_la = sum_la / max(n_batch, 1)
        avg_lb = sum_lb / max(n_batch, 1)
        dt = time.time() - t0
        line = (f"[{epoch:02d}] forget={rate:.2f} LA={avg_la:.4f} LB={avg_lb:.4f}  "
                f"Val Acc={val_acc:.3%} P={v_p:.1%} R={v_r:.1%} F1={v_f1:.1%}  {dt:.0f}s")
        print(line)
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_la:.6f},{avg_lb:.6f},0.0,{val_acc:.6f},{v_p:.6f},"
                    f"{v_r:.6f},{v_f1:.6f},{rate:.4f},{dt:.1f}\n")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model_a.state_dict(), os.path.join(CHECKPOINT_DIR, "physprior_coteaching_a_best.pth"))
            torch.save(model_b.state_dict(), os.path.join(CHECKPOINT_DIR, "physprior_coteaching_b_best.pth"))
            print(f"  → Best: Val Acc={val_acc:.3%}")

        if early_stop.step(val_acc):
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"\nBest Val: {best_val_acc:.3%} at epoch {best_epoch}")
    print(f"Total time: {time.time()-t_overall:.0f}s")

    # ── Test eval on the best ──
    print("\n>>> Test evaluation with best ensemble (avg of A+B logits)")
    test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False)
    model_a.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "physprior_coteaching_a_best.pth"),
                                        map_location=DEVICE, weights_only=True))
    model_b.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "physprior_coteaching_b_best.pth"),
                                        map_location=DEVICE, weights_only=True))
    model_a.eval(); model_b.eval()
    tp = fp = tn = fn = 0
    for i in range(len(test_ds)):
        s = test_ds[i]
        x = s["image"].unsqueeze(0).to(DEVICE)
        moon = torch.tensor([s["moon_phase"]], device=DEVICE)
        sza  = torch.tensor([s["solar_zenith"]], device=DEVICE)
        with torch.no_grad():
            pa, _ = model_a(x, moon_phase=moon, solar_zenith=sza)
            pb, _ = model_b(x, moon_phase=moon, solar_zenith=sza)
            pred = (pa + pb) / 2
            pred_cls = pred.argmax(1)[0].cpu().numpy()
        ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
        cl = s["center_label"]
        if cl != cl: continue
        y0, y1 = max(0, ry - 1), min(128, ry + 2)
        x0, x1 = max(0, rx - 1), min(128, rx + 2)
        win = pred_cls[y0:y1, x0:x1]
        bin_win = (win >= 2).astype(int)
        pbin = 1 if bin_win.sum() > 0 else (0 if bin_win.mean() <= 0.5 else 1)
        if   pbin == 1 and int(cl) == 1: tp += 1
        elif pbin == 1 and int(cl) == 0: fp += 1
        elif pbin == 0 and int(cl) == 0: tn += 1
        else: fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / max(total, 1)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    sp = tn / max(tn + fp, 1)
    print(f"Test Radar Acc: {acc:.3%}  n={total}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}  P={p:.1%} R={r:.1%} F1={f1:.1%} Sp={sp:.1%}")
    out = {"method": "physprior_coteaching", "best_epoch": best_epoch,
           "best_val_acc": best_val_acc, "test_acc": acc, "test_n": total,
           "tp": tp, "fp": fp, "tn": tn, "fn": fn,
           "precision": p, "recall": r, "f1": f1, "specificity": sp}
    out_path = r"E:/Claude code/project/noise-label-cloud/output/physprior_coteaching_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    train()
