"""论文说:'训练数据上 PhysPrior 校正后的中心像素准确率 = 69.5% = 模型预测准
确率(DL 没有超过校正标签)'(Conslusion 段落)。
现在对 Test 上做同样验证:
  1. 对每个 Test 样本,对 CLDMSK 应用 moderate PhysPrior (T=264K, σ=1.5K)
  2. 计算:校正后 CLDMSK 的雷达准确率 = ?
  3. 与 phase4_physprior_moderate_best.pth 在 radar_loc 像素预测的准确率 = 69.5% 进行比较
"""
import os, sys, glob
import numpy as np
import torch
from scipy.ndimage import uniform_filter

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet
from methods.physical_prior import apply_physical_correction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MTUNET_DATASET = r"E:/Data/Unet_Dataset"
CKPT = r"E:/Claude code/project/noise-label-cloud/output/checkpoints/phase4_physprior_moderate_best.pth"

# ── 1. apply PhysPrior to test, count radar accuracy at center pixel ──
files = sorted(glob.glob(os.path.join(MTUNET_DATASET, "Test", "*.npz")))
correct_label = 0
total_label = 0
for fp in files:
    d = np.load(fp)
    if d["Center_Label"] != d["Center_Label"]: continue
    cldmsk = d["Y_mask"].astype(np.int32)
    m15 = d["X_m15"]
    corrected, _, _ = apply_physical_correction(
        cldmsk, m15, m15_min=264.0, std_max=1.5,
    )
    ry, rx = int(d["Radar_Loc"][0]), int(d["Radar_Loc"][1])
    pred = 1 if corrected[ry, rx] in (2, 3) else 0
    truth = int(d["Center_Label"])
    if pred == truth: correct_label += 1
    total_label += 1
acc_label = correct_label / max(total_label, 1)
print(f"[1] PhysPrior 校正后标签的 Radar 中心像素准确率 = {acc_label:.3%}  (n={total_label})")
print(f"    论文报: 69.5%   →  偏差 = {(acc_label-0.695)*100:+.1f}pp\n")

# ── 2. phase4 ckpt predictions at center pixel (3×3 window majority) ──
model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                use_film=True, use_evidential=False).to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(sd, strict=False)
model.eval()

ds = CloudDataset(MTUNET_DATASET, "Test", augment=False, use_basemap=True)
correct_model = 0
total_model = 0
tp = fp = tn = fn = 0
for i in range(len(ds)):
    s = ds[i]
    cl = s["center_label"]
    if cl != cl: continue
    x = s["image"].unsqueeze(0).to(DEVICE)
    moon = torch.tensor([s["moon_phase"]], device=DEVICE)
    sza  = torch.tensor([s["solar_zenith"]], device=DEVICE)
    with torch.no_grad():
        logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = logits.argmax(1)[0].cpu().numpy()
    ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
    y0, y1 = max(0, ry - 1), min(128, ry + 2)
    x0, x1 = max(0, rx - 1), min(128, rx + 2)
    win = pred_cls[y0:y1, x0:x1]
    bin_win = (win >= 2).astype(int)
    pbin = 1 if bin_win.sum() > 0 else (0 if bin_win.mean() <= 0.5 else 1)
    if pbin == int(cl): correct_model += 1
    total_model += 1
    if   pbin == 1 and int(cl) == 1: tp += 1
    elif pbin == 1 and int(cl) == 0: fp += 1
    elif pbin == 0 and int(cl) == 0: tn += 1
    else: fn += 1
acc_model = correct_model / max(total_model, 1)
print(f"[2] phase4_physprior_moderate 模型预测的 Radar 中心像素准确率 = {acc_model:.3%}  (n={total_model})")
print(f"    论文报: 69.5%   →  偏差 = {(acc_model-0.695)*100:+.1f}pp")
print(f"    TP={tp} FP={fp} TN={tn} FN={fn}")
p = tp / max(tp+fp,1); r = tp / max(tp+fn,1); f1 = 2*p*r/max(p+r,1e-9)
sp = tn / max(tn+fp,1)
print(f"    P={p:.1%} R={r:.1%} F1={f1:.1%} Sp={sp:.1%}")

# ── 3. delta: model - label ──
print(f"\n[3] Δ = 模型 - 校正后标签 = {(acc_model - acc_label)*100:+.2f}pp")
print(f"    论文说:'下游模型准确率(69.5%) = 校正后标签中心像素准确率(69.5%) —— DL 没有超过校正'")
print(f"    实测:{'一致 ✓' if abs(acc_model - acc_label) < 0.005 else '有差异! 注意'}")
