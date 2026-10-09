"""Bootstrap 95% CI for PhysPrior+Co-Teaching long-run (74.6%)."""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet

DEVICE = "cuda"
CKPT_A = r"E:/Claude code/project/noise-label-cloud/output/checkpoints/physprior_coteaching_long_a_best.pth"
CKPT_B = r"E:/Claude code/project/noise-label-cloud/output/checkpoints/physprior_coteaching_long_b_best.pth"

model_a = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                  use_film=True, use_evidential=False).to(DEVICE)
model_a.load_state_dict(torch.load(CKPT_A, map_location=DEVICE, weights_only=True))
model_a.eval()
model_b = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                  use_film=True, use_evidential=False).to(DEVICE)
model_b.load_state_dict(torch.load(CKPT_B, map_location=DEVICE, weights_only=True))
model_b.eval()

ds = CloudDataset(r"E:/Data/Unet_Dataset", "Test", augment=False, use_basemap=True)
corrects = []
for i in range(len(ds)):
    s = ds[i]
    cl = s["center_label"]
    if cl != cl: continue
    x = s["image"].unsqueeze(0).to(DEVICE)
    moon = torch.tensor([s["moon_phase"]], device=DEVICE)
    sza  = torch.tensor([s["solar_zenith"]], device=DEVICE)
    with torch.no_grad():
        pa, _ = model_a(x, moon_phase=moon, solar_zenith=sza)
        pb, _ = model_b(x, moon_phase=moon, solar_zenith=sza)
        pred = (pa + pb) / 2
        pred_cls = pred.argmax(1)[0].cpu().numpy()
    ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
    y0, y1 = max(0, ry - 1), min(128, ry + 2)
    x0, x1 = max(0, rx - 1), min(128, rx + 2)
    win = pred_cls[y0:y1, x0:x1]
    bin_win = (win >= 2).astype(int)
    pbin = 1 if bin_win.sum() > 0 else (0 if bin_win.mean() <= 0.5 else 1)
    corrects.append(int(pbin == int(cl)))

corrects = np.array(corrects, dtype=float)
n = len(corrects)
print(f"n={n}, mean={corrects.mean()*100:.3f}%")

np.random.seed(42)
means = np.empty(2000)
for b in range(2000):
    idx = np.random.choice(n, n, replace=True)
    means[b] = corrects[idx].mean()

lo = float(np.percentile(means, 2.5))
hi = float(np.percentile(means, 97.5))
print(f"95% CI: [{lo*100:.1f}%, {hi*100:.1f}%]  (width = {(hi-lo)*100:.1f}pp)")

# Confusion matrix at the best ensemble
tp=fp=tn=fn=0
# need the raw preds again
preds_records = []
for i in range(len(ds)):
    s = ds[i]
    cl = s["center_label"]
    if cl != cl: continue
    x = s["image"].unsqueeze(0).to(DEVICE)
    moon = torch.tensor([s["moon_phase"]], device=DEVICE)
    sza  = torch.tensor([s["solar_zenith"]], device=DEVICE)
    with torch.no_grad():
        pa, _ = model_a(x, moon_phase=moon, solar_zenith=sza)
        pb, _ = model_b(x, moon_phase=moon, solar_zenith=sza)
        pred = (pa + pb) / 2
        pred_cls = pred.argmax(1)[0].cpu().numpy()
    ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
    y0, y1 = max(0, ry - 1), min(128, ry + 2)
    x0, x1 = max(0, rx - 1), min(128, rx + 2)
    win = pred_cls[y0:y1, x0:x1]
    bin_win = (win >= 2).astype(int)
    pbin = 1 if bin_win.sum() > 0 else (0 if bin_win.mean() <= 0.5 else 1)
    preds_records.append((int(pbin), int(cl)))

for pbin, cl in preds_records:
    if   pbin==1 and cl==1: tp+=1
    elif pbin==1 and cl==0: fp+=1
    elif pbin==0 and cl==0: tn+=1
    else: fn+=1
total = tp+fp+tn+fn
p = tp/max(tp+fp,1); r = tp/max(tp+fn,1); sp = tn/max(tn+fp,1)
f1 = 2*p*r/max(p+r,1e-9)
mcc = (tp*tn - fp*fn) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

print(f"\nTP={tp} FP={fp} TN={tn} FN={fn}")
print(f"P={p*100:.1f}% R={r*100:.1f}% F1={f1*100:.1f}% Sp={sp*100:.1f}%")
print(f"MCC={mcc:.3f}")

out = {"mean": float(corrects.mean()), "lo": lo, "hi": hi,
       "tp": tp, "fp": fp, "tn": tn, "fn": fn,
       "precision": float(p), "recall": float(r), "f1": float(f1),
       "specificity": float(sp), "mcc": float(mcc), "n": int(total)}
with open(r"E:/Claude code/project/noise-label-cloud/output/physprior_coteaching_long_ci.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved → output/physprior_coteaching_long_ci.json")
