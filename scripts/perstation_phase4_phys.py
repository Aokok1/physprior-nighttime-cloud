"""Per-station + seasonal breakdown for phase4_physprior_moderate (no-leakage).
Goal: produce the EXACT numbers needed to repair GRSL §III-C and §III-D.
"""
import sys, os, json
sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
import numpy as np
import torch
from data.dataset import CloudDataset
from models.mt_unet import MT_UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = r"E:/Claude code/project/noise-label-cloud/output/checkpoints/phase4_physprior_moderate_best.pth"

model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                use_film=True, use_evidential=False).to(DEVICE)
sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(sd, strict=False)
model.eval()

ds = CloudDataset(r"E:/Data/Unet_Dataset", "Test", augment=False, use_basemap=True)
records = []
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
    x0, x1 = max(0, rx - 1), min(128, ry + 2)
    x0 = max(0, rx - 1); x1 = min(128, rx + 2)
    win = pred_cls[y0:y1, x0:x1]
    bin_win = (win >= 2).astype(int)
    pbin = 1 if bin_win.sum() > 0 else (0 if bin_win.mean() <= 0.5 else 1)
    fname = s["filename"]
    station = "CS" if "Changsha" in fname else "LM" if "Longmen" in fname else "?"
    records.append({"fname": fname, "station": station, "pred": int(pbin), "truth": int(cl)})


def _bucket(rs, key_fn):
    buckets = {}
    for r in rs:
        k = key_fn(r)
        buckets.setdefault(k, {"tp":0,"fp":0,"tn":0,"fn":0,"names":[]})
        b = buckets[k]
        b["names"].append(r["fname"])
        if   r["pred"]==1 and r["truth"]==1: b["tp"]+=1
        elif r["pred"]==1 and r["truth"]==0: b["fp"]+=1
        elif r["pred"]==0 and r["truth"]==0: b["tn"]+=1
        else: b["fn"]+=1
    return buckets


def _season(fname):
    # 文件名 Tensor_Changsha_A2020134.1824.npz → 2020134 -> DOY 134 -> May (~May)
    # 真实月份:更精确用 1950-based date
    try:
        doy = int(fname.split("_A")[1].split(".")[0])
        year = doy // 1000
        ddd  = doy % 1000
        # 1970+ 任意非闰年的近似
        # 用真实日历
        from datetime import date, timedelta
        d = date(year, 1, 1) + timedelta(days=ddd - 1)
        m = d.month
    except Exception:
        return "Unknown"
    if m in (3, 4, 5): return "Spring"
    if m in (6, 7, 8): return "Summer"
    if m in (9, 10, 11): return "Autumn"
    return "Winter"


def _metrics(b):
    nn = b["tp"]+b["fp"]+b["tn"]+b["fn"]
    if nn == 0: return None
    acc = (b["tp"]+b["tn"])/nn*100
    p = b["tp"]/(b["tp"]+b["fp"])*100 if b["tp"]+b["fp"] else 0
    r = b["tp"]/(b["tp"]+b["fn"])*100 if b["tp"]+b["fn"] else 0
    sp = b["tn"]/(b["tn"]+b["fp"])*100 if b["tn"]+b["fp"] else 0
    f1 = 2*p*r/(p+r) if (p+r) else 0
    return {"n": nn, "tp": b["tp"], "fp": b["fp"], "tn": b["tn"], "fn": b["fn"],
            "acc": round(acc, 1), "p": round(p, 1), "r": round(r, 1),
            "f1": round(f1, 1), "sp": round(sp, 1)}


print("="*72)
print("Phase4_physprior_moderate (NO LEAKAGE) per-station & seasonal breakdown")
print("="*72)

# Per-station
print("\n--- PER-STATION ---")
buckets = _bucket(records, lambda r: r["station"])
out = {}
for st in ("CS", "LM"):
    if st in buckets:
        m = _metrics(buckets[st])
        out[st] = m
        print(f"  {st}: n={m['n']:>3}  TP={m['tp']:>2} FP={m['fp']:>2} TN={m['tn']:>2} FN={m['fn']:>2}  "
              f"Acc={m['acc']:5.1f}%  P={m['p']:5.1f}%  R={m['r']:5.1f}%  F1={m['f1']:5.1f}%  Sp={m['sp']:5.1f}%")

# Combined
all_b = {"tp":sum(buckets[s]["tp"] for s in buckets),
         "fp":sum(buckets[s]["fp"] for s in buckets),
         "tn":sum(buckets[s]["tn"] for s in buckets),
         "fn":sum(buckets[s]["fn"] for s in buckets)}
m_all = _metrics(all_b)
out["Combined"] = m_all
print(f"  COMBINED: n={m_all['n']:>3}  TP={m_all['tp']:>2} FP={m_all['fp']:>2} TN={m_all['tn']:>2} FN={m_all['fn']:>2}  "
      f"Acc={m_all['acc']:5.1f}%  P={m_all['p']:5.1f}%  R={m_all['r']:5.1f}%  F1={m_all['f1']:5.1f}%  Sp={m_all['sp']:5.1f}%")

# Seasonal (accurate calendar)
print("\n--- SEASONAL (calendar-accurate) ---")
sb = _bucket(records, lambda r: _season(r["fname"]))
out_s = {}
for sname in ("Spring", "Summer", "Autumn", "Winter"):
    if sname not in sb: continue
    m = _metrics(sb[sname])
    out_s[sname] = m
    print(f"  {sname:>6}: n={m['n']:>3}  TP={m['tp']:>2} FP={m['fp']:>2} TN={m['tn']:>2} FN={m['fn']:>2}  "
          f"Acc={m['acc']:5.1f}%  P={m['p']:5.1f}%  R={m['r']:5.1f}%  F1={m['f1']:5.1f}%  Sp={m['sp']:5.1f}%")

# Save
out_path = r"E:/Claude code/project/noise-label-cloud/output/perstation_phase4_phys.json"
with open(out_path, "w") as f:
    json.dump({"station": out, "season": out_s}, f, indent=2)
print(f"\nSaved → {out_path}")

# 关键差异提示
print("\n" + "="*72)
print("CRITICAL FOR PAPER REVISION (R2 critique on §III-C per-station numbers)")
print("="*72)
print("Paper §III-C text says:")
print('  "Longmen achieves 72.0% accuracy with high specificity (69.1%)"')
print('  "Changsha achieves 55.2% with moderate specificity (62.5%)"')
print("But Table III says:")
print('  "CS Acc=62.1, Sp=31.2"')
print('  "LM Acc=80.4, Sp=87.0"')
print("Phase4 (no-leakage) actual numbers:")
print(f"  CS: Acc={out['CS']['acc']}, Sp={out['CS']['sp']}")
print(f"  LM: Acc={out['LM']['acc']}, Sp={out['LM']['sp']}")
