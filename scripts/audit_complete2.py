"""Complete data audit — fixed season extraction."""
import os, sys, math
import numpy as np
import torch

PROJECT_ROOT = "E:/Claude code/project/noise-label-cloud"
sys.path.insert(0, PROJECT_ROOT)

import config
config.MTUNET_DATASET = "E:/Data/Unet_Dataset"
from config import MTUNET_DATASET, CHECKPOINT_DIR
from models.mt_unet import MT_UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = os.path.join(CHECKPOINT_DIR, "phase4_physprior_moderate_best.pth")
model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                use_film=True, use_evidential=False).to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()

test_dir = os.path.join(MTUNET_DATASET, "Test")
files = sorted([f for f in os.listdir(test_dir) if f.endswith(".npz")])

records = []
for fname in files:
    site = "Changsha" if "Changsha" in fname else "Longmen"
    d = np.load(os.path.join(test_dir, fname))
    
    dnb = d["X_dnb"].astype(np.float32)
    bm = d["X_basemap"].astype(np.float32)
    m15 = d.get("X_m15", d.get("X_mod", np.zeros_like(dnb))).astype(np.float32)
    
    dnb_norm = np.log10(np.clip(dnb, 1e-10, None))
    dnb_norm = (dnb_norm - (-8.85)) / 0.54
    bm_norm = np.log10(np.clip(bm, 1e-10, None))
    bm_norm = (bm_norm - (-8.85)) / 0.54
    m15_norm = (m15 - 274.0) / 16.7
    
    image = np.stack([dnb_norm, bm_norm, m15_norm], axis=0)
    image_t = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    
    cl = float(d.get("Center_Label", float("nan")))
    moon = float(d.get("Moon_Phase", 180.0))
    sza = float(d.get("Solar_Zenith", 100.0))
    
    # CORRECTED: extract year and julian day from "Tensor_Site_A{YYYY}{DDD}.HHMM.npz"
    tag = fname.split("_A")[1]  # "2020286.1836.npz"
    year = int(tag[:4])
    julian = int(tag[4:7])
    month = min((julian - 1) // 30 + 1, 12)
    if month in [12, 1, 2]: season = "winter"
    elif month in [3, 4, 5]: season = "spring"
    elif month in [6, 7, 8]: season = "summer"
    else: season = "autumn"
    
    with torch.no_grad():
        pred_mask, _ = model(image_t,
                           moon_phase=torch.tensor([moon], dtype=torch.float32).to(DEVICE),
                           solar_zenith=torch.tensor([sza], dtype=torch.float32).to(DEVICE))
        pred_cls = torch.argmax(pred_mask, dim=1)[0, 64, 64].item()
    
    pred_binary = 1 if pred_cls >= 2 else 0
    cldmsk_raw = int(d["Y_mask"].squeeze()[64, 64])
    cldmsk_cloud = 1 if cldmsk_raw >= 2 else 0
    
    records.append({
        "filename": fname.replace("Tensor_","").replace(".npz",""),
        "site": site, "season": season,
        "radar": int(cl) if not np.isnan(cl) else None,
        "pred": pred_binary, "cldmsk": cldmsk_cloud,
    })

def compute(recs, col="pred"):
    valid = [r for r in recs if r["radar"] is not None]
    if not valid: return "n=0"
    preds = np.array([r[col] for r in valid])
    labels = np.array([r["radar"] for r in valid])
    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    tn = int(((preds==0)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    n = len(valid)
    acc = (tp+tn)/max(n,1)
    prec = tp/max(tp+fp,1)
    rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-8)
    spec = tn/max(tn+fp,1)
    bal = (rec+spec)/2
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc = (tp*tn-fp*fn)/max(denom,1e-8) if denom > 0 else 0
    from scipy.stats import binomtest
    ci = binomtest(tp+tn,n).proportion_ci()
    return f"Acc={acc:.1%} [{ci[0]:.1%},{ci[1]:.1%}] TP={tp}FP={fp}TN={tn}FN={fn} P={prec:.1%}R={rec:.1%}F1={f1:.1%}"

print("="*70)
print("COMPLETE DATA AUDIT")
print("="*70)

for title, col in [("CLDMSK RAW", "cldmsk"), ("PHYSPRIOR Phase4", "pred")]:
    print(f"\n═══ {title} ═══")
    for g in ["Overall","Changsha","Longmen","spring","summer","autumn","winter"]:
        if g in ["Changsha","Longmen"]:
            r = [r for r in records if r["site"]==g]
        elif g == "Overall":
            r = records
        else:
            r = [r for r in records if r["season"]==g]
        print(f"  {g:10s}: {compute(r,col)}")

# Paper summary
print("\n" + "="*70)
print("PAPER NUMBERS vs AUDIT")
print("="*70)
c = compute(records, "cldmsk")
p = compute(records, "pred")
print(f"  CLDMSK:   44.2% (paper) vs {c.split()[0]} (audit)")
print(f"  PhysPrior: 69.5% (paper) vs {p.split()[0]} (audit)")
