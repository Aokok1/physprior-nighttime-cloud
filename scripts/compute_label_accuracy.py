"""Critical fix: compute PhysPrior pure label accuracy vs radar.

This addresses the reviewer's core complaint:
  "Paper confuses label correction accuracy with downstream model accuracy."
  
Computes:
  1. PhysPrior-corrected label (center pixel) directly vs radar → label_correction_accuracy
  2. Pure M15 threshold baseline vs radar → physics_baseline_accuracy
  3. Pure M15+std baseline vs radar
"""
import os, sys, json
import numpy as np

PROJECT_ROOT = "E:/Claude code/project/noise-label-cloud"
sys.path.insert(0, PROJECT_ROOT)

from methods.physical_prior import apply_physical_correction

TEST_DIR = "E:/Data/Unet_Dataset/Test"
files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".npz")])

def compute_accuracy(predictions, ground_truth):
    """Compute binary accuracy and confusion matrix."""
    valid = [(p,g) for p,g in zip(predictions, ground_truth) if not np.isnan(g)]
    if not valid: return None
    preds = np.array([v[0] for v in valid])
    labels = np.array([v[1] for v in valid])
    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(((preds==0)&(labels==0)).sum())  # wait, let me be more careful
    # Actually: pred=1 means cloud, label=1 means radar says cloud
    tp = int(((preds==1)&(labels==1)).sum())
    fp = int(((preds==1)&(labels==0)).sum())
    tn = int(((preds==0)&(labels==0)).sum())
    fn = int(((preds==0)&(labels==1)).sum())
    n = len(valid)
    acc = (tp+tn)/max(n,1)
    prec = tp/max(tp+fp,1)
    rec = tp/max(tp+fn,1)
    f1 = 2*prec*rec/max(prec+rec,1e-8) if (prec+rec)>0 else 0
    return {"n":n,"acc":acc,"tp":tp,"fp":fp,"tn":tn,"fn":fn,"prec":prec,"rec":rec,"f1":f1}

# ── 1. CLDMSK raw vs radar ──
cldmsk_preds = []
cldmsk_labels = []
for f in files:
    d = np.load(os.path.join(TEST_DIR, f))
    y_mask = int(d["Y_mask"].squeeze()[64,64])
    cl = float(d.get("Center_Label", float("nan")))
    cldmsk_preds.append(1 if y_mask>=2 else 0)
    cldmsk_labels.append(cl)

r_cldmsk = compute_accuracy(cldmsk_preds, cldmsk_labels)
print("=== CLDMSK raw vs radar ===")
print(f"  Acc={r_cldmsk['acc']:.1%}  TP={r_cldmsk['tp']} FP={r_cldmsk['fp']} TN={r_cldmsk['tn']} FN={r_cldmsk['fn']}")
print(f"  P={r_cldmsk['prec']:.1%}  R={r_cldmsk['rec']:.1%}  F1={r_cldmsk['f1']:.1%}")

# ── 2. PhysPrior-corrected labels vs radar (NO MODEL, pure correction) ──
for preset_name, (m15_min, std_max) in [
    ("moderate", (264.0, 1.5)),
    ("aggressive", (262.0, 2.0)),
    ("conservative", (266.0, 1.0)),
]:
    phys_preds = []
    phys_labels = []
    stats = {"total_corrected": 0, "total_pixels": 0}
    for f in files:
        d = np.load(os.path.join(TEST_DIR, f))
        cldmsk = d["Y_mask"].squeeze().astype(np.int32)
        m15 = d.get("X_m15", d.get("X_mod", np.ones_like(cldmsk)*np.nan)).astype(np.float64)
        cl = float(d.get("Center_Label", float("nan")))
        
        # Apply PhysPrior correction
        corrected, n_corr, _ = apply_physical_correction(
            cldmsk, m15, m15_min=m15_min, std_max=std_max
        )
        stats["total_corrected"] += n_corr
        stats["total_pixels"] += cldmsk.size
        
        # Center pixel: what does corrected label say?
        center_corrected = corrected[64, 64]
        phys_preds.append(1 if center_corrected>=2 else 0)  # still using same threshold
        phys_labels.append(cl)
    
    r = compute_accuracy(phys_preds, phys_labels)
    print(f"\n=== PhysPrior ({preset_name}) LABEL CORRECTION vs radar (no model) ===")
    print(f"  Acc={r['acc']:.1%}  TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")
    print(f"  P={r['prec']:.1%}  R={r['rec']:.1%}  F1={r['f1']:.1%}")
    print(f"  Pixels corrected: {stats['total_corrected']}/{stats['total_pixels']} ({stats['total_corrected']/stats['total_pixels']*100:.1f}%)")

# ── 3. Pure M15 threshold baseline vs radar (no model, no texture) ──
for t in [260, 262, 264, 266, 268, 270, 272]:
    m15_preds = []
    m15_labels = []
    for f in files:
        d = np.load(os.path.join(TEST_DIR, f))
        m15 = d.get("X_m15", d.get("X_mod", np.ones(1)*np.nan)).astype(np.float64)
        cl = float(d.get("Center_Label", float("nan")))
        center_m15 = m15[64, 64] if len(m15.shape)>=2 else m15
        m15_preds.append(0 if center_m15 > t else 1)  # cold = cloud
        m15_labels.append(cl)
    
    r = compute_accuracy(m15_preds, m15_labels)
    print(f"\n=== Pure M15 threshold T={t}K vs radar (no texture, no model) ===")
    print(f"  Acc={r['acc']:.1%}  TP={r['tp']} FP={r['fp']} TN={r['tn']} FN={r['fn']}")

# ── 4. Summary ──
print("\n" + "=" * 70)
print("SUMMARY FOR PAPER")
print("=" * 70)
print(f"  CLDMSK raw:             {r_cldmsk['acc']:.1%}")
# Use moderate preset for final
for t in [264]:
    r = compute_accuracy([
        0 if np.load(os.path.join(TEST_DIR,f)).get("X_m15",np.ones(1))[64,64]>t else 1
        for f in files
    ], [
        float(np.load(os.path.join(TEST_DIR,f)).get("Center_Label",float("nan")))
        for f in files
    ])
    print(f"  Pure M15 T={t}K:         {r['acc']:.1%}")
# Need to report the PhysPrior label correction accuracy
