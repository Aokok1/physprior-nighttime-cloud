"""Evaluate PhysPrior moderate model on TEST set with confusion matrix."""
import os, sys, json
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import config
config.MTUNET_DATASET = "E:/Data/Unet_Dataset"

from config import MTUNET_DATASET, CHECKPOINT_DIR
from data import make_dataloaders
from models.mt_unet import MT_UNet
from eval import _collect_radar_predictions, _bootstrap_ci
from scipy.stats import binomtest

# Load test data
_, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=16, use_phase3=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load PhysPrior moderate best checkpoint
ckpt_path = os.path.join(CHECKPOINT_DIR, "physprior_moderate_best.pth")
print(f"Loading checkpoint: {ckpt_path}")
model = MT_UNet(backbone="resnet34", in_channels=3, mask_classes=4,
                use_film=True, use_evidential=False).to(device)
state = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()

# Collect predictions
records = _collect_radar_predictions(model, val_loader, device)
print(f"\nTotal samples evaluated: {len(records)}")

# Per-station breakdown
for site in ["Changsha", "Longmen"]:
    sr = [r for r in records if site in r["filename"]]
    if not sr:
        continue
    tp = sum(1 for r in sr if r["radar_truth"]==1 and r["pred_binary"]==1)
    fn = sum(1 for r in sr if r["radar_truth"]==1 and r["pred_binary"]==0)
    fp = sum(1 for r in sr if r["radar_truth"]==0 and r["pred_binary"]==1)
    tn = sum(1 for r in sr if r["radar_truth"]==0 and r["pred_binary"]==0)
    total = tp + fn + fp + tn
    correct = tp + tn
    acc_ci = binomtest(correct, total).proportion_ci()
    precision = tp/(tp+fp) if (tp+fp) > 0 else 0
    recall = tp/(tp+fn) if (tp+fn) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    specificity = tn/(tn+fp) if (tn+fp) > 0 else 0
    
    print(f"\n=== PhysPrior Moderate - {site} (n={total}) ===")
    print(f"  Confusion Matrix:")
    print(f"    TP={tp}  FN={fn}")
    print(f"    FP={fp}  TN={tn}")
    print(f"  Accuracy:    {correct}/{total} = {correct/total:.1%} [{acc_ci[0]:.1%}, {acc_ci[1]:.1%}]")
    print(f"  Precision:   {precision:.1%}  (TP/(TP+FP))")
    print(f"  Recall:      {recall:.1%}  (TP/(TP+FN))  [Cloud sensitivity]")
    print(f"  Specificity: {specificity:.1%}  (TN/(TN+FP))  [Clear selectivity]")
    print(f"  F1:          {f1:.1%}")
    print(f"  FP rate:     {fp}/{tn+fp} = {fp/(tn+fp):.1%}  (clear misclassified as cloud)")
    print(f"  FN rate:     {fn}/{tp+fn} = {fn/(tp+fn):.1%}  (cloud misclassified as clear)")

# Combined
tp_all = sum(1 for r in records if r["radar_truth"]==1 and r["pred_binary"]==1)
fn_all = sum(1 for r in records if r["radar_truth"]==1 and r["pred_binary"]==0)
fp_all = sum(1 for r in records if r["radar_truth"]==0 and r["pred_binary"]==1)
tn_all = sum(1 for r in records if r["radar_truth"]==0 and r["pred_binary"]==0)
total_all = len(records)
correct_all = tp_all + tn_all

acc_ci_all = binomtest(correct_all, total_all).proportion_ci()
precision_all = tp_all/(tp_all+fp_all) if (tp_all+fp_all) > 0 else 0
recall_all = tp_all/(tp_all+fn_all) if (tp_all+fn_all) > 0 else 0
f1_all = 2*precision_all*recall_all/(precision_all+recall_all) if (precision_all+recall_all) > 0 else 0
specificity_all = tn_all/(tn_all+fp_all) if (tn_all+fp_all) > 0 else 0

print(f"\n{'='*60}")
print(f"=== PhysPrior Moderate - COMBINED (n={total_all}) ===")
print(f"{'='*60}")
print(f"  Confusion Matrix:")
print(f"    TP={tp_all}  FN={fn_all}")
print(f"    FP={fp_all}  TN={tn_all}")
print(f"  Accuracy:    {correct_all}/{total_all} = {correct_all/total_all:.1%} [{acc_ci_all[0]:.1%}, {acc_ci_all[1]:.1%}]")
print(f"  Precision:   {precision_all:.1%}")
print(f"  Recall:      {recall_all:.1%}  [Cloud sensitivity]")
print(f"  Specificity: {specificity_all:.1%}  [Clear selectivity]")
print(f"  F1:          {f1_all:.1%}")
print(f"  FP rate:     {fp_all}/{tn_all+fp_all} = {fp_all/(tn_all+fp_all):.1%}")
print(f"  FN rate:     {fn_all}/{tp_all+fn_all} = {fn_all/(tp_all+fn_all):.1%}")

# Bootstrap CI
boot = _bootstrap_ci(records)
print(f"\n  Bootstrap 95% CI: [{boot['ci_lower']:.1%}, {boot['ci_upper']:.1%}]")

# Compare with CLDMSK baseline (from eval_per_site.py)
print(f"\n{'='*60}")
print(f"COMPARISON: CLDMSK raw vs PhysPrior Moderate")
print(f"{'='*60}")

# CLDMSK stats from eval_per_site.py output
cldmsk_tp = 55; cldmsk_fn = 3; cldmsk_fp = 107; cldmsk_tn = 32
cldmsk_total = 197
print(f"                    {'CLDMSK':>10s}  {'PhysPrior':>10s}")
print(f"  TP:               {cldmsk_tp:>10d}  {tp_all:>10d}")
print(f"  FN:               {cldmsk_fn:>10d}  {fn_all:>10d}")
print(f"  FP:               {cldmsk_fp:>10d}  {fp_all:>10d}")
print(f"  TN:               {cldmsk_tn:>10d}  {tn_all:>10d}")
print(f"  Accuracy:         {cldmsk_tp+cldmsk_tn:>10d}/{cldmsk_total}  {correct_all}/{total_all}")
print(f"                    {(cldmsk_tp+cldmsk_tn)/cldmsk_total:>9.1%}  {correct_all/total_all:>9.1%}")
