"""Re-estimate the Loss Correction transition matrix from train+val radar samples (NOT test).

The radar-collocated samples of the active training and validation splits are
independent of the 197-sample radar-reference test set. On the default split
these number 167; on the group-disjoint split (NL_SPLIT_TAG=grp) they number 160.

The freshly estimated matrix is lifted to 4x4 and used for training in the same
run, so a re-run can never train with a stale matrix.
"""
import sys, os, json, numpy as np
sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
from determinism import seed_everything
seed_everything(int(os.environ.get("NL_SEED", "42")))  # fixed seed: the pipeline must be bit-reproducible

from config import MTUNET_DATASET, train_cfg, CHECKPOINT_DIR, LOG_DIR
from data import make_dataloaders
from data.dataset import CloudDataset
import torch

print("=== Loss Correction: re-estimate transition matrix from train+val radar ===")

# Load train_phase4 and val_phase4 datasets to extract radar-labeled samples
# Need to read .npz files directly since dataloaders don't carry file-level radar info
import pandas as pd
_mf = ("sample_manifest_grp.csv" if os.environ.get("NL_SPLIT_TAG") == "grp"
       else "sample_manifest.csv")
df = pd.read_csv(os.path.join(r"E:/Claude code/project/noise-label-cloud/output/manifest", _mf))
_tag = os.environ.get('NL_SPLIT_TAG', 'phase4')
_tr, _va = f'train_{_tag}', f'val_{_tag}'
_suf = f'_{_tag}' if _tag != 'phase4' else ''
radar_non_test = df[(df.has_radar == True) & (df.split.isin([_tr, _va]))]
print(f"Radar-labeled non-test samples: {len(radar_non_test)}")
print(f"  Split counts: {radar_non_test.split.value_counts().to_dict()}")

# For each, load the .npz, extract Y_mask (CLDMSK 4-class) and Center_Label (radar binary)
# Map Y_mask to binary: class in {2,3} = cloud
binary_mapping = lambda y: ((y == 2) | (y == 3)).astype(int)

cl_4class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
radar_4class_counts = {}  # will be keyed by CLDMSK class -> radar counts

# Accumulate confusion
conf = np.zeros((4, 2), dtype=int)  # rows=CLDMSK class (0-3), cols=radar label (0=clear,1=cloud)

for idx, row in radar_non_test.iterrows():
    # Find the .npz file
    fname = row['fname']
    # Search in Train_phase4 or Val_phase4
    found = False
    for sub in [f'Train_{_tag}', f'Val_{_tag}', 'Train_phase4', 'Val_phase4']:
        fpath = os.path.join(MTUNET_DATASET, sub, fname)
        if os.path.exists(fpath):
            data = np.load(fpath, allow_pickle=True)
            cl_4class = np.squeeze(data['Y_mask']).astype(np.int32)
            radar_label = float(data['Center_Label'])
            found = True
            break
    if not found:
        # Try Test directory
        fpath = os.path.join(MTUNET_DATASET, 'Test', fname)
        if os.path.exists(fpath):
            data = np.load(fpath, allow_pickle=True)
            cl_4class = np.squeeze(data['Y_mask']).astype(np.int32)
            radar_label = float(data['Center_Label'])
            found = True
    if not found:
        print(f"  WARN: file not found {fname}")
        continue
    if np.isnan(radar_label):
        continue
    # Get center pixel CLDMSK class
    center_y = cl_4class[64, 64]  # 128x128 center
    center_cl = int(center_y)
    if center_cl not in [0, 1, 2, 3]:
        continue
    radar_bin = int(radar_label)  # 0=clear, 1=cloud
    conf[center_cl, radar_bin] += 1
    cl_4class_counts[center_cl] += 1

print(f"CLDMSK_RADAR CONFUSION")
print(f"{'CLDMSK\\Radar':>12s} {'Clear':>8s} {'Cloud':>8s}")
for cl in range(4):
    print(f"  Class {cl}:           {conf[cl,0]:6d}  {conf[cl,1]:6d}")

# Normalize: T[i, j] = P(Radar=j | CLDMSK=i)
T = conf.astype(float)
for i in range(4):
    if conf[i].sum() > 0:
        T[i] = conf[i] / conf[i].sum()
print(f"Transition matrix P(Radar|CLDMSK):")
print(f"{'CLDMSK\\Radar':>12s} {'Clear':>8s} {'Cloud':>8s}")
for cl in range(4):
    print(f"  Class {cl}:           {T[cl,0]:8.4f}  {T[cl,1]:8.4f}")

out_path = rf"E:/Claude code/project/noise-label-cloud/output/noise_matrix_from_train_val{_suf}.json"
with open(out_path, 'w') as f:
    json.dump({"n_samples": int(conf.sum()), "T": T.tolist(), "conf_matrix": conf.tolist()}, f, indent=2)
print(f"\nSaved → {out_path}")

# Lift the freshly estimated 4x2 matrix into the 4x4 shape the corrected loss
# expects: columns 1 and 2 are structural zeros because the radar reference is
# binary, column 0 carries P(radar = clear) and column 3 carries P(radar = cloud).
# Previously this step loaded a pre-existing noise_matrix_4x4.json, which meant a
# re-run estimated a new matrix and then trained with the old one.
T4 = np.zeros((4, 4), dtype=float)
T4[:, 0] = T[:, 0]
T4[:, 3] = T[:, 1]
T = T4
print("Lifted 4x4 transition matrix P(Radar|CLDMSK):")
print(f"{'CLDMSK\\Radar':>12s} {'clear':>8s} {'prob_c':>8s} {'prob_k':>8s} {'cloud':>8s}")
for cl in range(4):
    print(f"  Class {cl}:           {T[cl,0]:8.4f}  {T[cl,1]:8.4f}  {T[cl,2]:8.4f}  {T[cl,3]:8.4f}")
_lift_path = rf"E:/Claude code/project/noise-label-cloud/output/noise_matrix_4x4{_suf}.json"
with open(_lift_path, 'w') as f:
    json.dump({"T": T.tolist(), "n": int(conf.sum())}, f, indent=2)
print(f"Saved → {_lift_path}")

train_loader, val_loader = make_dataloaders(
    MTUNET_DATASET, batch_size=train_cfg.batch_size,
    use_basemap=True, use_phase4=True)

from methods.loss_correction import LossCorrectionTrainer
trainer = LossCorrectionTrainer(
    train_loader, val_loader,
    transition_matrix=T,
    device=torch.device("cuda")
)
model, best_val = trainer.run()

# Test on test set
model.eval()
test_ds = CloudDataset(MTUNET_DATASET, "Test", augment=False, use_basemap=True)
records = []
with torch.no_grad():
    for i in range(len(test_ds)):
        s = test_ds[i]
        cl = s["center_label"]
        if cl != cl: continue
        x = s["image"].unsqueeze(0).cuda()
        moon = torch.tensor([s["moon_phase"]], device="cuda")
        sza = torch.tensor([s["solar_zenith"]], device="cuda")
        logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = logits.argmax(1)[0].cpu().numpy()
        ry, rx = int(s["radar_loc"][0]), int(s["radar_loc"][1])
        y0, y1 = max(0, ry - 1), min(128, ry + 2)
        x0, x1 = max(0, rx - 1), min(128, rx + 2)
        win = pred_cls[y0:y1, x0:x1]
        bw = (win >= 2).astype(int)
        pbin = 1 if bw.sum() > 0 else (0 if bw.mean() <= 0.5 else 1)
        truth = int(cl)
        records.append({"fname": s["filename"], "pred": int(pbin), "truth": truth})

arr = np.array([(r["pred"], r["truth"]) for r in records])
tp = int(((arr[:,0]==1)&(arr[:,1]==1)).sum())
fp = int(((arr[:,0]==1)&(arr[:,1]==0)).sum())
tn = int(((arr[:,0]==0)&(arr[:,1]==0)).sum())
fn = int(((arr[:,0]==0)&(arr[:,1]==1)).sum())
n = tp+fp+tn+fn
acc = (tp+tn)/max(n,1)
print(f"\n=== Test result (n={n}) ===")
print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
print(f"  Acc={acc*100:.2f}% (was 61.9% with old matrix)")

result = {
    "method": "Loss Correction (re-estimated)",
    "n": n, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    "acc": float(acc),
    "transition_matrix_source": f"{_tr} + {_va} radar samples (n={int(conf.sum())})",
    "old_acc": 0.619
}
with open(rf"E:/Claude code/project/noise-label-cloud/output/loss_correction_redo{_suf}.json", 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved → {out_path}")
