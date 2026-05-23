import numpy as np, os, glob

TEST_DIR = r"E:\Data\Unet_Dataset\Test"

for site in ["Changsha", "Longmen"]:
    tp = fn = fp = tn = 0
    for fpath in sorted(glob.glob(os.path.join(TEST_DIR, "*.npz"))):
        if site not in os.path.basename(fpath):
            continue
        d = np.load(fpath, allow_pickle=True)
        rl = float(d["Center_Label"])
        if np.isnan(rl):
            continue
        cldmsk_cloud = d["Y_mask"][64, 64] >= 2
        radar_cloud = rl == 1
        if radar_cloud and cldmsk_cloud: tp += 1
        elif radar_cloud and not cldmsk_cloud: fn += 1
        elif not radar_cloud and cldmsk_cloud: fp += 1
        else: tn += 1
    
    total = tp + fn + fp + tn
    print(f"=== {site} (n={total}) ===")
    print(f"  Clear accuracy: {tn}/{tn+fp} = {tn/(tn+fp):.1%}  (CLDMSK correct when actually clear)")
    print(f"  Cloud accuracy: {tp}/{tp+fn} = {tp/(tp+fn):.1%}  (CLDMSK correct when actually cloudy)")
    print(f"  Overall:        ({tp+tn})/{total} = {(tp+tn)/total:.1%}")
    print(f"  FP rate:        {fp}/{tn+fp} = {fp/(tn+fp):.1%}  (clear misclassified as cloud)")
    print(f"  FN rate:        {fn}/{tp+fn} = {fn/(tp+fn):.1%}  (cloud misclassified as clear)")
    print()

# Combined
tp_all = fn_all = fp_all = tn_all = 0
for fpath in sorted(glob.glob(os.path.join(TEST_DIR, "*.npz"))):
    d = np.load(fpath, allow_pickle=True)
    rl = float(d["Center_Label"])
    if np.isnan(rl): continue
    cldmsk_cloud = d["Y_mask"][64, 64] >= 2
    radar_cloud = rl == 1
    if radar_cloud and cldmsk_cloud: tp_all += 1
    elif radar_cloud and not cldmsk_cloud: fn_all += 1
    elif not radar_cloud and cldmsk_cloud: fp_all += 1
    else: tn_all += 1

total_all = tp_all + fn_all + fp_all + tn_all
print(f"=== COMBINED (n={total_all}) ===")
print(f"  Clear accuracy: {tn_all}/{tn_all+fp_all} = {tn_all/(tn_all+fp_all):.1%}")
print(f"  Cloud accuracy: {tp_all}/{tp_all+fn_all} = {tp_all/(tp_all+fn_all):.1%}")
print(f"  Overall:        ({tp_all+tn_all})/{total_all} = {(tp_all+tn_all)/total_all:.1%}")
