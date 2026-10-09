import sys, pandas as pd, numpy as np
sys.path.insert(0, r"E:/Claude code/project/noise-label-cloud")
from datetime import date, timedelta

df = pd.read_csv(r"E:/Claude code/project/noise-label-cloud/output/manifest/sample_manifest.csv")

print("=== Per-directory counts (any .npz that lives there) ===")
print("Train dir:", df.in_sub_train.sum())
print("Train_no_overlap:", df.in_sub_train_no_overlap.sum())
print("Train_phase4:", df.in_sub_train_phase4.sum())
print("Val_phase4:", df.in_sub_val_phase4.sum())
print("Test dir:", df.split.eq("test").sum())
print()

print("=== By station across full universe ===")
print(df.groupby("station").size())
print()

# Per row, in_sub_test_backup should be True if Test_backup
print("By per subdirectories by station:")
for col in ["in_sub_train", "in_sub_train_no_overlap", "in_sub_train_phase4", "in_sub_val_phase4"]:
    print(f"\n{col}:")
    print(df[df[col]].groupby("station").size())

print("\n=== Test split per station ===")
print(df[df.split == "test"].groupby("station").size())

def season_of(doy):
    if pd.isna(doy): return "Unknown"
    year = int(doy) // 1000
    ddd = int(doy) % 1000
    try:
        d = date(year, 1, 1) + timedelta(days=ddd - 1)
        m = d.month
    except: return "Unknown"
    if m in (3, 4, 5): return "Spring"
    if m in (6, 7, 8): return "Summer"
    if m in (9, 10, 11): return "Autumn"
    return "Winter"

print("\n=== Test split per station per season ===")
test_df = df[df.split == "test"].copy()
test_df["season"] = test_df["doy_year"].apply(season_of)
print(test_df.groupby(["station", "season"]).size())
print(f"\nTotal test rows: {len(test_df)}")
print(f"  CS: {(test_df.station == 'CS').sum()}")
print(f"  LM: {(test_df.station == 'LM').sum()}")
print()

# What is "901" + "1713"? Search closest
print("Most plausible interpretation of the paper text 'Changsha 901, Longmen 1713':")
print(f"  CS in Train_no_overlap: {(df.in_sub_train_no_overlap & (df.station=='CS')).sum()}")
print(f"  LM in Train_no_overlap: {(df.in_sub_train_no_overlap & (df.station=='LM')).sum()}")
print(f"  Sum: {(df.in_sub_train_no_overlap).sum()}")
print()

# Look at the original Train/ dir (not overlapping)
print("All CS .npz files:")
cs_files = df[df.station == "CS"]["fname"].unique()
print(f"  Total CS files (across all subdirs union): {len(cs_files)}")
print(f"  CS in Train dir: {(df.in_sub_train & (df.station=='CS')).sum()}")
print(f"  CS in Train_no_overlap (sub): {(df.in_sub_train_no_overlap & (df.station=='CS')).sum()}")
print(f"  CS in Train_phase4: {(df.in_sub_train_phase4 & (df.station=='CS')).sum()}")
print(f"  CS in Val_phase4: {(df.in_sub_val_phase4 & (df.station=='CS')).sum()}")
print(f"  CS in Test: {((df.split=='test') & (df.station=='CS')).sum()}")
print()
print("All LM .npz files:")
lm_files = df[df.station == "LM"]["fname"].unique()
print(f"  Total LM files (across all subdirs union): {len(lm_files)}")
print(f"  LM in Train dir: {(df.in_sub_train & (df.station=='LM')).sum()}")
print(f"  LM in Train_no_overlap: {(df.in_sub_train_no_overlap & (df.station=='LM')).sum()}")
print(f"  LM in Train_phase4: {(df.in_sub_train_phase4 & (df.station=='LM')).sum()}")
print(f"  LM in Val_phase4: {(df.in_sub_val_phase4 & (df.station=='LM')).sum()}")
print(f"  LM in Test: {((df.split=='test') & (df.station=='LM')).sum()}")
print()
print(f"Closure: 1977 + 495 + 197 = {1977 + 495 + 197} ≈ 2,669")
