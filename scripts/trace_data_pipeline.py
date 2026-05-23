"""Trace data pipeline: why is our dataset so small?"""
import numpy as np
import os, glob
from collections import Counter

TRAIN_DIR = "E:/Data/Unet_Dataset/Train"
TEST_DIR = "E:/Data/Unet_Dataset/Test"

print("=" * 60)
print(" 数据源追溯: 为什么数据这么少?")
print("=" * 60)

# 1. Count samples per split
for split, ddir in [("Train", TRAIN_DIR), ("Test", TEST_DIR)]:
    files = sorted(glob.glob(os.path.join(ddir, "*.npz")))
    has_radar = 0
    no_radar = 0
    stations = Counter()
    years = Counter()
    seasons = Counter()
    
    for f in files:
        d = np.load(f, allow_pickle=True)
        rl = float(d["Center_Label"])
        basename = os.path.basename(f)
        
        if "Changsha" in basename:
            stations["Changsha"] += 1
        elif "Longmen" in basename:
            stations["Longmen"] += 1
        
        # Extract year from filename: A2020286 -> 2020
        if "A2" in basename:
            year = int(basename.split("A")[1][:4])
            julian = int(basename.split("A")[1][4:7])
            years[year] += 1
            month = (julian - 1) // 30 + 1
            if month > 12: month = 12
            if month in [12, 1, 2]: seasons["Winter"] += 1
            elif month in [3, 4, 5]: seasons["Spring"] += 1
            elif month in [6, 7, 8]: seasons["Summer"] += 1
            else: seasons["Autumn"] += 1
        
        if np.isnan(rl):
            no_radar += 1
        else:
            has_radar += 1
    
    print(f"\n--- {split} ({len(files)} samples) ---")
    print(f"  Radar coverage: {has_radar} ({has_radar/len(files)*100:.1f}%)")
    print(f"  No radar:       {no_radar} ({no_radar/len(files)*100:.1f}%)")
    print(f"  Stations:       {dict(stations)}")
    print(f"  Years:          {dict(sorted(years.items()))}")
    print(f"  Seasons:        {dict(seasons)}")

# 2. Key insight
print("\n" + "=" * 60)
print(" 关键发现")
print("=" * 60)
print("""
数据管道瓶颈分析:

1. Ka波段雷达仅在特定年份运行:
   - 龙门: 2020年 (约1年)
   - 长沙: 2022年 (约1年)
   → 雷达数据极度稀缺

2. 训练集(2585样本)中:
   - 仅196个(7.5%)有雷达标签
   - 其余2389个(92.5%)只有CLDMSK标签(含噪)
   → 大部分训练样本没有干净标签

3. 测试集(197样本):
   - 100%有雷达覆盖(这是选择标准)
   - 龙门168 + 长沙29(冬季补充)
   → 测试集小是因为雷达覆盖有限

4. 数据漏斗:
   原始VIIRS过境 → 空间匹配(128x128 patch) → 
   时间匹配(雷达运行期) → 质量筛选 → 最终数据集
   
5. 为什么不多站点?
   - Ka波段雷达站点稀少(全球<100个)
   - 中国境内已知: 长沙、龙门、北京、合肥等
   - 需要与VIIRS过境时空共定位
   → 每增加一个站点需要新的数据获取和处理
""")

# 3. Check if there's more source data somewhere
print("=" * 60)
print(" 潜在扩展数据源")
print("=" * 60)
# Check other possible data locations
for path in [
    "E:/Data/",
    "H:/code/Data/",
    "H:/code/server_package/data/",
]:
    if os.path.exists(path):
        items = os.listdir(path)
        print(f"\n{path}: {len(items)} items")
        for item in items[:10]:
            full = os.path.join(path, item)
            if os.path.isdir(full):
                sub_items = os.listdir(full)
                print(f"  📁 {item}/ ({len(sub_items)} files)")
            else:
                print(f"  📄 {item}")
    else:
        print(f"\n{path}: NOT ACCESSIBLE")
