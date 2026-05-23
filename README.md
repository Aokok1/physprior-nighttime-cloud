# Noise-Label Cloud Detection

从噪声 CLDMSK 标签中学习夜间云检测——将 CLDMSK 从"坏基线"（42% 准确率）转变为"全球弱监督资源"。

## 核心思路

CLDMSK 在雷达验证点上的准确率仅 42%，但它是**全球覆盖**的——地球上每个 VIIRS 像素都有 CLDMSK 标签。本项目的核心假设：**用 300 个干净的雷达标注样本，可以"净化"2089 个带噪 CLDMSK 标注**。

## 三种方法

| 方法 | 原理 | 关键参数 |
|------|------|---------|
| **Loss Correction** | 用雷达估计 CLDMSK 混淆矩阵，修正 CE 损失 | 混淆矩阵 (4×4) |
| **Co-Teaching** | 双模型互相筛选"干净"样本，仅用对方认可的数据更新 | forget_rate=0.2 |
| **SELFIE** | 教师(CLDMSK+雷达) → 净化标签 → 学生(纯净化标签) | confidence_thresh=0.85 |

## 项目结构

```
noise-label-cloud/
├── config.py           # 全局配置（路径、超参）
├── train.py            # 主入口（--method, --all）
├── eval.py             # 评估 + 方法对比
├── data/
│   ├── dataset.py      # Dataset 类（适配 MT-UNet .npz）
│   └── noise_analysis.py  # CLDMSK 噪声模式分析
├── models/
│   └── mt_unet.py      # MT-UNet + CorrectedLoss
├── methods/
│   ├── baseline.py     # 标准训练（对照）
│   ├── loss_correction.py
│   ├── coteaching.py
│   └── selfie.py       # 教师-学生净化框架
├── output/             # checkpoint + log + 报告
└── tests/
```

## 快速开始

```bash
# 1. 分析 CLDMSK 噪声模式（必须先做）
python train.py --analyze

# 2. 运行单个方法
python train.py --method baseline
python train.py --method loss_correction
python train.py --method coteaching
python train.py --method selfie --teacher-ckpt output/checkpoints/baseline_best.pth

# 3. 评估所有方法
python eval.py

# 4. 一键运行全部
python train.py --all
```

## 数据依赖

共享 MT-UNet 的数据管线：

- `E:\Claude code\project\mtunet\pipeline\MiniData\Unet_Dataset\` — .npz 样本
- `E:\Claude code\project\mtunet\pipeline\norm_stats.npz` — 归一化统计量

每个 .npz 文件包含：
- `X_dnb, X_basemap, X_mod` — 3 通道 VIIRS 图像
- `Y_mask` — CLDMSK 4 类标签（**噪声标签**）
- `Center_Label` — 雷达验证标签 0/1（**干净标签**，仅 1 像素）

## 预期产出

| 方法 | 预期效果 |
|------|---------|
| Baseline | Binary IoU ≈ 0.88-0.90（复现 MT-UNet） |
| Loss Correction | 在 CLDMSK 系统性偏差场景下优于 Baseline |
| Co-Teaching | 大噪声比例场景（>50% noisy）下鲁棒 |
| SELFIE | 学生模型超过教师（证明标签净化有效） |

## 课题来源

三专家交叉评价（2026-05-19），详见：

- `E:\Claude code\project\mtunet\docs\supplement_cloud_detection_topics.md` — 课题提案
- `E:\Claude code\project\mtunet\docs\plan_1.1_evidential_uncertainty.md` — EDL 课题（可作为 SELFIE 教师增强）
