"""Phase 3: Train core models on cleaned dataset.

Dataset: Train_no_overlap (2472 samples) + Test with winter (197 samples)
- Removed 113 same-overpass Changsha samples
- Added 29 winter samples to Test

Methods to train:
  1. 3ch baseline
  2. 3ch PhysPrior moderate
  3. 2ch baseline (DNB+M15)
  4. 2ch PhysPrior moderate
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Override config for Phase 3
import config
config.MTUNET_DATASET = "E:/Data/Unet_Dataset"

from config import MTUNET_DATASET, CHECKPOINT_DIR, LOG_DIR, train_cfg
from data import make_dataloaders
from methods.physical_prior import PhysicalPriorCorrector
from methods.baseline import BaselineTrainer


def train_phase3_baseline(use_basemap=True, epochs=80):
    """Train baseline on Phase 3 dataset."""
    ch_str = "3ch" if use_basemap else "2ch"
    print(f"\n{'='*60}")
    print(f"Phase 3: {ch_str} Baseline (Train_no_overlap + Test_winter)")
    print(f"{'='*60}")
    
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size,
        use_basemap=use_basemap, use_phase3=True
    )
    
    trainer = BaselineTrainer(train_loader, val_loader)
    model, best_loss = trainer.run()
    
    # Rename checkpoint
    src = os.path.join(CHECKPOINT_DIR, "baseline_best.pth")
    dst = os.path.join(CHECKPOINT_DIR, f"phase3_{ch_str}_baseline_best.pth")
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"Checkpoint saved: {dst}")


def train_phase3_physprior(use_basemap=True, preset="moderate"):
    """Train PhysPrior on Phase 3 dataset."""
    ch_str = "3ch" if use_basemap else "2ch"
    print(f"\n{'='*60}")
    print(f"Phase 3: {ch_str} PhysPrior {preset} (Train_no_overlap + Test_winter)")
    print(f"{'='*60}")
    
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size,
        use_basemap=use_basemap, use_phase3=True
    )
    
    corrector = PhysicalPriorCorrector(train_loader, val_loader, preset=preset)
    model, best_loss, stats = corrector.run()
    
    # Rename checkpoint
    src = os.path.join(CHECKPOINT_DIR, f"physprior_{preset}_best.pth")
    dst = os.path.join(CHECKPOINT_DIR, f"phase3_{ch_str}_physprior_{preset}_best.pth")
    if os.path.exists(src):
        os.rename(src, dst)
        print(f"Checkpoint saved: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        choices=["baseline-3ch", "baseline-2ch", 
                                 "physprior-3ch", "physprior-2ch", "all"],
                        help="Method to train")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    
    if args.method == "baseline-3ch":
        train_phase3_baseline(use_basemap=True, epochs=args.epochs)
    elif args.method == "baseline-2ch":
        train_phase3_baseline(use_basemap=False, epochs=args.epochs)
    elif args.method == "physprior-3ch":
        train_phase3_physprior(use_basemap=True)
    elif args.method == "physprior-2ch":
        train_phase3_physprior(use_basemap=False)
    elif args.method == "all":
        train_phase3_baseline(use_basemap=True, epochs=args.epochs)
        train_phase3_baseline(use_basemap=False, epochs=args.epochs)
        train_phase3_physprior(use_basemap=True)
        train_phase3_physprior(use_basemap=False)


if __name__ == "__main__":
    main()
