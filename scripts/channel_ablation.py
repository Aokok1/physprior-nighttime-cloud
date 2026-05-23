"""Channel ablation study for noise-label cloud detection.

Tests 7 configurations:
  1. Full (DNB + Basemap + M15) - baseline
  2. -DNB (Basemap + M15)
  3. -M15 (DNB + Basemap)
  4. -Basemap (DNB + M15)
  5. TIR-only (M15 only) - CLDMSK equivalent
  6. DNB-only
  7. Basemap-only

Each configuration requires retraining the model from scratch.
"""
import os
import sys
import json
import time
import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET, CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR, train_cfg
from data import make_dataloaders
from models.mt_unet import MT_UNet, MT_Loss
from methods.baseline import EarlyStopping, compute_binary_metrics


# Channel configurations
CHANNEL_CONFIGS = {
    "full": {
        "description": "DNB + Basemap + M15 (baseline)",
        "channels": [0, 1, 2],
        "in_channels": 3,
    },
    "no_dnb": {
        "description": "Basemap + M15 (remove DNB)",
        "channels": [1, 2],
        "in_channels": 2,
    },
    "no_m15": {
        "description": "DNB + Basemap (remove M15)",
        "channels": [0, 1],
        "in_channels": 2,
    },
    "no_basemap": {
        "description": "DNB + M15 (remove Basemap)",
        "channels": [0, 2],
        "in_channels": 2,
    },
    "tir_only": {
        "description": "M15 only (TIR-only, CLDMSK equivalent)",
        "channels": [2],
        "in_channels": 1,
    },
    "dnb_only": {
        "description": "DNB only",
        "channels": [0],
        "in_channels": 1,
    },
    "basemap_only": {
        "description": "Basemap only",
        "channels": [1],
        "in_channels": 1,
    },
}


class ChannelSelectDataset(torch.utils.data.Dataset):
    """Wrapper to select specific channels from the original dataset."""
    
    def __init__(self, original_dataset, channel_indices):
        self.dataset = original_dataset
        self.channel_indices = channel_indices
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item["image"] = item["image"][self.channel_indices]
        return item


class AblationTrainer:
    """Train MT-UNet with specific channel configuration."""
    
    def __init__(self, config_name, config, epochs=80, device=None):
        self.config_name = config_name
        self.config = config
        self.epochs = epochs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def setup(self):
        self.model = MT_UNet(
            backbone="resnet34",
            in_channels=self.config["in_channels"],
            mask_classes=4,
            use_film=True if self.config["in_channels"] == 3 else False,
        ).to(self.device)
        
        self.criterion = MT_Loss(
            weight_center=train_cfg.weight_center,
            dice_weight=train_cfg.dice_weight,
        ).to(self.device)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=train_cfg.T_0, T_mult=train_cfg.T_mult,
            eta_min=1e-6,
        )
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for batch in train_loader:
            x = batch["image"].to(self.device)
            y_mask = batch["mask_noisy"].to(self.device)
            y_height = batch["height"].to(self.device)
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            radar_loc = batch["radar_loc"].to(self.device)
            center = batch["center_label"].to(self.device)
            
            self.optimizer.zero_grad()
            pred_mask, pred_hgt = self.model(x, moon_phase=moon, solar_zenith=sza)
            loss, _, _, _ = self.criterion(
                pred_mask, pred_hgt, y_mask, y_height, radar_loc, center
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_cfg.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        for batch in val_loader:
            x = batch["image"].to(self.device)
            center = batch["center_label"]
            radar_loc = batch["radar_loc"]
            moon = batch["moon_phase"].to(self.device)
            sza = batch["solar_zenith"].to(self.device)
            
            pred_logits, _ = self.model(x, moon_phase=moon, solar_zenith=sza)
            pred_cls = torch.argmax(pred_logits, dim=1)
            
            for b in range(x.size(0)):
                cl = center[b].item()
                if cl != cl:
                    continue
                ry, rx = radar_loc[b, 0].item(), radar_loc[b, 1].item()
                y0, y1 = max(0, ry - 1), min(128, ry + 2)
                x0, x1 = max(0, rx - 1), min(128, rx + 2)
                win = pred_cls[b, y0:y1, x0:x1]
                bin_win = (win >= 2).int()
                pred_bin = 1 if bin_win.sum() > 0 else 0
                if pred_bin == int(cl):
                    correct += 1
                total += 1
        
        return correct / max(total, 1)
    
    def run(self):
        if not hasattr(self, 'model'):
            self.setup()
        
        # Create dataloaders with channel selection
        train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size)
        
        train_dataset = ChannelSelectDataset(train_loader.dataset, self.config["channels"])
        val_dataset = ChannelSelectDataset(val_loader.dataset, self.config["channels"])
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_cfg.batch_size, shuffle=True, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=train_cfg.batch_size, shuffle=False, num_workers=0
        )
        
        early_stop = EarlyStopping(patience=train_cfg.patience)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"ablation_{self.config_name}_best.pth")
        
        print(f"\n{'='*60}")
        print(f"Training ablation: {self.config_name}")
        print(f"  {self.config['description']}")
        print(f"  Channels: {self.config['channels']}")
        print(f"  In channels: {self.config['in_channels']}")
        print(f"{'='*60}")
        
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(train_loader)
            
            if (epoch + 1) % 5 == 0:
                val_acc = self.evaluate(val_loader)
                elapsed = time.time() - start_time
                
                print(f"  Epoch {epoch+1}/{self.epochs} | "
                      f"Loss: {train_loss:.4f} | "
                      f"Radar Acc: {val_acc:.1%} | "
                      f"Time: {elapsed:.0f}s")
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(self.model.state_dict(), checkpoint_path)
                    early_stop.counter = 0
                else:
                    early_stop.step(train_loss)
                
                if early_stop.counter >= early_stop.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        print(f"  Best accuracy: {best_acc:.1%}")
        return checkpoint_path, best_acc


def main():
    """Run ablation study."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Configs to run (default: all)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per config")
    args = parser.parse_args()
    
    configs_to_run = args.configs or list(CHANNEL_CONFIGS.keys())
    
    results = {}
    
    for config_name in configs_to_run:
        if config_name not in CHANNEL_CONFIGS:
            print(f"Unknown config: {config_name}")
            continue
        
        config = CHANNEL_CONFIGS[config_name]
        
        # Check if checkpoint already exists
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"ablation_{config_name}_best.pth")
        if os.path.exists(ckpt_path):
            print(f"\nSkipping {config_name} (checkpoint exists)")
            continue
        
        trainer = AblationTrainer(config_name, config, epochs=args.epochs)
        ckpt_path, best_acc = trainer.run()
        
        results[config_name] = {
            "description": config["description"],
            "channels": config["channels"],
            "in_channels": config["in_channels"],
            "best_accuracy": best_acc,
        }
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
