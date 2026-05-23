"""Generate failure case visualization for winter samples.

Shows cases where PhysPrior fails due to extreme radiative cooling.
"""
import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET, CHECKPOINT_DIR, OUTPUT_DIR
from data import make_dataloaders
from models.mt_unet import MT_UNet
import re

def get_season(filename):
    match = re.search(r'A\d{4}(\d{3})\.', filename)
    if match:
        jd = int(match.group(1))
        if jd >= 335 or jd < 60: return 'winter'
        elif jd < 152: return 'spring'
        elif jd < 244: return 'summer'
        else: return 'autumn'
    return 'unknown'

@torch.no_grad()
def find_failure_cases(model, dataloader, device, n_samples=3):
    """Find winter samples where model fails."""
    model.eval()
    failures = []
    
    for batch in dataloader:
        if len(failures) >= n_samples:
            break
        
        x = batch["image"].to(device)
        moon = batch["moon_phase"].to(device)
        sza = batch["solar_zenith"].to(device)
        center = batch["center_label"]
        radar_loc = batch["radar_loc"]
        filenames = batch["filename"]
        
        pred_logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = torch.argmax(pred_logits, dim=1)
        
        for b in range(x.size(0)):
            if len(failures) >= n_samples:
                break
            
            cl = center[b].item()
            if cl != cl:  # NaN
                continue
            
            season = get_season(filenames[b])
            if season != 'winter':
                continue
            
            ry, rx = radar_loc[b, 0].item(), radar_loc[b, 1].item()
            y0, y1 = max(0, ry-1), min(128, ry+2)
            x0, x1 = max(0, rx-1), min(128, rx+2)
            win = pred_cls[b, y0:y1, x0:x1]
            pred_bin = 1 if (win >= 2).any() else 0
            
            if pred_bin != int(cl):  # Failure case
                failures.append({
                    'filename': filenames[b],
                    'season': season,
                    'radar_truth': int(cl),
                    'prediction': pred_bin,
                    'image': x[b].cpu().numpy(),
                    'pred_map': pred_cls[b].cpu().numpy(),
                })
    
    return failures

def visualize_failures(failures, output_dir):
    """Create visualization of failure cases."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, fail in enumerate(failures):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # DNB channel
        dnb = fail['image'][0]
        axes[0].imshow(dnb, cmap='gray')
        axes[0].set_title(f"DNB\n{fail['filename'][:30]}...", fontsize=9)
        axes[0].axis('off')
        
        # M15 channel (brightness temperature)
        m15 = fail['image'][2]
        im = axes[1].imshow(m15, cmap='coolwarm', vmin=-3, vmax=1)
        axes[1].set_title(f"M15 (BT)\nRadar={'Cloud' if fail['radar_truth']==1 else 'Clear'}", fontsize=9)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Prediction
        pred = fail['pred_map']
        axes[2].imshow(pred, cmap='RdYlBu_r', vmin=0, vmax=3)
        axes[2].set_title(f"Prediction\nModel={'Cloud' if fail['prediction']==1 else 'Clear'}", fontsize=9)
        axes[2].axis('off')
        
        # Add failure annotation
        truth_str = "Cloud" if fail['radar_truth'] == 1 else "Clear"
        pred_str = "Cloud" if fail['prediction'] == 1 else "Clear"
        status = "FN" if fail['radar_truth'] == 1 and fail['prediction'] == 0 else "FP"
        
        fig.suptitle(f"Winter Failure Case {i+1}: {status} (Truth={truth_str}, Pred={pred_str})", 
                    fontsize=11, fontweight='bold', color='red')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f"failure_winter_{i+1}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = MT_UNet(in_channels=3, mask_classes=4).to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "physprior_moderate_best.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded: {ckpt_path}")
    
    # Load data
    _, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=1, use_phase3=True)
    
    # Find failure cases
    print("Finding winter failure cases...")
    failures = find_failure_cases(model, val_loader, device, n_samples=3)
    
    if not failures:
        print("No winter failure cases found!")
        # Try any failure cases
        print("Looking for any failure cases...")
        failures = find_failure_cases(model, val_loader, device, n_samples=3)
    
    print(f"Found {len(failures)} failure cases")
    
    # Visualize
    output_dir = os.path.join(OUTPUT_DIR, "failure_cases")
    visualize_failures(failures, output_dir)
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
