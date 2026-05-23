"""Grad-CAM visualization for noise-label cloud detection.

Generates heatmaps showing which image regions the model focuses on
when making predictions. Uses the last decoder conv layer as target.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET, CHECKPOINT_DIR, OUTPUT_DIR
from data import make_dataloaders
from models.mt_unet import MT_UNet


class GradCAM:
    """Grad-CAM implementation for MT-UNet."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Use full backward hook
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, moon_phase=None, solar_zenith=None, target_class=None):
        """Generate Grad-CAM heatmap."""
        self.model.eval()
        
        # Forward pass
        output, _ = self.model(input_tensor, moon_phase=moon_phase, solar_zenith=solar_zenith)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            print("Warning: hooks didn't capture gradients/activations")
            return np.zeros(input_tensor.shape[2:])
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input size
        cam = torch.nn.functional.interpolate(
            cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )
        
        return cam.squeeze().cpu().numpy()


def find_last_conv_layer(model):
    """Find the last Conv2d layer in the decoder."""
    last_conv = None
    for name, module in model.base_model.decoder.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        # Fallback: use segmentation head
        last_conv = model.mask_head
    return last_conv


def visualize_gradcam(model, dataloader, device, n_samples=5, output_dir=None):
    """Generate Grad-CAM visualizations for sample images."""
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "gradcam")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find target layer
    target_layer = find_last_conv_layer(model)
    grad_cam = GradCAM(model, target_layer)
    
    # Custom colormap
    colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]
    cmap = LinearSegmentedColormap.from_list("gradcam", colors)
    
    samples_processed = 0
    
    for batch in dataloader:
        if samples_processed >= n_samples:
            break
        
        images = batch["image"].to(device)
        moon = batch["moon_phase"].to(device)
        sza = batch["solar_zenith"].to(device)
        filenames = batch["filename"]
        center_label = batch["center_label"]
        
        for i in range(min(images.size(0), n_samples - samples_processed)):
            img = images[i:i+1]
            filename = filenames[i]
            cl = center_label[i].item()
            
            # Reset gradients for each sample
            grad_cam.gradients = None
            grad_cam.activations = None
            
            # Generate Grad-CAM for cloud class (class 3)
            heatmap = grad_cam.generate(
                img, 
                moon_phase=moon[i:i+1],
                solar_zenith=sza[i:i+1],
                target_class=3
            )
            
            # Get original image (DNB channel)
            dnb = img[0, 0].cpu().numpy()
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            
            axes[0].imshow(dnb, cmap='gray')
            radar_str = f"Radar={'Cloud' if cl==1 else 'Clear'}" if cl == cl else "Radar=N/A"
            axes[0].set_title(f'DNB ({radar_str})')
            axes[0].axis('off')
            
            axes[1].imshow(heatmap, cmap=cmap, vmin=0, vmax=1)
            axes[1].set_title('Grad-CAM (Cloud Class)')
            axes[1].axis('off')
            
            axes[2].imshow(dnb, cmap='gray', alpha=0.7)
            axes[2].imshow(heatmap, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
            plt.suptitle(f'{filename}', fontsize=9)
            plt.tight_layout()
            
            out_name = f"gradcam_{filename.replace('.npz', '.png')}"
            output_path = os.path.join(output_dir, out_name)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Saved: {out_name}")
            samples_processed += 1
    
    print(f"\nGenerated {samples_processed} Grad-CAM visualizations in {output_dir}")


def main():
    """Generate Grad-CAM visualizations."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--method", type=str, default="physprior_moderate")
    parser.add_argument("--n-samples", type=int, default=5)
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MT_UNet(in_channels=3, mask_classes=4).to(device)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{args.method}_best.pth")
    
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")
    
    _, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=1)
    visualize_gradcam(model, val_loader, device, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
