"""Generate GRSL paper figures.

Figure 1: Study area and data flow diagram
Figure 2: Channel ablation and baseline comparison
Figure 3: Seasonal accuracy distribution
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
})


def fig1_study_area():
    """Figure 1: Study area and data flow diagram."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    
    # Left panel: Study area map (schematic)
    ax = axes[0]
    ax.set_xlim(108, 115)
    ax.set_ylim(24, 30)
    
    # Draw simplified China outline (Hunan region)
    ax.fill([108, 115, 115, 108], [24, 24, 30, 30], color='#f0f0f0', edgecolor='gray', linewidth=0.5)
    
    # Station locations
    changsha = (112.98, 28.23)
    longmen = (112.80, 25.40)
    
    ax.plot(*changsha, 'r^', markersize=12, label='Changsha (Train)')
    ax.plot(*longmen, 'bs', markersize=10, label='Longmen (Test)')
    
    ax.annotate('Changsha\n(901 samples)', changsha, textcoords="offset points",
                xytext=(15, 10), fontsize=8, color='red')
    ax.annotate('Longmen\n(168 samples)', longmen, textcoords="offset points",
                xytext=(15, -15), fontsize=8, color='blue')
    
    # Draw radar coverage circle
    circle = plt.Circle(longmen, 0.3, fill=False, color='blue', linestyle='--', linewidth=1)
    ax.add_patch(circle)
    ax.annotate('Ka-band\nRadar', longmen, textcoords="offset points",
                xytext=(20, 5), fontsize=7, color='blue', style='italic')
    
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('(a) Study Area', fontweight='bold')
    ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Right panel: Data flow diagram
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Draw boxes
    boxes = [
        (5, 9, 'VIIRS Nighttime Data\n(DNB + M15 + Basemap)', '#e6f3ff'),
        (5, 7, 'CLDMSK Cloud Mask\n(Noisy Labels, 52.75%)', '#fff3e6'),
        (5, 5, 'PhysPrior Denoising\n(M15≥264K & std≤1.5K)', '#e6ffe6'),
        (5, 3, 'MT-UNet Training\n(3ch, ResNet34)', '#f3e6ff'),
        (5, 1, 'Radar Validation\n(Ka-band, n=197)', '#ffe6e6'),
    ]
    
    for x, y, text, color in boxes:
        box = FancyBboxPatch((x-2, y-0.5), 4, 1, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='gray', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Draw arrows
    for i in range(len(boxes)-1):
        ax.annotate('', xy=(5, boxes[i+1][1]+0.5), xytext=(5, boxes[i][1]-0.5),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    ax.set_title('(b) Data Flow', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig1_study_area.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def fig2_ablation():
    """Figure 2: Channel ablation and baseline comparison."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Data
    configs = ['Basemap\nOnly', 'Full\n(3ch)', 'DNB\nOnly', 'DNB+BM', 'BM+M15', 'DNB+M15', 'M15\nOnly']
    accuracies = [26.8, 50.6, 56.0, 56.5, 58.9, 69.0, 71.4]
    colors = ['#d32f2f', '#ff9800', '#ffc107', '#ffc107', '#4caf50', '#2196f3', '#1976d2']
    
    # Create bars
    bars = ax.bar(range(len(configs)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Highlight key findings
    ax.axhline(y=50.6, color='red', linestyle='--', linewidth=1, alpha=0.7, label='3ch Baseline (50.6%)')
    ax.axhline(y=71.4, color='blue', linestyle='--', linewidth=1, alpha=0.7, label='M15 Only (71.4%)')
    
    # Labels
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel('Radar Accuracy (%)')
    ax.set_xlabel('Channel Configuration')
    ax.set_title('Channel Ablation Study: "Less is More"', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 85)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotations
    ax.annotate('Basemap is harmful!\n(26.8% alone)', xy=(0, 26.8), xytext=(1.5, 35),
               arrowprops=dict(arrowstyle='->', color='red'),
               fontsize=8, color='red', fontweight='bold')
    
    ax.annotate('M15 is most valuable\n(71.4% alone)', xy=(6, 71.4), xytext=(5, 80),
               arrowprops=dict(arrowstyle='->', color='blue'),
               fontsize=8, color='blue', fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig2_ablation.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def fig3_seasonal():
    """Figure 3: Seasonal accuracy distribution."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Data
    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    accuracies = [79.6, 84.9, 87.8, 58.6]
    n_samples = [54, 73, 41, 29]
    colors = ['#4caf50', '#ff9800', '#f44336', '#2196f3']
    
    # Create bars
    bars = ax.bar(range(len(seasons)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add value labels and sample sizes
    for bar, acc, n in zip(bars, accuracies, n_samples):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add overall accuracy line
    ax.axhline(y=80.2, color='black', linestyle='--', linewidth=1.5, label='Overall: 80.2%')
    
    # Labels
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, fontsize=11)
    ax.set_ylabel('Radar Accuracy (%)')
    ax.set_xlabel('Season')
    ax.set_title('Seasonal Accuracy Distribution (PhysPrior Moderate)', fontweight='bold', fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation for winter
    ax.annotate('Winter: Radiative cooling\ncauses cold-surface FP',
               xy=(3, 58.6), xytext=(1.5, 45),
               arrowprops=dict(arrowstyle='->', color='blue'),
               fontsize=8, color='blue', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "fig3_seasonal.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("Generating GRSL paper figures...")
    fig1_study_area()
    fig2_ablation()
    fig3_seasonal()
    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
