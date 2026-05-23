"""Run PhysPrior moderate 3 times to get mean±std."""
import os
import sys
import json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Override config for Phase 3
import config
config.MTUNET_DATASET = "E:/Data/Unet_Dataset"

from config import MTUNET_DATASET, CHECKPOINT_DIR, OUTPUT_DIR, train_cfg
from data import make_dataloaders
from methods.physical_prior import PhysicalPriorCorrector
from eval import compute_radar_accuracy_with_ci
import torch

def run_once(run_id):
    """Run PhysPrior moderate once and return accuracy."""
    print(f"\n{'='*60}")
    print(f"Run {run_id}/3")
    print(f"{'='*60}")
    
    train_loader, val_loader = make_dataloaders(
        MTUNET_DATASET, batch_size=train_cfg.batch_size, use_phase3=True
    )
    
    corrector = PhysicalPriorCorrector(train_loader, val_loader, preset="moderate")
    model, best_loss, stats = corrector.run()
    
    # Evaluate
    result = compute_radar_accuracy_with_ci(model, val_loader, "cuda")
    acc = result["overall"]["mean"]
    ci_lower = result["overall"]["ci_lower"]
    ci_upper = result["overall"]["ci_upper"]
    
    # Save checkpoint with run_id
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"physprior_moderate_run{run_id}.pth")
    torch.save(model.state_dict(), ckpt_path)
    
    print(f"\nRun {run_id}: {acc:.1%} [{ci_lower:.1%}, {ci_upper:.1%}]")
    return acc, ci_lower, ci_upper

def main():
    results = []
    for run_id in range(1, 4):
        acc, ci_lower, ci_upper = run_once(run_id)
        results.append({
            "run": run_id,
            "accuracy": acc,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        })
    
    # Compute statistics
    accs = [r["accuracy"] for r in results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    
    print(f"\n{'='*60}")
    print(f"RESULTS (3 runs)")
    print(f"{'='*60}")
    for r in results:
        print(f"  Run {r['run']}: {r['accuracy']:.1%}")
    print(f"\n  Mean: {mean_acc:.1%} ± {std_acc:.1%}")
    print(f"  Range: [{min(accs):.1%}, {max(accs):.1%}]")
    
    # Save results
    output = {
        "runs": results,
        "mean": float(mean_acc),
        "std": float(std_acc),
        "min": float(min(accs)),
        "max": float(max(accs)),
    }
    output_path = os.path.join(OUTPUT_DIR, "physprior_3runs.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

if __name__ == "__main__":
    main()
