"""Main training entry point — run any noise-label method.

Usage:
    # 1. First, analyze CLDMSK noise patterns
    python train.py --analyze

    # 2. Run methods
    python train.py --method baseline
    python train.py --method loss_correction
    python train.py --method coteaching
    python train.py --method selfie [--teacher-ckpt PATH]

    # 3. Run all methods sequentially and compare
    python train.py --all
"""
import argparse
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from determinism import seed_everything
seed_everything(int(os.environ.get("NL_SEED", "42")))  # fixed seed: the pipeline must be bit-reproducible


from config import MTUNET_DATASET, OUTPUT_DIR, train_cfg
from data import make_dataloaders, CLDMSKNoiseAnalyzer, print_noise_report
from methods import (BaselineTrainer, LossCorrectionTrainer, CoTeachingTrainer,
                       SELFIE, GCEBaseline, PhysicalPriorCorrector, PRESETS,
                       PhysicsGuidedTrainer, MixupTrainer)


def cmd_analyze():
    """Analyze CLDMSK noise patterns using radar ground truth."""
    analyzer = CLDMSKNoiseAnalyzer(MTUNET_DATASET)
    summary = print_noise_report(analyzer)

    # Save to JSON for downstream use
    report_path = os.path.join(OUTPUT_DIR, "noise_analysis.json")
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Noise analysis saved to {report_path}")

    # Also save the correction matrix as .npy
    correction_matrix = analyzer.get_loss_correction_matrix()
    matrix_path = os.path.join(OUTPUT_DIR, "correction_matrix.npy")
    import numpy as np
    np.save(matrix_path, correction_matrix)
    print(f"Correction matrix saved to {matrix_path}")

    return summary, correction_matrix


def cmd_baseline():
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    trainer = BaselineTrainer(train_loader, val_loader)
    model, best_loss = trainer.run()
    return model


def cmd_loss_correction():
    import numpy as np
    matrix_path = os.path.join(OUTPUT_DIR, "correction_matrix.npy")
    if not os.path.exists(matrix_path):
        print("Run --analyze first to generate correction matrix")
        sys.exit(1)
    T = np.load(matrix_path)
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    trainer = LossCorrectionTrainer(train_loader, val_loader, T)
    model, best_loss = trainer.run()
    return model


def cmd_coteaching():
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    trainer = CoTeachingTrainer(train_loader, val_loader)
    models, best_loss = trainer.run()
    return models


def cmd_selfie(teacher_ckpt=None):
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    selfie = SELFIE(train_loader, val_loader, MTUNET_DATASET)
    results = selfie.run_full(teacher_ckpt=teacher_ckpt)

    summary_path = os.path.join(OUTPUT_DIR, "selfie_results.json")
    clean = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean[k] = {kk: float(vv) if hasattr(vv, "item") else vv
                       for kk, vv in v.items()}
    with open(summary_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"SELFIE results saved to {summary_path}")
    return results


def cmd_selfie_dual():
    """SELFIE with dual-teacher consensus (P0-2 fix)."""
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    selfie = SELFIE(train_loader, val_loader, MTUNET_DATASET)
    results = selfie.run_full_dual()

    summary_path = os.path.join(OUTPUT_DIR, "selfie_dual_results.json")
    clean = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean[k] = {kk: float(vv) if hasattr(vv, "item") else vv
                       for kk, vv in v.items()}
    with open(summary_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"SELFIE-Dual results saved to {summary_path}")
    return results


def cmd_selfie_phys():
    """SELFIE with PhysPrior-corrected teacher (breaks circular reasoning)."""
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    selfie = SELFIE(train_loader, val_loader, MTUNET_DATASET)
    results = selfie.run_full_phys_teacher(phys_preset="moderate")

    summary_path = os.path.join(OUTPUT_DIR, "selfie_phys_results.json")
    clean = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean[k] = {kk: float(vv) if hasattr(vv, "item") else vv
                       for kk, vv in v.items()}
    with open(summary_path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"SELFIE-Phys results saved to {summary_path}")
    return results


def cmd_gce(q=0.7):
    """GCE baseline (P1-6)."""
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    trainer = GCEBaseline(train_loader, val_loader, q=q)
    model, best_loss = trainer.run()
    return model


def cmd_mixup(alpha=0.4):
    """Mixup baseline — noise-robust regularization."""
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    trainer = MixupTrainer(train_loader, val_loader, alpha=alpha)
    model, best_loss = trainer.run()
    return model


def cmd_physical_prior(preset="conservative"):
    """Physical Prior Correction — ablation (Expert 1's cold-surface insight)."""
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    corrector = PhysicalPriorCorrector(train_loader, val_loader, preset=preset)
    model, best_loss, stats = corrector.run()

    summary_path = os.path.join(OUTPUT_DIR, f"physprior_{preset}_stats.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Physical Prior stats saved to {summary_path}")
    return model


def cmd_physics_guided(preset="moderate"):
    """Physics-Guided Loss — soft, differentiable physical constraint."""
    train_loader, val_loader = make_dataloaders(MTUNET_DATASET, batch_size=train_cfg.batch_size,
                                     use_basemap=True, use_phase4=True)
    trainer = PhysicsGuidedTrainer(train_loader, val_loader, preset=preset)
    model, best_loss = trainer.run()
    return model


def cmd_all():
    """Run noise analysis + all four methods and produce comparison."""
    print("\n" + "=" * 60)
    print("  E1: Noise-Label Cloud Detection — Full Pipeline")
    print("=" * 60)

    # Step 0: Analyze noise
    print("\n>>> Step 0: Noise Analysis")
    summary, T = cmd_analyze()

    results = {}

    # Step 1: Baseline
    print("\n>>> Step 1: Baseline (CLDMSK as ground truth)")
    cmd_baseline()
    results["baseline"] = {"note": "see logs/baseline_train_log.txt"}

    # Step 2: Loss Correction
    print("\n>>> Step 2: Loss Correction")
    cmd_loss_correction()
    results["loss_correction"] = {"note": "see logs/loss_correction_train_log.txt"}

    # Step 3: Co-teaching
    print("\n>>> Step 3: Co-Teaching")
    cmd_coteaching()
    results["coteaching"] = {"note": "see logs/coteaching_train_log.txt"}

    # Step 4: SELFIE (single teacher)
    print("\n>>> Step 4: SELFIE (Teacher-Student Purification)")
    teacher_ckpt = os.path.join(OUTPUT_DIR, "checkpoints", "baseline_best.pth")
    if not os.path.exists(teacher_ckpt):
        teacher_ckpt = None
    cmd_selfie(teacher_ckpt=teacher_ckpt)

    # Step 5: SELFIE-Dual (P0-2 fix)
    print("\n>>> Step 5: SELFIE-Dual (Dual-Teacher Consensus)")
    cmd_selfie_dual()
    results["selfie_dual"] = {"note": "see output/selfie_dual_results.json"}

    # Step 6: GCE baseline
    print("\n>>> Step 6: GCE Baseline (q=0.7)")
    cmd_gce(q=0.7)
    results["gce_q07"] = {"note": "see logs/gce_q0.7_train_log.txt"}

    print("\n" + "=" * 60)
    print("  Full pipeline complete. Results in output/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise-Label Cloud Detection")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze CLDMSK noise patterns")
    parser.add_argument("--method", type=str, default=None,
                        choices=["baseline", "loss_correction", "coteaching", "selfie",
                                 "selfie-dual", "selfie-phys", "gce", "mixup",
                                 "phys-cons", "phys-mod", "phys-agg",
                                 "physguide-cons", "physguide-mod", "physguide-agg"],
                        help="Which method to run")
    parser.add_argument("--teacher-ckpt", type=str, default=None,
                        help="Pre-trained teacher checkpoint for SELFIE")
    parser.add_argument("--gce-q", type=float, default=0.7,
                        help="q parameter for GCE loss (0=CE, 1=MAE)")
    parser.add_argument("--all", action="store_true",
                        help="Run all methods sequentially")
    args = parser.parse_args()

    if args.analyze:
        cmd_analyze()
    elif args.all:
        cmd_all()
    elif args.method == "baseline":
        cmd_baseline()
    elif args.method == "loss_correction":
        cmd_loss_correction()
    elif args.method == "coteaching":
        cmd_coteaching()
    elif args.method == "selfie":
        cmd_selfie(teacher_ckpt=args.teacher_ckpt)
    elif args.method == "selfie-dual":
        cmd_selfie_dual()
    elif args.method == "selfie-phys":
        cmd_selfie_phys()
    elif args.method == "gce":
        cmd_gce(q=args.gce_q)
    elif args.method == "mixup":
        cmd_mixup()
    elif args.method == "phys-cons":
        cmd_physical_prior(preset="conservative")
    elif args.method == "phys-mod":
        cmd_physical_prior(preset="moderate")
    elif args.method == "phys-agg":
        cmd_physical_prior(preset="aggressive")
    elif args.method == "physguide-cons":
        cmd_physics_guided(preset="conservative")
    elif args.method == "physguide-mod":
        cmd_physics_guided(preset="moderate")
    elif args.method == "physguide-agg":
        cmd_physics_guided(preset="aggressive")
    else:
        parser.print_help()
