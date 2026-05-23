"""Four-layer evaluation for noise-label cloud detection.

Layer 1 — Radar Accuracy (PRIMARY, unbiased): 3×3 window match at radar pixels,
         with Bootstrap 95% CI.
Layer 2 — CLDMSK Disagreement Rate (purification signal): high disagreement +
         high RadarAcc = genuine purification.
Layer 3 — Human Audit Export: stratified random sample of corrected pixels
         for manual verification.
Layer 4 — CLDMSK-IoU / F1 (REFERENCE ONLY, downgraded): may penalize models
         that correctly fix CLDMSK errors.
"""
import os
import sys
import json
import random
import torch
import numpy as np
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET, CHECKPOINT_DIR, OUTPUT_DIR
from data import make_dataloaders
from models.mt_unet import MT_UNet
from methods.baseline import compute_binary_metrics

random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════════
# Layer 1: Radar Accuracy with Bootstrap CI (PRIMARY METRIC)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _collect_radar_predictions(model, dataloader, device):
    """Collect all (pred_binary, radar_truth, cldmsk_class, filename) tuples."""
    model.eval()
    records = []
    for batch in dataloader:
        x = batch["image"].to(device)
        moon = batch["moon_phase"].to(device)
        sza = batch["solar_zenith"].to(device)
        radar_loc = batch["radar_loc"]
        center = batch["center_label"]
        fnames = batch["filename"]

        pred_logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = torch.argmax(pred_logits, dim=1)

        for b in range(x.size(0)):
            cl = center[b].item()
            if cl != cl:  # NaN
                continue
            ry, rx = radar_loc[b, 0].item(), radar_loc[b, 1].item()
            y0, y1 = max(0, ry - 1), min(128, ry + 2)
            x0, x1 = max(0, rx - 1), min(128, rx + 2)
            win = pred_cls[b, y0:y1, x0:x1]
            bin_win = (win >= 2).int()
            pred_bin = 1 if bin_win.sum() > 0 else (0 if bin_win.float().mean() <= 0.5 else 1)

            records.append({
                "filename": fnames[b],
                "radar_truth": int(cl),
                "pred_binary": int(pred_bin),
                "correct": int(pred_bin == int(cl)),
                "cldmsk_class": int(batch["mask_noisy"][b, ry, rx].item()),
            })
    return records


def _bootstrap_ci(samples, n_bootstrap=2000, alpha=0.05):
    """Bootstrap 95% CI for accuracy from list of binary (correct/incorrect)."""
    corrects = np.array([s["correct"] for s in samples])
    n = len(corrects)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    means = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        means.append(corrects[idx].mean())
    means = np.array(means)
    return {
        "mean": float(corrects.mean()),
        "ci_lower": float(np.percentile(means, alpha / 2 * 100)),
        "ci_upper": float(np.percentile(means, (1 - alpha / 2) * 100)),
        "n": n,
    }


def compute_radar_accuracy_with_ci(model, dataloader, device="cuda"):
    """Layer 1: Radar accuracy as primary unbiased metric, with Bootstrap CI."""
    records = _collect_radar_predictions(model, dataloader, device)
    overall = _bootstrap_ci(records)

    # Per-CLDMSK-class breakdown
    per_class = {}
    for c in range(4):
        class_records = [r for r in records if r["cldmsk_class"] == c]
        if len(class_records) >= 5:
            per_class[f"cldmsk_class_{c}"] = _bootstrap_ci(class_records)

    return {"overall": overall, "per_class": per_class, "records": records}


# ═══════════════════════════════════════════════════════════════════════
# Layer 2: CLDMSK Disagreement Rate (purification signal)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_disagreement_rate(model, dataloader, device="cuda"):
    """Layer 2: What fraction of pixels does the model disagree with CLDMSK?

    High disagreement + high Radar Accuracy = genuine purification.
    Low disagreement + low Radar Accuracy = model is overfitting CLDMSK noise.
    """
    model.eval()
    total_disagree = 0
    total_pixels = 0
    # Per-CLDMSK-class disagreement
    per_class_disagree = defaultdict(int)
    per_class_total = defaultdict(int)

    for batch in dataloader:
        x = batch["image"].to(device)
        y_mask = batch["mask_noisy"].to(device)
        moon = batch["moon_phase"].to(device)
        sza = batch["solar_zenith"].to(device)

        pred_logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = torch.argmax(pred_logits, dim=1)

        disagree = (pred_cls != y_mask)
        total_disagree += disagree.sum().item()
        total_pixels += disagree.numel()

        for c in range(4):
            class_mask = (y_mask == c)
            per_class_disagree[c] += (disagree & class_mask).sum().item()
            per_class_total[c] += class_mask.sum().item()

    overall_rate = total_disagree / max(total_pixels, 1)

    per_class_rates = {}
    class_names = ["True Clear", "prob_clear", "prob_cloud", "True Cloud"]
    for c in range(4):
        if per_class_total[c] > 0:
            per_class_rates[f"class_{c}_{class_names[c]}"] = \
                per_class_disagree[c] / per_class_total[c]

    return {
        "overall_disagreement_rate": round(overall_rate, 4),
        "total_disagree_pixels": total_disagree,
        "total_pixels": total_pixels,
        "per_class_disagreement": per_class_rates,
    }


# ═══════════════════════════════════════════════════════════════════════
# Layer 3: Human Audit Export
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def export_audit_samples(model, dataloader, device="cuda", n_per_stratum=5):
    """Layer 3: Export stratified samples for human visual audit.

    Strata (4 × 2):
      CLDMSK class {0,1,2,3} × model agrees/disagrees with CLDMSK

    For each pixel, export:
      - filename, pixel location
      - CLDMSK label, model prediction, model confidence
      - Whether model corrected CLDMSK (and to what)
    """
    model.eval()
    candidates = {c: {"agree": [], "disagree": []} for c in range(4)}

    for batch in dataloader:
        x = batch["image"].to(device)
        y_mask = batch["mask_noisy"]
        moon = batch["moon_phase"].to(device)
        sza = batch["solar_zenith"].to(device)
        fnames = batch["filename"]

        pred_logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        probs = torch.softmax(pred_logits, dim=1)
        pred_cls = torch.argmax(probs, dim=1)
        conf = probs.max(dim=1).values

        B, H, W = pred_cls.shape
        for b in range(B):
            for py in [H // 4, H // 2, 3 * H // 4]:
                for px in [W // 4, W // 2, 3 * W // 4]:
                    cldmsk_c = int(y_mask[b, py, px])
                    pred_c = int(pred_cls[b, py, px])
                    pred_conf = float(conf[b, py, px])

                    entry = {
                        "filename": fnames[b],
                        "pixel": [py, px],
                        "cldmsk_class": cldmsk_c,
                        "model_class": pred_c,
                        "model_confidence": round(pred_conf, 4),
                        "corrected": cldmsk_c != pred_c,
                    }

                    key = "disagree" if cldmsk_c != pred_c else "agree"
                    if len(candidates[cldmsk_c][key]) < n_per_stratum * 5:
                        candidates[cldmsk_c][key].append(entry)

    # Sample n_per_stratum from each stratum
    audit = []
    for c in range(4):
        for key in ["agree", "disagree"]:
            pool = candidates[c][key]
            selected = random.sample(pool, min(n_per_stratum, len(pool)))
            audit.extend(selected)

    return audit


# ═══════════════════════════════════════════════════════════════════════
# Layer 4: Legacy CLDMSK Metrics (REFERENCE ONLY)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_cldmsk_metrics(model, dataloader, device="cuda"):
    """Layer 4: CLDMSK-IoU and F1 — REFERENCE ONLY.

    These may UNDERESTIMATE models that correctly fix CLDMSK errors.
    """
    model.eval()
    tp_all = fp_all = fn_all = tn_all = 0

    for batch in dataloader:
        x = batch["image"].to(device)
        y_mask = batch["mask_noisy"].to(device)
        moon = batch["moon_phase"].to(device)
        sza = batch["solar_zenith"].to(device)

        pred_logits, _ = model(x, moon_phase=moon, solar_zenith=sza)
        pred_cls = torch.argmax(pred_logits, dim=1)

        tp, fp, fn, tn = compute_binary_metrics(pred_cls, y_mask)
        tp_all += tp; fp_all += fp; fn_all += fn; tn_all += tn

    iou = tp_all / (tp_all + fp_all + fn_all + 1e-6)
    f1 = 2 * tp_all / (2 * tp_all + fp_all + fn_all + 1e-6)

    return {
        "cldmsk_iou_REFERENCE_ONLY": round(iou, 4),
        "cldmsk_f1_REFERENCE_ONLY": round(f1, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Full four-layer evaluation for a checkpoint
# ═══════════════════════════════════════════════════════════════════════

def evaluate_full(ckpt_path, val_loader, device="cuda"):
    """Run all four layers of evaluation on a checkpoint."""
    model = MT_UNet(
        backbone="resnet34", in_channels=3, mask_classes=4,
        use_film=True, use_evidential=False,
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    print(f"  Evaluating {os.path.basename(ckpt_path)} ...")

    # Layer 1
    radar = compute_radar_accuracy_with_ci(model, val_loader, device)
    # Layer 2
    disagree = compute_disagreement_rate(model, val_loader, device)
    # Layer 3
    audit = export_audit_samples(model, val_loader, device)
    # Layer 4
    cldmsk = compute_cldmsk_metrics(model, val_loader, device)

    return {
        "layer1_radar_accuracy": radar["overall"],
        "layer1_radar_per_class": radar["per_class"],
        "layer2_disagreement": disagree,
        "layer3_audit_samples": audit[:20],  # top 20 for manual review
        "layer4_cldmsk_reference": cldmsk,
    }


# ═══════════════════════════════════════════════════════════════════════
# Comparison table (updated for new metrics)
# ═══════════════════════════════════════════════════════════════════════

def compare_all():
    """Compare all available checkpoints with four-layer evaluation."""
    _, val_loader = make_dataloaders(MTUNET_DATASET)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoints = {
        "Baseline": os.path.join(CHECKPOINT_DIR, "baseline_best.pth"),
        "GCE (q=0.7)": os.path.join(CHECKPOINT_DIR, "gce_q0.7_best.pth"),
        "Loss Correction": os.path.join(CHECKPOINT_DIR, "loss_correction_best.pth"),
        "Co-Teaching (A)": os.path.join(CHECKPOINT_DIR, "coteaching_a_best.pth"),
        "Co-Teaching (B)": os.path.join(CHECKPOINT_DIR, "coteaching_b_best.pth"),
        "SELFIE Student": os.path.join(CHECKPOINT_DIR, "selfie_student_best.pth"),
        "SELFIE Teacher": os.path.join(CHECKPOINT_DIR, "selfie_teacher.pth"),
        "SELFIE-Dual Student": os.path.join(CHECKPOINT_DIR, "selfie_student_best.pth"),
        "SELFIE-Dual TeacherA": os.path.join(CHECKPOINT_DIR, "selfie_teacher_a.pth"),
        "SELFIE-Dual TeacherB": os.path.join(CHECKPOINT_DIR, "selfie_teacher_b.pth"),
        "PhysPrior (cons)": os.path.join(CHECKPOINT_DIR, "physprior_conservative_best.pth"),
        "PhysPrior (mod)": os.path.join(CHECKPOINT_DIR, "physprior_moderate_best.pth"),
        "PhysPrior (agg)": os.path.join(CHECKPOINT_DIR, "physprior_aggressive_best.pth"),
        "Mixup": os.path.join(CHECKPOINT_DIR, "mixup_best.pth"),
    }

    results = {}
    print("\n" + "=" * 75)
    print("  Four-Layer Evaluation Results")
    print("=" * 75)

    # Header
    print(f"\n  Layer 1 — Radar Accuracy (PRIMARY, unbiased)")
    print(f"  {'Method':<20s} {'RadarAcc':>9s} {'95% CI':>18s}  {'N':>5s}")
    print(f"  {'-'*56}")

    for name, ckpt_path in checkpoints.items():
        if not os.path.exists(ckpt_path):
            continue

        metrics = evaluate_full(ckpt_path, val_loader, device)
        results[name] = metrics

        ra = metrics["layer1_radar_accuracy"]
        ci_str = f"[{ra['ci_lower']:.4f}, {ra['ci_upper']:.4f}]"
        print(f"  {name:<20s} {ra['mean']:>9.4f} {ci_str:>18s}  {ra['n']:>5d}")

    # Layer 2 table
    print(f"\n  Layer 2 — Disagreement Rate (purification signal)")
    print(f"  {'Method':<20s} {'Disagree':>9s} {'+RadarAcc':>10s} {'Verdict':>15s}")
    print(f"  {'-'*58}")

    for name, metrics in results.items():
        dr = metrics["layer2_disagreement"]["overall_disagreement_rate"]
        ra = metrics["layer1_radar_accuracy"]["mean"]
        # Heuristic: high disagreement + high radar accuracy = purification
        if dr > 0.15 and ra > 0.85:
            verdict = "LIKELY PURIFIED"
        elif dr > 0.15 and ra <= 0.85:
            verdict = "DEGRADED"
        elif dr <= 0.15 and ra > 0.85:
            verdict = "CONSERVATIVE"
        else:
            verdict = "NO IMPROVEMENT"
        print(f"  {name:<20s} {dr:>9.4f} {ra:>10.4f} {verdict:>15s}")

    # Layer 4 (reference only)
    print(f"\n  Layer 4 — CLDMSK Metrics (REFERENCE ONLY — may penalize corrections)")
    print(f"  {'Method':<20s} {'CLDMSK-IoU':>11s} {'CLDMSK-F1':>11s}")
    print(f"  {'-'*45}")
    for name, metrics in results.items():
        c = metrics["layer4_cldmsk_reference"]
        print(f"  {name:<20s} {c['cldmsk_iou_REFERENCE_ONLY']:>11.4f} "
              f"{c['cldmsk_f1_REFERENCE_ONLY']:>11.4f}")

    print("=" * 75)

    # Save full results
    # Strip audit samples from saved JSON (too large)
    save_results = {}
    for name, metrics in results.items():
        save_results[name] = {
            "layer1_radar_accuracy": metrics["layer1_radar_accuracy"],
            "layer1_radar_per_class": metrics["layer1_radar_per_class"],
            "layer2_disagreement": metrics["layer2_disagreement"],
            "layer4_cldmsk_reference": metrics["layer4_cldmsk_reference"],
        }

    out_path = os.path.join(OUTPUT_DIR, "comparison_four_layer.json")
    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n  Full results saved to {out_path}")

    # Save audit samples separately
    audit_dir = os.path.join(OUTPUT_DIR, "audit")
    os.makedirs(audit_dir, exist_ok=True)
    for name, metrics in results.items():
        audit_path = os.path.join(audit_dir, f"{name.replace(' ', '_')}_audit.json")
        with open(audit_path, "w") as f:
            json.dump(metrics["layer3_audit_samples"], f, indent=2)
    print(f"  Audit samples saved to {audit_dir}/")

    return results


if __name__ == "__main__":
    compare_all()
