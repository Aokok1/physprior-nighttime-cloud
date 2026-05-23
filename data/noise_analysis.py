"""Analyze CLDMSK label noise using radar ground truth as clean reference.

This is the core analysis that transforms CLDMSK from "bad baseline" to
"usable weak supervision": by quantifying exactly how CLDMSK is wrong,
we can correct it.

Key outputs:
  1. Noise transition matrix:  P(CLDMSK=j | Radar=i)
     For the 4 CLDMSK classes and 2 radar classes (clear/cloud)
  2. Per-class noise rates
  3. Confidence calibration analysis
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

from config import data_cfg, noise_cfg, MTUNET_DATASET
from data.dataset import CloudDataset


class CLDMSKNoiseAnalyzer:
    """Estimate noise patterns in CLDMSK labels using radar ground truth.

    For each radar-hit pixel, we know:
      - Radar truth:   center_label (0=clear, 1=cloud)
      - CLDMSK label:  Y_mask[ry, rx] ∈ {0,1,2,3}

    This gives us a 2×4 confusion matrix: P(CLDMSK class | radar binary).
    """

    def __init__(self, dataset_root: str = MTUNET_DATASET):
        self.dataset_root = dataset_root
        self.num_classes = data_cfg.num_classes
        # Counts: [radar_class=2, cldmsk_class=4]
        self.counts = np.zeros((2, self.num_classes), dtype=np.int64)
        self.total_radar = 0
        self.total_clean = 0  # samples with valid center_label

    def scan(self) -> dict:
        """Scan all samples and accumulate radar-vs-CLDMSK statistics.

        Returns summary dict with noise matrix and per-class metrics.
        """
        modes = ["Train", "Test"]
        for mode in modes:
            ds = CloudDataset(self.dataset_root, mode, augment=False)
            for i in range(len(ds)):
                sample = ds[i]
                center = sample["center_label"]
                if np.isnan(center):
                    continue

                self.total_clean += 1
                radar_class = int(center)                   # 0 or 1
                cldmsk_class = int(sample["mask_noisy"][
                    sample["radar_loc"][0], sample["radar_loc"][1]
                ])
                radar_class = min(radar_class, 1)            # clamp
                self.counts[radar_class, cldmsk_class] += 1
                self.total_radar += 1

        return self._summarize()

    def _summarize(self) -> dict:
        """Compute noise metrics from accumulated counts."""
        if self.total_radar == 0:
            return {"error": "No radar samples found"}

        # Noise transition matrix: P(CLDMSK | Radar)
        T = np.zeros((2, self.num_classes))
        for r in range(2):
            row_sum = self.counts[r].sum()
            if row_sum > 0:
                T[r] = self.counts[r] / row_sum

        # Bootstrap CI for transition matrix entries and overall accuracy
        T_ci_lower, T_ci_upper, reliable = self._bootstrap_ci(n_bootstrap=2000)

        # Key noise metrics
        clear_actual = self.counts[0, 0] / max(self.counts[:, 0].sum(), 1)
        cloud_actual = self.counts[1, 3] / max(self.counts[:, 3].sum(), 1)

        correct = self.counts[0, 0] + self.counts[0, 1] + \
                  self.counts[1, 2] + self.counts[1, 3]
        accuracy = correct / max(self.total_radar, 1)

        fp_rate = self.counts[0, 2:].sum() / max(self.counts[0].sum(), 1)
        fn_rate = self.counts[1, :2].sum() / max(self.counts[1].sum(), 1)

        # CI for overall accuracy
        pixels = []
        for r in range(2):
            for c in range(self.num_classes):
                is_correct = (r == 0 and c in (0, 1)) or (r == 1 and c in (2, 3))
                pixels.extend([int(is_correct)] * int(self.counts[r, c]))
        pixels = np.array(pixels)
        acc_ci = self._bootstrap_accuracy_ci(pixels)

        summary = {
            "cldmsk_accuracy_radar_ci": acc_ci,
            "total_radar_pixels": int(self.total_radar),
            "total_clean_samples": int(self.total_clean),
            "counts": self.counts.tolist(),
            "transition_matrix": T.tolist(),
            "transition_matrix_ci_lower": T_ci_lower.tolist(),
            "transition_matrix_ci_upper": T_ci_upper.tolist(),
            "reliable_classes": reliable.tolist(),
            "cldmsk_accuracy_radar": round(accuracy, 4),
            "cldmsk_accuracy_radar_ci": acc_ci,
            "cldmsk_false_positive_rate": round(fp_rate, 4),
            "cldmsk_false_negative_rate": round(fn_rate, 4),
            "true_clear_purity": round(clear_actual, 4),
            "true_cloud_purity": round(cloud_actual, 4),
        }

        # Per-CLDMSK-class statistics
        class_names = ["True Clear", "prob_clear", "prob_cloud", "True Cloud"]
        for c in range(self.num_classes):
            total_c = int(self.counts[:, c].sum())
            if total_c > 0:
                n_cloud = int(self.counts[1, c])
                summary[f"class_{c}_{class_names[c]}_total"] = total_c
                summary[f"class_{c}_cloud_fraction"] = round(n_cloud / total_c, 4)
                summary[f"class_{c}_reliable"] = bool(reliable[c])

        return summary

    def _bootstrap_ci(self, n_bootstrap=2000, alpha=0.05):
        """Bootstrap 95% CI for transition matrix entries.

        Returns:
            T_lower: (2, 4) lower bounds
            T_upper: (2, 4) upper bounds
            reliable: (4,) bool — True if enough samples for this class
        """
        # Flatten all radar pixels as (radar_class, cldmsk_class) pairs
        pixel_pairs = []
        for r in range(2):
            for c in range(self.num_classes):
                pixel_pairs.extend([(r, c)] * int(self.counts[r, c]))
        n = len(pixel_pairs)
        if n == 0:
            zero = np.zeros((2, self.num_classes))
            return zero, zero, np.zeros(self.num_classes, dtype=bool)

        T_bootstrap = np.zeros((n_bootstrap, 2, self.num_classes))
        for b in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            counts_b = np.zeros((2, self.num_classes))
            for i in idx:
                r, c = pixel_pairs[i]
                counts_b[r, c] += 1
            for r in range(2):
                row_sum = counts_b[r].sum()
                if row_sum > 0:
                    T_bootstrap[b, r] = counts_b[r] / row_sum

        T_lower = np.percentile(T_bootstrap, alpha / 2 * 100, axis=0)
        T_upper = np.percentile(T_bootstrap, (1 - alpha / 2) * 100, axis=0)

        # Reliable: each column has >= min_samples_per_class total
        min_s = noise_cfg.min_samples_per_class
        reliable = self.counts.sum(axis=0) >= min_s

        return T_lower, T_upper, reliable

    def _bootstrap_accuracy_ci(self, corrects, n_bootstrap=2000, alpha=0.05):
        """Bootstrap CI for overall CLDMSK accuracy at radar pixels."""
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

    def get_loss_correction_matrix(self, radar_confidence: float = 0.9) -> np.ndarray:
        """Build 4×4 weighted loss correction matrix C where C[i,j] ≈ P(true=i | noisy=j).

        Radar gives binary (clear/cloud), CLDMSK has 4 classes.
        Weight strategy: when radar says "clear", mass is weighted toward the
        extreme classes (True Clear over prob_clear) because radar is a physical
        measurement with high confidence.

        For each CLDMSK class j:
          frac_cloud = fraction of radar hits at class j that are actually cloud
          C[0,j] = (1 - frac_cloud) * radar_confidence       (True Clear)
          C[1,j] = (1 - frac_cloud) * (1 - radar_confidence)  (prob_clear)
          C[2,j] = frac_cloud * (1 - radar_confidence)        (prob_cloud)
          C[3,j] = frac_cloud * radar_confidence              (True Cloud)

        radar_confidence=0.9: 90% mass to extreme classes (True Clear/True Cloud).
        Physical basis: Ka-band radar cloud detection is a direct measurement,
        with very low false detection rate.
        """
        C = np.zeros((self.num_classes, self.num_classes))

        for c in range(self.num_classes):
            total_c = self.counts[:, c].sum()
            if total_c == 0:
                C[c, c] = 1.0
                continue

            frac_cloud = self.counts[1, c] / total_c

            C[0, c] = (1.0 - frac_cloud) * radar_confidence
            C[1, c] = (1.0 - frac_cloud) * (1.0 - radar_confidence)
            C[2, c] = frac_cloud * (1.0 - radar_confidence)
            C[3, c] = frac_cloud * radar_confidence

        return C


def print_noise_report(analyzer: CLDMSKNoiseAnalyzer):
    """Pretty-print the noise analysis report."""
    summary = analyzer.scan()

    if "error" in summary:
        print(f"[ERROR] {summary['error']}")
        return

    print("\n" + "=" * 65)
    print("  CLDMSK Noise Analysis Report (Radar as Clean Reference)")
    print("=" * 65)
    print(f"  Radar-validated pixels: {summary['total_radar_pixels']}")
    print(f"  Samples with clean label: {summary['total_clean_samples']}")
    print(f"\n  CLDMSK Accuracy at Radar Pixels: {summary['cldmsk_accuracy_radar']:.2%}")
    print(f"  False Positive Rate (say cloud, is clear): {summary['cldmsk_false_positive_rate']:.2%}")
    print(f"  False Negative Rate (say clear, is cloud): {summary['cldmsk_false_negative_rate']:.2%}")

    print(f"\n  Transition Matrix P(CLDMSK | Radar) with 95% Bootstrap CI:")
    print(f"                   TrueClear        prob_clear       prob_cloud       TrueCloud")
    for r, rname in enumerate(["Radar Clear  ", "Radar Cloud  "]):
        row = ""
        for c in range(4):
            v = summary['transition_matrix'][r][c]
            lo = summary['transition_matrix_ci_lower'][r][c]
            hi = summary['transition_matrix_ci_upper'][r][c]
            row += f"{v:.3f} [{lo:.3f},{hi:.3f}]  "
        print(f"  {rname}  {row}")

    # Mark unreliable classes
    unreliable = [c for c in range(4) if not summary.get(f"class_{c}_reliable", True)]
    if unreliable:
        names = ["True Clear", "prob_clear", "prob_cloud", "True Cloud"]
        print(f"\n  [WARN] Unreliable classes (<{noise_cfg.min_samples_per_class} radar pixels):")
        for c in unreliable:
            print(f"    - {names[c]}: {summary.get(f'class_{c}_{names[c]}_total', '?')} radar pixels")

    print(f"\n  Per-Class Purity:")
    class_names = ["True Clear", "prob_clear", "prob_cloud", "True Cloud"]
    for c in range(4):
        key = f"class_{c}_cloud_fraction"
        if key in summary:
            print(f"    {class_names[c]:12s}: {summary[key]:.1%} actually cloud")

    print("=" * 65 + "\n")
    return summary


# ── Moon-Phase-Dependent Noise Analysis (P2-7) ──────────────────────────
MOON_BINS = [
    (0, 60, "new_crescent"),
    (60, 120, "first_quarter"),
    (120, 180, "waxing_gibbous"),
    (180, 240, "full_waning_gibbous"),
    (240, 300, "last_quarter"),
    (300, 360, "waning_crescent"),
]


def should_use_moon_segmentation(analyzer: CLDMSKNoiseAnalyzer,
                                  threshold: float = 0.10) -> bool:
    """Check if moon-dependent noise analysis is warranted.

    Scans dark (0-90°, 270-360°) vs bright (90-270°) phases.
    If CLDMSK accuracy differs by >threshold, recommend segmentation.
    """
    ds = CloudDataset(MTUNET_DATASET, "Train", augment=False)
    ds2 = CloudDataset(MTUNET_DATASET, "Test", augment=False)

    dark_correct = dark_total = 0
    bright_correct = bright_total = 0

    for dataset in [ds, ds2]:
        for i in range(len(dataset)):
            sample = dataset[i]
            center = sample["center_label"]
            if np.isnan(center):
                continue
            moon = sample["moon_phase"]
            ry, rx = int(sample["radar_loc"][0]), int(sample["radar_loc"][1])
            cldmsk_cls = int(sample["mask_noisy"][ry, rx])
            radar_cls = int(center)

            # Dark: 0-90 or 270-360; Bright: 90-270
            is_dark = (moon < 90) or (moon > 270)
            is_correct = ((radar_cls == 0 and cldmsk_cls < 2) or
                          (radar_cls == 1 and cldmsk_cls >= 2))

            if is_dark:
                dark_total += 1
                dark_correct += int(is_correct)
            else:
                bright_total += 1
                bright_correct += int(is_correct)

    if dark_total < 20 or bright_total < 20:
        return False

    dark_acc = dark_correct / max(dark_total, 1)
    bright_acc = bright_correct / max(bright_total, 1)
    diff = abs(dark_acc - bright_acc)

    print(f"  Moon-phase check: dark_acc={dark_acc:.2%} (n={dark_total}), "
          f"bright_acc={bright_acc:.2%} (n={bright_total}), diff={diff:.2%}")
    print(f"  Threshold={threshold:.2%} → "
          f"{'ENABLE' if diff > threshold else 'SKIP'} moon segmentation")

    return diff > threshold


# ── Quick test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = CLDMSKNoiseAnalyzer()
    print_noise_report(analyzer)
    should_use_moon_segmentation(analyzer)
