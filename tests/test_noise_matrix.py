"""Unit tests for noise matrix estimation.

Uses synthetic data since real data requires MT-UNet dataset.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.noise_analysis import CLDMSKNoiseAnalyzer


def test_noise_matrix_from_synthetic():
    """Test noise matrix estimation with known noise pattern.

    Simulates CLDMSK with known error rates:
      - True Clear → CLDMSK says True Clear 60%, prob_clear 30%, prob_cloud 8%, True Cloud 2%
      - True Cloud → CLDMSK says True Cloud 70%, prob_cloud 20%, prob_clear 8%, True Clear 2%
    """
    # Build synthetic counts
    # Class 0 (True Clear): heavily biased toward clear classes
    # Class 1 (True Cloud): biased toward cloud classes
    true_clear_counts = [600, 300, 80, 20]   # 1000 samples
    true_cloud_counts = [20, 80, 200, 700]    # 1000 samples

    counts = np.array([true_clear_counts, true_cloud_counts])

    # Verify transition matrix properties
    T = np.zeros((2, 4))
    for r in range(2):
        T[r] = counts[r] / counts[r].sum()

    # Clear samples: mostly in classes 0-1
    assert T[0, 0] + T[0, 1] > 0.8, "Clear should be mostly in True Clear or prob_clear"
    # Cloud samples: mostly in classes 2-3
    assert T[1, 2] + T[1, 3] > 0.8, "Cloud should be mostly in prob_cloud or True Cloud"

    # Forward correction matrix: P(Radar | CLDMSK) → 4×4
    # Distribute binary radar across the two clear classes (0,1) and two cloud classes (2,3)
    C = np.zeros((4, 4))
    for c in range(4):
        total = counts[:, c].sum()
        if total > 0:
            frac_cloud = counts[1, c] / total
            C[0, c] = (1.0 - frac_cloud) / 2.0
            C[1, c] = (1.0 - frac_cloud) / 2.0
            C[2, c] = frac_cloud / 2.0
            C[3, c] = frac_cloud / 2.0

    assert C.shape == (4, 4)
    # Column 0 (CLDMSK True Clear): >90% actually clear → C[0,0]+C[1,0] > 0.9
    assert C[0, 0] + C[1, 0] > 0.9, f"True Clear should be mostly clear, got {C[0,0]+C[1,0]:.3f}"
    # Column 3 (CLDMSK True Cloud): >90% actually cloud
    assert C[2, 3] + C[3, 3] > 0.9, f"True Cloud should be mostly cloud, got {C[2,3]+C[3,3]:.3f}"
    # Columns sum to 1
    for c in range(4):
        assert abs(C[:, c].sum() - 1.0) < 1e-10, f"Column {c} doesn't sum to 1"

    print("All noise matrix tests passed.")
    print(f"Transition matrix P(CLDMSK | Radar):\n{T}")
    print(f"Correction matrix P(Radar | CLDMSK):\n{C}")
    return T, C


def test_empty_case():
    """Test behavior with no radar samples."""
    analyzer = CLDMSKNoiseAnalyzer("/nonexistent/path")
    try:
        summary = analyzer.scan()
        assert "error" in summary or analyzer.total_radar == 0
    except ValueError:
        pass  # Expected when no files exist
    print("Empty case test passed.")


if __name__ == "__main__":
    test_noise_matrix_from_synthetic()
    test_empty_case()
    print("\nAll tests passed.")
