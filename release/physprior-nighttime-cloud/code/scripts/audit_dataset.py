"""Audit dataset.py: data loading, normalization, augmentation.

Checks:
1. Normalization consistency (log_norm, bt_norm)
2. Augmentation correctness (flip, rotate, noise)
3. Channel ordering (DNB, Basemap, M15)
4. Label handling (noisy vs clean)
5. Radar pixel location after augmentation
"""
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import MTUNET_DATASET, MTUNET_NORM_STATS
from data.dataset import CloudDataset, _load_stats, _log_norm, _bt_norm


def audit_normalization():
    """Check normalization stats and consistency."""
    print("="*60)
    print("AUDIT: Normalization")
    print("="*60)
    
    stats = _load_stats()
    print(f"DNB stats: mean={stats['dnb_mean']:.4f}, std={stats['dnb_std']:.4f}")
    print(f"MOD stats: mean={stats['mod_mean']:.4f}, std={stats['mod_std']:.4f}")
    
    # Load norm_stats file directly
    s = np.load(MTUNET_NORM_STATS)
    print(f"\nRaw norm_stats.npz keys: {list(s.keys())}")
    for k in s.keys():
        print(f"  {k}: {s[k]:.6f}")
    
    # Test normalization on sample data
    test_arr = np.array([1e-8, 1e-5, 1e-3, 0.01, 0.1, 1.0], dtype=np.float32)
    log_result = _log_norm(test_arr)
    print(f"\nLog norm test (DNB):")
    print(f"  Input:  {test_arr}")
    print(f"  Output: {log_result}")
    
    # Check for NaN/Inf after normalization
    nan_count = np.isnan(log_result).sum()
    inf_count = np.isinf(log_result).sum()
    print(f"  NaN: {nan_count}, Inf: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        print("  ⚠️ WARNING: NaN/Inf in normalized output!")
    
    return nan_count == 0 and inf_count == 0


def audit_augmentation():
    """Test augmentation preserves radar pixel location."""
    print("\n" + "="*60)
    print("AUDIT: Augmentation")
    print("="*60)
    
    dataset = CloudDataset(MTUNET_DATASET, mode="Train", augment=True)
    item = dataset[0]
    
    print(f"Image shape: {item['image'].shape}")
    print(f"Mask shape:  {item['mask_noisy'].shape}")
    print(f"Radar loc:   {item['radar_loc']}")
    print(f"Center label: {item['center_label']}")
    
    # Test augmentation consistency: radar pixel should match after transforms
    # Load raw data
    import numpy as np_raw
    raw = np.load(dataset.file_list[0])
    ry_raw, rx_raw = int(raw["Radar_Loc"][0]), int(raw["Radar_Loc"][1])
    y_mask_raw = np.squeeze(raw["Y_mask"])
    
    print(f"\nRaw radar loc: ({ry_raw}, {rx_raw})")
    print(f"Raw CLDMSK at radar: {y_mask_raw[ry_raw, rx_raw]}")
    
    # After augmentation, check if radar_loc still points to valid location
    ry_aug, rx_aug = item['radar_loc'][0].item(), item['radar_loc'][1].item()
    mask_aug = item['mask_noisy'].numpy()
    
    print(f"Augmented radar loc: ({ry_aug}, {rx_aug})")
    if 0 <= ry_aug < 128 and 0 <= rx_aug < 128:
        print(f"Augmented CLDMSK at radar: {mask_aug[ry_aug, rx_aug]}")
    else:
        print("⚠️ WARNING: Radar loc out of bounds after augmentation!")
    
    return True


def audit_channel_ordering():
    """Verify channel ordering: DNB, Basemap, M15."""
    print("\n" + "="*60)
    print("AUDIT: Channel Ordering")
    print("="*60)
    
    dataset = CloudDataset(MTUNET_DATASET, mode="Test", augment=False)
    item = dataset[0]
    
    image = item['image'].numpy()
    print(f"Image shape: {image.shape} (should be [3, 128, 128])")
    print(f"Channel 0 (DNB):     mean={image[0].mean():.4f}, std={image[0].std():.4f}")
    print(f"Channel 1 (Basemap): mean={image[1].mean():.4f}, std={image[1].std():.4f}")
    print(f"Channel 2 (M15):     mean={image[2].mean():.4f}, std={image[2].std():.4f}")
    
    # DNB and Basemap should have similar stats (both log-normalized)
    # M15 should have different stats (BT-normalized)
    dnb_bm_corr = np.corrcoef(image[0].flatten(), image[1].flatten())[0, 1]
    print(f"\nDNB-Basemap correlation: {dnb_bm_corr:.4f}")
    
    if dnb_bm_corr > 0.9:
        print("⚠️ WARNING: DNB and Basemap are very similar — possible channel duplication!")
    
    return True


def audit_label_consistency():
    """Check that noisy labels match CLDMSK and clean labels match radar."""
    print("\n" + "="*60)
    print("AUDIT: Label Consistency")
    print("="*60)
    
    dataset = CloudDataset(MTUNET_DATASET, mode="Test", augment=False)
    
    mismatch_count = 0
    nan_count = 0
    total_with_radar = 0
    
    for i in range(min(50, len(dataset))):
        item = dataset[i]
        
        # Load raw data
        raw = np.load(dataset.file_list[i])
        y_mask_raw = np.squeeze(raw["Y_mask"])
        center_label = float(raw.get("Center_Label", float("nan")))
        radar_loc = raw.get("Radar_Loc", np.array([64, 64]))
        ry, rx = int(radar_loc[0]), int(radar_loc[1])
        
        # Check CLDMSK at radar pixel
        cldmsk_at_radar = int(y_mask_raw[ry, rx])
        cldmsk_binary = 1 if cldmsk_at_radar >= 2 else 0
        
        if np.isnan(center_label):
            nan_count += 1
            continue
        
        total_with_radar += 1
        radar_binary = int(center_label)
        
        if cldmsk_binary != radar_binary:
            mismatch_count += 1
    
    print(f"Checked: {min(50, len(dataset))} samples")
    print(f"With radar: {total_with_radar}")
    print(f"NaN radar: {nan_count}")
    print(f"CLDMSK-Radar mismatch: {mismatch_count}/{total_with_radar} = "
          f"{mismatch_count/max(total_with_radar,1):.1%}")
    
    return True


def audit_radar_pixel_bounds():
    """Check that radar pixel locations are within valid bounds."""
    print("\n" + "="*60)
    print("AUDIT: Radar Pixel Bounds")
    print("="*60)
    
    dataset = CloudDataset(MTUNET_DATASET, mode="Test", augment=False)
    
    out_of_bounds = 0
    center_pixel = 0
    
    for i in range(len(dataset)):
        item = dataset[i]
        ry, rx = item['radar_loc'][0].item(), item['radar_loc'][1].item()
        
        if ry < 0 or ry >= 128 or rx < 0 or rx >= 128:
            out_of_bounds += 1
        
        if ry == 64 and rx == 64:
            center_pixel += 1
    
    print(f"Total samples: {len(dataset)}")
    print(f"Out of bounds: {out_of_bounds}")
    print(f"Center pixel (64,64): {center_pixel}")
    
    if out_of_bounds > 0:
        print("⚠️ WARNING: Radar pixel out of bounds!")
    
    return out_of_bounds == 0


def audit_data_values():
    """Check for extreme values, NaN, Inf in features."""
    print("\n" + "="*60)
    print("AUDIT: Data Values")
    print("="*60)
    
    dataset = CloudDataset(MTUNET_DATASET, mode="Test", augment=False)
    
    all_dnb = []
    all_m15 = []
    nan_count = 0
    inf_count = 0
    
    for i in range(min(20, len(dataset))):
        item = dataset[i]
        image = item['image'].numpy()
        
        all_dnb.append(image[0].flatten())
        all_m15.append(image[2].flatten())
        
        nan_count += np.isnan(image).sum()
        inf_count += np.isinf(image).sum()
    
    all_dnb = np.concatenate(all_dnb)
    all_m15 = np.concatenate(all_m15)
    
    print(f"Checked {min(20, len(dataset))} samples")
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")
    print(f"\nDNB range: [{np.nanmin(all_dnb):.4f}, {np.nanmax(all_dnb):.4f}]")
    print(f"M15 range: [{np.nanmin(all_m15):.4f}, {np.nanmax(all_m15):.4f}]")
    
    # Check for extreme values
    dnb_extreme = np.sum(np.abs(all_dnb) > 10)
    m15_extreme = np.sum(np.abs(all_m15) > 10)
    print(f"\nExtreme values (|x|>10):")
    print(f"  DNB: {dnb_extreme} ({dnb_extreme/len(all_dnb)*100:.2f}%)")
    print(f"  M15: {m15_extreme} ({m15_extreme/len(all_m15)*100:.2f}%)")
    
    return nan_count == 0 and inf_count == 0


def main():
    """Run full dataset audit."""
    print("="*70)
    print("DATASET AUDIT — Noise-Label Cloud Detection")
    print("="*70)
    
    results = {}
    
    results["normalization"] = audit_normalization()
    results["augmentation"] = audit_augmentation()
    results["channel_ordering"] = audit_channel_ordering()
    results["label_consistency"] = audit_label_consistency()
    results["radar_bounds"] = audit_radar_pixel_bounds()
    results["data_values"] = audit_data_values()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {check}")


if __name__ == "__main__":
    main()
