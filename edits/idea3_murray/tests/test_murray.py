"""
Unit Verification Suite for Murray's Law Bifurcation Module (Idea 3)
===================================================================
Tests synthetic bifurcation detection, Murray ratio computation,
and differentiable loss backpropagation.
"""
import os
import sys
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(project_root, "edits", "idea3_murray"))

from murray_loss import (
    detect_bifurcations_2d,
    compute_murray_junction_radii,
    generate_murray_bifurcation_map,
    MurrayBifurcationLoss
)


def create_synthetic_y_junction(h=64, w=64):
    """Creates a synthetic vascular Y-junction with parent trunk and 2 daughters."""
    mask = np.zeros((h, w), dtype=np.float32)
    # Parent trunk (vertical, radius ~ 3)
    mask[32:60, 29:35] = 1.0
    # Left daughter (angled left, radius ~ 2)
    for i in range(20):
        mask[32 - i, max(0, 31 - i - 1):min(w, 31 - i + 2)] = 1.0
    # Right daughter (angled right, radius ~ 2)
    for i in range(20):
        mask[32 - i, max(0, 32 + i - 1):min(w, 32 + i + 2)] = 1.0

    skel = skeletonize(mask > 0).astype(np.uint8)
    edt = ndi.distance_transform_edt(mask > 0).astype(np.float32)
    return mask, skel, edt


def test_bifurcation_detection():
    print("[1/4] Testing synthetic Y-junction detection...")
    mask, skel, edt = create_synthetic_y_junction()
    coords, bif_mask = detect_bifurcations_2d(skel)
    assert len(coords) == 1, f"Expected exactly 1 bifurcation, got {len(coords)}"
    cy, cx = coords[0]
    assert 28 <= cy <= 36 and 28 <= cx <= 36, f"Bifurcation coordinates ({cy}, {cx}) out of expected range"
    print(f"  -> Passed! Detected 1 bifurcation at ({cy}, {cx}).")


def test_murray_radii_calculation():
    print("[2/4] Testing Murray's Law branching ratio calculation...")
    mask, skel, edt = create_synthetic_y_junction()
    coords, _ = detect_bifurcations_2d(skel)
    junc_data = compute_murray_junction_radii(coords, skel, edt)
    assert len(junc_data) == 1, f"Expected 1 valid junction analysis, got {len(junc_data)}"
    j = junc_data[0]
    r0, r1, r2 = j['r0'], j['r1'], j['r2']
    ratio = j['murray_ratio']
    print(f"  -> Measured Radii: r0={r0:.2f}, r1={r1:.2f}, r2={r2:.2f}")
    print(f"  -> Murray Ratio: (r1^3 + r2^3)/r0^3 = {ratio:.3f} (Deviation: {j['deviation']:.3f})")
    assert r0 >= r1 and r1 >= r2, "Parent trunk must be the largest vessel caliber"
    assert ratio > 0.0, "Murray ratio must be strictly positive"
    print("  -> Passed! Murray branching ratio correctly computed.")


def test_bifurcation_map_generation():
    print("[3/4] Testing continuous Murray bifurcation map generation...")
    mask, skel, edt = create_synthetic_y_junction()
    bif_map = generate_murray_bifurcation_map(mask, skel, edt)
    assert bif_map.shape == mask.shape, "Bifurcation map shape mismatch"
    assert bif_map.max() > 0.5, "Bifurcation map peak must be > 0.5 near junction"
    assert np.all(bif_map >= 0.0) and np.all(bif_map <= 1.0), "Map values must be in [0, 1]"
    print(f"  -> Passed! Bifurcation map generated with peak: {bif_map.max():.3f}.")


def test_murray_loss_autograd():
    print("[4/4] Testing differentiable MurrayBifurcationLoss autograd pass...")
    mask, skel, edt = create_synthetic_y_junction()
    bif_map = generate_murray_bifurcation_map(mask, skel, edt)

    pred = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # Add perturbance and require grad
    pred = (pred * 0.8 + 0.1).clone().detach().requires_grad_(True)
    target = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    bif_t = torch.tensor(bif_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    loss_fn = MurrayBifurcationLoss()
    loss = loss_fn(pred, target, bif_t)
    loss.backward()

    assert loss.item() > 0.0, "Loss must be positive under perturbation"
    assert pred.grad is not None, "Gradients must exist"
    assert pred.grad.abs().sum().item() > 0.0, "Gradients must be non-zero"
    print(f"  -> Murray Loss: {loss.item():.4f} | Gradient Sum: {pred.grad.abs().sum().item():.4f}")
    print("  -> Passed! Autograd backward flow fully verified.")


if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING MURRAY'S LAW BIFURCATION UNIT VERIFICATION SUITE")
    print("=" * 60)
    test_bifurcation_detection()
    test_murray_radii_calculation()
    test_bifurcation_map_generation()
    test_murray_loss_autograd()
    print("=" * 60)
    print("ALL 4 MURRAY'S LAW UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
