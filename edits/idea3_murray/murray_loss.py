"""
Murray's Law Bifurcation Loss & Prior Map Generation (Idea 3)
=============================================================
Formulates Murray's Law of minimum metabolic work (r0^3 = r1^3 + r2^3)
to penalize branch severance and junction pinching during training.
"""
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_bifurcations_2d(skeleton_np):
    """
    Detects vascular bifurcations (degree >= 3 branch points) on a 1-pixel skeleton.
    Returns:
        bif_coords: List of (y, x) coordinates for each bifurcation centroid.
        bif_mask: Binary 2D numpy array where bifurcation points are 1.
    """
    skel = (skeleton_np > 0).astype(np.uint8)
    if skel.sum() == 0:
        return [], np.zeros_like(skel)

    # 3x3 neighbor convolution
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = ndi.convolve(skel, kernel, mode='constant', cval=0) * skel
    raw_bif_mask = (neighbor_count >= 3)

    # Group contiguous junction pixels into single centroids
    labeled, num_features = ndi.label(raw_bif_mask, structure=ndi.generate_binary_structure(2, 2))
    bif_coords = []
    bif_mask = np.zeros_like(skel, dtype=np.float32)

    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        cy, cx = int(np.round(np.mean(ys))), int(np.round(np.mean(xs)))
        bif_coords.append((cy, cx))
        bif_mask[cy, cx] = 1.0

    return bif_coords, bif_mask


def compute_murray_junction_radii(bif_coords, skeleton_np, edt_np, search_dist=3):
    """
    Measures parent (r0) and daughter (r1, r2) radii around each bifurcation point.
    Returns:
        r0, r1, r2, murray_ratio, murray_deviation for each valid junction.
    """
    h, w = skeleton_np.shape
    skel = (skeleton_np > 0).astype(np.uint8)
    junction_data = []

    for cy, cx in bif_coords:
        y_min = max(0, cy - search_dist)
        y_max = min(h, cy + search_dist + 1)
        x_min = max(0, cx - search_dist)
        x_max = min(w, cx + search_dist + 1)

        patch_skel = skel[y_min:y_max, x_min:x_max]
        patch_edt = edt_np[y_min:y_max, x_min:x_max]

        # Sample radii on the skeleton perimeter of the search window
        py, px = np.where(patch_skel > 0)
        radii = []
        for y, x in zip(py, px):
            # Distance from center
            dist = np.hypot(y - (cy - y_min), x - (cx - x_min))
            if dist >= 1.5:  # away from junction core
                radii.append(float(patch_edt[y, x]))

        if len(radii) >= 2:
            radii = sorted(radii, reverse=True)
            r0 = max(1.0, radii[0])
            r1 = max(1.0, radii[1])
            r2 = max(1.0, radii[2]) if len(radii) >= 3 else max(0.5, r1 * 0.7)

            murray_ratio = (r1**3 + r2**3) / (r0**3 + 1e-6)
            deviation = abs(1.0 - murray_ratio)
            junction_data.append({
                'coord': (cy, cx),
                'r0': r0,
                'r1': r1,
                'r2': r2,
                'murray_ratio': murray_ratio,
                'deviation': deviation
            })

    return junction_data


def generate_murray_bifurcation_map(gt_mask_np, skeleton_np, edt_np):
    """
    Generates a spatial Gaussian bifurcation guidance map M_bif(x, y) in [0, 1].
    Points around genuine bifurcations have high weight, proportional to vessel caliber.
    """
    h, w = gt_mask_np.shape
    bif_coords, _ = detect_bifurcations_2d(skeleton_np)
    junction_data = compute_murray_junction_radii(bif_coords, skeleton_np, edt_np)

    bif_map = np.zeros((h, w), dtype=np.float32)
    for junc in junction_data:
        cy, cx = junc['coord']
        r0 = junc['r0']
        sigma = max(2.5, float(r0) * 1.2)
        radius_cutoff = int(np.ceil(3.0 * sigma))

        y_min = max(0, cy - radius_cutoff)
        y_max = min(h, cy + radius_cutoff + 1)
        x_min = max(0, cx - radius_cutoff)
        x_max = min(w, cx + radius_cutoff + 1)

        y_idx = np.arange(y_min, y_max)[:, None]
        x_idx = np.arange(x_min, x_max)[None, :]
        dist_sq = (y_idx - cy)**2 + (x_idx - cx)**2
        gaussian_patch = np.exp(-dist_sq / (2.0 * sigma**2))

        bif_map[y_min:y_max, x_min:x_max] = np.maximum(
            bif_map[y_min:y_max, x_min:x_max],
            gaussian_patch.astype(np.float32)
        )

    # Restrict guidance to vessel lumen neighborhood
    vessel_dilated = ndi.binary_dilation(gt_mask_np > 0, iterations=2)
    bif_map = bif_map * vessel_dilated.astype(np.float32)
    return bif_map.astype(np.float32)


class MurrayBifurcationLoss(nn.Module):
    """
    Murray's Law Bifurcation Regularization Loss:
    Penalizes deviations in predicted probability specifically within vascular bifurcation zones.
    """
    def __init__(self, eps=1e-7):
        super(MurrayBifurcationLoss, self).__init__()
        self.eps = eps

    def forward(self, pred_prob, target_mask, bif_map):
        """
        Args:
            pred_prob: [B, 1, H, W] in [0, 1]
            target_mask: [B, 1, H, W] in {0, 1}
            bif_map: [B, 1, H, W] in [0, 1] Gaussian bifurcation spatial map
        """
        sq_err = (pred_prob - target_mask) ** 2
        weighted_err = sq_err * bif_map
        norm = bif_map.sum(dim=(2, 3)) + self.eps
        loss_per_sample = weighted_err.sum(dim=(2, 3)) / norm
        return loss_per_sample.mean()
