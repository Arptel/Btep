"""
Connectivity & Topological Evaluation Script for Retinal Vessel Segmentation.
Computes peer-reviewed topological metrics:
1. clDice (Centerline Dice, CVPR 2021)
2. Topology Precision (Tprec) & Topology Sensitivity (Tsens)
3. Betti-0 Count (Connected Components) & Absolute Betti-0 Error
4. Vascular Tree Fragmentation Ratio (Pred CC / GT CC)
5. Largest Connected Component Ratio (LCCR)
"""
import os
import glob
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage as ndi


def compute_sample_connectivity(pred_mask, gt_mask):
    """
    Computes clDice, Betti-0, Fragmentation, and LCCR for a single pair of binary masks.
    pred_mask: 2D boolean numpy array
    gt_mask: 2D boolean numpy array
    """
    # 1. Morphological Skeletonization for clDice
    s_pred = morph.skeletonize(pred_mask)
    s_gt = morph.skeletonize(gt_mask)

    t_prec = np.sum(s_pred & gt_mask) / (np.sum(s_pred) + 1e-8)
    t_sens = np.sum(s_gt & pred_mask) / (np.sum(s_gt) + 1e-8)
    cldice = 2.0 * t_prec * t_sens / (t_prec + t_sens + 1e-8)

    # 2. Connected Component Analysis (Betti-0 with 8-neighborhood)
    structure_8 = np.ones((3, 3), dtype=int)
    lbl_pred, n_pred = ndi.label(pred_mask, structure=structure_8)
    lbl_gt, n_gt = ndi.label(gt_mask, structure=structure_8)

    b0_err = abs(n_pred - n_gt)
    frag_ratio = n_pred / (n_gt + 1e-8)

    # 3. Largest Connected Component Ratio (LCCR)
    if n_pred > 0:
        sizes_pred = ndi.sum(pred_mask, lbl_pred, range(1, n_pred + 1))
        lcc_pred = np.max(sizes_pred) / (np.sum(pred_mask) + 1e-8)
    else:
        lcc_pred = 0.0

    if n_gt > 0:
        sizes_gt = ndi.sum(gt_mask, lbl_gt, range(1, n_gt + 1))
        lcc_gt = np.max(sizes_gt) / (np.sum(gt_mask) + 1e-8)
    else:
        lcc_gt = 0.0

    return {
        'cldice': float(cldice),
        'tprec': float(t_prec),
        'tsens': float(t_sens),
        'b0_gt': int(n_gt),
        'b0_pred': int(n_pred),
        'b0_err': int(b0_err),
        'frag_ratio': float(frag_ratio),
        'lcc_gt': float(lcc_gt),
        'lcc_pred': float(lcc_pred)
    }


def evaluate_dataset_connectivity(dataset_name="DRIVE", 
                                  pred_dir="results/predictions", 
                                  gt_dir="Drive/DRIVE/test/1st_manual"):
    # Fallback to parent path if run inside baseline_reproduction
    if not os.path.exists(pred_dir):
        alt_pred = os.path.join("..", pred_dir)
        if os.path.exists(alt_pred):
            pred_dir = alt_pred

    if not os.path.exists(gt_dir):
        alt_gt = os.path.join("..", gt_dir)
        if os.path.exists(alt_gt):
            gt_dir = alt_gt

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.png")))
    if not pred_files:
        print(f"Error: No prediction files found in '{pred_dir}'!")
        return

    print("=" * 75)
    print(f" TOPOLOGICAL CONNECTIVITY EVALUATION: {dataset_name} ({len(pred_files)} Images)")
    print("=" * 75)

    metrics = []
    for pf in pred_files:
        basename = os.path.basename(pf).replace('_pred.png', '')
        
        # Resolve ground truth path
        if dataset_name.upper() == "DRIVE":
            num = basename.split('_')[0]
            gt_path = os.path.join(gt_dir, f"{num}_manual1.png")
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_dir, f"{num}_manual1.gif")
        else: # STARE
            gt_path = os.path.join(gt_dir, f"{basename}.ppm")

        if not os.path.exists(gt_path):
            print(f"Warning: GT not found for '{pf}' at '{gt_path}'")
            continue

        pred_img = np.array(Image.open(pf)) > 127
        gt_img = np.array(Image.open(gt_path)) > 0
        if pred_img.shape != gt_img.shape:
            pred_img = pred_img[:gt_img.shape[0], :gt_img.shape[1]]

        m = compute_sample_connectivity(pred_img, gt_img)
        metrics.append(m)

    # Compute Averages
    cldice_mean = np.mean([m['cldice'] for m in metrics]) * 100
    tprec_mean = np.mean([m['tprec'] for m in metrics]) * 100
    tsens_mean = np.mean([m['tsens'] for m in metrics]) * 100
    b0_gt_mean = np.mean([m['b0_gt'] for m in metrics])
    b0_pred_mean = np.mean([m['b0_pred'] for m in metrics])
    b0_err_mean = np.mean([m['b0_err'] for m in metrics])
    frag_mean = np.mean([m['frag_ratio'] for m in metrics])
    lcc_gt_mean = np.mean([m['lcc_gt'] for m in metrics]) * 100
    lcc_pred_mean = np.mean([m['lcc_pred'] for m in metrics]) * 100

    print(f"{'Topological Metric':<35} | {'Ground Truth':<16} | {'Model Prediction':<16}")
    print("-" * 75)
    print(f"{'Centerline Dice (clDice)':<35} | {'-':<16} | {cldice_mean:6.2f}%")
    print(f"{'Topology Precision (Tprec)':<35} | {'-':<16} | {tprec_mean:6.2f}%")
    print(f"{'Topology Sensitivity (Tsens)':<35} | {'-':<16} | {tsens_mean:6.2f}%")
    print(f"{'Connected Components (Betti-0)':<35} | {b0_gt_mean:6.1f}           | {b0_pred_mean:6.1f}")
    print(f"{'Absolute Betti-0 Error (Disconnections)':<35} | {'0.0':<16} | {b0_err_mean:6.1f}")
    print(f"{'Vascular Fragmentation Ratio':<35} | {'1.00x':<16} | {frag_mean:6.2f}x")
    print(f"{'Largest Connected Component (LCCR)':<35} | {lcc_gt_mean:6.2f}%          | {lcc_pred_mean:6.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    print("\n--- [1] DRIVE TEST SET CONNECTIVITY ---")
    evaluate_dataset_connectivity(dataset_name="DRIVE", 
                                  pred_dir="results/predictions", 
                                  gt_dir="Drive/DRIVE/test/1st_manual")

    print("\n--- [2] STARE TEST SET CONNECTIVITY ---")
    evaluate_dataset_connectivity(dataset_name="STARE", 
                                  pred_dir="results/stare/predictions", 
                                  gt_dir="Stare data/benchmark_20/labels")
