"""
GLOBAL STANDARDIZED TOPOLOGICAL & CONNECTIVITY TEST SUITE
=========================================================
Applies standard peer-reviewed connectivity metrics across ALL models:
- Baseline Reproduction (SA-UNetv2)
- Idea 1 (clDice Loss)
- Future Ideas (ACE-Gate, Prototypes, etc.)

Metrics Computed:
1. clDice (Centerline Dice Index - CVPR 2021)
2. Topology Precision (Tprec) & Topology Sensitivity (Tsens)
3. Connected Components (Betti-0, beta_0)
4. Absolute Disconnection Error (|beta_0_pred - beta_0_gt|)
5. Vascular Tree Fragmentation Ratio (beta_0_pred / beta_0_gt)
6. Largest Connected Component Ratio (LCCR)
"""
import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage as ndi


def find_path(rel_path):
    """Robust path finder checking current dir, parent, and grandparent."""
    if os.path.exists(rel_path):
        return rel_path
    p1 = os.path.join("..", rel_path)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join("..", "..", rel_path)
    if os.path.exists(p2):
        return p2
    return rel_path


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

    # 2. Connected Component Analysis (Betti-0 with 8-connectivity)
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


def evaluate_directory(pred_dir, gt_dir, dataset_name="DRIVE"):
    pred_dir = find_path(pred_dir)
    gt_dir = find_path(gt_dir)

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred.png")))
    if not pred_files:
        print(f"[-] No prediction files found in: {pred_dir}")
        return None

    metrics = []
    for pf in pred_files:
        basename = os.path.basename(pf).replace('_pred.png', '')
        
        if dataset_name.upper() == "DRIVE":
            num = basename.split('_')[0]
            gt_path = os.path.join(gt_dir, f"{num}_manual1.png")
            if not os.path.exists(gt_path):
                gt_path = os.path.join(gt_dir, f"{num}_manual1.gif")
        else: # STARE
            gt_path = os.path.join(gt_dir, f"{basename}.ppm")

        if not os.path.exists(gt_path):
            continue

        pred_img = np.array(Image.open(pf)) > 127
        gt_img = np.array(Image.open(gt_path)) > 0
        if pred_img.shape != gt_img.shape:
            pred_img = pred_img[:gt_img.shape[0], :gt_img.shape[1]]

        m = compute_sample_connectivity(pred_img, gt_img)
        metrics.append(m)

    if not metrics:
        return None

    return {
        'cldice': np.mean([m['cldice'] for m in metrics]) * 100,
        'tprec': np.mean([m['tprec'] for m in metrics]) * 100,
        'tsens': np.mean([m['tsens'] for m in metrics]) * 100,
        'b0_gt': np.mean([m['b0_gt'] for m in metrics]),
        'b0_pred': np.mean([m['b0_pred'] for m in metrics]),
        'b0_err': np.mean([m['b0_err'] for m in metrics]),
        'frag_ratio': np.mean([m['frag_ratio'] for m in metrics]),
        'lcc_gt': np.mean([m['lcc_gt'] for m in metrics]) * 100,
        'lcc_pred': np.mean([m['lcc_pred'] for m in metrics]) * 100,
        'count': len(metrics)
    }


def print_single_report(res, title, dataset_name):
    print("=" * 75)
    print(f" {title.upper()} - {dataset_name} ({res['count']} Images)")
    print("=" * 75)
    print(f"{'Topological Metric':<35} | {'Ground Truth':<16} | {'Model Prediction':<16}")
    print("-" * 75)
    print(f"{'Centerline Dice (clDice)':<35} | {'-':<16} | {res['cldice']:6.2f}%")
    print(f"{'Topology Precision (Tprec)':<35} | {'-':<16} | {res['tprec']:6.2f}%")
    print(f"{'Topology Sensitivity (Tsens)':<35} | {'-':<16} | {res['tsens']:6.2f}%")
    print(f"{'Connected Components (Betti-0)':<35} | {res['b0_gt']:6.1f}           | {res['b0_pred']:6.1f}")
    print(f"{'Absolute Betti-0 Error':<35} | {'0.0':<16} | {res['b0_err']:6.1f}")
    print(f"{'Vascular Fragmentation Ratio':<35} | {'1.00x':<16} | {res['frag_ratio']:6.2f}x")
    print(f"{'Largest Connected Component (LCCR)':<35} | {res['lcc_gt']:6.2f}%          | {res['lcc_pred']:6.2f}%")
    print("=" * 75)


def print_comparison_report(res_base, res_idea, idea_name, dataset_name):
    print("=" * 82)
    print(f" TOPOLOGICAL COMPARISON: BASELINE vs. {idea_name.upper()} ({dataset_name})")
    print("=" * 82)
    print(f"{'Topological Metric':<30} | {'Baseline':<12} | {idea_name:<12} | {'Change / Delta':<16}")
    print("-" * 82)
    
    def format_delta(val_idea, val_base, is_higher_better=True, unit="%"):
        delta = val_idea - val_base
        sign = "+" if delta > 0 else ""
        if (delta > 0 and is_higher_better) or (delta < 0 and not is_higher_better):
            status = "improved"
        else:
            status = "worse"
        return f"{sign}{delta:5.2f}{unit} ({status})"

    print(f"{'Centerline Dice (clDice)':<30} | {res_base['cldice']:5.2f}%     | {res_idea['cldice']:5.2f}%     | {format_delta(res_idea['cldice'], res_base['cldice'], True)}")
    print(f"{'Topology Precision (Tprec)':<30} | {res_base['tprec']:5.2f}%     | {res_idea['tprec']:5.2f}%     | {format_delta(res_idea['tprec'], res_base['tprec'], True)}")
    print(f"{'Topology Sensitivity (Tsens)':<30} | {res_base['tsens']:5.2f}%     | {res_idea['tsens']:5.2f}%     | {format_delta(res_idea['tsens'], res_base['tsens'], True)}")
    print(f"{'Connected Components (Betti-0)':<30} | {res_base['b0_pred']:5.1f}      | {res_idea['b0_pred']:5.1f}      | {format_delta(res_idea['b0_pred'], res_base['b0_pred'], False, '')}")
    print(f"{'Disconnection Error (Betti-0)':<30} | {res_base['b0_err']:5.1f}      | {res_idea['b0_err']:5.1f}      | {format_delta(res_idea['b0_err'], res_base['b0_err'], False, '')}")
    print(f"{'Fragmentation Ratio':<30} | {res_base['frag_ratio']:5.2f}x     | {res_idea['frag_ratio']:5.2f}x     | {format_delta(res_idea['frag_ratio'], res_base['frag_ratio'], False, 'x')}")
    print(f"{'Largest Connected Tree (LCCR)':<30} | {res_base['lcc_pred']:5.2f}%     | {res_idea['lcc_pred']:5.2f}%     | {format_delta(res_idea['lcc_pred'], res_base['lcc_pred'], True)}")
    print("=" * 82)


TARGET_DIRS = {
    'baseline': {
        'drive': 'results/predictions',
        'stare': 'results/stare/predictions'
    },
    'idea1': {
        'drive': 'edits/idea1_cldice/results/predictions',
        'stare': 'edits/idea1_cldice/results/stare/predictions'
    }
}


def main():
    parser = argparse.ArgumentParser(description="Global Topological & Connectivity Test Suite")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "idea1"], 
                        help="Target model to evaluate")
    parser.add_argument("--compare", type=str, default=None, choices=["idea1"],
                        help="Compare baseline against specified idea")
    parser.add_argument("--dataset", type=str, default="both", choices=["drive", "stare", "both"])
    args = parser.parse_args()

    gt_drive = "Drive/DRIVE/test/1st_manual"
    gt_stare = "Stare data/benchmark_20/labels"

    if args.compare:
        idea_key = args.compare
        if args.dataset in ["drive", "both"]:
            base_drive = evaluate_directory(TARGET_DIRS['baseline']['drive'], gt_drive, "DRIVE")
            idea_drive = evaluate_directory(TARGET_DIRS[idea_key]['drive'], gt_drive, "DRIVE")
            if base_drive and idea_drive:
                print_comparison_report(base_drive, idea_drive, idea_key, "DRIVE")
            else:
                print("[-] Ensure predictions exist for both baseline and target before comparing.")

        if args.dataset in ["stare", "both"]:
            base_stare = evaluate_directory(TARGET_DIRS['baseline']['stare'], gt_stare, "STARE")
            idea_stare = evaluate_directory(TARGET_DIRS[idea_key]['stare'], gt_stare, "STARE")
            if base_stare and idea_stare:
                print_comparison_report(base_stare, idea_stare, idea_key, "STARE")
        return

    # Single model report
    target_key = args.model
    if args.dataset in ["drive", "both"]:
        res_d = evaluate_directory(TARGET_DIRS[target_key]['drive'], gt_drive, "DRIVE")
        if res_d:
            print_single_report(res_d, f"{target_key} Connectivity", "DRIVE")

    if args.dataset in ["stare", "both"]:
        res_s = evaluate_directory(TARGET_DIRS[target_key]['stare'], gt_stare, "STARE")
        if res_s:
            print_single_report(res_s, f"{target_key} Connectivity", "STARE")


if __name__ == "__main__":
    main()
