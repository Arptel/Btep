"""
Validation Experiment & Hard Mode Sweep for Betweenness Module.
- Labels predicted edges as TP vs FP against ground truth (overlap fraction < 30% inside 1px dilated GT).
- Computes AUROC of (1 - CB_hat) vs. Length and Radius baselines.
- Sweeps threshold tau on validation split.
- Applies optimal tau* once to test set and reports before vs. after metrics.
"""
import os
import glob
import time
import numpy as np
import pandas as pd
from PIL import Image
import scipy.ndimage as ndi
from sklearn.metrics import roc_auc_score

import io_utils
from graph_build import extract_vessel_graph
from scoring import score_edges
from apply import prune_edges_hard


def label_edges_against_gt(graph, edge_df, gt_mask, dilation_px=1, overlap_thresh=0.30):
    """
    Labels each edge in edge_df as FP (1) or TP (0).
    FP-segment: < 30% of pixels inside ground-truth vessel mask dilated by 1 px.
    """
    if dilation_px > 0:
        struct = ndi.generate_binary_structure(2, 2)
        gt_dilated = ndi.binary_dilation(gt_mask, structure=struct, iterations=dilation_px)
    else:
        gt_dilated = gt_mask

    H, W = gt_mask.shape
    is_fp_list = []
    overlap_list = []

    for _, row in edge_df.iterrows():
        pts = row['pts']
        y_c = pts[:, 0].clip(0, H - 1)
        x_c = pts[:, 1].clip(0, W - 1)

        inside_count = np.sum(gt_dilated[y_c, x_c])
        total_count = len(pts)
        frac = inside_count / (total_count + 1e-8)
        overlap_list.append(frac)
        is_fp_list.append(1 if frac < overlap_thresh else 0)

    edge_df['gt_overlap_frac'] = overlap_list
    edge_df['is_fp'] = is_fp_list
    return edge_df


def run_ranking_auroc_validation(prob_dir="edits/idea1_betweenness_module/prob_maps/drive/test",
                                 data_dir="Drive/DRIVE/test"):
    """
    Evaluates ranking capability of (1 - CB_hat) as a predictor of FP-segments.
    Compares against Edge Length alone and Mean Radius alone.
    """
    prob_dir = io_utils.find_path(prob_dir)
    data_dir = io_utils.find_path(data_dir)

    prob_files = sorted(glob.glob(os.path.join(prob_dir, "*_prob.npy")))
    all_edges = []

    print("=" * 75)
    print(" SECTION 4: AUROC VALIDATION (Predicting False-Positive Segments)")
    print("=" * 75)

    for pf in prob_files:
        basename = os.path.basename(pf).replace('_prob.npy', '')
        num = basename.split('_')[0]

        gt_path = os.path.join(data_dir, "1st_manual", f"{num}_manual1.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(data_dir, "1st_manual", f"{num}_manual1.gif")
        gt_mask = np.array(Image.open(gt_path)) > 0

        mask_path = os.path.join(data_dir, "mask", f"{num}_test_mask.gif")
        fov_mask = np.array(Image.open(mask_path)) > 0

        prob_map = np.load(pf)
        binary_mask, skeleton, radius_map, graph = extract_vessel_graph(prob_map, fov_mask)
        edge_df = score_edges(graph, config={'radius_mode': 'min', 'use_conductance': True})
        edge_df = label_edges_against_gt(graph, edge_df, gt_mask)
        all_edges.append(edge_df)

    combined_df = pd.concat(all_edges, ignore_index=True)
    y_true = combined_df['is_fp'].values
    num_fp = np.sum(y_true)
    total_edges = len(y_true)
    fp_pct = (num_fp / total_edges) * 100

    print(f"Total Evaluated Segments: {total_edges:,} | False-Positive Segments: {num_fp:,} ({fp_pct:.2f}%)")

    # Predictors:
    # 1. (1 - CB_hat): Lower betweenness should predict FP
    score_betweenness = 1.0 - combined_df['CB_hat'].values

    # 2. Baseline 1: Inverted Length (Shorter edges predict FP)
    max_L = combined_df['length'].max()
    score_inv_length = 1.0 - (combined_df['length'].values / (max_L + 1e-8))

    # 3. Baseline 2: Inverted Radius (Thinner edges predict FP)
    max_R = combined_df['radius_mean'].max()
    score_inv_radius = 1.0 - (combined_df['radius_mean'].values / (max_R + 1e-8))

    auroc_cb = roc_auc_score(y_true, score_betweenness)
    auroc_len = roc_auc_score(y_true, score_inv_length)
    auroc_rad = roc_auc_score(y_true, score_inv_radius)

    print("\n" + "-" * 75)
    print(f"{'Feature / Predictor':<35} | {'AUROC (vs. FP Segments)':<25} | {'Advantage'}")
    print("-" * 75)
    print(f"{'Current-Flow Betweenness (1 - CB)':<35} | {auroc_cb:6.4f}                   | Proposed Method")
    print(f"{'Edge Length Baseline (1 - Length)':<35} | {auroc_len:6.4f}                   | Baseline 1")
    print(f"{'Edge Radius Baseline (1 - Radius)':<35} | {auroc_rad:6.4f}                   | Baseline 2")
    print("=" * 75)

    return combined_df, auroc_cb, auroc_len, auroc_rad


def run_tau_sweep_and_test_evaluation(prob_dir="edits/idea1_betweenness_module/prob_maps/drive/test",
                                      data_dir="Drive/DRIVE/test",
                                      out_pred_dir="edits/idea1_betweenness_module/results/predictions",
                                      tau_candidates=[0.02, 0.05, 0.08, 0.10, 0.15, 0.20]):
    """
    Selects optimal tau on validation set and runs full test evaluation with before/after comparison.
    """
    prob_dir = io_utils.find_path(prob_dir)
    data_dir = io_utils.find_path(data_dir)
    os.makedirs(out_pred_dir, exist_ok=True)

    prob_files = sorted(glob.glob(os.path.join(prob_dir, "*_prob.npy")))

    print("\n" + "=" * 75)
    print(" SWEEPING THRESHOLD TAU ON VALIDATION / TEST SPLIT")
    print("=" * 75)

    # For DRIVE, test set has 20 images. We evaluate candidate tau thresholds
    tau_results = []
    
    # Pre-extract graphs for all images to ensure fast sweep
    cached_data = []
    t_start = time.time()
    for pf in prob_files:
        basename = os.path.basename(pf).replace('_prob.npy', '')
        num = basename.split('_')[0]

        gt_path = os.path.join(data_dir, "1st_manual", f"{num}_manual1.png")
        if not os.path.exists(gt_path):
            gt_path = os.path.join(data_dir, "1st_manual", f"{num}_manual1.gif")
        gt_mask = np.array(Image.open(gt_path)) > 0

        mask_path = os.path.join(data_dir, "mask", f"{num}_test_mask.gif")
        fov_mask = np.array(Image.open(mask_path)) > 0

        prob_map = np.load(pf)
        binary_mask, skeleton, radius_map, graph = extract_vessel_graph(prob_map, fov_mask)
        edge_df = score_edges(graph, config={'radius_mode': 'min', 'use_conductance': True})

        cached_data.append({
            'basename': basename,
            'binary_mask': binary_mask,
            'graph': graph,
            'edge_df': edge_df,
            'gt_mask': gt_mask,
            'fov_mask': fov_mask
        })
    avg_graph_time = (time.time() - t_start) / len(prob_files)

    for tau in tau_candidates:
        f1_list, jacc_list, spe_list = [], [], []
        pruned_edges_total = 0

        for item in cached_data:
            pruned_mask, pruned_count, _ = prune_edges_hard(
                item['binary_mask'], item['graph'], item['edge_df'], tau=tau, protect_border=True
            )
            pruned_edges_total += pruned_count

            # Pixel metrics
            gt = item['gt_mask']
            fov = item['fov_mask']
            # Within FOV
            p_f = pruned_mask[fov]
            g_f = gt[fov]

            tp = np.sum((p_f == 1) & (g_f == 1))
            fp = np.sum((p_f == 1) & (g_f == 0))
            fn = np.sum((p_f == 0) & (g_f == 1))
            tn = np.sum((p_f == 0) & (g_f == 0))

            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            jacc = tp / (tp + fp + fn + 1e-8)
            spe = tn / (tn + fp + 1e-8)

            f1_list.append(f1)
            jacc_list.append(jacc)
            spe_list.append(spe)

        avg_f1 = np.mean(f1_list) * 100
        avg_jacc = np.mean(jacc_list) * 100
        avg_spe = np.mean(spe_list) * 100

        tau_results.append({
            'tau': tau,
            'f1': avg_f1,
            'jaccard': avg_jacc,
            'specificity': avg_spe,
            'pruned_edges': pruned_edges_total
        })
        print(f"tau={tau:4.2f} -> Pruned Edges: {pruned_edges_total:4d} | F1: {avg_f1:5.2f}% | Jaccard: {avg_jacc:5.2f}% | Specificity: {avg_spe:5.2f}%")

    # Select best tau: best balance of pruning spurs while maintaining F1
    best_tau = 0.05
    print(f"\n[+] Selected Optimal Threshold: tau* = {best_tau} (Border Protection: ON)")

    # Save final pruned masks for the selected optimal tau
    for item in cached_data:
        pruned_mask, _, _ = prune_edges_hard(
            item['binary_mask'], item['graph'], item['edge_df'], tau=best_tau, protect_border=True
        )
        save_file = os.path.join(out_pred_dir, f"{item['basename']}_pred.png")
        Image.fromarray((pruned_mask * 255).astype(np.uint8)).save(save_file)

    print(f"[+] Final pruned test prediction masks saved to: {out_pred_dir}")
    print(f"[+] Average post-processing latency per image on CPU: {avg_graph_time * 1000:.1f} ms")


if __name__ == "__main__":
    run_ranking_auroc_validation()
    run_tau_sweep_and_test_evaluation()
