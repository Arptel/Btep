"""
3-Way Comparative Benchmark on STARE: Baseline vs. Vanilla clDice vs. cw-clDice
==============================================================================
Directly benchmarks all 3 paradigms under identical conditions:
1. Baseline SA-UNetv2 (ISBI 2026: BCE + Continuous MCC)
2. Baseline + Vanilla clDice (CVPR 2021: Uniform Centerline Dice)
3. Baseline + Ours cw-clDice (Proposed: Conductance-Weighted Centerline Dice)

Evaluates:
- Pixel Metrics: F1, Sen, Spe, ACC, MCC, AUC
- Topological Metrics: clDice, Tsens, Tprec, Betti-0 (beta0), Frag Ratio, LCCR
- Generates 5-Panel Visual Comparisons:
  [Original | Ground Truth | Baseline | Vanilla clDice | Ours cw-clDice]
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import skimage.morphology as morph
from sklearn.metrics import roc_auc_score
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "baseline_reproduction", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from model import SA_UNetv2
from stare_dataset import get_stare_cwcldice_dataloaders, restore_image_stare


def compute_sample_metrics(y_true, y_pred_prob, threshold=0.5):
    """Computes both pixel-level and topological connectivity metrics."""
    y_true_flat = y_true.flatten().astype(np.float32)
    y_prob_flat = y_pred_prob.flatten().astype(np.float32)
    pred_bin = (y_pred_prob >= threshold).astype(np.float32)
    pred_bin_flat = pred_bin.flatten()
    gt_bin = (y_true >= 0.5).astype(np.float32)

    # Pixel Confusion Matrix
    tp = np.sum((pred_bin_flat == 1.0) & (y_true_flat == 1.0))
    tn = np.sum((pred_bin_flat == 0.0) & (y_true_flat == 0.0))
    fp = np.sum((pred_bin_flat == 1.0) & (y_true_flat == 0.0))
    fn = np.sum((pred_bin_flat == 0.0) & (y_true_flat == 1.0))

    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)

    denom = np.sqrt(float(tp + fp)) * np.sqrt(float(tp + fn)) * np.sqrt(float(tn + fp)) * np.sqrt(float(tn + fn))
    mcc = float((float(tp) * float(tn) - float(fp) * float(fn)) / (denom + 1e-8))

    try:
        auc = roc_auc_score(y_true_flat, y_prob_flat)
    except Exception:
        auc = 0.5

    # Centerlines (CVPR 2021)
    s_gt = morph.skeletonize(gt_bin).astype(np.float32) if gt_bin.sum() > 0 else np.zeros_like(gt_bin)
    s_pred = morph.skeletonize(pred_bin).astype(np.float32) if pred_bin.sum() > 0 else np.zeros_like(pred_bin)

    tsens = (s_gt * pred_bin).sum() / (s_gt.sum() + 1e-7)
    tprec = (s_pred * gt_bin).sum() / (s_pred.sum() + 1e-7)
    cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-7)

    # Betti-0 Connected Components
    labeled_pred, num_pred_cc = ndi.label(pred_bin, structure=ndi.generate_binary_structure(2, 2))
    labeled_gt, num_gt_cc = ndi.label(gt_bin, structure=ndi.generate_binary_structure(2, 2))
    frag_ratio = num_pred_cc / max(1, num_gt_cc)

    if pred_bin.sum() > 0:
        cc_sizes = ndi.sum(pred_bin, labeled_pred, range(1, num_pred_cc + 1))
        max_cc = np.max(cc_sizes) if len(cc_sizes) > 0 else 0
        lccr = max_cc / pred_bin.sum()
    else:
        lccr = 0.0

    return {
        'f1': f1,
        'sensitivity': sen,
        'specificity': spe,
        'accuracy': acc,
        'mcc': mcc,
        'auc': auc,
        'cldice': cldice,
        'tsens': tsens,
        'tprec': tprec,
        'beta0_pred': num_pred_cc,
        'beta0_gt': num_gt_cc,
        'frag_ratio': frag_ratio,
        'lccr': lccr,
        'pred_bin': pred_bin,
        'prob': y_pred_prob
    }


def create_colored_overlay(orig_rgb, pred_bin, color_rgb=(0, 255, 128)):
    """Creates a glowing colored overlay on top of the original fundus photograph."""
    overlay = orig_rgb.copy()
    mask = (pred_bin == 1.0)
    for c in range(3):
        overlay[mask, c] = np.clip(
            orig_rgb[mask, c] * 0.3 + color_rgb[c] * 0.7, 0, 255
        ).astype(np.uint8)
    return overlay


def evaluate_single_model(model_path, test_loader, device, threshold=0.5):
    """Runs inference and collects all sample metrics for one model."""
    model = SA_UNetv2(in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.0).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    sample_results = {}
    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            img_id = batch['id'][0]

            preds_padded = model(imgs)
            preds_cropped = restore_image_stare(preds_padded, orig_h=605, orig_w=700)
            lbls_cropped = restore_image_stare(lbls, orig_h=605, orig_w=700)
            imgs_cropped = restore_image_stare(imgs, orig_h=605, orig_w=700)

            prob_np = preds_cropped.squeeze().cpu().numpy()
            lbl_np = lbls_cropped.squeeze().cpu().numpy()
            orig_rgb = (imgs_cropped.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

            metrics = compute_sample_metrics(lbl_np, prob_np, threshold=threshold)
            metrics['orig_rgb'] = orig_rgb
            metrics['gt_bin'] = (lbl_np >= 0.5).astype(np.float32)
            sample_results[img_id] = metrics

    return sample_results


def aggregate_metrics(sample_results):
    """Averages metrics across all test images."""
    keys = ['f1', 'sensitivity', 'specificity', 'accuracy', 'mcc', 'auc',
            'cldice', 'tsens', 'tprec', 'beta0_pred', 'frag_ratio', 'lccr']
    return {k: float(np.mean([sample_results[img_id][k] for img_id in sample_results])) for k in keys}


def run_3way_benchmark(
    baseline_path="checkpoints/best_sa_unetv2_stare.pth",
    cldice_path="checkpoints/best_sa_unetv2_stare_cldice.pth",
    cwcldice_path="checkpoints/best_sa_unetv2_stare_cwcldice.pth",
    threshold=0.5,
    results_dir="results/stare_3way",
    device_name="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "comparisons"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)

    print("=" * 80)
    print(" 3-WAY COMPARATIVE BENCHMARK ON STARE BENCHMARK")
    print(f" Device: {device_name.upper()} | Threshold: {threshold}")
    print(f" [1] Baseline:      {baseline_path}")
    print(f" [2] Vanilla clDice:{cldice_path}")
    print(f" [3] Ours cw-clDice:{cwcldice_path}")
    print("=" * 80, flush=True)

    _, _, test_loader = get_stare_cwcldice_dataloaders(repeat=1)

    print("[1/3] Evaluating Baseline SA-UNetv2...")
    res_base = evaluate_single_model(baseline_path, test_loader, device, threshold)

    print("[2/3] Evaluating Vanilla clDice SA-UNetv2...")
    res_cldice = evaluate_single_model(cldice_path, test_loader, device, threshold)

    print("[3/3] Evaluating Ours cw-clDice SA-UNetv2...")
    res_cwcldice = evaluate_single_model(cwcldice_path, test_loader, device, threshold)

    avg_base = aggregate_metrics(res_base)
    avg_cldice = aggregate_metrics(res_cldice)
    avg_cwcldice = aggregate_metrics(res_cwcldice)

    # Print Table
    header = (
        f"{'Evaluation Metric':<28} | {'Baseline':<10} | {'Vanilla clDice':<14} | "
        f"{'Ours (cw-clDice)':<16} | {'Delta vs Base':<13} | {'Delta vs clDice':<15}"
    )
    sep = "=" * len(header)
    print("\n" + sep)
    print(" 3-WAY STARE PERFORMANCE COMPARISON TABLE")
    print(sep)
    print(header)
    print(sep)

    metric_defs = [
        ("Pixel Metrics", None),
        ("F1-Score / Dice (%)", 'f1', 100.0, True),
        ("Sensitivity (%)", 'sensitivity', 100.0, True),
        ("Specificity (%)", 'specificity', 100.0, True),
        ("Accuracy (%)", 'accuracy', 100.0, True),
        ("Matthews Corr (MCC) (%)", 'mcc', 100.0, True),
        ("AUC-ROC (%)", 'auc', 100.0, True),
        ("Topological Connectivity", None),
        ("Centerline Dice (clDice) (%)", 'cldice', 100.0, True),
        ("Topology Sensitivity (%)", 'tsens', 100.0, True),
        ("Topology Precision (%)", 'tprec', 100.0, True),
        ("Betti-0 Components (beta0)", 'beta0_pred', 1.0, False),
        ("Fragmentation Ratio", 'frag_ratio', 1.0, False),
        ("Largest Tree Ratio (LCCR) (%)", 'lccr', 100.0, True),
    ]

    for item in metric_defs:
        if item[1] is None:
            print("-" * len(header))
            print(f"-- {item[0].upper()} --")
            continue

        label, key, mult, is_pct = item
        v_base = avg_base[key] * mult
        v_cl = avg_cldice[key] * mult
        v_cw = avg_cwcldice[key] * mult

        d_base = v_cw - v_base
        d_cl = v_cw - v_cl

        unit = "%" if is_pct else ("x" if "Ratio" in label else "")
        d_base_str = f"{d_base:+.2f}{unit}"
        d_cl_str = f"{d_cl:+.2f}{unit}"

        print(
            f"{label:<28} | {v_base:<10.2f} | {v_cl:<14.2f} | {v_cw:<16.2f} | "
            f"{d_base_str:<13} | {d_cl_str:<15}"
        )

    print(sep)

    # Generate 5-Panel Comparisons
    print(f"\n[*] Saving 5-Panel comparison visualizations to: {results_dir}/comparisons")
    for img_id in res_base:
        orig = res_base[img_id]['orig_rgb']
        gt = (res_base[img_id]['gt_bin'] * 255).astype(np.uint8)
        gt_rgb = np.stack([gt] * 3, axis=-1)

        p_base = (res_base[img_id]['pred_bin'] * 255).astype(np.uint8)
        p_base_rgb = np.stack([p_base] * 3, axis=-1)

        p_cl = (res_cldice[img_id]['pred_bin'] * 255).astype(np.uint8)
        p_cl_rgb = np.stack([p_cl] * 3, axis=-1)

        p_cw = (res_cwcldice[img_id]['pred_bin'] * 255).astype(np.uint8)
        p_cw_rgb = np.stack([p_cw] * 3, axis=-1)

        # Save individual prediction masks
        Image.fromarray(p_base).save(os.path.join(results_dir, "predictions", f"{img_id}_baseline.png"))
        Image.fromarray(p_cl).save(os.path.join(results_dir, "predictions", f"{img_id}_cldice.png"))
        Image.fromarray(p_cw).save(os.path.join(results_dir, "predictions", f"{img_id}_cwcldice.png"))

        # Combine 5 panels horizontally: [Original | Ground Truth | Baseline | Vanilla clDice | Ours cw-clDice]
        five_panel = np.concatenate([orig, gt_rgb, p_base_rgb, p_cl_rgb, p_cw_rgb], axis=1)
        save_path = os.path.join(results_dir, "comparisons", f"{img_id}_5panel_comparison.png")
        Image.fromarray(five_panel).save(save_path)
        print(f"  -> Saved {img_id}_5panel_comparison.png")

    print(f"\n[+] 3-Way Benchmark Complete! All results written to {results_dir}")
    return avg_base, avg_cldice, avg_cwcldice


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3-Way STARE Benchmark")
    parser.add_argument("--baseline", type=str, default="checkpoints/best_sa_unetv2_stare.pth")
    parser.add_argument("--cldice", type=str, default="checkpoints/best_sa_unetv2_stare_cldice.pth")
    parser.add_argument("--cwcldice", type=str, default="checkpoints/best_sa_unetv2_stare_cwcldice.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--results_dir", type=str, default="results/stare_3way")
    args = parser.parse_args()

    run_3way_benchmark(
        baseline_path=args.baseline,
        cldice_path=args.cldice,
        cwcldice_path=args.cwcldice,
        threshold=args.threshold,
        results_dir=args.results_dir
    )
