"""
Multi-Way Comparative Benchmark on STARE:
=========================================
Compares across all 5 paradigms:
1. Baseline SA-UNetv2 (ISBI 2026: BCE + MCC)
2. Vanilla clDice (CVPR 2021: BCE + MCC + clDice)
3. cw-BCE Standalone (Track 1 Sub-step 1A: cw-BCE + MCC, no skeletonizer)
4. Ours cw-clDice (Proposed: BCE + MCC + cw-clDice)
5. Ours Unified Caliber (Track 1 Sub-step 1B: cw-BCE + MCC + cw-clDice)
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
    y_true_flat = y_true.flatten().astype(np.float32)
    y_prob_flat = y_pred_prob.flatten().astype(np.float32)
    pred_bin = (y_pred_prob >= threshold).astype(np.float32)
    pred_bin_flat = pred_bin.flatten()
    gt_bin = (y_true >= 0.5).astype(np.float32)

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

    s_gt = morph.skeletonize(gt_bin).astype(np.float32) if gt_bin.sum() > 0 else np.zeros_like(gt_bin)
    s_pred = morph.skeletonize(pred_bin).astype(np.float32) if pred_bin.sum() > 0 else np.zeros_like(pred_bin)

    tsens = (s_gt * pred_bin).sum() / (s_gt.sum() + 1e-7)
    tprec = (s_pred * gt_bin).sum() / (s_pred.sum() + 1e-7)
    cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-7)

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


def evaluate_model(model_path, test_loader, device, threshold=0.5):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    use_cad = any('conv_trunk' in k for k in state_dict.keys())
    model = SA_UNetv2(in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.0, use_cad_topo_csa=use_cad).to(device)
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

            prob_np = preds_cropped.squeeze().cpu().numpy()
            lbl_np = lbls_cropped.squeeze().cpu().numpy()

            metrics = compute_sample_metrics(lbl_np, prob_np, threshold=threshold)
            sample_results[img_id] = metrics

    keys = ['f1', 'sensitivity', 'specificity', 'accuracy', 'mcc', 'auc',
            'cldice', 'tsens', 'tprec', 'beta0_pred', 'frag_ratio', 'lccr']
    return {k: float(np.mean([sample_results[img_id][k] for img_id in sample_results])) for k in keys}


def run_multiway_benchmark(threshold=0.5, device_name="cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device_name)
    _, _, test_loader = get_stare_cwcldice_dataloaders(repeat=1)

    models = [
        ("Baseline (ISBI 2026)", "checkpoints/best_sa_unetv2_stare.pth"),
        ("Vanilla clDice", "checkpoints/best_sa_unetv2_stare_cldice.pth"),
        ("Standalone cw-BCE (1A)", "checkpoints/best_sa_unetv2_stare_cwbce.pth"),
        ("Ours cw-clDice", "checkpoints/best_sa_unetv2_stare_cwcldice.pth"),
        ("Ours Unified (1B)", "checkpoints/best_sa_unetv2_stare_unified.pth"),
        ("CAD-Topo-CSA (Track 2)", "checkpoints/best_sa_unetv2_stare_cadtocsa.pth"),
        ("CAD-Topo-CSA + Unified (Both)", "checkpoints/best_sa_unetv2_stare_cadtocsa_unified.pth"),
        ("CAD-Topo-CSA + Murray (Idea 1+2+3)", "checkpoints/best_sa_unetv2_stare_cadtocsa_murray.pth"),
    ]

    results = {}
    for name, path in models:
        if os.path.exists(path):
            print(f"[*] Evaluating {name} from {path}...")
            results[name] = evaluate_model(path, test_loader, device, threshold)
        else:
            print(f"[!] Checkpoint not found: {path} (skipping {name})")

    # Header
    col_names = list(results.keys())
    metric_defs = [
        ("Topological Connectivity", None),
        ("Centerline Dice (clDice) (%)", 'cldice', 100.0, True),
        ("Topology Sensitivity (%)", 'tsens', 100.0, True),
        ("Topology Precision (%)", 'tprec', 100.0, True),
        ("Betti-0 Stumps (beta0)", 'beta0_pred', 1.0, False),
        ("Fragmentation Ratio", 'frag_ratio', 1.0, False),
        ("Largest Tree Ratio (LCCR) (%)", 'lccr', 100.0, True),
        ("Pixel Metrics", None),
        ("F1-Score / Dice (%)", 'f1', 100.0, True),
        ("Sensitivity (%)", 'sensitivity', 100.0, True),
        ("Specificity (%)", 'specificity', 100.0, True),
        ("Global Accuracy (%)", 'accuracy', 100.0, True),
        ("Matthews Corr (MCC) (%)", 'mcc', 100.0, True),
        ("AUC-ROC (%)", 'auc', 100.0, True),
    ]

    header_str = f"{'Evaluation Metric':<28} | " + " | ".join([f"{col:<22}" for col in col_names])
    sep = "=" * len(header_str)
    print("\n" + sep)
    print(" COMPLETE ABLATION COMPARISON TABLE ON STARE BENCHMARK")
    print(sep)
    print(header_str)
    print(sep)

    for item in metric_defs:
        if item[1] is None:
            print("-" * len(header_str))
            print(f"-- {item[0].upper()} --")
            continue

        label, key, mult, is_pct = item
        vals = [f"{results[col][key] * mult:.2f}{'%' if is_pct else ('x' if 'Ratio' in label else '')}" for col in col_names]
        row_str = f"{label:<28} | " + " | ".join([f"{v:<22}" for v in vals])
        print(row_str)

    print(sep)
    return results


if __name__ == '__main__':
    run_multiway_benchmark()
