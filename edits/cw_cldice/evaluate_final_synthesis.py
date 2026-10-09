"""
Comparative Evaluation Suite: Final Combined Synthesis (Step 4)
================================================================
Section 7, Step 4: Dual-Axis Geometric Defense + Physical Synergy on STARE.
Combines:
- Idea 1: Unified Conductance-Weighted Supervision (cw-BCE + cw-clDice)
- Idea 2: Optimal h=3 Transverse Contrast Strip Attention (3x21)
- Idea 3: Murray's Law Anatomical Bifurcation Physics
- Idea 4: Orthogonal Cross-Strip Gating Inhibition

Evaluates:
- Baseline SA-UNetv2
- CAD-Topo-CSA h=1 (Uninhibited)
- CAD-Topo-CSA h=3 (Idea 2 alone)
- CAD-Topo + Murray (Idea 1+2+3 with h=1)
- Final Combined Synthesis (Idea 1 + Idea 2 h=3 + Idea 3 + Idea 4)
"""

import os
import sys
import numpy as np
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

    struct = ndi.generate_binary_structure(2, 2)
    labeled_pred, num_pred_cc = ndi.label(pred_bin, structure=struct)
    labeled_gt, num_gt_cc = ndi.label(gt_bin, structure=struct)
    frag_ratio = num_pred_cc / max(1, num_gt_cc)

    if pred_bin.sum() > 0:
        cc_sizes = ndi.sum(pred_bin, labeled_pred, range(1, num_pred_cc + 1))
        trunk_label = int(np.argmax(cc_sizes) + 1)
        max_cc = np.max(cc_sizes) if len(cc_sizes) > 0 else 0
        lccr = max_cc / pred_bin.sum()
        
        floater_sizes = [int(cc_sizes[i - 1]) for i in range(1, num_pred_cc + 1) if i != trunk_label]
        fp_floaters = sum(1 for i in range(1, num_pred_cc + 1) if i != trunk_label and np.sum((labeled_pred == i) & (gt_bin == 1.0)) == 0)
    else:
        lccr = 0.0
        floater_sizes = []
        fp_floaters = 0

    num_floaters = len(floater_sizes)
    mean_floater_size = float(np.mean(floater_sizes)) if num_floaters > 0 else 0.0
    fp_floater_rate = (fp_floaters / num_floaters * 100.0) if num_floaters > 0 else 0.0

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
        'frag_ratio': frag_ratio,
        'lccr': lccr,
        'num_floaters': num_floaters,
        'mean_floater_size': mean_floater_size,
        'fp_floater_rate': fp_floater_rate
    }


def evaluate_model(model_path, test_loader, device, threshold=0.5):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    use_cad = any('conv_trunk' in k for k in state_dict.keys())
    strip_h = 1
    for k, v in state_dict.items():
        if 'conv_strip_h' in k:
            strip_h = v.shape[2]
            break
    use_ortho = any('gate_h' in k for k in state_dict.keys())

    model = SA_UNetv2(
        in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.0,
        use_cad_topo_csa=use_cad, strip_height=strip_h,
        use_orthogonal_inhibition=use_ortho
    ).to(device)
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
            'cldice', 'tsens', 'tprec', 'beta0_pred', 'frag_ratio', 'lccr',
            'num_floaters', 'mean_floater_size', 'fp_floater_rate']
    return {k: float(np.mean([sample_results[img_id][k] for img_id in sample_results])) for k in keys}


def run_final_synthesis_benchmark(threshold=0.5, device_name="cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device_name)
    _, _, test_loader = get_stare_cwcldice_dataloaders(repeat=1)

    models = [
        ("Baseline (ISBI 2026)", "checkpoints/best_sa_unetv2_stare.pth"),
        ("CAD-Topo-CSA h=1", "checkpoints/best_sa_unetv2_stare_cadtocsa.pth"),
        ("CAD-Topo-CSA h=3", "checkpoints/best_sa_unetv2_stare_cadtocsa_h3.pth"),
        ("CAD + Murray (1+2+3)", "checkpoints/best_sa_unetv2_stare_cadtocsa_murray.pth"),
        ("Final Synthesis (1+2+3+4)", "checkpoints/best_sa_unetv2_stare_final_synthesis.pth"),
    ]

    results = {}
    for name, path in models:
        if os.path.exists(path):
            print(f"[*] Evaluating {name} from {path}...")
            results[name] = evaluate_model(path, test_loader, device, threshold)
        else:
            print(f"[-] Checkpoint {path} not found. Skipping {name}.")

    print("\n" + "=" * 145)
    print("FINAL COMBINED SYNTHESIS BENCHMARK ON STARE (IDEA 1 + IDEA 2 (h=3) + IDEA 3 + IDEA 4)")
    print("=" * 145)

    header = f"{'Metric':<25} | " + " | ".join([f"{name:<22}" for name in results.keys()])
    print(header)
    print("-" * 145)

    display_metrics = [
        ("Centerline Dice (clDice)", 'cldice', True, "%"),
        ("Topology Sensitivity (Tsens)", 'tsens', True, "%"),
        ("Topology Precision (Tprec)", 'tprec', True, "%"),
        ("Betti-0 Stumps (beta0)", 'beta0_pred', False, ""),
        ("Fragmentation Ratio", 'frag_ratio', False, "x"),
        ("Largest Tree Ratio (LCCR)", 'lccr', True, "%"),
        ("F1-Score / Dice", 'f1', True, "%"),
        ("Sensitivity (Recall)", 'sensitivity', True, "%"),
        ("Specificity", 'specificity', True, "%"),
        ("AUC-ROC", 'auc', True, "%"),
        ("Disconnected Floater Count", 'num_floaters', False, ""),
        ("Mean Floater Size", 'mean_floater_size', False, "px"),
        ("Floater Hallucination Rate", 'fp_floater_rate', False, "%"),
    ]

    for label, key, is_pct, suffix in display_metrics:
        row = f"{label:<25} | "
        vals = []
        for name in results.keys():
            val = results[name][key]
            if is_pct:
                vals.append(f"{val*100:6.2f}%{'':<15}")
            elif suffix == "x":
                vals.append(f"{val:6.2f}x{'':<15}")
            elif suffix == "px":
                vals.append(f"{val:6.1f}px{'':<14}")
            else:
                vals.append(f"{val:6.2f}{'':<16}")
        row += " | ".join(vals)
        print(row)
    print("=" * 145)

    return results


if __name__ == '__main__':
    run_final_synthesis_benchmark()
