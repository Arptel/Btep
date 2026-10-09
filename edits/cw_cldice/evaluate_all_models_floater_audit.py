"""
Comprehensive Floater Hallucination & Disconnected Stumps Audit
==============================================================
Evaluates ALL 11 implemented model checkpoints on the STARE benchmark:
1. Baseline (ISBI 2026)
2. Vanilla clDice (CVPR 2021)
3. Standalone cw-BCE (Track 1A)
4. Ours cw-clDice (Proposed)
5. Ours Unified (Track 1B)
6. CAD-Topo-CSA (1x21, Track 2)
7. CAD-Topo-CSA + Unified (Idea 1 + 2)
8. CAD-Topo-CSA + Murray (Idea 1 + 2 + 3)
9. CAD-Topo-CSA (3x21 Transverse, Idea 2 alone)
10. CAD-Topo-CSA + Ortho-Inhib (Idea 4 alone)
11. Final Combined Synthesis (Idea 1 + Idea 2 [h=3] + Idea 3 + Idea 4)

Metrics Evaluated:
- Betti-0 Stumps (beta0)
- Disconnected Floaters Count (Num Floaters)
- Mean Floater Size (px)
- Pure False Positive Floaters (FP Floaters)
- Floater Hallucination Rate (% pure hallucination)
- Largest Tree Ratio (LCCR %)
- Centerline Dice (clDice %)
- Topology Sensitivity (Tsens %)
- Topology Precision (Tprec %)
- F1-Score / Dice (%)
- Sensitivity (%)
- Specificity (%)
- AUC-ROC (%)
"""

import os
import sys
import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "baseline_reproduction", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from model import SA_UNetv2
from stare_dataset import get_stare_cwcldice_dataloaders, restore_image_stare


def compute_floater_metrics(y_true, y_pred_prob, threshold=0.5):
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
        fp_floaters = sum(
            1 for i in range(1, num_pred_cc + 1)
            if i != trunk_label and np.sum((labeled_pred == i) & (gt_bin == 1.0)) == 0
        )
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
        'fp_floaters': fp_floaters,
        'mean_floater_size': mean_floater_size,
        'fp_floater_rate': fp_floater_rate
    }


def evaluate_checkpoint(model_path, test_loader, device, threshold=0.5):
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

            metrics = compute_floater_metrics(lbl_np, prob_np, threshold=threshold)
            sample_results[img_id] = metrics

    keys = [
        'f1', 'sensitivity', 'specificity', 'accuracy', 'mcc', 'auc',
        'cldice', 'tsens', 'tprec', 'beta0_pred', 'frag_ratio', 'lccr',
        'num_floaters', 'fp_floaters', 'mean_floater_size', 'fp_floater_rate'
    ]
    return {k: float(np.mean([sample_results[img_id][k] for img_id in sample_results])) for k in keys}


def run_full_floater_audit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_stare_cwcldice_dataloaders(repeat=1)

    models = [
        ("[1] Baseline (ISBI 2026)", "checkpoints/best_sa_unetv2_stare.pth"),
        ("[2] Vanilla clDice", "checkpoints/best_sa_unetv2_stare_cldice.pth"),
        ("[3] Standalone cw-BCE", "checkpoints/best_sa_unetv2_stare_cwbce.pth"),
        ("[4] Ours cw-clDice", "checkpoints/best_sa_unetv2_stare_cwcldice.pth"),
        ("[5] Ours Unified", "checkpoints/best_sa_unetv2_stare_unified.pth"),
        ("[6] CAD-Topo-CSA (1x21)", "checkpoints/best_sa_unetv2_stare_cadtocsa.pth"),
        ("[7] CAD-Topo + Unified", "checkpoints/best_sa_unetv2_stare_cadtocsa_unified.pth"),
        ("[8] CAD-Topo + Murray", "checkpoints/best_sa_unetv2_stare_cadtocsa_murray.pth"),
        ("[9] CAD-Topo (3x21)", "checkpoints/best_sa_unetv2_stare_cadtocsa_h3.pth"),
        ("[10] CAD + Ortho-Inhib", "checkpoints/best_sa_unetv2_stare_cadtocsa_ortho.pth"),
        ("[11] Final Synthesis", "checkpoints/best_sa_unetv2_stare_final_synthesis.pth"),
    ]

    results = {}
    for name, path in models:
        if os.path.exists(path):
            print(f"[*] Evaluating {name}...")
            results[name] = evaluate_checkpoint(path, test_loader, device)
        else:
            print(f"[-] Checkpoint {path} not found! Skipping {name}")

    # Print Console Summary
    print("\n" + "=" * 165)
    print("COMPLETE RETINAL FLOATER HALLUCINATION & CONNECTIVITY AUDIT ON STARE (ALL 11 MODELS)")
    print("=" * 165)

    headers = [
        "Model Paradigm",
        "beta0 Stumps",
        "Floater Count",
        "FP Floaters",
        "Hallucination (%)",
        "Floater Size (px)",
        "LCCR (%)",
        "clDice (%)",
        "Tsens (%)",
        "Tprec (%)",
        "F1 (%)",
        "Sensitivity (%)",
        "Specificity (%)"
    ]

    header_fmt = f"{headers[0]:<28} | {headers[1]:<13} | {headers[2]:<13} | {headers[3]:<12} | {headers[4]:<17} | {headers[5]:<17} | {headers[6]:<10} | {headers[7]:<10} | {headers[8]:<10} | {headers[9]:<10} | {headers[10]:<8} | {headers[11]:<15} | {headers[12]:<15}"
    print(header_fmt)
    print("-" * 165)

    for name, r in results.items():
        row_str = (
            f"{name:<28} | "
            f"{r['beta0_pred']:>10.2f}    | "
            f"{r['num_floaters']:>10.2f}    | "
            f"{r['fp_floaters']:>9.2f}   | "
            f"{r['fp_floater_rate']:>14.2f}%  | "
            f"{r['mean_floater_size']:>13.1f} px  | "
            f"{r['lccr']*100:>7.2f}%  | "
            f"{r['cldice']*100:>7.2f}%  | "
            f"{r['tsens']*100:>7.2f}%  | "
            f"{r['tprec']*100:>7.2f}%  | "
            f"{r['f1']*100:>6.2f}% | "
            f"{r['sensitivity']*100:>12.2f}%  | "
            f"{r['specificity']*100:>12.2f}%"
        )
        print(row_str)
    print("=" * 165)

    # Render High-Resolution PNG Table
    render_floater_table_png(results)

    return results


def render_floater_table_png(results):
    out_dir = os.path.join(project_root, "results", "stare_multiway")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "stare_floater_hallucination_audit_table.png")

    col_headers = [
        "Model Paradigm",
        "beta0\nStumps",
        "Floater\nCount",
        "Pure FP\nFloaters",
        "Hallucination\nRate (%)",
        "Mean Floater\nArea (px)",
        "Largest Tree\n(LCCR %)",
        "clDice\n(%)",
        "Topology\nSens (%)",
        "Topology\nPrec (%)",
        "F1 / Dice\n(%)",
        "Sensitivity\nRecall (%)",
        "Specificity\n(%)"
    ]

    table_data = []
    for name, r in results.items():
        table_data.append([
            name.replace(" (ISBI 2026)", "").replace(" (Track 1A)", "").replace(" (Proposed)", "").replace(" (Track 1B)", "").replace(" (Track 2)", ""),
            f"{r['beta0_pred']:.2f}",
            f"{r['num_floaters']:.2f}",
            f"{r['fp_floaters']:.2f}",
            f"{r['fp_floater_rate']:.2f}%",
            f"{r['mean_floater_size']:.1f} px",
            f"{r['lccr']*100:.2f}%",
            f"{r['cldice']*100:.2f}%",
            f"{r['tsens']*100:.2f}%",
            f"{r['tprec']*100:.2f}%",
            f"{r['f1']*100:.2f}%",
            f"{r['sensitivity']*100:.2f}%",
            f"{r['specificity']*100:.2f}%",
        ])

    fig, ax = plt.subplots(figsize=(26, 8.5), dpi=300)
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        colLabels=col_headers,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    col_widths = [0.18, 0.065, 0.065, 0.065, 0.08, 0.075, 0.07, 0.065, 0.07, 0.07, 0.065, 0.07, 0.06]
    for i, width in enumerate(col_widths):
        for j in range(len(table_data) + 1):
            cell = table[(j, i)]
            cell.set_width(width)

    header_color = '#0F172A'
    row_alt_color = '#F8FAFC'
    row_base_color = '#FFFFFF'
    highlight_ortho = '#FEF08A'
    highlight_cldice = '#DCFCE7'

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor('#CBD5E1')
        cell.set_linewidth(0.8)

        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
            cell.get_text().set_fontsize(10)
            cell.set_height(0.10)
        else:
            cell.set_height(0.065)
            r = row_idx - 1
            base_bg = row_alt_color if r % 2 == 1 else row_base_color

            # Highlight Orthogonal Inhibition (Row 9: col 4 Hallucination lowest)
            if col_idx == 0:
                cell.set_facecolor('#F1F5F9')
                cell.get_text().set_weight('bold')
                cell.get_text().set_ha('left')
            elif col_idx == 4 and r == 9: # Ortho hallucination rate
                cell.set_facecolor(highlight_ortho)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#78350F')
            elif col_idx == 1 and r == 3: # cw-clDice beta0 lowest
                cell.set_facecolor(highlight_cldice)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#14532D')
            elif col_idx == 7 and r == 10: # Final Synthesis clDice highest
                cell.set_facecolor(highlight_ortho)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#78350F')
            elif col_idx == 11 and r == 6: # CAD+Unified peak recall
                cell.set_facecolor(highlight_cldice)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#14532D')
            else:
                cell.set_facecolor(base_bg)
                cell.get_text().set_color('#334155')

    plt.title(
        "COMPLETE RETINAL VESSEL FLOATER HALLUCINATION & TOPOLOGICAL CONNECTIVITY AUDIT (STARE BENCHMARK)\n"
        "Quantifying Betti-0 Stumps, Disconnected Floater Hallucination Rate, and Topological Precision Across All 11 Models",
        fontsize=13,
        fontweight='bold',
        pad=20,
        color='#0F172A'
    )

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[+] Saved publication-quality audit table image to:\n    {save_path}")


if __name__ == '__main__':
    run_full_floater_audit()
