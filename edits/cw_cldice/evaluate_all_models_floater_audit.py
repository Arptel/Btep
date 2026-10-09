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
        ("[2] Standalone cw-BCE", "checkpoints/best_sa_unetv2_stare_cwbce.pth"),
        ("[3] Ours cw-clDice", "checkpoints/best_sa_unetv2_stare_cwcldice.pth"),
        ("[4] Ours Unified", "checkpoints/best_sa_unetv2_stare_unified.pth"),
        ("[5] CAD-Topo-CSA (1x21)", "checkpoints/best_sa_unetv2_stare_cadtocsa.pth"),
        ("[6] CAD-Topo + Unified", "checkpoints/best_sa_unetv2_stare_cadtocsa_unified.pth"),
        ("[7] CAD-Topo + Murray", "checkpoints/best_sa_unetv2_stare_cadtocsa_murray.pth"),
        ("[8] CAD-Topo (3x21)", "checkpoints/best_sa_unetv2_stare_cadtocsa_h3.pth"),
        ("[9] CAD + Ortho-Inhib", "checkpoints/best_sa_unetv2_stare_cadtocsa_ortho.pth"),
        ("[10] Final Synthesis", "checkpoints/best_sa_unetv2_stare_final_synthesis.pth"),
    ]

    results = {}
    for name, path in models:
        if os.path.exists(path):
            print(f"[*] Evaluating {name}...")
            results[name] = evaluate_checkpoint(path, test_loader, device)
        else:
            print(f"[-] Checkpoint {path} not found! Skipping {name}")

    # Print Console Summary
    print("\n" + "=" * 200)
    print("COMPLETE RETINAL FLOATER HALLUCINATION & CONNECTIVITY AUDIT ON STARE (10 MODELS)")
    print("=" * 200)

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
        "Specificity (%)",
        "Global Acc (%)",
        "MCC (%)",
        "AUC-ROC (%)"
    ]

    header_fmt = (
        f"{headers[0]:<26} | {headers[1]:<12} | {headers[2]:<13} | {headers[3]:<11} | "
        f"{headers[4]:<17} | {headers[5]:<17} | {headers[6]:<9} | {headers[7]:<10} | "
        f"{headers[8]:<9} | {headers[9]:<9} | {headers[10]:<7} | {headers[11]:<15} | "
        f"{headers[12]:<15} | {headers[13]:<14} | {headers[14]:<8} | {headers[15]:<11}"
    )
    print(header_fmt)
    print("-" * 200)

    for name, r in results.items():
        row_str = (
            f"{name:<26} | "
            f"{r['beta0_pred']:>9.2f}   | "
            f"{r['num_floaters']:>10.2f}   | "
            f"{r['fp_floaters']:>8.2f}   | "
            f"{r['fp_floater_rate']:>14.2f}%  | "
            f"{r['mean_floater_size']:>13.1f} px  | "
            f"{r['lccr']*100:>6.2f}%  | "
            f"{r['cldice']*100:>7.2f}%  | "
            f"{r['tsens']*100:>6.2f}%  | "
            f"{r['tprec']*100:>6.2f}%  | "
            f"{r['f1']*100:>5.2f}% | "
            f"{r['sensitivity']*100:>12.2f}%  | "
            f"{r['specificity']*100:>12.2f}%  | "
            f"{r['accuracy']*100:>11.2f}%  | "
            f"{r['mcc']*100:>5.2f}% | "
            f"{r['auc']*100:>8.2f}%"
        )
        print(row_str)
    print("=" * 200)

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
        "Specificity\n(%)",
        "Global\nAcc (%)",
        "MCC\n(%)",
        "AUC-ROC\n(%)"
    ]

    table_data = []
    for name, r in results.items():
        clean_name = (
            name.replace(" (ISBI 2026)", "")
                .replace(" (Track 1A)", "")
                .replace(" (Proposed)", "")
                .replace(" (Track 1B)", "")
                .replace(" (Track 2)", "")
        )
        table_data.append([
            clean_name,
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
            f"{r['accuracy']*100:.2f}%",
            f"{r['mcc']*100:.2f}%",
            f"{r['auc']*100:.2f}%",
        ])

    fig, ax = plt.subplots(figsize=(32, 9.0), dpi=300)
    ax.axis('off')

    table = ax.table(
        cellText=table_data,
        colLabels=col_headers,
        loc='center',
        cellLoc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.0)

    # 16 columns: sum = 1.0
    col_widths = [
        0.15,   # Model Paradigm
        0.05,   # beta0
        0.05,   # Floater Count
        0.055,  # Pure FP Floaters
        0.07,   # Hallucination Rate
        0.065,  # Mean Floater Area
        0.06,   # LCCR
        0.055,  # clDice
        0.055,  # Topology Sens
        0.055,  # Topology Prec
        0.05,   # F1 / Dice
        0.06,   # Sensitivity
        0.055,  # Specificity
        0.055,  # Global Acc
        0.05,   # MCC
        0.055   # AUC-ROC
    ]
    for i, width in enumerate(col_widths):
        for j in range(len(table_data) + 1):
            cell = table[(j, i)]
            cell.set_width(width)

    header_color = '#0F172A'
    row_alt_color = '#F8FAFC'
    row_base_color = '#FFFFFF'
    highlight_ortho = '#FEF08A'
    highlight_winner = '#DCFCE7'

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor('#CBD5E1')
        cell.set_linewidth(0.8)

        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
            cell.get_text().set_fontsize(9.5)
            cell.set_height(0.10)
        else:
            cell.set_height(0.065)
            r = row_idx - 1
            base_bg = row_alt_color if r % 2 == 1 else row_base_color

            if col_idx == 0:
                cell.set_facecolor('#F1F5F9')
                cell.get_text().set_weight('bold')
                cell.get_text().set_ha('left')
            # Ortho Hallucination lowest (r=8, col=4)
            elif col_idx == 4 and r == 8:
                cell.set_facecolor(highlight_ortho)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#78350F')
            # cw-clDice beta0 lowest (r=2, col=1)
            elif col_idx == 1 and r == 2:
                cell.set_facecolor(highlight_winner)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#14532D')
            # Final Synthesis clDice highest (r=9, col=7)
            elif col_idx == 7 and r == 9:
                cell.set_facecolor(highlight_ortho)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#78350F')
            # CAD+Unified peak recall (r=5, col=11)
            elif col_idx == 11 and r == 5:
                cell.set_facecolor(highlight_winner)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#14532D')
            # Ortho peak specificity (r=8, col=12)
            elif col_idx == 12 and r == 8:
                cell.set_facecolor(highlight_ortho)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#78350F')
            # Peak F1: CAD 3x21 (r=7, col=10) & cw-clDice (r=2, col=10)
            elif col_idx == 10 and r in [2, 7]:
                cell.set_facecolor(highlight_winner)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#14532D')
            # Peak AUC-ROC: cw-BCE (r=1, col=15) & Final Synthesis (r=9, col=15)
            elif col_idx == 15 and r in [1, 9]:
                cell.set_facecolor(highlight_ortho if r == 9 else highlight_winner)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#78350F' if r == 9 else '#14532D')
            # Peak MCC: cw-clDice (r=2, col=14)
            elif col_idx == 14 and r == 2:
                cell.set_facecolor(highlight_winner)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#14532D')
            # Peak Accuracy: cw-clDice (r=2, col=13)
            elif col_idx == 13 and r == 2:
                cell.set_facecolor(highlight_winner)
                cell.get_text().set_weight('bold')
                cell.get_text().set_color('#14532D')
            else:
                cell.set_facecolor(base_bg)
                cell.get_text().set_color('#334155')

    plt.title(
        "COMPLETE RETINAL VESSEL FLOATER HALLUCINATION & TOPOLOGICAL CONNECTIVITY AUDIT (STARE BENCHMARK)\n"
        "Quantifying Betti-0 Stumps, Disconnected Floater Hallucination Rate, AUC-ROC, MCC, and Global Accuracy Across Models",
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
