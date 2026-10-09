"""
Murray's Law Anatomical Compliance Audit Suite (Idea 3)
======================================================
Evaluates vascular segmentations on physical fluid transport criteria:
1. Mean Murray Branching Ratio (R_bar): Ideal = 1.00
2. Mean Murray Compliance Deviation (Delta_Murray): Lower is better (0.00 = perfect)
3. Bifurcation Preservation Rate (B_recall): % GT branch points preserved
4. Bifurcation Precision (B_prec): % predicted branch points that are anatomically valid
"""
import os
import sys
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "baseline_reproduction", "src"))
sys.path.insert(0, os.path.join(project_root, "edits", "cw_cldice"))
sys.path.insert(0, os.path.dirname(__file__))

from model import SA_UNetv2
from stare_dataset import get_stare_cwcldice_dataloaders, restore_image_stare
from murray_loss import detect_bifurcations_2d, compute_murray_junction_radii


def evaluate_murray_compliance(gt_bin, pred_bin, tolerance=3.5):
    """
    Computes Murray's Law anatomical metrics between ground truth and predicted binary masks.
    """
    skel_gt = skeletonize(gt_bin > 0).astype(np.uint8)
    skel_pred = skeletonize(pred_bin > 0).astype(np.uint8)

    edt_gt = ndi.distance_transform_edt(gt_bin > 0).astype(np.float32)
    edt_pred = ndi.distance_transform_edt(pred_bin > 0).astype(np.float32)

    bif_gt, _ = detect_bifurcations_2d(skel_gt)
    bif_pred, _ = detect_bifurcations_2d(skel_pred)

    num_gt = len(bif_gt)
    num_pred = len(bif_pred)

    if num_gt == 0:
        return {
            'b_recall': 1.0,
            'b_prec': 1.0 if num_pred == 0 else 0.0,
            'murray_ratio': 1.0,
            'murray_dev': 0.0,
            'num_gt_bif': 0,
            'num_pred_bif': num_pred
        }

    # Match GT bifurcations with predicted bifurcations within tolerance
    matched_gt = 0
    matched_pred = 0
    preserved_deviations = []
    preserved_ratios = []

    pred_matched_indices = set()

    for cy_gt, cx_gt in bif_gt:
        best_dist = float('inf')
        best_p_idx = -1
        for p_idx, (cy_p, cx_p) in enumerate(bif_pred):
            d = np.hypot(cy_gt - cy_p, cx_gt - cx_p)
            if d < best_dist:
                best_dist = d
                best_p_idx = p_idx

        if best_dist <= tolerance:
            matched_gt += 1
            pred_matched_indices.add(best_p_idx)

    b_recall = matched_gt / max(1, num_gt)
    b_prec = len(pred_matched_indices) / max(1, num_pred)

    # Compute Murray's law metrics on predicted bifurcations
    pred_junc_data = compute_murray_junction_radii(bif_pred, skel_pred, edt_pred)
    if len(pred_junc_data) > 0:
        murray_ratio = float(np.mean([j['murray_ratio'] for j in pred_junc_data]))
        murray_dev = float(np.mean([j['deviation'] for j in pred_junc_data]))
    else:
        murray_ratio = 1.0
        murray_dev = 0.0

    return {
        'b_recall': b_recall,
        'b_prec': b_prec,
        'murray_ratio': murray_ratio,
        'murray_dev': murray_dev,
        'num_gt_bif': num_gt,
        'num_pred_bif': num_pred
    }


def audit_all_models_on_stare(threshold=0.5, device_name="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Audits all 7 trained model checkpoints on Murray's Law anatomical compliance across STARE.
    """
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
        ("Final Synthesis (1+2+3+4)", "checkpoints/best_sa_unetv2_stare_final_synthesis.pth"),
    ]

    results = {}

    for name, path in models:
        if not os.path.exists(path):
            print(f"[!] Checkpoint not found: {path}")
            continue

        print(f"[*] Auditing Murray's Law Compliance for: {name}...")
        ckpt = torch.load(path, map_location=device, weights_only=False)
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

        sample_audits = []
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch['image'].to(device)
                lbls = batch['label'].to(device)

                preds_padded = model(imgs)
                preds_cropped = restore_image_stare(preds_padded, orig_h=605, orig_w=700)
                lbls_cropped = restore_image_stare(lbls, orig_h=605, orig_w=700)

                prob_np = preds_cropped.squeeze().cpu().numpy()
                lbl_np = lbls_cropped.squeeze().cpu().numpy()

                pred_bin = (prob_np >= threshold).astype(np.float32)
                gt_bin = (lbl_np >= 0.5).astype(np.float32)

                audit = evaluate_murray_compliance(gt_bin, pred_bin)
                sample_audits.append(audit)

        keys = ['b_recall', 'b_prec', 'murray_ratio', 'murray_dev', 'num_gt_bif', 'num_pred_bif']
        results[name] = {k: float(np.mean([s[k] for s in sample_audits])) for k in keys}

    # Print Report
    print("\n" + "=" * 120)
    print(" MURRAY'S LAW ANATOMICAL BIFURCATION AUDIT (STARE BENCHMARK)")
    print("=" * 120)
    header = f"{'Model Configuration':<32} | {'Bif Recall (%)':<15} | {'Bif Prec (%)':<15} | {'Murray Ratio (R)':<18} | {'Murray Deviation':<18} | {'Pred Bifurcations':<16}"
    print(header)
    print("-" * 120)
    for name in results:
        r = results[name]
        print(f"{name:<32} | {r['b_recall']*100.0:>13.2f}% | {r['b_prec']*100.0:>13.2f}% | {r['murray_ratio']:>16.3f} | {r['murray_dev']:>16.3f} | {r['num_pred_bif']:>16.1f}")
    print("=" * 120)
    return results


if __name__ == '__main__':
    audit_all_models_on_stare()
