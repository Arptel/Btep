"""
Diagnostic Visualizer: Disconnected Vascular Pools & Directional Smear Analysis
================================================================================
Section 7, Step 1: Visual Inspection of Disconnected Pools in STARE Final Model.

Performs:
1. Full test-set inference on STARE using:
   - Model [8]: CAD-Topo-CSA + Murray's Law (checkpoints/best_sa_unetv2_stare_cadtocsa_murray.pth)
   - Model [7]: CAD-Topo-CSA + Unified (checkpoints/best_sa_unetv2_stare_cadtocsa_unified.pth)
   - Model [1]: Baseline SA-UNetv2 (checkpoints/best_sa_unetv2_stare.pth)
2. Quantitative connected component decomposition:
   - Primary Vascular Trunk (largest connected component)
   - Disconnected Floating Stumps / Pools (all secondary components)
   - Floater size distribution, True Positive vs False Positive ratio
3. High-resolution diagnostic figures:
   - 5-Panel Full Retina Map (RGB, GT, Pred, Component Trunk vs Float, Error Map)
   - Magnified Region of Interest (ROI) zoom showing directional pooling smears
   - Multi-Case Summary Board saved to results/stare_final_visual_comparisons/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from PIL import Image
import scipy.ndimage as ndi
import skimage.morphology as morph
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "baseline_reproduction", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from model import SA_UNetv2
from stare_dataset import get_stare_cwcldice_dataloaders, restore_image_stare


def load_model(checkpoint_path, device):
    """Loads checkpoint and initializes model with appropriate skip attention setting."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    use_cad = any('conv_trunk' in k for k in state_dict.keys())
    strip_h = 1
    for k, v in state_dict.items():
        if 'conv_strip_h' in k:
            strip_h = v.shape[2]
            break
    model = SA_UNetv2(in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.0, use_cad_topo_csa=use_cad, strip_height=strip_h).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def analyze_connected_components(pred_bin, gt_bin):
    """
    Decomposes binary prediction into primary trunk and disconnected floating pools.
    Audits size distribution and floater validity (TP vs FP).
    """
    struct = ndi.generate_binary_structure(2, 2)
    labeled_pred, num_cc = ndi.label(pred_bin, structure=struct)
    
    if num_cc == 0:
        return {
            'num_cc': 0,
            'trunk_mask': np.zeros_like(pred_bin),
            'floaters_mask': np.zeros_like(pred_bin),
            'labeled_pred': labeled_pred,
            'floater_sizes': [],
            'tp_floaters': 0,
            'fp_floaters': 0,
            'trunk_ratio': 0.0,
            'floater_centroids': []
        }

    # Measure sizes
    cc_sizes = ndi.sum(pred_bin, labeled_pred, range(1, num_cc + 1))
    trunk_label = int(np.argmax(cc_sizes) + 1)
    
    trunk_mask = (labeled_pred == trunk_label).astype(np.float32)
    floaters_mask = ((pred_bin == 1.0) & (labeled_pred != trunk_label)).astype(np.float32)
    
    floater_sizes = []
    floater_centroids = []
    tp_floaters = 0
    fp_floaters = 0
    
    for i in range(1, num_cc + 1):
        if i == trunk_label:
            continue
        c_mask = (labeled_pred == i)
        c_size = int(np.sum(c_mask))
        floater_sizes.append(c_size)
        
        # Center of mass
        cy, cx = ndi.center_of_mass(c_mask)
        floater_centroids.append((cy, cx, c_size))
        
        # Check if overlaps with true vessels
        if np.sum(c_mask & (gt_bin == 1.0)) > 0:
            tp_floaters += 1
        else:
            fp_floaters += 1
            
    trunk_ratio = float(cc_sizes[trunk_label - 1] / max(1.0, float(pred_bin.sum())))
    
    return {
        'num_cc': num_cc,
        'trunk_mask': trunk_mask,
        'floaters_mask': floaters_mask,
        'labeled_pred': labeled_pred,
        'floater_sizes': floater_sizes,
        'tp_floaters': tp_floaters,
        'fp_floaters': fp_floaters,
        'trunk_ratio': trunk_ratio,
        'floater_centroids': floater_centroids
    }


def find_densest_floater_roi(floater_centroids, h, w, crop_size=160):
    """Finds the bounding crop centered on the highest spatial density of floaters."""
    if not floater_centroids:
        # Default to center of image
        cy, cx = h // 2, w // 2
    else:
        # Compute 2D spatial histogram or density
        pts = np.array([[c[0], c[1]] for c in floater_centroids])
        # Find point with highest number of neighbors within radius 100
        best_count = -1
        best_center = (h // 2, w // 2)
        for p in pts:
            dists = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
            count = np.sum(dists < 100)
            if count > best_count:
                best_count = count
                best_center = (int(p[0]), int(p[1]))
        cy, cx = best_center

    y1 = max(0, min(h - crop_size, cy - crop_size // 2))
    x1 = max(0, min(w - crop_size, cx - crop_size // 2))
    return y1, y1 + crop_size, x1, x1 + crop_size


def create_diagnostic_figure(img_rgb, gt_bin, pred_bin, prob_map, comp_info, img_id, out_path):
    """
    Renders high-resolution 6-panel diagnostic figure:
    [1] Input Fundus RGB
    [2] Ground Truth Annotation
    [3] Model [8] Prediction (Prob/Binary)
    [4] Component Topology Audit (Cyan=Main Trunk, Magenta=Floating Pools)
    [5] Pixel Error Confusion Map (Green=TP, Red=FP hallucination, Blue=FN missed)
    [6] High-Mag ROI Zoom (160x160) showing directional smear structure
    """
    h, w = gt_bin.shape
    y1, y2, x1, x2 = find_densest_floater_roi(comp_info['floater_centroids'], h, w, crop_size=160)

    fig = plt.figure(figsize=(24, 15), facecolor='#0D1117')
    fig.suptitle(
        f"Diagnostic Inspection of Disconnected Vascular Pools & Directional Smears\n"
        f"STARE Test Sample: {img_id} | Model [8]: CAD-Topo-CSA + Murray's Law | Total Disconnected Stumps: {comp_info['num_cc'] - 1}",
        fontsize=20, color='white', fontweight='bold', y=0.98
    )

    gs = fig.add_gridspec(2, 3, wspace=0.08, hspace=0.18, left=0.03, right=0.97, top=0.91, bottom=0.04)

    # Panel 1: Input RGB Fundus
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title("Panel 1: Input Retinal Fundus (CLAHE Green Channel)", color='#58A6FF', fontsize=14, pad=10, fontweight='semibold')
    rect1 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='#F59E0B', facecolor='none', linestyle='--')
    ax1.add_patch(rect1)
    ax1.text(x1 + 4, y1 + 18, "ROI Zoom", color='#F59E0B', fontsize=11, fontweight='bold')
    ax1.axis('off')

    # Panel 2: Ground Truth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_bin, cmap='gray')
    ax2.set_title(f"Panel 2: Ground Truth Annotation (Vascular Tree)", color='#58A6FF', fontsize=14, pad=10, fontweight='semibold')
    rect2 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='#F59E0B', facecolor='none', linestyle='--')
    ax2.add_patch(rect2)
    ax2.axis('off')

    # Panel 3: Model [8] Prediction
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(prob_map, cmap='magma', vmin=0.0, vmax=1.0)
    ax3.set_title(f"Panel 3: Model [8] Predicted Probability Map (Threshold = 0.50)", color='#58A6FF', fontsize=14, pad=10, fontweight='semibold')
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.035, pad=0.02)
    cbar3.ax.tick_params(colors='white')
    rect3 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='#F59E0B', facecolor='none', linestyle='--')
    ax3.add_patch(rect3)
    ax3.axis('off')

    # Panel 4: Component Topology Audit (Trunk vs Floaters)
    ax4 = fig.add_subplot(gs[1, 0])
    # Build RGB map for components: Black background, Cyan trunk, Vivid Red floaters with glowing dilation
    comp_rgb = np.zeros((h, w, 3), dtype=np.float32)
    # Background faint fundus outline
    gray_bg = np.mean(img_rgb, axis=2) / 255.0 * 0.15
    comp_rgb[:, :, 0] = gray_bg
    comp_rgb[:, :, 1] = gray_bg
    comp_rgb[:, :, 2] = gray_bg

    # Main trunk: bright cyan (0.0, 0.9, 1.0)
    comp_rgb[comp_info['trunk_mask'] == 1.0] = [0.0, 0.95, 0.95]

    # Floating fragments: Dilated halo in yellow, core in bright crimson red
    dilated_floaters = morph.dilation(comp_info['floaters_mask'] > 0, morph.disk(2))
    comp_rgb[dilated_floaters & (comp_info['trunk_mask'] == 0)] = [1.0, 0.8, 0.1]
    comp_rgb[comp_info['floaters_mask'] > 0] = [1.0, 0.1, 0.3]

    ax4.imshow(comp_rgb)
    ax4.set_title(
        f"Panel 4: Topological Decomposition Audit\n"
        f"Cyan = Main Trunk ({comp_info['trunk_ratio']*100:.1f}%) | Red = {len(comp_info['floater_sizes'])} Floating Pools",
        color='#58A6FF', fontsize=14, pad=10, fontweight='semibold'
    )
    rect4 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='#F59E0B', facecolor='none', linestyle='--')
    ax4.add_patch(rect4)
    ax4.axis('off')

    # Panel 5: Pixel Error Classification Overlay
    ax5 = fig.add_subplot(gs[1, 1])
    error_rgb = np.zeros((h, w, 3), dtype=np.float32)
    # Background
    error_rgb[:, :, :] = np.expand_dims(gray_bg, axis=-1)

    tp_mask = (pred_bin == 1.0) & (gt_bin == 1.0)
    fp_mask = (pred_bin == 1.0) & (gt_bin == 0.0)
    fn_mask = (pred_bin == 0.0) & (gt_bin == 1.0)

    error_rgb[tp_mask] = [0.18, 0.80, 0.44]  # Green TP
    error_rgb[fp_mask] = [0.91, 0.30, 0.24]  # Red FP
    error_rgb[fn_mask] = [0.20, 0.60, 0.86]  # Blue FN

    ax5.imshow(error_rgb)
    ax5.set_title(
        f"Panel 5: Error Confusion Map\n"
        f"Green=TP | Red=FP Hallucination ({int(fp_mask.sum())} px) | Blue=FN Missed",
        color='#58A6FF', fontsize=14, pad=10, fontweight='semibold'
    )
    rect5 = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='#F59E0B', facecolor='none', linestyle='--')
    ax5.add_patch(rect5)
    ax5.axis('off')

    # Panel 6: High-Magnification ROI Zoom (160x160)
    ax6 = fig.add_subplot(gs[1, 2])
    roi_comp = comp_rgb[y1:y2, x1:x2].copy()
    ax6.imshow(roi_comp)
    ax6.set_title(
        f"Panel 6: Magnified ROI ({x2-x1}x{y2-y1} px) — Smear Analysis\n"
        f"Reveals 1x21 Directional Strip Texture Smear in Low-Contrast Tissue",
        color='#F59E0B', fontsize=14, pad=10, fontweight='bold'
    )
    # Annotate individual floaters in ROI
    for (cy, cx, c_sz) in comp_info['floater_centroids']:
        if y1 <= cy < y2 and x1 <= cx < x2:
            ry = cy - y1
            rx = cx - x1
            circ = patches.Circle((rx, ry), radius=max(4, int(np.sqrt(c_sz)*1.5)), linewidth=1.5, edgecolor='#FBBF24', facecolor='none')
            ax6.add_patch(circ)
            ax6.text(rx + 5, ry - 3, f"{c_sz}px", color='#FDE68A', fontsize=9, fontweight='bold')
    ax6.axis('off')

    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Diagnostic figure saved: {out_path}")


def create_comparative_roi_triptych(img_rgb, gt_bin, pred_base, pred_uni, pred_murray, img_id, out_path):
    """
    Generates a direct 4-model zoom comparison in the dense floater region:
    [GT | Model 1 Baseline | Model 7 CAD+Unified | Model 8 CAD+Murray]
    Visually proving how the 1x21 directional strip pools emerge and how Murray stabilizes them.
    """
    h, w = gt_bin.shape
    struct = ndi.generate_binary_structure(2, 2)
    _, num_m = ndi.label(pred_murray, structure=struct)
    
    comp_murray = analyze_connected_components(pred_murray, gt_bin)
    y1, y2, x1, x2 = find_densest_floater_roi(comp_murray['floater_centroids'], h, w, crop_size=180)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5.5), facecolor='#0D1117')
    fig.suptitle(
        f"Ablation Evolution of Directional Smears: STARE {img_id} (ROI Zoom {x2-x1}x{y2-y1} px)\n"
        f"Tracking Disconnected Pool Formation: Baseline vs. CAD+Unified vs. CAD+Murray",
        fontsize=16, color='white', fontweight='bold', y=0.98
    )

    panels = [
        ("Input RGB Fundus", img_rgb[y1:y2, x1:x2], None),
        ("Ground Truth", gt_bin[y1:y2, x1:x2], 'gray'),
        ("Model [1]: Baseline SA-UNetv2", pred_base[y1:y2, x1:x2], 'gray'),
        ("Model [7]: CAD+Unified (1+2)", pred_uni[y1:y2, x1:x2], 'gray'),
        ("Model [8]: CAD+Murray (1+2+3)", pred_murray[y1:y2, x1:x2], 'gray')
    ]

    for ax, (title, data, cmap) in zip(axes, panels):
        if cmap is None:
            ax.imshow(data)
        else:
            # Color floaters red
            lab, n_cc = ndi.label(data, structure=struct)
            if n_cc > 0:
                sizes = ndi.sum(data, lab, range(1, n_cc + 1))
                trunk_id = int(np.argmax(sizes) + 1)
                rgb_roi = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.float32)
                rgb_roi[lab == trunk_id] = [0.0, 0.9, 0.9]  # Cyan trunk
                rgb_roi[(data == 1.0) & (lab != trunk_id)] = [1.0, 0.15, 0.35]  # Red floaters
                ax.imshow(rgb_roi)
            else:
                ax.imshow(data, cmap=cmap)

        ax.set_title(title, color='#58A6FF', fontsize=12, pad=8, fontweight='semibold')
        ax.axis('off')

    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Evolution triptych saved: {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Diagnostic Visualizer on {device}...")

    out_dir = os.path.join(project_root, "results", "stare_final_visual_comparisons")
    os.makedirs(out_dir, exist_ok=True)

    # Load dataloader
    _, _, test_loader = get_stare_cwcldice_dataloaders(repeat=1)

    # Checkpoint paths
    ckpt_base = os.path.join(project_root, "checkpoints", "best_sa_unetv2_stare.pth")
    ckpt_uni = os.path.join(project_root, "checkpoints", "best_sa_unetv2_stare_cadtocsa_unified.pth")
    ckpt_murray = os.path.join(project_root, "checkpoints", "best_sa_unetv2_stare_cadtocsa_murray.pth")

    print("[*] Loading models...")
    model_base = load_model(ckpt_base, device)
    model_uni = load_model(ckpt_uni, device)
    model_murray = load_model(ckpt_murray, device)

    stats_summary = []

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            img_id = batch['id'][0]
            print(f"\n[*] Processing STARE test case: {img_id}...")

            # Model 8 (Murray Final)
            p8_padded = model_murray(imgs)
            p8 = restore_image_stare(p8_padded, 605, 700).squeeze().cpu().numpy()
            pred8_bin = (p8 >= 0.5).astype(np.float32)

            # Model 7 (Unified)
            p7_padded = model_uni(imgs)
            p7 = restore_image_stare(p7_padded, 605, 700).squeeze().cpu().numpy()
            pred7_bin = (p7 >= 0.5).astype(np.float32)

            # Model 1 (Baseline)
            p1_padded = model_base(imgs)
            p1 = restore_image_stare(p1_padded, 605, 700).squeeze().cpu().numpy()
            pred1_bin = (p1 >= 0.5).astype(np.float32)

            # Label and RGB
            gt_cropped = restore_image_stare(lbls, 605, 700).squeeze().cpu().numpy()
            gt_bin = (gt_cropped >= 0.5).astype(np.float32)

            # Restored input image
            raw_img = restore_image_stare(imgs, 605, 700).squeeze().permute(1, 2, 0).cpu().numpy()
            raw_img = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)

            # Analyze components
            comp_info = analyze_connected_components(pred8_bin, gt_bin)
            sizes = comp_info['floater_sizes']
            print(f"    - Total Components (Beta-0): {comp_info['num_cc']}")
            print(f"    - Main Trunk Pixels: {int(comp_info['trunk_mask'].sum())} ({comp_info['trunk_ratio']*100:.2f}% of vessel mass)")
            print(f"    - Disconnected Floaters: {len(sizes)} components")
            if len(sizes) > 0:
                print(f"    - Floater Area: Mean={np.mean(sizes):.1f}px, Median={np.median(sizes):.1f}px, Max={np.max(sizes)}px, Min={np.min(sizes)}px")
                print(f"    - Floater Precision: TP={comp_info['tp_floaters']} (true fragments), FP={comp_info['fp_floaters']} (hallucinations)")
                fp_rate = comp_info['fp_floaters'] / len(sizes) * 100.0
                print(f"    - Hallucination Rate among Floaters: {fp_rate:.1f}%")

            stats_summary.append({
                'id': img_id,
                'beta0': comp_info['num_cc'],
                'trunk_ratio': comp_info['trunk_ratio'],
                'num_floaters': len(sizes),
                'mean_size': np.mean(sizes) if len(sizes) > 0 else 0,
                'median_size': np.median(sizes) if len(sizes) > 0 else 0,
                'tp_floaters': comp_info['tp_floaters'],
                'fp_floaters': comp_info['fp_floaters']
            })

            # Render 6-panel diagnostic
            fig_path = os.path.join(out_dir, f"{img_id}_disconnected_pools_diagnostic.png")
            create_diagnostic_figure(raw_img, gt_bin, pred8_bin, p8, comp_info, img_id, fig_path)

            # Render evolutionary triptych
            triptych_path = os.path.join(out_dir, f"{img_id}_ablation_evolution_roi.png")
            create_comparative_roi_triptych(raw_img, gt_bin, pred1_bin, pred7_bin, pred8_bin, img_id, triptych_path)

    # Print summary statistics table
    print("\n" + "=" * 80)
    print("SUMMARY AUDIT: DISCONNECTED POOLS ACROSS ALL STARE TEST SAMPLES")
    print("=" * 80)
    print(f"{'Image ID':<10} | {'Beta-0':<8} | {'Trunk %':<10} | {'Floaters':<10} | {'Mean Sz':<10} | {'TP Floaters':<12} | {'FP Floaters':<12}")
    print("-" * 80)
    for s in stats_summary:
        print(f"{s['id']:<10} | {s['beta0']:<8} | {s['trunk_ratio']*100:<9.1f}% | {s['num_floaters']:<10} | {s['mean_size']:<9.1f}px | {s['tp_floaters']:<12} | {s['fp_floaters']:<12}")
    print("=" * 80)


if __name__ == "__main__":
    main()
