"""
Stage F: Soft mode (Heatmap overlays & CSV exports) and Hard mode (Pruning by threshold tau).
"""
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from graph_build import extract_vessel_graph
from scoring import score_edges


def generate_soft_heatmap(binary_mask, graph, edge_df):
    """
    Paints each edge's skeleton pixels with its normalized betweenness score CB_hat.
    Ensures all segmented vessels have a baseline floor so spurs/fragments are visible.
    """
    H, W = binary_mask.shape
    # Baseline floor of 0.05 for all vessel pixels so low-scoring spurs are visible as deep purple
    heatmap = np.zeros((H, W), dtype=np.float32)

    for _, row in edge_df.iterrows():
        pts = row['pts']
        # CB_hat ranges from 0.0 to 1.0; map to [0.05, 1.0] for visible vessel pixels
        cb = float(row['CB_hat'])
        val = 0.05 + 0.95 * cb
        r = max(1, int(round(row['radius_mean'])))

        for (y, x) in pts:
            y_min, y_max = max(0, y - r), min(H, y + r + 1)
            x_min, x_max = max(0, x - r), min(W, x + r + 1)
            mask_patch = binary_mask[y_min:y_max, x_min:x_max]
            heatmap[y_min:y_max, x_min:x_max] = np.where(
                mask_patch, 
                np.maximum(heatmap[y_min:y_max, x_min:x_max], val), 
                heatmap[y_min:y_max, x_min:x_max]
            )

    return heatmap


def prune_edges_hard(binary_mask, graph, edge_df, tau=0.1, protect_border=True):
    """
    Hard mode: Remove edges with CB_hat < tau, setting their pixels to background (False).
    Edges touching the border are protected from pruning if protect_border is True.
    """
    pruned_mask = binary_mask.copy()
    pruned_edge_count = 0
    protected_edge_count = 0

    for _, row in edge_df.iterrows():
        cb = float(row['CB_hat'])
        touches_border = bool(row['touches_border'])

        if cb < tau:
            if protect_border and touches_border:
                protected_edge_count += 1
                continue

            pruned_edge_count += 1
            pts = row['pts']
            r = max(1, int(round(row['radius_mean'])))
            H, W = pruned_mask.shape

            for (y, x) in pts:
                y_min, y_max = max(0, y - r), min(H, y + r + 1)
                x_min, x_max = max(0, x - r), min(W, x + r + 1)
                pruned_mask[y_min:y_max, x_min:x_max] = False

    return pruned_mask, pruned_edge_count, protected_edge_count


def render_betweenness_heatmap(orig_img, binary_mask, heatmap, save_path, title="Current-Flow Betweenness Heatmap"):
    """
    Renders 3-panel visualization:
    1. Original Image
    2. Baseline Binary Mask
    3. Structural Validity Heatmap (CB_hat) overlaid with colorbar (High=Yellow/Red, Low=Deep Purple)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.patch.set_facecolor('#121212')

    # Panel 1: Original
    axes[0].imshow(orig_img)
    axes[0].set_title("1. Original Fundus Image", color='white', fontsize=13, pad=10)
    axes[0].axis('off')

    # Panel 2: Baseline Mask
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title(f"2. Baseline Vessel Mask", color='white', fontsize=13, pad=10)
    axes[1].axis('off')

    # Panel 3: Heatmap Overlay on Dark Background
    cmap = plt.colormaps['plasma'].copy()
    cmap.set_under('#000000')

    # Mask out non-vessel background pixels (< 0.01)
    masked_hm = np.ma.masked_where(heatmap < 0.01, heatmap)
    axes[2].imshow(np.zeros_like(binary_mask), cmap='gray')
    im = axes[2].imshow(masked_hm, cmap=cmap, vmin=0.05, vmax=1.0)
    axes[2].set_title("3. Betweenness Validity: Bright=Trunk, Dark=Spur", color='white', fontsize=13, pad=10)
    axes[2].axis('off')

    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    cbar.set_label('Normalized Current-Flow Betweenness', color='white', fontsize=11, labelpad=8)

    plt.suptitle(title, color='#38bdf8', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"[+] Saved betweenness heatmap to: {save_path}")


def generate_5_stage_e_heatmaps(prob_dir="edits/idea1_betweenness_module/prob_maps/drive/test",
                                data_dir="Drive/DRIVE/test",
                                out_dir="edits/idea1_betweenness_module/results/stage_e_heatmaps"):
    prob_dir = os.path.abspath(prob_dir)
    data_dir = os.path.abspath(data_dir)
    os.makedirs(out_dir, exist_ok=True)

    prob_files = sorted(glob.glob(os.path.join(prob_dir, "*_prob.npy")))[:5]
    print(f"[*] Generating Stage E Betweenness Heatmaps for {len(prob_files)} images...")

    for pf in prob_files:
        basename = os.path.basename(pf).replace('_prob.npy', '')
        num = basename.split('_')[0]

        # Load image & FOV mask
        img_path = os.path.join(data_dir, "images", f"{num}_test.tif")
        orig_img = Image.open(img_path).convert('RGB')

        mask_path = os.path.join(data_dir, "mask", f"{num}_test_mask.gif")
        fov_mask = np.array(Image.open(mask_path)) > 0

        prob_map = np.load(pf)

        # Graph extraction (Stages A-C)
        binary_mask, skeleton, radius_map, graph = extract_vessel_graph(prob_map, fov_mask)

        # Edge scoring (Stages D-E)
        edge_df = score_edges(graph, config={'radius_mode': 'min', 'use_conductance': True})

        # Save edge CSV
        csv_path = os.path.join(out_dir, f"{basename}_edges.csv")
        edge_df.drop(columns=['pts']).to_csv(csv_path, index=False)

        # Generate Soft Heatmap
        heatmap = generate_soft_heatmap(binary_mask, graph, edge_df)

        # Render visual figure
        save_path = os.path.join(out_dir, f"{basename}_betweenness_heatmap.png")
        render_betweenness_heatmap(orig_img, binary_mask, heatmap, save_path, 
                                   title=f"Stage E: Current-Flow Betweenness Heatmap ({basename})")

    print(f"[+] All 5 Stage E heatmaps successfully generated in: {out_dir}")


if __name__ == "__main__":
    generate_5_stage_e_heatmaps()
