"""
Stage A-C: Binarize, Clean, Skeletonize, Distance Transform, and Graph Extraction via sknw.
"""
import os
import glob
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import skimage.morphology as morph
import sknw
import networkx as nx
import matplotlib.pyplot as plt


def binarize_and_clean(prob_map, fov_mask=None, threshold=0.5, min_cc_size=10):
    """
    Stage A: Threshold probability map, apply FOV mask, and filter small CC noise.
    """
    binary = (prob_map >= threshold).astype(bool)
    if fov_mask is not None:
        binary = binary & (fov_mask > 0)

    # Remove small CCs (< min_cc_size)
    lbl, num = ndi.label(binary, structure=np.ones((3, 3), dtype=int))
    if num > 0:
        sizes = ndi.sum(binary, lbl, range(1, num + 1))
        small_mask = np.isin(lbl, np.where(sizes < min_cc_size)[0] + 1)
        binary[small_mask] = False

    return binary


def prune_tiny_skeleton_spurs(skeleton, max_spur_length=5):
    """
    Prunes tiny skeleton artifact spurs (< 5px) created by morphological skeletonization.
    """
    skel = skeleton.copy()
    for _ in range(max_spur_length):
        # Endpoints have exactly 1 neighbor in 8-neighborhood
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=int)
        neighbor_count = ndi.convolve(skel.astype(int), kernel, mode='constant', cval=0)
        endpoints = (skel) & (neighbor_count == 1)
        if not np.any(endpoints):
            break
        skel[endpoints] = False
    return skel


def extract_vessel_graph(prob_map, fov_mask=None, threshold=0.5, min_cc_size=10, min_spur_len=4):
    """
    Stages A-C: Full pipeline from probability map to annotated networkx graph.
    """
    # Stage A
    binary_mask = binarize_and_clean(prob_map, fov_mask, threshold, min_cc_size)

    # Stage B
    raw_skeleton = morph.skeletonize(binary_mask)
    radius_map = ndi.distance_transform_edt(binary_mask)
    clean_skeleton = prune_tiny_skeleton_spurs(raw_skeleton, max_spur_length=min_spur_len)

    # Stage C: Graph via sknw
    graph = sknw.build_sknw(clean_skeleton.astype(np.uint16), multi=True)

    # Annotate edges with length, radius (min & mean), and border touch
    H, W = binary_mask.shape
    for (u, v, k) in list(graph.edges(keys=True)):
        edge_data = graph[u][v][k]
        pts = edge_data['pts']  # Array of (y, x) coordinates

        # 1. Length calculation (Euclidean step summation)
        if len(pts) > 1:
            diffs = np.diff(pts, axis=0)
            step_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
            L_e = float(np.sum(step_lengths))
        else:
            L_e = 1.0
        edge_data['length'] = max(L_e, 0.5)

        # 2. Radius calculation (min & mean along path)
        y_coords = pts[:, 0].clip(0, H - 1)
        x_coords = pts[:, 1].clip(0, W - 1)
        r_vals = radius_map[y_coords, x_coords]
        edge_data['radius_min'] = float(np.min(r_vals)) if len(r_vals) > 0 else 0.5
        edge_data['radius_mean'] = float(np.mean(r_vals)) if len(r_vals) > 0 else 0.5

        # 3. Border touching check (endpoint near image boundary or FOV edge)
        touches_border = False
        margin = 3
        for y, x in [pts[0], pts[-1]]:
            if y <= margin or y >= H - 1 - margin or x <= margin or x >= W - 1 - margin:
                touches_border = True
            elif fov_mask is not None and fov_mask[int(y), int(x)] == 0:
                touches_border = True
        edge_data['touches_border'] = touches_border

    return binary_mask, clean_skeleton, radius_map, graph


def render_stage_ac_overlay(orig_img, binary_mask, skeleton, graph, save_path, title="Vessel Graph Extraction"):
    """
    Renders 4-panel visual verification overlay:
    Panel 1: Original Image
    Panel 2: Cleaned Binary Mask (Stage A)
    Panel 3: Skeleton with Radius Map (Stage B)
    Panel 4: Extracted sknw Graph Overlay (Stage C)
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), dpi=150)
    fig.patch.set_facecolor('#141414')

    # Panel 1: Original Image
    axes[0].imshow(orig_img)
    axes[0].set_title("1. Original Fundus Image", color='white', fontsize=13, pad=10)
    axes[0].axis('off')

    # Panel 2: Cleaned Binary Mask
    axes[1].imshow(binary_mask, cmap='gray')
    axes[1].set_title(f"2. Stage A: Binary Mask ({np.sum(binary_mask):,} px)", color='white', fontsize=13, pad=10)
    axes[1].axis('off')

    # Panel 3: Skeleton & Distance Transform
    axes[2].imshow(skeleton, cmap='gray')
    axes[2].set_title(f"3. Stage B: Skeleton ({np.sum(skeleton):,} px)", color='white', fontsize=13, pad=10)
    axes[2].axis('off')

    # Panel 4: Graph Overlay
    axes[3].imshow(np.zeros_like(binary_mask), cmap='gray')
    # Draw edges
    for (u, v, k) in graph.edges(keys=True):
        pts = graph[u][v][k]['pts']
        axes[3].plot(pts[:, 1], pts[:, 0], color='#00ff88', linewidth=1.2, alpha=0.85)

    # Draw nodes
    for node in graph.nodes():
        ps = graph.nodes[node]['o']  # centroid/coordinate (y, x)
        deg = graph.degree(node)
        if deg == 1:
            # Endpoint (Cyan)
            axes[3].plot(ps[1], ps[0], 'o', color='#00d9ff', markersize=3.5, alpha=0.9)
        elif deg >= 3:
            # Junction (Red/Orange)
            axes[3].plot(ps[1], ps[0], 'o', color='#ff3366', markersize=4.5, alpha=0.95)

    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    axes[3].set_title(f"4. Stage C: Graph ({num_nodes} nodes, {num_edges} edges)", color='white', fontsize=13, pad=10)
    axes[3].axis('off')

    plt.suptitle(title, color='#38bdf8', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"[+] Saved Stage A-C overlay to: {save_path}")


def generate_5_stage_ac_overlays(prob_dir="edits/idea1_betweenness_module/prob_maps/drive/test",
                                 data_dir="Drive/DRIVE/test",
                                 out_dir="edits/idea1_betweenness_module/results/stage_ac_overlays"):
    import io_utils
    prob_dir = io_utils.find_path(prob_dir)
    data_dir = io_utils.find_path(data_dir)
    os.makedirs(out_dir, exist_ok=True)

    prob_files = sorted(glob.glob(os.path.join(prob_dir, "*_prob.npy")))[:5]
    print(f"[*] Generating Stage A-C inspection overlays for {len(prob_files)} images...")

    for pf in prob_files:
        basename = os.path.basename(pf).replace('_prob.npy', '')
        num = basename.split('_')[0]

        # Load original image and FOV mask
        img_path = os.path.join(data_dir, "images", f"{num}_test.tif")
        if not os.path.exists(img_path):
            img_path = glob.glob(os.path.join(data_dir, "images", f"{num}_*.*"))[0]
        orig_img = Image.open(img_path).convert('RGB')

        mask_path = os.path.join(data_dir, "mask", f"{num}_test_mask.gif")
        if not os.path.exists(mask_path):
            mask_path = glob.glob(os.path.join(data_dir, "mask", f"{num}_*.*"))[0]
        fov_mask = np.array(Image.open(mask_path)) > 0

        prob_map = np.load(pf)

        # Run Stage A-C
        binary_mask, skeleton, radius_map, graph = extract_vessel_graph(prob_map, fov_mask)

        # Render inspection overlay
        save_path = os.path.join(out_dir, f"{basename}_graph_overlay.png")
        render_stage_ac_overlay(orig_img, binary_mask, skeleton, graph, save_path, 
                                title=f"Stage A-C Inspection: {basename} (Graph Parity)")

    print(f"[+] All 5 inspection overlays successfully generated in: {out_dir}")


if __name__ == "__main__":
    generate_5_stage_ac_overlays()
