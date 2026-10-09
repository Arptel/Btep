"""
Visual Comparison: Strip Height Ablation (h=1 vs h=3 vs h=5)
============================================================
Section 7, Step 2: Generates high-resolution comparative fundus ROI figures
visually proving how h=3 transverse contrast extinguishes linear streak artifacts.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage as ndi
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "baseline_reproduction", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from model import SA_UNetv2
from stare_dataset import get_stare_cwcldice_dataloaders, restore_image_stare


def load_model(checkpoint_path, device):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(project_root, "results", "stare_final_visual_comparisons")
    os.makedirs(out_dir, exist_ok=True)

    _, _, test_loader = get_stare_cwcldice_dataloaders(repeat=1)

    ckpt_base = os.path.join(project_root, "checkpoints", "best_sa_unetv2_stare.pth")
    ckpt_h1 = os.path.join(project_root, "checkpoints", "best_sa_unetv2_stare_cadtocsa.pth")
    ckpt_h3 = os.path.join(project_root, "checkpoints", "best_sa_unetv2_stare_cadtocsa_h3.pth")
    ckpt_h5 = os.path.join(project_root, "checkpoints", "best_sa_unetv2_stare_cadtocsa_h5.pth")

    model_base = load_model(ckpt_base, device)
    model_h1 = load_model(ckpt_h1, device)
    model_h3 = load_model(ckpt_h3, device)
    model_h5 = load_model(ckpt_h5, device)

    struct = ndi.generate_binary_structure(2, 2)

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            img_id = batch['id'][0]

            # Inferences
            p_base = (restore_image_stare(model_base(imgs), 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)
            p_h1 = (restore_image_stare(model_h1(imgs), 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)
            p_h3 = (restore_image_stare(model_h3(imgs), 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)
            p_h5 = (restore_image_stare(model_h5(imgs), 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)

            gt_bin = (restore_image_stare(lbls, 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)
            raw_img = restore_image_stare(imgs, 605, 700).squeeze().permute(1, 2, 0).cpu().numpy()
            raw_img = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)

            # Center ROI on floater region
            lab_h1, num_h1 = ndi.label(p_h1, structure=struct)
            sizes_h1 = ndi.sum(p_h1, lab_h1, range(1, num_h1 + 1))
            trunk_id = int(np.argmax(sizes_h1) + 1)
            floaters_h1 = (p_h1 == 1.0) & (lab_h1 != trunk_id)
            
            # Find center of mass of floaters
            if np.sum(floaters_h1) > 0:
                cy, cx = ndi.center_of_mass(floaters_h1)
                cy, cx = int(cy), int(cx)
            else:
                cy, cx = 300, 350

            crop = 180
            y1 = max(0, min(605 - crop, cy - crop // 2))
            x1 = max(0, min(700 - crop, cx - crop // 2))
            y2, x2 = y1 + crop, x1 + crop

            fig, axes = plt.subplots(1, 6, figsize=(28, 5.5), facecolor='#0D1117')
            fig.suptitle(
                f"STARE {img_id}: Strip Height Search (h in {{1, 3, 5}}) Transverse Contrast Ablation\n"
                f"Proving h=3 Extinguishes 1x21 Directional Smear Artifacts (ROI Zoom {crop}x{crop} px)",
                fontsize=16, color='white', fontweight='bold', y=0.98
            )

            panels = [
                ("Input RGB Fundus", raw_img[y1:y2, x1:x2], None),
                ("Ground Truth", gt_bin[y1:y2, x1:x2], 'gt'),
                ("Baseline SA-UNetv2", p_base[y1:y2, x1:x2], 'pred'),
                ("h=1 (1x21 Strip)", p_h1[y1:y2, x1:x2], 'pred'),
                ("h=3 (3x21 Strip) [Optimal]", p_h3[y1:y2, x1:x2], 'pred'),
                ("h=5 (5x21 Strip) [Over-wide]", p_h5[y1:y2, x1:x2], 'pred'),
            ]

            for ax, (title, data, mode) in zip(axes, panels):
                if mode is None:
                    ax.imshow(data)
                elif mode == 'gt':
                    ax.imshow(data, cmap='gray')
                else:
                    lab, n_cc = ndi.label(data, structure=struct)
                    rgb_roi = np.zeros((crop, crop, 3), dtype=np.float32)
                    if n_cc > 0:
                        sizes = ndi.sum(data, lab, range(1, n_cc + 1))
                        t_id = int(np.argmax(sizes) + 1)
                        rgb_roi[lab == t_id] = [0.0, 0.9, 0.9]  # Cyan
                        rgb_roi[(data == 1.0) & (lab != t_id)] = [1.0, 0.15, 0.35]  # Red floaters
                    ax.imshow(rgb_roi)

                color = '#10B981' if '[Optimal]' in title else ('#EF4444' if '[Over-wide]' in title else '#58A6FF')
                ax.set_title(title, color=color, fontsize=12, pad=8, fontweight='semibold')
                ax.axis('off')

            out_path = os.path.join(out_dir, f"{img_id}_strip_height_ablation_roi.png")
            plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
            plt.close(fig)
            print(f"[OK] Saved: {out_path}")

    print("\n[OK] All strip height ablation visualizations rendered successfully!")


if __name__ == '__main__':
    main()
