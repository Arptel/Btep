"""
Visual Comparison: Final Combined Synthesis (Step 4)
====================================================
Section 7, Step 4: Generates high-resolution comparative fundus ROI figures
visually proving how combining h=3 Transverse Contrast + Orthogonal Inhibition
extinguishes linear streak floaters while sustaining peak micro-capillary continuity.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
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
    use_ortho = any('gate_h' in k for k in state_dict.keys())
    model = SA_UNetv2(
        in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.0,
        use_cad_topo_csa=use_cad, strip_height=strip_h,
        use_orthogonal_inhibition=use_ortho
    ).to(device)
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
    ckpt_murray = os.path.join(project_root, "checkpoints", "best_sa_unetv2_stare_cadtocsa_murray.pth")
    ckpt_final = os.path.join(project_root, "checkpoints", "best_sa_unetv2_stare_final_synthesis.pth")

    if not os.path.exists(ckpt_final):
        print(f"Error: {ckpt_final} does not exist yet. Please wait for training to finish.")
        return

    model_base = load_model(ckpt_base, device)
    model_h1 = load_model(ckpt_h1, device)
    model_murray = load_model(ckpt_murray, device)
    model_final = load_model(ckpt_final, device)

    struct = ndi.generate_binary_structure(2, 2)

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            img_id = batch['id'][0]

            p_base = (restore_image_stare(model_base(imgs), 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)
            p_h1 = (restore_image_stare(model_h1(imgs), 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)
            p_murray = (restore_image_stare(model_murray(imgs), 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)
            p_final = (restore_image_stare(model_final(imgs), 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)

            gt_bin = (restore_image_stare(lbls, 605, 700).squeeze().cpu().numpy() >= 0.5).astype(np.float32)
            raw_img = restore_image_stare(imgs, 605, 700).squeeze().permute(1, 2, 0).cpu().numpy()
            raw_img = np.clip(raw_img * 255.0, 0, 255).astype(np.uint8)

            # Center ROI on floater region where h=1 creates hallucinations
            lab_h1, num_h1 = ndi.label(p_h1, structure=struct)
            sizes_h1 = ndi.sum(p_h1, lab_h1, range(1, num_h1 + 1))
            trunk_id = int(np.argmax(sizes_h1) + 1)
            floaters_h1 = (p_h1 == 1.0) & (lab_h1 != trunk_id)
            
            if np.sum(floaters_h1) > 0:
                cy, cx = ndi.center_of_mass(floaters_h1)
                cy, cx = int(cy), int(cx)
            else:
                cy, cx = 300, 350

            crop = 180
            y1 = max(0, min(605 - crop, cy - crop // 2))
            x1 = max(0, min(700 - crop, cx - crop // 2))
            y2, x2 = y1 + crop, x1 + crop

            fig, axes = plt.subplots(1, 6, figsize=(28, 6.0), facecolor='#0D1117')
            fig.suptitle(
                f"STARE {img_id}: Step 4 Final Combined Synthesis (Idea 1 + Idea 2 (h=3) + Idea 3 + Idea 4)\n"
                f"Clean Tree Continuity via Transverse-Contrast & Cross-Strip Suppression (ROI Zoom {crop}x{crop} px)",
                fontsize=15, color='white', fontweight='bold', y=0.99
            )

            panels = [
                ("Input RGB Fundus", raw_img[y1:y2, x1:x2], None),
                ("Ground Truth GT", gt_bin[y1:y2, x1:x2], 'gray'),
                ("Baseline (ISBI 2026)\n(Fragmented Endings)", p_base[y1:y2, x1:x2], 'gray'),
                ("CAD-Topo-CSA h=1\n(Spurious 1x21 Smears)", p_h1[y1:y2, x1:x2], 'inferno'),
                ("CAD + Murray (1+2+3)\n(Uninhibited h=1)", p_murray[y1:y2, x1:x2], 'viridis'),
                ("Final Combined Synthesis\n(1+2 [h=3] + 3 + 4 [Ortho])", p_final[y1:y2, x1:x2], 'magma')
            ]

            for ax, (title, img_data, cmap) in zip(axes, panels):
                if cmap is None:
                    ax.imshow(img_data)
                else:
                    ax.imshow(img_data, cmap=cmap)
                ax.set_title(title, fontsize=11, color='#58A6FF', fontweight='bold', pad=12)
                ax.axis('off')
                for spine in ax.spines.values():
                    spine.set_color('#30363D')
                    spine.set_linewidth(1.5)

            plt.tight_layout(rect=[0, 0, 1, 0.92])
            save_path = os.path.join(out_dir, f"{img_id}_final_synthesis_diagnostic.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
            plt.close()
            print(f"[+] Saved diagnostic visualization: {save_path}")


if __name__ == '__main__':
    main()
