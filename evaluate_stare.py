"""
Evaluation script for SA-UNetv2 on the STARE dataset test set.
Computes F1, Jaccard, Sensitivity, Specificity, Accuracy, MCC, and AUC-ROC,
and compares them directly against Table 2 of the ISBI 2026 paper.
Generates binary prediction masks, glowing green overlays, and visual comparisons.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from model import SA_UNetv2
from stare_dataset import get_stare_dataloaders, restore_image_stare

def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    y_true_flat = y_true.flatten().astype(np.float32)
    y_prob_flat = y_pred_prob.flatten().astype(np.float32)
    y_pred_bin = (y_prob_flat >= threshold).astype(np.float32)

    tp = np.sum((y_pred_bin == 1.0) & (y_true_flat == 1.0))
    tn = np.sum((y_pred_bin == 0.0) & (y_true_flat == 0.0))
    fp = np.sum((y_pred_bin == 1.0) & (y_true_flat == 0.0))
    fn = np.sum((y_pred_bin == 0.0) & (y_true_flat == 1.0))

    sen = tp / (tp + fn + 1e-8)
    spe = tn / (tn + fp + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    jacc = tp / (tp + fp + fn + 1e-8)

    denom = (np.sqrt(float(tp + fp)) * 
             np.sqrt(float(tp + fn)) * 
             np.sqrt(float(tn + fp)) * 
             np.sqrt(float(tn + fn)))
    mcc = float((float(tp) * float(tn) - float(fp) * float(fn)) / (denom + 1e-8))

    try:
        auc = roc_auc_score(y_true_flat, y_prob_flat)
    except Exception:
        auc = 0.5

    return {
        'f1': float(f1),
        'jaccard': float(jacc),
        'sensitivity': float(sen),
        'specificity': float(spe),
        'accuracy': float(acc),
        'mcc': float(mcc),
        'auc': float(auc)
    }

def create_overlay(orig_rgb, pred_bin):
    """
    Generate diagnostic glowing neon green vessel overlay on original eye.
    """
    overlay = orig_rgb.copy()
    green_mask = (pred_bin == 1.0)
    # Tint vessel pixels green: keep red/blue reduced, boost green
    overlay[green_mask, 0] = (overlay[green_mask, 0] * 0.2).astype(np.uint8)
    overlay[green_mask, 1] = np.clip(overlay[green_mask, 1] * 0.5 + 180, 0, 255).astype(np.uint8)
    overlay[green_mask, 2] = (overlay[green_mask, 2] * 0.2).astype(np.uint8)
    return overlay

def evaluate_stare_model(
    checkpoint_path="checkpoints/best_sa_unetv2_stare.pth",
    benchmark_dir="Stare data/benchmark_20",
    results_dir="results/stare",
    threshold=0.5,
    device_name="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_name)
    os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "overlays"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "visual_comparisons"), exist_ok=True)

    print("=" * 65)
    print(" Evaluating SA-UNetv2 on STARE Dataset (Test Set)")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Device: {device_name.upper()}")
    print("=" * 65)

    # 1. Load Model
    model = SA_UNetv2(in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"[+] Loaded weights from Epoch {ckpt.get('epoch', 'N/A')} (Val Loss: {ckpt.get('val_loss', 'N/A'):.4f})")

    # 2. Get Test DataLoader
    _, _, test_loader = get_stare_dataloaders(benchmark_dir=benchmark_dir)
    print(f"[*] Total test images to evaluate: {len(test_loader.dataset)}")

    metrics_list = []

    with torch.no_grad():
        for batch in test_loader:
            img_tensor = batch['image'].to(device)
            lbl_tensor = batch['label'].to(device)
            img_id = batch['id'][0]

            # Forward pass (704 x 704)
            preds_padded = model(img_tensor)
            preds_cropped = restore_image_stare(preds_padded, orig_h=605, orig_w=700)
            lbls_cropped = restore_image_stare(lbl_tensor, orig_h=605, orig_w=700)
            imgs_cropped = restore_image_stare(img_tensor, orig_h=605, orig_w=700)

            pred_prob = preds_cropped.squeeze().cpu().numpy()
            lbl_np = lbls_cropped.squeeze().cpu().numpy()
            orig_rgb = (imgs_cropped.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)

            m = compute_metrics(lbl_np, pred_prob, threshold=threshold)
            metrics_list.append(m)

            # Save Visual Outputs
            pred_bin = (pred_prob >= threshold).astype(np.float32)
            pred_img = Image.fromarray((pred_bin * 255.0).astype(np.uint8))
            pred_path = os.path.join(results_dir, "predictions", f"{img_id}_pred.png")
            pred_img.save(pred_path)

            overlay_arr = create_overlay(orig_rgb, pred_bin)
            overlay_img = Image.fromarray(overlay_arr)
            overlay_path = os.path.join(results_dir, "overlays", f"{img_id}_overlay.png")
            overlay_img.save(overlay_path)

            # 4-Panel comparison: Image | Ground Truth | Prediction | Overlay
            gt_rgb = np.stack([(lbl_np * 255).astype(np.uint8)] * 3, axis=-1)
            pred_rgb = np.stack([(pred_bin * 255).astype(np.uint8)] * 3, axis=-1)
            four_panel = np.concatenate([orig_rgb, gt_rgb, pred_rgb, overlay_arr], axis=1)
            four_panel_img = Image.fromarray(four_panel)
            four_panel_path = os.path.join(results_dir, "visual_comparisons", f"{img_id}_comparison.png")
            four_panel_img.save(four_panel_path)

            print(f"  --> Processed {img_id}: F1={m['f1']*100:.2f}%, Jacc={m['jaccard']*100:.2f}%, Sen={m['sensitivity']*100:.2f}%, Spe={m['specificity']*100:.2f}%")

    # Aggregate
    avg_m = {k: float(np.mean([m[k] for m in metrics_list])) for k in metrics_list[0]}

    # Table 2 Paper Reported Numbers on STARE
    paper_stare = {
        'f1': 0.8281,
        'jaccard': 0.7082,
        'sensitivity': 0.8535,
        'specificity': 0.9871,
        'accuracy': 0.9783,
        'mcc': 0.8179,
        'auc': 0.9913
    }

    print("\n" + "=" * 80)
    print(f" FINAL BENCHMARK COMPARISON ON STARE DATASET (Threshold: {threshold})")
    print("=" * 80)
    header = f"{'Metric':<25} | {'Paper (Table 2)':<16} | {'Reproduced (Ours)':<18} | {'Delta':<10}"
    print(header)
    print("-" * 80)

    for k, name in [
        ('f1', 'F1-Score / Dice'),
        ('jaccard', 'Jaccard Index (IoU)'),
        ('sensitivity', 'Sensitivity (Recall)'),
        ('specificity', 'Specificity'),
        ('accuracy', 'Accuracy'),
        ('mcc', 'Matthews Corr (MCC)'),
        ('auc', 'AUC-ROC')
    ]:
        p_val = paper_stare[k] * 100
        o_val = avg_m[k] * 100
        delta = o_val - p_val
        print(f"{name:<25} | {p_val:>14.2f}% | {o_val:>16.2f}% | {delta:>+8.2f}%")

    print("=" * 80)
    print(f"[+] All prediction masks saved to: {os.path.join(results_dir, 'predictions')}")
    print(f"[+] Glowing green overlays saved to: {os.path.join(results_dir, 'overlays')}")
    print(f"[+] 4-Panel comparison images saved to: {os.path.join(results_dir, 'visual_comparisons')}")
    return avg_m

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SA-UNetv2 on STARE")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_sa_unetv2_stare.pth")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    evaluate_stare_model(checkpoint_path=args.checkpoint, threshold=args.threshold)
