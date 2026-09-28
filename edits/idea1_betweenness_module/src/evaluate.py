import os
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, recall_score, f1_score, jaccard_score, matthews_corrcoef
from dataset import restore_image


def compute_metrics_flat(y_true_flat, y_pred_prob_flat, threshold=0.5):
    """
    Computes all standard metrics given 1D flattened ground truth and probabilities.
    """
    y_pred_bin = (y_pred_prob_flat >= threshold).astype(np.uint8)
    y_true_bin = (y_true_flat >= 0.5).astype(np.uint8)

    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    sen = tp / (tp + fn + 1e-7)
    spe = tn / (tn + fp + 1e-7)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
    jacc = tp / (tp + fp + fn + 1e-7)
    
    # MCC
    mcc_denom = np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn))
    mcc = (tp * tn - fp * fn) / (mcc_denom + 1e-7)

    # AUC
    try:
        auc = roc_auc_score(y_true_bin, y_pred_prob_flat)
    except Exception:
        auc = 0.0

    return {
        'f1': f1 * 100.0,
        'jacc': jacc * 100.0,
        'sen': sen * 100.0,
        'spe': spe * 100.0,
        'acc': acc * 100.0,
        'mcc': mcc * 100.0,
        'auc': auc * 100.0
    }


def evaluate_model(model, dataloader, device, output_dir=None, threshold=0.5):
    """
    Evaluates model across test dataset, computing metrics both
    WITHOUT FOV MASK and WITH FOV MASK as in the paper.
    """
    model.eval()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

    metrics_no_fov = []
    metrics_with_fov = []

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            images = sample['image'].to(device)
            labels = sample['label'].numpy()  # (B, 1, 592, 592)
            masks = sample.get('mask', None)
            if masks is not None:
                masks = masks.numpy()

            probs = model(images).cpu().numpy()  # (B, 1, 592, 592)

            for b in range(images.size(0)):
                # Crop back to original DRIVE resolution 584x565
                prob_crop = probs[b, 0, :584, :565]
                label_crop = labels[b, 0, :584, :565]
                mask_crop = masks[b, 0, :584, :565] if masks is not None else None

                # Protocol A: Without FOV mask (full 584x565)
                m_no_fov = compute_metrics_flat(label_crop.ravel(), prob_crop.ravel(), threshold)
                metrics_no_fov.append(m_no_fov)

                # Protocol B: With FOV mask
                if mask_crop is not None:
                    fov_indices = np.where(mask_crop.ravel() > 0.5)
                    gt_fov = label_crop.ravel()[fov_indices]
                    pred_fov = prob_crop.ravel()[fov_indices]
                    m_fov = compute_metrics_flat(gt_fov, pred_fov, threshold)
                    metrics_with_fov.append(m_fov)

                # Save visual prediction & overlay
                if output_dir:
                    fname = sample['filename'][b]
                    base_name = os.path.splitext(fname)[0]
                    pred_bin = ((prob_crop >= threshold) * 255).astype(np.uint8)
                    Image.fromarray(pred_bin).save(os.path.join(output_dir, "predictions", f"{base_name}_pred.png"))

                    # Generate color overlay
                    raw_img = (sample['image'][b, :, :584, :565].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    overlay = raw_img.copy()
                    overlay[pred_bin > 0] = [0, 255, 60]
                    Image.fromarray(overlay).save(os.path.join(output_dir, "overlays", f"{base_name}_overlay.png"))

    # Compute averages
    avg_no_fov = {k: np.mean([m[k] for m in metrics_no_fov]) for k in metrics_no_fov[0].keys()}
    avg_with_fov = {}
    if metrics_with_fov:
        avg_with_fov = {k: np.mean([m[k] for m in metrics_with_fov]) for k in metrics_with_fov[0].keys()}

    return avg_no_fov, avg_with_fov
