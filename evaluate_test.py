import os
import sys
import time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, 'src')
from model import SA_UNetv2
from dataset import get_drive_datasets
from evaluate import evaluate_model


def run_evaluation(
    model_path="checkpoints/best_sa_unetv2.pth",
    data_dir="Drive/DRIVE",
    output_dir="results",
    threshold=0.5
):
    print("=" * 70)
    print(" EVALUATING REPRODUCED SA-UNetv2 ON TEST SET (20 Images)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Accelerator: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint '{model_path}' not found!")
        return

    # Load model
    model = SA_UNetv2().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint trained to Epoch {checkpoint.get('epoch', 'N/A')} (Best Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f})")

    # Load test set
    _, _, test_ds = get_drive_datasets(data_dir)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Evaluating {len(test_ds)} test images...")

    start_lat = time.time()
    res_no_fov, res_with_fov = evaluate_model(
        model, test_loader, device, output_dir=output_dir, threshold=threshold
    )
    total_inf_time = time.time() - start_lat
    avg_inf_per_img = total_inf_time / len(test_ds)

    print("\n" + "=" * 70)
    print(" REPRODUCTION BENCHMARK VS. ISBI 2026 PAPER (Table 1)")
    print("=" * 70)
    
    print("\n[Protocol A: WITHOUT FOV Mask]")
    print(f"{'Metric':<18} | {'Paper (SA-UNetv2)':<18} | {'Reproduced (Ours)':<18} | {'Difference':<12}")
    print("-" * 72)
    paper_no_fov = {
        'F1 (%)': (82.82, res_no_fov['f1']),
        'Jaccard (%)': (70.69, res_no_fov['jacc']),
        'Sensitivity (%)': (83.64, res_no_fov['sen']),
        'Specificity (%)': (98.28, res_no_fov['spe']),
        'Accuracy (%)': (96.98, res_no_fov['acc']),
        'MCC (%)': (81.27, res_no_fov['mcc']),
        'AUC (%)': (98.71, res_no_fov['auc'])
    }
    for m_name, (p_val, r_val) in paper_no_fov.items():
        diff = r_val - p_val
        sign = "+" if diff >= 0 else ""
        print(f"{m_name:<18} | {p_val:<18.2f} | {r_val:<18.2f} | {sign}{diff:<11.2f}")

    if res_with_fov:
        print("\n[Protocol B: WITH FOV Mask]")
        print(f"{'Metric':<18} | {'Paper (SA-UNetv2)':<18} | {'Reproduced (Ours)':<18} | {'Difference':<12}")
        print("-" * 72)
        paper_with_fov = {
            'F1 (%)': (82.84, res_with_fov['f1']),
            'Jaccard (%)': (70.73, res_with_fov['jacc']),
            'Sensitivity (%)': (83.67, res_with_fov['sen']),
            'Specificity (%)': (97.39, res_with_fov['spe']),
            'Accuracy (%)': (95.61, res_with_fov['acc']),
            'MCC (%)': (80.44, res_with_fov['mcc']),
            'AUC (%)': (98.08, res_with_fov['auc'])
        }
        for m_name, (p_val, r_val) in paper_with_fov.items():
            diff = r_val - p_val
            sign = "+" if diff >= 0 else ""
            print(f"{m_name:<18} | {p_val:<18.2f} | {r_val:<18.2f} | {sign}{diff:<11.2f}")

    print(f"\nAverage Inference Time: {avg_inf_per_img * 1000:.1f} ms / image on {device}")
    print(f"Predictions and visual masks saved to: {output_dir}/predictions/")
    print("=" * 70)


if __name__ == '__main__':
    run_evaluation()
