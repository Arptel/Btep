"""
Training Script for SA-UNetv2 with Conductance-Weighted clDice (cw-clDice)
========================================================================
Integrates the novel cw-clDice loss into the PyTorch training loop.

Compound Objective:
L_total = 0.5 * L_BCE + 0.5 * L_MCC + lambda_cw * L_cw-clDice
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add project roots to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "baseline_reproduction", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from model import SA_UNetv2, count_parameters
from losses import CompoundCwclDiceLoss
from dataset import get_cwcldice_drive_datasets


def train_sa_unetv2_cwcldice(
    data_dir="Drive/DRIVE",
    epochs=50,
    batch_size=2,
    lr=5e-4,
    lambda_cw=0.2,
    alpha=2.0,
    beta=1.5,
    val_ratio=0.1,
    patience_early_stop=15,
    patience_lr_plateau=7,
    init_from_baseline=True,
    loss_mode="cw_cldice",
    save_dir="checkpoints",
    save_name=None,
    device_name="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device_name)

    # Configure loss parameters based on loss_mode
    if loss_mode == "cw_bce_only":
        use_cw_bce = True
        eff_lambda_cw = 0.0
        mode_desc = "Standalone cw-BCE (Sub-step 1A: 0.5*cw-BCE + 0.5*MCC, lambda_cw=0.0)"
        default_save_name = "best_sa_unetv2_cwbce.pth"
    elif loss_mode == "unified":
        use_cw_bce = True
        eff_lambda_cw = lambda_cw
        mode_desc = f"Unified Caliber (Sub-step 1B: 0.5*cw-BCE + 0.5*MCC + {lambda_cw}*cw-clDice)"
        default_save_name = "best_sa_unetv2_unified.pth"
    elif loss_mode == "vanilla_cldice":
        use_cw_bce = False
        eff_lambda_cw = lambda_cw
        alpha = 0.0
        mode_desc = f"Vanilla clDice (0.5*BCE + 0.5*MCC + {lambda_cw}*clDice, alpha=0.0)"
        default_save_name = "best_sa_unetv2_cldice.pth"
    else:  # 'cw_cldice'
        use_cw_bce = False
        eff_lambda_cw = lambda_cw
        mode_desc = f"cw-clDice (0.5*BCE + 0.5*MCC + {lambda_cw}*cw-clDice, alpha={alpha}, beta={beta})"
        default_save_name = "best_sa_unetv2_cwcldice.pth"

    target_save_name = save_name if save_name is not None else default_save_name
    save_path = os.path.join(save_dir, target_save_name)

    print("=" * 65)
    print(" TRAINING SA-UNetv2 WITH LOSS MODE:", loss_mode.upper())
    print(f" Device: {device_name.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Objective: {mode_desc}")
    print(f" Target Checkpoint: {save_path}")
    print(f" Batch Size: {batch_size} | Epochs: {epochs} | Initial LR: {lr:.2e}")
    print("=" * 65, flush=True)

    # 1. Datasets and DataLoaders
    train_ds, val_ds, test_ds = get_cwcldice_drive_datasets(
        data_dir, val_ratio=val_ratio, alpha=alpha, beta=beta
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Dataset split: {len(train_ds)} train samples, {len(val_ds)} validation samples")

    # 2. Instantiate Model
    model = SA_UNetv2(in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.15, block_size=7).to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params / 1e6:.4f}M ({total_params:,} parameters)")

    # 3. Optional Warm-Start from Baseline Weights
    baseline_ckpt = os.path.join(save_dir, "best_sa_unetv2.pth")
    if init_from_baseline and os.path.exists(baseline_ckpt):
        print(f"Warm-starting model weights from baseline checkpoint: {baseline_ckpt}")
        ckpt = torch.load(baseline_ckpt, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("-> Baseline weights loaded successfully!")
    else:
        print("Training model from scratch.")

    # 4. Optimizer, Criterion & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CompoundCwclDiceLoss(
        lambda_bce=0.5, lambda_mcc=0.5, lambda_cw=eff_lambda_cw,
        use_cw_bce=use_cw_bce, num_skel_iter=4, alpha=alpha, beta=beta
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience_lr_plateau, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Training Loop ---
        model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_mcc = 0.0
        running_cw = 0.0
        running_score = 0.0

        for batch in train_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            skels = batch['skeleton'].to(device)
            weights = batch['weight_map'].to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss, bce, mcc, cw_l, cw_s = criterion(preds, lbls, s_gt=skels, weight_map=weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_bce += bce.item() * imgs.size(0)
            running_mcc += mcc.item() * imgs.size(0)
            running_cw += cw_l.item() * imgs.size(0)
            running_score += cw_s.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_ds)
        epoch_bce = running_bce / len(train_ds)
        epoch_mcc = running_mcc / len(train_ds)
        epoch_cw = running_cw / len(train_ds)
        epoch_score = running_score / len(train_ds)

        # --- Validation Loop ---
        model.eval()
        val_running_loss = 0.0
        val_running_score = 0.0
        with torch.no_grad():
            for batch in val_loader:
                v_imgs = batch['image'].to(device)
                v_lbls = batch['label'].to(device)
                v_skels = batch['skeleton'].to(device)
                v_weights = batch['weight_map'].to(device)

                v_preds = model(v_imgs)
                v_loss, _, _, _, v_score = criterion(v_preds, v_lbls, s_gt=v_skels, weight_map=v_weights)
                val_running_loss += v_loss.item() * v_imgs.size(0)
                val_running_score += v_score.item() * v_imgs.size(0)

        val_loss = val_running_loss / len(val_ds)
        val_score = val_running_score / len(val_ds)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {epoch_loss:.4f} (BCE: {epoch_bce:.4f}, MCC: {epoch_mcc:.4f}, cw-clDice: {epoch_cw:.4f}) | Val Loss: {val_loss:.4f} | Val cw-Score: {val_score:.4f} | LR: {current_lr:.2e}", flush=True)

        # Checkpoint Saving & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_score': val_score,
                'total_params': total_params,
                'loss_config': {'lambda_cw': lambda_cw, 'alpha': alpha, 'beta': beta}
            }, save_path)
            print(f"  -> Best model saved to {save_path} (Val Loss: {val_loss:.4f})", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_early_stop:
                print(f"Early stopping triggered after {patience_early_stop} epochs without improvement.")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SA-UNetv2 with cw-clDice loss")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lambda_cw", type=float, default=0.2, help="Weight of cw-clDice loss")
    parser.add_argument("--alpha", type=float, default=2.0, help="Capillary boost factor")
    parser.add_argument("--beta", type=float, default=1.5, help="Decay curvature exponent")
    parser.add_argument("--loss_mode", type=str, default="cw_cldice",
                        choices=["cw_cldice", "cw_bce_only", "unified", "vanilla_cldice"],
                        help="Loss objective paradigm")
    parser.add_argument("--save_name", type=str, default=None, help="Custom filename for best checkpoint")
    parser.add_argument("--scratch", action="store_true", help="Train from scratch rather than warm-start")
    args = parser.parse_args()

    train_sa_unetv2_cwcldice(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_cw=args.lambda_cw,
        alpha=args.alpha,
        beta=args.beta,
        loss_mode=args.loss_mode,
        save_name=args.save_name,
        init_from_baseline=not args.scratch
    )
