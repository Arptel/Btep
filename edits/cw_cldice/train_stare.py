"""
Training Script for SA-UNetv2 on STARE with clDice / cw-clDice
==============================================================
Fine-tunes the baseline SA-UNetv2 weights with topological loss:
- Mode 1 (vanilla clDice): alpha=0.0 -> Uniform centerline weighting
- Mode 2 (cw-clDice):      alpha=2.0, beta=1.5 -> Conductance-weighted capillary boost

Compound Objective:
L_total = 0.5 * L_BCE + 0.5 * L_MCC + lambda_cw * L_clDice
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "baseline_reproduction", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from model import SA_UNetv2, count_parameters
from losses import CompoundCwclDiceLoss
from stare_dataset import get_stare_cwcldice_dataloaders


def train_stare_cwcldice(
    benchmark_dir="Stare data/benchmark_20",
    epochs=35,
    batch_size=2,
    lr=3e-4,
    lambda_cw=0.2,
    alpha=2.0,
    beta=1.5,
    repeat=2,
    patience_early_stop=15,
    patience_lr_plateau=7,
    loss_mode="cw_cldice",
    use_cad_topo_csa=False,
    init_checkpoint="checkpoints/best_sa_unetv2_stare.pth",
    save_path=None,
    device_name="cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_name)

    if loss_mode == "cw_bce_only":
        use_cw_bce = True
        eff_lambda_cw = 0.0
        mode_desc = "Standalone cw-BCE (Sub-step 1A: 0.5*cw-BCE + 0.5*MCC, lambda_cw=0.0)"
        default_save_path = "checkpoints/best_sa_unetv2_stare_cwbce.pth"
    elif loss_mode == "unified":
        use_cw_bce = True
        eff_lambda_cw = lambda_cw
        eff_lambda_murray = 0.0
        mode_desc = f"Unified Caliber (Sub-step 1B: 0.5*cw-BCE + 0.5*MCC + {lambda_cw}*cw-clDice)"
        default_save_path = "checkpoints/best_sa_unetv2_stare_cadtocsa_unified.pth" if use_cad_topo_csa else "checkpoints/best_sa_unetv2_stare_unified.pth"
    elif loss_mode == "murray_unified":
        use_cw_bce = True
        eff_lambda_cw = lambda_cw
        eff_lambda_murray = 0.15
        mode_desc = f"Murray-Unified (Idea 1 + Idea 3: 0.5*cw-BCE + 0.5*MCC + {lambda_cw}*cw-clDice + {eff_lambda_murray}*Murray)"
        default_save_path = "checkpoints/best_sa_unetv2_stare_cadtocsa_murray.pth" if use_cad_topo_csa else "checkpoints/best_sa_unetv2_stare_murray.pth"
    elif loss_mode == "vanilla_cldice":
        use_cw_bce = False
        eff_lambda_cw = lambda_cw
        eff_lambda_murray = 0.0
        alpha = 0.0
        mode_desc = f"Vanilla clDice (0.5*BCE + 0.5*MCC + {lambda_cw}*clDice, alpha=0.0)"
        default_save_path = "checkpoints/best_sa_unetv2_stare_cldice.pth"
    else:  # cw_cldice
        use_cw_bce = False
        eff_lambda_cw = lambda_cw
        eff_lambda_murray = 0.0
        mode_desc = f"cw-clDice (0.5*BCE + 0.5*MCC + {lambda_cw}*cw-clDice, alpha={alpha}, beta={beta})"
        default_save_path = "checkpoints/best_sa_unetv2_stare_cwcldice.pth"

    target_save_path = save_path if save_path is not None else default_save_path
    os.makedirs(os.path.dirname(target_save_path), exist_ok=True)

    print("=" * 68)
    print(f" TRAINING SA-UNetv2 ON STARE WITH LOSS MODE: {loss_mode.upper()}")
    print(f" Device: {device_name.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Objective: {mode_desc}")
    print(f" Batch Size: {batch_size} | Epochs: {epochs} | Initial LR: {lr:.2e}")
    print(f" Target Checkpoint: {target_save_path}")
    print("=" * 68, flush=True)

    # 1. Datasets & Loaders
    train_loader, val_loader, test_loader = get_stare_cwcldice_dataloaders(
        benchmark_dir=benchmark_dir,
        batch_size=batch_size,
        repeat=repeat,
        alpha=alpha,
        beta=beta
    )
    print(f"[*] Train batches per epoch: {len(train_loader)} ({len(train_loader.dataset)} samples)")
    print(f"[*] Validation samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # 2. Model
    model = SA_UNetv2(in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.15, block_size=7, use_cad_topo_csa=use_cad_topo_csa).to(device)
    total_params = count_parameters(model)
    arch_name = "SA-UNetv2 + CAD-Topo-CSA" if use_cad_topo_csa else "Baseline SA-UNetv2"
    print(f"[*] Architecture: {arch_name} | Parameters: {total_params / 1e6:.4f}M ({total_params:,})")

    # 3. Warm-Start from STARE Baseline Weights
    if os.path.exists(init_checkpoint):
        print(f"[*] Warm-starting weights from baseline: {init_checkpoint}")
        ckpt = torch.load(init_checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        res = model.load_state_dict(state_dict, strict=not use_cad_topo_csa)
        if use_cad_topo_csa:
            print(f"-> Warm-started backbone! (Missing keys initialized: {len(res.missing_keys)})")
        else:
            print("-> Baseline weights loaded successfully!")
    else:
        print(f"[!] Warning: Initial checkpoint {init_checkpoint} not found. Training from scratch.")

    # 4. Criterion, Optimizer, Scheduler
    criterion = CompoundCwclDiceLoss(
        lambda_bce=0.5, lambda_mcc=0.5, lambda_cw=eff_lambda_cw, lambda_murray=eff_lambda_murray,
        use_cw_bce=use_cw_bce, num_skel_iter=4, alpha=alpha, beta=beta
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience_lr_plateau, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    save_path = target_save_path
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_mcc = 0.0
        running_cw = 0.0
        running_score = 0.0
        total_samples = 0

        for batch in train_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)
            skels = batch['skeleton'].to(device)
            weights = batch['weight_map'].to(device)
            bifs = batch['bif_map'].to(device) if 'bif_map' in batch else None

            optimizer.zero_grad()
            preds = model(imgs)
            loss, bce, mcc, cw_l, cw_s = criterion(preds, lbls, s_gt=skels, weight_map=weights, bif_map=bifs)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            running_bce += bce.item() * bs
            running_mcc += mcc.item() * bs
            running_cw += cw_l.item() * bs
            running_score += cw_s.item() * bs
            total_samples += bs

        epoch_loss = running_loss / total_samples
        epoch_bce = running_bce / total_samples
        epoch_mcc = running_mcc / total_samples
        epoch_cw = running_cw / total_samples
        epoch_score = running_score / total_samples

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_score = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device)
                lbls = batch['label'].to(device)
                skels = batch['skeleton'].to(device)
                weights = batch['weight_map'].to(device)
                bifs = batch['bif_map'].to(device) if 'bif_map' in batch else None

                preds = model(imgs)
                v_loss, _, _, _, v_s = criterion(preds, lbls, s_gt=skels, weight_map=weights, bif_map=bifs)
                val_loss += v_loss.item()
                val_score += v_s.item()

        val_loss /= len(val_loader)
        val_score /= len(val_loader)
        curr_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch:03d}/{epochs:03d}] | "
            f"Train: {epoch_loss:.4f} (BCE: {epoch_bce:.4f}, MCC: {epoch_mcc:.4f}, clDice: {epoch_cw:.4f}) | "
            f"Val Loss: {val_loss:.4f} | Val Score: {val_score:.4f} | LR: {curr_lr:.2e}",
            flush=True
        )

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
                'loss_config': {'alpha': alpha, 'beta': beta, 'lambda_cw': lambda_cw}
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
    parser = argparse.ArgumentParser(description="Train SA-UNetv2 on STARE with clDice / cw-clDice")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--lambda_cw", type=float, default=0.2, help="Weight of clDice loss")
    parser.add_argument("--alpha", type=float, default=2.0, help="Capillary boost factor (0.0 for vanilla clDice)")
    parser.add_argument("--beta", type=float, default=1.5, help="Decay curvature exponent")
    parser.add_argument("--repeat", type=int, default=2, help="Dataset repeat multiplier per epoch")
    parser.add_argument("--loss_mode", type=str, default="cw_cldice",
                        choices=["cw_cldice", "cw_bce_only", "unified", "vanilla_cldice", "murray_unified"],
                        help="Loss objective paradigm")
    parser.add_argument("--use_cad_topo_csa", action="store_true",
                        help="Enable CAD-Topo-CSA multi-scale skip attention")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                        help="Initial checkpoint to warm-start weights from")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    init_ckpt = args.init_checkpoint
    if init_ckpt is None:
        if args.use_cad_topo_csa and os.path.exists("checkpoints/best_sa_unetv2_stare_cadtocsa.pth"):
            init_ckpt = "checkpoints/best_sa_unetv2_stare_cadtocsa.pth"
        else:
            init_ckpt = "checkpoints/best_sa_unetv2_stare.pth"

    train_stare_cwcldice(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_cw=args.lambda_cw,
        alpha=args.alpha,
        beta=args.beta,
        repeat=args.repeat,
        loss_mode=args.loss_mode,
        use_cad_topo_csa=args.use_cad_topo_csa,
        init_checkpoint=init_ckpt,
        save_path=args.save_path
    )
