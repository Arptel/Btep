"""
Training script for SA-UNetv2 on the STARE dataset.
Reproduces Table 2 of ISBI 2026 paper:
- Batch size: 2
- Padded input: 704 x 704
- Compound Loss: 0.5 * BCE + 0.5 * Continuous MCC
- Optimizer: Adam (lr = 1e-3)
- Scheduler: ReduceLROnPlateau(factor=0.5, patience=10)
- Epochs: 150
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from model import SA_UNetv2, count_parameters
from losses import CompoundLoss
from stare_dataset import get_stare_dataloaders

def train_stare(
    benchmark_dir="Stare data/benchmark_20",
    epochs=150,
    batch_size=2,
    lr=1e-3,
    repeat=2,
    patience_early_stop=25,
    patience_lr_plateau=10,
    save_path="checkpoints/best_sa_unetv2_stare.pth",
    device_name="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = torch.device(device_name)
    print("=" * 60)
    print(" Training SA-UNetv2 on STARE Dataset (ISBI 2026 Baseline)")
    print(f" Device: {device_name.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Benchmark Directory: {benchmark_dir}")
    print(f" Batch Size: {batch_size} | Epochs: {epochs} | Initial LR: {lr}")
    print("=" * 60)

    # 1. Loaders
    train_loader, val_loader, test_loader = get_stare_dataloaders(
        benchmark_dir=benchmark_dir,
        batch_size=batch_size,
        repeat=repeat # 28 augmented iterations per epoch (~5.5s/epoch)
    )
    print(f"[*] Training samples per epoch: {len(train_loader.dataset)} ({len(train_loader)} batches)")
    print(f"[*] Validation samples: {len(val_loader.dataset)} | Test samples: {len(test_loader.dataset)}")

    # 2. Model
    model = SA_UNetv2(in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.15, block_size=7).to(device)
    params = count_parameters(model)
    print(f"[*] Model Parameters: {params / 1e6:.4f}M ({params:,})")

    # 3. Loss, Optimizer, Scheduler
    criterion = CompoundLoss(lambda_bce=0.5, lambda_mcc=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience_lr_plateau, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_bce = 0.0
        running_mcc = 0.0

        for batch in train_loader:
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss, bce, mcc = criterion(preds, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_bce += bce.item() * imgs.size(0)
            running_mcc += mcc.item() * imgs.size(0)

        total_train_samples = len(train_loader.dataset)
        train_loss = running_loss / total_train_samples
        train_bce = running_bce / total_train_samples
        train_mcc = running_mcc / total_train_samples

        # Validation
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                v_imgs = batch['image'].to(device)
                v_lbls = batch['label'].to(device)
                v_preds = model(v_imgs)
                v_loss, _, _ = criterion(v_preds, v_lbls)
                val_running_loss += v_loss.item() * v_imgs.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        epoch_sec = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, save_path)
            flag = f"--> Saved Best Model ({val_loss:.4f})"
        else:
            epochs_no_improve += 1
            flag = ""

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] ({epoch_sec:.1f}s) - "
                  f"Train Loss: {train_loss:.4f} (BCE: {train_bce:.4f}, MCC: {train_mcc:.4f}) | "
                  f"Val Loss: {val_loss:.4f} | LR: {current_lr:.1e} {flag}")

        if epochs_no_improve >= patience_early_stop:
            print(f"\n[*] Early stopping triggered after {patience_early_stop} epochs without improvement.")
            break

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"[+] STARE Training Completed in {total_time / 60:.2f} minutes!")
    print(f"[+] Best Validation Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    print(f"[+] Optimal Checkpoint Saved to: {save_path}")
    print("=" * 60)

    # Save history
    history_path = os.path.join(os.path.dirname(save_path), "history_stare.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SA-UNetv2 on STARE")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    train_stare(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, repeat=args.repeat)
