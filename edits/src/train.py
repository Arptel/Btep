import os
import time
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SA_UNetv2, count_parameters
from losses import CompoundLoss
from dataset import get_drive_datasets
from evaluate import evaluate_model


def train_sa_unetv2(
    data_dir="Drive/DRIVE",
    epochs=150,
    batch_size=8,
    lr=1e-3,
    val_ratio=0.1,
    patience_early_stop=20,
    patience_lr_plateau=10,
    save_dir="checkpoints",
    device_name="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device_name)
    print(f"==================================================")
    print(f" Training SA-UNetv2 (ISBI 2026 Base Model)")
    print(f" Device: {device_name.upper()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Dataset Directory: {data_dir}")
    print(f" Batch Size: {batch_size} | Epochs: {epochs} | Initial LR: {lr}")
    print(f"==================================================")

    # 1. Datasets and DataLoaders
    train_ds, val_ds, test_ds = get_drive_datasets(data_dir, val_ratio=val_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Dataset split: {len(train_ds)} train samples, {len(val_ds)} validation samples")

    # 2. Instantiate Model
    model = SA_UNetv2(in_channels=3, out_channels=1, start_neurons=16, drop_prob=0.15, block_size=7).to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params / 1e6:.4f}M ({total_params:,} parameters)")

    # 3. Optimizer, Loss & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = CompoundLoss(lambda_bce=0.5, lambda_mcc=0.5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience_lr_plateau, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Training Loop ---
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

        epoch_loss = running_loss / len(train_ds)
        epoch_bce = running_bce / len(train_ds)
        epoch_mcc = running_mcc / len(train_ds)

        # --- Validation Loop ---
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                v_imgs = batch['image'].to(device)
                v_lbls = batch['label'].to(device)
                v_preds = model(v_imgs)
                v_loss, _, _ = criterion(v_preds, v_lbls)
                val_running_loss += v_loss.item() * v_imgs.size(0)

        val_loss = val_running_loss / len(val_ds)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)

        if epoch % 2 == 0 or epoch == 1:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Loss: {epoch_loss:.4f} (BCE: {epoch_bce:.4f}, MCC: {epoch_mcc:.4f}) | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}", flush=True)

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
                'total_params': total_params
            }, os.path.join(save_dir, "best_sa_unetv2.pth"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_early_stop:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch}.")
                break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60.0:.2f} minutes.")

    # Save history
    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=4)

    return os.path.join(save_dir, "best_sa_unetv2.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Drive/DRIVE')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    train_sa_unetv2(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
