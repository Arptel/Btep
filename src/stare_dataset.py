"""
STARE Dataset Loader for SA-UNetv2.
Handles zero-padding from (605, 700) to (704, 704),
restoration cropping, data augmentations, and caching.
"""
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random

STARE_BENCHMARK_20 = [
    "im0001", "im0002", "im0003", "im0004", "im0005",
    "im0044", "im0077", "im0081", "im0082", "im0139",
    "im0162", "im0163", "im0235", "im0236", "im0239",
    "im0240", "im0255", "im0291", "im0319", "im0324"
]

# Standard balanced 16 train / 4 test split (2 normal + 2 pathological)
DEFAULT_TEST_IDS = ["im0001", "im0002", "im0162", "im0163"]

def pad_image_stare(img_arr, target_h=704, target_w=704):
    """
    Zero-pad image or mask to target dimensions.
    img_arr: (H, W, C) or (H, W)
    """
    if img_arr.ndim == 3:
        h, w, c = img_arr.shape
        padded = np.zeros((target_h, target_w, c), dtype=img_arr.dtype)
        padded[:h, :w, :] = img_arr
    else:
        h, w = img_arr.shape
        padded = np.zeros((target_h, target_w), dtype=img_arr.dtype)
        padded[:h, :w] = img_arr
    return padded

def restore_image_stare(padded_arr, orig_h=605, orig_w=700):
    """
    Crop padded tensor or numpy array back to original dimensions.
    """
    if isinstance(padded_arr, torch.Tensor):
        if padded_arr.ndim == 4:
            return padded_arr[:, :, :orig_h, :orig_w]
        elif padded_arr.ndim == 3:
            return padded_arr[:, :orig_h, :orig_w]
        return padded_arr[:orig_h, :orig_w]
    else:
        if padded_arr.ndim == 3:
            return padded_arr[:orig_h, :orig_w, :]
        return padded_arr[:orig_h, :orig_w]

class STAREDataset(Dataset):
    """
    PyTorch Dataset for STARE Retinal Vessel Segmentation.
    """
    def __init__(self, image_paths, label_paths, is_train=True, target_size=(704, 704), repeat=1):
        self.image_paths = sorted(image_paths)
        self.label_paths = sorted(label_paths)
        self.is_train = is_train
        self.target_size = target_size
        self.repeat = repeat if is_train else 1
        self.num_base = len(self.image_paths)

        assert len(self.image_paths) == len(self.label_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"

        self.cached_images = []
        self.cached_labels = []

        th, tw = self.target_size
        for idx in range(self.num_base):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img_np = np.array(img, dtype=np.float32) / 255.0
            self.cached_images.append(pad_image_stare(img_np, th, tw))

            lbl = Image.open(self.label_paths[idx]).convert('L')
            lbl_np = (np.array(lbl, dtype=np.float32) > 127.0).astype(np.float32)
            self.cached_labels.append(pad_image_stare(lbl_np, th, tw))

    def __len__(self):
        return self.num_base * self.repeat

    def __getitem__(self, idx):
        base_idx = idx % self.num_base
        img_padded = self.cached_images[base_idx].copy()
        lbl_padded = self.cached_labels[base_idx].copy()

        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float()
        lbl_tensor = torch.from_numpy(lbl_padded).unsqueeze(0).float()

        if self.is_train:
            # Horizontal Flip
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                lbl_tensor = TF.hflip(lbl_tensor)

            # Vertical Flip
            if random.random() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                lbl_tensor = TF.vflip(lbl_tensor)

            # Random 90 deg rotation
            rot_k = random.randint(0, 3)
            if rot_k > 0:
                img_tensor = torch.rot90(img_tensor, rot_k, [1, 2])
                lbl_tensor = torch.rot90(lbl_tensor, rot_k, [1, 2])

            # Subtle photometric jitter
            if random.random() > 0.5:
                factor = 1.0 + random.uniform(-0.15, 0.15)
                img_tensor = torch.clamp(img_tensor * factor, 0.0, 1.0)

        img_id = os.path.basename(self.image_paths[base_idx]).split('.')[0]
        return {
            'image': img_tensor,
            'label': lbl_tensor,
            'id': img_id
        }

def get_stare_dataloaders(benchmark_dir="Stare data/benchmark_20", 
                          test_ids=None, 
                          batch_size=2, 
                          num_workers=0, 
                          repeat=10):
    if test_ids is None:
        test_ids = DEFAULT_TEST_IDS

    img_dir = os.path.join(benchmark_dir, "images")
    lbl_dir = os.path.join(benchmark_dir, "labels")

    train_imgs, train_lbls = [], []
    val_imgs, val_lbls = [], []
    test_imgs, test_lbls = [], []

    train_candidate_ids = [img_id for img_id in STARE_BENCHMARK_20 if img_id not in test_ids]
    # Use 2 images for validation, 14 for pure training
    val_ids = train_candidate_ids[:2]
    pure_train_ids = train_candidate_ids[2:]

    for img_id in pure_train_ids:
        train_imgs.append(os.path.join(img_dir, f"{img_id}.ppm"))
        train_lbls.append(os.path.join(lbl_dir, f"{img_id}.ppm"))

    for img_id in val_ids:
        val_imgs.append(os.path.join(img_dir, f"{img_id}.ppm"))
        val_lbls.append(os.path.join(lbl_dir, f"{img_id}.ppm"))

    for img_id in test_ids:
        test_imgs.append(os.path.join(img_dir, f"{img_id}.ppm"))
        test_lbls.append(os.path.join(lbl_dir, f"{img_id}.ppm"))

    train_dataset = STAREDataset(train_imgs, train_lbls, is_train=True, repeat=repeat)
    val_dataset = STAREDataset(val_imgs, val_lbls, is_train=False, repeat=1)
    test_dataset = STAREDataset(test_imgs, test_lbls, is_train=False, repeat=1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
