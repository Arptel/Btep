import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


def pad_image(img_arr, target_h=592, target_w=592):
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


def restore_image(padded_arr, orig_h=584, orig_w=565):
    """
    Crop padded image/mask back to original dimensions.
    """
    if padded_arr.ndim == 3:
        return padded_arr[:orig_h, :orig_w, :]
    elif padded_arr.ndim == 4:
        return padded_arr[:, :, :orig_h, :orig_w]
    return padded_arr[:orig_h, :orig_w]


class RetinalDataset(Dataset):
    """
    PyTorch Dataset for DRIVE / STARE Retinal Vessel Segmentation.
    """
    def __init__(self, image_paths, label_paths, mask_paths=None, is_train=True, target_size=(592, 592), repeat=1):
        self.image_paths = sorted(image_paths)
        self.label_paths = sorted(label_paths)
        self.mask_paths = sorted(mask_paths) if mask_paths is not None else None
        self.is_train = is_train
        self.target_size = target_size
        self.repeat = repeat if is_train else 1
        self.num_base = len(self.image_paths)

        assert len(self.image_paths) == len(self.label_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"

        # Pre-cache images, labels, and masks in memory (RAM)
        self.cached_images = []
        self.cached_labels = []
        self.cached_masks = []

        th, tw = self.target_size
        for idx in range(self.num_base):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img_np = np.array(img, dtype=np.float32) / 255.0
            self.cached_images.append(pad_image(img_np, th, tw))

            lbl = Image.open(self.label_paths[idx]).convert('L')
            lbl_np = (np.array(lbl, dtype=np.float32) > 127.0).astype(np.float32)
            self.cached_labels.append(pad_image(lbl_np, th, tw))

            if self.mask_paths is not None and idx < len(self.mask_paths):
                msk = Image.open(self.mask_paths[idx]).convert('L')
                msk_np = (np.array(msk, dtype=np.float32) > 127.0).astype(np.float32)
                self.cached_masks.append(pad_image(msk_np, th, tw))
            else:
                self.cached_masks.append(None)

    def __len__(self):
        return self.num_base * self.repeat

    def __getitem__(self, idx):
        base_idx = idx % self.num_base
        
        # 1. Retrieve cached padded image & label
        img_padded = self.cached_images[base_idx].copy()
        lbl_padded = self.cached_labels[base_idx].copy()
        mask_padded = self.cached_masks[base_idx].copy() if self.cached_masks[base_idx] is not None else None

        # 2. Convert to PyTorch Tensors (C, H, W)
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float()
        lbl_tensor = torch.from_numpy(lbl_padded).unsqueeze(0).float()

        # 6. Data Augmentation during training
        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                lbl_tensor = TF.hflip(lbl_tensor)

            # Random Vertical Flip
            if random.random() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                lbl_tensor = TF.vflip(lbl_tensor)

            # Random 90-degree rotations
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img_tensor = torch.rot90(img_tensor, k, [1, 2])
                lbl_tensor = torch.rot90(lbl_tensor, k, [1, 2])

        sample = {
            'image': img_tensor,
            'label': lbl_tensor,
            'filename': os.path.basename(self.image_paths[base_idx])
        }
        if mask_padded is not None:
            sample['mask'] = torch.from_numpy(mask_padded).unsqueeze(0).float()
            sample['raw_mask'] = torch.from_numpy(self.cached_masks[base_idx][:584, :565]).float()

        return sample


def get_drive_datasets(base_dir, val_ratio=0.1, random_seed=42, repeat=1):
    """
    Discovers DRIVE dataset paths and creates train, val, test datasets.
    """
    train_img_dir = os.path.join(base_dir, "training", "images")
    train_lbl_dir = os.path.join(base_dir, "training", "1st_manual")
    train_msk_dir = os.path.join(base_dir, "training", "mask")

    test_img_dir = os.path.join(base_dir, "test", "images")
    test_lbl_dir = os.path.join(base_dir, "test", "1st_manual")
    test_msk_dir = os.path.join(base_dir, "test", "mask")

    train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.*")))
    train_lbls = sorted(glob.glob(os.path.join(train_lbl_dir, "*.*")))
    train_msks = sorted(glob.glob(os.path.join(train_msk_dir, "*.*")))

    test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.*")))
    test_lbls = sorted(glob.glob(os.path.join(test_lbl_dir, "*.*")))
    test_msks = sorted(glob.glob(os.path.join(test_msk_dir, "*.*")))

    # Random 10% validation split from training data as per paper protocol
    indices = list(range(len(train_imgs)))
    random.seed(random_seed)
    random.shuffle(indices)

    val_size = max(1, int(len(train_imgs) * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    val_imgs = [train_imgs[i] for i in val_indices]
    val_lbls = [train_lbls[i] for i in val_indices]
    val_msks = [train_msks[i] for i in val_indices] if train_msks else None

    tr_imgs = [train_imgs[i] for i in train_indices]
    tr_lbls = [train_lbls[i] for i in train_indices]
    tr_msks = [train_msks[i] for i in train_indices] if train_msks else None

    train_ds = RetinalDataset(tr_imgs, tr_lbls, tr_msks, is_train=True, repeat=repeat)
    val_ds = RetinalDataset(val_imgs, val_lbls, val_msks, is_train=False, repeat=1)
    test_ds = RetinalDataset(test_imgs, test_lbls, test_msks, is_train=False, repeat=1) if test_lbls else None

    return train_ds, val_ds, test_ds
