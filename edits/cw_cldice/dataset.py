"""
High-Performance Dataset with Precomputed Caliber & Skeleton Caching
===================================================================
Precomputes Medial-Axis Skeletons S_gt and Physiological Weight Maps W(x, y)
during initialization and caches them in RAM.

Augmentations:
Applies identical isometric transforms (hflip, vflip, rot90) across:
- Input Image
- Ground Truth Vessel Mask
- Ground Truth Skeleton S_gt
- Caliber Weight Map W(x, y)

Result: Zero GPU/CPU bottleneck during training epochs.
"""
import os
import glob
import random
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import skimage.morphology as morph
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


def pad_image(img_arr, target_h=592, target_w=592):
    """Zero-pads image or mask to target dimensions."""
    if img_arr.ndim == 3:
        h, w, c = img_arr.shape
        padded = np.zeros((target_h, target_w, c), dtype=img_arr.dtype)
        padded[:h, :w, :] = img_arr
    else:
        h, w = img_arr.shape
        padded = np.zeros((target_h, target_w), dtype=img_arr.dtype)
        padded[:h, :w] = img_arr
    return padded


def compute_caliber_weight_map(mask, alpha=2.0, beta=1.5):
    """
    Computes Euclidean Distance Transform and Poiseuille Conductance Weight Map W(x, y).
    """
    vessel_bin = (mask > 0.5).astype(np.float32)
    if vessel_bin.sum() == 0:
        return np.ones_like(mask, dtype=np.float32), np.zeros_like(mask, dtype=np.float32)

    # 1. Medial-axis skeleton (1-pixel wide)
    skeleton = morph.skeletonize(vessel_bin).astype(np.float32)

    # 2. Euclidean Distance Transform (vessel radius at each pixel)
    edt = ndi.distance_transform_edt(vessel_bin).astype(np.float32)

    # 3. Caliber-dependent conductance weight
    r_max = float(np.max(edt))
    r_min = 1.0

    inv_radius = (r_max - edt) / (r_max - r_min + 1e-4)
    inv_radius = np.clip(inv_radius, 0.0, 1.0) * vessel_bin

    weight_map = 1.0 + alpha * np.power(inv_radius, beta)
    return weight_map.astype(np.float32), skeleton.astype(np.float32)


class CwclDiceRetinalDataset(Dataset):
    """
    Retinal Dataset with cached physiological weight maps and skeletons.
    """
    def __init__(self, image_paths, label_paths, mask_paths=None, is_train=True,
                 target_size=(592, 592), repeat=1, alpha=2.0, beta=1.5):
        self.image_paths = sorted(image_paths)
        self.label_paths = sorted(label_paths)
        self.mask_paths = sorted(mask_paths) if mask_paths is not None else None
        self.is_train = is_train
        self.target_size = target_size
        self.repeat = repeat if is_train else 1
        self.num_base = len(self.image_paths)

        self.cached_images = []
        self.cached_labels = []
        self.cached_masks = []
        self.cached_skeletons = []
        self.cached_weights = []

        th, tw = self.target_size
        for idx in range(self.num_base):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img_np = np.array(img, dtype=np.float32) / 255.0
            self.cached_images.append(pad_image(img_np, th, tw))

            lbl = Image.open(self.label_paths[idx]).convert('L')
            lbl_np = (np.array(lbl, dtype=np.float32) > 127.0).astype(np.float32)
            lbl_padded = pad_image(lbl_np, th, tw)
            self.cached_labels.append(lbl_padded)

            # Precompute skeleton and caliber weight map
            w_map, skel = compute_caliber_weight_map(lbl_padded, alpha=alpha, beta=beta)
            self.cached_weights.append(w_map)
            self.cached_skeletons.append(skel)

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

        img_arr = self.cached_images[base_idx].copy()
        lbl_arr = self.cached_labels[base_idx].copy()
        skel_arr = self.cached_skeletons[base_idx].copy()
        w_arr = self.cached_weights[base_idx].copy()
        mask_arr = self.cached_masks[base_idx].copy() if self.cached_masks[base_idx] is not None else None

        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()
        lbl_tensor = torch.from_numpy(lbl_arr).unsqueeze(0).float()
        skel_tensor = torch.from_numpy(skel_arr).unsqueeze(0).float()
        w_tensor = torch.from_numpy(w_arr).unsqueeze(0).float()

        if self.is_train:
            # Random Horizontal Flip
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                lbl_tensor = TF.hflip(lbl_tensor)
                skel_tensor = TF.hflip(skel_tensor)
                w_tensor = TF.hflip(w_tensor)

            # Random Vertical Flip
            if random.random() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                lbl_tensor = TF.vflip(lbl_tensor)
                skel_tensor = TF.vflip(skel_tensor)
                w_tensor = TF.vflip(w_tensor)

            # Random 90-degree rotations
            k = random.choice([0, 1, 2, 3])
            if k > 0:
                img_tensor = torch.rot90(img_tensor, k, [1, 2])
                lbl_tensor = torch.rot90(lbl_tensor, k, [1, 2])
                skel_tensor = torch.rot90(skel_tensor, k, [1, 2])
                w_tensor = torch.rot90(w_tensor, k, [1, 2])

        sample = {
            'image': img_tensor,
            'label': lbl_tensor,
            'skeleton': skel_tensor,
            'weight_map': w_tensor,
            'filename': os.path.basename(self.image_paths[base_idx])
        }
        if mask_arr is not None:
            sample['mask'] = torch.from_numpy(mask_arr).unsqueeze(0).float()

        return sample


def get_cwcldice_drive_datasets(base_dir="Drive/DRIVE", val_ratio=0.1, random_seed=42, repeat=1, alpha=2.0, beta=1.5):
    """
    Finds DRIVE dataset and initializes training, validation, and test datasets.
    """
    candidates = [
        base_dir,
        os.path.join("Drive", "DRIVE"),
        "Drive",
        os.path.join("..", base_dir),
        os.path.join("..", "Drive", "DRIVE"),
        os.path.join("..", "..", base_dir),
        os.path.join("..", "..", "Drive", "DRIVE"),
    ]
    resolved_dir = None
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "training", "images")):
            resolved_dir = cand
            break

    if resolved_dir is None:
        raise FileNotFoundError(f"Could not find DRIVE dataset in candidates: {candidates}")

    train_img_dir = os.path.join(resolved_dir, "training", "images")
    train_lbl_dir = os.path.join(resolved_dir, "training", "1st_manual")
    train_msk_dir = os.path.join(resolved_dir, "training", "mask")

    test_img_dir = os.path.join(resolved_dir, "test", "images")
    test_lbl_dir = os.path.join(resolved_dir, "test", "1st_manual")
    test_msk_dir = os.path.join(resolved_dir, "test", "mask")

    train_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.*")))
    train_lbls = sorted(glob.glob(os.path.join(train_lbl_dir, "*.*")))
    train_msks = sorted(glob.glob(os.path.join(train_msk_dir, "*.*")))

    test_imgs = sorted(glob.glob(os.path.join(test_img_dir, "*.*")))
    test_lbls = sorted(glob.glob(os.path.join(test_lbl_dir, "*.*")))
    test_msks = sorted(glob.glob(os.path.join(test_msk_dir, "*.*")))

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

    train_ds = CwclDiceRetinalDataset(tr_imgs, tr_lbls, tr_msks, is_train=True, repeat=repeat, alpha=alpha, beta=beta)
    val_ds = CwclDiceRetinalDataset(val_imgs, val_lbls, val_msks, is_train=False, repeat=1, alpha=alpha, beta=beta)
    test_ds = CwclDiceRetinalDataset(test_imgs, test_lbls, test_msks, is_train=False, repeat=1, alpha=alpha, beta=beta)

    return train_ds, val_ds, test_ds
