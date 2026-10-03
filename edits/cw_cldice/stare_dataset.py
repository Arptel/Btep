"""
STARE Dataset Loader for clDice and cw-clDice Training
======================================================
Precomputes Medial-Axis Skeletons S_gt and Conductance Weight Maps W(x, y)
during initialization and caches them in RAM.

Supports:
- Vanilla clDice: alpha=0.0 -> Uniform weight map W(x, y) = 1.0
- cw-clDice:      alpha=2.0, beta=1.5 -> Caliber-dependent conductance weight
"""
import os
import random
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import skimage.morphology as morph
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

STARE_BENCHMARK_20 = [
    "im0001", "im0002", "im0003", "im0004", "im0005",
    "im0044", "im0077", "im0081", "im0082", "im0139",
    "im0162", "im0163", "im0235", "im0236", "im0239",
    "im0240", "im0255", "im0291", "im0319", "im0324"
]

DEFAULT_TEST_IDS = ["im0001", "im0002", "im0162", "im0163"]


def pad_image_stare(img_arr, target_h=704, target_w=704):
    """Zero-pad image or mask to target dimensions (704, 704)."""
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
    """Crop padded tensor or numpy array back to original dimensions (605, 700)."""
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


import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(project_root, "edits", "idea3_murray"))
try:
    from murray_loss import generate_murray_bifurcation_map
except ImportError:
    generate_murray_bifurcation_map = None


def compute_caliber_weight_map_stare(mask, alpha=2.0, beta=1.5):
    """
    Computes Euclidean Distance Transform, Conductance Weight Map W(x, y),
    and Murray Bifurcation Map M_bif(x, y).
    """
    vessel_bin = (mask > 0.5).astype(np.float32)
    if vessel_bin.sum() == 0:
        return np.ones_like(mask, dtype=np.float32), np.zeros_like(mask, dtype=np.float32), np.zeros_like(mask, dtype=np.float32)

    # 1. Medial-axis skeleton (1-pixel wide)
    skeleton = morph.skeletonize(vessel_bin).astype(np.float32)
    edt = ndi.distance_transform_edt(vessel_bin).astype(np.float32)

    # 2. Murray Bifurcation Map
    if generate_murray_bifurcation_map is not None:
        try:
            bif_map = generate_murray_bifurcation_map(vessel_bin, skeleton, edt)
        except Exception:
            bif_map = np.zeros_like(mask, dtype=np.float32)
    else:
        bif_map = np.zeros_like(mask, dtype=np.float32)

    # 3. Vanilla clDice (alpha == 0): return uniform 1.0 weights
    if alpha == 0.0:
        weight_map = np.ones_like(mask, dtype=np.float32)
        return weight_map, skeleton, bif_map

    # 4. cw-clDice: Euclidean Distance Transform
    r_max = float(np.max(edt))
    r_min = 1.0

    inv_radius = (r_max - edt) / (r_max - r_min + 1e-4)
    inv_radius = np.clip(inv_radius, 0.0, 1.0) * vessel_bin

    weight_map = 1.0 + alpha * np.power(inv_radius, beta)
    return weight_map.astype(np.float32), skeleton.astype(np.float32), bif_map.astype(np.float32)


class STARECwclDiceDataset(Dataset):
    """
    PyTorch Dataset for STARE with cached skeletons, physiological weight maps, and Murray bifurcation maps.
    """
    def __init__(self, image_paths, label_paths, is_train=True,
                 target_size=(704, 704), repeat=1, alpha=2.0, beta=1.5):
        self.image_paths = sorted(image_paths)
        self.label_paths = sorted(label_paths)
        self.is_train = is_train
        self.target_size = target_size
        self.repeat = repeat if is_train else 1
        self.alpha = alpha
        self.beta = beta
        self.num_base = len(self.image_paths)

        assert len(self.image_paths) == len(self.label_paths), \
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"

        self.cached_images = []
        self.cached_labels = []
        self.cached_skeletons = []
        self.cached_weights = []
        self.cached_bif_maps = []

        th, tw = self.target_size
        for idx in range(self.num_base):
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img_np = np.array(img, dtype=np.float32) / 255.0
            self.cached_images.append(pad_image_stare(img_np, th, tw))

            lbl = Image.open(self.label_paths[idx]).convert('L')
            lbl_np = (np.array(lbl, dtype=np.float32) > 127.0).astype(np.float32)
            lbl_padded = pad_image_stare(lbl_np, th, tw)
            self.cached_labels.append(lbl_padded)

            w_map, skel, bif_map = compute_caliber_weight_map_stare(lbl_padded, alpha=alpha, beta=beta)
            self.cached_weights.append(w_map)
            self.cached_skeletons.append(skel)
            self.cached_bif_maps.append(bif_map)

    def __len__(self):
        return self.num_base * self.repeat

    def __getitem__(self, idx):
        base_idx = idx % self.num_base
        img_padded = self.cached_images[base_idx].copy()
        lbl_padded = self.cached_labels[base_idx].copy()
        skel_padded = self.cached_skeletons[base_idx].copy()
        w_padded = self.cached_weights[base_idx].copy()
        bif_padded = self.cached_bif_maps[base_idx].copy()

        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float()
        lbl_tensor = torch.from_numpy(lbl_padded).unsqueeze(0).float()
        skel_tensor = torch.from_numpy(skel_padded).unsqueeze(0).float()
        w_tensor = torch.from_numpy(w_padded).unsqueeze(0).float()
        bif_tensor = torch.from_numpy(bif_padded).unsqueeze(0).float()

        if self.is_train:
            # Isometric spatial augmentations
            if random.random() > 0.5:
                img_tensor = TF.hflip(img_tensor)
                lbl_tensor = TF.hflip(lbl_tensor)
                skel_tensor = TF.hflip(skel_tensor)
                w_tensor = TF.hflip(w_tensor)
                bif_tensor = TF.hflip(bif_tensor)

            if random.random() > 0.5:
                img_tensor = TF.vflip(img_tensor)
                lbl_tensor = TF.vflip(lbl_tensor)
                skel_tensor = TF.vflip(skel_tensor)
                w_tensor = TF.vflip(w_tensor)
                bif_tensor = TF.vflip(bif_tensor)

            rot_k = random.randint(0, 3)
            if rot_k > 0:
                img_tensor = torch.rot90(img_tensor, rot_k, [1, 2])
                lbl_tensor = torch.rot90(lbl_tensor, rot_k, [1, 2])
                skel_tensor = torch.rot90(skel_tensor, rot_k, [1, 2])
                w_tensor = torch.rot90(w_tensor, rot_k, [1, 2])
                bif_tensor = torch.rot90(bif_tensor, rot_k, [1, 2])

            if random.random() > 0.5:
                factor = 1.0 + random.uniform(-0.15, 0.15)
                img_tensor = torch.clamp(img_tensor * factor, 0.0, 1.0)

        img_id = os.path.basename(self.image_paths[base_idx]).split('.')[0]
        return {
            'image': img_tensor,
            'label': lbl_tensor,
            'skeleton': skel_tensor,
            'weight_map': w_tensor,
            'bif_map': bif_tensor,
            'id': img_id
        }


def get_stare_cwcldice_dataloaders(benchmark_dir="Stare data/benchmark_20",
                                   test_ids=None,
                                   batch_size=2,
                                   num_workers=0,
                                   repeat=2,
                                   alpha=2.0,
                                   beta=1.5):
    """
    Constructs train, val, and test DataLoaders for STARE.
    alpha=0.0 -> vanilla clDice
    alpha=2.0 -> cw-clDice
    """
    if test_ids is None:
        test_ids = DEFAULT_TEST_IDS

    if not os.path.exists(benchmark_dir):
        alt_dir = os.path.join("..", benchmark_dir)
        if os.path.exists(alt_dir):
            benchmark_dir = alt_dir

    img_dir = os.path.join(benchmark_dir, "images")
    lbl_dir = os.path.join(benchmark_dir, "labels")

    train_imgs, train_lbls = [], []
    val_imgs, val_lbls = [], []
    test_imgs, test_lbls = [], []

    train_candidate_ids = [img_id for img_id in STARE_BENCHMARK_20 if img_id not in test_ids]
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

    train_ds = STARECwclDiceDataset(train_imgs, train_lbls, is_train=True, repeat=repeat, alpha=alpha, beta=beta)
    val_ds = STARECwclDiceDataset(val_imgs, val_lbls, is_train=False, repeat=1, alpha=alpha, beta=beta)
    test_ds = STARECwclDiceDataset(test_imgs, test_lbls, is_train=False, repeat=1, alpha=alpha, beta=beta)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
