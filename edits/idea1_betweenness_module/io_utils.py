"""
I/O utilities for dumping and loading probability maps, ground truth, and FOV masks.
"""
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

def find_path(rel_path):
    if os.path.exists(rel_path):
        return rel_path
    p1 = os.path.join("..", rel_path)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join("..", "..", rel_path)
    if os.path.exists(p2):
        return p2
    return rel_path

def dump_drive_prob_maps(checkpoint_path="checkpoints/best_sa_unetv2.pth", 
                         data_dir="Drive/DRIVE", 
                         out_dir="edits/idea1_betweenness_module/prob_maps/drive"):
    checkpoint_path = find_path(checkpoint_path)
    data_dir = find_path(data_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Import SA_UNetv2 and dataloader
    import sys
    src_dir = find_path("baseline_reproduction/src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from model import SA_UNetv2
    from dataset import RetinalDataset, pad_image, restore_image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SA_UNetv2().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    splits = {
        'test': (os.path.join(data_dir, "test", "images"), os.path.join(data_dir, "test", "1st_manual"), os.path.join(data_dir, "test", "mask")),
        'train': (os.path.join(data_dir, "training", "images"), os.path.join(data_dir, "training", "1st_manual"), os.path.join(data_dir, "training", "mask"))
    }

    print(f"[*] Dumping DRIVE probability maps using checkpoint: {checkpoint_path}")
    for split_name, (img_d, lbl_d, msk_d) in splits.items():
        img_paths = sorted(glob.glob(os.path.join(img_d, "*.*")))
        lbl_paths = sorted(glob.glob(os.path.join(lbl_d, "*.*")))
        msk_paths = sorted(glob.glob(os.path.join(msk_d, "*.*")))

        split_out = os.path.join(out_dir, split_name)
        os.makedirs(split_out, exist_ok=True)

        for ipath in img_paths:
            basename = os.path.splitext(os.path.basename(ipath))[0]
            img = Image.open(ipath).convert('RGB')
            orig_w, orig_h = img.size
            img_arr = np.array(img, dtype=np.float32) / 255.0
            padded_img = pad_image(img_arr, target_h=592, target_w=592)
            tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(tensor_img)
                pred_prob = pred.squeeze().cpu().numpy()

            # Crop back to original dimensions
            prob_map = restore_image(pred_prob, orig_h=orig_h, orig_w=orig_w)
            save_path = os.path.join(split_out, f"{basename}_prob.npy")
            np.save(save_path, prob_map.astype(np.float32))

    print(f"[+] DRIVE probability maps dumped to: {out_dir}")

if __name__ == "__main__":
    dump_drive_prob_maps()
