import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, 'src')
from model import SA_UNetv2
from dataset import pad_image, restore_image


def segment_fundus_image(
    image_path,
    model_path="checkpoints/best_sa_unetv2.pth",
    output_dir="results/inference_outputs",
    threshold=0.5,
    device_name="cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Takes an input Color Fundus Photograph (CFG) and outputs:
    1. Binary vessel segmentation map (0 or 255)
    2. Continuous probability heatmap
    3. Color visual overlay (Green vessels overlaid on the original fundus image)
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_name)

    # 1. Load trained model
    model = SA_UNetv2().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Load input image
    orig_img = Image.open(image_path).convert('RGB')
    orig_np = np.array(orig_img, dtype=np.float32)
    h, w, c = orig_np.shape

    # Normalize to [0, 1]
    norm_img = orig_np / 255.0

    # 3. Zero-pad to nearest multiple of 8 (default 592x592 or dynamic)
    target_h = ((h + 7) // 8) * 8
    target_w = ((w + 7) // 8) * 8
    target_h = max(target_h, 592)
    target_w = max(target_w, 592)

    padded_np = pad_image(norm_img, target_h, target_w)
    input_tensor = torch.from_numpy(padded_np).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 4. Inference
    with torch.no_grad():
        prob_padded = model(input_tensor).squeeze().cpu().numpy()

    # 5. Restore original dimensions
    prob_map = prob_padded[:h, :w]
    binary_mask = (prob_map >= threshold).astype(np.uint8) * 255

    # 6. Create Visual Overlay (Green vessels overlaid on RGB fundus)
    overlay = orig_np.copy().astype(np.uint8)
    vessel_mask = binary_mask > 0
    # Highlight vessels in vibrant bright green (0, 255, 0)
    overlay[vessel_mask] = [0, 255, 60]

    # Side-by-Side Visual
    side_by_side = np.hstack([orig_np.astype(np.uint8), np.repeat(binary_mask[:, :, None], 3, axis=2), overlay])

    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    bin_path = os.path.join(output_dir, f"{base_name}_vessel_mask.png")
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
    comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")

    Image.fromarray(binary_mask).save(bin_path)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(side_by_side).save(comparison_path)

    print(f"\n=======================================================")
    print(f" SA-UNetv2 Retinal Vessel Segmentation Result")
    print(f"=======================================================")
    print(f" Input Image:       {image_path} ({w}x{h})")
    print(f" Binary Vessel Mask: {bin_path}")
    print(f" Green Overlay:      {overlay_path}")
    print(f" Side-by-Side View:  {comparison_path}")
    print(f"=======================================================\n")
    return bin_path, overlay_path, comparison_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segment retinal vessels from a color fundus image")
    parser.add_argument('--image', type=str, required=True, help="Path to input color fundus image (.tif, .png, .jpg, .ppm)")
    parser.add_argument('--model', type=str, default="checkpoints/best_sa_unetv2.pth", help="Path to model weights")
    parser.add_argument('--out_dir', type=str, default="results/inference_outputs", help="Output directory")
    parser.add_argument('--threshold', type=float, default=0.5, help="Binarization threshold (default: 0.5)")
    args = parser.parse_args()

    segment_fundus_image(
        image_path=args.image,
        model_path=args.model,
        output_dir=args.out_dir,
        threshold=args.threshold
    )
