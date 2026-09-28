"""
Download and extract the 20 standard STARE ground-truth vessel annotations
created by Adam Hoover from Clemson University.
Organizes images and labels into:
  Stare data/benchmark_20/images/
  Stare data/benchmark_20/labels/
"""
import os
import shutil
import tarfile
import gzip
import io
import ssl
import urllib.request
from PIL import Image

CLEMSON_URL = "https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar"

BENCHMARK_20 = [
    "im0001", "im0002", "im0003", "im0004", "im0005",
    "im0044", "im0077", "im0081", "im0082", "im0139",
    "im0162", "im0163", "im0235", "im0236", "im0239",
    "im0240", "im0255", "im0291", "im0319", "im0324"
]

def setup_stare_benchmark(stare_root="Stare data", out_dir="Stare data/benchmark_20"):
    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"[*] Downloading Hoover vessel labels from {CLEMSON_URL}...")
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(CLEMSON_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
        tar_bytes = response.read()

    print(f"[+] Downloaded {len(tar_bytes)} bytes. Extracting archives...")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for member in tf.getmembers():
            base_name = member.name.split(".")[0] # e.g. im0001
            if base_name in BENCHMARK_20:
                f_obj = tf.extractfile(member)
                if f_obj is not None:
                    # File inside tar is .ppm.gz
                    with gzip.GzipFile(fileobj=f_obj) as gz:
                        decompressed = gz.read()
                        out_label_path = os.path.join(labels_dir, f"{base_name}.ppm")
                        with open(out_label_path, "wb") as out_f:
                            out_f.write(decompressed)
                        print(f"  [Label] Saved {base_name}.ppm")

    print("[*] Copying corresponding 20 benchmark images from raw data...")
    for base_name in BENCHMARK_20:
        src_img = os.path.join(stare_root, f"{base_name}.ppm")
        dst_img = os.path.join(images_dir, f"{base_name}.ppm")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            print(f"  [Image] Copied {base_name}.ppm")
        else:
            print(f"  [!] Missing image: {src_img}")

    # Verify all pairs
    print("\n[*] Verifying 20 image-label pairs:")
    verified = 0
    for base_name in BENCHMARK_20:
        img_p = os.path.join(images_dir, f"{base_name}.ppm")
        lbl_p = os.path.join(labels_dir, f"{base_name}.ppm")
        if os.path.exists(img_p) and os.path.exists(lbl_p):
            with Image.open(img_p) as img, Image.open(lbl_p) as lbl:
                if img.size == lbl.size:
                    verified += 1
                else:
                    print(f"  [Mismatch] {base_name}: img {img.size} vs lbl {lbl.size}")
    print(f"[+] Successfully prepared and verified {verified}/{len(BENCHMARK_20)} STARE benchmark pairs!")

if __name__ == "__main__":
    setup_stare_benchmark()
