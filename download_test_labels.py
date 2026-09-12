import os
import urllib.request

def download_drive_test_labels():
    target_dir = os.path.join("Drive", "DRIVE", "test", "1st_manual")
    os.makedirs(target_dir, exist_ok=True)
    base_url = "https://huggingface.co/datasets/Zomba/DRIVE-digital-retinal-images-for-vessel-extraction/resolve/main/val/label"

    print(f"Downloading 20 DRIVE test ground-truth masks to {target_dir}...")
    headers = {'User-Agent': 'Mozilla/5.0'}

    for i in range(1, 21):
        filename = f"{i:02d}_manual1.png"
        filepath = os.path.join(target_dir, filename)
        if os.path.exists(filepath):
            print(f"[{i:02d}/20] Already exists: {filename}")
            continue

        url = f"{base_url}/{filename}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp, open(filepath, 'wb') as f:
                f.write(resp.read())
            print(f"[{i:02d}/20] Successfully downloaded: {filename}")
        except Exception as e:
            print(f"[{i:02d}/20] Failed to download {filename}: {e}")

    print("All DRIVE test masks are ready!")

if __name__ == '__main__':
    download_drive_test_labels()
