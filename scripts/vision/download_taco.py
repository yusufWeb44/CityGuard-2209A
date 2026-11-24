import json
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

TACO_ANN_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
TARGET_DIR = Path("data/raw/Urban_Trash")

def download_single_image(img_info):
    img_id = img_info["id"]
    url = img_info.get("flickr_url") or img_info.get("flickr_640_url")
    if not url:
        return False
    out_path = TARGET_DIR / f"taco_{img_id:05d}.jpg"
    if out_path.exists():
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
        return True
    except Exception:
        return False

def download_taco():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print("1/2 جاري جلب فهرس الصور الرسمي لـ TACO Dataset...")
    req = urllib.request.Request(TACO_ANN_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    images = data.get("images", [])
    print(f"2/2 جاري تحميل {len(images)} صورة إلى {TARGET_DIR}...")

    success = 0
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(download_single_image, img) for img in images]
        for future in tqdm(as_completed(futures), total=len(futures), desc="TACO Trash", unit="صورة"):
            if future.result():
                success += 1

    print(f"\nتم بنجاح! عدد الصور المحملة في Urban_Trash: {len(list(TARGET_DIR.glob('*.jpg')))}")

if __name__ == "__main__":
    download_taco()
