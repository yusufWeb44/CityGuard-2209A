import urllib.request
import zipfile
import os
from pathlib import Path
from tqdm import tqdm

COCO_URL = "http://images.cocodataset.org/zips/val2017.zip"
ZIP_PATH = Path("data/raw/val2017.zip")
TARGET_DIR = Path("data/raw/Spam_Irrelevant")
TARGET_COUNT = 1600  # نأخذ 1600 لضمان بقاء 1500 بعد التنظيف

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract_coco():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print("1/3 جاري تحميل MS COCO 2017 Validation Set (~780 MB)...")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="COCO val2017") as t:
        urllib.request.urlretrieve(COCO_URL, filename=ZIP_PATH, reporthook=t.update_to)

    print(f"\n2/3 جاري استخراج أول {TARGET_COUNT} صورة إلى {TARGET_DIR}...")
    extracted = 0
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        for file_info in zf.infolist():
            if file_info.filename.endswith(".jpg"):
                filename = os.path.basename(file_info.filename)
                out_path = TARGET_DIR / filename
                with zf.open(file_info) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted += 1
                if extracted >= TARGET_COUNT:
                    break

    print("3/3 جاري حذف الملف المضغوط المؤقت لتوفير المساحة...")
    ZIP_PATH.unlink(missing_ok=True)
    print(f"تم بنجاح! عدد الصور في Spam_Irrelevant: {len(list(TARGET_DIR.glob('*.jpg')))}")

if __name__ == "__main__":
    download_and_extract_coco()
