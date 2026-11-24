import random
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import imagehash
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = Path("manifests/image_manifest.csv")

CLASSES = [
    "Pothole",
    "Urban_Trash",
    "Water_Leakage",
    "Streetlight_Fault",
    "Spam_Irrelevant"
]

TARGET_SIZE = (224, 224)
MAX_IMAGES_PER_CLASS = 1500
RANDOM_SEED = 42
HASH_THRESHOLD = 4

def is_near_duplicate(new_hash, existing_hashes, threshold=HASH_THRESHOLD):
    for h in existing_hashes:
        if (new_hash - h) <= threshold:
            return True
    return False

def clean_and_collect_images():
    valid_records = []

    print("\n--- [1/3] Perceptual Hash (pHash) Deduplication & RGB Validation ---")
    for cls in CLASSES:
        cls_raw_dir = RAW_DIR / cls
        if not cls_raw_dir.exists():
            print(f"[WARNING] Directory not found: {cls_raw_dir}")
            continue

        image_files = [
            f for f in cls_raw_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ]
        
        cls_valid = []
        seen_phashes = []
        corrupted_count = 0
        duplicate_count = 0

        for img_path in tqdm(image_files, desc=f"Scanning {cls:<17}", unit="img"):
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
                    p_hash = imagehash.phash(img_rgb)

                    if is_near_duplicate(p_hash, seen_phashes):
                        duplicate_count += 1
                        continue

                    img_resized = img_rgb.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    seen_phashes.append(p_hash)
                    cls_valid.append((img_path, img_resized, cls))

            except (UnidentifiedImageError, OSError, ValueError):
                corrupted_count += 1

        random.Random(RANDOM_SEED).shuffle(cls_valid)
        selected = cls_valid[:MAX_IMAGES_PER_CLASS]
        valid_records.extend(selected)

        print(f" -> {cls:<18}: Unique={len(cls_valid):<4} | Selected={len(selected)} | Near-Duplicates={duplicate_count:<3} | Corrupted={corrupted_count}")

    return valid_records

def split_and_save(records):
    if not records:
        print("\n[ERROR] No valid images found.")
        return

    print("\n--- [2/3] Stratified Dataset Splitting (70% Train / 15% Val / 15% Test) ---")
    labels = [r[2] for r in records]

    train_recs, temp_recs = train_test_split(
        records, test_size=0.30, stratify=labels, random_state=RANDOM_SEED
    )
    temp_labels = [r[2] for r in temp_recs]
    val_recs, test_recs = train_test_split(
        temp_recs, test_size=0.50, stratify=temp_labels, random_state=RANDOM_SEED
    )

    splits = {"train": train_recs, "val": val_recs, "test": test_recs}
    manifest_rows = []
    print("\n--- [3/3] Resizing (224x224) & Generating image_manifest.csv ---")

    for split_name, split_items in splits.items():
        for idx, (orig_path, img_obj, cls) in enumerate(tqdm(split_items, desc=f"Saving {split_name:<5}", unit="img")):
            out_dir = PROCESSED_DIR / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)

            new_filename = f"{cls}_{split_name}_{idx:04d}.jpg"
            out_path = out_dir / new_filename
            img_obj.save(out_path, format="JPEG", quality=95)

            manifest_rows.append({
                "image_id": f"{cls}_{split_name}_{idx:04d}",
                "class_label": cls,
                "split": split_name,
                "file_path": out_path.as_posix(),
                "original_source_file": orig_path.name
            })

    df = pd.DataFrame(manifest_rows)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    print("\n========================================================")
    print("FINAL DATASET DISTRIBUTION SUMMARY (7,500 IMAGES):")
    print("========================================================")
    print(pd.crosstab(df["class_label"], df["split"], margins=True))
    print("========================================================")
    print(f"Manifest saved successfully at: {MANIFEST_PATH}\n")

if __name__ == "__main__":
    cleaned_data = clean_and_collect_images()
    split_and_save(cleaned_data)
