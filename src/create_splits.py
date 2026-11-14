import hashlib
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
RANDOM_SEED = 42

def get_image_pixel_hash(img_path: Path) -> str:
    """Computes an MD5 hash of the grayscale pixel bytes to catch exact duplicates."""
    with Image.open(img_path) as img:
        # Convert to grayscale and standardize size briefly just for fingerprinting
        gray_img = img.convert("L")
        return hashlib.md5(gray_img.tobytes()).hexdigest()

def collect_clean_records():
    seen_hashes = set()
    train_val_records = []
    test_records = []
    duplicates_removed = 0
    leaked_between_splits = 0

    print("Scanning Training images and computing pixel fingerprints...")
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / "Training" / cls
        for img_path in sorted(cls_dir.iterdir()):
            if not img_path.is_file():
                continue
            img_hash = get_image_pixel_hash(img_path)
            if img_hash in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes.add(img_hash)
            train_val_records.append({
                "filepath": str(img_path.as_posix()),
                "class_name": cls,
                "label": label_idx
            })

    print("Scanning Testing images and checking for cross-split leakage...")
    for label_idx, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / "Testing" / cls
        for img_path in sorted(cls_dir.iterdir()):
            if not img_path.is_file():
                continue
            img_hash = get_image_pixel_hash(img_path)
            if img_hash in seen_hashes:
                # Caught an image in Testing that already appeared in Training (or earlier in Testing)!
                leaked_between_splits += 1
                continue
            seen_hashes.add(img_hash)
            test_records.append({
                "filepath": str(img_path.as_posix()),
                "class_name": cls,
                "label": label_idx
            })

    print(f"\n--- Deduplication & Leakage Report ---")
    print(f"Within-Training duplicates removed : {duplicates_removed}")
    print(f"Testing/Training leaked duplicates : {leaked_between_splits}")
    return pd.DataFrame(train_val_records), pd.DataFrame(test_records)

def create_and_save_splits():
    train_val_df, test_df = collect_clean_records()

    # Stratified split: 85% Training, 15% Validation
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=train_val_df["label"]
    )

    # Save CSV manifests
    train_df.to_csv(DATA_DIR / "train_split.csv", index=False)
    val_df.to_csv(DATA_DIR / "val_split.csv", index=False)
    test_df.to_csv(DATA_DIR / "test_split.csv", index=False)

    print("\n--- Final Clean Split Counts ---")
    for name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n[{name} Set] Total: {len(df)} images")
        counts = df["class_name"].value_counts()
        for cls in CLASSES:
            c = counts.get(cls, 0)
            pct = (c / len(df)) * 100
            print(f"  - {cls:<12}: {c:>4} ({pct:.1f}%)")

    print("\n[SUCCESS] Saved train_split.csv, val_split.csv, and test_split.csv inside 'data/'!")

if __name__ == "__main__":
    create_and_save_splits()