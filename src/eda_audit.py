import os
from pathlib import Path
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt

# Define path to dataset
DATA_DIR = Path("data")
SPLITS = ["Training", "Testing"]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def audit_dataset():
    print("--- Starting Dataset Audit ---\n")
    summary_counts = {}
    dimensions = Counter()
    color_modes = Counter()
    corrupted_files = []

    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"[ERROR] Folder not found: {split_dir.resolve()}")
            print("Please check that 'Training' and 'Testing' are directly inside 'data/'.")
            return

        summary_counts[split] = {}
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"[WARNING] Class folder missing: {cls_dir}")
                continue

            files = [f for f in cls_dir.iterdir() if f.is_file()]
            summary_counts[split][cls] = len(files)

            # Inspect each image
            for img_path in files:
                try:
                    with Image.open(img_path) as img:
                        dimensions[img.size] += 1  # (width, height)
                        color_modes[img.mode] += 1 # 'L' = grayscale, 'RGB' = 3-channel
                except Exception as e:
                    corrupted_files.append((str(img_path), str(e)))

    # 1. Print Class Counts
    print("1. Class Distribution:")
    total_images = 0
    for split, counts in summary_counts.items():
        split_total = sum(counts.values())
        total_images += split_total
        print(f"\n  [{split}] (Total: {split_total})")
        for cls, count in counts.items():
            pct = (count / split_total * 100) if split_total > 0 else 0
            print(f"    - {cls:<12}: {count:>4} images ({pct:.1f}%)")

    print(f"\n  Grand Total Images: {total_images}")

    # 2. Print Image Modes (Grayscale vs RGB)
    print("\n2. Color Modes Found (L = Grayscale, RGB = 3-Channel Color):")
    for mode, count in color_modes.most_common():
        print(f"    - {mode}: {count} images")

    # 3. Print Top 5 Most Common Image Dimensions
    print("\n3. Top 5 Most Common Image Sizes (Width x Height):")
    for size, count in dimensions.most_common(5):
        print(f"    - {size[0]}x{size[1]}: {count} images")
    print(f"    (Total unique image resolutions found: {len(dimensions)})")

    # 4. Report Corrupted Files
    print(f"\n4. Corrupted / Unreadable Images: {len(corrupted_files)}")
    for bad_file, err in corrupted_files[:5]:
        print(f"    - {bad_file}: {err}")

    # 5. Save a visual grid of sample MRI images
    save_sample_grid()

def save_sample_grid():
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("Sample Brain MRI Scans (Experimental / Educational Only)", fontsize=14)

    for idx, cls in enumerate(CLASSES):
        cls_dir = DATA_DIR / "Training" / cls
        sample_files = list(cls_dir.glob("*"))
        if sample_files:
            with Image.open(sample_files[0]) as img:
                axes[idx].imshow(img, cmap="gray")
                axes[idx].set_title(f"Class: {cls}\nSize: {img.size}")
                axes[idx].axis("off")

    plt.tight_layout()
    output_path = Path("notebooks/sample_mri_grid.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\n[SUCCESS] Saved sample visual grid to: {output_path}")

if __name__ == "__main__":
    audit_dataset()