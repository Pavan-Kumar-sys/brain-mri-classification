import shutil
from pathlib import Path
import kagglehub

def setup_dataset():
    print("Downloading Brain Tumor MRI Dataset from Kaggle (around 150 MB)...")
    # Downloads the dataset to a local cache folder and returns the path
    cache_path = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
    print(f"Downloaded to cache: {cache_path}")

    target_dir = Path("data")
    target_dir.mkdir(exist_ok=True)

    # Copy 'Training' and 'Testing' folders directly into our project's 'data/' folder
    for folder_name in ["Training", "Testing"]:
        src_folder = cache_path / folder_name
        dst_folder = target_dir / folder_name

        if dst_folder.exists():
            print(f"'{dst_folder}' already exists. Skipping copy.")
        elif src_folder.exists():
            print(f"Copying {folder_name} images into {dst_folder}...")
            shutil.copytree(src_folder, dst_folder)
        else:
            print(f"[ERROR] Could not find {folder_name} inside {cache_path}")

    print("\n[SUCCESS] Dataset is ready inside 'data/Training' and 'data/Testing'!")

if __name__ == "__main__":
    setup_dataset()