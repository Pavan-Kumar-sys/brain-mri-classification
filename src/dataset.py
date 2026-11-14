from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Standard constants
IMG_SIZE = 224
BATCH_SIZE = 32
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Standard ImageNet normalization stats (works well for both custom CNNs and transfer learning)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class BrainMRIDataset(Dataset):
    """Custom PyTorch Dataset reading directly from our deduplicated CSV manifests."""
    def __init__(self, csv_path: Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["filepath"])
        label = int(row["label"])

        # Always convert to 3-channel RGB to standardize 'L' and 'RGB' files
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """Returns separate transformation pipelines for training vs. evaluation."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), # Scales pixels from [0, 255] to [0.0, 1.0]
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    return train_transform, eval_transform

def get_dataloaders(data_dir: Path = Path("data"), batch_size: int = BATCH_SIZE):
    """Builds and returns Train, Validation, and Test DataLoaders."""
    train_tf, eval_tf = get_transforms()

    train_ds = BrainMRIDataset(data_dir / "train_split.csv", transform=train_tf)
    val_ds = BrainMRIDataset(data_dir / "val_split.csv", transform=eval_tf)
    test_ds = BrainMRIDataset(data_dir / "test_split.csv", transform=eval_tf)

    # num_workers=0 ensures maximum compatibility on Windows without multiprocessing errors
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    print("--- Testing PyTorch DataLoaders ---")
    train_loader, val_loader, test_loader = get_dataloaders()

    # Fetch a single batch from the training loader
    images, labels = next(iter(train_loader))
    print(f"Train batches available : {len(train_loader)}")
    print(f"Val batches available   : {len(val_loader)}")
    print(f"Test batches available  : {len(test_loader)}")
    print(f"Single batch image shape: {images.shape}  (Expected: [32, 3, 224, 224])")
    print(f"Single batch label shape: {labels.shape}  (Expected: [32])")
    print(f"Sample labels in batch  : {labels[:10].tolist()}")
    print("\n[SUCCESS] DataLoaders are verified and ready for model training!")