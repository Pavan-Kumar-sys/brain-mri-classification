import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import BrainMRIDataset, get_transforms
from models import build_transfer_model, count_parameters

# Hyperparameters
EPOCHS = 5
LEARNING_RATE = 5e-4
BATCH_SIZE = 32
MODEL_SAVE_PATH = Path("models/resnet18_best.pth")
PLOT_SAVE_PATH = Path("notebooks/resnet18_learning_curves.png")

def get_fast_loaders(data_dir: Path = Path("data"), batch_size: int = BATCH_SIZE):
    train_tf, eval_tf = get_transforms()
    train_ds = BrainMRIDataset(data_dir / "train_split.csv", transform=train_tf)
    val_ds = BrainMRIDataset(data_dir / "val_split.csv", transform=eval_tf)

    # Using 2 background workers speeds up image loading while CPU computes
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)
    return train_loader, val_loader

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Training  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, (correct / total) * 100.0

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / total, (correct / total) * 100.0

def save_learning_curves(history):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ResNet-18 Transfer Learning Curves (Educational Only)", fontsize=13)

    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", marker="o")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["train_acc"], label="Train Acc (%)", marker="o")
    ax2.plot(epochs_range, history["val_acc"], label="Val Acc (%)", marker="o")
    ax2.set_title("Classification Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    PLOT_SAVE_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(PLOT_SAVE_PATH, dpi=150)
    plt.close()
    print(f"\n[INFO] Saved learning curves plot to: {PLOT_SAVE_PATH}")

def main():
    print("--- Starting ResNet-18 Transfer Learning (Fine-Tuning) ---")
    print("Disclaimer: Experimental educational model only. Not for clinical use.\n")

    # Optimize CPU thread usage
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(cpu_cores)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} (Using {cpu_cores} CPU threads)")

    train_loader, val_loader = get_fast_loaders(batch_size=BATCH_SIZE)
    model = build_transfer_model(num_classes=4, freeze_early_layers=True).to(device)

    trainable_params, total_params = count_parameters(model)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    criterion = nn.CrossEntropyLoss()
    # Pass only trainable parameters to AdamW optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        saved_marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            saved_marker = " -> [Saved Best Model]"

        print(
            f"Epoch [{epoch}/{EPOCHS}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f} ({train_acc:.1f}%) | "
            f"Val Loss: {val_loss:.4f} ({val_acc:.1f}%){saved_marker}"
        )

    total_minutes = (time.time() - start_time) / 60
    print(f"\n--- Transfer Learning Complete in {total_minutes:.1f} minutes ---")
    print(f"Baseline Best Val Accuracy : 87.44%")
    print(f"ResNet-18 Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Best weights saved to      : {MODEL_SAVE_PATH}")

    save_learning_curves(history)

if __name__ == "__main__":
    main()