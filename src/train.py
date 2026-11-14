import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from models import BaselineCNN

# Hyperparameters
EPOCHS = 8
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MODEL_SAVE_PATH = Path("models/baseline_best.pth")
PLOT_SAVE_PATH = Path("notebooks/baseline_learning_curves.png")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Training  ", leave=False):
        images, labels = images.to(device), labels.to(device)

        # 1. Zero out gradients from the previous batch
        optimizer.zero_grad()

        # 2. Forward pass: compute predicted logits
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 3. Backward pass: compute gradients and update weights
        loss.backward()
        optimizer.step()

        # 4. Track batch statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # Disable gradient tracking during validation to save memory and speed up computation
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100.0
    return epoch_loss, epoch_acc

def save_learning_curves(history):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Baseline CNN Training & Validation Curves (Educational Only)", fontsize=13)

    # Plot Loss
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", marker="o")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
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
    print("--- Starting Baseline CNN Training ---")
    print("Disclaimer: Experimental educational model only. Not for clinical use.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)
    model = BaselineCNN(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # Save model checkpoint only when validation accuracy improves
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

    total_Minutes = (time.time() - start_time) / 60
    print(f"\n--- Training Complete in {total_Minutes:.1f} minutes ---")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.2f}%")
    print(f"Best weights saved to: {MODEL_SAVE_PATH}")

    save_learning_curves(history)

if __name__ == "__main__":
    main()