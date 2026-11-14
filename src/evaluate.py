from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from dataset import get_dataloaders, CLASSES
from models import BaselineCNN, build_transfer_model

BASELINE_WEIGHTS = Path("models/baseline_best.pth")
RESNET_WEIGHTS = Path("models/resnet18_best.pth")
CM_PLOT_PATH = Path("notebooks/confusion_matrices_comparison.png")
REPORT_TXT_PATH = Path("notebooks/test_evaluation_report.txt")

def get_predictions(model, loader, device):
    """Runs inference on the test set and returns true labels, predicted labels, and probabilities."""
    model.eval()
    all_Labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Evaluating Test Set", leave=False):
            images = images.to(device)
            logits = model(images)
            # Convert raw logits into probabilities (0.0 to 1.0) using Softmax
            probs = F.softmax(logits, dim=1)
            _, preds = torch.max(probs, dim=1)

            all_Labels.extend(labels.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_Labels), np.array(all_preds), np.array(all_probs)

def evaluate_model(name, model, weights_path, test_loader, device):
    print(f"\n==========================================")
    print(f" Evaluating: {name}")
    print(f"==========================================")

    if not weights_path.exists():
        print(f"[ERROR] Weights not found at {weights_path}")
        return None

    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)

    y_true, y_pred, y_prob = get_predictions(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro") * 100.0
    # One-vs-Rest (OvR) multi-class ROC-AUC
    roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") * 100.0

    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Test Accuracy : {acc:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.2f}%")
    print(f"Macro ROC-AUC : {roc_auc:.2f}%\n")
    print("Per-Class Classification Report:")
    print(report)

    return {
        "name": name,
        "acc": acc,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "report": report,
        "cm": cm
    }

def plot_confusion_matrices(results_list):
    fig, axes = plt.subplots(1, len(results_list), figsize=(14, 5.5))
    fig.suptitle("Held-Out Test Set Confusion Matrices (Experimental / Educational Only)", fontsize=14)

    if len(results_list) == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        sns.heatmap(
            res["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            ax=ax,
            cbar=False
        )
        ax.set_title(f"{res['name']}\nAcc: {res['acc']:.2f}% | Macro F1: {res['macro_f1']:.2f}%")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")

    plt.tight_layout()
    CM_PLOT_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(CM_PLOT_PATH, dpi=150)
    plt.close()
    print(f"\n[SUCCESS] Saved side-by-side Confusion Matrices to: {CM_PLOT_PATH}")

def main():
    print("--- Unsealing Held-Out Test Set for Final Evaluation ---")
    print("Disclaimer: Educational image classification benchmark only. Never use for clinical diagnosis.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(batch_size=32)

    results = []

    # 1. Evaluate Baseline CNN
    baseline_model = BaselineCNN(num_classes=4)
    res_base = evaluate_model("Baseline Custom CNN", baseline_model, BASELINE_WEIGHTS, test_loader, device)
    if res_base:
        results.append(res_base)

    # 2. Evaluate Fine-Tuned ResNet-18
    resnet_model = build_transfer_model(num_classes=4, freeze_early_layers=False)
    res_resnet = evaluate_model("ResNet-18 (Transfer Learning)", resnet_model, RESNET_WEIGHTS, test_loader, device)
    if res_resnet:
        results.append(res_resnet)

    # Save visual confusion matrices and text report
    if results:
        plot_confusion_matrices(results)
        with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
            f.write("=== BRAIN MRI CLASSIFICATION: HELD-OUT TEST SET REPORT ===\n")
            f.write("Disclaimer: Educational experiment only. Not a medical diagnostic tool.\n\n")
            for r in results:
                f.write(f"Model: {r['name']}\n")
                f.write(f"Test Accuracy : {r['acc']:.2f}%\n")
                f.write(f"Macro F1-Score: {r['macro_f1']:.2f}%\n")
                f.write(f"Macro ROC-AUC : {r['roc_auc']:.2f}%\n\n")
                f.write(r["report"] + "\n" + "="*50 + "\n\n")
        print(f"[SUCCESS] Saved detailed text metrics report to: {REPORT_TXT_PATH}")

if __name__ == "__main__":
    main()