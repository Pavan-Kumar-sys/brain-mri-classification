from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import get_transforms, CLASSES, IMG_SIZE
from models import build_transfer_model

RESNET_WEIGHTS = Path("models/resnet18_best.pth")
TEST_CSV = Path("data/test_split.csv")
GRADCAM_PLOT = Path("notebooks/gradcam_explanations.png")
ERROR_PLOT = Path("notebooks/error_analysis_misclassified.png")

class GradCAM:
    """Computes Grad-CAM heatmaps for a target convolutional layer in PyTorch."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture forward activations and backward gradients
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = torch.argmax( logits, dim=1).item()

        confidence = probs[0, class_idx].item() * 100.0

        # Backpropagate the score of the predicted class
        score = logits[0, class_idx]
        score.backward()

        # Global average pooling over the gradients: [1, C, H, W] -> [C]
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32, device=input_tensor.device)

        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]

        # Apply ReLU so we only highlight features that positively support the class
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam.cpu().numpy(), class_idx, confidence

def overlay_heatmap(orig_pil_img, cam_2d, alpha=0.45):
    """Resizes the 7x7 Grad-CAM grid to 224x224 and blends it over the MRI image."""
    orig_resized = orig_pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB")
    orig_np = np.array(orig_resized) / 255.0

    # Resize CAM using PIL bilinear interpolation
    cam_pil = Image.fromarray(np.uint8(255 * cam_2d)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    cam_resized = np.array(cam_pil) / 255.0

    # Apply 'jet' colormap to heatmap
    heatmap_rgb = cm.jet(cam_resized)[:, :, :3]
    blended = (1 - alpha) * orig_np + alpha * heatmap_rgb
    blended = np.clip(blended, 0, 1)
    return orig_np, blended

def main():
    print("--- Running Grad-CAM Visual Explainability & Error Analysis ---")
    print("Disclaimer: Experimental educational visualization only. Not for clinical diagnosis.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_tf = get_transforms()

    # Load fine-tuned ResNet-18 and ensure layer4 tracks gradients for Grad-CAM
    model = build_transfer_model(num_classes=4, freeze_early_layers=False)
    model.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    for param in model.layer4.parameters():
        param.requires_grad = True

    grad_cam = GradCAM(model, target_layer=model.layer4[-1])
    test_df = pd.read_csv(TEST_CSV)

    # 1. Generate Grad-CAM for 1 sample of each class
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    fig.suptitle("ResNet-18 Grad-CAM Attention Maps on Held-Out Test Set (Educational Only)", fontsize=14)

    for col_idx, cls_name in enumerate(CLASSES):
        sample_row = test_df[test_df["class_name"] == cls_name].iloc[0]
        img_path = Path(sample_row["filepath"])

        with Image.open(img_path) as raw_img:
            rgb_img = raw_img.convert("RGB")
            input_tensor = eval_tf(rgb_img).unsqueeze(0).to(device)
            cam_2d, pred_idx, conf = grad_cam.generate_heatmap(input_tensor)
            orig_np, blended_np = overlay_heatmap(rgb_img, cam_2d)

        pred_name = CLASSES[pred_idx]
        axes[0, col_idx].imshow(orig_np)
        axes[0, col_idx].set_title(f"True: {cls_name}\nOriginal MRI", fontsize=11)
        axes[0, col_idx].axis("off")

        axes[1, col_idx].imshow(blended_np)
        axes[1, col_idx].set_title(f"Pred: {pred_name} ({conf:.1f}%)\nGrad-CAM Focus", fontsize=11)
        axes[1, col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(GRADCAM_PLOT, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved Grad-CAM visual report to: {GRADCAM_PLOT}")

    # 2. Find Misclassified Test Samples for Error Analysis
    print("Scanning test set for misclassified examples...")
    misclassified = []
    with torch.no_grad():
        for _, row in test_df.iterrows():
            img_path = Path(row["filepath"])
            true_label = int(row["label"])
            with Image.open(img_path) as raw_img:
                input_tensor = eval_tf(raw_img.convert("RGB")).unsqueeze(0).to(device)
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                if pred_idx.item() != true_label:
                    misclassified.append({
                        "filepath": img_path,
                        "true_name": CLASSES[true_label],
                        "pred_name": CLASSES[pred_idx.item()],
                        "conf": conf.item() * 100.0
                    })
                    if len(misclassified) == 4:
                        break

    if misclassified:
        fig, axes = plt.subplots(1, len(misclassified), figsize=(14, 4.2))
        fig.suptitle("Error Analysis: Misclassified Test Set Examples (ResNet-18)", fontsize=14)
        if len(misclassified) == 1:
            axes = [axes]
        for ax, item in zip(axes, misclassified):
            with Image.open(item["filepath"]) as img:
                ax.imshow(img.convert("L"), cmap="gray")
                ax.set_title(
                    f"True: {item['true_name']}\nPred: {item['pred_name']} ({item['conf']:.1f}%)",
                    color="darkred",
                    fontsize=11
                )
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(ERROR_PLOT, dpi=150)
        plt.close()
        print(f"[SUCCESS] Saved Error Analysis plot to: {ERROR_PLOT}")

if __name__ == "__main__":
    main()