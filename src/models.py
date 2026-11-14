import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaselineCNN(nn.Module):
    """
    A custom 3-block Convolutional Neural Network built from scratch
    to serve as our experimental baseline for 4-class Brain MRI classification.
    """
    def __init__(self, num_classes: int = 4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        logits = self.classifier(x)
        return logits

def build_transfer_model(num_classes: int = 4, freeze_early_layers: bool = True) -> nn.Module:
    """
    Loads a pre-trained ResNet-18 model, freezes early feature-extraction layers
    for fast CPU training, and replaces the head for 4-class MRI classification.
    """
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    if freeze_early_layers:
        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only 'layer4' (the high-level feature block) so it adapts to MRI scans
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Replace the final 1000-class ImageNet head with our 4-class MRI classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )

    return model

def count_parameters(model: nn.Module):
    """Returns (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

if __name__ == "__main__":
    model = build_transfer_model(num_classes=4)
    trainable, total = count_parameters(model)
    print(f"ResNet-18 Total Parameters    : {total:,}")
    print(f"ResNet-18 Trainable Parameters: {trainable:,} ({trainable/total*100:.1f}%)")