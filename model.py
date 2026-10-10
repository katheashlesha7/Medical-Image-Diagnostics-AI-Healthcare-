"""ResNet-18 transfer-learning model and image transforms."""

from torch import nn
from torchvision import models, transforms


def build_model(num_labels: int, pretrained: bool = True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_labels)
    return model


def make_transforms(training: bool = False):
    steps = [transforms.Resize(256), transforms.CenterCrop(224)]
    if training:
        # Small geometric changes only; avoid left/right flips of anatomy.
        steps.append(transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)))
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(steps)
