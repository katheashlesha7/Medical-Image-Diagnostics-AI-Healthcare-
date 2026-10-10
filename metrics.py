"""Evaluation helpers for imbalanced multi-label classification."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import torch


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probabilities = []
    targets = []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(labels.cpu().numpy())

    probs = np.concatenate(probabilities, axis=0)
    truth = np.concatenate(targets, axis=0).astype(np.int32)
    predictions = (probs >= 0.5).astype(np.int32)
    # Import here to keep the label order aligned with the dataset/model head.
    from .data import FINDINGS

    auc_values = []
    per_label = {}
    for index, name in enumerate(FINDINGS):
        if np.unique(truth[:, index]).size == 2:
            auc = float(roc_auc_score(truth[:, index], probs[:, index]))
            auc_values.append(auc)
        else:
            auc = None
        per_label[name] = {
            "auroc": auc,
            "precision_at_0_5": float(precision_score(truth[:, index], predictions[:, index], zero_division=0)),
            "recall_at_0_5": float(recall_score(truth[:, index], predictions[:, index], zero_division=0)),
            "f1_at_0_5": float(f1_score(truth[:, index], predictions[:, index], zero_division=0)),
            "positive_count": int(truth[:, index].sum()),
        }
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "macro_auroc": float(np.mean(auc_values)) if auc_values else None,
        "macro_f1_at_0_5": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "macro_recall_at_0_5": float(recall_score(truth, predictions, average="macro", zero_division=0)),
        "per_label": per_label,
    }
    return metrics
