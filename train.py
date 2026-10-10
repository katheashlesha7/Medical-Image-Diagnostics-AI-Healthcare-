"""Train a ResNet-18 model on the NIH ChestX-ray14 findings labels."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from cxr.data import FINDINGS, ChestXrayDataset, load_records
from cxr.metrics import evaluate
from cxr.model import build_model, make_transforms


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-images", type=int, default=0,
                        help="Optional quick-run limit; whole patient groups are kept. 0 uses all images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Keep 0 for easiest Windows compatibility.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download/use ImageNet weights (useful offline, usually less effective).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_records, val_records, test_records = load_records(
        args.data_dir, max_images=args.max_images, seed=args.seed
    )

    train_dataset = ChestXrayDataset(train_records, make_transforms(training=True))
    val_dataset = ChestXrayDataset(val_records, make_transforms())
    test_dataset = ChestXrayDataset(test_records, make_transforms())
    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=use_pin_memory,
    )

    model = build_model(len(FINDINGS), pretrained=not args.no_pretrained).to(device)
    positive_counts = train_records[FINDINGS].to_numpy(dtype=np.float32).sum(axis=0)
    negative_counts = len(train_records) - positive_counts
    positive_weights = np.clip(negative_counts / np.maximum(positive_counts, 1.0), 1.0, 20.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best.pt"
    history = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        train_loss = running_loss / len(train_dataset)
        val_metrics = evaluate(model, val_loader, device)
        auc = val_metrics["macro_auroc"]
        auc_text = f"{auc:.4f}" if auc is not None else "n/a"
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | val macro AUROC {auc_text} | "
            f"val macro F1@0.5 {val_metrics['macro_f1_at_0_5']:.4f}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_macro_auroc": auc,
            "val_macro_f1_at_0_5": val_metrics["macro_f1_at_0_5"],
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_labels": FINDINGS,
                    "threshold": 0.5,
                    "epoch": epoch,
                    "seed": args.seed,
                    "val_metrics": val_metrics,
                    "preprocessing": "RGB; resize 256; center crop 224; ImageNet normalization",
                },
                checkpoint_path,
            )

    pd.DataFrame(history).to_csv(args.output_dir / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2)

    test_auc = test_metrics["macro_auroc"]
    test_auc_text = f"{test_auc:.4f}" if test_auc is not None else "n/a"
    print(f"Saved best checkpoint: {checkpoint_path}")
    print(
        f"Held-out test results | macro AUROC {test_auc_text} | "
        f"macro F1@0.5 {test_metrics['macro_f1_at_0_5']:.4f} | "
        f"macro recall@0.5 {test_metrics['macro_recall_at_0_5']:.4f}"
    )
    print(f"Detailed test metrics: {args.output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
