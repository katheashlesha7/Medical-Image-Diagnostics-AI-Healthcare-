"""Dataset loading and patient-level splitting for NIH ChestX-ray14."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]


def load_records(data_dir: Path, max_images: int = 0, seed: int = 42):
    """Load labels and find images; optionally choose a patient-grouped subset."""
    metadata_path = data_dir / "Data_Entry_2017_v2020.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find {metadata_path}. Put Data_Entry_2017_v2020.csv in the data folder."
        )

    images_root = data_dir / "images"
    if not images_root.exists():
        images_root = data_dir
    image_paths = {}
    for path in images_root.rglob("*.png"):
        image_paths.setdefault(path.name, path)
    if not image_paths:
        raise FileNotFoundError(
            f"No PNG images found under {images_root}. Extract the NIH image archives there."
        )

    frame = pd.read_csv(metadata_path)
    required = {"Image Index", "Finding Labels", "Patient ID"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {', '.join(sorted(missing))}")

    frame["image_path"] = frame["Image Index"].map(image_paths)
    missing_count = int(frame["image_path"].isna().sum())
    frame = frame.dropna(subset=["image_path"]).copy()
    if frame.empty:
        raise ValueError("Metadata was read, but none of its images were found in the image folder.")
    if missing_count:
        print(f"Skipping {missing_count:,} metadata rows whose image files are not present.")

    for finding in FINDINGS:
        frame[finding] = frame["Finding Labels"].fillna("").map(
            lambda labels: float(finding in labels.split("|"))
        )
    frame["Patient ID"] = frame["Patient ID"].astype(str)

    # Choose whole patients, rather than individual images, for a quick demo subset.
    rng = np.random.default_rng(seed)
    patients = frame["Patient ID"].drop_duplicates().to_numpy()
    rng.shuffle(patients)
    if max_images > 0 and len(frame) > max_images:
        selected = []
        selected_count = 0
        for patient in patients:
            selected.append(patient)
            selected_count += int((frame["Patient ID"] == patient).sum())
            if selected_count >= max_images:
                break
        frame = frame[frame["Patient ID"].isin(selected)].copy()
        patients = np.asarray(selected)
        print(
            f"Using {len(frame):,} images from {len(patients):,} complete patients "
            f"(--max-images {max_images:,})."
        )

    if len(patients) < 3:
        raise ValueError("At least 3 distinct patients with images are needed for train/val/test splits.")

    rng.shuffle(patients)
    n_patients = len(patients)
    n_train = max(1, int(n_patients * 0.70))
    n_val = max(1, int(n_patients * 0.15))
    if n_train + n_val >= n_patients:
        n_train = n_patients - 2
        n_val = 1
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients = set(patients[n_train + n_val :])

    train = frame[frame["Patient ID"].isin(train_patients)].reset_index(drop=True)
    val = frame[frame["Patient ID"].isin(val_patients)].reset_index(drop=True)
    test = frame[frame["Patient ID"].isin(test_patients)].reset_index(drop=True)
    print(
        f"Patient-level split: train {len(train):,} images / {len(train_patients):,} patients; "
        f"validation {len(val):,} / {len(val_patients):,}; "
        f"test {len(test):,} / {len(test_patients):,}."
    )
    return train, val, test


class ChestXrayDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform=None):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        labels = torch.tensor(row[FINDINGS].to_numpy(dtype=np.float32), dtype=torch.float32)
        return image, labels
