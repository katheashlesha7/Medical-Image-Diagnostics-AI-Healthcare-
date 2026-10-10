# Medical Image Diagnostics: Chest X-ray Classification

An educational machine-learning project that fine-tunes a pretrained ResNet-18 to predict 14 potentially co-occurring findings from frontal chest X-rays in the NIH ChestX-ray14 dataset. It includes a patient-level training split, evaluation output, command-line inference, and an optional local Streamlit upload demo.

> **Research/education only.** This code is not a medical device, has not been clinically validated, and must not be used to diagnose a person, rule out disease, prioritize care, or make treatment decisions. The displayed scores are model outputs, not calibrated clinical probabilities. A clinician should interpret medical imaging in context.

## What the project does

Given a chest X-ray image, the model returns one score for each of 14 findings. More than one finding can be present, so this is a **multi-label classification** problem rather than a 15-way choice. The output layer has 14 independent sigmoid scores and training uses binary cross-entropy with a capped positive-class weight to reduce the effect of label imbalance.

The project replaces ResNet-18's ImageNet classification head with a 14-output linear layer, then fine-tunes the network. The image pipeline converts PNGs to RGB, resizes and center-crops to 224 × 224, and applies ImageNet normalization. Small rotations/translations are used during training; left/right flips are deliberately omitted.

### Findings

Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia. “No Finding” is represented by all 14 outputs being zero; it is not a fifteenth output.

## Dataset

Get the NIH ChestX-ray14 files from the [NIH Clinical Center download page](https://nihcc.app.box.com/v/ChestXray-NIHCC). This commonly cited release contains 112,120 frontal-view images from 30,805 patients, with image-level labels mined from radiology reports. Download/extract the metadata CSV (`Data_Entry_2017.csv`) and the image archive parts. The full image set takes substantial disk space; for a first run you can extract a subset of archives and train with `--max-images`.

Place files like this (the image loader searches recursively for PNGs):

```text
data/
├── Data_Entry_2017.csv
└── images/
    ├── images_001/
    │   ├── 00000001_000.png
    │   └── ...
    ├── images_002/
    └── ...
```

The loader expects these CSV columns: `Image Index`, `Finding Labels`, and `Patient ID`. It skips rows whose image files are not extracted. The NIH dataset attribution requirements are to link to the NIH download page, cite the original CVPR 2017 paper, and acknowledge the NIH Clinical Center as the data provider. This project acknowledges the **NIH Clinical Center** as the source of the dataset; see the [dataset documentation](https://cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest) and [original paper](https://arxiv.org/abs/1705.02315).

## Project layout

```text
app.py                 Optional local Streamlit demo
train.py               Training and held-out evaluation entry point
predict.py             Single-image command-line inference
cxr/data.py            Label parsing, image loading, patient splits
cxr/model.py           ResNet-18 and preprocessing transforms
cxr/metrics.py         AUROC, F1, precision, and recall
data/                  Put downloaded dataset files here (not committed)
models/                Checkpoint and metrics output (not committed)
RESUME.md              Resume wording and interview notes
```

## Set up and run (Windows PowerShell)

Use Python 3.10 or newer. From the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, open a new terminal with an allowed execution policy or run the environment's interpreter directly as `.\.venv\Scripts\python.exe` in the commands below.

### Train a quick subset

After placing metadata and some extracted PNG archives in `data/`:

```powershell
python train.py --data-dir data --epochs 3 --batch-size 8 --max-images 3000
```

The script keeps whole patients together, then creates a deterministic 70% train / 15% validation / 15% test split. The `--max-images` option chooses complete patient groups and can therefore include slightly more than the requested number. If enabled, ImageNet weights download on the first run. If you are offline, add `--no-pretrained` (this changes the experiment and usually makes it harder to learn well).

For a fuller run, extract the complete dataset, omit `--max-images`, and increase epochs as compute allows:

```powershell
python train.py --data-dir data --epochs 10 --batch-size 16
```

On CPU, training can be slow. Lower `--batch-size` if memory is limited. If a CUDA-enabled PyTorch installation is needed, select the appropriate command from the [official PyTorch install selector](https://pytorch.org/get-started/locally/).

Training writes:

- `models/best.pt` — checkpoint with the best validation loss.
- `models/training_history.csv` — per-epoch losses and validation metrics.
- `models/test_metrics.json` — final metrics on the held-out patient test split.

The printed and saved metrics include macro AUROC (only labels with both classes represented in the test split), macro F1 and recall at the fixed 0.5 cutoff, and per-label precision, recall, F1, AUROC, and positive counts. Rare findings may be missing from a small subset's validation/test split; their AUROC is then reported as `null` and excluded from macro AUROC. The 0.5 cutoff is for demonstration, not a clinically selected threshold.

### Predict one image

```powershell
python predict.py path\to\chest_xray.png
```

Optionally specify a checkpoint and score cutoff:

```powershell
python predict.py path\to\chest_xray.png --checkpoint models\best.pt --threshold 0.35
```

The command prints model scores and whether each score is above or below the demo cutoff. It does not return a diagnosis.

### Open the local web demo

Train a checkpoint first, then:

```powershell
streamlit run app.py
```

Streamlit prints a local address to open in your browser. Keep the app local; if you deploy or host it, uploaded images will be sent to that server. Do not upload identifiable or sensitive patient images to a public demo.

## Run on macOS/Linux

Create and activate a virtual environment with `python3 -m venv .venv` and `source .venv/bin/activate`, install dependencies with `python -m pip install -r requirements.txt`, then use the same `python train.py`, `python predict.py`, and `streamlit run app.py` commands with POSIX paths.

## Evaluation and limitations

- The split is generated by this project at the patient level. It is not the official NIH benchmark split, so its metrics should not be compared directly with benchmark papers.
- The original dataset's image labels were mined from reports and are weak/noisy labels, not a complete radiologist-confirmed ground truth for every image. In this starter pipeline, findings absent from a row's label string are set to zero; “not mentioned” is not guaranteed to mean “clinically absent.”
- Random splits can have few examples of rare classes, especially when training only on a small subset. Report per-label support alongside aggregate metrics.
- Macro AUROC describes ranking on this particular held-out split. It does not show calibration, clinical utility, subgroup fairness, prospective performance, or safety.
- The default score threshold is 0.5. It has not been calibrated against a target sensitivity, clinical workflow, or expert reference standard.
- The model is trained on one dataset source and may fail on different hospitals, populations, equipment, acquisition protocols, or patient groups.

## Resume entry

See [RESUME.md](RESUME.md) for a copy-ready project description and a results line to fill only after training. Do not add invented metrics or imply clinical validation.

## References

1. Wang et al., [ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks for Weakly-Supervised Classification and Localization of Common Thorax Diseases](https://arxiv.org/abs/1705.02315), CVPR 2017. The paper describes the original collection and text-mined multi-label setup.
2. NIH Clinical Center, [ChestX-ray-NIHCC download](https://nihcc.app.box.com/v/ChestXray-NIHCC).
3. TorchVision, [ResNet-18 model and pretrained weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
