# Steel inclusion 4-class pipeline

Target classes:

- non-inclusion
- oxide
- sulfide
- oxy-sulfide

## Setup

```bash
cd steel_4class_pipeline
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put these files in one folder, for example `data/raw/`:

```text
2018_AlCaSMn_20kV_4.xlsx
Heat 1 images.zip
Heat 2 images.zip
Heat 3 images.zip
Heat 4 images.zip
```

## 1. Prepare manifests

```bash
python 01_prepare_4class_manifest.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --seed 42
```

This creates:

```text
data/processed/manifest_4class_all.csv
/data/processed/manifest_4class_balanced.csv
```

The balanced manifest is the paper-style comparison set. The all manifest is for high-accuracy training with imbalance handling.

## 2. Train image-only baselines

Paper-style balanced EfficientNet-B0:

```bash
python 02_train_4class_image.py \
  --manifest data/processed/manifest_4class_balanced.csv \
  --model efficientnet_b0 \
  --out_dir runs/effb0_balanced \
  --epochs 30 \
  --batch_size 32 \
  --img_size 224
```

Swin-Tiny transformer:

```bash
python 02_train_4class_image.py \
  --manifest data/processed/manifest_4class_balanced.csv \
  --model swin_tiny_patch4_window7_224 \
  --out_dir runs/swin_tiny_balanced \
  --epochs 35 \
  --batch_size 16 \
  --img_size 224
```

High-accuracy all-data training:

```bash
python 02_train_4class_image.py \
  --manifest data/processed/manifest_4class_all.csv \
  --model efficientnet_b0 \
  --out_dir runs/effb0_all_weighted \
  --epochs 35 \
  --batch_size 32 \
  --img_size 224 \
  --use_class_weights
```

## 3. Train hybrid model

Image + morphology + grayscale histogram features:

```bash
python 03_train_4class_hybrid.py \
  --manifest data/processed/manifest_4class_all.csv \
  --model efficientnet_b0 \
  --out_dir runs/hybrid_effb0_all_weighted \
  --epochs 35 \
  --batch_size 32 \
  --img_size 224 \
  --use_class_weights
```

## What to compare

Check each run's:

```text
metrics.json
classification_report.txt
confusion_matrix.png
best_model.pt
```

The benchmark to beat is around 78% accuracy for the 4-class CNN setup from the RF-vs-CNN paper.
