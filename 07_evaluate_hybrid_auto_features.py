
import argparse
import json
from pathlib import Path
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


feat_mod = SourceFileLoader("auto_features", "04_auto_extract_features.py").load_module()
train_mod = SourceFileLoader("hybrid_train", "03_train_4class_hybrid.py").load_module()


class AutoFeatureHybridDataset(Dataset):
    def __init__(self, df, labels, feature_cols, mean, scale, img_size):
        self.df = df.reset_index(drop=True)
        self.labels = labels
        self.label_to_id = {label: i for i, label in enumerate(labels)}
        self.feature_cols = feature_cols
        self.mean = mean
        self.scale = scale

        self.tfm = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = row["label"]

        img = Image.open(image_path).convert("L")
        x_img = self.tfm(img)

        feats, _ = feat_mod.extract_features(image_path)

        x_tab = np.array(
            [float(feats.get(c, 0.0)) for c in self.feature_cols],
            dtype=np.float32
        )
        x_tab = np.nan_to_num(x_tab, nan=0.0, posinf=0.0, neginf=0.0)
        x_tab = (x_tab - self.mean) / self.scale

        y = self.label_to_id[label]

        return x_img, torch.tensor(x_tab, dtype=torch.float32), y, image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest_4class_all.csv")
    parser.add_argument("--ckpt", default="runs/hybrid_effb0_all_weighted/best_model.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--out_dir", default="runs/hybrid_auto_feature_eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)

    labels = ckpt["labels"]
    model_name = ckpt["model_name"]
    img_size = ckpt["img_size"]
    feature_cols = ckpt["feature_cols"]
    mean = np.array(ckpt["tabular_mean"], dtype=np.float32)
    scale = np.array(ckpt["tabular_scale"], dtype=np.float32)

    print("Model:", model_name)
    print("Labels:", labels)
    print("Image size:", img_size)
    print("Auto-extracted tabular features:", len(feature_cols))

    df = pd.read_csv(args.manifest)

    if args.split:
        df = df[df["split"] == args.split].copy()

    df = df[df["label"].isin(labels)].copy()

    if args.max_rows is not None:
        df = df.sample(args.max_rows, random_state=42).reset_index(drop=True)

    print("Evaluation rows:", len(df))
    print("Class counts:")
    print(df["label"].value_counts())

    dataset = AutoFeatureHybridDataset(
        df=df,
        labels=labels,
        feature_cols=feature_cols,
        mean=mean,
        scale=scale,
        img_size=img_size
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    model = train_mod.HybridModel(
        model_name,
        n_tab=len(feature_cols),
        n_classes=len(labels),
        pretrained=False
    )

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for x_img, x_tab, y, paths in tqdm(loader, desc="Evaluating"):
            x_img = x_img.to(device)
            x_tab = x_tab.to(device)

            logits = model(x_img, x_tab)
            probs = F.softmax(logits, dim=1)

            preds = probs.argmax(dim=1).cpu().numpy()

            y_true.extend(y.numpy().tolist())
            y_pred.extend(preds.tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_paths.extend(list(paths))

    y_true_labels = [labels[i] for i in y_true]
    y_pred_labels = [labels[i] for i in y_pred]

    acc = accuracy_score(y_true_labels, y_pred_labels)
    bal_acc = balanced_accuracy_score(y_true_labels, y_pred_labels)
    macro_f1 = f1_score(y_true_labels, y_pred_labels, average="macro")
    weighted_f1 = f1_score(y_true_labels, y_pred_labels, average="weighted")

    report_txt = classification_report(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        digits=4
    )

    report_dict = classification_report(
        y_true_labels,
        y_pred_labels,
        labels=labels,
        output_dict=True
    )

    cm = confusion_matrix(
        y_true_labels,
        y_pred_labels,
        labels=labels
    )

    metrics = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "num_samples": len(df),
        "split": args.split,
        "model": model_name,
        "feature_mode": "auto_extracted_from_image"
    }

    print("\n==============================")
    print("FRONTEND-STYLE HYBRID RESULTS")
    print("==============================")
    print(json.dumps(metrics, indent=2))

    print("\nClassification report:")
    print(report_txt)

    print("Confusion matrix labels:")
    print(labels)
    print(cm)

    # Save outputs
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report_txt)

    with open(out_dir / "classification_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")

    result_rows = []
    for path, true_label, pred_label, probs in zip(all_paths, y_true_labels, y_pred_labels, all_probs):
        row = {
            "image_path": path,
            "true_label": true_label,
            "pred_label": pred_label,
            "correct": true_label == pred_label
        }
        for label, prob in zip(labels, probs):
            row[f"prob_{label}"] = prob
        result_rows.append(row)

    pd.DataFrame(result_rows).to_csv(out_dir / "predictions.csv", index=False)

    print("\nSaved results to:", out_dir)


if __name__ == "__main__":
    main()
