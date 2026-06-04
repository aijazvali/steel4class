import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

LABELS = ["non-inclusion", "oxide", "sulfide", "oxy-sulfide"]
BASE_FEATURES = [
    "Intensity", "Davg", "Dcirc", "Dmax", "Dmin", "Dperp", "Aspect",
    "Area", "Perim", "Roundness", "Elongation", "Pixel.Size...m.per.pixel."
]
G_FEATURES = [f"g{i}" for i in range(256)]
FEATURE_COLS = BASE_FEATURES + G_FEATURES


class HybridDataset(Dataset):
    def __init__(self, df, feature_cols, scaler, transform=None):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.transform = transform
        x = self.df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype("float32")
        self.x_tab = self.scaler.transform(x).astype("float32")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        tab = torch.tensor(self.x_tab[idx], dtype=torch.float32)
        y = int(row["label_id"])
        return img, tab, y


class HybridModel(nn.Module):
    def __init__(self, backbone_name, n_tab, n_classes=4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        img_dim = self.backbone.num_features
        self.tab_net = nn.Sequential(
            nn.Linear(n_tab, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(img_dim + 128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, n_classes),
        )

    def forward(self, img, tab):
        img_feat = self.backbone(img)
        tab_feat = self.tab_net(tab)
        return self.head(torch.cat([img_feat, tab_feat], dim=1))


def make_transforms(img_size):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=180),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.90, 1.05)),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, eval_tfms


def evaluate(model, loader, device):
    model.eval()
    all_y, all_pred = [], []
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for img, tab, y in loader:
            img, tab, y = img.to(device), tab.to(device), y.to(device)
            logits = model(img, tab)
            loss = ce(logits, y)
            pred = logits.argmax(dim=1)
            loss_sum += loss.item() * len(y)
            n += len(y)
            all_y.extend(y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
    all_y = np.array(all_y); all_pred = np.array(all_pred)
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": accuracy_score(all_y, all_pred),
        "balanced_accuracy": balanced_accuracy_score(all_y, all_pred),
        "macro_f1": f1_score(all_y, all_pred, average="macro"),
        "y_true": all_y,
        "y_pred": all_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="efficientnet_b0")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(feature_cols) < 20:
        raise ValueError("Not enough BSE-derived feature columns found in manifest. Re-run 01_prepare_4class_manifest.py.")
    print(f"Using {len(feature_cols)} tabular BSE-derived features")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    scaler = StandardScaler()
    train_x = train_df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values.astype("float32")
    scaler.fit(train_x)
    np.savez(out_dir / "tabular_scaler.npz", mean=scaler.mean_, scale=scaler.scale_, feature_cols=np.array(feature_cols))

    train_tfms, eval_tfms = make_transforms(args.img_size)
    train_loader = DataLoader(HybridDataset(train_df, feature_cols, scaler, train_tfms), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(HybridDataset(val_df, feature_cols, scaler, eval_tfms), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(HybridDataset(test_df, feature_cols, scaler, eval_tfms), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    model = HybridModel(args.model, n_tab=len(feature_cols), n_classes=len(LABELS), pretrained=True)
    model.to(device)

    if args.use_class_weights:
        weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(LABELS)), y=train_df["label_id"].values)
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Class weights:", weights.detach().cpu().numpy())
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, total = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for img, tab, y in pbar:
            img, tab, y = img.to(device), tab.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(img, tab)
                loss = criterion(logits, y)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            running_loss += loss.item() * len(y)
            total += len(y)
            pbar.set_postfix(loss=running_loss / max(total, 1))
        scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(total, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
        }
        history.append(row)
        print(row)

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "labels": LABELS,
                "img_size": args.img_size,
                "feature_cols": feature_cols,
                "tabular_mean": scaler.mean_,
                "tabular_scale": scaler.scale_,
                "args": vars(args),
            }, out_dir / "best_model.pt")
            print("Saved best hybrid model")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics = evaluate(model, test_loader, device)

    report = classification_report(test_metrics["y_true"], test_metrics["y_pred"], target_names=LABELS, digits=4)
    cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=np.arange(len(LABELS)))
    metrics = {
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_loss": float(test_metrics["loss"]),
        "model": args.model,
        "manifest": args.manifest,
        "num_tabular_features": len(feature_cols),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)
    print("\nTEST METRICS")
    print(json.dumps(metrics, indent=2))
    print(report)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(values_format="d", xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
