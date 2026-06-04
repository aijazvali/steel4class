import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
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


class BSEDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = int(row["label_id"])
        return img, y


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
    all_y, all_pred, all_prob = [], [], []
    loss_sum, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            prob = torch.softmax(logits, dim=1)
            pred = prob.argmax(dim=1)
            loss_sum += loss.item() * len(y)
            n += len(y)
            all_y.extend(y.cpu().numpy())
            all_pred.extend(pred.cpu().numpy())
            all_prob.extend(prob.cpu().numpy())
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
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_tfms, eval_tfms = make_transforms(args.img_size)
    train_loader = DataLoader(BSEDataset(train_df, train_tfms), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(BSEDataset(val_df, eval_tfms), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(BSEDataset(test_df, eval_tfms), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    model = timm.create_model(args.model, pretrained=True, num_classes=len(LABELS))
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
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, total = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
                "args": vars(args),
            }, out_dir / "best_model.pt")
            print("Saved best model")

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
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
