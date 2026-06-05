
import argparse
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# Load our existing files
feat_mod = SourceFileLoader("auto_features", "04_auto_extract_features.py").load_module()
train_mod = SourceFileLoader("hybrid_train", "03_train_4class_hybrid.py").load_module()


def load_image_tensor(image_path, img_size):
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(image_path).convert("L")
    x = tfm(img).unsqueeze(0)
    return x


def predict(image_path, ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )

    labels = ckpt["labels"]
    model_name = ckpt["model_name"]
    img_size = ckpt["img_size"]
    feature_cols = ckpt["feature_cols"]
    mean = np.array(ckpt["tabular_mean"], dtype=np.float32)
    scale = np.array(ckpt["tabular_scale"], dtype=np.float32)

    # Load model architecture from training script
    model = train_mod.HybridModel(
        model_name,
        n_tab=len(feature_cols),
        n_classes=len(labels),
        pretrained=False
    )

    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    # Image tensor
    img_tensor = load_image_tensor(image_path, img_size).to(device)

    # Auto features
    feats, _ = feat_mod.extract_features(image_path)
    x_tab = np.array(
        [float(feats.get(c, 0.0)) for c in feature_cols],
        dtype=np.float32
    )

    x_tab = np.nan_to_num(x_tab, nan=0.0, posinf=0.0, neginf=0.0)
    x_tab = (x_tab - mean) / scale
    x_tab = torch.tensor(x_tab, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor, x_tab)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    ranked = sorted(
        zip(labels, probs),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--ckpt",
        default="runs/hybrid_effb0_all_weighted/best_model.pt"
    )
    args = parser.parse_args()

    ranked = predict(args.image, args.ckpt)

    print("Image:", args.image)
    print("\nPrediction ranking:")
    for label, prob in ranked:
        print(f"{label:15s}: {prob:.4f}")

    print("\nFinal prediction:", ranked[0][0])
