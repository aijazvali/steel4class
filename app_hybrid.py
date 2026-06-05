
import warnings
from importlib.machinery import SourceFileLoader

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torchvision import transforms


warnings.filterwarnings("ignore")

CKPT_PATH = "runs/hybrid_effb0_all_weighted/best_model.pt"

feat_mod = SourceFileLoader("auto_features", "04_auto_extract_features.py").load_module()
train_mod = SourceFileLoader("hybrid_train", "03_train_4class_hybrid.py").load_module()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(
    CKPT_PATH,
    map_location=device,
    weights_only=False
)

labels = ckpt["labels"]
model_name = ckpt["model_name"]
img_size = ckpt["img_size"]
feature_cols = ckpt["feature_cols"]
tabular_mean = np.array(ckpt["tabular_mean"], dtype=np.float32)
tabular_scale = np.array(ckpt["tabular_scale"], dtype=np.float32)

model = train_mod.HybridModel(
    model_name,
    n_tab=len(feature_cols),
    n_classes=len(labels),
    pretrained=False
)

model.load_state_dict(ckpt["state_dict"])
model.to(device)
model.eval()

img_tfm = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


MAIN_FEATURES = [
    "Intensity",
    "Davg",
    "Dcirc",
    "Dmax",
    "Dmin",
    "Dperp",
    "Aspect",
    "Area",
    "Perim",
    "Roundness",
    "Elongation",
    "Pixel.Size...m.per.pixel.",
]


def make_feature_table(feats):
    rows = []

    for c in MAIN_FEATURES:
        rows.append({
            "Feature": c,
            "Value": round(float(feats.get(c, 0.0)), 6),
            "Group": "Morphology / intensity"
        })

    for i in range(256):
        c = f"g{i}"
        rows.append({
            "Feature": c,
            "Value": int(feats.get(c, 0)),
            "Group": "Grayscale histogram"
        })

    return pd.DataFrame(rows)


def predict(image_path):
    if image_path is None:
        empty_df = pd.DataFrame(columns=["Feature", "Value", "Group"])
        return {}, "Please upload a SEM/BSE image.", empty_df, None

    img = Image.open(image_path).convert("L")
    x_img = img_tfm(img).unsqueeze(0).to(device)

    feats, mask = feat_mod.extract_features(image_path)

    x_tab = np.array(
        [float(feats.get(c, 0.0)) for c in feature_cols],
        dtype=np.float32
    )

    x_tab = np.nan_to_num(x_tab, nan=0.0, posinf=0.0, neginf=0.0)
    x_tab = (x_tab - tabular_mean) / tabular_scale
    x_tab = torch.tensor(x_tab, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x_img, x_tab)
        probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()

    result = {labels[i]: float(probs[i]) for i in range(len(labels))}
    top_idx = int(np.argmax(probs))

    details = (
        f"Final prediction: {labels[top_idx]}\n"
        f"Confidence: {probs[top_idx]:.4f}\n\n"
        f"Model: Hybrid EfficientNet-B0\n"
        f"Input used: image + automatically extracted 268 BSE features\n\n"
        f"Features calculated:\n"
        f"- 12 morphology/intensity features\n"
        f"- 256 grayscale histogram features g0 to g255"
    )

    feature_table = make_feature_table(feats)

    # Convert mask to RGB for Gradio display
    mask_rgb = cv2.cvtColor(mask.astype("uint8"), cv2.COLOR_GRAY2RGB)

    return result, details, feature_table, mask_rgb


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload SEM/BSE particle image"),
    outputs=[
        gr.Label(num_top_classes=4, label="Prediction probabilities"),
        gr.Textbox(label="Details"),
        gr.Dataframe(
            headers=["Feature", "Value", "Group"],
            label="Automatically calculated features",
            interactive=False
        ),
        gr.Image(label="Auto-detected particle mask")
    ],
    title="Steel Inclusion Classification - Hybrid Model",
    description=(
        "Upload a SEM/BSE particle image. "
        "The app automatically extracts morphology features and grayscale histogram features, "
        "then predicts one of: non-inclusion, oxide, sulfide, oxy-sulfide."
    ),
    examples=[
        ["data/raw/heat_4/imgs/03999.TIF"]
    ]
)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
