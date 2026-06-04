import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import gradio as gr

from PIL import Image
from torchvision import transforms
from pathlib import Path


LABELS = ["non-inclusion", "oxide", "sulfide", "oxy-sulfide"]

IMAGE_CKPT_PATH = "runs/effb0_balanced/best_model.pt"
HYBRID_CKPT_PATH = "runs/hybrid_effb0_all_weighted/best_model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class HybridModel(nn.Module):
    def __init__(self, backbone_name, n_tab, n_classes=4):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
        )

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


def load_image_model():
    ckpt = torch.load(IMAGE_CKPT_PATH, map_location=DEVICE, weights_only=False)

    model_name = ckpt.get("model_name", "efficientnet_b0")
    img_size = ckpt.get("img_size", 224)
    labels = ckpt.get("labels", LABELS)

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(labels),
    )

    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, img_size, labels


def load_hybrid_model():
    ckpt = torch.load(HYBRID_CKPT_PATH, map_location=DEVICE, weights_only=False)

    model_name = ckpt.get("model_name", "efficientnet_b0")
    img_size = ckpt.get("img_size", 224)
    labels = ckpt.get("labels", LABELS)

    feature_cols = ckpt["feature_cols"]
    tabular_mean = np.array(ckpt["tabular_mean"], dtype=np.float32)
    tabular_scale = np.array(ckpt["tabular_scale"], dtype=np.float32)

    model = HybridModel(
        backbone_name=model_name,
        n_tab=len(feature_cols),
        n_classes=len(labels),
    )

    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, img_size, labels, feature_cols, tabular_mean, tabular_scale


image_model, image_img_size, image_labels = load_image_model()
hybrid_model, hybrid_img_size, hybrid_labels, hybrid_feature_cols, hybrid_mean, hybrid_scale = load_hybrid_model()

image_tfms = make_eval_transform(image_img_size)
hybrid_tfms = make_eval_transform(hybrid_img_size)


def preprocess_image(img, tfms):
    if img is None:
        raise gr.Error("Please upload an image.")

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img = img.convert("L").convert("RGB")
    x = tfms(img).unsqueeze(0).to(DEVICE)
    return x


def format_prediction(prob, labels):
    prob = prob.detach().cpu().numpy()[0]
    result = {labels[i]: float(prob[i]) for i in range(len(labels))}
    top_idx = int(np.argmax(prob))
    top_text = f"Prediction: {labels[top_idx]} | Confidence: {prob[top_idx] * 100:.2f}%"
    return result, top_text


def predict_image_only(img):
    x = preprocess_image(img, image_tfms)

    with torch.no_grad():
        logits = image_model(x)
        prob = torch.softmax(logits, dim=1)

    return format_prediction(prob, image_labels)


def predict_hybrid(img, csv_file):
    if csv_file is None:
        raise gr.Error("Please upload a CSV containing one feature row.")

    x_img = preprocess_image(img, hybrid_tfms)

    df = pd.read_csv(csv_file)

    if len(df) == 0:
        raise gr.Error("CSV is empty.")

    row = df.iloc[[0]].copy()

    missing = [c for c in hybrid_feature_cols if c not in row.columns]
    if missing:
        raise gr.Error(
            f"CSV is missing {len(missing)} required feature columns. "
            f"First missing columns: {missing[:10]}"
        )

    x_tab = row[hybrid_feature_cols]
    x_tab = x_tab.fillna(0).replace([np.inf, -np.inf], 0)
    x_tab = x_tab.values.astype("float32")

    x_tab = (x_tab - hybrid_mean) / hybrid_scale
    x_tab = torch.tensor(x_tab, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = hybrid_model(x_img, x_tab)
        prob = torch.softmax(logits, dim=1)

    return format_prediction(prob, hybrid_labels)


with gr.Blocks(title="Steel Inclusion Classifier") as demo:
    gr.Markdown("# Steel Inclusion Classifier")
    gr.Markdown(
        "Classes: **non-inclusion**, **oxide**, **sulfide**, **oxy-sulfide**"
    )

    with gr.Tab("Image-only model"):
        gr.Markdown("Use this when you only have the BSE/SEM image.")
        img1 = gr.Image(type="pil", label="Upload BSE image")
        btn1 = gr.Button("Predict using image-only model")
        out1 = gr.Label(num_top_classes=4, label="Class probabilities")
        txt1 = gr.Textbox(label="Top prediction")
        btn1.click(
            fn=predict_image_only,
            inputs=img1,
            outputs=[out1, txt1],
        )

    with gr.Tab("Hybrid model"):
        gr.Markdown(
            "Use this when you have the BSE/SEM image plus a CSV row containing the 268 BSE-derived features."
        )
        img2 = gr.Image(type="pil", label="Upload BSE image")
        csv2 = gr.File(
            label="Upload feature CSV",
            file_types=[".csv"],
            type="filepath",
        )
        btn2 = gr.Button("Predict using hybrid model")
        out2 = gr.Label(num_top_classes=4, label="Class probabilities")
        txt2 = gr.Textbox(label="Top prediction")
        btn2.click(
            fn=predict_hybrid,
            inputs=[img2, csv2],
            outputs=[out2, txt2],
        )

demo.launch(share=True, debug=True)
