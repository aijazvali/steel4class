
import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


PIXEL_SIZE = 0.092773


FEATURE_COLS = [
    "Intensity", "Davg", "Dcirc", "Dmax", "Dmin", "Dperp",
    "Aspect", "Area", "Perim", "Roundness", "Elongation",
    "Pixel.Size...m.per.pixel."
] + [f"g{i}" for i in range(256)]


def read_gray(path):
    # Use PIL for loading TIFF images to avoid OpenCV TIFF metadata warnings.
    img = Image.open(path).convert("L")
    img = np.array(img)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    return img


def get_histogram(img):
    hist = np.bincount(img.ravel(), minlength=256).astype(float)
    return {f"g{i}": hist[i] for i in range(256)}


def choose_particle_mask(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)

    _, th = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    candidates = [th, 255 - th]

    best_mask = None
    best_score = -1
    h, w = img.shape
    center = np.array([w / 2, h / 2])
    total_area = h * w

    for cand in candidates:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cand.astype("uint8"), connectivity=8
        )

        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]

            if area < 20:
                continue

            # Reject huge regions because they are usually background, not particle.
            # In our 128x128 particle crops, the actual inclusion/non-inclusion
            # particle is normally much smaller than the full image.
            if area > 0.60 * total_area:
                continue

            cx, cy = centroids[lab]
            dist = np.linalg.norm(np.array([cx, cy]) - center)

            # prefer large central particle-like region
            score = area - 3.0 * dist

            if score > best_score:
                best_score = score
                best_mask = (labels == lab).astype("uint8") * 255

    if best_mask is None:
        # fallback: entire non-zero image
        best_mask = np.ones_like(img, dtype="uint8") * 255

    kernel = np.ones((3, 3), np.uint8)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
    best_mask = cv2.morphologyEx(best_mask, cv2.MORPH_OPEN, kernel)

    return best_mask


def contour_features(img, mask, pixel_size=PIXEL_SIZE):
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No contour found")

    contour = max(contours, key=cv2.contourArea)

    area_px = float(cv2.contourArea(contour))
    perim_px = float(cv2.arcLength(contour, True))

    if area_px <= 0:
        area_px = float(np.count_nonzero(mask))

    area_um = area_px * pixel_size * pixel_size
    perim_um = perim_px * pixel_size

    rect = cv2.minAreaRect(contour)
    (_, _), (rw, rh), _ = rect

    dmax_px = max(rw, rh)
    dmin_px = min(rw, rh)

    if dmax_px <= 0 or dmin_px <= 0:
        x, y, w, h = cv2.boundingRect(contour)
        dmax_px = max(w, h)
        dmin_px = min(w, h)

    dmax = dmax_px * pixel_size
    dmin = dmin_px * pixel_size
    dperp = dmin

    dcirc = math.sqrt(4.0 * area_um / math.pi) if area_um > 0 else 0.0
    davg = (dmax + dmin) / 2.0 if (dmax + dmin) > 0 else 0.0
    aspect = dmax / dmin if dmin > 0 else 0.0

    # matches the dataset style better than 4*pi*A/P^2
    roundness = 4.0 * area_um / (math.pi * dmax * dmax) if dmax > 0 else 0.0
    elongation = dperp / dmax if dmax > 0 else 0.0

    intensity = float(img[mask > 0].mean()) if np.any(mask > 0) else float(img.mean())

    return {
        "Intensity": intensity,
        "Davg": davg,
        "Dcirc": dcirc,
        "Dmax": dmax,
        "Dmin": dmin,
        "Dperp": dperp,
        "Aspect": aspect,
        "Area": area_um,
        "Perim": perim_um,
        "Roundness": roundness,
        "Elongation": elongation,
        "Pixel.Size...m.per.pixel.": pixel_size,
    }


def extract_features(image_path):
    img = read_gray(image_path)
    mask = choose_particle_mask(img)

    feats = {}
    feats.update(contour_features(img, mask))
    feats.update(get_histogram(img))

    return feats, mask


def compare_with_manifest(manifest, row_idx=0):
    df = pd.read_csv(manifest)
    row = df.iloc[row_idx]
    image_path = row["image_path"]

    auto, mask = extract_features(image_path)

    compare_cols = [
        "Intensity", "Davg", "Dcirc", "Dmax", "Dmin", "Dperp",
        "Aspect", "Area", "Perim", "Roundness", "Elongation",
        "Pixel.Size...m.per.pixel."
    ]

    print("Image:", image_path)
    print("Label:", row["label"])
    print()

    print("Feature comparison:")
    for c in compare_cols:
        original = float(row[c])
        predicted = float(auto[c])
        diff = predicted - original
        print(f"{c:30s} original={original:10.4f} auto={predicted:10.4f} diff={diff:10.4f}")

    hist_ok = True
    for i in range(256):
        if float(row[f"g{i}"]) != float(auto[f"g{i}"]):
            hist_ok = False
            break

    print()
    print("Histogram exact match:", hist_ok)

    out_mask = Path("auto_feature_debug_mask.png")
    cv2.imwrite(str(out_mask), mask)
    print("Saved mask:", out_mask)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest_4class_all.csv")
    parser.add_argument("--row_idx", type=int, default=0)
    args = parser.parse_args()

    compare_with_manifest(args.manifest, args.row_idx)
