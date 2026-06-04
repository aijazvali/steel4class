import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["non-inclusion", "oxide", "sulfide", "oxy-sulfide"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}

BSE_FEATURES = [
    "Intensity", "Davg", "Dcirc", "Dmax", "Dmin", "Dperp", "Aspect",
    "Area", "Perim", "Roundness", "Elongation", "Pixel.Size...m.per.pixel."
] + [f"g{i}" for i in range(256)]


def norm_text(x):
    return str(x).strip().replace("_", "-").replace("–", "-").replace("—", "-").lower()


def find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    raise KeyError(f"Could not find any of these columns: {candidates}")


def add_split_column(df, seed):
    train_df, temp_df = train_test_split(
        df, test_size=0.40, random_state=seed, stratify=df["label_id"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df["label_id"]
    )
    train_df = train_df.copy(); train_df["split"] = "train"
    val_df = val_df.copy(); val_df["split"] = "val"
    test_df = test_df.copy(); test_df["split"] = "test"
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def zip_for_heat(raw_dir: Path, heat: int):
    candidates = [
        raw_dir / f"Heat {heat} images.zip",
        raw_dir / f"Heat_{heat}_images.zip",
        raw_dir / f"heat{heat}.zip",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(raw_dir.glob(f"*Heat*{heat}*images*.zip")) + list(raw_dir.glob(f"*heat*{heat}*images*.zip"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find zip for Heat {heat} in {raw_dir}")


def build_image_paths(df, raw_dir: Path, out_dir: Path, heat_col: str, part_col: str):
    image_root = out_dir / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    out_paths = []
    missing = []

    zip_cache = {}
    zip_names_cache = {}

    for _, row in df.iterrows():
        heat = int(row[heat_col])
        part = int(row[part_col])
        internal_name = f"imgs/{part:05d}.TIF"
        local_path = image_root / f"heat{heat}" / f"{part:05d}.TIF"

        if not local_path.exists():
            zpath = zip_cache.get(heat)
            if zpath is None:
                zpath = zip_for_heat(raw_dir, heat)
                zip_cache[heat] = zpath
                with zipfile.ZipFile(zpath) as zf:
                    zip_names_cache[heat] = set(zf.namelist())

            names = zip_names_cache[heat]
            chosen = None
            for candidate in [internal_name, internal_name.lower(), internal_name.replace(".TIF", ".tif")]:
                if candidate in names:
                    chosen = candidate
                    break

            if chosen is None:
                missing.append((heat, part, internal_name))
                out_paths.append(None)
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath) as zf:
                with zf.open(chosen) as src, open(local_path, "wb") as dst:
                    dst.write(src.read())

        out_paths.append(str(local_path))

    if missing:
        print(f"WARNING: {len(missing)} images were missing. First 10: {missing[:10]}")
    return out_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="Folder containing xlsx and Heat image zip files")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder for extracted images and manifests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xlsx", type=str, default="2018_AlCaSMn_20kV_4.xlsx")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = raw_dir / args.xlsx
    if not xlsx_path.exists():
        matches = list(raw_dir.glob("*.xlsx"))
        if not matches:
            raise FileNotFoundError(f"No xlsx found in {raw_dir}")
        xlsx_path = matches[0]
        print(f"Using xlsx: {xlsx_path}")

    print("Reading Excel file...")
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    heat_col = find_col(df, ["Heat", "HEAT"])
    part_col = find_col(df, ["Part", "particle", "Particle"])
    type_col = find_col(df, ["type", "Type"])

    df["label"] = df[type_col].apply(norm_text)
    df = df[df["label"].isin(LABELS)].copy()
    df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)

    # Keep columns useful for hybrid training, but never include EDS chemistry features.
    keep_cols = [heat_col, part_col, "label", "label_id"]
    for c in ["SN", "Fld"] + BSE_FEATURES:
        if c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    df = df[keep_cols].copy()
    df = df.rename(columns={heat_col: "Heat", part_col: "Part"})

    print("Extracting / linking images...")
    df["image_path"] = build_image_paths(df, raw_dir, out_dir, "Heat", "Part")
    df = df[df["image_path"].notna()].copy()

    all_df = add_split_column(df, args.seed)
    all_path = out_dir / "manifest_4class_all.csv"
    all_df.to_csv(all_path, index=False)

    min_count = all_df["label"].value_counts().min()
    balanced_df = (
        all_df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min_count, random_state=args.seed))
        .reset_index(drop=True)
    )
    balanced_df = add_split_column(balanced_df.drop(columns=["split"]), args.seed)
    balanced_path = out_dir / "manifest_4class_balanced.csv"
    balanced_df.to_csv(balanced_path, index=False)

    print("\nClass counts, all:")
    print(all_df["label"].value_counts())
    print("\nClass counts, balanced:")
    print(balanced_df["label"].value_counts())
    print("\nSplit counts, balanced:")
    print(pd.crosstab(balanced_df["label"], balanced_df["split"]))
    print(f"\nSaved: {all_path}")
    print(f"Saved: {balanced_path}")


if __name__ == "__main__":
    main()
