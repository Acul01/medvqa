"""
Download PathVQA, SLAKE, and VQA-Rad from HuggingFace into ./data/
(layout matches utils/load_data.MedicalVQALoader).

Usage on cluster:
    conda activate llava_env
    pip install datasets huggingface_hub pandas
    python download_datasets.py
"""

import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import zipfile
import pandas as pd
import shutil

DATA_ROOT = Path("data")


# ═══════════════════════════════════════════════════════════════════════════════
# PathVQA
# ═══════════════════════════════════════════════════════════════════════════════

def download_pathvqa():
    print("=" * 60)
    print("Downloading PathVQA...")
    print("=" * 60)

    ds = load_dataset("flaviagiammarino/path-vqa")

    img_dir = DATA_ROOT / "PathVQA" / "pathvqa_images"
    ann_dir = DATA_ROOT / "PathVQA" / "pathvqa_annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    split_mapping = {
        "train": "pathvqa_train.csv",
        "validation": "pathvqa_validation.csv",
        "test": "pathvqa_test.csv",
    }

    for split_name, csv_filename in split_mapping.items():
        print(f"\nProcessing {split_name} split...")
        split_data = ds[split_name]
        rows = []

        for idx, sample in enumerate(split_data):
            image = sample["image"]
            question = sample["question"]
            answer = sample["answer"]

            # Save image as: pathvqa_images/{split}_{idx}.jpg
            img_filename = f"{split_name}_{idx}.jpg"
            img_path = img_dir / img_filename
            if not img_path.exists():
                image.convert("RGB").save(img_path)

            rows.append({
                "image_path": img_filename,   # relative to pathvqa_images/
                "question": question,
                "answer": answer,
            })

        df = pd.DataFrame(rows)
        df.to_csv(ann_dir / csv_filename, index=False)
        print(f"  Saved {len(df)} samples -> {csv_filename}")

    print(f"\nPathVQA done! {len(list(img_dir.iterdir()))} images saved.")


# ═══════════════════════════════════════════════════════════════════════════════
# SLAKE
# ═══════════════════════════════════════════════════════════════════════════════

def download_slake():
    print("=" * 60)
    print("Downloading SLAKE...")
    print("=" * 60)

    slake_dir = DATA_ROOT / "SLAKE"
    img_dir = slake_dir / "slake_images"
    ann_dir = slake_dir / "slake_annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # Download imgs.zip and extract
    print("Downloading imgs.zip...")
    imgs_zip_path = hf_hub_download(
        repo_id="BoKelvin/SLAKE",
        filename="imgs.zip",
        repo_type="dataset",
    )
    print("Extracting images...")
    with zipfile.ZipFile(imgs_zip_path, "r") as zf:
        zf.extractall(slake_dir / "_tmp_imgs")

    # Move extracted images into slake_images/
    # imgs.zip typically extracts to an "imgs/" subdirectory
    extracted = slake_dir / "_tmp_imgs"
    src = extracted / "imgs" if (extracted / "imgs").exists() else extracted
    for item in src.iterdir():
        dest = img_dir / item.name
        if not dest.exists():
            shutil.move(str(item), str(dest))
    shutil.rmtree(extracted, ignore_errors=True)
    print(f"  Images extracted to {img_dir}")

    # Download JSON annotation files
    # HuggingFace repo has: train.json, validation.json, test.json
    # But load_data.py expects: train.json, validate.json, test.json
    json_files = {
        "train.json": "train.json",           # HF name -> local name
        "test.json": "test.json",
        "validation.json": "validate.json",    # IMPORTANT: rename!
    }

    for hf_name, local_name in json_files.items():
        print(f"Downloading {hf_name} -> {local_name}")
        downloaded = hf_hub_download(
            repo_id="BoKelvin/SLAKE",
            filename=hf_name,
            repo_type="dataset",
        )
        dest = ann_dir / local_name
        shutil.copy2(downloaded, dest)

    # Verify
    for f in ann_dir.iterdir():
        with open(f) as fh:
            data = json.load(fh)
        print(f"  {f.name}: {len(data)} entries")

    print(f"\nSLAKE done!")


# ═══════════════════════════════════════════════════════════════════════════════
# VQA-Rad
# ═══════════════════════════════════════════════════════════════════════════════

def download_vqarad():
    print("=" * 60)
    print("Downloading VQA-Rad...")
    print("=" * 60)

    ds = load_dataset("flaviagiammarino/vqa-rad")

    root = DATA_ROOT / "VQA-Rad"
    img_dir = root / "vqarad_images"
    ann_dir = root / "vqarad_annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    split_mapping = {
        "train": "vqarad_train.csv",
        "test": "vqarad_test.csv",
    }

    for split_name, csv_filename in split_mapping.items():
        print(f"\nProcessing {split_name} split...")
        split_data = ds[split_name]
        rows = []

        for idx, sample in enumerate(split_data):
            image = sample["image"]
            question = sample["question"]
            answer = sample["answer"]

            img_filename = f"{split_name}_{idx}.jpg"
            img_path = img_dir / img_filename
            if not img_path.exists():
                image.convert("RGB").save(img_path)

            rows.append({
                "image_path": img_filename,
                "question": question,
                "answer": answer,
            })

        df = pd.DataFrame(rows)
        df.to_csv(ann_dir / csv_filename, index=False)
        print(f"  Saved {len(df)} samples -> {csv_filename}")

    print(f"\nVQA-Rad done! {len(list(img_dir.iterdir()))} image files in {img_dir}.")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    download_pathvqa()
    print()
    download_slake()
    print()
    download_vqarad()
    print("\nAll datasets downloaded under", DATA_ROOT.resolve())