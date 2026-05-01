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
import re

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
# Med-VQA-2019
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_medvqa_split(split_name: str) -> str | None:
    value = split_name.strip().lower()
    if value in {"train", "training"}:
        return "train"
    if value in {"validation", "valid", "val", "dev"}:
        return "validation"
    if value in {"test", "testing"}:
        return "test"
    return None


def _pick_existing_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_to_original = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_to_original:
            return lower_to_original[c]
    raise ValueError(f"None of expected columns found: {candidates}. Available: {list(df.columns)}")


def _safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("._")
    return cleaned or "image"


def download_medvqa2019():
    print("=" * 60)
    print("Downloading Med-VQA-2019...")
    print("=" * 60)

    ds = load_dataset("claudioreeves/imageclef-vqa-med-2019")

    root = DATA_ROOT / "Med-VQA-2019"
    img_dir = root / "medvqa2019_images"
    ann_dir = root / "medvqa2019_annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    csv_targets = {
        "train": ann_dir / "medvqa2019_train.csv",
        "validation": ann_dir / "medvqa2019_validation.csv",
        "test": ann_dir / "medvqa2019_test.csv",
    }
    rows_by_split = {k: [] for k in csv_targets.keys()}

    all_rows = []
    for split_name in ds.keys():
        split_data = ds[split_name]
        split_df = split_data.to_pandas()
        split_df["_source_split"] = split_name
        all_rows.append(split_df)

    if not all_rows:
        raise ValueError("HF dataset contains no splits.")

    df = pd.concat(all_rows, ignore_index=True)
    image_col = _pick_existing_column(df, ["image", "img", "image_path", "img_path"])
    question_col = _pick_existing_column(df, ["question", "query"])
    answer_col = _pick_existing_column(df, ["answer", "answers", "label"])

    # Optional explicit split column in examples.
    split_col = None
    for candidate in ["split", "subset", "set", "fold"]:
        if candidate in {c.lower() for c in df.columns}:
            split_col = next(c for c in df.columns if c.lower() == candidate)
            break

    print(f"Using columns -> image:{image_col}, question:{question_col}, answer:{answer_col}, split:{split_col or '_source_split'}")

    image_counter = 0
    for idx, row in df.iterrows():
        raw_split = row[split_col] if split_col is not None else row["_source_split"]
        target_split = _normalize_medvqa_split(str(raw_split))
        if target_split is None:
            continue

        image_value = row[image_col]
        question = str(row[question_col])
        answer = row[answer_col]
        if isinstance(answer, (list, tuple)):
            answer = answer[0] if answer else ""
        answer = str(answer)

        image = image_value
        if image is None:
            continue

        if hasattr(image, "convert"):
            stem = _safe_filename(f"{target_split}_{idx}")
            img_filename = f"{stem}.jpg"
            img_path = img_dir / img_filename
            if not img_path.exists():
                image.convert("RGB").save(img_path)
        else:
            src_path = Path(str(image))
            if not src_path.exists():
                # Skip rows where image path is not directly accessible.
                continue
            stem = _safe_filename(src_path.stem)
            img_filename = f"{target_split}_{stem}_{idx}{src_path.suffix or '.jpg'}"
            img_path = img_dir / img_filename
            if not img_path.exists():
                shutil.copy2(src_path, img_path)

        rows_by_split[target_split].append(
            {
                "image_path": img_filename,
                "question": question,
                "answer": answer,
            }
        )
        image_counter += 1

    for split_name, csv_path in csv_targets.items():
        split_rows = rows_by_split[split_name]
        split_df = pd.DataFrame(split_rows, columns=["image_path", "question", "answer"])
        split_df.to_csv(csv_path, index=False)
        print(f"  Saved {len(split_df)} samples -> {csv_path.name}")

    print(f"\nMed-VQA-2019 done! {image_counter} image files processed in {img_dir}.")


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    download_pathvqa()
    print()
    download_slake()
    print()
    download_vqarad()
    print()
    download_medvqa2019()
    print("\nAll datasets downloaded under", DATA_ROOT.resolve())