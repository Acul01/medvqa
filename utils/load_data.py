"""
Unified Data Loader for Medical VQA Datasets.

By default this loader reads the multiple-choice exports produced by
build_mc_datasets_openai.py (under ``02_data/data_mc_openai`` in the default
layout). Set mc_data_root=None to use legacy free-text annotations from
data_root.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from utils.mc_vqa import correct_index_to_letter


def combined_dataset_instruction(dataset_display_name: str) -> str:
    """Legacy free-text instruction text for joint training."""
    return (
        f"You are a medical expert. You answer a question from the {dataset_display_name} dataset. "
        "Answer the question about the medical image concisely.\n"
        "Keep the answer as short as possible. If it is a yes/no question, answer only 'yes' or 'no'."
    )


@dataclass
class VQASample:
    """Unified data structure for VQA samples."""

    dataset: str
    split: str
    image_path: str
    question: str
    answer: str
    metadata: Optional[Dict] = None
    mc_options: Optional[List[str]] = None
    mc_correct_index: Optional[int] = None
    question_type: Optional[str] = None


class MedicalVQALoader:
    """Unified loader for Medical VQA datasets."""

    def __init__(
        self,
        data_root: str = "02_data",
        mc_data_root: Optional[str] = "02_data/data_mc_openai",
    ):
        self.data_root = Path(data_root)
        self.mc_data_root = Path(mc_data_root) if mc_data_root else None

        self.dataset_configs = {
            "PathVQA": {
                "annotations_dir": "pathvqa_annotations",
                "images_dir": "pathvqa_images",
                "annotation_files": {
                    "train": "pathvqa_train.csv",
                    "validation": "pathvqa_validation.csv",
                    "test": "pathvqa_test.csv",
                },
                "format": "csv",
            },
            "SLAKE": {
                "annotations_dir": "slake_annotations",
                "images_dir": "slake_images",
                "annotation_files": {
                    "train": "train.json",
                    "validation": "validate.json",
                    "test": "test.json",
                },
                "format": "json",
            },
            "VQA-Rad": {
                "annotations_dir": "vqarad_annotations",
                "images_dir": "vqarad_images",
                "annotation_files": {
                    "train": "vqarad_train.csv",
                    "test": "vqarad_test.csv",
                },
                "format": "csv",
            },
            "Med-VQA-2019": {
                "annotations_dir": "medvqa2019_annotations",
                "images_dir": "medvqa2019_images",
                "annotation_files": {
                    "train": "medvqa2019_train.csv",
                    "validation": "medvqa2019_validation.csv",
                    "test": "medvqa2019_test.csv",
                },
                "format": "csv",
            },
        }

        self.dataset_name_mapping = {
            "pathvqa": "PathVQA",
            "slake": "SLAKE",
            "vqarad": "VQA-Rad",
            "medvqa2019": "Med-VQA-2019",
        }
        self.canonical_to_slug = {v: k for k, v in self.dataset_name_mapping.items()}

    def _normalize_dataset_name(self, dataset_name: str) -> str:
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower in self.dataset_name_mapping:
            return self.dataset_name_mapping[dataset_name_lower]
        if dataset_name in self.dataset_configs:
            return dataset_name
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {list(self.dataset_name_mapping.keys())}"
        )

    def _resolve_image_path(self, dataset_name: str, config: Dict, image_path: str) -> Path:
        dataset_root = self.data_root / dataset_name
        image_path_str = str(image_path).strip()
        if not image_path_str:
            raise ValueError("empty image_path")
        if os.path.isabs(image_path_str):
            return Path(image_path_str)
        # MC records may store a repo-relative path (e.g. data/VQA-Rad/vqarad_images/train_0.jpg).
        # Do not prefix dataset_root/images_dir again in that case.
        rel = Path(image_path_str)
        if rel.exists():
            return rel.resolve()
        if image_path_str.startswith(config["images_dir"]):
            return dataset_root / image_path_str
        return dataset_root / config["images_dir"] / image_path_str

    def _mc_paths(self, dataset_name: str, split: str) -> tuple[Path, Path]:
        if self.mc_data_root is None:
            raise ValueError("mc_data_root is disabled")
        slug = self.canonical_to_slug[dataset_name]
        base = self.mc_data_root / slug / f"{slug}_{split}_mc_openai"
        return base.with_suffix(".json"), base.with_suffix(".csv")

    def _load_mc_records(self, dataset_name: str, split: str, config: Dict) -> List[VQASample]:
        mc_json_path, mc_csv_path = self._mc_paths(dataset_name, split)

        records = None
        if mc_json_path.exists():
            with open(mc_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, list):
                raise ValueError(f"MC JSON must contain a list: {mc_json_path}")
            records = payload
        elif mc_csv_path.exists():
            with open(mc_csv_path, "r", encoding="utf-8") as f:
                records = list(csv.DictReader(f))

        if records is None:
            raise FileNotFoundError(
                "Missing MC annotations. Expected one of: "
                f"{mc_json_path} or {mc_csv_path}. "
                "Build MC data first with utils/build_mc_datasets_openai.py."
            )

        samples: List[VQASample] = []
        for row in records:
            options = row.get("options")
            if isinstance(options, str):
                try:
                    options = json.loads(options)
                except json.JSONDecodeError:
                    options = None
            if not isinstance(options, list) or len(options) < 2:
                continue

            raw_correct_index = row.get("correct_index")
            try:
                correct_index = int(raw_correct_index)
            except (TypeError, ValueError):
                continue
            if correct_index < 0 or correct_index >= len(options):
                continue

            question = str(row.get("question", "")).strip()
            if not question:
                continue

            image_path = self._resolve_image_path(
                dataset_name=dataset_name,
                config=config,
                image_path=str(row.get("image_path", "")),
            )

            metadata = {
                "gt_answer_text": row.get("answer", ""),
            }
            for key, value in row.items():
                if key not in {"dataset", "split", "image_path", "question", "answer", "options", "correct_index", "question_type"}:
                    metadata[key] = value

            samples.append(
                VQASample(
                    dataset=dataset_name,
                    split=split,
                    image_path=str(image_path),
                    question=question,
                    answer=correct_index_to_letter(correct_index),
                    metadata=metadata,
                    mc_options=[str(opt) for opt in options],
                    mc_correct_index=correct_index,
                    question_type=str(row.get("question_type", "")).strip() or None,
                )
            )
        return samples

    def load_dataset(self, dataset_name: str, split: str) -> List[VQASample]:
        dataset_name = self._normalize_dataset_name(dataset_name)
        if dataset_name not in self.dataset_configs:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported datasets: {list(self.dataset_configs.keys())}"
            )

        config = self.dataset_configs[dataset_name]
        if split not in config["annotation_files"]:
            available_splits = list(config["annotation_files"].keys())
            raise ValueError(
                f"Split '{split}' not available for {dataset_name}. "
                f"Available splits: {available_splits}"
            )

        if self.mc_data_root is not None:
            return self._load_mc_records(dataset_name, split, config)

        annotation_file = config["annotation_files"][split]
        annotation_path = self.data_root / dataset_name / config["annotations_dir"] / annotation_file
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        if config["format"] == "csv":
            return self._load_csv(dataset_name, annotation_path, split, config)
        if config["format"] == "json":
            return self._load_json(dataset_name, annotation_path, split, config)
        raise ValueError(f"Unknown format: {config['format']}")

    def load_combined(self, split: str, vqarad_split: Optional[str] = None) -> List[VQASample]:
        combined: List[VQASample] = []
        for slug in ("pathvqa", "slake", "vqarad", "medvqa2019"):
            use_split = split if slug != "vqarad" or vqarad_split is None else vqarad_split
            combined.extend(self.load_dataset(slug, use_split))
        return combined

    def _load_csv(self, dataset_name: str, annotation_path: Path, split: str, config: Dict) -> List[VQASample]:
        df = pd.read_csv(annotation_path)
        samples = []

        for _, row in df.iterrows():
            if "image_path" not in df.columns:
                raise ValueError(f"CSV file missing 'image_path' column: {annotation_path}")

            full_image_path = self._resolve_image_path(dataset_name, config, row["image_path"])
            question = row.get("question", "")
            answer = row.get("answer", "")
            metadata = {k: v for k, v in row.items() if k not in ["image_path", "question", "answer", "split"]}

            samples.append(
                VQASample(
                    dataset=dataset_name,
                    split=split,
                    image_path=str(full_image_path),
                    question=question,
                    answer=answer,
                    metadata=metadata if metadata else None,
                )
            )
        return samples

    def _load_json(self, dataset_name: str, annotation_path: Path, split: str, config: Dict) -> List[VQASample]:
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        dataset_root = self.data_root / dataset_name
        for item in data:
            if item.get("q_lang") == "zh":
                continue

            img_name = item.get("img_name", "")
            if img_name.startswith(config["images_dir"]):
                full_image_path = dataset_root / img_name
            else:
                full_image_path = dataset_root / config["images_dir"] / img_name

            question = item.get("question", "")
            answer = item.get("answer", "")
            metadata = {k: v for k, v in item.items() if k not in ["question", "answer"]}

            samples.append(
                VQASample(
                    dataset=dataset_name,
                    split=split,
                    image_path=str(full_image_path),
                    question=question,
                    answer=answer,
                    metadata=metadata if metadata else None,
                )
            )
        return samples

    def get_available_datasets(self) -> List[str]:
        return list(self.dataset_configs.keys())

    def get_available_splits(self, dataset_name: str) -> List[str]:
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return list(self.dataset_configs[dataset_name]["annotation_files"].keys())

    def load_all_splits(self, dataset_name: str) -> Dict[str, List[VQASample]]:
        splits = self.get_available_splits(dataset_name)
        return {split: self.load_dataset(dataset_name, split) for split in splits}


# Convenience functions for quick access

def load_pathvqa(data_root: str = "02_data", split: str = "train") -> List[VQASample]:
    loader = MedicalVQALoader(data_root)
    return loader.load_dataset("PathVQA", split)


def load_slake(data_root: str = "02_data", split: str = "train") -> List[VQASample]:
    loader = MedicalVQALoader(data_root)
    return loader.load_dataset("SLAKE", split)


def load_vqarad(data_root: str = "02_data", split: str = "train") -> List[VQASample]:
    loader = MedicalVQALoader(data_root)
    return loader.load_dataset("VQA-Rad", split)


def load_medvqa2019(data_root: str = "02_data", split: str = "train") -> List[VQASample]:
    loader = MedicalVQALoader(data_root)
    return loader.load_dataset("Med-VQA-2019", split)
