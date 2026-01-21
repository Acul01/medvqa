"""
Unified Data Loader for Medical VQA Datasets

Supports:
- PathVQA
- SLAKE
- VQA-Rad
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class VQASample:
    """Unified data structure for VQA samples"""
    dataset: str
    split: str
    image_path: str
    question: str
    answer: str
    # Additional metadata (dataset-specific)
    metadata: Optional[Dict] = None


class MedicalVQALoader:
    """
    Unified loader for Medical VQA datasets.
    
    Usage:
        loader = MedicalVQALoader(data_root="data")
        pathvqa_train = loader.load_dataset("PathVQA", split="train")
        slake_test = loader.load_dataset("SLAKE", split="test")
    """
    
    def __init__(self, data_root: str = "data"):
        """
        Initialize the loader.
        
        Args:
            data_root: Root directory containing all datasets (default: "data")
        """
        self.data_root = Path(data_root)
        
        # Dataset configurations
        self.dataset_configs = {
            "PathVQA": {
                "annotations_dir": "pathvqa_annotations",
                "images_dir": "pathvqa_images",
                "annotation_files": {
                    "train": "pathvqa_train.csv",
                    "validation": "pathvqa_validation.csv",
                    "test": "pathvqa_test.csv"
                },
                "format": "csv"
            },
            "SLAKE": {
                "annotations_dir": "slake_annotations",
                "images_dir": "slake_images",
                "annotation_files": {
                    "train": "train.json",
                    "validation": "validate.json",  # Note: validate, not validation
                    "test": "test.json"
                },
                "format": "json"
            },
            "VQA-Rad": {
                "annotations_dir": "vqarad_annotations",
                "images_dir": "vqarad_images",
                "annotation_files": {
                    "train": "vqarad_train.csv",
                    "test": "vqarad_test.csv"
                },
                "format": "csv"
            }
        }
        
        # Mapping from lowercase to canonical names
        self.dataset_name_mapping = {
            "pathvqa": "PathVQA",
            "slake": "SLAKE",
            "vqarad": "VQA-Rad"
        }
    
    def _normalize_dataset_name(self, dataset_name: str) -> str:
        """Convert lowercase dataset name to canonical name"""
        dataset_name_lower = dataset_name.lower()
        if dataset_name_lower in self.dataset_name_mapping:
            return self.dataset_name_mapping[dataset_name_lower]
        # If already canonical, return as-is
        if dataset_name in self.dataset_configs:
            return dataset_name
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported datasets: {list(self.dataset_name_mapping.keys())}"
        )
    
    def load_dataset(self, dataset_name: str, split: str) -> List[VQASample]:
        """
        Load a specific dataset split.
        
        Args:
            dataset_name: Name of the dataset ("pathvqa", "slake", "vqarad" or canonical names)
            split: Split to load ("train", "validation", or "test")
            
        Returns:
            List of VQASample objects
            
        Raises:
            ValueError: If dataset_name or split is not supported
            FileNotFoundError: If annotation file is not found
        """
        # Normalize dataset name (convert lowercase to canonical)
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
        
        annotation_file = config["annotation_files"][split]
        annotation_path = self.data_root / dataset_name / config["annotations_dir"] / annotation_file
        
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_path}"
            )
        
        # Load based on format
        if config["format"] == "csv":
            samples = self._load_csv(dataset_name, annotation_path, split, config)
        elif config["format"] == "json":
            samples = self._load_json(dataset_name, annotation_path, split, config)
        else:
            raise ValueError(f"Unknown format: {config['format']}")
        
        return samples
    
    def _load_csv(self, dataset_name: str, annotation_path: Path, split: str, config: Dict) -> List[VQASample]:
        """Load CSV annotation file"""
        df = pd.read_csv(annotation_path)
        
        samples = []
        dataset_root = self.data_root / dataset_name
        
        for _, row in df.iterrows():
            # Handle different CSV structures
            if "image_path" in df.columns:
                image_path = row["image_path"]
            else:
                raise ValueError(f"CSV file missing 'image_path' column: {annotation_path}")
            
            # Resolve image path relative to dataset root
            if not os.path.isabs(image_path):
                # If path starts with images_dir, use it as-is, otherwise prepend images_dir
                if image_path.startswith(config["images_dir"]):
                    full_image_path = dataset_root / image_path
                else:
                    full_image_path = dataset_root / config["images_dir"] / image_path
            else:
                full_image_path = Path(image_path)
            
            # Extract question and answer
            question = row.get("question", "")
            answer = row.get("answer", "")
            
            # Store additional metadata
            metadata = {k: v for k, v in row.items() 
                       if k not in ["image_path", "question", "answer", "split"]}
            
            sample = VQASample(
                dataset=dataset_name,
                split=split,
                image_path=str(full_image_path),
                question=question,
                answer=answer,
                metadata=metadata if metadata else None
            )
            samples.append(sample)
        
        return samples
    
    def _load_json(self, dataset_name: str, annotation_path: Path, split: str, config: Dict) -> List[VQASample]:
        """Load JSON annotation file (for SLAKE)"""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        dataset_root = self.data_root / dataset_name
        
        # SLAKE JSON is a list of objects
        for item in data:
            # Extract image name (e.g., "xmlab102/source.jpg")
            img_name = item.get("img_name", "")
            
            # Resolve image path
            if img_name.startswith(config["images_dir"]):
                full_image_path = dataset_root / img_name
            else:
                full_image_path = dataset_root / config["images_dir"] / img_name
            
            # Extract question and answer
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            # Store all other fields as metadata
            metadata = {k: v for k, v in item.items() 
                       if k not in ["question", "answer"]}
            
            sample = VQASample(
                dataset=dataset_name,
                split=split,
                image_path=str(full_image_path),
                question=question,
                answer=answer,
                metadata=metadata if metadata else None
            )
            samples.append(sample)
        
        return samples
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names"""
        return list(self.dataset_configs.keys())
    
    def get_available_splits(self, dataset_name: str) -> List[str]:
        """
        Get list of available splits for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of available split names
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return list(self.dataset_configs[dataset_name]["annotation_files"].keys())
    
    def load_all_splits(self, dataset_name: str) -> Dict[str, List[VQASample]]:
        """
        Load all available splits for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping split names to lists of VQASample objects
        """
        splits = self.get_available_splits(dataset_name)
        return {split: self.load_dataset(dataset_name, split) for split in splits}


# Convenience functions for quick access
def load_pathvqa(data_root: str = "data", split: str = "train") -> List[VQASample]:
    """Convenience function to load PathVQA dataset"""
    loader = MedicalVQALoader(data_root)
    return loader.load_dataset("PathVQA", split)


def load_slake(data_root: str = "data", split: str = "train") -> List[VQASample]:
    """Convenience function to load SLAKE dataset"""
    loader = MedicalVQALoader(data_root)
    return loader.load_dataset("SLAKE", split)


def load_vqarad(data_root: str = "data", split: str = "train") -> List[VQASample]:
    """Convenience function to load VQA-Rad dataset"""
    loader = MedicalVQALoader(data_root)
    return loader.load_dataset("VQA-Rad", split)
