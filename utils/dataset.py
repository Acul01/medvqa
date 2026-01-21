"""
PyTorch Dataset class for Medical VQA with LLaVA formatting
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional

from utils.load_data import VQASample, MedicalVQALoader
from utils.prompt_formatter import PromptFormatter


class MedicalVQADataset(Dataset):
    """
    PyTorch Dataset for Medical VQA tasks.
    
    Formats data for LLaVA/LLaVA-Med models with question/answer tokens.
    For training: Labels are set so that loss is only computed for tokens after answer_token.
    """
    
    def __init__(
        self,
        samples: List[VQASample],
        image_processor,
        tokenizer,
        model_type: str = "llava",
        max_length: int = 512,
        for_training: bool = True,
        question_token: str = "<question>",
        answer_token: str = "<answer>"
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of VQASample objects
            image_processor: Image processor from transformers (e.g., CLIPImageProcessor)
            tokenizer: Text tokenizer from transformers
            model_type: "llava" or "llava_med"
            max_length: Maximum sequence length for tokenization
            for_training: If True, include answers in prompts; if False, prepare for inference
            question_token: Token marking question start
            answer_token: Token marking answer start
        """
        self.samples = samples
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.for_training = for_training
        
        # Initialize prompt formatter
        self.formatter = PromptFormatter(
            question_token=question_token,
            answer_token=answer_token,
            model_type=model_type
        )
        
        # Set tokenizer padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
                - pixel_values: Processed image tensor
                - input_ids: Tokenized text input
                - attention_mask: Attention mask
                - labels: Token IDs for loss computation (if training)
                    Only tokens after answer_token contribute to loss
        """
        sample = self.samples[idx]
        
        # Load and process image
        image_path = Path(sample.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        
        if self.for_training:
            # TRAINING: Format [question_token; question; answer_token; answer]
            full_prompt = self.formatter.format_training_prompt(
                question=sample.question,
                answer=sample.answer
            )
            
            # Tokenize full prompt
            full_encoding = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = full_encoding["input_ids"].squeeze(0)
            attention_mask = full_encoding["attention_mask"].squeeze(0)
            
            # Create labels: only compute loss for tokens after answer_token
            labels = input_ids.clone()
            
            # Find position of answer_token in tokenized sequence
            # Tokenize answer_token separately to find its token IDs
            answer_token_text = self.formatter.answer_token
            answer_token_ids = self.tokenizer.encode(
                answer_token_text,
                add_special_tokens=False
            )
            
            # Find where answer_token appears in the sequence
            answer_token_start_idx = None
            input_ids_list = input_ids.tolist()
            
            for i in range(len(input_ids_list) - len(answer_token_ids) + 1):
                if input_ids_list[i:i+len(answer_token_ids)] == answer_token_ids:
                    answer_token_start_idx = i + len(answer_token_ids)
                    break
            
            if answer_token_start_idx is not None:
                # Set labels to -100 (ignored in loss) for everything before answer_token
                labels[:answer_token_start_idx] = -100
                # Keep labels for answer part (after answer_token)
            else:
                # Fallback: if answer_token not found, try to find assistant prefix
                # This handles cases where tokenization splits the token differently
                assistant_prefix_ids = self.tokenizer.encode(
                    self.formatter.assistant_prefix,
                    add_special_tokens=False
                )
                for i in range(len(input_ids_list) - len(assistant_prefix_ids) + 1):
                    if input_ids_list[i:i+len(assistant_prefix_ids)] == assistant_prefix_ids:
                        # Start loss computation after assistant prefix
                        answer_token_start_idx = i + len(assistant_prefix_ids)
                        labels[:answer_token_start_idx] = -100
                        break
                
                if answer_token_start_idx is None:
                    # Last resort: set all to -100 except last part (likely the answer)
                    # This shouldn't happen, but safety check
                    labels[:len(labels)//2] = -100
            
            # Set padding tokens to -100 (ignored in loss)
            labels[labels == self.tokenizer.pad_token_id] = -100
            
        else:
            # INFERENCE: Format [question_token; question; answer_token]
            prompt = self.formatter.format_inference_prompt(
                question=sample.question
            )
            
            # Tokenize text
            text_encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            input_ids = text_encoding["input_ids"].squeeze(0)
            attention_mask = text_encoding["attention_mask"].squeeze(0)
            labels = None  # No labels for inference
        
        result = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Add labels for training
        if labels is not None:
            result["labels"] = labels
        
        # Add metadata fields only for inference (not for training)
        # These cause issues with the Trainer's data collator
        if not self.for_training:
            result["image_path"] = str(image_path)
            result["question"] = sample.question
            result["answer"] = sample.answer
        
        return result


def create_dataset(
    dataset_name: str,
    split: str,
    data_root: str,
    image_processor,
    tokenizer,
    model_type: str = "llava",
    max_length: int = 512,
    for_training: bool = True,
    question_token: str = "<question>",
    answer_token: str = "<answer>"
) -> MedicalVQADataset:
    """
    Convenience function to create a dataset.
    
    Args:
        dataset_name: Name of dataset ("pathvqa", "slake", "vqarad" or canonical names)
        split: Split to load ("train", "validation", "test")
        data_root: Root directory containing datasets
        image_processor: Image processor from transformers
        tokenizer: Text tokenizer from transformers
        model_type: "llava" or "llava_med"
        max_length: Maximum sequence length
        for_training: Whether to prepare for training or inference
        question_token: Token marking question start
        answer_token: Token marking answer start
        
    Returns:
        MedicalVQADataset instance
    """
    # Load samples
    loader = MedicalVQALoader(data_root=data_root)
    samples = loader.load_dataset(dataset_name, split=split)
    
    # Create dataset
    dataset = MedicalVQADataset(
        samples=samples,
        image_processor=image_processor,
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=max_length,
        for_training=for_training,
        question_token=question_token,
        answer_token=answer_token
    )
    
    return dataset
