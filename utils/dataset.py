"""
PyTorch Dataset class for Medical VQA with LLaVA formatting
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional
import json
import os

from utils.load_data import VQASample, MedicalVQALoader
from utils.prompt_formatter import PromptFormatter

# #region agent log
LOG_PATH = "/Users/luca/Desktop/Thesis+Work/medvqa/.cursor/debug.log"
def _log(session_id, run_id, hypothesis_id, location, message, data):
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({"sessionId": session_id, "runId": run_id, "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": os.times().elapsed}) + "\n")
    except: pass
# #endregion


class MedicalVQADataset(Dataset):
    """
    PyTorch Dataset for Medical VQA tasks.
    
    Formats data for LLaVA/LLaVA-Med models using apply_chat_template.
    Uses PromptFormatter with apply_chat_template to ensure correct format.
    For training: Labels are set so that loss is only computed for tokens after ASSISTANT:.
    """
    
    def __init__(
        self,
        samples: List[VQASample],
        image_processor,
        tokenizer,
        model_id: str,  # Changed from model_type to model_id
        max_length: int = 512,
        for_training: bool = True,
        question_token: str = "<question>",  # Deprecated, kept for compatibility
        answer_token: str = "<answer>",  # Deprecated, kept for compatibility
        num_image_tokens: int = 576  # Number of image tokens (patches) - default for LLaVA-1.5 with 336x336
    ):
        """
        Initialize dataset.
        
        Args:
            samples: List of VQASample objects
            image_processor: Image processor from transformers (e.g., CLIPImageProcessor)
            tokenizer: Text tokenizer from transformers
            model_id: Model ID or path (e.g., "llava-hf/llava-1.5-7b-hf")
            max_length: Maximum sequence length for tokenization
            for_training: If True, include answers in prompts; if False, prepare for inference
            question_token: Token marking question start (deprecated, kept for compatibility)
            answer_token: Token marking answer start (deprecated, kept for compatibility)
        """
        self.samples = samples
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.for_training = for_training
        self.num_image_tokens = num_image_tokens  # Store number of image tokens needed
        
        # Initialize prompt formatter with model_id
        self.formatter = PromptFormatter(model_id=model_id)
        
        # Set tokenizer padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get image token placeholder from tokenizer (LLaVA uses special image tokens)
        # Try to find image token - LLaVA typically uses "<image>" or similar
        self.image_token = None
        if hasattr(tokenizer, 'image_token_id') and tokenizer.image_token_id is not None:
            # Tokenizer has explicit image token ID
            self.image_token_id = tokenizer.image_token_id
        else:
            # Try to find image token by encoding "<image>"
            try:
                image_token_ids = tokenizer.encode("<image>", add_special_tokens=False)
                if image_token_ids:
                    self.image_token_id = image_token_ids[0]
                    self.image_token = "<image>"
            except:
                pass
            
            # If still not found, try common LLaVA image token IDs
            if self.image_token_id is None:
                # LLaVA-1.5 typically uses token ID 32000 or similar
                # Try to find it by checking if tokenizer has vocab with image-related tokens
                if hasattr(tokenizer, 'vocab'):
                    for token, token_id in tokenizer.vocab.items():
                        if 'image' in token.lower() or token == '<image>':
                            self.image_token_id = token_id
                            self.image_token = token
                            break
        
        # #region agent log
        _log("debug-session", "run1", "C", "dataset.py:95", "Image token detection", {"image_token": self.image_token, "image_token_id": self.image_token_id if hasattr(self, 'image_token_id') else None})
        # #endregion
    
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
        
        # Use the configured number of image tokens
        num_image_tokens = self.num_image_tokens
        
        # #region agent log
        _log("debug-session", "run1", "B", "dataset.py:132", "Image processed", {"pixel_values_shape": list(pixel_values.shape), "num_image_tokens": num_image_tokens})
        # #endregion
        
        # Use apply_chat_template to get formatted prompt text
        if self.for_training:
            text_prompt = self.formatter.format_training_prompt(
                question=sample.question,
                answer=sample.answer,
                include_image=True
            )
        else:
            text_prompt = self.formatter.format_inference_prompt(
                question=sample.question,
                include_image=True
            )
        
        # Tokenize the prompt
        # We need to manually insert image tokens after the user prefix
        # First, find where to insert image tokens
        if "llava-med-v1.5-mistral" in self.formatter.model_id or "llava_mistral" in self.formatter.model_id:
            # For Mistral format, <image> is already in the text
            # Tokenize and replace <image> token with multiple image tokens
            encoded = self.tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids_list = encoded["input_ids"].squeeze(0).tolist()
            
            # Find <image> token ID
            image_token_id = None
            try:
                image_token_ids = self.tokenizer.encode("<image>", add_special_tokens=False)
                if image_token_ids:
                    image_token_id = image_token_ids[0]
            except:
                pass
            
            if image_token_id is None:
                image_token_id = 32000  # Default
            
            # Replace single <image> token with multiple image tokens
            new_input_ids = []
            for token_id in input_ids_list:
                if token_id == image_token_id:
                    new_input_ids.extend([image_token_id] * self.num_image_tokens)
                else:
                    new_input_ids.append(token_id)
            input_ids_list = new_input_ids
        else:
            # Standard LLaVA: tokenize and insert image tokens after user prefix
            # The prompt from apply_chat_template already has the right structure
            encoded = self.tokenizer(text_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids_list = encoded["input_ids"].squeeze(0).tolist()
            
            # Find image token ID
            image_token_id = None
            try:
                image_token_ids = self.tokenizer.encode("<image>", add_special_tokens=False)
                if image_token_ids:
                    image_token_id = image_token_ids[0]
            except:
                pass
            
            if image_token_id is None:
                image_token_id = 32000  # Default
            
            # Replace single <image> token with multiple image tokens
            new_input_ids = []
            for token_id in input_ids_list:
                if token_id == image_token_id:
                    new_input_ids.extend([image_token_id] * self.num_image_tokens)
                else:
                    new_input_ids.append(token_id)
            input_ids_list = new_input_ids
        
        # Truncate if needed
        if len(input_ids_list) > self.max_length:
            input_ids_list = input_ids_list[:self.max_length]
        
        # Pad to max_length
        if len(input_ids_list) < self.max_length:
            input_ids_list = input_ids_list + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids_list))
        
        # Create tensors
        input_ids = torch.tensor(input_ids_list)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # For training: create labels (only compute loss for assistant response)
        if self.for_training:
            labels = input_ids.clone()
            
            # Find where assistant response starts
            # Look for "ASSISTANT:" in the tokenized sequence
            assistant_text = "ASSISTANT:"
            assistant_ids = self.tokenizer.encode(assistant_text, add_special_tokens=False)
            input_ids_list_for_labels = input_ids.tolist()
            
            answer_start_idx = None
            for i in range(len(input_ids_list_for_labels) - len(assistant_ids) + 1):
                if input_ids_list_for_labels[i:i+len(assistant_ids)] == assistant_ids:
                    answer_start_idx = i + len(assistant_ids)
                    break
            
            if answer_start_idx is not None:
                labels[:answer_start_idx] = -100
            else:
                # Fallback: assume last third is the answer
                labels[:len(labels) * 2 // 3] = -100
            
            # Set padding tokens to -100
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = None
        
        result = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Add labels for training
        if labels is not None:
            result["labels"] = labels
        
        # #region agent log
        _log("debug-session", "run1", "D", "dataset.py:177", "Dataset item created", {"has_pixel_values": "pixel_values" in result, "has_input_ids": "input_ids" in result, "input_ids_shape": list(result["input_ids"].shape) if "input_ids" in result else None, "pixel_values_shape": list(result["pixel_values"].shape) if "pixel_values" in result else None})
        # #endregion
        
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
    model_id: str,  # Changed from model_type to model_id
    max_length: int = 512,
    for_training: bool = True,
    question_token: str = "<question>",  # Deprecated, kept for compatibility
    answer_token: str = "<answer>",  # Deprecated, kept for compatibility
    num_image_tokens: int = 576  # Number of image tokens (patches) - default for LLaVA-1.5
) -> MedicalVQADataset:
    """
    Convenience function to create a dataset.
    
    Args:
        dataset_name: Name of dataset ("pathvqa", "slake", "vqarad" or canonical names)
        split: Split to load ("train", "validation", "test")
        data_root: Root directory containing datasets
        image_processor: Image processor from transformers
        tokenizer: Text tokenizer from transformers
        model_id: Model ID or path (e.g., "llava-hf/llava-1.5-7b-hf")
        max_length: Maximum sequence length
        for_training: Whether to prepare for training or inference
        question_token: Token marking question start (deprecated)
        answer_token: Token marking answer start (deprecated)
        
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
        model_id=model_id,
        max_length=max_length,
        for_training=for_training,
        question_token=question_token,
        answer_token=answer_token,
        num_image_tokens=num_image_tokens
    )
    
    return dataset
