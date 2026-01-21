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
        answer_token: str = "<answer>",
        num_image_tokens: int = 576  # Number of image tokens (patches) - default for LLaVA-1.5 with 336x336
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
        
        # Calculate number of image patches
        # LLaVA vision tower typically outputs features with shape [batch, num_patches, hidden_dim]
        # For CLIP-based vision towers: num_patches = (image_size / patch_size)^2 + 1 (for CLS token)
        # We need to determine how many image token placeholders to insert
        if len(pixel_values.shape) == 4:  # [batch, channels, height, width]
            # This is raw image tensor, will be processed by vision tower
            # Estimate patches: typically 14x14 = 196 patches for 224x224 image
            # Or 24x24 = 576 patches for 336x336 image
            # We'll calculate based on actual processed features later
            num_image_tokens = None  # Will be determined after processing
        else:
            # Already processed features
            num_image_tokens = pixel_values.shape[1] if len(pixel_values.shape) > 1 else 1
        
        # Use the configured number of image tokens
        num_image_tokens = self.num_image_tokens
        
        # #region agent log
        _log("debug-session", "run1", "B", "dataset.py:110", "Image processed", {"pixel_values_shape": list(pixel_values.shape), "num_image_tokens": num_image_tokens})
        # #endregion
        
        if self.for_training:
            # TRAINING: Format [question_token; question; answer_token; answer]
            full_prompt = self.formatter.format_training_prompt(
                question=sample.question,
                answer=sample.answer
            )
            
            # Insert image token placeholders at the beginning of USER message
            # LLaVA expects multiple image tokens (one per image patch)
            image_token_string = ""
            if hasattr(self, 'image_token') and self.image_token:
                # Repeat image token for each patch
                image_token_string = " ".join([self.image_token] * num_image_tokens)
            elif hasattr(self, 'image_token_id') and self.image_token_id is not None:
                # Use "<image>" string, will be converted to token IDs
                image_token_string = " ".join(["<image>"] * num_image_tokens)
            else:
                # Fallback: try "<image>"
                image_token_string = " ".join(["<image>"] * num_image_tokens)
            
            # Insert image tokens at the start of USER message
            if image_token_string:
                full_prompt = full_prompt.replace(
                    f"{self.formatter.user_prefix}",
                    f"{self.formatter.user_prefix} {image_token_string}"
                )
            
            # #region agent log
            _log("debug-session", "run1", "A", "dataset.py:130", "Prompt formatted", {"prompt_preview": full_prompt[:200], "has_image_token": "<image>" in full_prompt or (hasattr(self, 'image_token') and self.image_token and self.image_token in full_prompt)})
            # #endregion
            
            # Tokenize full prompt
            full_encoding = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            # If we have image_token_id but it's not in the tokenized sequence, insert it manually
            if hasattr(self, 'image_token_id') and self.image_token_id is not None:
                input_ids_list = full_encoding["input_ids"].squeeze(0).tolist()
                # Count how many image tokens are already in the sequence
                num_existing_image_tokens = input_ids_list.count(self.image_token_id)
                
                if num_existing_image_tokens < num_image_tokens:
                    # Need to insert more image tokens
                    user_prefix_ids = self.tokenizer.encode(
                        self.formatter.user_prefix,
                        add_special_tokens=False
                    )
                    if user_prefix_ids:
                        # Find first occurrence of user prefix
                        for i in range(len(input_ids_list) - len(user_prefix_ids) + 1):
                            if input_ids_list[i:i+len(user_prefix_ids)] == user_prefix_ids:
                                # Insert remaining image tokens after user prefix
                                tokens_to_insert = num_image_tokens - num_existing_image_tokens
                                for j in range(tokens_to_insert):
                                    input_ids_list.insert(i + len(user_prefix_ids) + j, self.image_token_id)
                                # Truncate if needed
                                if len(input_ids_list) > self.max_length:
                                    input_ids_list = input_ids_list[:self.max_length]
                                # Update encoding
                                full_encoding["input_ids"] = torch.tensor([input_ids_list])
                                break
            
            input_ids = full_encoding["input_ids"].squeeze(0)
            attention_mask = full_encoding["attention_mask"].squeeze(0)
            
            # #region agent log
            _log("debug-session", "run1", "A", "dataset.py:107", "Text tokenized", {"input_ids_shape": list(input_ids.shape), "input_ids_sample": input_ids[:20].tolist() if len(input_ids) > 20 else input_ids.tolist()})
            # #endregion
            
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
            
            # Insert image token placeholders at the beginning of USER message
            image_token_string = ""
            if hasattr(self, 'image_token') and self.image_token:
                image_token_string = " ".join([self.image_token] * num_image_tokens)
            elif hasattr(self, 'image_token_id') and self.image_token_id is not None:
                image_token_string = " ".join(["<image>"] * num_image_tokens)
            else:
                image_token_string = " ".join(["<image>"] * num_image_tokens)
            
            if image_token_string:
                prompt = prompt.replace(
                    f"{self.formatter.user_prefix}",
                    f"{self.formatter.user_prefix} {image_token_string}"
                )
            
            # Tokenize text
            text_encoding = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )
            
            # Insert image token IDs if needed (same as training)
            if hasattr(self, 'image_token_id') and self.image_token_id is not None:
                input_ids_list = text_encoding["input_ids"].squeeze(0).tolist()
                num_existing_image_tokens = input_ids_list.count(self.image_token_id)
                
                if num_existing_image_tokens < num_image_tokens:
                    user_prefix_ids = self.tokenizer.encode(
                        self.formatter.user_prefix,
                        add_special_tokens=False
                    )
                    if user_prefix_ids:
                        for i in range(len(input_ids_list) - len(user_prefix_ids) + 1):
                            if input_ids_list[i:i+len(user_prefix_ids)] == user_prefix_ids:
                                tokens_to_insert = num_image_tokens - num_existing_image_tokens
                                for j in range(tokens_to_insert):
                                    input_ids_list.insert(i + len(user_prefix_ids) + j, self.image_token_id)
                                if len(input_ids_list) > self.max_length:
                                    input_ids_list = input_ids_list[:self.max_length]
                                text_encoding["input_ids"] = torch.tensor([input_ids_list])
                                break
            
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
    model_type: str = "llava",
    max_length: int = 512,
    for_training: bool = True,
    question_token: str = "<question>",
    answer_token: str = "<answer>",
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
        answer_token=answer_token,
        num_image_tokens=num_image_tokens
    )
    
    return dataset
