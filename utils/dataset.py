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
    Format: USER: [question_token; question; answer_token] ASSISTANT: [answer]
    For training: Labels are set so that loss is only computed for tokens after ASSISTANT:.
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
        self.num_image_tokens = num_image_tokens  # Store number of image tokens needed
        
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
        
        # Use the configured number of image tokens
        num_image_tokens = self.num_image_tokens
        
        # #region agent log
        _log("debug-session", "run1", "B", "dataset.py:132", "Image processed", {"pixel_values_shape": list(pixel_values.shape), "num_image_tokens": num_image_tokens})
        # #endregion
        
        if self.for_training:
            # TRAINING: Format USER: [question_token; question; answer_token] ASSISTANT: [answer]
            full_prompt = self.formatter.format_training_prompt(
                question=sample.question,
                answer=sample.answer
            )
            
            # Insert image token placeholders at the beginning of USER message
            # LLaVA expects multiple image tokens (one per image patch)
            # We'll insert them directly as token IDs to avoid tokenization issues
            # First, tokenize the prompt without image tokens
            user_prefix_ids = self.tokenizer.encode(
                self.formatter.user_prefix,
                add_special_tokens=False
            )
            
            # Tokenize the rest of the prompt
            rest_of_prompt = full_prompt.replace(f"{self.formatter.user_prefix}", "").strip()
            rest_ids = self.tokenizer.encode(rest_of_prompt, add_special_tokens=False)
            
            # Get the image token ID
            image_token_id = None
            if hasattr(self, 'image_token_id') and self.image_token_id is not None:
                image_token_id = self.image_token_id
            else:
                # Try to encode "<image>" to get the token ID
                try:
                    image_token_ids = self.tokenizer.encode("<image>", add_special_tokens=False)
                    if image_token_ids:
                        image_token_id = image_token_ids[0]
                except:
                    pass
            
            if image_token_id is None:
                # Last resort: try common LLaVA image token IDs
                # LLaVA-1.5 typically uses 32000
                image_token_id = 32000
            
            # Build input_ids with image tokens inserted
            input_ids_list = user_prefix_ids + [image_token_id] * num_image_tokens + rest_ids
            
            # Truncate carefully: keep user_prefix and image tokens, truncate rest
            # CRITICAL: Image tokens must ALWAYS be preserved completely
            prefix_and_image_length = len(user_prefix_ids) + num_image_tokens
            
            # Ensure max_length is at least large enough for prefix + image tokens
            if self.max_length < prefix_and_image_length:
                # This is an error condition - max_length is too small
                raise ValueError(
                    f"max_length ({self.max_length}) is too small for {num_image_tokens} image tokens + "
                    f"prefix ({len(user_prefix_ids)} tokens). "
                    f"Minimum required: {prefix_and_image_length + 50} (recommended: {prefix_and_image_length + 200})"
                )
            
            if len(input_ids_list) > self.max_length:
                # Keep prefix + image tokens, truncate rest
                max_rest_length = self.max_length - prefix_and_image_length
                input_ids_list = user_prefix_ids + [image_token_id] * num_image_tokens + rest_ids[:max_rest_length]
            
            # Pad to max_length
            if len(input_ids_list) < self.max_length:
                input_ids_list = input_ids_list + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids_list))
            
            # Create encoding from list
            input_ids = torch.tensor([input_ids_list])
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
            
            # #region agent log
            num_inserted = input_ids.tolist().count(image_token_id)
            _log("debug-session", "run1", "A", "dataset.py:185", "Text tokenized with image tokens", {"input_ids_shape": list(input_ids.shape), "num_image_tokens_inserted": num_inserted, "expected": num_image_tokens, "image_token_id": image_token_id})
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
            
            # Insert image token placeholders directly as token IDs (same as training)
            user_prefix_ids = self.tokenizer.encode(
                self.formatter.user_prefix,
                add_special_tokens=False
            )
            
            # Get the image token ID
            image_token_id = None
            if hasattr(self, 'image_token_id') and self.image_token_id is not None:
                image_token_id = self.image_token_id
            else:
                try:
                    image_token_ids = self.tokenizer.encode("<image>", add_special_tokens=False)
                    if image_token_ids:
                        image_token_id = image_token_ids[0]
                except:
                    pass
            
            if image_token_id is None:
                image_token_id = 32000  # Default LLaVA image token ID
            
            # Tokenize the prompt without USER prefix
            prompt_without_prefix = prompt.replace(f"{self.formatter.user_prefix}", "").strip()
            rest_ids = self.tokenizer.encode(prompt_without_prefix, add_special_tokens=False)
            
            # Build input_ids: [user_prefix] + [image_token] * num_image_tokens + [rest_of_prompt]
            input_ids_list = user_prefix_ids + [image_token_id] * num_image_tokens + rest_ids
            
            # Truncate if needed (keep prefix and image tokens)
            # CRITICAL: Image tokens must ALWAYS be preserved completely
            prefix_and_image_length = len(user_prefix_ids) + num_image_tokens
            
            # Ensure max_length is at least large enough for prefix + image tokens
            if self.max_length < prefix_and_image_length:
                # This is an error condition - max_length is too small
                raise ValueError(
                    f"max_length ({self.max_length}) is too small for {num_image_tokens} image tokens + "
                    f"prefix ({len(user_prefix_ids)} tokens). "
                    f"Minimum required: {prefix_and_image_length + 50} (recommended: {prefix_and_image_length + 200})"
                )
            
            if len(input_ids_list) > self.max_length:
                # Keep prefix + image tokens, truncate rest
                max_rest_length = self.max_length - prefix_and_image_length
                input_ids_list = user_prefix_ids + [image_token_id] * num_image_tokens + rest_ids[:max_rest_length]
            
            # Pad to max_length
            if len(input_ids_list) < self.max_length:
                input_ids_list = input_ids_list + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids_list))
            
            # Create tensors
            input_ids = torch.tensor([input_ids_list]).squeeze(0)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
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
