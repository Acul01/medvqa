"""
Prompt formatting for LLaVA and LLaVA-Med models

Standard LLaVA format: USER: <image>\n[question] ASSISTANT: [answer]
For inference: answer is removed, model generates after ASSISTANT:
Note: Image tokens are inserted by the dataset after "USER:"
"""

from typing import Optional, List, Dict, Any


class PromptFormatter:
    """Format prompts for LLaVA models with question/answer tokens"""
    
    def __init__(
        self,
        question_token: str = "<question>",
        answer_token: str = "<answer>",
        model_type: str = "llava"  # "llava" or "llava_med"
    ):
        """
        Initialize prompt formatter.
        
        Args:
            question_token: Token to mark the start of a question
            answer_token: Token to mark the start of an answer
            model_type: Type of model ("llava" or "llava_med")
        """
        self.question_token = question_token
        self.answer_token = answer_token
        self.model_type = model_type
        
        # Standard LLaVA conversation format
        if model_type == "llava":
            self.system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            self.user_prefix = "USER:"
            self.assistant_prefix = "ASSISTANT:"
        elif model_type == "llava_med":
            # LLaVA-Med might have medical-specific system message
            self.system_message = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions about medical images."
            self.user_prefix = "USER:"
            self.assistant_prefix = "ASSISTANT:"
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'llava' or 'llava_med'")
    
    def format_training_prompt(
        self,
        question: str,
        answer: str,
        include_image: bool = True
    ) -> str:
        """
        Format prompt for training (includes answer).
        
        Format: <|im_start|>user <image>\n[question]<|im_end|><|im_start|>assistant [answer]
        Note: Image tokens are inserted by the dataset after "<|im_start|>user"
        
        Args:
            question: The question to ask
            answer: The ground truth answer
            include_image: Whether to include image placeholder (for documentation)
            
        Returns:
            Formatted prompt string
        """
        # LLaVA-interleave format: <|im_start|>user <image>\n[question]<|im_end|><|im_start|>assistant [answer]
        # Image tokens are inserted by dataset after "<|im_start|>user"
        conversation = f"{self.user_prefix} <image>\n{question}{self.user_suffix}{self.assistant_prefix} {answer}"
        
        return conversation
    
    def format_inference_prompt(
        self,
        question: str,
        include_image: bool = True
    ) -> str:
        """
        Format prompt for inference (answer removed, model generates).
        
        Format: USER: <image>\n[question] ASSISTANT:
        Note: Image tokens are inserted by the dataset after "USER:"
        
        Args:
            question: The question to ask
            include_image: Whether to include image placeholder (for documentation)
            
        Returns:
            Formatted prompt string without answer
        """
        # Standard LLaVA format: USER: <image>\n[question] ASSISTANT:
        # Image tokens are inserted by dataset after "USER:"
        conversation = f"{self.user_prefix} <image>\n{question} {self.assistant_prefix}"
        
        return conversation
    
    def format_simple(self, question: str, answer: Optional[str] = None) -> str:
        """
        Simple format without LLaVA conversation wrapper.
        
        Args:
            question: The question
            answer: Optional answer (if None, only question is returned)
            
        Returns:
            Formatted string
        """
        if answer:
            return f"{self.question_token} {question} {self.answer_token} {answer}"
        else:
            return f"{self.question_token} {question} {self.answer_token}"
    
    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract answer from model response.
        Extracts text after ASSISTANT: and removes any remaining format tokens.
        
        Args:
            response: Model's raw response (should be only newly generated tokens)
            
        Returns:
            Cleaned answer string
        """
        answer = response.strip()
        
        # Remove assistant prefix if present at the start
        if answer.startswith(self.assistant_prefix):
            answer = answer[len(self.assistant_prefix):].strip()
        
        # Remove any occurrences of format tokens
        # Remove ASSISTANT: prefix
        answer = answer.replace(self.assistant_prefix, "").strip()
        # Remove USER: prefix (in case it appears)
        answer = answer.replace(self.user_prefix, "").strip()
        
        # Remove any remaining format tokens that might appear
        import re
        # Remove any <|im_start|> or <|im_end|> tokens (leftover from other formats)
        answer = re.sub(r'<\|im_(start|end)\|>', '', answer)
        answer = re.sub(r'\|im_(start|end)\|?>', '', answer)
        
        # Clean up any extra whitespace
        answer = ' '.join(answer.split())
        
        return answer
    
    def format_chat_template_training(
        self,
        question: str,
        answer: str,
        image_path: Optional[str] = None
    ) -> list:
        """
        Format conversation in chat template format for training.
        
        Format matches HuggingFace transformers apply_chat_template format:
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What's shown in this image?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "This image shows..."},]
            },
        ]
        
        Args:
            question: The question to ask
            answer: The ground truth answer
            image_path: Optional path to image (for reference, not used in template)
            
        Returns:
            List of conversation dictionaries
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer},]
            },
        ]
        return conversation
    
    def format_chat_template_inference(
        self,
        question: str,
        image_path: Optional[str] = None
    ) -> list:
        """
        Format conversation in chat template format for inference.
        
        Format matches HuggingFace transformers apply_chat_template format:
        [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What's shown in this image?"},
                ],
            },
        ]
        
        Args:
            question: The question to ask
            image_path: Optional path to image (for reference, not used in template)
            
        Returns:
            List of conversation dictionaries
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        return conversation


# Convenience functions
def format_llava_prompt(question: str, answer: Optional[str] = None, for_training: bool = True) -> str:
    """
    Quick function to format LLaVA prompt.
    
    Args:
        question: The question
        answer: The answer (required if for_training=True)
        for_training: If True, include answer; if False, prepare for inference
        
    Returns:
        Formatted prompt string
    """
    formatter = PromptFormatter(model_type="llava")
    if for_training:
        if answer is None:
            raise ValueError("Answer required for training prompt")
        return formatter.format_training_prompt(question, answer)
    else:
        return formatter.format_inference_prompt(question)


def format_llava_med_prompt(question: str, answer: Optional[str] = None, for_training: bool = True) -> str:
    """
    Quick function to format LLaVA-Med prompt.
    
    Args:
        question: The question
        answer: The answer (required if for_training=True)
        for_training: If True, include answer; if False, prepare for inference
        
    Returns:
        Formatted prompt string
    """
    formatter = PromptFormatter(model_type="llava_med")
    if for_training:
        if answer is None:
            raise ValueError("Answer required for training prompt")
        return formatter.format_training_prompt(question, answer)
    else:
        return formatter.format_inference_prompt(question)


# Chat template convenience functions
def format_llava_chat_template(
    question: str,
    answer: Optional[str] = None,
    for_training: bool = True,
    image_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Format LLaVA conversation in chat template format.
    
    Args:
        question: The question to ask
        answer: The answer (required if for_training=True)
        for_training: If True, include answer; if False, prepare for inference
        image_path: Optional path to image (for reference)
        
    Returns:
        List of conversation dictionaries in chat template format
    """
    formatter = PromptFormatter(model_type="llava")
    if for_training:
        if answer is None:
            raise ValueError("Answer required for training prompt")
        return formatter.format_chat_template_training(question, answer, image_path)
    else:
        return formatter.format_chat_template_inference(question, image_path)


def format_llava_med_chat_template(
    question: str,
    answer: Optional[str] = None,
    for_training: bool = True,
    image_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Format LLaVA-Med conversation in chat template format.
    
    Args:
        question: The question to ask
        answer: The answer (required if for_training=True)
        for_training: If True, include answer; if False, prepare for inference
        image_path: Optional path to image (for reference)
        
    Returns:
        List of conversation dictionaries in chat template format
    """
    formatter = PromptFormatter(model_type="llava_med")
    if for_training:
        if answer is None:
            raise ValueError("Answer required for training prompt")
        return formatter.format_chat_template_training(question, answer, image_path)
    else:
        return formatter.format_chat_template_inference(question, image_path)
