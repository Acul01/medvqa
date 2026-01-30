"""
Prompt formatting for LLaVA and LLaVA-Med models

Format: USER: [question_token; question; answer_token] ASSISTANT: [answer]
For inference: answer is removed, model generates after ASSISTANT:
"""

from typing import Optional


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
        
        # LLaVA standard conversation format
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
        
        Format: USER: [question_token; question; answer_token] ASSISTANT: [answer]
        
        Args:
            question: The question to ask
            answer: The ground truth answer
            include_image: Whether to include image placeholder (for documentation)
            
        Returns:
            Formatted prompt string
        """
        # Format: [question_token; question; answer_token] (answer only after ASSISTANT:)
        prompt = f"{self.question_token} {question} {self.answer_token}"
        
        # Wrap in LLaVA conversation format - answer comes after ASSISTANT:
        conversation = f"{self.user_prefix} {prompt}\n{self.assistant_prefix} {answer}"
        
        return conversation
    
    def format_inference_prompt(
        self,
        question: str,
        include_image: bool = True
    ) -> str:
        """
        Format prompt for inference (answer removed, model generates).
        
        Format: [question_token; question; answer_token]
        
        Args:
            question: The question to ask
            include_image: Whether to include image placeholder (for documentation)
            
        Returns:
            Formatted prompt string without answer
        """
        # Format: [question_token; question; answer_token] (no answer)
        prompt = f"{self.question_token} {question} {self.answer_token}"
        
        # Wrap in LLaVA conversation format
        conversation = f"{self.user_prefix} {prompt}\n{self.assistant_prefix}"
        
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
        Removes the answer_token and any prefixes.
        
        Args:
            response: Model's raw response
            
        Returns:
            Cleaned answer string
        """
        # Remove answer token if present
        if self.answer_token in response:
            parts = response.split(self.answer_token, 1)
            if len(parts) > 1:
                answer = parts[1].strip()
            else:
                answer = response.strip()
        else:
            answer = response.strip()
        
        # Remove assistant prefix if present
        if self.assistant_prefix in answer:
            answer = answer.replace(self.assistant_prefix, "").strip()
        
        return answer


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
