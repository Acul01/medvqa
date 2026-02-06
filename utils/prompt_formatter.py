"""
Prompt formatting for LLaVA and LLaVA-Med models using apply_chat_template
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from transformers import AutoProcessor


@dataclass
class PromptFormatter:
    model_id: str

    def __post_init__(self):
        # Processor kapselt Tokenizer + Bildvorverarbeitung und stellt apply_chat_template bereit
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _build_messages_llava15(
        self, question: str, answer: Optional[str], include_image: bool
    ) -> List[Dict[str, Any]]:
        # LLaVA-Doku: content ist Liste aus {"type": "image"} und {"type": "text"}
        user_content = []
        if include_image:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": user_content}]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def _build_messages_llava_med_mistral(
        self, question: str, answer: Optional[str], include_image: bool
    ) -> List[Dict[str, str]]:
        # Dieses Modell nutzt ein [INST]...[/INST]-Template laut tokenizer_config.json
        # => content ist STRING (nicht multimodale content-Liste).
        # Bild-Platzhalter wird als Literal "<image>" in den User-Text geschrieben (übliches LLaVA-Schema).
        content = f"<image>\n{question}" if include_image else question

        messages = [{"role": "user", "content": content}]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        return messages

    def format_training_prompt(self, question: str, answer: str, include_image: bool = True) -> str:
        # Training: vollständige Konversation inkl. Assistant-Antwort (kein generation prompt)
        if "llava-med-v1.5-mistral" in self.model_id or "llava_mistral" in self.model_id:
            messages = self._build_messages_llava_med_mistral(question, answer, include_image)
        else:
            messages = self._build_messages_llava15(question, answer, include_image)

        return self.processor.apply_chat_template(messages, add_generation_prompt=False)

    def format_inference_prompt(self, question: str, include_image: bool = True) -> str:
        # Inferenz: nur bis Assistant-Start; Modell generiert danach
        if "llava-med-v1.5-mistral" in self.model_id or "llava_mistral" in self.model_id:
            messages = self._build_messages_llava_med_mistral(question, None, include_image)
        else:
            messages = self._build_messages_llava15(question, None, include_image)

        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def extract_answer_from_response(self, response: str) -> str:
        """
        Extract answer from model response.
        Removes any format tokens and cleans up the response.
        
        Args:
            response: Model's raw response (should be only newly generated tokens)
            
        Returns:
            Cleaned answer string
        """
        answer = response.strip()
        
        # Remove common format tokens
        import re
        # Remove ASSISTANT: prefix if present
        answer = re.sub(r'ASSISTANT:\s*', '', answer, flags=re.IGNORECASE)
        # Remove USER: prefix if present
        answer = re.sub(r'USER:\s*', '', answer, flags=re.IGNORECASE)
        # Remove any <|im_start|> or <|im_end|> tokens
        answer = re.sub(r'<\|im_(start|end)\|>', '', answer)
        answer = re.sub(r'\|im_(start|end)\|?>', '', answer)
        
        # Clean up any extra whitespace
        answer = ' '.join(answer.split())
        
        return answer
