"""Prompt formatting helpers for LLaVA and LLaVA-Med models."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from transformers import AutoProcessor

from utils.mc_vqa import format_mc_user_text


@dataclass
class PromptFormatter:
    model_id: str

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _build_messages_llava15(
        self,
        question: str,
        answer: Optional[str],
        include_image: bool,
        mc_options: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        user_text = format_mc_user_text(question, mc_options) if mc_options else question
        user_content = []
        if include_image:
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": user_text})

        messages = [{"role": "user", "content": user_content}]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return messages

    def _build_messages_llava_med_mistral(
        self,
        question: str,
        answer: Optional[str],
        include_image: bool,
        mc_options: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        user_text = format_mc_user_text(question, mc_options) if mc_options else question
        content = f"<image>\n{user_text}" if include_image else user_text

        messages = [{"role": "user", "content": content}]
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        return messages

    def format_training_prompt(
        self,
        question: str,
        answer: str,
        include_image: bool = True,
        mc_options: Optional[List[str]] = None,
    ) -> str:
        if "llava-med-v1.5-mistral" in self.model_id or "llava_mistral" in self.model_id:
            messages = self._build_messages_llava_med_mistral(question, answer, include_image, mc_options)
        else:
            messages = self._build_messages_llava15(question, answer, include_image, mc_options)

        return self.processor.apply_chat_template(messages, add_generation_prompt=False)

    def format_inference_prompt(
        self,
        question: str,
        include_image: bool = True,
        mc_options: Optional[List[str]] = None,
    ) -> str:
        if "llava-med-v1.5-mistral" in self.model_id or "llava_mistral" in self.model_id:
            messages = self._build_messages_llava_med_mistral(question, None, include_image, mc_options)
        else:
            messages = self._build_messages_llava15(question, None, include_image, mc_options)

        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def extract_answer_from_response(self, response: str) -> str:
        answer = response.strip()

        import re

        answer = re.sub(r"ASSISTANT:\s*", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"USER:\s*", "", answer, flags=re.IGNORECASE)
        answer = re.sub(r"<\|im_(start|end)\|>", "", answer)
        answer = re.sub(r"\|im_(start|end)\|?>", "", answer)

        return " ".join(answer.split())
