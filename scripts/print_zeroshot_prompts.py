import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoProcessor
from utils.mc_json_zeroshot import (
    build_mc_zeroshot_training_conversation,
    build_qwen_json_conversation,
)
from utils.qwen_vl import configure_qwen_processor_pixels, resolve_processor_path

QUESTION = "What is the primary finding on this chest X-ray?"
OPTIONS = ["Normal", "Pneumonia", "Pleural effusion", "Cardiomegaly"]
GT_LETTER = "B"
INFERRED_RATIONALE = (
    "There is increased opacity in the right lower lobe suggesting airspace consolidation."
)
JSON_ORDER = "answer_first"
MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(resolve_processor_path(MODEL_PATH))
configure_qwen_processor_pixels(processor)

infer_conv = build_qwen_json_conversation(
    QUESTION, OPTIONS, json_order=JSON_ORDER, image="<PIL.Image>"
)
infer_prompt = processor.apply_chat_template(
    infer_conv, tokenize=False, add_generation_prompt=True
)

train_conv = build_mc_zeroshot_training_conversation(
    QUESTION,
    OPTIONS,
    rationale_raw=INFERRED_RATIONALE,
    answer=GT_LETTER,
    json_order=JSON_ORDER,
    image="<PIL.Image>",
)
train_prompt = processor.apply_chat_template(
    train_conv, tokenize=False, add_generation_prompt=False
)

print("=" * 72)
print("INFERENCE PROMPT")
print("=" * 72)
print(infer_prompt)

print("\n" + "=" * 72)
print("TRAINING PROMPT")
print("=" * 72)
print(train_prompt)
