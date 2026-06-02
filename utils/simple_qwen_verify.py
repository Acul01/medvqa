import json
import re
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch

# =========================
# CONFIG
# =========================
INPUT_JSON = "input.json"
OUTPUT_JSON = "output.json"

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# =========================
# LOAD MODEL
# =========================
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

# =========================
# HELPERS
# =========================
LETTER_TO_INDEX = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}

def parse_letter(text):
    text = text.strip().upper()

    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)

    return None

# =========================
# LOAD DATA
# =========================
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

results = []
num_correct = 0

# =========================
# INFERENCE LOOP
# =========================
for sample in tqdm(data):

    image_path = sample["image_path"]
    question = sample["question"]
    options = sample["options"]
    pseudo_reasoning_raw = sample["pseudo_reasoning_raw"]
    correct_index = sample["correct_index"]

    allowed = ", ".join([chr(65 + i) for i in range(len(options))])

    correct_letter = chr(65 + correct_index)

    verify_prompt = (
        "You answer a multiple-choice medical question. "
        "The reasoning below describes visual findings from a medical image. "
        "The image itself is not shown. Base your choice solely on the reasoning text.\n\n"
        f"Reasoning:\n{pseudo_reasoning_raw}\n\n"
        f"Question:\n{str(question).strip()}\n\n"
        f"Options:\n" + "\n".join(options) + "\n\n"
        f"Respond with ONLY the single MC option letter ({allowed}) that is most supported by the reasoning."
        " Do not output any other text."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": verify_prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=10
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    predicted_letter = parse_letter(output_text)

    is_correct = predicted_letter == correct_letter

    if is_correct:
        num_correct += 1

    results.append({
        "image_path": image_path,
        "question": question,
        "options": options,
        "pseudo_reasoning_raw": pseudo_reasoning_raw,
        "correct_index": correct_index,
        "correct_letter": correct_letter,
        "predicted_letter": predicted_letter,
        "prediction_raw": output_text,
        "is_correct": is_correct
    })

# =========================
# SAVE OUTPUT
# =========================
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)

# =========================
# FINAL ACCURACY
# =========================
accuracy = num_correct / len(results)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Correct: {num_correct}/{len(results)}")