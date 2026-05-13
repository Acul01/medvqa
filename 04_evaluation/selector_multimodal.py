"""
Paper-near selector workflow for reliable VQA with LLaVA-Med.

Subcommands:
1) extract: build multimodal features + labels
2) train: train selector MLP regressor
3) eval: threshold selector scores, compute risk/coverage/AURC and plot
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from utils.load_data import MedicalVQALoader
from utils.mc_vqa import format_mc_user_text, parse_predicted_letter
from utils.paths import (
    get_data_root,
    get_mc_data_root,
    get_selector_eval_dir,
    get_selector_features_dir,
    get_selector_models_dir,
)
from utils.save_results import clean_answer

DEFAULT_MODEL_PATH = "microsoft/llava-med-v1.5-mistral-7b"
DEFAULT_FEATURES_DIR = get_selector_features_dir()
DEFAULT_SELECTOR_DIR = get_selector_models_dir()


def exact_match_letter(predicted_text: str, gt_letter: str, n_options: int) -> float:
    pred_letter = parse_predicted_letter(predicted_text, num_options=n_options)
    return float(pred_letter == (gt_letter or "").strip().upper())


def is_valid_number(v) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(v)


def summarize_tensor(t: torch.Tensor) -> List[float]:
    """Summarize arbitrary tensor into compact scalar statistics."""
    x = t.detach().float().reshape(-1)
    if x.numel() == 0:
        return [0.0] * 5
    mean = x.mean().item()
    std = x.std(unbiased=False).item()
    vmax = x.max().item()
    vmin = x.min().item()
    l2 = torch.norm(x, p=2).item() / max(1, x.numel())
    return [mean, std, vmax, vmin, l2]


def mean_token_entropy(scores: List[torch.Tensor]) -> float:
    entropies = []
    for step_logits in scores:
        probs = torch.softmax(step_logits[0], dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()
        entropies.append(entropy)
    if not entropies:
        return float("nan")
    return float(sum(entropies) / len(entropies))


def mean_top1_conf(scores: List[torch.Tensor]) -> float:
    vals = []
    for step_logits in scores:
        probs = torch.softmax(step_logits[0], dim=-1)
        vals.append(torch.max(probs).item())
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def load_llava_med(model_path: str, device: torch.device):
    """
    Load LLaVA-Med model/tooling from the Microsoft codebase environment.
    """
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=str(device),
    )
    model.eval()
    return tokenizer, model, image_processor


def build_prompt(question: str, options: List[str], conv_template: str, default_image_token: str, conv_templates):
    conv = conv_templates[conv_template].copy()
    user_text = f"{default_image_token}\n{format_mc_user_text(question, options)}"
    conv.append_message(conv.roles[0], user_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def extract_features(args):
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model, image_processor = load_llava_med(args.model_path, device=device)

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(args.dataset, split=args.split)
    if args.n_samples is not None:
        samples = samples[: args.n_samples]

    features = []
    labels = []
    entropies = []
    records = []

    emb_layer = model.get_input_embeddings()

    with torch.no_grad():
        for sample in tqdm(samples, desc="Extracting selector features"):
            if not sample.mc_options or sample.mc_correct_index is None:
                continue

            image = Image.open(sample.image_path).convert("RGB")
            image_tensor = process_images([image], image_processor, model.config).to(device, dtype=torch.float16)
            prompt = build_prompt(
                question=sample.question,
                options=sample.mc_options,
                conv_template=args.conv_template,
                default_image_token=DEFAULT_IMAGE_TOKEN,
                conv_templates=conv_templates,
            )
            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(device)

            generation = model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

            gen_ids = generation.sequences[0]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred = clean_answer(pred)
            label = exact_match_letter(pred, sample.answer, n_options=len(sample.mc_options))

            # Build multimodal features from model forward.
            outputs = model(
                input_ids=input_ids,
                images=image_tensor,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

            # Text-only question embedding summary
            mc_text = format_mc_user_text(sample.question, sample.mc_options)
            q_ids = tokenizer(mc_text, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(device)
            q_embed = emb_layer(q_ids)

            # Answer token embedding summary from generated ids
            a_ids = gen_ids.unsqueeze(0).to(device)
            a_embed = emb_layer(a_ids)

            entropy = mean_token_entropy(generation.scores)
            top1_conf = mean_top1_conf(generation.scores)

            feat = []
            feat.extend(summarize_tensor(image_tensor))     # visual
            feat.extend(summarize_tensor(q_embed))          # question
            feat.extend(summarize_tensor(last_hidden))      # multimodal fusion proxy
            feat.extend(summarize_tensor(a_embed))          # answer representation
            feat.extend([entropy if is_valid_number(entropy) else 0.0, top1_conf, float(a_ids.shape[-1])])

            features.append(feat)
            labels.append(label)
            entropies.append(entropy if is_valid_number(entropy) else float("nan"))
            records.append(
                {
                    "image_path": sample.image_path,
                    "question": sample.question,
                    "ground_truth_letter": sample.answer,
                    "options": sample.mc_options,
                    "predicted_answer": pred,
                    "correct": int(label),
                    "entropy": entropy,
                }
            )

    if not features:
        raise ValueError("No features extracted. Check dataset split and annotations.")

    out_dir = Path(args.features_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.dataset}_{args.split}_{len(features)}"
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}_records.json"

    np.savez_compressed(
        npz_path,
        X=np.asarray(features, dtype=np.float32),
        y=np.asarray(labels, dtype=np.float32),
        entropy=np.asarray(entropies, dtype=np.float32),
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Extracted samples: {len(features)}")
    print(f"Saved features: {npz_path}")
    print(f"Saved records: {json_path}")


class SelectorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def split_indices(n: int, val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    idxs = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    n_val = int(n * val_fraction)
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    return train_idx, val_idx


def train_selector(args):
    data = np.load(args.features_file)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)

    train_idx, val_idx = split_indices(len(X), args.val_fraction, args.seed)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = SelectorMLP(in_dim=X.shape[1], hidden_dim=args.hidden_dim, dropout=args.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, y_val).item()

        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_loss": float(val_loss)})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    out_dir = Path(args.selector_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{Path(args.features_file).stem}_selector.pt"
    hist_path = out_dir / f"{Path(args.features_file).stem}_train_history.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": X.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "best_val_loss": best_val,
        },
        ckpt_path,
    )
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best val loss: {best_val:.6f}")
    print(f"Saved selector checkpoint: {ckpt_path}")
    print(f"Saved train history: {hist_path}")


def compute_curve(scores: np.ndarray, labels: np.ndarray, num_thresholds: int) -> List[Dict]:
    uniq = sorted(set(float(v) for v in scores.tolist()))
    if len(uniq) <= num_thresholds:
        thresholds = uniq
    else:
        idxs = [round(i * (len(uniq) - 1) / (num_thresholds - 1)) for i in range(num_thresholds)]
        thresholds = [uniq[i] for i in sorted(set(idxs))]

    rows = []
    n = len(scores)
    for thr in thresholds:
        answer_mask = scores >= thr
        answered = int(answer_mask.sum())
        if answered == 0:
            rows.append({"threshold": thr, "coverage": 0.0, "risk": 0.0, "answered": 0})
            continue
        answered_labels = labels[answer_mask]
        correct = float(answered_labels.sum())
        errors = answered - correct
        rows.append(
            {
                "threshold": thr,
                "coverage": answered / n,
                "risk": errors / answered,
                "answered": answered,
            }
        )
    return sorted(rows, key=lambda x: x["coverage"])


def aurc(curve: List[Dict]) -> float:
    if len(curve) < 2:
        return 0.0
    area = 0.0
    for a, b in zip(curve[:-1], curve[1:]):
        area += (b["coverage"] - a["coverage"]) * 0.5 * (a["risk"] + b["risk"])
    return area


def save_curve(curve: List[Dict], csv_path: Path, png_path: Path, title: str):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "coverage", "risk", "answered"])
        writer.writeheader()
        writer.writerows(curve)

    x = [r["coverage"] for r in curve]
    y = [r["risk"] for r in curve]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o", linewidth=2)
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=180)
    plt.close()


def selector_eval(args):
    data = np.load(args.features_file)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = data["y"].astype(np.float32)
    entropy = data["entropy"].astype(np.float32) if "entropy" in data.files else None

    ckpt = torch.load(args.selector_ckpt, map_location="cpu")
    model = SelectorMLP(
        in_dim=int(ckpt["in_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt["dropout"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        scores = model(X).numpy().astype(np.float32)

    curve_selector = compute_curve(scores=scores, labels=y, num_thresholds=args.num_thresholds)
    aurc_selector = aurc(curve_selector)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.features_file).stem

    save_curve(
        curve_selector,
        csv_path=out_dir / f"{stem}_selector_risk_coverage.csv",
        png_path=out_dir / f"{stem}_selector_risk_coverage.png",
        title=f"Selector Risk-Coverage ({stem})",
    )

    summary = {"aurc_selector": aurc_selector, "n_samples": int(len(y))}

    # Optional baseline comparison from entropy thresholding on same feature set.
    if entropy is not None and np.isfinite(entropy).any():
        entropy_scores = -entropy  # lower entropy => more confident, invert for thresholding
        curve_entropy = compute_curve(scores=entropy_scores, labels=y, num_thresholds=args.num_thresholds)
        aurc_entropy = aurc(curve_entropy)
        save_curve(
            curve_entropy,
            csv_path=out_dir / f"{stem}_entropy_risk_coverage.csv",
            png_path=out_dir / f"{stem}_entropy_risk_coverage.png",
            title=f"Entropy Baseline Risk-Coverage ({stem})",
        )
        summary["aurc_entropy_baseline"] = aurc_entropy

        # Combined comparison plot
        plt.figure(figsize=(7, 5))
        plt.plot(
            [r["coverage"] for r in curve_entropy],
            [r["risk"] for r in curve_entropy],
            label=f"Entropy baseline (AURC={aurc_entropy:.4f})",
            linewidth=2,
        )
        plt.plot(
            [r["coverage"] for r in curve_selector],
            [r["risk"] for r in curve_selector],
            label=f"Selector (AURC={aurc_selector:.4f})",
            linewidth=2,
        )
        plt.xlabel("Coverage")
        plt.ylabel("Risk")
        plt.title(f"Risk-Coverage Comparison ({stem})")
        plt.grid(alpha=0.25)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_selector_vs_entropy.png", dpi=180)
        plt.close()

    with open(out_dir / f"{stem}_selector_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Selector AURC: {aurc_selector:.6f}")
    if "aurc_entropy_baseline" in summary:
        print(f"Entropy baseline AURC: {summary['aurc_entropy_baseline']:.6f}")
    print(f"Saved outputs to: {out_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Multimodal selector workflow for reliable LLaVA-Med VQA")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="Extract multimodal features + labels")
    p_extract.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p_extract.add_argument("--data_root", type=str, default=get_data_root())
    p_extract.add_argument("--mc_data_root", type=str, default=get_mc_data_root())
    p_extract.add_argument("--dataset", type=str, required=True, choices=["pathvqa", "slake", "vqarad", "medvqa2019"])
    p_extract.add_argument("--split", type=str, default="validation")
    p_extract.add_argument("--features_dir", type=str, default=DEFAULT_FEATURES_DIR)
    p_extract.add_argument("--n_samples", type=int, default=None)
    p_extract.add_argument("--max_new_tokens", type=int, default=8)
    p_extract.add_argument("--conv_template", type=str, default="mistral_instruct")

    p_train = sub.add_parser("train", help="Train selector MLP regressor")
    p_train.add_argument("--features_file", type=str, required=True)
    p_train.add_argument("--selector_dir", type=str, default=DEFAULT_SELECTOR_DIR)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight_decay", type=float, default=1e-4)
    p_train.add_argument("--hidden_dim", type=int, default=64)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--val_fraction", type=float, default=0.2)
    p_train.add_argument("--seed", type=int, default=42)

    p_eval = sub.add_parser("eval", help="Evaluate selector with threshold sweep")
    p_eval.add_argument("--features_file", type=str, required=True)
    p_eval.add_argument("--selector_ckpt", type=str, required=True)
    p_eval.add_argument("--output_dir", type=str, default=get_selector_eval_dir())
    p_eval.add_argument("--num_thresholds", type=int, default=200)

    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "extract":
        extract_features(args)
    elif args.command == "train":
        train_selector(args)
    elif args.command == "eval":
        selector_eval(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
