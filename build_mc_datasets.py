"""
Build multiple-choice variants of PathVQA, SLAKE, and VQA-Rad.

For each sample in each split, adds:
- options: GT answer + top-k distractors from train-answer vocabulary (BERTScore similarity)
- correct_index: index of GT inside shuffled options
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from bert_score import BERTScorer
from tqdm import tqdm

from utils.load_data import MedicalVQALoader, VQASample


DATASETS = ["pathvqa", "slake", "vqarad"]
CANONICAL_NAME = {
    "pathvqa": "PathVQA",
    "slake": "SLAKE",
    "vqarad": "VQA-Rad",
}


def normalize_answer(text: str, strip_punctuation: bool = True) -> str:
    """Normalize answers for dedup/comparison while keeping originals for output."""
    normalized = (text or "").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    if strip_punctuation:
        normalized = normalized.translate(str.maketrans("", "", string.punctuation))
        normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_train_vocab(
    train_samples: List[VQASample],
    strip_punctuation: bool = True,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Build normalized unique vocabulary from train answers.

    Returns:
      norm_to_canonical: normalized answer -> chosen canonical original string
      norm_to_originals: normalized answer -> all observed original strings
    """
    originals_by_norm: Dict[str, List[str]] = defaultdict(list)
    for sample in train_samples:
        original = str(sample.answer or "").strip()
        norm = normalize_answer(original, strip_punctuation=strip_punctuation)
        if not norm:
            continue
        originals_by_norm[norm].append(original)

    norm_to_canonical: Dict[str, str] = {}
    for norm, originals in originals_by_norm.items():
        # Canonical surface form = most frequent observed original answer.
        chosen = Counter(originals).most_common(1)[0][0]
        norm_to_canonical[norm] = chosen

    return norm_to_canonical, originals_by_norm


def bertscore_topk(
    scorer: BERTScorer,
    gt_text: str,
    candidates: List[str],
    top_k: int,
    batch_size: int,
) -> Tuple[List[int], List[float]]:
    """Return indices and scores for top-k candidates by BERTScore F1."""
    if not candidates or top_k <= 0:
        return [], []

    refs = [gt_text] * len(candidates)
    _, _, f1 = scorer.score(candidates, refs, batch_size=batch_size)
    scores = f1.tolist()
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = ranked_idx[:top_k]
    return keep, [scores[i] for i in keep]


class EmbeddingPreRetriever:
    """Optional semantic pre-retrieval to reduce BERTScore candidate set."""

    def __init__(self, model_name: str, candidate_norms: List[str], candidate_texts: List[str], device: str | None):
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError as exc:
            raise ImportError(
                "Pre-retrieval requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self._util = util
        self.model = SentenceTransformer(model_name, device=device)
        self.candidate_norms = candidate_norms
        self.candidate_embeddings = self.model.encode(
            candidate_texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def top_k_norms(self, query_text: str, top_k: int) -> List[str]:
        if top_k <= 0 or not self.candidate_norms:
            return []
        query_emb = self.model.encode(
            [query_text],
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        sims = self._util.cos_sim(query_emb, self.candidate_embeddings)[0]
        k = min(top_k, len(self.candidate_norms))
        top_idx = sims.topk(k=k).indices.tolist()
        return [self.candidate_norms[i] for i in top_idx]


def sample_fallback_distractors(
    all_candidate_norms: List[str],
    forbidden_norms: set[str],
    need: int,
    rng: random.Random,
) -> List[str]:
    """Fallback sampling if BERT ranking yields fewer than required distractors."""
    pool = [n for n in all_candidate_norms if n not in forbidden_norms]
    if not pool or need <= 0:
        return []
    if len(pool) <= need:
        return pool[:]
    return rng.sample(pool, need)


def infer_splits(loader: MedicalVQALoader, dataset_slug: str) -> List[str]:
    canonical = CANONICAL_NAME[dataset_slug]
    return loader.dataset_configs[canonical]["annotation_files"].keys()


def sample_to_record(sample: VQASample) -> Dict:
    record = {
        "dataset": sample.dataset,
        "split": sample.split,
        "image_path": sample.image_path,
        "question": sample.question,
        "answer": sample.answer,
    }
    if sample.metadata:
        for key, value in sample.metadata.items():
            if key not in record:
                record[key] = value
    return record


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_mc_for_dataset(
    loader: MedicalVQALoader,
    dataset_slug: str,
    output_root: Path,
    rng: random.Random,
    num_distractors: int,
    batch_size: int,
    scorer: BERTScorer,
    strip_punctuation: bool,
    use_pre_retrieval: bool,
    pre_retrieval_top_k: int,
    embedding_model: str,
    device: str | None,
) -> None:
    print(f"\n=== Dataset: {dataset_slug} ===")
    splits = list(infer_splits(loader, dataset_slug))
    if "train" not in splits:
        raise ValueError(f"{dataset_slug}: missing required train split")

    train_samples = loader.load_dataset(dataset_slug, "train")
    norm_to_canonical, _ = build_train_vocab(
        train_samples,
        strip_punctuation=strip_punctuation,
    )
    all_vocab_norms = list(norm_to_canonical.keys())
    print(f"Train vocab size (normalized): {len(all_vocab_norms)}")

    dataset_out = output_root / dataset_slug
    ensure_output_dir(dataset_out)

    pre_retriever = None
    if use_pre_retrieval:
        print(
            f"Initializing embedding pre-retrieval with model={embedding_model} "
            f"(top_k={pre_retrieval_top_k})"
        )
        try:
            pre_retriever = EmbeddingPreRetriever(
                model_name=embedding_model,
                candidate_norms=all_vocab_norms,
                candidate_texts=[norm_to_canonical[n] for n in all_vocab_norms],
                device=device,
            )
        except ImportError as exc:
            print(f"Warning: {exc}")
            print("Continuing without pre-retrieval for this dataset.")

    for split in splits:
        samples = loader.load_dataset(dataset_slug, split)
        output_records: List[Dict] = []

        fallback_count = 0
        hard_fallback_count = 0
        all_selected_scores: List[float] = []
        short_answer_count = 0

        # Rank once per unique normalized GT answer in this split.
        unique_gt_norm_to_original: Dict[str, str] = {}
        for sample in samples:
            gt_original = str(sample.answer or "").strip()
            gt_norm = normalize_answer(gt_original, strip_punctuation=strip_punctuation)
            if gt_norm and gt_norm not in unique_gt_norm_to_original:
                unique_gt_norm_to_original[gt_norm] = gt_original

        top_cache: Dict[str, Tuple[List[str], List[float]]] = {}
        ranking_iter = tqdm(
            unique_gt_norm_to_original.items(),
            total=len(unique_gt_norm_to_original),
            desc=f"{dataset_slug}/{split} ranking",
            unit="gt",
        )
        for gt_norm, gt_original in ranking_iter:
            candidate_norms = [n for n in all_vocab_norms if n != gt_norm]
            if pre_retriever:
                pre_norms = pre_retriever.top_k_norms(gt_original, pre_retrieval_top_k)
                pre_norms = [n for n in pre_norms if n != gt_norm]
                if pre_norms:
                    pre_set = set(pre_norms)
                    candidate_norms = [n for n in candidate_norms if n in pre_set]

            candidate_texts = [norm_to_canonical[n] for n in candidate_norms]
            idxs, scores = bertscore_topk(
                scorer=scorer,
                gt_text=gt_original,
                candidates=candidate_texts,
                top_k=num_distractors,
                batch_size=batch_size,
            )
            ranked_norms = [candidate_norms[i] for i in idxs]
            top_cache[gt_norm] = (ranked_norms, scores)

        sample_iter = tqdm(
            samples,
            total=len(samples),
            desc=f"{dataset_slug}/{split} building",
            unit="sample",
        )
        for sample in sample_iter:
            gt_original = str(sample.answer or "").strip()
            gt_norm = normalize_answer(gt_original, strip_punctuation=strip_punctuation)
            if len(gt_norm.split()) <= 1:
                short_answer_count += 1

            ranked_norms, ranked_scores = top_cache.get(gt_norm, ([], []))

            selected_norms = list(ranked_norms[:num_distractors])
            selected_scores = list(ranked_scores[:num_distractors])

            if len(selected_norms) < num_distractors:
                fallback_count += 1
                need = num_distractors - len(selected_norms)
                forbidden = set(selected_norms)
                forbidden.add(gt_norm)
                sampled = sample_fallback_distractors(
                    all_candidate_norms=all_vocab_norms,
                    forbidden_norms=forbidden,
                    need=need,
                    rng=rng,
                )
                selected_norms.extend(sampled)
                selected_scores.extend([0.0] * len(sampled))
                if len(selected_norms) < num_distractors:
                    hard_fallback_count += 1

            # Keep options duplicate-free by normalized form.
            option_texts = [gt_original]
            seen_norms = {gt_norm}
            for norm in selected_norms:
                if norm in seen_norms:
                    continue
                option_texts.append(norm_to_canonical[norm])
                seen_norms.add(norm)
                if len(option_texts) == num_distractors + 1:
                    break

            rng.shuffle(option_texts)
            correct_index = option_texts.index(gt_original)

            all_selected_scores.extend(selected_scores[:num_distractors])

            record = sample_to_record(sample)
            record["options"] = option_texts
            record["correct_index"] = correct_index
            output_records.append(record)

        # Validation checks
        total = len(output_records)
        gt_once_ok = 0
        unique_opts_ok = 0
        target_len_ok = 0
        for rec in output_records:
            opts = rec["options"]
            gt = rec["answer"]
            if len(opts) == (num_distractors + 1):
                target_len_ok += 1
            if opts.count(gt) == 1:
                gt_once_ok += 1
            if len(set(opts)) == len(opts):
                unique_opts_ok += 1

        mean_sim = sum(all_selected_scores) / len(all_selected_scores) if all_selected_scores else 0.0

        print(
            f"[{dataset_slug}/{split}] samples={total} "
            f"| exact_len={target_len_ok}/{total} "
            f"| gt_once={gt_once_ok}/{total} "
            f"| unique_opts={unique_opts_ok}/{total}"
        )
        print(
            f"[{dataset_slug}/{split}] fallback={fallback_count}, "
            f"hard_fallback={hard_fallback_count}, "
            f"mean_gt_to_distractor_sim={mean_sim:.4f}, "
            f"short_answers={short_answer_count}"
        )

        json_path = dataset_out / f"{dataset_slug}_{split}_mc.json"
        csv_path = dataset_out / f"{dataset_slug}_{split}_mc.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_records, f, ensure_ascii=False, indent=2)

        df = pd.DataFrame(output_records)
        if "options" in df.columns:
            df["options"] = df["options"].apply(lambda x: json.dumps(x, ensure_ascii=False))
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Saved: {json_path}")
        print(f"Saved: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MC datasets for PathVQA, SLAKE, and VQA-Rad."
    )
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_root", type=str, default="data_mc")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--num_distractors", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--bert_model_type", type=str, default="roberta-large")
    parser.add_argument("--num_layers", type=int, default=17)
    parser.add_argument("--device", type=str, default=None, help="Optional torch device for BERTScore.")
    parser.add_argument(
        "--disable_pre_retrieval",
        action="store_true",
        help="Disable optional embedding pre-retrieval stage.",
    )
    parser.add_argument(
        "--pre_retrieval_top_k",
        type=int,
        default=200,
        help="Candidate count kept by embedding pre-retrieval before BERTScore re-ranking.",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformer model for pre-retrieval.",
    )
    parser.add_argument(
        "--keep_punctuation_in_normalization",
        action="store_true",
        help="If set, punctuation is preserved during answer normalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_distractors < 1:
        raise ValueError("--num_distractors must be >= 1")

    loader = MedicalVQALoader(data_root=args.data_root)
    output_root = Path(args.output_root)
    ensure_output_dir(output_root)
    rng = random.Random(args.seed)

    print("Building MC datasets with settings:")
    print(f"  data_root={args.data_root}")
    print(f"  output_root={args.output_root}")
    print(f"  datasets={args.datasets}")
    print(f"  num_distractors={args.num_distractors}")
    print(f"  seed={args.seed}")
    print(f"  batch_size={args.batch_size}")
    print(f"  lang={args.lang}")
    print(f"  bert_model_type={args.bert_model_type}")
    print(f"  num_layers={args.num_layers}")
    print(f"  device={args.device}")
    print(f"  use_pre_retrieval={not args.disable_pre_retrieval}")
    print(f"  pre_retrieval_top_k={args.pre_retrieval_top_k}")
    print(f"  embedding_model={args.embedding_model}")
    print(f"  strip_punctuation={not args.keep_punctuation_in_normalization}")

    print("Initializing BERTScorer once...")
    scorer_kwargs = {
        "lang": args.lang,
        "model_type": args.bert_model_type,
        "num_layers": args.num_layers,
        "rescale_with_baseline": False,
    }
    if args.device:
        scorer_kwargs["device"] = args.device
    scorer = BERTScorer(**scorer_kwargs)

    for dataset_slug in args.datasets:
        build_mc_for_dataset(
            loader=loader,
            dataset_slug=dataset_slug,
            output_root=output_root,
            rng=rng,
            num_distractors=args.num_distractors,
            batch_size=args.batch_size,
            scorer=scorer,
            strip_punctuation=not args.keep_punctuation_in_normalization,
            use_pre_retrieval=not args.disable_pre_retrieval,
            pre_retrieval_top_k=args.pre_retrieval_top_k,
            embedding_model=args.embedding_model,
            device=args.device,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
