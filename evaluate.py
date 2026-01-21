"""
Evaluation script for Medical VQA predictions
Computes accuracy and other metrics
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import re


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.
    Lowercase, remove punctuation, extra whitespace.
    """
    # Lowercase
    answer = answer.lower()
    
    # Remove punctuation
    answer = re.sub(r'[^\w\s]', '', answer)
    
    # Remove extra whitespace
    answer = ' '.join(answer.split())
    
    return answer


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer exactly matches ground truth"""
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def contains_answer(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer contains ground truth (or vice versa)"""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    return gt_norm in pred_norm or pred_norm in gt_norm


def evaluate_predictions(results: List[Dict]) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth.
    
    Args:
        results: List of result dictionaries with 'predicted_answer' and 'ground_truth_answer'
        
    Returns:
        Dictionary with evaluation metrics
    """
    exact_matches = 0
    contains_matches = 0
    total = 0
    
    for result in results:
        if result.get("ground_truth_answer") is None:
            continue
        
        predicted = result.get("predicted_answer", "")
        ground_truth = result.get("ground_truth_answer", "")
        
        if exact_match(predicted, ground_truth):
            exact_matches += 1
        
        if contains_answer(predicted, ground_truth):
            contains_matches += 1
        
        total += 1
    
    metrics = {
        "exact_match_accuracy": exact_matches / total if total > 0 else 0.0,
        "contains_accuracy": contains_matches / total if total > 0 else 0.0,
        "total_samples": total,
    }
    
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Medical VQA predictions")
    parser.add_argument("--results_file", type=str, required=True,
                       help="JSON file with inference results")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output file for evaluation metrics")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load results
    results_path = Path(args.results_file)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} results from {results_path}")
    
    # Evaluate
    metrics = evaluate_predictions(results)
    
    # Print metrics
    print("\n" + "=" * 60)
    print("Evaluation Metrics")
    print("=" * 60)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f} ({metrics['exact_match_accuracy']*100:.2f}%)")
    print(f"Contains Accuracy: {metrics['contains_accuracy']:.4f} ({metrics['contains_accuracy']*100:.2f}%)")
    print("=" * 60)
    
    # Save metrics
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
