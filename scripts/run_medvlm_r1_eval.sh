#!/usr/bin/env bash
# Off-the-shelf MedVLM-R1 MC evaluation on SLAKE, VQA-Rad, Med-VQA-2019.
# Usage:
#   ./scripts/run_medvlm_r1_eval.sh smoke   # 20 samples per dataset
#   ./scripts/run_medvlm_r1_eval.sh full    # full test splits

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-smoke}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

EXTRA=()
if [[ "$MODE" == "smoke" ]]; then
  EXTRA+=(--n_samples 20 --disable_wandb)
elif [[ "$MODE" != "full" ]]; then
  echo "Usage: $0 {smoke|full}" >&2
  exit 1
fi

for ds in slake vqarad medvqa2019; do
  echo "=== MedVLM-R1 eval: $ds ($MODE) ==="
  python 04_evaluation/inference_medvlm_r1.py \
    --dataset "$ds" \
    --split test \
    --model_path JZPeterPan/MedVLM-R1 \
    --max_new_tokens 1024 \
    --agreement_samples 5 \
    --attn_implementation eager \
    "${EXTRA[@]}"
done

echo "Done. Results under $(python -c 'from utils.paths import get_results_dir; print(get_results_dir())' 2>/dev/null || echo '06_results/')."
