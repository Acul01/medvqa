# MedVQA

Training and evaluation code for medical visual question answering on **PathVQA**, **SLAKE**, **VQA-Rad**, and **Med-VQA-2019**.

The repository fine-tunes and evaluates **LLaVA 1.5**, **LLaVA-Med**, **Qwen2.5-VL**, and **Qwen3-VL**, and can score an off-the-shelf **MedVLM-R1** checkpoint. Questions are cast as multiple choice. Runs record answer accuracy together with uncertainty signals (token entropy, sample agreement, confidence) and support selective prediction via risk–coverage curves and AURC. A second set of experiments supervises short JSON rationales (region, observation, reasoning, answer letter), including rationales distilled from GPT-4o or Qwen3-VL.

This is the code reference for a master's thesis. Model weights, datasets, and result dumps are not in the repository.

## Layout

| Path | Role |
|------|------|
| `03_training/` | Fine-tuning for LLaVA, LLaVA-Med, and Qwen-VL, plus spatial-filter variants |
| `04_evaluation/` | Inference, metrics, risk–coverage, RCC plots, SHAP figures |
| `utils/` | Data loader, prompts, result I/O, Qwen helpers, SoM / MedSAM utilities |
| `configs/` | Figure configs for the reported RCC and SHAP tables |
| `scripts/` | SoM precomputation, prompt dump, MedVLM-R1 sweep |
| `patches/` | Small LLaVA-Med dtype patch |
| `iterative_calibration_ft_experiments/` | Side experiment on calibration-aware fine-tuning |

Run every command from the repository root so `utils` imports resolve.

## Setup

Python 3.10+ and a CUDA GPU are expected for the 7B/8B models.

```bash
pip install -r requirements.txt
```

Qwen3-VL needs a recent Transformers build (`transformers>=4.57`). If `Qwen3VLForConditionalGeneration` is missing:

```bash
pip install "git+https://github.com/huggingface/transformers" accelerate qwen-vl-utils
```

**LLaVA-Med** is not pip-installed from this repo. Clone [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) and install it in the same environment, then point `LLAVA_MED_HOME` at that checkout if it is not on `PYTHONPATH`. For bfloat16 training, apply the included projector-dtype patch inside that checkout:

```bash
patch -d "$LLAVA_MED_HOME" -p1 < patches/llava_med_encode_images_projector_dtype.patch
```

Paths default to directories next to the code. Override them when data or outputs live elsewhere:

```bash
export MEDVQA_DATA_ROOT=/path/to/02_data
export MEDVQA_MC_ROOT=/path/to/02_data/data_mc_openai
export MEDVQA_OUTPUTS_DIR=/path/to/05_outputs
export MEDVQA_RESULTS_DIR=/path/to/06_results
```

OpenAI calls (`build_mc_datasets_openai.py`, GPT pseudo-rationales) read `OPENAI_API_KEY` from the environment or a local `.env` file. Do not commit that file.

## Data

`MedicalVQALoader` (`utils/load_data.py`) expects this layout under the data root:

```text
02_data/PathVQA/{pathvqa_annotations,pathvqa_images}/
02_data/SLAKE/{slake_annotations,slake_images}/
02_data/VQA-Rad/{vqarad_annotations,vqarad_images}/
02_data/Med-VQA-2019/{medvqa2019_annotations,medvqa2019_images}/
```

SLAKE's validation split is named `validate`, not `validation`. VQA-Rad has no validation split in this loader.

```bash
python utils/download_datasets.py
```

The downloader pulls PathVQA, SLAKE, VQA-Rad, and Med-VQA-2019 (ImageCLEF VQA-Med 2019 via Hugging Face). Each dataset keeps its own license; this repository does not redistribute the images.

### Multiple choice and pseudo-rationales

Closed questions become `yes`/`no`. Open questions get four generated distractors plus the gold answer.

```bash
python utils/build_mc_datasets_openai.py \
  --datasets pathvqa slake vqarad medvqa2019 \
  --openai_model gpt-4o-mini \
  --resume
```

Pseudo-rationales are written onto those MC files, then checked by a pretrained Qwen2.5-VL that sees the rationale and the options but not the image:

```bash
python utils/build_pseudo_reasonings_openai.py \
  --datasets vqarad \
  --model gpt-4o-mini \
  --n_reasonings 5

python 04_evaluation/verify_pseudo_reasoning_qwen.py \
  --input_json 02_data/data_mc_openai/vqarad/vqarad_train_mc_openai_pseudo.json \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --resume
```

Use `--model qwen3-vl-8b` in the builder to distill from local Qwen3-VL instead of GPT-4o. Training with `--use_reasoning` keeps only rows that passed this check.

## Training

Loss is computed on the assistant span only (the answer letter, or the JSON rationale when `--use_reasoning` is set). `--both_json_orders` trains both rationale-first and answer-first JSON on the same checkpoint.

```bash
# Qwen2.5-VL, answer only
python 03_training/train_qwen.py \
  --dataset vqarad \
  --val_split test \
  --bf16 --gradient_checkpointing \
  --output_dir 05_outputs/qwen_vqarad

# Qwen3-VL, verified rationales, both JSON orders
python 03_training/train_qwen.py \
  --dataset vqarad \
  --model_path Qwen/Qwen3-VL-8B-Instruct \
  --mc_export_suffix _verify \
  --use_reasoning --both_json_orders \
  --max_length 2048 \
  --bf16 --gradient_checkpointing \
  --output_dir 05_outputs/qwen3_vqarad_both
```

The same MC flags exist on `03_training/train_llava.py` and `03_training/train_llava-med.py`. Defaults:

| Script | Base model |
|--------|------------|
| `train_llava.py` | `llava-hf/llava-1.5-7b-hf` |
| `train_llava-med.py` | `microsoft/llava-med-v1.5-mistral-7b` |
| `train_qwen.py` | `Qwen/Qwen2.5-VL-7B-Instruct` |

`train_qwen_modified.py` and `train_llava_modified.py` add a learnable spatial filter on visual tokens and save `spatial_filter.pt` next to the checkpoint. Pass `--use_lora` for LoRA and `--use_8bit_optimizer` when `bitsandbytes` is installed.

## Inference

Each inference script writes `{model}_{dataset}_{split tag}.json` plus CSV (and XLSX if `openpyxl` is installed) under the results directory, then a sibling `*_eval.json` with closed / open / overall accuracy and mean uncertainty.

```bash
python 04_evaluation/inference_qwen.py \
  --dataset vqarad \
  --split test \
  --model_path 05_outputs/qwen_vqarad \
  --finetuned \
  --results_dir 06_results
```

| Script | Use |
|--------|-----|
| `inference_qwen.py` | Fine-tuned or base Qwen2.5-VL / Qwen3-VL |
| `inference_llava.py` | Hugging Face LLaVA |
| `inference_llava-med.py` | LLaVA-Med |
| `inference_*_zeroshot_json.py` | Base model, JSON rationale, no task fine-tuning |
| `inference_medvlm_r1.py` | `JZPeterPan/MedVLM-R1` |

`--use_reasoning_json` (and `--answer_first` when the checkpoint was trained that way) scores explanation checkpoints. `--agreement_samples` (default 5) draws extra stochastic answers for agreement, confidence, and the scalar entropy. `--ablation_mode` supports `black`, `shuffle`, and FGR crops.

Dual-order checkpoints are evaluated twice and pooled:

```bash
python 04_evaluation/merge_rf_af_results.py \
  --rf 06_results/vqarad_rf.json \
  --af 06_results/vqarad_af.json \
  --output 06_results/vqarad_merged.json \
  --evaluate
```

Re-score an existing JSON without rerunning the model:

```bash
python 04_evaluation/evaluate.py --results_file 06_results/qwen2.5-vl_vqarad_all.json
```

## Risk–coverage and figures

```bash
python 04_evaluation/risk_coverage_entropy.py \
  --results_file 06_results/qwen2.5-vl_vqarad_all.json \
  --score_field entropy \
  --risk_metric match
```

`04_evaluation/plot_rcc_model_comparison.py` reads a JSON config and writes one risk–coverage figure per dataset and question type. The configs in `configs/` name the result files used for the thesis figures (`rcc_model_compare.json`, `rcc_experiment_specs.json`, `rcc_best_of_split.json`). Missing result files are skipped.

SHAP tables are driven by `04_evaluation/run_shap_table.py` and `04_evaluation/shap_visualize.py`, with sample lists in `configs/shap_*.json`.

## Models

| Model | Hugging Face id |
|-------|-----------------|
| LLaVA 1.5 7B | `llava-hf/llava-1.5-7b-hf` |
| LLaVA-Med | `microsoft/llava-med-v1.5-mistral-7b` |
| Qwen2.5-VL 7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Qwen3-VL 8B | `Qwen/Qwen3-VL-8B-Instruct` |
| MedVLM-R1 | `JZPeterPan/MedVLM-R1` |

## License

Code in this repository is released under the [MIT License](LICENSE). Model weights and datasets follow the licenses of their original authors.
