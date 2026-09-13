#!/usr/bin/env python3
"""SHAP overlays for the 5 SLAKE Pseudo-SFT rescue samples.

For each sample writes into one folder:
  <id>/original.png
  <id>/answer_only.png + _heatmap.png
  <id>/zeroshot.png + _heatmap.png
  <id>/pseudo_sft.png + _heatmap.png
  <id>/meta.json
plus a root manifest.json.

Example:
  CUDA_VISIBLE_DEVICES=0 python -u 04_evaluation/run_slake_rescue_shap.py \\
      --config configs/shap_slake_rescue_compare.json --skip_existing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from find_pseudo_sft_rescues import normalize_image_key
from utils.load_data import MedicalVQALoader
from utils.mc_json_zeroshot import parse_mc_answer_rationale_json
from utils.paths import get_data_root, get_mc_data_root
from utils.qwen_vl import parse_mc_reasoning_json_output

SHAP_SCRIPT = Path(__file__).with_name("shap_visualize.py")
SLOT_ORDER = ("answer_only", "zeroshot", "pseudo_sft")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_question(text: str) -> str:
    return " ".join(str(text or "").split())


def match_sample(samples: list, spec: dict) -> tuple[int, object]:
    want_img = normalize_image_key(spec["image_path"])
    want_q = norm_question(spec["question"])
    hits = []
    for i, sample in enumerate(samples):
        if normalize_image_key(str(sample.image_path)) != want_img:
            continue
        if norm_question(sample.question) != want_q:
            continue
        hits.append((i, sample))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"No MC sample for {spec['id']}: {want_img} | {want_q}")
    raise SystemExit(f"Ambiguous match for {spec['id']}: {len(hits)} hits")


def slot_spec(cfg: dict, sample_cfg: dict, slot: str) -> dict:
    if slot == "pseudo_sft":
        key = sample_cfg["pseudo_model"]
        spec = dict(cfg["pseudo_models"][key])
        spec["slot"] = slot
        spec["pseudo_model_id"] = key
        return spec
    spec = dict(cfg["slots"][slot])
    spec["slot"] = slot
    return spec


def shap_out_dir(root: Path, spec: dict, dataset: str) -> Path:
    path = root / spec["shap_model"] / dataset / spec["experiment"] / spec["json_order"]
    if spec.get("run_tag"):
        path = path / spec["run_tag"]
    return path


def job_key(spec: dict) -> tuple:
    return (
        spec["shap_model"],
        spec["model_path"],
        spec["experiment"],
        spec["json_order"],
        spec.get("run_tag") or "",
    )


def extract_explanation(raw: str, experiment: str, json_order: str, num_options: int) -> str:
    if not raw or experiment == "answer_only" or json_order == "answer_only":
        return ""
    if experiment == "pseudo":
        parsed = parse_mc_reasoning_json_output(raw, num_options=num_options)
        return parsed.reasoning_text or ""
    parsed = parse_mc_answer_rationale_json(raw, num_options=num_options)
    return parsed.rationale_text or ""


def write_original(sample, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(sample.image_path).convert("RGB")
    image.save(dest)


def copy_shap_assets(src_dir: Path, idx: int, dest_dir: Path, slot: str) -> dict[str, str]:
    stem = f"sample_{idx:04d}"
    mapping = {
        f"{stem}.png": f"{slot}.png",
        f"{stem}_heatmap.png": f"{slot}_heatmap.png",
        f"{stem}.json": f"{slot}_shap.json",
    }
    written: dict[str, str] = {}
    for src_name, dest_name in mapping.items():
        src = src_dir / src_name
        if not src.exists():
            continue
        shutil.copy2(src, dest_dir / dest_name)
        written[dest_name] = dest_name
    return written


def build_meta(sample_cfg: dict, sample, idx: int, cfg: dict, dest_dir: Path) -> dict:
    options = list(sample.mc_options or [])
    gt_letter = sample_cfg.get("gt_letter") or (sample.answer or "")
    gt_answer = sample_cfg.get("gt_answer") or ""
    if not gt_answer and gt_letter and options:
        pos = ord(str(gt_letter)[0]) - 65
        if 0 <= pos < len(options):
            gt_answer = options[pos]
    meta = {
        "id": sample_cfg["id"],
        "index": idx,
        "image_path": str(sample.image_path),
        "original": "original.png",
        "question": sample.question,
        "question_type": sample.question_type,
        "options": options,
        "gt_letter": gt_letter,
        "gt_answer": gt_answer,
        "slots": {},
    }
    for slot in SLOT_ORDER:
        spec = slot_spec(cfg, sample_cfg, slot)
        shap_meta_path = dest_dir / f"{slot}_shap.json"
        shap_meta = load_json(shap_meta_path) if shap_meta_path.exists() else {}
        raw = str(shap_meta.get("predicted_raw") or "")
        pred_letter = shap_meta.get("predicted_letter") or ""
        pred_answer = shap_meta.get("predicted_text") or ""
        explanation = extract_explanation(raw, spec["experiment"], spec["json_order"], len(options))
        meta["slots"][slot] = {
            "label": spec.get("label"),
            "shap_model": spec["shap_model"],
            "model_path": spec["model_path"],
            "experiment": spec["experiment"],
            "json_order": spec["json_order"],
            "pseudo_model": spec.get("pseudo_model_id"),
            "pred_letter": pred_letter,
            "pred_answer": pred_answer,
            "answer": f"{pred_letter}) {pred_answer}".strip() if pred_letter else pred_answer,
            "explanation": explanation,
            "predicted_raw": raw,
            "correct": bool(pred_letter) and str(pred_letter).upper() == str(gt_letter).upper(),
            "overlay": f"{slot}.png",
            "heatmap": f"{slot}_heatmap.png",
        }
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHAP overlays for SLAKE rescue samples")
    parser.add_argument("--config", type=Path, default=Path("configs/shap_slake_rescue_compare.json"))
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--data_root", default=get_data_root())
    parser.add_argument("--mc_data_root", default=get_mc_data_root())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"CUDA_VISIBLE_DEVICES={args.gpu}", flush=True)

    cfg = load_json(args.config)
    out_root = Path(cfg["output_dir"])
    shap_scratch = out_root / "_shap_runs"
    out_root.mkdir(parents=True, exist_ok=True)

    loader = MedicalVQALoader(data_root=args.data_root, mc_data_root=args.mc_data_root)
    samples = loader.load_dataset(cfg["dataset"], split=cfg.get("split", "test"))
    samples = [s for s in samples if s.mc_options and s.mc_correct_index is not None]

    matched = []
    for spec in cfg["samples"]:
        idx, sample = match_sample(samples, spec)
        matched.append((spec, idx, sample))
        print(f"{spec['id']}: index={idx}  {sample.image_path}", flush=True)

    jobs: dict[tuple, dict] = {}
    for spec, idx, _sample in matched:
        for slot in SLOT_ORDER:
            slot_cfg = slot_spec(cfg, spec, slot)
            key = job_key(slot_cfg)
            bucket = jobs.setdefault(key, {"spec": slot_cfg, "indices": []})
            if idx not in bucket["indices"]:
                bucket["indices"].append(idx)

    for i, bucket in enumerate(jobs.values(), start=1):
        spec = bucket["spec"]
        indices = sorted(bucket["indices"])
        dests_ready = True
        for sample_cfg, idx, _sample in matched:
            if idx not in indices:
                continue
            case_dir = out_root / sample_cfg["id"]
            overlay = case_dir / f"{spec['slot']}.png"
            if not overlay.exists():
                dests_ready = False
                break
        print(f"\n=== [{i}/{len(jobs)}] {spec['slot']} {spec.get('label')} indices={indices} ===", flush=True)
        if args.skip_existing and dests_ready:
            print("skip existing overlays", flush=True)
            continue
        cmd = [
            sys.executable, "-u", str(SHAP_SCRIPT),
            "--model", spec["shap_model"],
            "--model_path", spec["model_path"],
            "--dataset", cfg["dataset"],
            "--split", cfg.get("split", "test"),
            "--experiment", spec["experiment"],
            "--json_order", spec["json_order"],
            "--sample_indices", ",".join(str(n) for n in indices),
            "--run_tag", spec.get("run_tag") or spec["slot"],
            "--max_evals", str(cfg.get("max_evals", 300)),
            "--output_dir", str(shap_scratch),
            "--dtype", args.dtype,
            "--data_root", args.data_root,
            "--mc_data_root", args.mc_data_root,
            "--shap_target", cfg.get("shap_target", "auto"),
        ]
        print(" ".join(cmd), flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    manifest_samples = []
    for spec, idx, sample in matched:
        case_dir = out_root / spec["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        original = case_dir / "original.png"
        if not original.exists() or not args.skip_existing:
            write_original(sample, original)
        if not args.dry_run:
            for slot in SLOT_ORDER:
                slot_cfg = slot_spec(cfg, spec, slot)
                src_dir = shap_out_dir(shap_scratch, slot_cfg, cfg["dataset"])
                copy_shap_assets(src_dir, idx, case_dir, slot)
            meta = build_meta(spec, sample, idx, cfg, case_dir)
            (case_dir / "meta.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            manifest_samples.append(meta)
            print(f"Wrote {case_dir}", flush=True)

    manifest = {
        "dataset": cfg["dataset"],
        "output_dir": str(out_root),
        "n_samples": len(manifest_samples),
        "samples": manifest_samples,
    }
    if not args.dry_run:
        (out_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Wrote {out_root / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
