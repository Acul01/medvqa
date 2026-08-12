"""Plot per-token entropy trajectories from MC inference results.

Reads ``entropy_tokens_all`` from a results CSV or JSON and can plot:

* the first ``--n_samples`` trajectories as a subplot grid (+ optional overlay)
* the dataset-mean trajectory (``--mean_trajectory``), absolute or relative

Example::

    python 04_evaluation/plot_token_entropy_trajectories.py \\
        --input results/llava_vqarad_af.csv \\
        --n_samples 10 \\
        --mean_trajectory
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot token-entropy trajectories from inference results (CSV/JSON)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to inference results (.csv or .json) containing entropy_tokens_all",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Number of per-sample trajectories from the start of the file (default: 10; "
        "use 0 with --mean_trajectory to skip per-sample plots)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for PNGs (default: <input_parent>/plots)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix (default: derived from input stem)",
    )
    parser.add_argument(
        "--no_overlay",
        action="store_true",
        help="Skip the per-sample overlay figure",
    )
    parser.add_argument(
        "--mean_trajectory",
        action="store_true",
        help="Also plot mean (±std) token-entropy trajectory over the full dataset",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="absolute",
        choices=["absolute", "relative"],
        help="Mean-trajectory x-axis: absolute token step, or relative position 0-100%% "
        "(default: absolute)",
    )
    parser.add_argument(
        "--relative_bins",
        type=int,
        default=100,
        help="Number of bins when --normalize relative (default: 100)",
    )
    parser.add_argument(
        "--no_split_correct",
        action="store_true",
        help="Do not split mean trajectory into correct vs incorrect (overall only)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="PNG DPI (default: 160)",
    )
    return parser.parse_args()


def load_rows(path: Path, n_samples: int | None) -> list[dict[str, Any]]:
    """Load result rows. ``n_samples=None`` loads the full file; otherwise first N rows."""
    if n_samples is not None and n_samples < 0:
        raise ValueError("--n_samples must be >= 0")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, nrows=n_samples if n_samples and n_samples > 0 else None)
        rows = df.to_dict(orient="records")
        if n_samples is not None and n_samples > 0:
            rows = rows[:n_samples]
    elif suffix == ".json":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "results" in data:
            data = data["results"]
        if not isinstance(data, list):
            raise ValueError(f"JSON must be a list of result rows: {path}")
        rows = data if n_samples is None or n_samples == 0 else data[:n_samples]
    else:
        raise ValueError(f"Unsupported input type {suffix!r}; use .csv or .json")

    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def _parse_entropy_tokens_all(raw: Any) -> dict[str, float]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        return {str(k): float(v) for k, v in json.loads(raw).items()}
    raise TypeError(f"Unexpected entropy_tokens_all type: {type(raw)}")


def answer_letter_entropy(toks: dict[str, float], letter: str | None) -> tuple[float | None, int | None]:
    """Return (entropy, 1-based step) of the first answer-letter token in generation order."""
    if not letter or (isinstance(letter, float) and math.isnan(letter)):
        return None, None
    letter = str(letter).strip().upper()
    if not letter:
        return None, None

    for i, (tok, ent) in enumerate(toks.items()):
        if tok.strip().upper() == letter:
            return max(0.0, float(ent)), i + 1
    for i, (tok, ent) in enumerate(toks.items()):
        cleaned = tok.strip().strip('"').strip("'").upper()
        if cleaned == letter:
            return max(0.0, float(ent)), i + 1
    return None, None


def _fmt_letter(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "?"
    s = str(value).strip().upper()
    return s if s else "?"


def _subplot_grid(n: int) -> tuple[int, int]:
    cols = min(5, n)
    rows = math.ceil(n / cols)
    return rows, cols


def _row_entropy_series(row: dict[str, Any]) -> list[float] | None:
    toks = _parse_entropy_tokens_all(row.get("entropy_tokens_all"))
    if not toks:
        return None
    return [max(0.0, float(v)) for v in toks.values()]


def _extract_series(
    rows: list[dict[str, Any]],
) -> tuple[list[list[float]], list[bool], list[float], list[int]]:
    """Return (series, is_correct, answer_entropies, answer_steps)."""
    series: list[list[float]] = []
    correct_flags: list[bool] = []
    ans_ents: list[float] = []
    ans_steps: list[int] = []
    for row in rows:
        ent = _row_entropy_series(row)
        if ent is None:
            continue
        series.append(ent)
        correct_flags.append(bool(row.get("is_correct")))
        toks = _parse_entropy_tokens_all(row.get("entropy_tokens_all"))
        pred = _fmt_letter(row.get("predicted_letter"))
        ans_ent, ans_step = answer_letter_entropy(toks, pred)
        if ans_ent is not None:
            ans_ents.append(ans_ent)
        if ans_step is not None:
            ans_steps.append(ans_step)
    return series, correct_flags, ans_ents, ans_steps


def mean_trajectory_absolute(
    series: list[list[float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Mean/std/count per absolute token step (1-based). Also returns median length."""
    if not series:
        raise ValueError("No series for mean trajectory")
    lengths = np.array([len(s) for s in series], dtype=int)
    max_len = int(lengths.max())
    median_len = float(np.median(lengths))
    stack = np.full((len(series), max_len), np.nan, dtype=float)
    for i, s in enumerate(series):
        stack[i, : len(s)] = s
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    count = np.sum(~np.isnan(stack), axis=0).astype(float)
    steps = np.arange(1, max_len + 1, dtype=float)
    return steps, mean, std, count, median_len


def mean_trajectory_relative(
    series: list[list[float]],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Interpolate each series onto relative positions in [0, 100], then mean/std."""
    if not series:
        raise ValueError("No series for mean trajectory")
    if n_bins < 2:
        raise ValueError("--relative_bins must be >= 2")
    x_rel = np.linspace(0.0, 100.0, n_bins)
    stacked = []
    for s in series:
        if len(s) == 1:
            stacked.append(np.full(n_bins, s[0], dtype=float))
            continue
        src_x = np.linspace(0.0, 100.0, len(s))
        stacked.append(np.interp(x_rel, src_x, s))
    arr = np.asarray(stacked, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    count = np.full(n_bins, float(len(series)))
    median_len = float(np.median([len(s) for s in series]))
    return x_rel, mean, std, count, median_len


def plot_subplots(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    n = len(rows)
    n_rows, n_cols = _subplot_grid(n)
    fig_w = max(3.4 * n_cols, 6)
    fig_h = max(3.4 * n_rows, 4)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharey=True)
    if n == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.flatten())

    for i, row in enumerate(rows):
        ax = axes_list[i]
        toks = _parse_entropy_tokens_all(row.get("entropy_tokens_all"))
        if not toks:
            ax.set_title(f"#{i + 1}: missing entropy_tokens_all", fontsize=8)
            ax.axis("off")
            continue

        tokens = list(toks.keys())
        entropies = [max(0.0, float(v)) for v in toks.values()]
        steps = list(range(1, len(entropies) + 1))

        pred = _fmt_letter(row.get("predicted_letter"))
        gt = _fmt_letter(row.get("ground_truth_letter"))
        correct = bool(row.get("is_correct"))
        color = "#2a9d8f" if correct else "#e76f51"
        ans_ent, ans_step = answer_letter_entropy(toks, pred)

        ax.plot(steps, entropies, color=color, lw=1.2)
        ax.fill_between(steps, entropies, alpha=0.15, color=color)
        if ans_step is not None and ans_ent is not None:
            ax.axvline(ans_step, color="#264653", ls="--", lw=0.9, alpha=0.7)
            ax.scatter([ans_step], [ans_ent], color="#264653", s=28, zorder=5)

        ans_str = f"{ans_ent:.4g}" if ans_ent is not None else "n/a"
        step_str = f" @step {ans_step}" if ans_step is not None else ""
        seq_mean = row.get("entropy_sequence_mean")
        if isinstance(seq_mean, (int, float)) and not math.isnan(float(seq_mean)):
            mean_str = f"{float(seq_mean):.3f}"
        else:
            mean_str = "n/a"

        ax.set_title(
            f"#{i + 1} pred={pred} gt={gt}\n"
            f"answer-token H({pred})={ans_str}{step_str}\n"
            f"seq-mean={mean_str}  n={len(entropies)}",
            fontsize=8.5,
        )
        ax.set_xlabel("token step", fontsize=8)
        if i % n_cols == 0:
            ax.set_ylabel("entropy", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

        top_idxs = sorted(range(len(entropies)), key=lambda j: entropies[j], reverse=True)[:2]
        for j in top_idxs:
            if entropies[j] > 0.2:
                label = tokens[j].replace("\n", "\\n")[:12]
                ax.annotate(
                    label,
                    (steps[j], entropies[j]),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=6,
                    color="#333",
                )

    for j in range(n, len(axes_list)):
        axes_list[j].axis("off")

    fig.suptitle(
        f"{title}\n"
        "green = correct, coral = incorrect; dashed line / marker = answer letter token",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_overlay(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, row in enumerate(rows):
        toks = _parse_entropy_tokens_all(row.get("entropy_tokens_all"))
        if not toks:
            continue
        entropies = [max(0.0, float(v)) for v in toks.values()]
        steps = list(range(1, len(entropies) + 1))
        pred = _fmt_letter(row.get("predicted_letter"))
        gt = _fmt_letter(row.get("ground_truth_letter"))
        correct = bool(row.get("is_correct"))
        color = "#2a9d8f" if correct else "#e76f51"
        ans_ent, ans_step = answer_letter_entropy(toks, pred)
        ans_str = f"{ans_ent:.3g}" if ans_ent is not None else "n/a"
        ax.plot(
            steps,
            entropies,
            color=color,
            alpha=0.75,
            lw=1.3,
            label=f"#{i + 1} {'✓' if correct else '✗'} {pred}/{gt}  H({pred})={ans_str}",
        )
        if ans_step is not None and ans_ent is not None:
            ax.scatter([ans_step], [ans_ent], color=color, s=36, edgecolors="#264653", zorder=5)

    ax.set_xlabel("token step")
    ax.set_ylabel("entropy")
    ax.set_title(f"{title}\nmarkers = answer letter token")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}")


def _plot_mean_curve(
    ax,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    count: np.ndarray,
    *,
    color: str,
    label: str,
    median_len: float | None,
    absolute: bool,
) -> None:
    n_total = int(count[0]) if len(count) else 0
    full_label = f"{label} (n={n_total})"

    if absolute and median_len is not None:
        before = x <= median_len
        after = x > median_len
        ax.plot(x[before], mean[before], color=color, lw=2.0, label=full_label)
        if after.any():
            # connect continuously: include last before-point
            idx = np.where(before)[0]
            start = int(idx[-1]) if len(idx) else 0
            ax.plot(x[start:], mean[start:], color=color, lw=2.0, alpha=0.35)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        ax.axvline(median_len, color="#6c757d", ls=":", lw=1.0, alpha=0.8)
    else:
        ax.plot(x, mean, color=color, lw=2.0, label=full_label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)


def plot_mean_trajectory(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    normalize: str,
    relative_bins: int,
    split_correct: bool,
    dpi: int,
) -> None:
    series, correct_flags, ans_ents, ans_steps = _extract_series(rows)
    if not series:
        raise ValueError("No usable entropy_tokens_all rows for mean trajectory")

    absolute = normalize == "absolute"
    compute = (
        (lambda s: mean_trajectory_absolute(s))
        if absolute
        else (lambda s: mean_trajectory_relative(s, relative_bins))
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 7.5),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )
    ax, ax_n = axes

    groups: list[tuple[str, list[list[float]], str]] = [
        ("overall", series, "#457b9d"),
    ]
    if split_correct:
        correct_series = [s for s, c in zip(series, correct_flags) if c]
        incorrect_series = [s for s, c in zip(series, correct_flags) if not c]
        if correct_series:
            groups.append(("correct", correct_series, "#2a9d8f"))
        if incorrect_series:
            groups.append(("incorrect", incorrect_series, "#e76f51"))

    median_len_overall = float(np.median([len(s) for s in series]))
    for name, group_series, color in groups:
        x, mean, std, count, _ = compute(group_series)
        _plot_mean_curve(
            ax,
            x,
            mean,
            std,
            count,
            color=color,
            label=name,
            median_len=median_len_overall if absolute and name == "overall" else None,
            absolute=absolute,
        )
        ax_n.plot(x, count, color=color, lw=1.5, label=name)

    if absolute and ans_steps:
        med_ans = float(np.median(ans_steps))
        ax.axvline(
            med_ans,
            color="#264653",
            ls="--",
            lw=1.2,
            alpha=0.85,
            label=f"median answer step ({med_ans:.0f})",
        )

    xlabel = "token step" if absolute else "relative position (% of sequence)"
    ax.set_ylabel("mean token entropy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")

    if ans_ents:
        mean_ans = float(np.mean(ans_ents))
        ax.text(
            0.01,
            0.98,
            f"mean answer-token H = {mean_ans:.4g}  (n={len(ans_ents)})",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#ccc"},
        )

    ax_n.set_xlabel(xlabel)
    ax_n.set_ylabel("# samples")
    ax_n.grid(True, alpha=0.3)
    if absolute:
        ax_n.axvline(median_len_overall, color="#6c757d", ls=":", lw=1.0, alpha=0.8)
        ax_n.text(
            0.99,
            0.95,
            f"median len = {median_len_overall:.0f}",
            transform=ax_n.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            color="#6c757d",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved {output_path}  (N={len(series)}, normalize={normalize})")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if args.n_samples < 0:
        raise ValueError("--n_samples must be >= 0")
    if args.n_samples == 0 and not args.mean_trajectory:
        raise ValueError("Nothing to plot: set --n_samples > 0 and/or --mean_trajectory")

    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent / "plots"
    prefix = args.prefix or input_path.stem

    if args.n_samples > 0:
        rows = load_rows(input_path, args.n_samples)
        missing = sum(1 for r in rows if not _parse_entropy_tokens_all(r.get("entropy_tokens_all")))
        if missing == len(rows):
            raise ValueError(
                f"No usable entropy_tokens_all in first {len(rows)} rows of {input_path}. "
                "Re-run inference with --use_reasoning_json so per-token entropies are saved."
            )
        if missing:
            print(f"Warning: {missing}/{len(rows)} rows missing entropy_tokens_all")

        title = f"{prefix} — token entropy trajectories (first {len(rows)} samples)"
        plot_subplots(
            rows,
            out_dir / f"{prefix}_entropy_traj_first{len(rows)}.png",
            title=title,
            dpi=args.dpi,
        )
        if not args.no_overlay:
            plot_overlay(
                rows,
                out_dir / f"{prefix}_entropy_traj_first{len(rows)}_overlay.png",
                title=f"{prefix} — token entropy overlay (first {len(rows)})",
                dpi=args.dpi,
            )

    if args.mean_trajectory:
        all_rows = load_rows(input_path, None)
        norm_tag = "abs" if args.normalize == "absolute" else f"rel{args.relative_bins}"
        plot_mean_trajectory(
            all_rows,
            out_dir / f"{prefix}_entropy_traj_mean_{norm_tag}.png",
            title=f"{prefix} — mean token entropy trajectory ({args.normalize})",
            normalize=args.normalize,
            relative_bins=args.relative_bins,
            split_correct=not args.no_split_correct,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
