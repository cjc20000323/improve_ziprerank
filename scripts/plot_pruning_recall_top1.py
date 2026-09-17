#!/usr/bin/env python3
"""Plot Recall@1/3/5-correct query counts from pruning analysis JSON.

This script is intentionally independent of model evaluation.  It reads the
``pruning_recall_query_groups.json`` produced by
``evaluate_pruning_recall_transitions.py`` and writes six PNG files: three line
charts and three grouped bar charts.  Each line chart contains:

1. Recall@K-correct query count at each exact pruning rate.
2. The distinct-query union across the current and all higher pruning rates.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pruning_recall_analysis import (  # noqa: E402
    RECALL_CUTOFFS,
    build_correct_query_curve,
    build_correct_query_higher_rate_breakdown,
)


DEFAULT_OUTPUT_FILENAMES = {
    1: "top1_correct_queries_by_pruning_rate.png",
    3: "recall3_correct_queries_by_pruning_rate.png",
    5: "recall5_correct_queries_by_pruning_rate.png",
}
DEFAULT_BREAKDOWN_OUTPUT_FILENAMES = {
    1: "recall1_correct_query_higher_rate_breakdown.png",
    3: "recall3_correct_query_higher_rate_breakdown.png",
    5: "recall5_correct_query_higher_rate_breakdown.png",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the standalone plotting arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot exact-rate and current-or-higher Recall@1/3/5-correct query "
            "counts plus higher-rate correctness breakdowns from "
            "pruning_recall_query_groups.json as six PNG files."
        )
    )
    parser.add_argument(
        "--analysis_json",
        type=str,
        required=True,
        help="Path to pruning_recall_query_groups.json.",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for all six PNG files. Defaults beside the input JSON.",
    )
    output_group.add_argument(
        "--output_png",
        type=str,
        default=None,
        help=(
            "Backward-compatible custom path for the Recall@1 PNG; Recall@3 and "
            "Recall@5 use their default filenames in the same directory."
        ),
    )
    return parser.parse_args(argv)


def load_analysis_json(path: Path) -> Dict[str, Any]:
    """Load and minimally validate a pruning analysis JSON file."""
    try:
        analysis = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Analysis JSON does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Analysis JSON is invalid: {path}: {exc}") from exc
    if not isinstance(analysis, dict):
        raise ValueError(f"Analysis JSON root must be an object: {path}")
    return analysis


def save_recall_correct_query_plot(
        analysis: Mapping[str, Any],
        k: int,
        output_path: Path,
) -> str:
    """Save the two requested Recall@K query-count curves as one PNG."""
    curve = build_correct_query_curve(analysis, k)
    points = curve.get("points")
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError(f"Recall@{k} correct-query curve contains no points")

    pruning_percentages = [int(point["pruning_percent"]) for point in points]
    exact_counts = [int(point["correct_at_pruning_rate"]) for point in points]
    union_counts = [
        int(point["correct_at_current_or_higher_pruning_union"])
        for point in points
    ]

    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        ticker = importlib.import_module("matplotlib.ticker")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to draw the pruning Recall@1 figure"
        ) from exc

    fig, axis = plt.subplots(figsize=(9.5, 6))
    axis.plot(
        pruning_percentages,
        exact_counts,
        color="#2563eb",
        marker="o",
        linewidth=2.2,
        markersize=6,
        label=f"Recall@{k} correct at exact pruning rate",
    )
    axis.plot(
        pruning_percentages,
        union_counts,
        color="#ea580c",
        marker="s",
        linewidth=2.2,
        markersize=6,
        label=(
            f"Distinct Recall@{k}-correct queries at current or higher rates"
        ),
    )
    axis.set_xlabel("Visual-token pruning rate (%)")
    axis.set_ylabel(f"Number of Recall@{k}-correct queries")
    axis.set_title(f"Recall@{k}-correct queries across pruning rates")
    axis.set_xticks(pruning_percentages)
    axis.set_xlim(min(pruning_percentages), max(pruning_percentages))
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    axis.legend()
    fig.tight_layout()

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return str(output_path)


def save_top1_correct_query_plot(
        analysis: Mapping[str, Any],
        output_path: Path,
) -> str:
    """Backward-compatible wrapper that saves only the Recall@1 figure."""
    return save_recall_correct_query_plot(analysis, 1, output_path)


def save_recall_higher_rate_breakdown_plot(
        analysis: Mapping[str, Any],
        k: int,
        output_path: Path,
) -> str:
    """Save the three-bar higher-pruning-rate breakdown for one Recall@K."""
    breakdown = build_correct_query_higher_rate_breakdown(analysis, k)
    points = breakdown.get("points")
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError(f"Recall@{k} higher-rate breakdown contains no points")

    pruning_percentages = [int(point["pruning_percent"]) for point in points]
    higher_wrong_counts = [
        int(point["current_correct_higher_rates_all_wrong"])
        for point in points
    ]
    higher_correct_counts = [
        int(point["current_correct_and_any_higher_rate_correct"])
        for point in points
    ]
    total_counts = [int(point["current_correct_total"]) for point in points]

    try:
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        plt = importlib.import_module("matplotlib.pyplot")
        ticker = importlib.import_module("matplotlib.ticker")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to draw the pruning Recall figures"
        ) from exc

    bar_width = 2.5
    fig, axis = plt.subplots(figsize=(11, 6.5))
    axis.bar(
        [value - bar_width for value in pruning_percentages],
        higher_wrong_counts,
        width=bar_width,
        color="#dc2626",
        label="Correct now; wrong at all higher pruning rates",
    )
    axis.bar(
        pruning_percentages,
        higher_correct_counts,
        width=bar_width,
        color="#16a34a",
        label="Correct now and at any higher pruning rate",
    )
    axis.bar(
        [value + bar_width for value in pruning_percentages],
        total_counts,
        width=bar_width,
        color="#2563eb",
        label="Total correct now",
    )
    axis.set_xlabel("Visual-token pruning rate (%)")
    axis.set_ylabel(f"Number of Recall@{k}-correct queries")
    axis.set_title(
        f"Recall@{k} correctness relative to higher pruning rates"
    )
    axis.set_xticks(pruning_percentages)
    axis.set_xlim(
        min(pruning_percentages) - 2 * bar_width,
        max(pruning_percentages) + 2 * bar_width,
    )
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axis.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    axis.legend()
    fig.tight_layout()

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    finally:
        plt.close(fig)
    return str(output_path)


def save_all_recall_correct_query_plots(
        analysis: Mapping[str, Any],
        output_dir: Path,
        top1_output_path: Optional[Path] = None,
) -> Dict[int, str]:
    """Save separate Recall@1, Recall@3, and Recall@5 figures."""
    output_dir = output_dir.expanduser().resolve()
    output_paths = {
        k: output_dir / DEFAULT_OUTPUT_FILENAMES[k]
        for k in RECALL_CUTOFFS
    }
    if top1_output_path is not None:
        output_paths[1] = top1_output_path.expanduser().resolve()
    return {
        k: save_recall_correct_query_plot(analysis, k, output_paths[k])
        for k in RECALL_CUTOFFS
    }


def save_all_recall_higher_rate_breakdown_plots(
        analysis: Mapping[str, Any],
        output_dir: Path,
) -> Dict[int, str]:
    """Save separate Recall@1, Recall@3, and Recall@5 grouped bar charts."""
    output_dir = output_dir.expanduser().resolve()
    return {
        k: save_recall_higher_rate_breakdown_plot(
            analysis,
            k,
            output_dir / DEFAULT_BREAKDOWN_OUTPUT_FILENAMES[k],
        )
        for k in RECALL_CUTOFFS
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Load one analysis JSON and draw all six standalone Recall figures."""
    args = parse_args(argv)
    analysis_path = Path(args.analysis_json).expanduser().resolve()
    top1_output_path = (
        Path(args.output_png).expanduser().resolve() if args.output_png else None
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            top1_output_path.parent
            if top1_output_path is not None
            else analysis_path.parent
        )
    )
    analysis = load_analysis_json(analysis_path)
    saved_paths = save_all_recall_correct_query_plots(
        analysis,
        output_dir,
        top1_output_path=top1_output_path,
    )
    breakdown_paths = save_all_recall_higher_rate_breakdown_plots(
        analysis,
        output_dir,
    )
    for k in RECALL_CUTOFFS:
        print(f"Saved Recall@{k} pruning line plot: {saved_paths[k]}")
        print(
            f"Saved Recall@{k} higher-rate breakdown plot: "
            f"{breakdown_paths[k]}"
        )


if __name__ == "__main__":
    main()
