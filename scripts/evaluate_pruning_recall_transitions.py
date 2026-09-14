#!/usr/bin/env python3
"""Run three visual-token pruning conditions and compare Recall@1 changes.

This is a standalone entry point built on top of ``scripts.evaluate``.  It loads
the model and MMDocIR inputs once, evaluates 10%, 50%, and 90% pruning with an
identical query set, and saves both transition lists and the existing token
similarity diagnostics for every condition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd
import torch
from transformers import AutoProcessor


# Allow both ``python scripts/...py`` and test-time ``scripts.<module>`` imports
# to resolve the repository's models, utilities, and original evaluation module.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen3vl_with_qi_early import Qwen3VLWithQIEarly  # noqa: E402
from scripts import evaluate as base_evaluate  # noqa: E402
from utils.pruning_recall_analysis import (
    PRUNING_TO_KEEP_RATIO,
    compare_pruning_recall1,
    save_pruning_recall1_analysis,
    write_jsonl,
)  # noqa: E402
from utils.similarity_analysis import (
    TokenSimilarityAnalysisCollector,
    ensure_matplotlib_available,
)  # noqa: E402


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the independent pruning-sweep entry point arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate page reranking at 10%, 50%, and 90% visual-token pruning, "
            "compare Recall@1 transitions, and save token-similarity distributions."
        )
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--first_stage_file", type=str, required=True)
    parser.add_argument(
        "--pages_parquet",
        type=str,
        default="MMDocIR/dataset/MMDocIR_pages.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pruning_recall1_analysis",
    )

    # All three conditions pass the same sampling controls into evaluate_mmdocir.
    # sample_size=0 evaluates every query; num_queries, when provided, deliberately
    # follows evaluate.py semantics and takes the first N instead.
    parser.add_argument("--num_queries", type=int, default=None)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--qi_early_temperature", type=float, default=0.1)

    # These options are intentionally identical to the previous similarity
    # analysis. Overall histograms use all eligible queries, while raw per-token
    # tensors and per-query plots use reservoir sampling within each Recall@1 group.
    parser.add_argument("--similarity_num_examples", type=int, default=5)
    parser.add_argument("--similarity_seed", type=int, default=42)
    parser.add_argument("--similarity_num_bins", type=int, default=80)
    parser.add_argument(
        "--similarity_comparison_num_ground_truth_candidates",
        type=int,
        default=0,
        help=(
            "Maximum GT candidates per query in all-query mean statistics; "
            "0 uses all GT candidates available in Top-K."
        ),
    )
    parser.add_argument(
        "--similarity_comparison_num_incorrect_candidates",
        type=int,
        default=3,
        help=(
            "Maximum top-ranked non-GT candidates per query in all-query mean "
            "statistics; 0 uses all non-GT candidates available in Top-K."
        ),
    )
    args = parser.parse_args(argv)

    if args.num_queries is not None and args.num_queries <= 0:
        parser.error("--num_queries must be positive when provided")
    if args.sample_size < 0:
        parser.error("--sample_size must be non-negative")
    if args.window_size <= 0 or args.stride <= 0:
        parser.error("--window_size and --stride must be positive")
    if args.similarity_num_examples <= 0:
        parser.error("--similarity_num_examples must be positive")
    if args.similarity_num_bins <= 1:
        parser.error("--similarity_num_bins must be greater than 1")
    if args.similarity_comparison_num_ground_truth_candidates < 0:
        parser.error(
            "--similarity_comparison_num_ground_truth_candidates must be "
            "non-negative"
        )
    if args.similarity_comparison_num_incorrect_candidates < 0:
        parser.error(
            "--similarity_comparison_num_incorrect_candidates must be non-negative"
        )
    return args


def _load_model_and_processor(
        model_path: str,
        temperature: float,
) -> Tuple[Qwen3VLWithQIEarly, Any]:
    """Load one QI-Early model that will be reused by all pruning conditions."""
    print(f"Loading processor and model from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = Qwen3VLWithQIEarly.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    # Transformers' generic from_pretrained typing does not expose custom model
    # members, so resolve the two QI-Early controls defensively at runtime.
    set_qi_early_enabled = getattr(model, "set_qi_early_enabled", None)
    pruner = getattr(model, "qi_early_pruner", None)
    if not callable(set_qi_early_enabled) or pruner is None:
        raise RuntimeError("Loaded model does not expose the QI-Early pruning API")
    cast(Callable[[bool], None], set_qi_early_enabled)(True)
    pruner.temperature = temperature
    return model, processor


def _run_pruning_condition(
        *,
        pruning_percent: int,
        keep_ratio: float,
        model: Qwen3VLWithQIEarly,
        processor: Any,
        first_stage_results: Dict[str, Any],
        parquet_df: pd.DataFrame,
        args: argparse.Namespace,
        output_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate one keep ratio and persist its ranking and similarity artifacts."""
    print()
    print("=" * 80)
    print(
        f"Pruning condition: remove {pruning_percent}% / "
        f"keep {keep_ratio:.0%} visual tokens"
    )
    print("=" * 80)

    # The model setter updates both the model attribute and the actual pruner.
    # Similarity collection is enabled only around inference so no stale tensor
    # from one condition can be consumed by the next condition.
    model.set_qi_early_keep_ratio(keep_ratio)
    pruner = model.qi_early_pruner
    pruner.temperature = args.qi_early_temperature
    collector = TokenSimilarityAnalysisCollector(
        num_examples=args.similarity_num_examples,
        seed=args.similarity_seed,
        num_bins=args.similarity_num_bins,
        comparison_num_ground_truth_candidates=(
            args.similarity_comparison_num_ground_truth_candidates
        ),
        comparison_num_incorrect_candidates=(
            args.similarity_comparison_num_incorrect_candidates
        ),
    )

    eval_stats = base_evaluate.EvalStats(use_logits=True)
    setattr(base_evaluate, "_eval_stats", eval_stats)
    pruner.set_collect_similarity_stats(True)
    try:
        results = base_evaluate.evaluate_mmdocir(
            model=model,
            processor=processor,
            first_stage_results=first_stage_results,
            parquet_df=parquet_df,
            mode="page",
            window_size=args.window_size,
            stride=args.stride,
            use_logits=True,
            num_queries=args.num_queries,
            sample_size=args.sample_size,
            seed=args.seed,
            log_file=None,
            similarity_collector=collector,
        )
    finally:
        pruner.set_collect_similarity_stats(False)

    # Save the complete ranking before plotting. If figure generation later fails,
    # the expensive inference results remain available for diagnosis or comparison.
    condition_name = f"prune{pruning_percent}"
    rerank_path = output_dir / f"rerank_results_{condition_name}.jsonl"
    write_jsonl(rerank_path, results)

    official_rerank_results = base_evaluate.convert_results_to_official_format(
        results,
        mode="page",
    )
    official_first_stage_results = base_evaluate.convert_first_stage_to_official_format(
        results,
        mode="page",
    )
    base_evaluate.compute_mmdocir_metrics(
        rerank_results=official_rerank_results,
        first_stage_results=official_first_stage_results,
        mode="page",
        model_name=f"ZipRerank-{condition_name}",
    )

    similarity_dir = output_dir / f"{condition_name}_similarity"

    # 三种剪枝率的相似度分析沿用与 evaluate.py 相同的图片恢复路径。读取器只在
    # collector.save() 处理最终抽中的样例时访问 parquet，因此不会为完整测试集
    # 额外解码图片，也不会改变任何候选的模型输入或重排结果。
    def load_similarity_image_binary(global_idx: int) -> bytes:
        row_index = int(global_idx)
        if not 0 <= row_index < len(parquet_df):
            raise IndexError(
                f"Candidate global_idx {row_index} is outside parquet row range "
                f"[0, {len(parquet_df)})"
            )
        return bytes(parquet_df.iloc[row_index]["image_binary"])

    similarity_paths = collector.save(
        output_dir=str(similarity_dir),
        keep_ratio=keep_ratio,
        image_binary_loader=load_similarity_image_binary,
    )
    eval_stats.total_queries = len(results)
    print(collector.report())
    print(eval_stats.report())

    artifacts = {
        "pruning_percent": pruning_percent,
        "keep_ratio": keep_ratio,
        "rerank_results": str(rerank_path),
        "similarity": similarity_paths,
        "selected_similarity_examples": {
            "recall1_correct": len(collector.correct_examples),
            "recall1_incorrect": len(collector.incorrect_examples),
        },
    }
    return results, artifacts


def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the three-condition experiment and return its transition summary."""
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Matplotlib is checked before model loading so a missing plotting dependency
    # cannot waste a long GPU evaluation. This is the same validation used by the
    # previous token-similarity analysis entry point.
    ensure_matplotlib_available()
    print(f"Loading page parquet: {args.pages_parquet}")
    parquet_df = pd.read_parquet(args.pages_parquet)
    print(f"Loading first-stage results: {args.first_stage_file}")
    first_stage_results = base_evaluate.load_first_stage_results(
        args.first_stage_file
    )

    model, processor = _load_model_and_processor(
        model_path=args.model_path,
        temperature=args.qi_early_temperature,
    )
    timer = base_evaluate.InferenceTimer(model)
    setattr(base_evaluate, "_inference_timer", timer)

    results_by_pruning: Dict[int, List[Dict[str, Any]]] = {}
    artifacts_by_condition: Dict[str, Any] = {}
    try:
        # Each evaluate_mmdocir call receives the same source dictionary, sampling
        # arguments, and seed. Its per-call seed reset therefore selects and orders
        # the same queries, which the comparison layer verifies again before output.
        for pruning_percent, keep_ratio in PRUNING_TO_KEEP_RATIO.items():
            results, artifacts = _run_pruning_condition(
                pruning_percent=pruning_percent,
                keep_ratio=keep_ratio,
                model=model,
                processor=processor,
                first_stage_results=first_stage_results,
                parquet_df=parquet_df,
                args=args,
                output_dir=output_dir,
            )
            results_by_pruning[pruning_percent] = results
            artifacts_by_condition[f"prune{pruning_percent}"] = artifacts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        analysis = compare_pruning_recall1(results_by_pruning)
        analysis["run_config"] = {
            "model_path": args.model_path,
            "first_stage_file": args.first_stage_file,
            "pages_parquet": args.pages_parquet,
            "num_queries": args.num_queries,
            "sample_size": args.sample_size,
            "seed": args.seed,
            "window_size": args.window_size,
            "stride": args.stride,
            "ranking_mode": "logits_single_token",
            "qi_early_temperature": args.qi_early_temperature,
            "similarity_num_examples_per_recall1_group": (
                args.similarity_num_examples
            ),
            "similarity_seed": args.similarity_seed,
            "similarity_num_bins": args.similarity_num_bins,
            "similarity_comparison_num_ground_truth_candidates": (
                args.similarity_comparison_num_ground_truth_candidates
            ),
            "similarity_comparison_num_incorrect_candidates": (
                args.similarity_comparison_num_incorrect_candidates
            ),
        }
        analysis["artifacts_by_condition"] = artifacts_by_condition
        transition_paths = save_pruning_recall1_analysis(analysis, output_dir)
        analysis["transition_artifacts"] = transition_paths
    finally:
        # InferenceTimer registers forward hooks. Removing them is important when
        # this entry point is invoked repeatedly from the same Python process.
        timer.remove_hooks()
        setattr(base_evaluate, "_inference_timer", None)
        setattr(base_evaluate, "_eval_stats", None)
        model.qi_early_pruner.set_collect_similarity_stats(False)

    return analysis


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI main entry point."""
    args = parse_args(argv)
    analysis = run(args)
    transitions = analysis["transitions"]

    print()
    print("=" * 80)
    print("Recall@1 pruning-transition analysis completed")
    print("=" * 80)
    print(f"Aligned samples: {analysis['num_aligned_samples']}")
    print(
        "Prune50 wrong -> Prune10 correct: "
        f"{transitions['prune50_wrong_prune10_correct']['count']}"
    )
    print(
        "Prune50 correct -> Prune90 wrong: "
        f"{transitions['prune50_correct_prune90_wrong']['count']}"
    )
    for name, path in analysis["transition_artifacts"].items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
