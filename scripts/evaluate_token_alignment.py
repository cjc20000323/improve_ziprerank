"""Inspect which QI text token maximizes similarity for each visual token.

This is a standalone diagnostic entry point.  It reuses stable loading and
reranking functions from ``scripts/evaluate.py`` but does not modify that file.
By default it randomly selects 20 page-mode queries whose first-stage Top-K
contains both a ground-truth and a non-ground-truth candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, cast

import pandas as pd
import torch
from transformers import AutoProcessor


# Make imports independent of the shell's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.qwen3vl_with_qi_early_token_alignment import (  # noqa: E402
    Qwen3VLWithQIEarlyTokenAlignment,
)
from scripts import evaluate as base_evaluate  # noqa: E402
from utils.token_alignment_analysis import (  # noqa: E402
    TokenAlignmentCollector,
    select_eligible_query_subset,
)


def parse_args() -> argparse.Namespace:
    # 参数尽量与常规 evaluate.py 保持一致，使诊断运行可以复用相同模型、候选集
    # 和重排方式；这里只额外提供样本筛选数量与对齐 JSON 的输出位置。
    parser = argparse.ArgumentParser(
        description=(
            "Rerank 20 MMDocIR queries and export, for every visual token in "
            "the best GT and best non-GT candidate, its maximum-similarity "
            "QI text token."
        )
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--first_stage_file", required=True)
    parser.add_argument(
        "--pages_parquet",
        default="MMDocIR/dataset/MMDocIR_pages.parquet",
    )
    parser.add_argument(
        "--output_file",
        default="outputs/token_alignment_analysis/token_alignment_20_queries.json",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=20,
        help="Number of eligible queries to sample uniformly (default: 20).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help=(
            "Must be at least the first-stage candidate count so each query "
            "uses exactly one model forward window."
        ),
    )
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument(
        "--use_logits",
        action="store_true",
        help="Use the faster first-token-logit reranking path.",
    )
    parser.add_argument(
        "--qi_early_keep_ratio",
        type=float,
        default=0.5,
        help="Visual-token retention ratio (default: 0.5).",
    )
    parser.add_argument("--qi_early_temperature", type=float, default=0.1)
    parser.add_argument(
        "--qi_early_text_mode",
        choices=["query_only", "all_before_image"],
        default=None,
        help=(
            "Optional override. If omitted, use the mode stored in the model "
            "configuration. The JSON always records the effective mode."
        ),
    )
    parser.add_argument(
        "--llm_log_file",
        default=None,
        help="Optional path for the same per-query LLM log as normal evaluation.",
    )
    return parser.parse_args()


def _consume_qi_token_alignment_stats(
    model,
    candidate_original_indices: Sequence[int],
) -> Sequence[Dict[str, Any]]:
    """Copy diagnostic GPU tensors to CPU immediately after one forward.

    ``base_evaluate.rerank_window_with_images`` already has one well-defined
    place where its standard similarity records are consumed.  The standalone
    script temporarily substitutes this richer consumer at runtime, leaving
    the source code and behavior of normal ``evaluate.py`` unchanged.
    """
    # noinspection PyProtectedMember
    pruner = base_evaluate._find_qi_early_pruner(model)

    # 这四份缓存来自同一次 forward：raw_stats 按候选图像顺序排列，后三项则是
    # 所有候选共享的查询文本坐标系。复制前先整体取出，避免清缓存时丢失引用。
    raw_stats = pruner.last_similarity_stats
    query_token_ids = getattr(pruner, "last_query_token_ids", None)
    query_input_positions = getattr(pruner, "last_query_input_positions", None)
    text_extraction_mode = getattr(pruner, "last_text_extraction_mode", None)

    # Drop model-owned GPU references early. Local references remain valid long
    # enough for the explicit CPU copy below and are released when this returns.
    pruner.last_similarity_stats = None
    pruner.last_query_token_ids = None
    pruner.last_query_input_positions = None
    pruner.last_text_extraction_mode = None

    if raw_stats is None:
        raise RuntimeError("QI-Early did not expose token-alignment statistics")
    if query_token_ids is None or query_input_positions is None:
        raise RuntimeError("QI-Early did not expose the text-token/input mapping")
    if text_extraction_mode is None:
        raise RuntimeError("QI-Early did not expose its text extraction mode")
    if len(raw_stats) != len(candidate_original_indices):
        raise RuntimeError(
            "QI-Early alignment/image count mismatch: "
            f"stats={len(raw_stats)}, images={len(candidate_original_indices)}"
        )

    if not isinstance(query_token_ids, torch.Tensor):
        raise TypeError("last_query_token_ids must be a torch.Tensor")
    if not isinstance(query_input_positions, torch.Tensor):
        raise TypeError("last_query_input_positions must be a torch.Tensor")

    # 查询元数据只复制一次到 CPU，随后由各候选记录共享。contiguous() 让后续
    # 序列化和相等性检查不依赖原 tensor 的切片步长或模型所在设备。
    shared_query_token_ids = query_token_ids.detach().to(
        device="cpu", dtype=torch.long
    ).contiguous()
    shared_query_input_positions = query_input_positions.detach().to(
        device="cpu", dtype=torch.long
    ).contiguous()

    copied_stats = []

    # candidate_original_indices 保存当前窗口图像在一阶段候选列表中的位置。
    # zip 后，每份视觉统计就获得稳定的 candidate_pos，可在最终排序完成后回接
    # 到正确的页面；逐视觉 token 的三个向量仍保持完全相同的原始 patch 顺序。
    for candidate_pos, image_stats in zip(candidate_original_indices, raw_stats):
        if "max_text_token_indices" not in image_stats:
            raise RuntimeError(
                "Missing per-visual-token text argmax. Ensure the diagnostic "
                "Qwen3VLWithQIEarlyTokenAlignment class is loaded."
            )
        copied_stats.append({
            "candidate_pos": int(candidate_pos),
            "scores": image_stats["scores"].detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous(),
            "kept_mask": image_stats["kept_mask"].detach().to(
                device="cpu", dtype=torch.bool
            ).contiguous(),
            "max_text_token_indices": image_stats[
                "max_text_token_indices"
            ].detach().to(device="cpu", dtype=torch.long).contiguous(),
            "threshold": float(
                image_stats["threshold"].detach().float().cpu().item()
            ),
            "num_original_tokens": int(image_stats["num_original_tokens"]),
            "num_kept_tokens": int(image_stats["num_kept_tokens"]),
            # These tensors are shared by all candidates in the current query;
            # they are not cloned twenty times in CPU memory.
            "query_token_ids": shared_query_token_ids,
            "query_input_positions": shared_query_input_positions,
            "text_extraction_mode": str(text_extraction_mode),
        })
    return copied_stats


def _open_log(path: Optional[str], args: argparse.Namespace):
    if path is None:
        return None
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    handle.write("=" * 80 + "\n")
    handle.write("TOKEN ALIGNMENT DIAGNOSTIC - ZipRerank\n")
    handle.write("=" * 80 + "\n")
    handle.write(f"Model: {args.model_path}\n")
    handle.write(f"Queries: {args.num_queries}\n")
    handle.write(f"Keep ratio: {args.qi_early_keep_ratio}\n")
    handle.flush()
    return handle


def main() -> None:
    args = parse_args()
    if args.num_queries <= 0:
        raise ValueError("--num_queries must be positive")
    if not 0.0 < args.qi_early_keep_ratio <= 1.0:
        raise ValueError("--qi_early_keep_ratio must be in (0, 1]")

    print("Loading MMDocIR page parquet...")
    parquet_df = pd.read_parquet(args.pages_parquet)
    print(f"Loaded {len(parquet_df)} pages")

    print("Loading first-stage retrieval results...")
    first_stage_results = base_evaluate.load_first_stage_results(
        args.first_stage_file
    )

    # 先过滤再随机抽样，保证每个入选 query 都能导出一个最高排名 GT 和一个
    # 最高排名非 GT；使用独立 seed 后，同一输入文件上的诊断样本可重复获得。
    selected_results, selection_metadata = select_eligible_query_subset(
        first_stage_results,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    max_candidates = max(
        len(item["top_k_global_indices"]) for item in selected_results.values()
    )

    # 对齐缓存只代表最近一次模型 forward，因此一个 query 必须完整放入同一窗口。
    # 若发生滑窗，多个 forward 会覆盖 query 元数据，候选与 argmax 将无法可靠配对。
    if max_candidates > args.window_size:
        raise ValueError(
            "Token alignment requires one reranking window per query, but the "
            f"selected queries contain up to {max_candidates} candidates and "
            f"--window_size={args.window_size}. Increase --window_size."
    )
    print(
        f"Found {len(selected_results)} eligible queries after inspecting "
        f"{selection_metadata['queries_inspected_before_early_stop']} entries "
        f"(seed={args.seed})"
    )

    print(f"Loading diagnostic model from {args.model_path}...")

    # 诊断子类沿用原检查点的全部参数，仅把无参数的 QI-Early pruner 换成会额外
    # 保存 argmax 的版本，所以模型打分和视觉 token 选择规则与正常评估一致。
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model = Qwen3VLWithQIEarlyTokenAlignment.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.set_qi_early_keep_ratio(args.qi_early_keep_ratio)
    model.qi_early_pruner.temperature = args.qi_early_temperature
    if args.qi_early_text_mode is not None:
        model.set_qi_early_text_mode(args.qi_early_text_mode)
    model.qi_early_pruner.set_collect_similarity_stats(True)
    model.eval()

    # TokenAlignmentCollector 实现了 evaluate_mmdocir 所需的 add_query 接口。
    # 初始化基础评估模块的两个全局统计对象后，可以直接复用完整的推理与排序链路。
    collector = TokenAlignmentCollector(processor.tokenizer)
    base_evaluate._inference_timer = base_evaluate.InferenceTimer(model)
    base_evaluate._eval_stats = base_evaluate.EvalStats(
        use_logits=args.use_logits
    )

    log_file = _open_log(args.llm_log_file, args)
    # noinspection PyProtectedMember
    original_consumer = base_evaluate._consume_qi_similarity_stats
    try:
        # Runtime substitution is local to this process. No line in the normal
        # evaluation source file is edited, and the original function is
        # restored even when evaluation raises an exception.
        base_evaluate._consume_qi_similarity_stats = (
            _consume_qi_token_alignment_stats
        )
        results = base_evaluate.evaluate_mmdocir(
            model=model,
            processor=processor,
            first_stage_results=selected_results,
            parquet_df=parquet_df,
            mode="page",
            window_size=args.window_size,
            stride=args.stride,
            use_logits=args.use_logits,
            num_queries=None,
            sample_size=0,
            seed=args.seed,
            log_file=log_file,
            similarity_collector=cast(Any, collector),
        )
    finally:
        base_evaluate._consume_qi_similarity_stats = original_consumer
        if log_file is not None:
            log_file.close()

    # 运行参数与抽样条件和逐 token 结果写在同一个 JSON 中，便于之后判断两次
    # 分析是否使用了相同配置，而不必依赖终端输出或另存的日志文件。
    run_metadata = {
        "model_path": args.model_path,
        "first_stage_file": args.first_stage_file,
        "pages_parquet": args.pages_parquet,
        "requested_queries": args.num_queries,
        "evaluated_queries": len(results),
        "window_size": args.window_size,
        "reranking_mode": "first-token logits" if args.use_logits else "generation",
        "qi_early_keep_ratio": args.qi_early_keep_ratio,
        "qi_early_temperature": args.qi_early_temperature,
        "effective_qi_early_text_mode": model.qi_early_text_mode,
        **selection_metadata,
    }
    output_path = collector.save(args.output_file, run_metadata=run_metadata)

    # collector 对不满足条件的 query 会选择跳过。脚本要求精确导出指定数量，
    # 因此先保存可诊断的部分结果，再用异常明确提示本次运行并未完整成功。
    if len(collector.query_records) != args.num_queries:
        raise RuntimeError(
            f"Expected {args.num_queries} exported queries, got "
            f"{len(collector.query_records)}; partial output was saved to {output_path}"
        )

    print(json.dumps({
        "output_file": str(output_path),
        "queries_exported": len(collector.query_records),
        "skipped_queries": dict(collector.skipped_counts),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
