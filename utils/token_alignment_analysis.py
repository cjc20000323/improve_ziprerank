"""Export visual-token to text-token maximum-similarity alignments.

The normal reranker only needs the maximum similarity value for each visual
token.  This diagnostic module additionally consumes the *argmax text-token
index* recorded by the diagnostic QI-Early model and converts it into readable
tokenizer information.

The JSON intentionally contains both:

* ``vocab_token``: the exact token string stored in the model vocabulary;
* ``decoded_text``: the result of decoding that single token ID.

These are tokenizer subword units rather than necessarily complete natural
language words.  Keeping both representations makes whitespace markers and
special tokens visible while still providing a human-readable form.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

import torch


def select_eligible_query_subset(
    first_stage_results: Mapping[Any, Mapping[str, Any]],
    num_queries: int = 20,
    seed: int = 42,
) -> Tuple[Dict[Any, Mapping[str, Any]], Dict[str, Any]]:
    """Randomly select queries whose Top-K contains both a GT and a non-GT.

    A correct/incorrect candidate pair cannot be constructed when first-stage
    retrieval misses every ground-truth page.  Filtering before model inference
    guarantees that the requested number of exported queries can each contain
    the two requested candidates.

    Query keys are shuffled with a local seeded generator. Eligibility is then
    checked in that order and stops immediately after ``num_queries`` matches,
    avoiding a complete eligibility scan when enough examples are found early.
    """
    if num_queries <= 0:
        raise ValueError(f"num_queries must be positive, got {num_queries}")

    eligible_keys: List[Any] = []
    skipped: DefaultDict[str, int] = defaultdict(int)
    candidate_keys = list(first_stage_results.keys())
    random.Random(seed).shuffle(candidate_keys)
    inspected_queries = 0

    # page_id 是文档内部的局部页号，而第一阶段结果保存的是数据集全局索引。
    # 先用 start_idx 转到同一坐标系，才能正确判断 Top-K 中是否同时存在正负例。
    # This randomized traversal is equivalent to uniform sampling without
    # replacement from the eligible set, but it can stop before scanning all data.
    for key in candidate_keys:
        inspected_queries += 1
        item = first_stage_results[key]
        gt_page_ids = {int(page_id) for page_id in item.get("page_id", [])}
        top_k_global_indices = list(item.get("top_k_global_indices", []))

        if not gt_page_ids:
            skipped["no_ground_truth_page"] += 1
            continue
        if not top_k_global_indices:
            skipped["empty_first_stage_candidates"] += 1
            continue

        start_idx = int(item["start_idx"])
        end_idx = int(item["end_idx"])
        local_page_ids = [int(global_idx) - start_idx for global_idx in top_k_global_indices]

        if any(not 0 <= page_id <= end_idx - start_idx for page_id in local_page_ids):
            skipped["candidate_outside_document_range"] += 1
            continue
        if not any(page_id in gt_page_ids for page_id in local_page_ids):
            skipped["correct_candidate_missing_from_topk"] += 1
            continue
        if not any(page_id not in gt_page_ids for page_id in local_page_ids):
            skipped["incorrect_candidate_missing_from_topk"] += 1
            continue

        eligible_keys.append(key)
        if len(eligible_keys) == num_queries:
            break

    # 在推理开始前一次性验证可用样本数，避免模型已经运行很久后才发现无法凑齐
    # 指定数量；局部 Random 实例不会改变调用方可能依赖的全局随机状态。
    if len(eligible_keys) < num_queries:
        raise ValueError(
            f"Requested {num_queries} analyzable queries, but only "
            f"{len(eligible_keys)} contain both GT and non-GT candidates"
        )

    selected = {key: first_stage_results[key] for key in eligible_keys}
    selection_metadata = {
        "selection_method": (
            "seeded random query permutation; first N eligible queries with early stop"
        ),
        "selection_seed": seed,
        "total_first_stage_queries": len(first_stage_results),
        "queries_inspected_before_early_stop": inspected_queries,
        "queries_not_inspected": len(first_stage_results) - inspected_queries,
        "eligible_queries_encountered": len(eligible_keys),
        "selected_queries": len(selected),
        "required_correct_candidates": 1,
        "required_incorrect_candidates": 1,
        "prefilter_skipped_queries": dict(sorted(skipped.items())),
        "eligibility_rule": (
            "First-stage Top-K must contain at least one ground-truth page and "
            "at least one non-ground-truth page."
        ),
    }
    return selected, selection_metadata


class TokenAlignmentCollector:
    """Collect two candidates per query and serialize token-level alignments."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.query_records: List[Dict[str, Any]] = []
        self.total_queries_seen = 0
        self.skipped_counts: DefaultDict[str, int] = defaultdict(int)
        self._special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

    def _describe_text_token(
        self,
        query_sequence_index: int,
        input_sequence_position: int,
        token_id: int,
    ) -> Dict[str, Any]:
        """Return both the exact vocabulary piece and its one-token decoding."""

        # vocabulary piece 忠实保留分词器内部的子词和空格标记，decoded_text 则
        # 更适合人工阅读。二者同时输出，避免仅解码后无法区分特殊 token 或边界。
        vocab_token = self.tokenizer.convert_ids_to_tokens(token_id)
        decoded_text = self.tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return {
            "query_sequence_index": query_sequence_index,
            "input_sequence_position": input_sequence_position,
            "token_id": token_id,
            "vocab_token": None if vocab_token is None else str(vocab_token),
            "decoded_text": decoded_text,
            "is_special_token": token_id in self._special_token_ids,
        }

    @staticmethod
    def _as_cpu_1d_tensor(value: Any, dtype: torch.dtype, name: str) -> torch.Tensor:
        # collector 只做诊断汇总，不应继续持有计算图或 GPU 内存。这里也统一形状
        # 和 dtype，使后续长度、索引范围及序列化检查具有确定的行为。
        tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=dtype).contiguous()
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape={tuple(tensor.shape)}")
        return tensor

    def _validate_shared_query_metadata(
        self,
        candidate_stats: Sequence[Mapping[str, Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Verify every image in a window used the same QI text-token sequence."""

        # 一次多图 forward 只构造一份查询 embedding，因此首个候选的元数据可以
        # 作为基准；其余候选必须逐项一致，否则不能共用同一个 argmax 下标空间。
        first = candidate_stats[0]
        query_token_ids = self._as_cpu_1d_tensor(
            first["query_token_ids"], torch.long, "query_token_ids"
        )
        query_input_positions = self._as_cpu_1d_tensor(
            first["query_input_positions"], torch.long, "query_input_positions"
        )
        text_extraction_mode = str(first["text_extraction_mode"])

        if query_token_ids.numel() == 0:
            raise ValueError("QI-Early text-token sequence must not be empty")
        if query_token_ids.numel() != query_input_positions.numel():
            raise ValueError(
                "query_token_ids/query_input_positions length mismatch: "
                f"{query_token_ids.numel()} vs {query_input_positions.numel()}"
            )

        for stats in candidate_stats[1:]:
            other_ids = self._as_cpu_1d_tensor(
                stats["query_token_ids"], torch.long, "query_token_ids"
            )
            other_positions = self._as_cpu_1d_tensor(
                stats["query_input_positions"], torch.long, "query_input_positions"
            )
            if (
                str(stats["text_extraction_mode"]) != text_extraction_mode
                or not torch.equal(other_ids, query_token_ids)
                or not torch.equal(other_positions, query_input_positions)
            ):
                raise ValueError("Candidates in one query used different text-token sequences")

        return query_token_ids, query_input_positions, text_extraction_mode

    def _build_candidate_record(
        self,
        *,
        role: str,
        candidate_pos: int,
        final_rank: int,
        global_idx: int,
        local_page_id: int,
        stats: Mapping[str, Any],
        query_tokens: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        scores = self._as_cpu_1d_tensor(stats["scores"], torch.float32, "scores")
        kept_mask = self._as_cpu_1d_tensor(
            stats["kept_mask"], torch.bool, "kept_mask"
        )
        max_text_indices = self._as_cpu_1d_tensor(
            stats["max_text_token_indices"],
            torch.long,
            "max_text_token_indices",
        )

        # 三个一维向量按剪枝前视觉 token 顺序严格对齐：scores[v] 是 token v 的
        # 最大相似度，kept_mask[v] 表示是否保留，max_text_indices[v] 是命中的
        # QI 文本序列下标。任一长度不一致都会破坏逐 token 记录的语义。
        if not (scores.numel() == kept_mask.numel() == max_text_indices.numel()):
            raise ValueError(
                "Visual-token statistics length mismatch: "
                f"scores={scores.numel()}, kept={kept_mask.numel()}, "
                f"argmax={max_text_indices.numel()}"
            )
        if scores.numel() == 0:
            raise ValueError("Candidate contains no visual tokens")
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite token similarity encountered")
        if int(max_text_indices.min()) < 0 or int(max_text_indices.max()) >= len(query_tokens):
            raise ValueError(
                "A max-text-token index falls outside the extracted QI text sequence"
            )

        visual_tokens = []
        all_match_counts: Counter[int] = Counter()
        kept_match_counts: Counter[int] = Counter()
        pruned_match_counts: Counter[int] = Counter()

        # 对每个视觉 token 只做一次 argmax 回查，同时累计文本 token 的吸附次数。
        # 详细列表用于逐 token 排查，三个 Counter 则用于构造更紧凑的聚合视图。
        for visual_token_index in range(scores.numel()):
            matched_index = int(max_text_indices[visual_token_index])
            matched_token = query_tokens[matched_index]
            is_kept = bool(kept_mask[visual_token_index])

            all_match_counts[matched_index] += 1
            if is_kept:
                kept_match_counts[matched_index] += 1
            else:
                pruned_match_counts[matched_index] += 1

            visual_tokens.append({
                "visual_token_index": visual_token_index,
                "max_similarity": float(scores[visual_token_index]),
                "kept": is_kept,
                "matched_query_sequence_index": matched_index,
                "matched_input_sequence_position": matched_token[
                    "input_sequence_position"
                ],
                "matched_token_id": matched_token["token_id"],
                "matched_vocab_token": matched_token["vocab_token"],
                "matched_decoded_text": matched_token["decoded_text"],
                "matched_is_special_token": matched_token["is_special_token"],
            })

        # This compact view makes it easy to see which text vocabulary pieces
        # attract many visual tokens without manually grouping the detailed rows.
        matching_summary = []
        for matched_index, count in sorted(
            all_match_counts.items(), key=lambda item: (-item[1], item[0])
        ):
            token = query_tokens[matched_index]
            matching_summary.append({
                **token,
                "matched_visual_token_count": count,
                "matched_kept_visual_token_count": kept_match_counts[matched_index],
                "matched_pruned_visual_token_count": pruned_match_counts[matched_index],
            })

        # kept/pruned 索引仍使用剪枝前的视觉 token 坐标，因此可以直接与
        # visual_tokens 列表下标对应，不会因 top-k 压缩后的重新编号而产生歧义。
        kept_indices = torch.where(kept_mask)[0].tolist()
        pruned_indices = torch.where(~kept_mask)[0].tolist()
        threshold = float(torch.as_tensor(stats["threshold"]).detach().float().cpu())

        return {
            "role": role,
            "candidate_pos": candidate_pos,
            "candidate_letter": chr(ord("A") + candidate_pos),
            "final_rank": final_rank,
            "global_idx": global_idx,
            "local_page_id": local_page_id,
            "is_ground_truth": role == "highest_ranked_correct",
            "pruning_threshold_similarity": threshold,
            "num_original_visual_tokens": scores.numel(),
            "num_kept_visual_tokens": len(kept_indices),
            "num_pruned_visual_tokens": len(pruned_indices),
            "kept_visual_token_indices": kept_indices,
            "pruned_visual_token_indices": pruned_indices,
            "matched_query_token_summary": matching_summary,
            "visual_tokens": visual_tokens,
        }

    def add_query(
        self,
        result: Mapping[str, Any],
        candidate_stats: Sequence[Mapping[str, Any]],
    ) -> None:
        """Select the best-ranked GT/non-GT candidate after final reranking."""
        # ranked_indices 排列的是 candidate_pos，而标注 page_id 使用 local_page_id
        # top_k_global_indices。一阶段检索到topk文档的全局id，按照从大到小的顺序
        # ranked_indices 重排后的一阶段文档列表id，也可代表选项，例如如果第一个是2，那么代表一阶段列表中第3个是重排后第一个，以此类推
        # gt_page_ids 看起来是ground-truth，但是是以局部id的形式来存的
        self.total_queries_seen += 1
        top_k_global_indices = [int(value) for value in result["top_k_global_indices"]]
        ranked_indices = [int(value) for value in result["ranked_indices"]]
        gt_page_ids = {int(value) for value in result["page_id"]}

        if not gt_page_ids:
            self.skipped_counts["no_ground_truth_page"] += 1
            return
        if not ranked_indices:
            self.skipped_counts["empty_ranking"] += 1
            return
        if len(candidate_stats) != len(top_k_global_indices):
            raise ValueError(
                "Similarity/image count mismatch for "
                f"{result.get('qid', '<unknown>')}: "
                f"stats={len(candidate_stats)}, candidates={len(top_k_global_indices)}"
            )

        # 统计数据的产生顺序未必等于最终重排顺序，先按 candidate_pos 建索引，
        # 后面即可用 ranked_indices 中的位置稳定地取得对应图像的逐 token 信息。
        stats_by_position = {
            int(stats["candidate_pos"]): stats for stats in candidate_stats
        }
        expected_positions = set(range(len(top_k_global_indices)))
        if set(stats_by_position) != expected_positions:
            raise ValueError("Alignment statistics do not cover every candidate position")
        if len(ranked_indices) != len(expected_positions) or set(ranked_indices) != expected_positions:
            raise ValueError("Final ranking is not a permutation of candidate positions")

        start_idx = int(result["start_idx"])
        end_idx = int(result["end_idx"])
        local_page_ids = [global_idx - start_idx for global_idx in top_k_global_indices]
        if any(not 0 <= page_id <= end_idx - start_idx for page_id in local_page_ids):
            raise ValueError("Candidate global index falls outside the document range")

        # Filtering in final reranking order means the first element of each
        # list is exactly the highest-ranked candidate for that class.
        correct_positions = [
            pos for pos in ranked_indices if local_page_ids[pos] in gt_page_ids
        ]
        incorrect_positions = [
            pos for pos in ranked_indices if local_page_ids[pos] not in gt_page_ids
        ]
        if not correct_positions:
            self.skipped_counts["correct_candidate_missing_from_topk"] += 1
            return
        if not incorrect_positions:
            self.skipped_counts["incorrect_candidate_missing_from_topk"] += 1
            return

        query_token_ids, query_input_positions, text_extraction_mode = (
            self._validate_shared_query_metadata(candidate_stats)
        )

        # 查询 token 只解码一次并保留两套位置：query_sequence_index 对应相似度
        # 矩阵行号，input_sequence_position 对应完整多模态输入中的原始位置。
        query_tokens = [
            self._describe_text_token(
                query_sequence_index=index,
                input_sequence_position=int(query_input_positions[index]),
                token_id=int(query_token_ids[index]),
            )
            for index in range(query_token_ids.numel())
        ]

        # 下面的candidate_pos描述的是，在一阶段排序local_id有序列表中第几项

        # rank_by_position 把“候选列表下标”反查为从 1 开始的最终名次；chosen
        # 固定只保留排名最靠前的一个正例和一个负例，控制 JSON 体积并便于比较。
        rank_by_position = {
            candidate_pos: rank + 1
            for rank, candidate_pos in enumerate(ranked_indices)
        }
        chosen = [
            ("highest_ranked_correct", correct_positions[0]),
            ("highest_ranked_incorrect", incorrect_positions[0]),
        ]
        candidates = []
        for role, candidate_pos in chosen:
            candidates.append(self._build_candidate_record(
                role=role,
                candidate_pos=candidate_pos,
                final_rank=rank_by_position[candidate_pos],
                global_idx=top_k_global_indices[candidate_pos],
                local_page_id=local_page_ids[candidate_pos],
                stats=stats_by_position[candidate_pos],
                query_tokens=query_tokens,
            ))

        top1_pos = ranked_indices[0]
        self.query_records.append({
            "qid": result.get("qid", f"{result['doc_name']}_{result['q_idx']}"),
            "doc_name": result["doc_name"],
            "domain": result["domain"],
            "q_idx": int(result["q_idx"]),
            "query": result["query"],
            "ground_truth_page_ids": sorted(gt_page_ids),
            "recall_at_1_correct": local_page_ids[top1_pos] in gt_page_ids,
            "text_extraction_mode": text_extraction_mode,
            "num_qi_text_tokens": len(query_tokens),
            "qi_text_token_sequence": query_tokens,
            "candidates": candidates,
        })

    def save(
        self,
        output_file: str,
        *,
        run_metadata: Mapping[str, Any],
    ) -> Path:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 顶层同时写入索引、字符串和并列 argmax 的解释，使 JSON 脱离代码后仍可
        # 独立理解；queries 中才是每个 query、候选和视觉 token 的实际诊断数据。
        payload = {
            "schema_version": 1,
            "description": (
                "For each visual token, records the QI-Early text token that "
                "achieved its maximum cosine similarity and whether the visual "
                "token survived pruning."
            ),
            "index_semantics": {
                "visual_token_index": "Zero-based index before visual-token pruning.",
                "query_sequence_index": (
                    "Zero-based index inside the text-token sequence actually used "
                    "by QI-Early similarity."
                ),
                "input_sequence_position": (
                    "Zero-based position in the full tokenized multimodal model input."
                ),
                "argmax_tie_rule": (
                    "PyTorch max returns the first query_sequence_index when values tie."
                ),
            },
            "token_string_semantics": {
                "vocab_token": "Exact tokenizer vocabulary piece (possibly a subword).",
                "decoded_text": "That one token ID decoded without skipping special tokens.",
            },
            "run": dict(run_metadata),
            "total_queries_seen": self.total_queries_seen,
            "num_queries_exported": len(self.query_records),
            "skipped_queries": dict(sorted(self.skipped_counts.items())),
            "queries": self.query_records,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path
