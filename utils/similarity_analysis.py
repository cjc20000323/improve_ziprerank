"""收集并绘制 QI-Early 视觉 token 相似度诊断结果。

这里同时维护两类数据：

1. 整体分布：每张候选图先转换成归一化直方图，同一 query 中重复的负例角色
   再取平均，最后跨 query 求均值，并绘制各 bin 在 query 样例间的最小—最大包络。
   这样每个 query 的权重相同，不会让高分辨率、token 更多的图像主导整体曲线。
2. 典型样例：Recall@1 正确和错误两组各用 reservoir sampling 均匀抽取固定数量
   的 query，并保留其原始 token 分数，用于绘制逐样例曲线。

因此，全部 query 只保存固定 bin 数量的直方图；只有最终抽中的少量 query 保留
原始 token tensor，避免分析功能持续占用大量内存。
"""

from __future__ import annotations

import json
import importlib
import math
import random
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from .similarity_image_export import (
    ImageBinaryLoader,
    export_selected_similarity_candidate_images,
)


TOP1_OUTCOME_GROUPS = ("top1_correct", "top1_incorrect")
OUTCOME_CANDIDATE_CLASSES = ("correct_candidates", "incorrect_candidates")
TOKEN_SCORE_GROUPS = ("all", "pruned", "kept")


def ensure_matplotlib_available() -> None:
    """在长时间评估开始前检查绘图库，并使用无界面的 Agg 后端。"""
    try:
        # 延迟导入使普通评估无需强制加载 matplotlib；Agg 可在服务器上无显示器画图。
        matplotlib = importlib.import_module("matplotlib")
        matplotlib.use("Agg")
        importlib.import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise RuntimeError(
            "Token-similarity analysis requires matplotlib. "
            "Install the updated requirements.txt before running evaluation."
        ) from exc


class TokenSimilarityAnalysisCollector:
    """维护 query 等权的整体直方图，并均匀抽取少量可视化样例。"""

    def __init__(
        self,
        num_examples: int = 5,
        seed: int = 42,
        num_bins: int = 80,
        comparison_num_ground_truth_candidates: int = 0,
        comparison_num_incorrect_candidates: int = 3,
    ) -> None:
        if num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {num_examples}")
        if num_bins <= 1:
            raise ValueError(f"num_bins must be greater than 1, got {num_bins}")
        if comparison_num_ground_truth_candidates < 0:
            raise ValueError(
                "comparison_num_ground_truth_candidates must be non-negative, "
                f"got {comparison_num_ground_truth_candidates}"
            )
        if comparison_num_incorrect_candidates < 0:
            raise ValueError(
                "comparison_num_incorrect_candidates must be non-negative, "
                f"got {comparison_num_incorrect_candidates}"
            )

        self.num_examples = num_examples
        self.seed = seed
        self.num_bins = num_bins
        # 这两个参数只控制“所有 query 的候选均值”统计，不影响 Recall@1 分组、
        # reservoir sampling 或逐样例图片。0 表示不限制数量，即使用该 query 在
        # 一阶段 Top-K 中实际存在的该类全部候选；正数表示按最终重排名次最多取 N 个。
        self.comparison_num_ground_truth_candidates = (
            comparison_num_ground_truth_candidates
        )
        self.comparison_num_incorrect_candidates = comparison_num_incorrect_candidates
        # 所有候选共用固定的 [0.0, 0.25] bin 边界，保证不同 query 的分布可直接平均。
        self.bin_edges = np.linspace(0.0, 0.25, num_bins + 1, dtype=np.float32)

        self.correct_examples: List[Dict[str, Any]] = []
        self.incorrect_examples: List[Dict[str, Any]] = []
        self.eligible_counts = {"correct": 0, "incorrect": 0}
        self.total_queries = 0
        self.skipped_counts: DefaultDict[str, int] = defaultdict(int)

        # 全量 query 的配对统计只维护浮点累加和，不保存所有候选的逐 token 分数，
        # 因此查询数量增加时也不会持续占用大量内存。
        self.comparison_eligible_queries = 0
        self.comparison_skipped_counts: DefaultDict[str, int] = defaultdict(int)
        self._comparison_sums: Dict[str, Dict[str, float]] = {
            role: self._new_candidate_metric_sums()
            for role in ("ground_truth_candidates", "top_ranked_incorrect_candidates")
        }
        self._comparison_candidate_counts = {
            role: 0 for role in self._comparison_sums
        }

        # 新增的 Top1 结果分组与原有 comparison 累计器完全独立。标量指标按候选
        # 等权累计；分布则先在单个 query 内对同类候选求平均，再跨 query 等权，
        # 这样 Top1 错误且前置错误候选较多的 query 不会主导总体曲线。
        self.outcome_group_query_counts = {
            group: 0 for group in TOP1_OUTCOME_GROUPS
        }
        self.outcome_group_skipped_counts: DefaultDict[str, int] = defaultdict(int)
        self._outcome_metric_sums = {
            group: {
                candidate_class: self._new_candidate_metric_sums()
                for candidate_class in OUTCOME_CANDIDATE_CLASSES
            }
            for group in TOP1_OUTCOME_GROUPS
        }
        self._outcome_candidate_counts = {
            group: {
                candidate_class: 0
                for candidate_class in OUTCOME_CANDIDATE_CLASSES
            }
            for group in TOP1_OUTCOME_GROUPS
        }
        self._outcome_histogram_sums = {
            group: {
                candidate_class: np.zeros(num_bins, dtype=np.float64)
                for candidate_class in OUTCOME_CANDIDATE_CLASSES
            }
            for group in TOP1_OUTCOME_GROUPS
        }
        self._outcome_histogram_mins = {
            group: {
                candidate_class: np.full(num_bins, np.inf, dtype=np.float64)
                for candidate_class in OUTCOME_CANDIDATE_CLASSES
            }
            for group in TOP1_OUTCOME_GROUPS
        }
        self._outcome_histogram_maxs = {
            group: {
                candidate_class: np.full(num_bins, -np.inf, dtype=np.float64)
                for candidate_class in OUTCOME_CANDIDATE_CLASSES
            }
            for group in TOP1_OUTCOME_GROUPS
        }

        # 两组使用独立随机数发生器：正确组抽到哪些 query，不受错误组数量影响。
        self._rng = {
            "correct": random.Random(seed),
            "incorrect": random.Random(seed + 1),
        }
        self._group_histograms: Dict[str, DefaultDict[str, List[np.ndarray]]] = {
            "correct": defaultdict(list),
            "incorrect": defaultdict(list),
        }

    @staticmethod
    def _new_candidate_metric_sums() -> Dict[str, float]:
        """Create independent accumulators for all/pruned/kept token metrics."""
        sums = {"pruning_threshold_similarity": 0.0}
        for token_group in TOKEN_SCORE_GROUPS:
            sums.update({
                f"{token_group}_token_similarity": 0.0,
                f"{token_group}_token_similarity_variance": 0.0,
                f"{token_group}_entropy_shannon_bits": 0.0,
                f"{token_group}_entropy_normalized": 0.0,
                f"{token_group}_entropy_effective_bins": 0.0,
            })
        return sums

    def add_query(
        self,
        result: Mapping[str, Any],
        candidate_stats: Sequence[Mapping[str, Any]],
    ) -> None:
        """判定一个 query 的 Recall@1 分组，更新整体分布并进行均匀抽样。"""
        self.total_queries += 1

        # 下面涉及三套索引：
        # 1) global_idx：parquet 全表的行号；
        # 2) local_page_id：文档内部页号，等于 global_idx - start_idx；
        # 3) candidate_pos：该页在一阶段 Top-K 候选列表中的位置。
        # ranked_indices 排列的是 candidate_pos，而标注 page_id 使用 local_page_id
        # top_k_global_indices。一阶段检索到topk文档的全局id，按照从大到小的顺序
        # ranked_indices 重排后的一阶段文档列表id，也可代表选项，例如如果第一个是2，那么代表一阶段列表中第3个是重排后第一个，以此类推
        # gt_page_ids 看起来是ground-truth，但是是以局部id的形式来存的
        top_k_global_indices = list(result["top_k_global_indices"])
        ranked_indices = list(result["ranked_indices"])
        gt_page_ids = {int(page_id) for page_id in result["page_id"]}

        if not gt_page_ids:
            self.skipped_counts["no_ground_truth_page"] += 1
            self.comparison_skipped_counts["no_ground_truth_page"] += 1
            return
        if not ranked_indices:
            self.skipped_counts["empty_ranking"] += 1
            self.comparison_skipped_counts["empty_ranking"] += 1
            return
        if len(candidate_stats) != len(top_k_global_indices):
            raise ValueError(
                "Similarity/image count mismatch for "
                f"{result.get('qid', '<unknown>')}: "
                f"stats={len(candidate_stats)}, candidates={len(top_k_global_indices)}"
            )

        # 模型返回的分数按输入图像顺序排列，先还原成 candidate_pos -> 分数记录，
        # 后续即使重排了候选，也能找到同一张图的 token 分数。
        stats_by_position = {
            int(item["candidate_pos"]): item for item in candidate_stats
        }
        expected_positions = set(range(len(top_k_global_indices)))
        if set(stats_by_position) != expected_positions:
            raise ValueError(
                "Similarity statistics do not cover every original candidate position "
                f"for {result.get('qid', '<unknown>')}"
            )
        entropy_similarity_range = self._query_similarity_range(candidate_stats)

        start_idx = int(result["start_idx"])
        end_idx = int(result["end_idx"])
        num_pages = end_idx - start_idx + 1
        local_page_ids = []
        # 这里是把全局id变成局部id，顺序还是一阶段检索的topk相似度从大到小
        for global_idx in top_k_global_indices:
            # MMDocIR 的一个文档在 parquet 中占连续行，因此减去该文档首行即可
            # 得到文档内部页号。范围检查可以及时发现候选跨文档或索引不匹配。
            local_page_id = int(global_idx) - start_idx
            if not 0 <= local_page_id < num_pages:
                raise ValueError(
                    f"Candidate global index {global_idx} is outside document range "
                    f"[{start_idx}, {end_idx}] for {result.get('qid', '<unknown>')}"
                )
            local_page_ids.append(local_page_id)

        if (
            len(ranked_indices) != len(expected_positions)
            or set(ranked_indices) != expected_positions
        ):
            raise ValueError(
                f"Ranking is not a permutation of all candidate positions: {ranked_indices} "
                f"for {result.get('qid', '<unknown>')}"
            )

        # 按最终重排次序分别取出 GT 和非 GT 候选；列表首项就是各类中排名最高者。

        gt_positions = [
            pos for pos in ranked_indices if local_page_ids[pos] in gt_page_ids
        ]
        negative_positions = [
            pos for pos in ranked_indices if local_page_ids[pos] not in gt_page_ids
        ]

        # 新分组只要求 Top-K 中同时存在正确和错误候选，不受原有逐样例绘图所需
        # “三个错误候选”条件限制，因此必须在后面的早退逻辑之前完成累计。
        if not gt_positions:
            self.outcome_group_skipped_counts[
                "correct_candidate_missing_from_topk"
            ] += 1
        elif not negative_positions:
            self.outcome_group_skipped_counts[
                "incorrect_candidate_missing_from_topk"
            ] += 1
        else:
            self._add_top1_outcome_query(
                ranked_indices=ranked_indices,
                gt_positions=gt_positions,
                negative_positions=negative_positions,
                stats_by_position=stats_by_position,
                entropy_similarity_range=entropy_similarity_range,
            )

        # 全查询聚合只要求 Top-K 中同时存在至少一个 GT 和一个非 GT，因此应在
        # “逐样例画图需要三个负例”等限制之前更新，避免漏掉本可用于成对比较的 query。
        if not gt_positions:
            self.comparison_skipped_counts["correct_candidate_missing_from_topk"] += 1
        elif not negative_positions:
            self.comparison_skipped_counts["incorrect_candidate_missing_from_topk"] += 1
        else:
            # gt_positions 与 negative_positions 已按最终重排顺序排列。默认配置下
            # GT 使用 Top-K 内全部正确候选，错误候选使用最终排名最靠前的 3 个；
            # 当命令行传入其他上限时，仅改变这里的全查询汇总候选集合。
            selected_gt_positions = self._select_comparison_positions(
                gt_positions,
                self.comparison_num_ground_truth_candidates,
            )
            selected_negative_positions = self._select_comparison_positions(
                negative_positions,
                self.comparison_num_incorrect_candidates,
            )
            self._add_query_candidate_set_comparison(
                incorrect_stats=[
                    stats_by_position[position]
                    for position in selected_negative_positions
                ],
                ground_truth_stats=[
                    stats_by_position[position] for position in selected_gt_positions
                ],
                entropy_similarity_range=entropy_similarity_range,
            )

        top1_pos = ranked_indices[0]
        # 多 GT query 的官方 recall@1 数值可能是 1 / GT数；这里的“正确/错误”分组
        # 只关心 Top1 是否命中任意 GT，因此使用布尔值。
        recall1_correct = local_page_ids[top1_pos] in gt_page_ids

        if recall1_correct:
            if len(negative_positions) < 3:
                self.skipped_counts["correct_not_enough_negative_candidates"] += 1
                return
            group = "correct"
            # 正确组：Top1 正确候选 + 最终排名最高的三个非正确候选。
            selected = [
                (top1_pos, "top1_correct", "Top1 correct (GT)"),
                (negative_positions[0], "negative_1", "Highest-ranked negative"),
                (negative_positions[1], "negative_2", "Negative #2"),
                (negative_positions[2], "negative_3", "Negative #3"),
            ]
        else:
            if not gt_positions:
                # The reranker cannot provide the requested correct option if
                # first-stage Top-K did not retrieve any ground-truth page.
                self.skipped_counts["incorrect_gt_missing_from_candidates"] += 1
                return
            other_negative_positions = [
                pos for pos in negative_positions if pos != top1_pos
            ]
            if len(other_negative_positions) < 2:
                self.skipped_counts["incorrect_not_enough_negative_candidates"] += 1
                return
            group = "incorrect"
            # 错误组：错误 Top1 + 其后的两个高排名非正确候选 + 排名最高的 GT。
            # 如果一阶段 Top-K 没有召回 GT，就无法画用户要求的“正确选项”，故跳过。
            selected = [
                (top1_pos, "top1_incorrect", "Top1 incorrect"),
                (other_negative_positions[0], "negative_1", "Other negative #1"),
                (other_negative_positions[1], "negative_2", "Other negative #2"),
                (gt_positions[0], "ground_truth", "Highest-ranked GT"),
            ]

        # 将 candidate_pos 反查为从 1 开始的最终名次，便于图例和 JSON 阅读。
        rank_by_position = {
            candidate_pos: rank + 1
            for rank, candidate_pos in enumerate(ranked_indices)
        }
        candidates = []
        for candidate_pos, role, role_label in selected:
            candidates.append(
                self._build_candidate_record(
                    candidate_pos=candidate_pos,
                    role=role,
                    role_label=role_label,
                    final_rank=rank_by_position[candidate_pos],
                    global_idx=int(top_k_global_indices[candidate_pos]),
                    local_page_id=local_page_ids[candidate_pos],
                    is_gt=local_page_ids[candidate_pos] in gt_page_ids,
                    stats=stats_by_position[candidate_pos],
                )
            )

        example = {
            "qid": result.get("qid", f"{result['doc_name']}_{result['q_idx']}"),
            "doc_name": result["doc_name"],
            "domain": result["domain"],
            "q_idx": int(result["q_idx"]),
            "query": result["query"],
            "ground_truth_page_ids": sorted(gt_page_ids),
            "recall_at_1": 1.0 / len(gt_page_ids) if recall1_correct else 0.0,
            "recall1_correct": recall1_correct,
            "entropy_similarity_range": list(entropy_similarity_range),
            "candidates": candidates,
        }

        self._add_group_histograms(group, candidates)
        self._reservoir_add(group, example)

    def _build_candidate_record(
        self,
        candidate_pos: int,
        role: str,
        role_label: str,
        final_rank: int,
        global_idx: int,
        local_page_id: int,
        is_gt: bool,
        stats: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """把某个候选的诊断 tensor 规范化为可长期保存的 CPU 记录。"""
        # 再次 detach/转 CPU 是防御性处理：分析数据不应保留计算图或占用显存。
        scores = torch.as_tensor(stats["scores"]).detach().float().cpu().contiguous()
        kept_mask = (
            torch.as_tensor(stats["kept_mask"])
            .detach()
            .to(dtype=torch.bool, device="cpu")
            .contiguous()
        )
        if scores.ndim != 1 or kept_mask.ndim != 1:
            raise ValueError("Similarity scores and kept_mask must both be one-dimensional")
        if scores.numel() != kept_mask.numel():
            raise ValueError(
                f"Similarity/mask length mismatch: {scores.numel()} vs {kept_mask.numel()}"
            )
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite token similarity score encountered")

        threshold_value = stats["threshold"]
        if isinstance(threshold_value, torch.Tensor):
            threshold = float(threshold_value.detach().float().cpu().item())
        else:
            threshold = float(threshold_value)

        return {
            "role": role,
            "role_label": role_label,
            "candidate_pos": candidate_pos,
            "candidate_letter": chr(ord("A") + candidate_pos),
            "final_rank": final_rank,
            "global_idx": global_idx,
            "local_page_id": local_page_id,
            "is_gt": is_gt,
            "scores": scores,
            "kept_mask": kept_mask,
            "threshold": threshold,
            "num_original_tokens": int(stats["num_original_tokens"]),
            "num_kept_tokens": int(stats["num_kept_tokens"]),
        }

    def _histogram_counts_and_density(
        self,
        scores: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """使用与绘图相同的边界计算原始 token 数量和归一化密度。"""
        # 将分数限制到当前展示区间，确保每个 token 都被计入首尾之间的某个 bin。
        values = np.clip(scores.numpy(), self.bin_edges[0], self.bin_edges[-1])
        counts, _ = np.histogram(values, bins=self.bin_edges)
        total = counts.sum()
        if total == 0:
            raise ValueError("No token similarities fell inside display range [0.0, 0.25]")
        widths = np.diff(self.bin_edges)
        # 除以 token 总数和 bin 宽度后，曲线面积为 1；不同 token 数的图才可比较。
        density = counts.astype(np.float64) / (total * widths)
        return counts, density

    def _normalized_histogram(self, scores: torch.Tensor) -> np.ndarray:
        """将一张图的所有 token 分数转换成积分为 1 的概率密度直方图。"""
        _, density = self._histogram_counts_and_density(scores)
        return density

    @staticmethod
    def _query_similarity_range(
        candidate_stats: Sequence[Mapping[str, Any]],
    ) -> tuple[float, float]:
        """Return the raw min/max over every candidate token in one query."""
        query_min = math.inf
        query_max = -math.inf
        for stats in candidate_stats:
            scores = (
                torch.as_tensor(stats["scores"])
                .detach()
                .float()
                .cpu()
                .contiguous()
            )
            if scores.ndim != 1 or scores.numel() == 0:
                raise ValueError(
                    "Query entropy range requires non-empty 1-D candidate scores"
                )
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite token similarity score encountered")
            query_min = min(query_min, float(scores.min().item()))
            query_max = max(query_max, float(scores.max().item()))

        if not math.isfinite(query_min) or not math.isfinite(query_max):
            raise ValueError("Query entropy range requires at least one candidate")
        return query_min, query_max

    def _entropy_histogram_counts(
        self,
        scores: torch.Tensor,
        similarity_range: tuple[float, float],
    ) -> np.ndarray:
        """Bin raw scores over their query's shared min/max interval."""
        range_min, range_max = similarity_range
        if (
            not math.isfinite(range_min)
            or not math.isfinite(range_max)
            or range_min > range_max
        ):
            raise ValueError(f"Invalid query similarity range: {similarity_range}")

        values = scores.detach().float().cpu().contiguous().numpy()
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Histogram entropy requires non-empty 1-D scores")
        if float(values.min()) < range_min or float(values.max()) > range_max:
            raise ValueError(
                "Token similarity falls outside its query entropy range: "
                f"scores=[{float(values.min())}, {float(values.max())}], "
                f"range={similarity_range}"
            )

        if range_min == range_max:
            counts = np.zeros(self.num_bins, dtype=np.int64)
            counts[0] = values.size
            return counts

        counts, _ = np.histogram(
            values,
            bins=self.num_bins,
            range=(range_min, range_max),
        )
        if int(counts.sum()) != values.size:
            raise ValueError("Query-range entropy histogram lost token scores")
        return counts

    def _histogram_entropy(self, counts: np.ndarray) -> Dict[str, float]:
        """基于逐 bin 概率计算 Shannon 熵及其便于比较的派生指标。"""
        total = int(counts.sum())
        if total <= 0:
            raise ValueError("Histogram entropy requires at least one token")

        probabilities = counts[counts > 0].astype(np.float64) / total
        shannon_bits = float(-np.sum(probabilities * np.log2(probabilities)))
        max_entropy_bits = math.log2(self.num_bins)
        # 浮点误差可能让理论边界轻微越过 0 或 1，因此在写入 JSON 前夹紧。
        normalized = float(np.clip(shannon_bits / max_entropy_bits, 0.0, 1.0))

        return {
            "shannon_bits": shannon_bits,
            "normalized": normalized,
            # 2**H 表示与当前分布具有相同熵的均匀分布会占据多少个 bin。
            "effective_bins": float(2.0 ** shannon_bits),
        }

    def _candidate_comparison_values(
        self,
        stats: Mapping[str, Any],
        entropy_similarity_range: tuple[float, float],
    ) -> Dict[str, Any]:
        """提取一个候选在全部、剪掉和保留 token 上的三类指标。"""
        scores = torch.as_tensor(stats["scores"]).detach().float().cpu().contiguous()
        kept_mask = (
            torch.as_tensor(stats["kept_mask"])
            .detach()
            .to(dtype=torch.bool, device="cpu")
            .contiguous()
        )
        if scores.ndim != 1 or kept_mask.ndim != 1 or scores.numel() == 0:
            raise ValueError(
                "Candidate comparison requires non-empty 1-D scores and kept_mask"
            )
        if scores.numel() != kept_mask.numel():
            raise ValueError(
                "Candidate comparison similarity/mask length mismatch: "
                f"{scores.numel()} vs {kept_mask.numel()}"
            )
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite token similarity score encountered")

        score_groups = {
            "all": scores,
            "pruned": scores[~kept_mask],
            "kept": scores[kept_mask],
        }
        if score_groups["pruned"].numel() == 0 or score_groups["kept"].numel() == 0:
            raise ValueError(
                "Candidate comparison requires both pruned and kept visual tokens"
            )

        threshold_value = stats["threshold"]
        if isinstance(threshold_value, torch.Tensor):
            threshold = float(threshold_value.detach().float().cpu().item())
        else:
            threshold = float(threshold_value)
        if not math.isfinite(threshold):
            raise ValueError("Non-finite pruning threshold encountered")

        token_group_metrics = {}
        for token_group, group_scores in score_groups.items():
            entropy_counts = self._entropy_histogram_counts(
                group_scores,
                entropy_similarity_range,
            )
            token_group_metrics[token_group] = {
                "mean_similarity": float(group_scores.mean().item()),
                # 每张候选图中当前 token 组是本次诊断关心的完整总体，因此使用
                # unbiased=False（ddof=0），单 token 分组也能得到定义良好的 0 方差。
                "similarity_variance": float(
                    group_scores.var(unbiased=False).item()
                ),
                "entropy": self._histogram_entropy(entropy_counts),
            }

        return {
            "threshold": threshold,
            "token_groups": token_group_metrics,
        }

    @staticmethod
    def _accumulate_candidate_metric_values(
        sums: Dict[str, float],
        values: Mapping[str, Any],
    ) -> None:
        """Add one candidate's all/pruned/kept scalar metrics to ``sums``."""
        sums["pruning_threshold_similarity"] += float(values["threshold"])
        for token_group in TOKEN_SCORE_GROUPS:
            metrics = values["token_groups"][token_group]
            entropy = metrics["entropy"]
            sums[f"{token_group}_token_similarity"] += metrics[
                "mean_similarity"
            ]
            sums[f"{token_group}_token_similarity_variance"] += metrics[
                "similarity_variance"
            ]
            sums[f"{token_group}_entropy_shannon_bits"] += entropy[
                "shannon_bits"
            ]
            sums[f"{token_group}_entropy_normalized"] += entropy["normalized"]
            sums[f"{token_group}_entropy_effective_bins"] += entropy[
                "effective_bins"
            ]

    @staticmethod
    def _mean_candidate_token_metrics(
        sums: Mapping[str, float],
        candidate_count: int,
    ) -> Dict[str, Any]:
        """Build parallel output fields for all/pruned/kept token groups."""
        output: Dict[str, Any] = {}
        for token_group in TOKEN_SCORE_GROUPS:
            if candidate_count == 0:
                mean_similarity = None
                mean_variance = None
                mean_entropy = {
                    "shannon_bits": None,
                    "normalized": None,
                    "effective_bins": None,
                }
            else:
                mean_similarity = (
                    sums[f"{token_group}_token_similarity"] / candidate_count
                )
                mean_variance = (
                    sums[f"{token_group}_token_similarity_variance"]
                    / candidate_count
                )
                mean_entropy = {
                    "shannon_bits": (
                        sums[f"{token_group}_entropy_shannon_bits"]
                        / candidate_count
                    ),
                    "normalized": (
                        sums[f"{token_group}_entropy_normalized"]
                        / candidate_count
                    ),
                    "effective_bins": (
                        sums[f"{token_group}_entropy_effective_bins"]
                        / candidate_count
                    ),
                }
            output.update({
                f"mean_{token_group}_token_similarity": mean_similarity,
                f"mean_{token_group}_token_similarity_variance": mean_variance,
                f"mean_{token_group}_token_similarity_entropy": mean_entropy,
            })
        return output


    def _add_top1_outcome_query(
        self,
        ranked_indices: Sequence[int],
        gt_positions: Sequence[int],
        negative_positions: Sequence[int],
        stats_by_position: Mapping[int, Mapping[str, Any]],
        entropy_similarity_range: tuple[float, float],
    ) -> None:
        """Apply the requested candidate rule for one reranked query."""
        top1_position = int(ranked_indices[0])
        gt_position_set = {int(position) for position in gt_positions}
        negative_position_set = {
            int(position) for position in negative_positions
        }

        if top1_position in gt_position_set:
            outcome_group = "top1_correct"
            selected_correct_positions = [top1_position]
            selected_incorrect_positions = [int(negative_positions[0])]
        else:
            outcome_group = "top1_incorrect"
            highest_correct_position = int(gt_positions[0])
            highest_correct_rank_index = list(ranked_indices).index(
                highest_correct_position
            )
            selected_correct_positions = [highest_correct_position]
            selected_incorrect_positions = [
                int(position)
                for position in ranked_indices[:highest_correct_rank_index]
            ]
            if (
                not selected_incorrect_positions
                or any(
                    position not in negative_position_set
                    for position in selected_incorrect_positions
                )
            ):
                raise ValueError(
                    "Candidates ahead of the highest-ranked GT must all be non-GT"
                )

        self._add_top1_outcome_candidate_sets(
            outcome_group=outcome_group,
            correct_stats=[
                stats_by_position[position]
                for position in selected_correct_positions
            ],
            incorrect_stats=[
                stats_by_position[position]
                for position in selected_incorrect_positions
            ],
            entropy_similarity_range=entropy_similarity_range,
        )

    def _add_top1_outcome_candidate_sets(
        self,
        outcome_group: str,
        correct_stats: Sequence[Mapping[str, Any]],
        incorrect_stats: Sequence[Mapping[str, Any]],
        entropy_similarity_range: tuple[float, float],
    ) -> None:
        """Accumulate scalar metrics and one query-balanced histogram per class."""
        if outcome_group not in TOP1_OUTCOME_GROUPS:
            raise ValueError(f"Unknown Top1 outcome group: {outcome_group!r}")
        stats_by_class = {
            "correct_candidates": list(correct_stats),
            "incorrect_candidates": list(incorrect_stats),
        }
        if any(not values for values in stats_by_class.values()):
            raise ValueError(
                "Top1 outcome statistics require both correct and incorrect candidates"
            )

        # Finish all validation and numerical extraction before touching any sum;
        # a malformed candidate can therefore never leave a half-updated query.
        values_by_class = {
            candidate_class: [
                self._candidate_comparison_values(
                    stats,
                    entropy_similarity_range,
                )
                for stats in candidate_stats
            ]
            for candidate_class, candidate_stats in stats_by_class.items()
        }
        query_histogram_by_class = {
            candidate_class: np.mean(
                np.stack([
                    self._normalized_histogram(
                        torch.as_tensor(stats["scores"])
                        .detach()
                        .float()
                        .cpu()
                        .contiguous()
                    )
                    for stats in candidate_stats
                ], axis=0),
                axis=0,
            )
            for candidate_class, candidate_stats in stats_by_class.items()
        }

        for candidate_class, candidate_values in values_by_class.items():
            sums = self._outcome_metric_sums[outcome_group][candidate_class]
            for values in candidate_values:
                self._accumulate_candidate_metric_values(sums, values)
            self._outcome_candidate_counts[outcome_group][candidate_class] += len(
                candidate_values
            )

            query_histogram = query_histogram_by_class[candidate_class]
            self._outcome_histogram_sums[outcome_group][
                candidate_class
            ] += query_histogram
            histogram_min = self._outcome_histogram_mins[outcome_group][
                candidate_class
            ]
            histogram_max = self._outcome_histogram_maxs[outcome_group][
                candidate_class
            ]
            np.minimum(histogram_min, query_histogram, out=histogram_min)
            np.maximum(histogram_max, query_histogram, out=histogram_max)

        self.outcome_group_query_counts[outcome_group] += 1

    @staticmethod
    def _select_comparison_positions(
        ranked_positions: Sequence[int],
        maximum_candidates: int,
    ) -> List[int]:
        """从一种候选中按最终名次取统计所需的位置。"""
        # 0 是“全部”的显式哨兵值；复制列表可以避免调用方意外共享并修改原列表。
        if maximum_candidates == 0:
            return list(ranked_positions)
        return list(ranked_positions[:maximum_candidates])

    def _add_query_candidate_comparison(
        self,
        highest_incorrect_stats: Mapping[str, Any],
        highest_correct_stats: Mapping[str, Any],
        entropy_similarity_range: tuple[float, float],
    ) -> None:
        """让一个 query 的最高排名错误候选和最高排名 GT 各贡献一次。"""
        self._add_query_candidate_set_comparison(
            incorrect_stats=[highest_incorrect_stats],
            ground_truth_stats=[highest_correct_stats],
            entropy_similarity_range=entropy_similarity_range,
        )

    def _add_query_candidate_set_comparison(
        self,
        incorrect_stats: Sequence[Mapping[str, Any]],
        ground_truth_stats: Sequence[Mapping[str, Any]],
        entropy_similarity_range: tuple[float, float],
    ) -> None:
        """累计一个 query 中被选中的 GT 与高排名错误候选。"""
        if not incorrect_stats or not ground_truth_stats:
            raise ValueError(
                "Candidate-set comparison requires at least one ground-truth and "
                "one incorrect candidate"
            )

        # 先完成两类候选的数值提取，再更新累计器。这样任一候选数据非法时，
        # 当前 query 不会只写入一半，从而保证候选计数和浮点累加和始终同步。
        values_by_role = {
            "ground_truth_candidates": [
                self._candidate_comparison_values(
                    stats,
                    entropy_similarity_range,
                )
                for stats in ground_truth_stats
            ],
            "top_ranked_incorrect_candidates": [
                self._candidate_comparison_values(
                    stats,
                    entropy_similarity_range,
                )
                for stats in incorrect_stats
            ],
        }

        for role, candidate_values in values_by_role.items():
            sums = self._comparison_sums[role]
            for values in candidate_values:
                self._accumulate_candidate_metric_values(sums, values)
            self._comparison_candidate_counts[role] += len(candidate_values)
        self.comparison_eligible_queries += 1

    def _all_query_candidate_comparison_summary(
        self,
        keep_ratio: float,
    ) -> Dict[str, Any]:
        """汇总所有可成对 query 的两类候选均值。"""
        query_count = self.comparison_eligible_queries

        def role_summary(role: str) -> Dict[str, Any]:
            sums = self._comparison_sums[role]
            candidate_count = self._comparison_candidate_counts[role]
            if candidate_count == 0:
                mean_threshold = None
            else:
                mean_threshold = (
                    sums["pruning_threshold_similarity"] / candidate_count
                )
            return {
                "num_queries": query_count,
                "num_candidates": candidate_count,
                "mean_pruning_threshold_similarity": mean_threshold,
                **self._mean_candidate_token_metrics(sums, candidate_count),
            }

        return {
            "keep_ratio": keep_ratio,
            "total_queries_seen": self.total_queries,
            "eligible_paired_queries": query_count,
            "skipped_queries": dict(sorted(self.comparison_skipped_counts.items())),
            "candidate_weighting": (
                "Each selected candidate contributes equally. Similarity mean, "
                "population variance, and histogram entropy are calculated "
                "separately for each token group within a candidate before "
                "candidate-level averaging."
            ),
            "token_group_definitions": {
                "all": "All visual tokens before pruning.",
                "pruned": (
                    "Visual tokens where the actual kept_mask is false at the "
                    "configured keep ratio."
                ),
                "kept": (
                    "Visual tokens where the actual kept_mask is true at the "
                    "configured keep ratio."
                ),
            },
            "variance_basis": {
                "score_groups": list(TOKEN_SCORE_GROUPS),
                "ddof": 0,
                "definition": (
                    "Population variance is calculated separately within every "
                    "selected candidate and token group using raw similarities."
                ),
                "aggregation": (
                    "Candidate variances are averaged with equal candidate weight "
                    "using the same selected candidates as the entropy metrics."
                ),
            },
            "selection_rule": {
                "ground_truth_candidates": {
                    "maximum_per_query": self.comparison_num_ground_truth_candidates,
                    "zero_means_all_available": True,
                    "description": (
                        "GT candidates available in the first-stage Top-K, ordered "
                        "by final reranking position."
                    ),
                },
                "top_ranked_incorrect_candidates": {
                    "maximum_per_query": self.comparison_num_incorrect_candidates,
                    "zero_means_all_available": True,
                    "description": (
                        "Non-GT candidates with the best final reranking positions."
                    ),
                },
            },
            "entropy_basis": {
                "score_groups": list(TOKEN_SCORE_GROUPS),
                "num_bins": self.num_bins,
                "range_scope": "per_query",
                "range_definition": (
                    "For each query, bin edges span the raw minimum and maximum "
                    "similarity across all tokens from all first-stage Top-K "
                    "candidates. The same edges are used for every selected "
                    "candidate and token group in that query."
                ),
                "degenerate_range_rule": (
                    "If a query minimum equals its maximum, all tokens are placed "
                    "in one bin and entropy is zero."
                ),
                "aggregation": (
                    "Entropy is calculated separately for every selected candidate, "
                    "then averaged with equal candidate weight."
                ),
            },
            "metrics": {
                "ground_truth_candidates": role_summary(
                    "ground_truth_candidates"
                ),
                "top_ranked_incorrect_candidates": role_summary(
                    "top_ranked_incorrect_candidates"
                ),
            },
        }


    def _top1_outcome_distribution_summary(
        self,
        outcome_group: str,
        candidate_class: str,
    ) -> Dict[str, Any]:
        """Return the mean density and per-bin range across query samples."""
        query_count = self.outcome_group_query_counts[outcome_group]
        if query_count == 0:
            return {
                "num_query_histograms": 0,
                "mean_density": None,
                "min_density": None,
                "max_density": None,
            }

        histogram_sum = self._outcome_histogram_sums[outcome_group][
            candidate_class
        ]
        mean_density = np.asarray(histogram_sum / query_count, dtype=np.float64)

        return {
            "num_query_histograms": query_count,
            "mean_density": mean_density.tolist(),
            "min_density": self._outcome_histogram_mins[outcome_group][
                candidate_class
            ].tolist(),
            "max_density": self._outcome_histogram_maxs[outcome_group][
                candidate_class
            ].tolist(),
        }

    def _top1_outcome_similarity_summary(
        self,
        keep_ratio: float,
    ) -> Dict[str, Any]:
        """Summarize the additive Top1-conditioned candidate statistics."""
        selection_rules = {
            "top1_correct": {
                "correct_candidates": "The final Top1 ground-truth candidate.",
                "incorrect_candidates": (
                    "The highest-ranked non-ground-truth candidate."
                ),
            },
            "top1_incorrect": {
                "correct_candidates": (
                    "The highest-ranked ground-truth candidate."
                ),
                "incorrect_candidates": (
                    "Every non-ground-truth candidate ranked ahead of the "
                    "highest-ranked ground-truth candidate."
                ),
            },
        }

        def candidate_class_summary(
            outcome_group: str,
            candidate_class: str,
        ) -> Dict[str, Any]:
            query_count = self.outcome_group_query_counts[outcome_group]
            candidate_count = self._outcome_candidate_counts[outcome_group][
                candidate_class
            ]
            sums = self._outcome_metric_sums[outcome_group][candidate_class]
            if candidate_count == 0:
                mean_threshold = None
            else:
                mean_threshold = (
                    sums["pruning_threshold_similarity"] / candidate_count
                )
            return {
                "num_queries": query_count,
                "num_candidates": candidate_count,
                "mean_candidates_per_query": (
                    candidate_count / query_count if query_count else None
                ),
                "mean_pruning_threshold_similarity": mean_threshold,
                **self._mean_candidate_token_metrics(sums, candidate_count),
                "token_similarity_distribution": (
                    self._top1_outcome_distribution_summary(
                        outcome_group,
                        candidate_class,
                    )
                ),
            }

        return {
            "keep_ratio": keep_ratio,
            "total_queries_seen": self.total_queries,
            "eligible_queries": dict(self.outcome_group_query_counts),
            "skipped_queries": dict(
                sorted(self.outcome_group_skipped_counts.items())
            ),
            "bin_edges": self.bin_edges.tolist(),
            "candidate_metric_weighting": (
                "Each selected candidate contributes equally. Similarity mean, "
                "population variance, and histogram entropy are calculated "
                "separately for all, pruned, and kept tokens within a candidate."
            ),
            "token_group_definitions": {
                "all": "All visual tokens before pruning.",
                "pruned": (
                    "Visual tokens where the actual kept_mask is false at the "
                    "configured keep ratio."
                ),
                "kept": (
                    "Visual tokens where the actual kept_mask is true at the "
                    "configured keep ratio."
                ),
            },
            "distribution_weighting": (
                "Each candidate is converted to a normalized density. Candidates "
                "of the same class are averaged within a query, then queries are "
                "averaged equally."
            ),
            "distribution_interval": (
                "For each histogram bin, min_density and max_density are the "
                "minimum and maximum query-level densities across eligible queries."
            ),
            "variance_basis": {
                "score_groups": list(TOKEN_SCORE_GROUPS),
                "ddof": 0,
                "definition": (
                    "Population variance is calculated within each candidate and "
                    "token group using raw similarities."
                ),
                "aggregation": "Candidate population variances are averaged equally.",
            },
            "entropy_basis": {
                "score_groups": list(TOKEN_SCORE_GROUPS),
                "num_bins": self.num_bins,
                "range_scope": "per_query",
                "range_definition": (
                    "For each query, bin edges span the raw minimum and maximum "
                    "similarity across all tokens from all first-stage Top-K "
                    "candidates. The same edges are used for every selected "
                    "candidate and token group in that query."
                ),
                "degenerate_range_rule": (
                    "If a query minimum equals its maximum, all tokens are placed "
                    "in one bin and entropy is zero."
                ),
            },
            "groups": {
                outcome_group: {
                    "num_queries": self.outcome_group_query_counts[outcome_group],
                    "selection_rule": selection_rules[outcome_group],
                    "correct_candidates": candidate_class_summary(
                        outcome_group,
                        "correct_candidates",
                    ),
                    "incorrect_candidates": candidate_class_summary(
                        outcome_group,
                        "incorrect_candidates",
                    ),
                }
                for outcome_group in TOP1_OUTCOME_GROUPS
            },
        }

    @staticmethod
    def _candidate_score_groups(
        candidate: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """按实际剪枝掩码把一个候选拆成全部、剪掉和保留三组 token。"""
        scores = candidate["scores"]
        kept_mask = candidate["kept_mask"]
        score_groups = {
            "all": scores,
            "pruned": scores[~kept_mask],
            "kept": scores[kept_mask],
        }
        if score_groups["pruned"].numel() == 0 or score_groups["kept"].numel() == 0:
            raise ValueError(
                f"Candidate {candidate['candidate_letter']} must contain both "
                "pruned and kept visual tokens"
            )
        return score_groups

    def _add_group_histograms(
        self,
        group: str,
        candidates: Sequence[Mapping[str, Any]],
    ) -> None:
        # 聚合顺序很重要：候选图各自归一化 -> 一个 query 内的多个负例取平均 ->
        # 保存该 query 的角色直方图。最终画图再跨 query 平均，因此每个 query 等权，
        # 不会因某张图分辨率更高、视觉 token 更多而获得更大权重。
        role_histograms: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
        for candidate in candidates:
            role = str(candidate["role"])
            if role.startswith("negative_"):
                aggregate_role = "other_negatives"
            else:
                aggregate_role = role
            role_histograms[aggregate_role].append(
                self._normalized_histogram(candidate["scores"])
            )

        for role, histograms in role_histograms.items():
            self._group_histograms[group][role].append(
                np.mean(np.stack(histograms, axis=0), axis=0)
            )

    def _reservoir_add(self, group: str, example: Dict[str, Any]) -> None:
        """用 reservoir sampling 从未知长度的数据流中等概率保留固定数量样例。"""
        self.eligible_counts[group] += 1
        bucket = self.correct_examples if group == "correct" else self.incorrect_examples
        seen = self.eligible_counts[group]

        if len(bucket) < self.num_examples:
            bucket.append(example)
            return

        # 看过 seen 个样例后，新样例以 num_examples / seen 的概率进入蓄水池；
        # 最终每个合格 query 被选中的概率相同，内存复杂度始终为 O(num_examples)。
        replacement_index = self._rng[group].randrange(seen)
        if replacement_index < self.num_examples:
            bucket[replacement_index] = example

    def save(
        self,
        output_dir: str,
        keep_ratio: float,
        image_binary_loader: Optional[ImageBinaryLoader] = None,
    ) -> Dict[str, str]:
        """保存原始抽样分数、统计 JSON、整体图和逐 query 样例图。"""
        ensure_matplotlib_available()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        keep_percent = int(round(keep_ratio * 100))
        raw_path = output_path / f"selected_token_similarity_examples_keep{keep_percent}.pt"
        summary_path = output_path / f"selected_token_similarity_examples_keep{keep_percent}.json"
        histogram_path = output_path / f"selected_token_similarity_histograms_keep{keep_percent}.json"
        comparison_path = output_path / (
            f"all_query_candidate_similarity_summary_keep{keep_percent}.json"
        )
        outcome_summary_path = output_path / (
            f"top1_outcome_token_similarity_summary_keep{keep_percent}.json"
        )
        overall_path = output_path / f"overall_token_similarity_keep{keep_percent}.png"
        outcome_figure_path = output_path / (
            f"top1_outcome_token_similarity_keep{keep_percent}.png"
        )
        correct_dir = output_path / f"recall1_correct_examples_keep{keep_percent}"
        incorrect_dir = output_path / f"recall1_incorrect_examples_keep{keep_percent}"

        comparison_summary = self._all_query_candidate_comparison_summary(keep_ratio)
        comparison_path.write_text(
            json.dumps(comparison_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outcome_summary = self._top1_outcome_similarity_summary(keep_ratio)
        outcome_summary_path.write_text(
            json.dumps(outcome_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # .pt 保留抽中样例的逐 token tensor，便于以后换 bin 或重新画图；
        # .json 只保存统计摘要和候选身份，方便直接检查而不依赖 PyTorch。
        payload = {
            "keep_ratio": keep_ratio,
            "seed": self.seed,
            "num_examples_requested": self.num_examples,
            "bin_edges": torch.from_numpy(self.bin_edges.copy()),
            "correct_examples": self.correct_examples,
            "incorrect_examples": self.incorrect_examples,
        }
        torch.save(payload, raw_path)

        summary = {
            "keep_ratio": keep_ratio,
            "total_queries_seen": self.total_queries,
            "eligible_queries": dict(self.eligible_counts),
            "selected_queries": {
                "correct": len(self.correct_examples),
                "incorrect": len(self.incorrect_examples),
            },
            "skipped_queries": dict(sorted(self.skipped_counts.items())),
            "selection_seed": self.seed,
            "selection_rule": {
                "correct": "final Top1 GT plus the three highest-ranked non-GT candidates",
                "incorrect": (
                    "final Top1 non-GT plus the next two highest-ranked non-GT "
                    "candidates and the highest-ranked GT candidate"
                ),
            },
            "all_query_candidate_comparison": comparison_summary,
            "correct_examples": [
                self._example_summary(item) for item in self.correct_examples
            ],
            "incorrect_examples": [
                self._example_summary(item) for item in self.incorrect_examples
            ],
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._plot_overall(overall_path, keep_ratio)
        self._plot_top1_outcome(
            outcome_figure_path,
            keep_ratio,
            outcome_summary,
        )
        correct_figure_paths = self._plot_examples(
            self.correct_examples,
            correct_dir,
            keep_ratio,
            group="correct",
        )
        incorrect_figure_paths = self._plot_examples(
            self.incorrect_examples,
            incorrect_dir,
            keep_ratio,
            group="incorrect",
        )

        # 这个文件与逐样例图片一一对应。每个候选都记录 all/pruned/kept 三条
        # 直方图在每个固定区间中的原始 token 数量，以及图片实际绘制的 density。
        histogram_export = self._histogram_export(keep_ratio)
        histogram_export["all_query_candidate_comparison"] = comparison_summary
        histogram_export["figure_files"] = {
            "correct": [
                path.relative_to(output_path).as_posix()
                for path in correct_figure_paths
            ],
            "incorrect": [
                path.relative_to(output_path).as_posix()
                for path in incorrect_figure_paths
            ],
        }
        histogram_path.write_text(
            json.dumps(histogram_export, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        output_paths = {
            "raw_examples": str(raw_path),
            "summary": str(summary_path),
            "histogram_counts": str(histogram_path),
            "all_query_candidate_summary": str(comparison_path),
            "top1_outcome_summary": str(outcome_summary_path),
            "overall_figure": str(overall_path),
            "top1_outcome_figure": str(outcome_figure_path),
            "correct_figure_directory": str(correct_dir),
            "incorrect_figure_directory": str(incorrect_dir),
        }

        # 图片读取器是可选依赖：普通单元测试或只重画历史 tensor 时保持原有行为；
        # evaluate.py 显式传入读取器后，才恢复最终抽中样例的候选页面。这里复用与
        # 分布图完全相同的两个 reservoir，保证图片和图中四个候选逐一对应。
        if image_binary_loader is not None:
            output_paths.update(
                export_selected_similarity_candidate_images(
                    example_groups={
                        "correct": self.correct_examples,
                        "incorrect": self.incorrect_examples,
                    },
                    output_dir=str(output_path),
                    keep_ratio=keep_ratio,
                    image_binary_loader=image_binary_loader,
                )
            )

        return output_paths

    @staticmethod
    def _candidate_summary(candidate: Mapping[str, Any]) -> Dict[str, Any]:
        scores = candidate["scores"].numpy()
        return {
            "role": candidate["role"],
            "role_label": candidate["role_label"],
            "candidate_pos": candidate["candidate_pos"],
            "candidate_letter": candidate["candidate_letter"],
            "final_rank": candidate["final_rank"],
            "global_idx": candidate["global_idx"],
            "local_page_id": candidate["local_page_id"],
            "is_gt": candidate["is_gt"],
            "num_original_tokens": candidate["num_original_tokens"],
            "num_kept_tokens": candidate["num_kept_tokens"],
            "threshold": candidate["threshold"],
            "similarity": {
                "min": float(np.min(scores)),
                "mean": float(np.mean(scores)),
                # 与全查询汇总保持同一总体方差定义，便于直接核对抽中候选。
                "variance": float(np.var(scores, ddof=0)),
                "median": float(np.median(scores)),
                "p90": float(np.quantile(scores, 0.9)),
                "max": float(np.max(scores)),
            },
        }

    def _example_summary(self, example: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "qid": example["qid"],
            "doc_name": example["doc_name"],
            "domain": example["domain"],
            "q_idx": example["q_idx"],
            "query": example["query"],
            "ground_truth_page_ids": example["ground_truth_page_ids"],
            "recall_at_1": example["recall_at_1"],
            "recall1_correct": example["recall1_correct"],
            "entropy_similarity_range": example["entropy_similarity_range"],
            "candidates": [
                self._candidate_summary(candidate)
                for candidate in example["candidates"]
            ],
        }

    def _candidate_histogram_export(
        self,
        candidate: Mapping[str, Any],
        entropy_similarity_range: tuple[float, float],
    ) -> Dict[str, Any]:
        """导出一个候选子图中三条直方图的逐 bin 数量和密度。"""
        exported = self._candidate_summary(candidate)
        distributions = {}
        for group_name, group_scores in self._candidate_score_groups(candidate).items():
            counts, density = self._histogram_counts_and_density(group_scores)
            entropy_counts = self._entropy_histogram_counts(
                group_scores,
                entropy_similarity_range,
            )
            raw_values = group_scores.numpy()
            distributions[group_name] = {
                "num_tokens": int(group_scores.numel()),
                # 绘图会把显示范围外的值夹到首尾 bin；单独记录数量便于识别这种情况。
                "num_clipped_below_range": int(
                    np.count_nonzero(raw_values < self.bin_edges[0])
                ),
                "num_clipped_above_range": int(
                    np.count_nonzero(raw_values > self.bin_edges[-1])
                ),
                "counts": counts.astype(np.int64).tolist(),
                "density": density.tolist(),
                # all、pruned、kept 都按各自包含的 token 计算均值和总体方差；
                # 其中 all 与 candidate.similarity 中的同名统计量一致。
                "mean": float(np.mean(raw_values)),
                "variance": float(np.var(raw_values, ddof=0)),
                "entropy": self._histogram_entropy(entropy_counts),
            }
        exported["distributions"] = distributions
        return exported

    def _example_histogram_export(
        self,
        example: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """导出图片中一整行 query 的候选身份和直方图数据。"""
        entropy_range_values = example["entropy_similarity_range"]
        entropy_similarity_range = (
            float(entropy_range_values[0]),
            float(entropy_range_values[1]),
        )
        return {
            "qid": example["qid"],
            "doc_name": example["doc_name"],
            "domain": example["domain"],
            "q_idx": example["q_idx"],
            "query": example["query"],
            "ground_truth_page_ids": example["ground_truth_page_ids"],
            "recall_at_1": example["recall_at_1"],
            "recall1_correct": example["recall1_correct"],
            "entropy_similarity_range": example["entropy_similarity_range"],
            "candidates": [
                self._candidate_histogram_export(
                    candidate,
                    entropy_similarity_range,
                )
                for candidate in example["candidates"]
            ],
        }

    def _histogram_export(self, keep_ratio: float) -> Dict[str, Any]:
        """构造可独立解读并能精确复现逐样例直方图的 JSON 数据。"""
        bin_definitions = []
        for bin_index, (left_edge, right_edge) in enumerate(
            zip(self.bin_edges[:-1], self.bin_edges[1:])
        ):
            bin_definitions.append({
                "bin_index": bin_index,
                "left_edge": float(left_edge),
                "right_edge": float(right_edge),
                "left_inclusive": True,
                # np.histogram 只有最后一个区间包含右端点。
                "right_inclusive": bin_index == self.num_bins - 1,
            })

        return {
            "keep_ratio": keep_ratio,
            "display_range": [float(self.bin_edges[0]), float(self.bin_edges[-1])],
            "num_bins": self.num_bins,
            "bins": bin_definitions,
            "distribution_order": ["all", "pruned", "kept"],
            "count_semantics": (
                "Raw visual-token counts after values outside display_range are "
                "clipped into the first or last bin."
            ),
            "density_semantics": (
                "counts / (num_tokens * bin_width); these are the y-values "
                "drawn by the example figures."
            ),
            "entropy_semantics": {
                "shannon_bits": "-sum(p_i * log2(p_i)) over non-empty bins.",
                "normalized": (
                    "shannon_bits / log2(num_bins): 0 means all mass is in one "
                    "bin; 1 means uniform mass across all bins."
                ),
                "effective_bins": (
                    "2 ** shannon_bits: the equivalent number of equally occupied bins."
                ),
                "bin_range": (
                    "Each query uses its raw minimum and maximum similarity across "
                    "all tokens from all first-stage Top-K candidates. The same "
                    "query-specific range is used for all candidate and token groups."
                ),
            },
            "groups": {
                "correct": [
                    self._example_histogram_export(example)
                    for example in self.correct_examples
                ],
                "incorrect": [
                    self._example_histogram_export(example)
                    for example in self.incorrect_examples
                ],
            },
        }

    def _plot_top1_outcome(
        self,
        output_path: Path,
        keep_ratio: float,
        summary: Mapping[str, Any],
    ) -> None:
        """Plot the two additive Top1-conditioned similarity distributions."""
        plt = importlib.import_module("matplotlib.pyplot")
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True, sharey=True)
        class_styles = {
            "correct_candidates": {
                "label": "Selected correct candidate(s)",
                "color": "#2ca02c",
                "linestyle": "-",
                "marker": "o",
                "markevery": (0, 10),
            },
            "incorrect_candidates": {
                "label": "Selected incorrect candidate(s)",
                "color": "#d62728",
                "linestyle": "--",
                "marker": "s",
                "markevery": (5, 10),
            },
        }
        group_titles = {
            "top1_correct": "Top1 correct",
            "top1_incorrect": "Top1 incorrect",
        }

        for axis, outcome_group in zip(axes, TOP1_OUTCOME_GROUPS):
            group_summary = summary["groups"][outcome_group]
            plotted = False
            for candidate_class in OUTCOME_CANDIDATE_CLASSES:
                candidate_summary = group_summary[candidate_class]
                distribution = candidate_summary[
                    "token_similarity_distribution"
                ]
                mean_density = distribution["mean_density"]
                if mean_density is None:
                    continue

                style = class_styles[candidate_class]
                axis.plot(
                    centers,
                    mean_density,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2,
                    marker=style["marker"],
                    markevery=style["markevery"],
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    label=style["label"],
                )
                axis.fill_between(
                    centers,
                    distribution["min_density"],
                    distribution["max_density"],
                    color=style["color"],
                    alpha=0.14,
                )
                mean_threshold = candidate_summary[
                    "mean_pruning_threshold_similarity"
                ]
                if mean_threshold is not None:
                    axis.axvline(
                        mean_threshold,
                        color=style["color"],
                        linestyle=":",
                        linewidth=1.2,
                        alpha=0.85,
                        label=(
                            f"{style['label']} mean threshold="
                            f"{mean_threshold:.4f}"
                        ),
                    )
                plotted = True

            axis.set_title(
                f"{group_titles[outcome_group]}\n"
                f"Eligible queries: {group_summary['num_queries']}"
            )
            axis.set_xlim(float(self.bin_edges[0]), float(self.bin_edges[-1]))
            axis.set_xlabel("Max cosine similarity per visual token")
            axis.grid(alpha=0.2)
            if plotted:
                axis.legend(fontsize=8)
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No eligible queries",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )

        axes[0].set_ylabel("Query-balanced density")
        fig.suptitle(
            f"Token similarity by reranking Top1 outcome (keep {keep_ratio:.0%})\n"
            "Shaded region: per-bin min-max range across query samples",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_overall(self, output_path: Path, keep_ratio: float) -> None:
        """绘制全部合格 query 的等权平均分布，而不是只统计抽中的 5 个样例。"""
        plt = importlib.import_module("matplotlib.pyplot")

        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)

        # 颜色之外再使用线型和错位 marker 区分角色。某些候选的总体分布几乎完全
        # 重合，仅依赖颜色时，后绘制的绿色曲线会覆盖红色曲线；dash 间隙和不同位置
        # 的 marker 可以同时暴露两组数据，又不需要人为平移曲线而改变横轴含义。
        specifications = {
            "correct": [
                ("top1_correct", "Top1 correct (GT)", "#2ca02c", "-", "o", 0),
                (
                    "other_negatives",
                    "Other 3 negatives (query mean)",
                    "#6c757d",
                    "--",
                    "s",
                    4,
                ),
            ],
            "incorrect": [
                ("top1_incorrect", "Top1 incorrect", "#d62728", "-", "o", 0),
                (
                    "other_negatives",
                    "Other 2 negatives (query mean)",
                    "#6c757d",
                    ":",
                    "s",
                    3,
                ),
                ("ground_truth", "Highest-ranked GT", "#2ca02c", "-.", "^", 6),
            ],
        }

        for ax, group in zip(axes, ("correct", "incorrect")):
            plotted = False
            for role, label, color, line_style, marker, marker_offset in specifications[group]:
                histograms = self._group_histograms[group].get(role, [])
                if not histograms:
                    continue
                values = np.stack(histograms, axis=0)
                mean = values.mean(axis=0)
                ax.plot(
                    centers,
                    mean,
                    color=color,
                    linestyle=line_style,
                    linewidth=2,
                    marker=marker,
                    markevery=(marker_offset, 10),
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    label=label,
                )
                # 每个 bin 独立取所有 query 样例的最小值和最大值；只有一个样例时，
                # 上下界都等于该样例曲线，因此阴影自然退化为零宽度。
                ax.fill_between(
                    centers,
                    values.min(axis=0),
                    values.max(axis=0),
                    color=color,
                    alpha=0.16,
                )
                plotted = True

            title_label = "Recall@1 correct" if group == "correct" else "Recall@1 incorrect (GT in Top-K)"
            ax.set_title(f"{title_label}\nEligible queries: {self.eligible_counts[group]}")
            ax.set_xlim(0.0, 0.25)
            ax.set_xlabel("Max cosine similarity per visual token")
            ax.grid(alpha=0.2)
            if plotted:
                ax.legend(fontsize=9)
            else:
                ax.text(0.5, 0.5, "No eligible queries", ha="center", va="center", transform=ax.transAxes)

        axes[0].set_ylabel("Query-balanced density")
        fig.suptitle(
            f"Overall QI-Early token similarity distributions (keep {keep_ratio:.0%})\n"
            "Shaded region: per-bin min-max range across query samples",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_examples(
        self,
        examples: Sequence[Mapping[str, Any]],
        output_dir: Path,
        keep_ratio: float,
        group: str,
    ) -> List[Path]:
        """每个 query 单独保存一张图，图内四个候选各占一个子图。"""
        plt = importlib.import_module("matplotlib.pyplot")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 同一候选子图中的颜色固定表示 token 状态，而候选角色由标题说明。
        # 三组分数分别归一化为面积为 1 的密度，比较的是分布形状而非 token 数量。
        distribution_styles = {
            "all": {
                "label": "All tokens",
                "color": "#4c4c4c",
                "linestyle": "-",
                "linewidth": 1.6,
            },
            "pruned": {
                "label": "Pruned tokens",
                "color": "#d62728",
                "linestyle": "--",
                "linewidth": 1.7,
            },
            "kept": {
                "label": "Kept tokens",
                "color": "#2ca02c",
                "linestyle": "-.",
                "linewidth": 1.7,
            },
        }
        generated_paths = []

        for example_index, example in enumerate(examples):
            # 一张图片只对应一个 query；四列分别展示该 query 选出的四个候选。
            # 共用纵轴尺度，便于在同一样例内部比较三个分布的形状和峰值。
            fig, axes = plt.subplots(
                1,
                4,
                figsize=(22, 5.4),
                squeeze=False,
                sharex=True,
                sharey=True,
            )
            candidate_axes = axes[0]
            legend_handles = None

            # 标题在每个候选子图中都保留 query ID 和截断后的 query 文本，确保单独
            # 查看或裁剪某个子图时，仍能知道它属于哪个查询。
            short_query = textwrap.shorten(
                str(example["query"]).replace("\n", " "),
                width=55,
                placeholder="...",
            )

            for column_index, candidate in enumerate(example["candidates"]):
                ax = candidate_axes[column_index]

                # all_scores 是剪枝前的全部 token；另外两组由实际 kept_mask 精确切分，
                # 而不是仅根据阈值重新推断，因此阈值处存在并列分数时也不会分错。
                score_groups = self._candidate_score_groups(candidate)

                subplot_handles = []
                for score_group, group_scores in score_groups.items():
                    style = distribution_styles[score_group]
                    histogram = self._normalized_histogram(group_scores)
                    subplot_handles.append(
                        ax.stairs(
                            histogram,
                            self.bin_edges,
                            color=style["color"],
                            linestyle=style["linestyle"],
                            linewidth=style["linewidth"],
                            label=style["label"],
                        )
                    )

                # 每个候选独立做 Top-K token 剪枝，所以阈值也按图像分别绘制。
                # 分布按 kept_mask 精确切分；虚线只用于直观标出 Top-K 的分界位置。
                threshold_handle = ax.axvline(
                    candidate["threshold"],
                    color="#9467bd",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.8,
                    label="Keep threshold",
                )
                if legend_handles is None:
                    legend_handles = [*subplot_handles, threshold_handle]

                ax.set_title(
                    f"Example {example_index + 1} | {example['qid']}\n"
                    f"{short_query}\n"
                    f"{candidate['role_label']}: [{candidate['candidate_letter']}] "
                    f"page={candidate['local_page_id']}, "
                    f"rank={candidate['final_rank']}, cut={candidate['threshold']:.3f}",
                    fontsize=8.5,
                )
                ax.set_xlim(0.0, 0.25)
                ax.set_xlabel("Max cosine similarity per visual token")
                if column_index == 0:
                    ax.set_ylabel("Density")
                ax.grid(alpha=0.2)

            # 正常情况下固定为四个候选；如果以后调整选择规则，隐藏多余子图。
            for unused_ax in candidate_axes[len(example["candidates"]):]:
                unused_ax.set_visible(False)

            group_title = (
                "Recall@1 correct"
                if group == "correct"
                else "Recall@1 incorrect (GT present in Top-K)"
            )
            fig.suptitle(
                f"{group_title}: example {example_index + 1}/{len(examples)} "
                f"(keep {keep_ratio:.0%})\n"
                "Each candidate subplot shows all, pruned, and kept visual-token distributions",
                fontsize=13,
            )
            if legend_handles is not None:
                fig.legend(
                    handles=legend_handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.90),
                    ncol=4,
                    fontsize=9,
                )
            fig.tight_layout(rect=(0, 0, 1, 0.82))

            output_path = output_dir / f"example_{example_index + 1:02d}.png"
            fig.savefig(output_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            generated_paths.append(output_path)

        return generated_paths

    def report(self) -> str:
        """生成便于写入评估日志的样例计数和跳过原因摘要。"""
        skipped = ", ".join(
            f"{reason}={count}" for reason, count in sorted(self.skipped_counts.items())
        ) or "none"
        return (
            f"Token-similarity analysis: queries={self.total_queries}, "
            f"eligible_correct={self.eligible_counts['correct']}, "
            f"eligible_incorrect={self.eligible_counts['incorrect']}, "
            f"selected_correct={len(self.correct_examples)}, "
            f"selected_incorrect={len(self.incorrect_examples)}, skipped=({skipped})"
        )
