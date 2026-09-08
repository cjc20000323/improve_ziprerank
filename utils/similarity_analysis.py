"""收集并绘制 QI-Early 视觉 token 相似度诊断结果。

这里同时维护两类数据：

1. 整体分布：每张候选图先转换成归一化直方图，同一 query 中重复的负例角色
   再取平均，最后跨 query 求均值。这样每个 query 的权重相同，不会让高分辨率、
   token 更多的图像主导整体曲线。
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
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence

import numpy as np
import torch


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
    ) -> None:
        if num_examples <= 0:
            raise ValueError(f"num_examples must be positive, got {num_examples}")
        if num_bins <= 1:
            raise ValueError(f"num_bins must be greater than 1, got {num_bins}")

        self.num_examples = num_examples
        self.seed = seed
        self.num_bins = num_bins
        # 所有候选共用固定的 [-0.25, 0.25] bin 边界，保证不同 query 的分布可直接平均。
        self.bin_edges = np.linspace(-0.25, 0.25, num_bins + 1, dtype=np.float32)

        self.correct_examples: List[Dict[str, Any]] = []
        self.incorrect_examples: List[Dict[str, Any]] = []
        self.eligible_counts = {"correct": 0, "incorrect": 0}
        self.total_queries = 0
        self.skipped_counts: DefaultDict[str, int] = defaultdict(int)

        # 两组使用独立随机数发生器：正确组抽到哪些 query，不受错误组数量影响。
        self._rng = {
            "correct": random.Random(seed),
            "incorrect": random.Random(seed + 1),
        }
        self._group_histograms: Dict[str, DefaultDict[str, List[np.ndarray]]] = {
            "correct": defaultdict(list),
            "incorrect": defaultdict(list),
        }

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

    def _normalized_histogram(self, scores: torch.Tensor) -> np.ndarray:
        """将一张图的所有 token 分数转换成积分为 1 的概率密度直方图。"""
        # 将分数限制到当前展示区间，确保每个 token 都被计入首尾之间的某个 bin。
        values = np.clip(scores.numpy(), -0.25, 0.25)
        counts, _ = np.histogram(values, bins=self.bin_edges)
        total = counts.sum()
        if total == 0:
            raise ValueError("No token similarities fell inside display range [-0.25, 0.25]")
        widths = np.diff(self.bin_edges)
        # 除以 token 总数和 bin 宽度后，曲线面积为 1；不同 token 数的图才可比较。
        return counts.astype(np.float64) / (total * widths)

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

    def save(self, output_dir: str, keep_ratio: float) -> Dict[str, str]:
        """保存原始抽样分数、可读元数据、整体图和两组逐样例图。"""
        ensure_matplotlib_available()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        keep_percent = int(round(keep_ratio * 100))
        raw_path = output_path / f"selected_token_similarity_examples_keep{keep_percent}.pt"
        summary_path = output_path / f"selected_token_similarity_examples_keep{keep_percent}.json"
        overall_path = output_path / f"overall_token_similarity_keep{keep_percent}.png"
        correct_path = output_path / f"recall1_correct_examples_keep{keep_percent}.png"
        incorrect_path = output_path / f"recall1_incorrect_examples_keep{keep_percent}.png"

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
        self._plot_examples(
            self.correct_examples,
            correct_path,
            keep_ratio,
            group="correct",
        )
        self._plot_examples(
            self.incorrect_examples,
            incorrect_path,
            keep_ratio,
            group="incorrect",
        )

        return {
            "raw_examples": str(raw_path),
            "summary": str(summary_path),
            "overall_figure": str(overall_path),
            "correct_figure": str(correct_path),
            "incorrect_figure": str(incorrect_path),
        }

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
            "candidates": [
                self._candidate_summary(candidate)
                for candidate in example["candidates"]
            ],
        }

    def _plot_overall(self, output_path: Path, keep_ratio: float) -> None:
        """绘制全部合格 query 的等权平均分布，而不是只统计抽中的 5 个样例。"""
        plt = importlib.import_module("matplotlib.pyplot")

        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)
        specifications = {
            "correct": [
                ("top1_correct", "Top1 correct (GT)", "#2ca02c"),
                ("other_negatives", "Other 3 negatives (query mean)", "#6c757d"),
            ],
            "incorrect": [
                ("top1_incorrect", "Top1 incorrect", "#d62728"),
                ("other_negatives", "Other 2 negatives (query mean)", "#6c757d"),
                ("ground_truth", "Highest-ranked GT", "#2ca02c"),
            ],
        }

        for ax, group in zip(axes, ("correct", "incorrect")):
            plotted = False
            for role, label, color in specifications[group]:
                histograms = self._group_histograms[group].get(role, [])
                if not histograms:
                    continue
                values = np.stack(histograms, axis=0)
                mean = values.mean(axis=0)
                ax.plot(centers, mean, color=color, linewidth=2, label=label)
                if len(values) > 1:
                    # 阴影是在各 query 直方图之间估计的均值 95% 置信区间。
                    ci = 1.96 * values.std(axis=0, ddof=1) / math.sqrt(len(values))
                    ax.fill_between(
                        centers,
                        np.maximum(mean - ci, 0.0),
                        mean + ci,
                        color=color,
                        alpha=0.16,
                    )
                plotted = True

            title_label = "Recall@1 correct" if group == "correct" else "Recall@1 incorrect (GT in Top-K)"
            ax.set_title(f"{title_label}\nEligible queries: {self.eligible_counts[group]}")
            ax.set_xlim(-0.25, 0.25)
            ax.set_xlabel("Max cosine similarity per visual token")
            ax.grid(alpha=0.2)
            if plotted:
                ax.legend(fontsize=9)
            else:
                ax.text(0.5, 0.5, "No eligible queries", ha="center", va="center", transform=ax.transAxes)

        axes[0].set_ylabel("Query-balanced density")
        fig.suptitle(
            f"Overall QI-Early token similarity distributions (keep {keep_ratio:.0%})\n"
            "Shaded region: query-level normal-approximation 95% CI",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_examples(
        self,
        examples: Sequence[Mapping[str, Any]],
        output_path: Path,
        keep_ratio: float,
        group: str,
    ) -> None:
        """每行展示一个 query，并把它的四个候选分别画在四个独立子图中。"""
        plt = importlib.import_module("matplotlib.pyplot")

        # 每个样例固定选择四个候选，因此使用“一行一个 query、四列四个候选”的布局。
        # sharey="row" 让同一 query 的四张图共用纵轴尺度，便于直接比较密度高低。
        num_rows = max(len(examples), 1)
        num_columns = 4
        fig, axes = plt.subplots(
            num_rows,
            num_columns,
            figsize=(22, 4.2 * num_rows),
            squeeze=False,
            sharex=True,
            sharey="row",
        )

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
        legend_handles = None

        if not examples:
            axes[0, 0].text(
                0.5,
                0.5,
                "No eligible queries",
                ha="center",
                va="center",
                transform=axes[0, 0].transAxes,
            )
            axes[0, 0].set_axis_off()
            for unused_ax in axes[0, 1:]:
                unused_ax.set_visible(False)

        for row_index, example in enumerate(examples):
            # 标题在每个候选子图中都保留 query ID 和截断后的 query 文本，确保单独
            # 查看或裁剪某个子图时，仍能知道它属于哪个查询。
            short_query = textwrap.shorten(
                str(example["query"]).replace("\n", " "),
                width=55,
                placeholder="...",
            )

            for column_index, candidate in enumerate(example["candidates"]):
                ax = axes[row_index, column_index]
                scores = candidate["scores"]
                kept_mask = candidate["kept_mask"]
                pruned_mask = ~kept_mask

                # all_scores 是剪枝前的全部 token；另外两组由实际 kept_mask 精确切分，
                # 而不是仅根据阈值重新推断，因此阈值处存在并列分数时也不会分错。
                score_groups = {
                    "all": scores,
                    "pruned": scores[pruned_mask],
                    "kept": scores[kept_mask],
                }
                if score_groups["pruned"].numel() == 0 or score_groups["kept"].numel() == 0:
                    raise ValueError(
                        f"Candidate {candidate['candidate_letter']} of {example['qid']} "
                        "must contain both pruned and kept visual tokens"
                    )

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
                    f"Example {row_index + 1} | {example['qid']}\n"
                    f"{short_query}\n"
                    f"{candidate['role_label']}: [{candidate['candidate_letter']}] "
                    f"page={candidate['local_page_id']}, "
                    f"rank={candidate['final_rank']}, cut={candidate['threshold']:.3f}",
                    fontsize=8.5,
                )
                ax.set_xlim(-0.25, 0.25)
                ax.set_xlabel("Max cosine similarity per visual token")
                if column_index == 0:
                    ax.set_ylabel("Density")
                ax.grid(alpha=0.2)

            # 正常情况下固定为四个候选；如果以后调整选择规则，隐藏该行多余子图。
            for unused_ax in axes[row_index, len(example["candidates"]):]:
                unused_ax.set_visible(False)

        group_title = "Recall@1 correct" if group == "correct" else "Recall@1 incorrect (GT present in Top-K)"
        fig.suptitle(
            f"{group_title}: selected token-similarity examples (keep {keep_ratio:.0%})\n"
            "Each candidate subplot shows all, pruned, and kept visual-token distributions",
            fontsize=13,
        )
        if legend_handles is not None:
            fig.legend(
                handles=legend_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=4,
                fontsize=9,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

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
