import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from utils.similarity_analysis import TokenSimilarityAnalysisCollector


class TokenSimilarityAnalysisCollectorTest(unittest.TestCase):
    @staticmethod
    def _candidate_stats(num_candidates=5):
        records = []
        for candidate_pos in range(num_candidates):
            scores = torch.tensor([-0.2, -0.1, 0.1, 0.2]) + candidate_pos * 0.01
            records.append({
                "candidate_pos": candidate_pos,
                "scores": scores,
                "kept_mask": torch.tensor([False, False, True, True]),
                "threshold": float(scores[2]),
                "num_original_tokens": 4,
                "num_kept_tokens": 2,
            })
        return records

    @staticmethod
    def _result(ranked_indices, ground_truth_page_ids):
        return {
            "qid": "doc_0",
            "doc_name": "doc",
            "domain": "News",
            "q_idx": 0,
            "query": "test query",
            "page_id": ground_truth_page_ids,
            "start_idx": 100,
            "end_idx": 109,
            # Candidate local page IDs are [1, 2, 3, 4, 5].
            "top_k_global_indices": [101, 102, 103, 104, 105],
            "ranked_indices": ranked_indices,
        }

    def test_correct_example_uses_top1_gt_and_three_highest_negatives(self):
        # Top1 的 candidate_pos=2 对应 GT 页；应再按最终名次取前三个非 GT。
        collector = TokenSimilarityAnalysisCollector(num_examples=5, seed=7)
        result = self._result(
            ranked_indices=[2, 0, 1, 3, 4],
            ground_truth_page_ids=[3],
        )

        collector.add_query(result, self._candidate_stats())

        self.assertEqual(collector.eligible_counts["correct"], 1)
        example = collector.correct_examples[0]
        self.assertEqual(
            [item["candidate_pos"] for item in example["candidates"]],
            [2, 0, 1, 3],
        )
        self.assertEqual(
            [item["role"] for item in example["candidates"]],
            ["top1_correct", "negative_1", "negative_2", "negative_3"],
        )

    def test_zero_pruning_supports_an_empty_pruned_token_group(self):
        collector = TokenSimilarityAnalysisCollector(num_examples=5, seed=7)
        candidate_stats = self._candidate_stats()
        for item in candidate_stats:
            item["kept_mask"] = torch.ones(4, dtype=torch.bool)
            item["threshold"] = float(item["scores"].min())
            item["num_kept_tokens"] = 4

        collector.add_query(
            self._result(
                ranked_indices=[2, 0, 1, 3, 4],
                ground_truth_page_ids=[3],
            ),
            candidate_stats,
        )

        summary = collector._all_query_candidate_comparison_summary(
            keep_ratio=1.0
        )
        metrics = summary["metrics"]["ground_truth_candidates"]
        self.assertEqual(metrics["num_candidates_with_pruned_tokens"], 0)
        self.assertIsNone(metrics["mean_pruned_token_similarity"])
        self.assertIsNone(metrics["mean_pruned_token_similarity_variance"])
        self.assertEqual(
            metrics["mean_pruned_token_similarity_entropy"],
            {
                "shannon_bits": None,
                "normalized": None,
                "effective_bins": None,
            },
        )
        self.assertEqual(metrics["num_candidates_with_kept_tokens"], 1)
        self.assertAlmostEqual(
            metrics["mean_all_token_similarity"],
            metrics["mean_kept_token_similarity"],
        )

        example = collector.correct_examples[0]
        exported = collector._candidate_histogram_export(
            example["candidates"][0],
            tuple(example["entropy_similarity_range"]),
        )
        pruned = exported["distributions"]["pruned"]
        self.assertEqual(pruned["num_tokens"], 0)
        self.assertEqual(sum(pruned["counts"]), 0)
        self.assertTrue(all(value == 0.0 for value in pruned["density"]))
        self.assertIsNone(pruned["mean"])
        self.assertIsNone(pruned["variance"])
        self.assertEqual(
            pruned["entropy"],
            {
                "shannon_bits": None,
                "normalized": None,
                "effective_bins": None,
            },
        )

    def test_incorrect_example_uses_top1_two_negatives_and_best_gt(self):
        # Top1 为非 GT；应保留它、其后两个非 GT，以及最终排名最高的 GT(pos=2)。
        collector = TokenSimilarityAnalysisCollector(num_examples=5, seed=7)
        result = self._result(
            ranked_indices=[0, 1, 3, 2, 4],
            ground_truth_page_ids=[3],
        )

        collector.add_query(result, self._candidate_stats())

        self.assertEqual(collector.eligible_counts["incorrect"], 1)
        example = collector.incorrect_examples[0]
        self.assertEqual(
            [item["candidate_pos"] for item in example["candidates"]],
            [0, 1, 3, 2],
        )
        self.assertEqual(
            [item["role"] for item in example["candidates"]],
            ["top1_incorrect", "negative_1", "negative_2", "ground_truth"],
        )

    def test_all_query_comparison_uses_best_ranked_gt_and_query_mean(self):
        collector = TokenSimilarityAnalysisCollector(
            num_examples=5,
            seed=7,
            comparison_num_ground_truth_candidates=1,
            comparison_num_incorrect_candidates=1,
        )
        result = self._result(
            # GT candidate_pos=4 排在 candidate_pos=2 之前，因此应选择 pos=4。
            ranked_indices=[0, 4, 1, 2, 3],
            ground_truth_page_ids=[3, 5],
        )

        first_stats = self._candidate_stats()
        collector.add_query(result, first_stats)

        second_stats = self._candidate_stats()
        for item in second_stats:
            item["threshold"] += 0.02
        collector.add_query(result, second_stats)

        summary = collector._all_query_candidate_comparison_summary(keep_ratio=0.5)
        self.assertEqual(summary["eligible_paired_queries"], 2)
        incorrect = summary["metrics"]["top_ranked_incorrect_candidates"]
        correct = summary["metrics"]["ground_truth_candidates"]
        # 两次分别为 0.10/0.12；query 等权均值为 0.11。
        self.assertAlmostEqual(
            incorrect["mean_pruning_threshold_similarity"],
            0.11,
        )
        # 最高排名 GT 是 candidate_pos=4，两次分别为 0.14/0.16。
        self.assertAlmostEqual(
            correct["mean_pruning_threshold_similarity"],
            0.15,
        )

    def test_all_query_comparison_defaults_to_all_gt_and_top_three_incorrect(self):
        collector = TokenSimilarityAnalysisCollector(num_examples=5, seed=7)
        result = self._result(
            ranked_indices=[0, 4, 1, 2, 3],
            ground_truth_page_ids=[3, 5],
        )

        first_stats = self._candidate_stats()
        collector.add_query(result, first_stats)
        second_stats = self._candidate_stats()
        for item in second_stats:
            item["threshold"] += 0.02
        collector.add_query(result, second_stats)

        # 每个 query 的 GT 为 pos=4/2，错误候选为最终排名最靠前的 pos=0/1/3。
        # 两个 query 共贡献 4 个 GT 与 6 个错误候选，所有均值按候选等权计算。
        summary = collector._all_query_candidate_comparison_summary(keep_ratio=0.5)
        ground_truth = summary["metrics"]["ground_truth_candidates"]
        incorrect = summary["metrics"]["top_ranked_incorrect_candidates"]
        self.assertEqual(ground_truth["num_queries"], 2)
        self.assertEqual(ground_truth["num_candidates"], 4)
        self.assertEqual(incorrect["num_candidates"], 6)
        self.assertAlmostEqual(
            ground_truth["mean_pruning_threshold_similarity"],
            0.14,
        )
        self.assertAlmostEqual(
            incorrect["mean_pruning_threshold_similarity"],
            0.12333333333333334,
        )
        self.assertAlmostEqual(
            ground_truth["mean_all_token_similarity"],
            0.03,
        )
        self.assertAlmostEqual(
            incorrect["mean_all_token_similarity"],
            0.013333333333333334,
        )
        self.assertAlmostEqual(
            ground_truth["mean_pruned_token_similarity"],
            -0.12,
        )
        self.assertAlmostEqual(
            ground_truth["mean_kept_token_similarity"],
            0.18,
        )
        self.assertAlmostEqual(
            incorrect["mean_pruned_token_similarity"],
            -0.13666666666666666,
        )
        self.assertAlmostEqual(
            incorrect["mean_kept_token_similarity"],
            0.16333333333333333,
        )
        # 所有候选只发生整体平移，token 离均差不变，因此两类候选的平均
        # 总体方差都应保持为 0.025。
        self.assertAlmostEqual(
            ground_truth["mean_all_token_similarity_variance"],
            0.025,
        )
        self.assertAlmostEqual(
            incorrect["mean_all_token_similarity_variance"],
            0.025,
        )
        for metrics in (ground_truth, incorrect):
            self.assertAlmostEqual(
                metrics["mean_pruned_token_similarity_variance"],
                0.0025,
            )
            self.assertAlmostEqual(
                metrics["mean_kept_token_similarity_variance"],
                0.0025,
            )
            self.assertAlmostEqual(
                metrics["mean_pruned_token_similarity_entropy"]["shannon_bits"],
                1.0,
            )
            self.assertAlmostEqual(
                metrics["mean_kept_token_similarity_entropy"]["shannon_bits"],
                1.0,
            )
        self.assertEqual(summary["variance_basis"]["ddof"], 0)
        self.assertEqual(
            summary["variance_basis"]["score_groups"],
            ["all", "pruned", "kept"],
        )
        self.assertEqual(
            summary["selection_rule"]["ground_truth_candidates"][
                "maximum_per_query"
            ],
            0,
        )
        self.assertEqual(
            summary["selection_rule"]["top_ranked_incorrect_candidates"][
                "maximum_per_query"
            ],
            3,
        )

    def test_top1_outcome_summary_uses_requested_candidates_and_statistics(self):
        collector = TokenSimilarityAnalysisCollector(num_examples=5, seed=7)
        collector.add_query(
            self._result(
                ranked_indices=[2, 0, 1, 3, 4],
                ground_truth_page_ids=[3],
            ),
            self._candidate_stats(),
        )
        collector.add_query(
            self._result(
                # The highest GT is rank 4, so ranks 1-3 must all contribute as
                # incorrect candidates; candidate_pos=4 after the GT is excluded.
                ranked_indices=[0, 1, 3, 2, 4],
                ground_truth_page_ids=[3],
            ),
            self._candidate_stats(),
        )

        summary = collector._top1_outcome_similarity_summary(keep_ratio=0.5)
        self.assertEqual(
            summary["eligible_queries"],
            {"top1_correct": 1, "top1_incorrect": 1},
        )

        top1_correct = summary["groups"]["top1_correct"]
        self.assertEqual(top1_correct["correct_candidates"]["num_candidates"], 1)
        self.assertEqual(top1_correct["incorrect_candidates"]["num_candidates"], 1)
        self.assertAlmostEqual(
            top1_correct["correct_candidates"][
                "mean_pruning_threshold_similarity"
            ],
            0.12,
        )
        self.assertAlmostEqual(
            top1_correct["incorrect_candidates"][
                "mean_pruning_threshold_similarity"
            ],
            0.10,
        )

        top1_incorrect = summary["groups"]["top1_incorrect"]
        correct = top1_incorrect["correct_candidates"]
        incorrect = top1_incorrect["incorrect_candidates"]
        self.assertEqual(correct["num_candidates"], 1)
        self.assertEqual(incorrect["num_candidates"], 3)
        self.assertAlmostEqual(incorrect["mean_candidates_per_query"], 3.0)
        self.assertAlmostEqual(
            incorrect["mean_pruning_threshold_similarity"],
            (0.10 + 0.11 + 0.13) / 3,
        )
        self.assertAlmostEqual(incorrect["mean_all_token_similarity"], 0.0133333333)
        self.assertAlmostEqual(
            incorrect["mean_all_token_similarity_variance"],
            0.025,
        )
        self.assertAlmostEqual(
            incorrect["mean_all_token_similarity_entropy"]["shannon_bits"],
            2.0,
        )
        self.assertAlmostEqual(
            incorrect["mean_pruned_token_similarity"],
            -0.1366666667,
        )
        self.assertAlmostEqual(
            incorrect["mean_kept_token_similarity"],
            0.1633333333,
        )
        self.assertAlmostEqual(
            incorrect["mean_pruned_token_similarity_variance"],
            0.0025,
        )
        self.assertAlmostEqual(
            incorrect["mean_kept_token_similarity_variance"],
            0.0025,
        )
        self.assertAlmostEqual(
            incorrect["mean_pruned_token_similarity_entropy"]["shannon_bits"],
            1.0,
        )
        self.assertAlmostEqual(
            incorrect["mean_kept_token_similarity_entropy"]["shannon_bits"],
            1.0,
        )

        distribution = incorrect["token_similarity_distribution"]
        self.assertEqual(distribution["num_query_histograms"], 1)
        self.assertEqual(len(distribution["mean_density"]), 80)
        self.assertEqual(distribution["min_density"], distribution["mean_density"])
        self.assertEqual(distribution["max_density"], distribution["mean_density"])
        self.assertNotIn("ci95_lower", distribution)
        self.assertNotIn("ci95_upper", distribution)
        bin_width = summary["bin_edges"][1] - summary["bin_edges"][0]
        self.assertAlmostEqual(
            sum(distribution["mean_density"]) * bin_width,
            1.0,
            places=5,
        )

    def test_top1_outcome_distribution_reports_query_min_max_envelope(self):
        collector = TokenSimilarityAnalysisCollector(
            num_examples=5,
            seed=7,
            num_bins=8,
        )
        result = self._result(
            ranked_indices=[2, 0, 1, 3, 4],
            ground_truth_page_ids=[3],
        )
        first_stats = self._candidate_stats()
        second_stats = self._candidate_stats()
        for item in second_stats:
            item["scores"] = item["scores"] + 0.04
            item["threshold"] += 0.04

        first_histogram = collector._normalized_histogram(first_stats[2]["scores"])
        second_histogram = collector._normalized_histogram(second_stats[2]["scores"])
        collector.add_query(result, first_stats)
        collector.add_query(result, second_stats)

        summary = collector._top1_outcome_similarity_summary(keep_ratio=0.5)
        distribution = summary["groups"]["top1_correct"][
            "correct_candidates"
        ]["token_similarity_distribution"]
        self.assertEqual(distribution["num_query_histograms"], 2)
        np.testing.assert_allclose(
            distribution["mean_density"],
            (first_histogram + second_histogram) / 2,
        )
        np.testing.assert_allclose(
            distribution["min_density"],
            np.minimum(first_histogram, second_histogram),
        )
        np.testing.assert_allclose(
            distribution["max_density"],
            np.maximum(first_histogram, second_histogram),
        )

    def test_entropy_uses_full_query_min_max_for_all_candidate_groups(self):
        collector = TokenSimilarityAnalysisCollector(
            num_examples=5,
            seed=7,
            num_bins=2,
        )
        result = self._result(
            ranked_indices=[2, 0, 1, 3, 4],
            ground_truth_page_ids=[3],
        )
        candidate_stats = self._candidate_stats()
        for item in candidate_stats:
            item["scores"] = torch.tensor([0.0, 1.0, 2.0, 3.0])
            item["threshold"] = 2.0
        # candidate_pos=4 不会进入 Top1 outcome 或默认全局汇总的候选集合，
        # 但它仍属于这个 query 的一阶段 Top-K，因此必须把熵范围扩到 100。
        candidate_stats[4]["scores"] = torch.tensor([0.0, 1.0, 2.0, 100.0])
        collector.add_query(result, candidate_stats)

        top1_summary = collector._top1_outcome_similarity_summary(keep_ratio=0.5)
        top1_correct = top1_summary["groups"]["top1_correct"][
            "correct_candidates"
        ]
        self.assertEqual(top1_summary["entropy_basis"]["range_scope"], "per_query")
        self.assertAlmostEqual(
            top1_correct["mean_all_token_similarity_entropy"]["shannon_bits"],
            0.0,
        )
        self.assertAlmostEqual(
            top1_correct["mean_pruned_token_similarity_entropy"]["shannon_bits"],
            0.0,
        )
        self.assertAlmostEqual(
            top1_correct["mean_kept_token_similarity_entropy"]["shannon_bits"],
            0.0,
        )

        global_summary = collector._all_query_candidate_comparison_summary(
            keep_ratio=0.5
        )
        global_correct = global_summary["metrics"]["ground_truth_candidates"]
        self.assertAlmostEqual(
            global_correct["mean_all_token_similarity_entropy"]["shannon_bits"],
            0.0,
        )

        example = collector._histogram_export(keep_ratio=0.5)["groups"]["correct"][0]
        self.assertEqual(example["entropy_similarity_range"], [0.0, 100.0])

        degenerate_counts = collector._entropy_histogram_counts(
            torch.ones(4),
            (1.0, 1.0),
        )
        self.assertEqual(degenerate_counts.tolist(), [4, 0])
        self.assertAlmostEqual(
            collector._histogram_entropy(degenerate_counts)["shannon_bits"],
            0.0,
        )

    def test_save_writes_per_bin_counts_for_each_candidate_distribution(self):
        collector = TokenSimilarityAnalysisCollector(
            num_examples=5,
            seed=7,
            comparison_num_ground_truth_candidates=1,
            comparison_num_incorrect_candidates=1,
        )
        collector.add_query(
            self._result(
                ranked_indices=[2, 0, 1, 3, 4],
                ground_truth_page_ids=[3],
            ),
            self._candidate_stats(),
        )

        # 这个测试只验证 JSON 数据，不依赖服务器是否装有可用的绘图后端。
        with tempfile.TemporaryDirectory() as temporary_dir:
            expected_correct_figure = (
                Path(temporary_dir)
                / "recall1_correct_examples_keep50"
                / "example_01.png"
            )
            with patch("utils.similarity_analysis.ensure_matplotlib_available"), \
                    patch.object(collector, "_plot_overall"), \
                    patch.object(collector, "_plot_top1_outcome") as plot_outcome, \
                    patch.object(
                        collector,
                        "_plot_examples",
                        side_effect=[[expected_correct_figure], []],
                    ) as plot_examples:
                paths = collector.save(temporary_dir, keep_ratio=0.5)

            histogram_path = Path(paths["histogram_counts"])
            self.assertTrue(histogram_path.is_file())
            exported = json.loads(histogram_path.read_text(encoding="utf-8"))
            comparison_path = Path(paths["all_query_candidate_summary"])
            self.assertTrue(comparison_path.is_file())
            comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
            outcome_path = Path(paths["top1_outcome_summary"])
            self.assertTrue(outcome_path.is_file())
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            self.assertEqual(plot_examples.call_count, 2)
            plot_outcome.assert_called_once()
            self.assertEqual(
                Path(paths["correct_figure_directory"]),
                Path(temporary_dir) / "recall1_correct_examples_keep50",
            )
            self.assertEqual(
                exported["figure_files"]["correct"],
                ["recall1_correct_examples_keep50/example_01.png"],
            )
            self.assertEqual(
                exported["all_query_candidate_comparison"],
                comparison,
            )
            self.assertEqual(
                outcome["eligible_queries"]["top1_correct"],
                1,
            )

        self.assertEqual(exported["num_bins"], 80)
        self.assertEqual(len(exported["bins"]), 80)
        self.assertEqual(exported["display_range"], [0.0, 0.25])
        self.assertEqual(exported["bins"][0]["left_edge"], 0.0)
        self.assertEqual(exported["bins"][-1]["right_edge"], 0.25)
        candidate = exported["groups"]["correct"][0]["candidates"][0]
        distributions = candidate["distributions"]
        self.assertEqual(set(distributions), {"all", "pruned", "kept"})
        self.assertAlmostEqual(candidate["similarity"]["variance"], 0.025)

        all_counts = distributions["all"]["counts"]
        pruned_counts = distributions["pruned"]["counts"]
        kept_counts = distributions["kept"]["counts"]
        self.assertEqual(len(all_counts), 80)
        self.assertEqual(sum(all_counts), 4)
        self.assertEqual(sum(pruned_counts), 2)
        self.assertEqual(sum(kept_counts), 2)
        self.assertEqual(
            all_counts,
            [pruned + kept for pruned, kept in zip(pruned_counts, kept_counts)],
        )

        self.assertAlmostEqual(
            exported["groups"]["correct"][0]["entropy_similarity_range"][0],
            -0.2,
        )
        self.assertAlmostEqual(
            exported["groups"]["correct"][0]["entropy_similarity_range"][1],
            0.24,
        )

        # 熵使用 query 全部候选的真实 [-0.2, 0.24] 区间。当前候选的四个值
        # 分别落入不同 bin，因此 all 分布的 Shannon 熵为 2 bits。
        all_entropy = distributions["all"]["entropy"]
        self.assertAlmostEqual(all_entropy["shannon_bits"], 2.0)
        self.assertAlmostEqual(
            all_entropy["normalized"],
            2.0 / math.log2(80),
        )
        self.assertAlmostEqual(all_entropy["effective_bins"], 4.0)
        self.assertAlmostEqual(distributions["all"]["variance"], 0.025)
        self.assertAlmostEqual(distributions["pruned"]["variance"], 0.0025)
        self.assertAlmostEqual(distributions["kept"]["variance"], 0.0025)
        self.assertAlmostEqual(distributions["all"]["mean"], 0.02)
        self.assertAlmostEqual(distributions["pruned"]["mean"], -0.13)
        self.assertAlmostEqual(distributions["kept"]["mean"], 0.17)

        # 两个被剪掉的负相似度在 query 实际区间内落入不同 bin，因此熵为 1。
        pruned_entropy = distributions["pruned"]["entropy"]
        self.assertAlmostEqual(pruned_entropy["shannon_bits"], 1.0)
        self.assertAlmostEqual(
            pruned_entropy["normalized"],
            1.0 / math.log2(80),
        )
        self.assertAlmostEqual(pruned_entropy["effective_bins"], 2.0)

        self.assertEqual(comparison["total_queries_seen"], 1)
        self.assertEqual(comparison["eligible_paired_queries"], 1)
        incorrect_metrics = comparison["metrics"][
            "top_ranked_incorrect_candidates"
        ]
        correct_metrics = comparison["metrics"]["ground_truth_candidates"]
        self.assertAlmostEqual(
            incorrect_metrics["mean_pruning_threshold_similarity"],
            0.10,
        )
        self.assertAlmostEqual(
            correct_metrics["mean_pruning_threshold_similarity"],
            0.12,
        )
        self.assertAlmostEqual(
            incorrect_metrics["mean_all_token_similarity_entropy"]["shannon_bits"],
            2.0,
        )
        self.assertAlmostEqual(
            correct_metrics["mean_all_token_similarity_entropy"]["shannon_bits"],
            2.0,
        )
        self.assertAlmostEqual(
            incorrect_metrics["mean_all_token_similarity_variance"],
            0.025,
        )
        self.assertAlmostEqual(
            correct_metrics["mean_all_token_similarity_variance"],
            0.025,
        )
        for metrics in (incorrect_metrics, correct_metrics):
            self.assertAlmostEqual(
                metrics["mean_pruned_token_similarity_variance"],
                0.0025,
            )
            self.assertAlmostEqual(
                metrics["mean_kept_token_similarity_variance"],
                0.0025,
            )
            self.assertAlmostEqual(
                metrics["mean_pruned_token_similarity_entropy"]["shannon_bits"],
                1.0,
            )
            self.assertAlmostEqual(
                metrics["mean_kept_token_similarity_entropy"]["shannon_bits"],
                1.0,
            )

        top1_metrics = outcome["groups"]["top1_correct"]["correct_candidates"]
        self.assertAlmostEqual(
            top1_metrics["mean_pruned_token_similarity"],
            -0.13,
        )
        self.assertAlmostEqual(
            top1_metrics["mean_kept_token_similarity"],
            0.17,
        )

    def test_overall_incorrect_curves_use_distinct_overlap_safe_styles(self):
        collector = TokenSimilarityAnalysisCollector(num_examples=1, seed=7)
        one_density_curve = torch.ones(collector.num_bins).numpy()
        three_density_curve = 3 * one_density_curve
        collector.eligible_counts["incorrect"] = 2
        for role in ("top1_incorrect", "other_negatives", "ground_truth"):
            collector._group_histograms["incorrect"][role].extend(
                [one_density_curve, three_density_curve]
            )

        # 使用模拟坐标轴捕获 plot 参数，验证完全重合的数据仍会获得不同线型、
        # marker 以及错开的 marker 起始位置，不依赖测试机器是否安装 matplotlib。
        fake_pyplot = MagicMock()
        correct_ax = MagicMock()
        incorrect_ax = MagicMock()
        fake_pyplot.subplots.return_value = (
            MagicMock(),
            [correct_ax, incorrect_ax],
        )
        with patch(
            "utils.similarity_analysis.importlib.import_module",
            return_value=fake_pyplot,
        ):
            collector._plot_overall(Path("unused.png"), keep_ratio=0.5)

        plot_calls = incorrect_ax.plot.call_args_list
        self.assertEqual(
            [call.kwargs["linestyle"] for call in plot_calls],
            ["-", ":", "-."],
        )
        self.assertEqual(
            [call.kwargs["marker"] for call in plot_calls],
            ["o", "s", "^"],
        )
        self.assertEqual(
            [call.kwargs["markevery"] for call in plot_calls],
            [(0, 10), (3, 10), (6, 10)],
        )
        fill_calls = incorrect_ax.fill_between.call_args_list
        self.assertEqual(len(fill_calls), 3)
        for fill_call in fill_calls:
            np.testing.assert_array_equal(fill_call.args[1], one_density_curve)
            np.testing.assert_array_equal(fill_call.args[2], three_density_curve)

    def test_incorrect_query_without_gt_in_topk_is_skipped(self):
        # GT local_page_id=8 不在候选 [1, 2, 3, 4, 5] 中，无法绘制正确选项。
        collector = TokenSimilarityAnalysisCollector(num_examples=5, seed=7)
        result = self._result(
            ranked_indices=[0, 1, 2, 3, 4],
            ground_truth_page_ids=[8],
        )

        collector.add_query(result, self._candidate_stats())

        self.assertEqual(collector.eligible_counts["incorrect"], 0)
        self.assertEqual(
            collector.skipped_counts["incorrect_gt_missing_from_candidates"],
            1,
        )
        self.assertEqual(collector.comparison_eligible_queries, 0)
        self.assertEqual(
            collector.comparison_skipped_counts[
                "correct_candidate_missing_from_topk"
            ],
            1,
        )
        self.assertEqual(
            collector.outcome_group_query_counts,
            {"top1_correct": 0, "top1_incorrect": 0},
        )
        self.assertEqual(
            collector.outcome_group_skipped_counts[
                "correct_candidate_missing_from_topk"
            ],
            1,
        )


if __name__ == "__main__":
    unittest.main()
