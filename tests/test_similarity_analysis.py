import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        collector = TokenSimilarityAnalysisCollector(num_examples=5, seed=7)
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
        incorrect = summary["metrics"]["highest_ranked_incorrect"]
        correct = summary["metrics"]["highest_ranked_correct"]
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

    def test_save_writes_per_bin_counts_for_each_candidate_distribution(self):
        collector = TokenSimilarityAnalysisCollector(num_examples=5, seed=7)
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
            self.assertEqual(plot_examples.call_count, 2)
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

        self.assertEqual(exported["num_bins"], 80)
        self.assertEqual(len(exported["bins"]), 80)
        self.assertEqual(exported["display_range"], [0.0, 0.25])
        self.assertEqual(exported["bins"][0]["left_edge"], 0.0)
        self.assertEqual(exported["bins"][-1]["right_edge"], 0.25)
        candidate = exported["groups"]["correct"][0]["candidates"][0]
        distributions = candidate["distributions"]
        self.assertEqual(set(distributions), {"all", "pruned", "kept"})

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

        # 第一个被选候选的四个分数落入三个 bin，概率为 [0.5, 0.25, 0.25]，
        # 因此 all 分布的 Shannon 熵应为 1.5 bits。
        all_entropy = distributions["all"]["entropy"]
        self.assertAlmostEqual(all_entropy["shannon_bits"], 1.5)
        self.assertAlmostEqual(
            all_entropy["normalized"],
            1.5 / math.log2(80),
        )
        self.assertAlmostEqual(all_entropy["effective_bins"], 2.0 ** 1.5)

        # 两个被剪掉的负相似度都会夹到第一个 bin，因此熵为 0。
        pruned_entropy = distributions["pruned"]["entropy"]
        self.assertAlmostEqual(pruned_entropy["shannon_bits"], 0.0)
        self.assertAlmostEqual(pruned_entropy["normalized"], 0.0)
        self.assertAlmostEqual(pruned_entropy["effective_bins"], 1.0)

        self.assertEqual(comparison["total_queries_seen"], 1)
        self.assertEqual(comparison["eligible_paired_queries"], 1)
        incorrect_metrics = comparison["metrics"]["highest_ranked_incorrect"]
        correct_metrics = comparison["metrics"]["highest_ranked_correct"]
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
            1.5,
        )
        self.assertAlmostEqual(
            correct_metrics["mean_all_token_similarity_entropy"]["shannon_bits"],
            1.5,
        )

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


if __name__ == "__main__":
    unittest.main()
