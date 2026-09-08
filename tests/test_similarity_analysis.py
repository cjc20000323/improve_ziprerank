import unittest

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


if __name__ == "__main__":
    unittest.main()
