import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.qwen3vl_with_qi_early_token_alignment import (
    QIEarlyTokenAlignmentPruner,
)
from utils.token_alignment_analysis import (
    TokenAlignmentCollector,
    select_eligible_query_subset,
)


class _FakeTokenizer:
    all_special_ids = [99]

    @staticmethod
    def convert_ids_to_tokens(token_id):
        return f"vocab_{token_id}"

    @staticmethod
    def decode(token_ids, **_kwargs):
        return f"decoded_{token_ids[0]}"


class TokenAlignmentCollectorTest(unittest.TestCase):
    @staticmethod
    def _candidate_stats():
        # 三个候选共享同一查询坐标系，但分数和阈值略有差异；这样既覆盖共享
        # 元数据校验，也能确认 collector 没有把不同候选的视觉统计混在一起。
        shared_ids = torch.tensor([10, 11, 99])
        shared_positions = torch.tensor([4, 5, 6])
        records = []
        for candidate_pos in range(3):
            records.append({
                "candidate_pos": candidate_pos,
                "scores": torch.tensor([0.1, 0.2, 0.3]) + candidate_pos * 0.01,
                "kept_mask": torch.tensor([False, True, True]),
                "max_text_token_indices": torch.tensor([1, 2, 1]),
                "threshold": 0.2 + candidate_pos * 0.01,
                "num_original_tokens": 3,
                "num_kept_tokens": 2,
                "query_token_ids": shared_ids,
                "query_input_positions": shared_positions,
                "text_extraction_mode": "all_before_image",
            })
        return records

    @staticmethod
    def _result():
        return {
            "qid": "doc_0",
            "doc_name": "doc",
            "domain": "News",
            "q_idx": 0,
            "query": "test query",
            "page_id": [3],
            "start_idx": 100,
            "end_idx": 109,
            # local page IDs: [1, 2, 3]; candidate_pos=2 is the GT.
            "top_k_global_indices": [101, 102, 103],
            # The best wrong candidate is pos=1; the best GT is pos=2.
            "ranked_indices": [1, 2, 0],
        }

    def test_exports_best_gt_best_wrong_argmax_and_keep_mask(self):
        collector = TokenAlignmentCollector(_FakeTokenizer())
        collector.add_query(self._result(), self._candidate_stats())

        # 先验证最终排序选择出的正负候选，再验证单个视觉 token 的文本回查；
        # 最后的聚合断言确保 kept/pruned 两条计数链路使用了同一份映射。
        self.assertEqual(len(collector.query_records), 1)
        query_record = collector.query_records[0]
        self.assertEqual(query_record["num_qi_text_tokens"], 3)
        self.assertTrue(query_record["qi_text_token_sequence"][2]["is_special_token"])

        correct, incorrect = query_record["candidates"]
        self.assertEqual(correct["role"], "highest_ranked_correct")
        self.assertEqual(correct["candidate_pos"], 2)
        self.assertEqual(correct["final_rank"], 2)
        self.assertEqual(incorrect["role"], "highest_ranked_incorrect")
        self.assertEqual(incorrect["candidate_pos"], 1)
        self.assertEqual(incorrect["final_rank"], 1)

        self.assertEqual(correct["kept_visual_token_indices"], [1, 2])
        self.assertEqual(correct["pruned_visual_token_indices"], [0])
        first_visual = correct["visual_tokens"][0]
        self.assertFalse(first_visual["kept"])
        self.assertEqual(first_visual["matched_query_sequence_index"], 1)
        self.assertEqual(first_visual["matched_input_sequence_position"], 5)
        self.assertEqual(first_visual["matched_token_id"], 11)
        self.assertEqual(first_visual["matched_vocab_token"], "vocab_11")
        self.assertEqual(first_visual["matched_decoded_text"], "decoded_11")

        # Query token index 1 attracts visual tokens 0 and 2, one kept and one pruned.
        most_common = correct["matched_query_token_summary"][0]
        self.assertEqual(most_common["query_sequence_index"], 1)
        self.assertEqual(most_common["matched_visual_token_count"], 2)
        self.assertEqual(most_common["matched_kept_visual_token_count"], 1)
        self.assertEqual(most_common["matched_pruned_visual_token_count"], 1)

    def test_save_writes_readable_json(self):
        collector = TokenAlignmentCollector(_FakeTokenizer())
        collector.add_query(self._result(), self._candidate_stats())

        # 使用真实临时目录完成一次写入和读回，覆盖 Path 创建、UTF-8 JSON
        # 序列化以及顶层 queries 结构，而不在仓库中留下测试产物。
        with tempfile.TemporaryDirectory() as temp_dir:
            output = collector.save(
                str(Path(temp_dir) / "alignment.json"),
                run_metadata={"test": True},
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_queries_exported"], 1)
            self.assertEqual(payload["queries"][0]["qid"], "doc_0")

    def test_query_prefilter_requires_both_gt_and_non_gt(self):
        # c 的 GT 不在 Top-K 内，不能组成诊断所需的正负候选对；固定 seed 下
        # 仅 a、b 应进入样本，同时跳过原因需要准确写入抽样元数据。
        eligible_a = self._result()
        eligible_b = {**self._result(), "qid": "doc_1", "q_idx": 1}
        missing_gt = {
            **self._result(),
            "qid": "doc_2",
            "q_idx": 2,
            "page_id": [8],
        }
        selected, metadata = select_eligible_query_subset(
            {"a": eligible_a, "b": eligible_b, "c": missing_gt},
            num_queries=2,
            seed=7,
        )

        self.assertEqual(set(selected), {"a", "b"})
        self.assertEqual(metadata["eligible_queries_encountered"], 2)
        inspected = metadata["queries_inspected_before_early_stop"]
        self.assertEqual(metadata["queries_not_inspected"], 3 - inspected)
        self.assertEqual(
            metadata["prefilter_skipped_queries"].get(
                "correct_candidate_missing_from_topk", 0
            ),
            inspected - 2,
        )

    def test_query_sampling_stops_after_twenty_eligible_queries(self):
        first_stage = {
            f"key_{index}": {
                **self._result(),
                "qid": f"doc_{index}",
                "q_idx": index,
            }
            for index in range(30)
        }

        # Run twice with the same local seed to verify deterministic ordering as
        # well as the early-stop boundary; ten remaining records are not inspected.
        selected_a, metadata_a = select_eligible_query_subset(
            first_stage, num_queries=20, seed=7
        )
        selected_b, metadata_b = select_eligible_query_subset(
            first_stage, num_queries=20, seed=7
        )

        self.assertEqual(list(selected_a), list(selected_b))
        self.assertEqual(len(selected_a), 20)
        self.assertEqual(metadata_a["queries_inspected_before_early_stop"], 20)
        self.assertEqual(metadata_a["queries_not_inspected"], 10)
        self.assertEqual(metadata_a["required_correct_candidates"], 1)
        self.assertEqual(metadata_a["required_incorrect_candidates"], 1)
        self.assertEqual(metadata_a, metadata_b)

    def test_diagnostic_pruner_records_text_argmax_without_changing_scores(self):
        pruner = QIEarlyTokenAlignmentPruner(keep_ratio=1.0)
        pruner.set_collect_similarity_stats(True)

        # 两组接近坐标轴的向量给出明确的 argmax [0, 1]，同时保留可手算的
        # 归一化余弦值，用来确认诊断分支只增加索引记录、没有改变基础分数。
        text_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        image_embeds = [torch.tensor([[0.9, 0.1], [0.1, 0.9]])]

        pruner(text_embeds=text_embeds, image_embeds_list=image_embeds)

        stats_list = pruner.last_similarity_stats
        self.assertIsNotNone(stats_list)
        if stats_list is None:
            self.fail("Diagnostic pruner did not record similarity statistics")
        stats = stats_list[0]
        max_text_indices = torch.as_tensor(stats["max_text_token_indices"])
        scores = torch.as_tensor(stats["scores"])
        self.assertEqual(max_text_indices.tolist(), [0, 1])
        expected_scores = torch.tensor([0.9938837, 0.9938837])
        self.assertTrue(torch.allclose(scores, expected_scores, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
