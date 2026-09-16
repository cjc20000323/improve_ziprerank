import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from models.qwen3vl_with_qi_early_token_alignment import (
    QIEarlyTokenAlignmentPruner,
)
from utils.token_alignment_analysis import (
    TokenAlignmentCollector,
    find_dataset_query_input_positions,
    select_all_eligible_queries,
    select_base_diagnostic_records,
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


class _CharacterOffsetTokenizer:
    def __call__(self, text, **_kwargs):
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [
                (index, index + 1) for index in range(len(text))
            ],
        }


def _image_bytes(width=90, height=30):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(120, 120, 120)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


class TokenAlignmentCollectorTest(unittest.TestCase):
    @staticmethod
    def _candidate_stats(num_candidates=3):
        # 三个候选共享同一查询坐标系，但分数和阈值略有差异；这样既覆盖共享
        # 元数据校验，也能确认 collector 没有把不同候选的视觉统计混在一起。
        shared_ids = torch.tensor([10, 11, 99])
        shared_positions = torch.tensor([4, 5, 6])
        records = []
        for candidate_pos in range(num_candidates):
            records.append({
                "candidate_pos": candidate_pos,
                "scores": torch.tensor([0.1, 0.2, 0.3]) + candidate_pos * 0.01,
                "query_max_similarities": (
                    torch.tensor([0.7, 0.8, 0.9]) + candidate_pos * 0.01
                ),
                "kept_mask": torch.tensor([False, True, True]),
                "max_text_token_indices": torch.tensor([1, 2, 1]),
                "threshold": 0.2 + candidate_pos * 0.01,
                "num_original_tokens": 3,
                "num_kept_tokens": 2,
                "image_grid_thw": [1, 1, 3],
                "query_token_ids": shared_ids,
                "query_input_positions": shared_positions,
                "text_extraction_mode": "dataset_query_only",
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
        collector = TokenAlignmentCollector(
            tokenizer=_FakeTokenizer(),
            image_binary_loader=lambda _global_idx: _image_bytes(),
            spatial_merge_size=1,
        )
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
        self.assertAlmostEqual(first_visual["max_similarity"], 0.72, places=6)
        self.assertAlmostEqual(first_visual["pruning_similarity"], 0.12, places=6)
        self.assertEqual(first_visual["patch_grid_position_trc"], [0, 0, 0])

        # Query token index 1 attracts visual tokens 0 and 2, one kept and one pruned.
        most_common = correct["matched_query_token_summary"][0]
        self.assertEqual(most_common["query_sequence_index"], 1)
        self.assertEqual(most_common["matched_visual_token_count"], 2)
        self.assertEqual(most_common["matched_kept_visual_token_count"], 1)
        self.assertEqual(most_common["matched_pruned_visual_token_count"], 1)

    def test_optional_collection_keeps_base_pair_and_adds_all_errors_before_gt(self):
        collector = TokenAlignmentCollector(
            tokenizer=_FakeTokenizer(),
            image_binary_loader=lambda _global_idx: _image_bytes(),
            spatial_merge_size=1,
            collect_incorrect_before_correct=True,
        )
        result = {
            **self._result(),
            "page_id": [4],
            "top_k_global_indices": [101, 102, 103, 104],
            # Wrong candidates occupy final ranks 1 and 2; the best GT is rank 3.
            "ranked_indices": [1, 0, 3, 2],
        }
        collector.add_query(result, self._candidate_stats(num_candidates=4))

        query_record = collector.query_records[0]
        self.assertEqual(len(query_record["candidates"]), 2)
        correct, highest_incorrect = query_record["candidates"]
        self.assertEqual(correct["final_rank"], 3)
        self.assertEqual(highest_incorrect["final_rank"], 1)

        additional = query_record[
            "additional_incorrect_candidates_before_correct"
        ]
        self.assertTrue(
            query_record["incorrect_candidates_before_correct_complete"]
        )
        self.assertEqual(len(additional), 1)
        self.assertEqual(additional[0]["role"], "incorrect_ranked_before_correct")
        self.assertEqual(additional[0]["candidate_pos"], 0)
        self.assertEqual(additional[0]["final_rank"], 2)

    def test_default_collection_does_not_add_outcome_extension_fields(self):
        collector = TokenAlignmentCollector(
            tokenizer=_FakeTokenizer(),
            image_binary_loader=lambda _global_idx: _image_bytes(),
            spatial_merge_size=1,
        )
        collector.add_query(self._result(), self._candidate_stats())

        query_record = collector.query_records[0]
        self.assertNotIn(
            "additional_incorrect_candidates_before_correct",
            query_record,
        )
        self.assertNotIn(
            "incorrect_candidates_before_correct_complete",
            query_record,
        )

    def test_base_record_selection_restores_sample_order_and_strips_extensions(self):
        all_records = [
            {
                "doc_name": "doc",
                "q_idx": 0,
                "marker": "dataset_first",
                "additional_incorrect_candidates_before_correct": [{"rank": 2}],
                "incorrect_candidates_before_correct_complete": True,
            },
            {
                "doc_name": "doc",
                "q_idx": 1,
                "marker": "dataset_second",
                "additional_incorrect_candidates_before_correct": [],
                "incorrect_candidates_before_correct_complete": True,
            },
        ]
        sampled_results = {
            "second": {"doc_name": "doc", "q_idx": 1},
            "first": {"doc_name": "doc", "q_idx": 0},
        }

        sampled_records = select_base_diagnostic_records(
            all_records,
            sampled_results,
        )

        self.assertEqual(
            [record["marker"] for record in sampled_records],
            ["dataset_second", "dataset_first"],
        )
        for record in sampled_records:
            self.assertNotIn(
                "additional_incorrect_candidates_before_correct",
                record,
            )
            self.assertNotIn(
                "incorrect_candidates_before_correct_complete",
                record,
            )
        self.assertIn(
            "additional_incorrect_candidates_before_correct",
            all_records[0],
        )

    def test_json_only_save_does_not_create_image_directory(self):
        collector = TokenAlignmentCollector(
            tokenizer=_FakeTokenizer(),
            image_binary_loader=lambda _global_idx: _image_bytes(),
            spatial_merge_size=1,
        )
        collector.add_query(self._result(), self._candidate_stats())

        with tempfile.TemporaryDirectory() as temp_dir:
            output = collector.save(
                str(Path(temp_dir) / "all_eligible.json"),
                run_metadata={"scope": "all_eligible"},
                write_images=False,
                total_queries_seen=7,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["num_queries_exported"], 1)
            self.assertEqual(payload["total_queries_seen"], 7)
            self.assertFalse(
                (Path(temp_dir) / "all_eligible_images").exists()
            )
            for candidate in payload["queries"][0]["candidates"]:
                self.assertNotIn("image_files", candidate)

    def test_save_writes_readable_json(self):
        collector = TokenAlignmentCollector(
            tokenizer=_FakeTokenizer(),
            image_binary_loader=lambda _global_idx: _image_bytes(),
            spatial_merge_size=1,
        )
        collector.add_query(self._result(), self._candidate_stats())

        # 使用真实临时目录完成一次写入和读回，覆盖 Path 创建、UTF-8 JSON
        # 序列化以及顶层 queries 结构，而不在仓库中留下测试产物。
        with tempfile.TemporaryDirectory() as temp_dir:
            stale_path = (
                Path(temp_dir)
                / "alignment_images"
                / "query_99_stale"
                / "stale.txt"
            )
            stale_path.parent.mkdir(parents=True)
            stale_path.write_text("stale", encoding="utf-8")
            output = collector.save(
                str(Path(temp_dir) / "alignment.json"),
                run_metadata={"test": True},
                reset_image_output_directory=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 4)
            self.assertEqual(payload["num_queries_exported"], 1)
            self.assertEqual(payload["queries"][0]["qid"], "doc_0")
            self.assertFalse(stale_path.exists())

            # Both selected candidates receive an untouched source image and an
            # overlay. A pixel in retained grid cell 1 must change while image size
            # remains identical to the source parquet bytes.
            for candidate in payload["queries"][0]["candidates"]:
                original_path = output.parent / candidate["image_files"]["original"]
                overlay_path = (
                    output.parent
                    / candidate["image_files"]["kept_patch_overlay"]
                )
                token_map_path = (
                    output.parent
                    / candidate["image_files"]["kept_patch_token_map"]
                )
                token_details_path = (
                    output.parent
                    / candidate["image_files"]["kept_patch_token_details"]
                )
                self.assertTrue(original_path.is_file())
                self.assertTrue(overlay_path.is_file())
                self.assertTrue(token_map_path.is_file())
                self.assertTrue(token_details_path.is_file())
                with Image.open(original_path) as original_image:
                    with Image.open(overlay_path) as overlay_image:
                        self.assertEqual(overlay_image.size, original_image.size)
                        self.assertNotEqual(
                            overlay_image.getpixel((45, 15)),
                            original_image.getpixel((45, 15)),
                        )
                    with Image.open(token_map_path) as token_map_image:
                        self.assertGreater(token_map_image.width, original_image.width)
                        self.assertNotEqual(
                            token_map_image.getpixel((45, 5)),
                            token_map_image.getpixel((75, 5)),
                        )
                    with Image.open(token_details_path) as token_details_image:
                        self.assertGreater(
                            token_details_image.width,
                            original_image.width,
                        )

    def test_finds_only_dataset_query_tokens_inside_prepared_input(self):
        query = "invoice total in 2024?"
        prompt = (
            "You are RankGPT.\n\n"
            "I will provide passages as images.\n\n"
            f"Search Query: {query}\n\n"
            "Rank the passages above based on their relevance to the search query.\n"
            "Only output the ranking results."
        )
        tokenizer = _CharacterOffsetTokenizer()
        prompt_ids = tokenizer(prompt)["input_ids"]
        full_input_ids = [200_000, 200_001] + prompt_ids + [200_002]

        # Character-level tokenization makes the expected boundary observable:
        # every returned ID must reconstruct the raw dataset field exactly, with
        # neither the field label nor the following fixed instruction included.
        positions = find_dataset_query_input_positions(
            prompt=prompt,
            input_ids=full_input_ids,
            tokenizer=tokenizer,
        )
        selected_text = "".join(chr(full_input_ids[index]) for index in positions)
        self.assertEqual(selected_text, query)
        self.assertNotIn("Search Query", selected_text)
        self.assertNotIn("Rank the passages", selected_text)

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

    def test_full_query_selection_scans_every_record_without_early_stop(self):
        eligible_a = self._result()
        eligible_b = {**self._result(), "qid": "doc_1", "q_idx": 1}
        missing_gt = {
            **self._result(),
            "qid": "doc_2",
            "q_idx": 2,
            "page_id": [8],
        }

        selected, metadata = select_all_eligible_queries({
            "a": eligible_a,
            "missing": missing_gt,
            "b": eligible_b,
        })

        self.assertEqual(list(selected), ["a", "b"])
        self.assertEqual(metadata["selected_queries"], 2)
        self.assertEqual(metadata["queries_inspected_before_early_stop"], 3)
        self.assertEqual(metadata["queries_not_inspected"], 0)
        self.assertEqual(
            metadata["prefilter_skipped_queries"][
                "correct_candidate_missing_from_topk"
            ],
            1,
        )

    def test_diagnostic_pruner_records_text_argmax_without_changing_scores(self):
        pruner = QIEarlyTokenAlignmentPruner(keep_ratio=1.0)
        pruner.set_collect_similarity_stats(True)

        # 两组接近坐标轴的向量给出明确的 argmax [0, 1]，同时保留可手算的
        # 归一化余弦值，用来确认诊断分支只增加索引记录、没有改变基础分数。
        text_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        image_embeds = [torch.tensor([[0.9, 0.1], [0.1, 0.9]])]
        pruner.set_dataset_query_alignment_metadata(
            token_ids=torch.tensor([10]),
            input_positions=torch.tensor([4]),
            query_embeds=text_embeds[:1],
        )

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

        dataset_indices = torch.as_tensor(
            stats["dataset_query_max_text_token_indices"]
        )
        dataset_scores = torch.as_tensor(stats["dataset_query_max_similarities"])
        self.assertEqual(dataset_indices.tolist(), [0, 0])
        expected_dataset_scores = torch.tensor([0.9938837, 0.1104315])
        self.assertTrue(
            torch.allclose(dataset_scores, expected_dataset_scores, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
