import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from unittest.mock import patch

import scripts.analyze_token_alignment_top5_pos as token_alignment_pos_cli
from utils.token_alignment_pos_analysis import (
    ContextualPosTagger,
    analyze_token_alignment_pos,
)


class _FakePosTagger(ContextualPosTagger):
    def __init__(self) -> None:
        self.num_calls = 0

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {"backend": "fake", "model_name_or_path": "unit-test"}

    def tag(self, text: str) -> Sequence[Mapping[str, Any]]:
        self.num_calls += 1
        if not text:
            return []
        return [
            {
                "text": "Find",
                "char_start": 0,
                "char_end": 4,
                "pos": "VERB",
                "fine_pos": "VB",
                "lemma": "find",
            },
            {
                "text": "red",
                "char_start": 5,
                "char_end": 8,
                "pos": "ADJ",
                "fine_pos": "JJ",
                "lemma": "red",
            },
            {
                "text": "invoices",
                "char_start": 9,
                "char_end": 17,
                "pos": "NOUN",
                "fine_pos": "NNS",
                "lemma": "invoice",
            },
            {
                "text": "quickly",
                "char_start": 18,
                "char_end": 25,
                "pos": "ADV",
                "fine_pos": "RB",
                "lemma": "quickly",
            },
            {
                "text": ".",
                "char_start": 25,
                "char_end": 26,
                "pos": "PUNCT",
                "fine_pos": ".",
                "lemma": ".",
            },
        ]


def _query_token(index, decoded_text):
    return {
        "query_sequence_index": index,
        "input_sequence_position": 100 + index,
        "token_id": 1000 + index,
        "vocab_token": decoded_text,
        "decoded_text": decoded_text,
        "is_special_token": False,
    }


def _summary(index, all_count, kept_count):
    return {
        "query_sequence_index": index,
        "matched_visual_token_count": all_count,
        "matched_kept_visual_token_count": kept_count,
        "matched_pruned_visual_token_count": all_count - kept_count,
    }


class TokenAlignmentPosAnalysisTest(unittest.TestCase):
    @staticmethod
    def _payload() -> Dict[str, Any]:
        query_tokens = [
            _query_token(0, "Find"),
            _query_token(1, " red"),
            _query_token(2, " invoice"),
            _query_token(3, "s"),
            _query_token(4, " quickly"),
            _query_token(5, "."),
        ]
        return {
            "schema_version": 4,
            "num_queries_exported": 1,
            "run": {"requested_queries": 1},
            "queries": [{
                "qid": "q0",
                "doc_name": "doc",
                "domain": "News",
                "q_idx": 0,
                "query": "Find red invoices quickly.",
                "qi_text_token_sequence": query_tokens,
                "candidates": [
                    {
                        "role": "highest_ranked_correct",
                        "candidate_letter": "C",
                        "final_rank": 2,
                        "local_page_id": 3,
                        "is_ground_truth": True,
                        "matched_query_token_summary": [
                            _summary(0, 4, 3),
                            _summary(1, 9, 5),
                            _summary(2, 7, 4),
                            _summary(3, 6, 4),
                            _summary(4, 3, 1),
                            _summary(5, 1, 0),
                        ],
                    },
                    {
                        "role": "highest_ranked_incorrect",
                        "candidate_letter": "B",
                        "final_rank": 1,
                        "local_page_id": 2,
                        "is_ground_truth": False,
                        "matched_query_token_summary": [
                            _summary(0, 8, 4),
                            _summary(1, 2, 1),
                            _summary(2, 6, 3),
                            _summary(3, 5, 2),
                            _summary(4, 7, 4),
                            _summary(5, 1, 1),
                        ],
                    },
                ],
            }],
        }

    def test_top_five_counts_and_pos_aggregates_are_role_specific(self):
        payload = self._payload()
        original_payload = copy.deepcopy(payload)

        aggregate_report, query_report = analyze_token_alignment_pos(
            payload,
            _FakePosTagger(),
            top_k=5,
            count_scope="all",
            expected_num_queries=1,
        )

        self.assertEqual(payload, original_payload)
        self.assertEqual(aggregate_report["schema_version"], 2)
        self.assertEqual(
            aggregate_report["report_type"],
            "aggregate_statistics",
        )
        self.assertNotIn("queries", aggregate_report)
        self.assertEqual(query_report["report_type"], "per_query_results")
        self.assertNotIn("aggregate", query_report)
        self.assertNotIn("aggregate_by_top1_outcome", query_report)
        self.assertEqual(query_report["num_queries"], 1)
        query_output = query_report["queries"][0]
        correct, incorrect = query_output["candidates"]
        self.assertEqual(
            [row["query_sequence_index"] for row in correct["top_query_tokens"]],
            [1, 2, 3, 0, 4],
        )
        self.assertEqual(
            [row["pos"] for row in correct["top_query_tokens"]],
            ["ADJ", "NOUN", "NOUN", "VERB", "ADV"],
        )
        self.assertEqual(
            [row["query_sequence_index"] for row in incorrect["top_query_tokens"]],
            [0, 4, 2, 3, 1],
        )

        correct_stats = aggregate_report["aggregate"]["by_candidate_role"][
            "highest_ranked_correct"
        ]
        correct_pos_counts = {
            row["pos"]: row["total_top_k_occurrences"]
            for row in correct_stats["pos_statistics"]
        }
        self.assertEqual(correct_pos_counts["NOUN"], 2)
        self.assertEqual(correct_pos_counts["ADJ"], 1)
        self.assertEqual(correct_stats["observed_top_k_slots"], 5)

        combined_stats = aggregate_report["aggregate"][
            "correct_and_incorrect_combined"
        ]
        combined_pos_counts = {
            row["pos"]: row["total_top_k_occurrences"]
            for row in combined_stats["pos_statistics"]
        }
        self.assertEqual(combined_pos_counts["NOUN"], 4)
        self.assertEqual(combined_stats["num_candidates"], 2)
        noun_row = next(
            row for row in combined_stats["pos_statistics"] if row["pos"] == "NOUN"
        )
        self.assertEqual(noun_row["mean_occurrences_per_query"], 4.0)
        self.assertEqual(noun_row["mean_occurrences_per_candidate"], 2.0)

    def test_kept_scope_ranks_by_kept_alignment_count(self):
        aggregate_report, query_report = analyze_token_alignment_pos(
            self._payload(),
            _FakePosTagger(),
            top_k=3,
            count_scope="kept",
        )

        correct = query_report["queries"][0]["candidates"][0]
        self.assertEqual(
            [row["query_sequence_index"] for row in correct["top_query_tokens"]],
            [1, 2, 3],
        )
        self.assertEqual(
            aggregate_report["configuration"]["ranking_count_field"],
            "matched_kept_visual_token_count",
        )

    def test_expected_query_count_prevents_partial_dataset_report(self):
        with self.assertRaises(ValueError):
            analyze_token_alignment_pos(
                self._payload(),
                _FakePosTagger(),
                expected_num_queries=2,
            )

    def test_additive_outcome_groups_use_all_errors_before_correct(self):
        payload = self._payload()
        top1_incorrect_query = payload["queries"][0]
        top1_incorrect_query["recall_at_1_correct"] = False
        top1_incorrect_query["candidates"][0]["final_rank"] = 3
        top1_incorrect_query[
            "additional_incorrect_candidates_before_correct"
        ] = [{
            "role": "incorrect_ranked_before_correct",
            "candidate_letter": "A",
            "final_rank": 2,
            "local_page_id": 1,
            "is_ground_truth": False,
            "matched_query_token_summary": [
                _summary(0, 1, 1),
                _summary(1, 8, 4),
                _summary(2, 7, 4),
                _summary(3, 6, 3),
                _summary(4, 5, 2),
                _summary(5, 4, 2),
            ],
        }]

        top1_correct_query = copy.deepcopy(top1_incorrect_query)
        top1_correct_query["qid"] = "q1"
        top1_correct_query["q_idx"] = 1
        top1_correct_query["recall_at_1_correct"] = True
        top1_correct_query["candidates"][0]["final_rank"] = 1
        top1_correct_query["candidates"][1]["final_rank"] = 2
        top1_correct_query[
            "additional_incorrect_candidates_before_correct"
        ] = []
        payload["queries"].append(top1_correct_query)
        payload["num_queries_exported"] = 2

        aggregate_report, query_report = analyze_token_alignment_pos(
            payload,
            _FakePosTagger(),
            top_k=5,
            count_scope="all",
        )

        # The original aggregate remains one correct and one highest incorrect
        # candidate per query; the additional rank-2 error only enters the new group.
        self.assertEqual(
            aggregate_report["aggregate"]["correct_and_incorrect_combined"][
                "num_candidates"
            ],
            4,
        )
        top1_correct = aggregate_report["aggregate_by_top1_outcome"][
            "top1_correct"
        ]
        self.assertEqual(top1_correct["num_queries"], 1)
        self.assertEqual(top1_correct["correct_candidates"]["num_candidates"], 1)
        self.assertEqual(top1_correct["incorrect_candidates"]["num_candidates"], 1)

        top1_incorrect = aggregate_report["aggregate_by_top1_outcome"][
            "top1_incorrect"
        ]
        self.assertEqual(top1_incorrect["num_queries"], 1)
        self.assertEqual(top1_incorrect["correct_candidates"]["num_candidates"], 1)
        self.assertEqual(top1_incorrect["incorrect_candidates"]["num_candidates"], 2)
        self.assertEqual(
            top1_incorrect["incorrect_candidates"]["observed_top_k_slots"],
            10,
        )
        self.assertEqual(
            top1_incorrect["incorrect_candidate_count"]["mean_per_query"],
            2.0,
        )
        self.assertEqual(
            [
                candidate["final_rank"]
                for candidate in query_report["queries"][0][
                    "outcome_group_candidate_selection"
                ]["incorrect_candidates"]
            ],
            [1, 2],
        )

    def test_rejects_aggregate_report_as_alignment_input(self):
        with self.assertRaisesRegex(ValueError, "per-query alignment JSON"):
            analyze_token_alignment_pos(
                {
                    "schema_version": 2,
                    "report_type": "aggregate_statistics",
                    "aggregate": {},
                },
                _FakePosTagger(),
            )

    def test_cli_writes_aggregate_and_query_results_to_separate_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "alignment.json"
            aggregate_path = Path(temp_dir) / "top5_pos.json"
            query_path = Path(temp_dir) / "top5_pos_queries.json"
            input_path.write_text(
                json.dumps(self._payload()),
                encoding="utf-8",
            )
            argv = [
                "analyze_token_alignment_top5_pos.py",
                "--input_file",
                str(input_path),
                "--output_file",
                str(aggregate_path),
            ]

            with patch.object(sys, "argv", argv):
                with patch(
                    "scripts.analyze_token_alignment_top5_pos.SpacyPosTagger",
                    lambda _model_name: _FakePosTagger(),
                ), redirect_stdout(io.StringIO()):
                    token_alignment_pos_cli.main()

            aggregate_payload = json.loads(
                aggregate_path.read_text(encoding="utf-8")
            )
            query_payload = json.loads(query_path.read_text(encoding="utf-8"))
            self.assertNotIn("queries", aggregate_payload)
            self.assertIn("aggregate", aggregate_payload)
            self.assertNotIn("aggregate", query_payload)
            self.assertEqual(len(query_payload["queries"]), 1)

    def test_outcome_group_rejects_missing_errors_before_correct(self):
        payload = self._payload()
        payload["queries"][0]["recall_at_1_correct"] = False
        payload["queries"][0]["candidates"][0]["final_rank"] = 3

        with self.assertRaisesRegex(
            ValueError,
            "--collect_incorrect_before_correct",
        ):
            analyze_token_alignment_pos(
                payload,
                _FakePosTagger(),
            )


if __name__ == "__main__":
    unittest.main()
