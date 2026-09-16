"""Aggregate POS statistics for the most frequently aligned query tokens.

This module is deliberately a post-processing layer over the existing token
alignment JSON. It never mutates that JSON and does not require another model
forward. Query-token counts come from each selected candidate's
``matched_query_token_summary``; contextual part-of-speech labels come from a
separately supplied tagger and are mapped back to the tokenizer subword spans.
"""

from __future__ import annotations

import importlib
import re
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple


CORRECT_ROLE = "highest_ranked_correct"
INCORRECT_ROLE = "highest_ranked_incorrect"
ANALYZED_ROLES = (CORRECT_ROLE, INCORRECT_ROLE)
TOP1_CORRECT_GROUP = "top1_correct"
TOP1_INCORRECT_GROUP = "top1_incorrect"
OUTCOME_GROUPS = (TOP1_CORRECT_GROUP, TOP1_INCORRECT_GROUP)
OUTCOME_CANDIDATE_CLASSES = ("correct_candidates", "incorrect_candidates")
COUNT_FIELDS = {
    "all": "matched_visual_token_count",
    "kept": "matched_kept_visual_token_count",
    "pruned": "matched_pruned_visual_token_count",
}


class ContextualPosTagger(Protocol):
    """Interface used by the production spaCy tagger and lightweight tests."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Describe the tagging implementation written into the output file."""

    def tag(self, text: str) -> Sequence[Mapping[str, Any]]:
        """Return word spans with universal and fine-grained POS labels."""


class SpacyPosTagger:
    """Use a spaCy pipeline to obtain contextual universal POS tags."""

    def __init__(self, model_name_or_path: str = "en_core_web_sm") -> None:
        try:
            spacy = importlib.import_module("spacy")
        except ImportError as exc:
            raise RuntimeError(
                "POS analysis requires spaCy. Install it in the evaluation "
                "environment, then install a language model such as "
                "en_core_web_sm."
            ) from exc

        try:
            self._nlp = spacy.load(
                model_name_or_path,
                disable=["ner", "parser"],
            )
        except OSError as exc:
            raise RuntimeError(
                f"Could not load spaCy POS model {model_name_or_path!r}. "
                "Install that model or pass --spacy_model with a local model path."
            ) from exc
        self.model_name_or_path = model_name_or_path

    @property
    def metadata(self) -> Mapping[str, Any]:
        return {
            "backend": "spacy",
            "model_name_or_path": self.model_name_or_path,
            "pipeline_language": str(self._nlp.meta.get("lang", "unknown")),
            "coarse_pos_standard": "spaCy Token.pos_ (Universal POS when supplied by the model)",
            "fine_pos_field": "spaCy Token.tag_",
        }

    def tag(self, text: str) -> Sequence[Mapping[str, Any]]:
        doc = self._nlp(text)
        tagged_spans = []
        for token in doc:
            if token.is_space:
                continue
            coarse_pos = str(token.pos_ or "").strip()
            if not coarse_pos:
                raise RuntimeError(
                    f"spaCy model {self.model_name_or_path!r} produced no POS "
                    "labels. Use a trained pipeline rather than spacy.blank()."
                )
            tagged_spans.append({
                "text": token.text,
                "char_start": int(token.idx),
                "char_end": int(token.idx + len(token.text)),
                "pos": coarse_pos,
                "fine_pos": str(token.tag_ or ""),
                "lemma": str(token.lemma_ or token.text),
            })
        return tagged_spans


def _token_piece(token_record: Mapping[str, Any]) -> str:
    """Choose the most readable tokenizer piece for character-span recovery."""
    decoded_text = str(token_record.get("decoded_text") or "")
    if decoded_text and "\ufffd" not in decoded_text:
        return decoded_text

    # Common SentencePiece/GPT-style word-boundary markers are converted only
    # as a recovery path when decoding one token yielded replacement characters.
    vocab_token = str(token_record.get("vocab_token") or "")
    return (
        vocab_token
        .replace("\u2581", " ")
        .replace("\u0120", " ")
        .replace("\u010a", "\n")
    )


def _whitespace_flexible_pattern(text: str) -> str:
    """Escape literal text while allowing equivalent whitespace runs."""
    parts = re.split(r"(\s+)", text)
    return "".join(r"\s+" if part.isspace() else re.escape(part) for part in parts)


def _locate_query_token_spans(
    query: str,
    query_tokens: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Recover each model subword's approximate character span in the query."""
    cursor = 0
    located_tokens = []
    for token_record in query_tokens:
        piece = _token_piece(token_record)
        visible_piece = piece.strip()
        located = dict(token_record)
        located["token_piece_for_pos_mapping"] = piece

        if not visible_piece:
            located.update({
                "query_char_span": None,
                "query_surface_text": "",
                "pos_mapping_status": "empty_or_whitespace_token",
            })
            located_tokens.append(located)
            continue

        while cursor < len(query) and query[cursor].isspace():
            cursor += 1
        match = re.match(
            _whitespace_flexible_pattern(visible_piece),
            query[cursor:],
        )
        if match is not None:
            char_start = cursor
            char_end = cursor + match.end()
            status = "sequential_exact"
        else:
            char_start = query.find(visible_piece, cursor)
            if char_start >= 0:
                char_end = char_start + len(visible_piece)
                status = "forward_search"
            else:
                char_start = -1
                char_end = -1
                status = "unmapped"

        if char_start >= 0:
            cursor = char_end
            located.update({
                "query_char_span": [char_start, char_end],
                "query_surface_text": query[char_start:char_end],
                "pos_mapping_status": status,
            })
        else:
            located.update({
                "query_char_span": None,
                "query_surface_text": visible_piece,
                "pos_mapping_status": status,
            })
        located_tokens.append(located)
    return located_tokens


def _attach_contextual_pos(
    query: str,
    query_tokens: Sequence[Mapping[str, Any]],
    pos_tagger: ContextualPosTagger,
) -> List[Dict[str, Any]]:
    """Map word-level contextual POS spans onto every overlapping model token."""
    tagged_words = list(pos_tagger.tag(query))
    located_tokens = _locate_query_token_spans(query, query_tokens)

    for token_record in located_tokens:
        token_span = token_record["query_char_span"]
        best_word = None
        best_overlap = 0
        if token_span is not None:
            token_start, token_end = (int(value) for value in token_span)
            for tagged_word in tagged_words:
                word_start = int(tagged_word["char_start"])
                word_end = int(tagged_word["char_end"])
                overlap = max(0, min(token_end, word_end) - max(token_start, word_start))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_word = tagged_word

        if best_word is None:
            token_record.update({
                "pos": "UNKNOWN",
                "fine_pos": "",
                "lemma": "",
                "pos_word_text": "",
            })
            if token_record["pos_mapping_status"] not in {
                "empty_or_whitespace_token",
                "unmapped",
            }:
                token_record["pos_mapping_status"] = "no_overlapping_pos_word"
        else:
            token_record.update({
                "pos": str(best_word.get("pos") or "UNKNOWN"),
                "fine_pos": str(best_word.get("fine_pos") or ""),
                "lemma": str(best_word.get("lemma") or ""),
                "pos_word_text": str(best_word.get("text") or ""),
            })
            token_record["pos_mapping_status"] += "+pos_overlap"
    return located_tokens


def _candidate_for_role(
    query_record: Mapping[str, Any],
    role: str,
) -> Mapping[str, Any]:
    matches = [
        candidate
        for candidate in query_record.get("candidates", [])
        if candidate.get("role") == role
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Query {query_record.get('qid', '<unknown>')} must contain exactly "
            f"one {role!r} candidate, found {len(matches)}"
        )
    return matches[0]


def _candidate_token_counts(
    candidate_record: Mapping[str, Any],
    annotated_query_tokens: Sequence[Mapping[str, Any]],
    count_scope: str,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Join alignment counts onto every query token and select positive-count Top-K."""
    summary_by_index = {
        int(row["query_sequence_index"]): row
        for row in candidate_record.get("matched_query_token_summary", [])
    }
    known_indices = {
        int(token["query_sequence_index"]) for token in annotated_query_tokens
    }
    unexpected_indices = sorted(set(summary_by_index) - known_indices)
    if unexpected_indices:
        raise ValueError(
            "Candidate alignment summary references unknown query-token indices: "
            f"{unexpected_indices}"
        )

    token_count_rows = []
    for token in annotated_query_tokens:
        query_sequence_index = int(token["query_sequence_index"])
        summary = summary_by_index.get(query_sequence_index, {})
        row = dict(token)
        row.update({
            "matched_visual_token_count": int(
                summary.get("matched_visual_token_count", 0)
            ),
            "matched_kept_visual_token_count": int(
                summary.get("matched_kept_visual_token_count", 0)
            ),
            "matched_pruned_visual_token_count": int(
                summary.get("matched_pruned_visual_token_count", 0)
            ),
        })
        row["selected_alignment_count"] = row[COUNT_FIELDS[count_scope]]
        token_count_rows.append(row)

    ranked_rows = sorted(
        (
            row
            for row in token_count_rows
            if row["selected_alignment_count"] > 0
            and not bool(row.get("is_special_token", False))
        ),
        key=lambda row: (
            -int(row["selected_alignment_count"]),
            int(row["query_sequence_index"]),
        ),
    )[:top_k]
    top_rows = [
        {"top_rank": rank, **row}
        for rank, row in enumerate(ranked_rows, start=1)
    ]
    return token_count_rows, top_rows


def _candidate_pos_output(
    candidate_record: Mapping[str, Any],
    annotated_query_tokens: Sequence[Mapping[str, Any]],
    count_scope: str,
    top_k: int,
) -> Dict[str, Any]:
    """Create the per-candidate token counts shared by old and new statistics."""
    token_count_rows, top_rows = _candidate_token_counts(
        candidate_record,
        annotated_query_tokens,
        count_scope,
        top_k,
    )
    return {
        "role": candidate_record.get("role"),
        "candidate_letter": candidate_record.get("candidate_letter"),
        "final_rank": candidate_record.get("final_rank"),
        "local_page_id": candidate_record.get("local_page_id"),
        "is_ground_truth": bool(candidate_record.get("is_ground_truth")),
        "num_positive_count_query_tokens": sum(
            row["selected_alignment_count"] > 0 for row in token_count_rows
        ),
        "query_token_alignment_counts": token_count_rows,
        "top_query_tokens": top_rows,
    }


def _select_outcome_group_candidates(
    query_record: Mapping[str, Any],
    base_candidate_outputs: Mapping[str, Mapping[str, Any]],
    additional_incorrect_outputs: Sequence[Mapping[str, Any]],
) -> Tuple[str, Mapping[str, Any], List[Mapping[str, Any]]]:
    """Select candidates using the requested Top1-correct/incorrect rules."""
    qid = query_record.get("qid", "<unknown>")
    correct_candidate = base_candidate_outputs[CORRECT_ROLE]
    highest_incorrect = base_candidate_outputs[INCORRECT_ROLE]
    correct_rank = int(correct_candidate["final_rank"])
    inferred_top1_correct = correct_rank == 1
    declared_top1_correct = query_record.get("recall_at_1_correct")
    if (
        declared_top1_correct is not None
        and bool(declared_top1_correct) != inferred_top1_correct
    ):
        raise ValueError(
            f"Query {qid} has inconsistent recall_at_1_correct and GT rank"
        )

    if inferred_top1_correct:
        if bool(highest_incorrect["is_ground_truth"]):
            raise ValueError(f"Query {qid} labels its highest incorrect as GT")
        return TOP1_CORRECT_GROUP, correct_candidate, [highest_incorrect]

    # Because this is the highest-ranked GT, every rank before it must be an
    # error. Exact rank coverage detects legacy JSON files that saved only the
    # first error and would otherwise silently undercount the Top1-wrong group.
    incorrect_candidates = [highest_incorrect, *additional_incorrect_outputs]
    if any(bool(candidate["is_ground_truth"]) for candidate in incorrect_candidates):
        raise ValueError(f"Query {qid} contains a GT in its incorrect-candidate set")
    incorrect_candidates = sorted(
        incorrect_candidates,
        key=lambda candidate: int(candidate["final_rank"]),
    )
    actual_ranks = [int(candidate["final_rank"]) for candidate in incorrect_candidates]
    expected_ranks = list(range(1, correct_rank))
    if actual_ranks != expected_ranks:
        raise ValueError(
            f"Query {qid} needs incorrect candidates at final ranks "
            f"{expected_ranks}, but the alignment JSON contains {actual_ranks}. "
            "Rerun evaluate_token_alignment.sh with "
            "--collect_incorrect_before_correct."
        )
    return TOP1_INCORRECT_GROUP, correct_candidate, incorrect_candidates


def _accumulate_top_token_pos(
    candidate_output: Mapping[str, Any],
    pos_counts: Counter,
    rank_counts: DefaultDict[int, Counter],
) -> None:
    """Accumulate POS occurrences while retaining their Top-K rank positions."""
    for top_row in candidate_output["top_query_tokens"]:
        pos = str(top_row.get("pos") or "UNKNOWN")
        top_rank = int(top_row["top_rank"])
        pos_counts[pos] += 1
        rank_counts[top_rank][pos] += 1


def _pos_statistics(
    pos_counts: Counter,
    rank_counts: Mapping[int, Counter],
    num_queries: int,
    num_candidates: int,
) -> Dict[str, Any]:
    total_slots = sum(pos_counts.values())
    rows = []
    for pos, count in sorted(pos_counts.items(), key=lambda item: (-item[1], item[0])):
        rows.append({
            "pos": pos,
            "total_top_k_occurrences": count,
            "mean_occurrences_per_query": count / num_queries if num_queries else 0.0,
            "mean_occurrences_per_candidate": (
                count / num_candidates if num_candidates else 0.0
            ),
            "fraction_of_observed_top_k_slots": (
                count / total_slots if total_slots else 0.0
            ),
        })
    return {
        "num_queries": num_queries,
        "num_candidates": num_candidates,
        "observed_top_k_slots": total_slots,
        "pos_statistics": rows,
        "pos_counts_by_top_rank": {
            str(rank): dict(sorted(counter.items()))
            for rank, counter in sorted(rank_counts.items())
        },
    }


def analyze_token_alignment_pos(
    alignment_payload: Mapping[str, Any],
    pos_tagger: ContextualPosTagger,
    *,
    top_k: int = 5,
    count_scope: str = "all",
    expected_num_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """Analyze every query record without changing the source alignment payload."""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if count_scope not in COUNT_FIELDS:
        raise ValueError(
            f"count_scope must be one of {sorted(COUNT_FIELDS)}, got {count_scope!r}"
        )

    source_queries = list(alignment_payload.get("queries", []))
    declared_queries = int(alignment_payload.get("num_queries_exported", len(source_queries)))
    if declared_queries != len(source_queries):
        raise ValueError(
            "Input num_queries_exported does not match the number of query records"
        )
    if expected_num_queries is not None and len(source_queries) != expected_num_queries:
        raise ValueError(
            f"Expected {expected_num_queries} input queries, found {len(source_queries)}"
        )

    role_pos_counts: Dict[str, Counter] = {
        role: Counter() for role in ANALYZED_ROLES
    }
    role_rank_counts: Dict[str, DefaultDict[int, Counter]] = {
        role: defaultdict(Counter) for role in ANALYZED_ROLES
    }
    combined_pos_counts: Counter = Counter()
    combined_rank_counts: DefaultDict[int, Counter] = defaultdict(Counter)
    outcome_query_counts: Counter = Counter()
    outcome_candidate_counts = {
        group: Counter() for group in OUTCOME_GROUPS
    }
    outcome_pos_counts = {
        group: {
            candidate_class: Counter()
            for candidate_class in OUTCOME_CANDIDATE_CLASSES
        }
        for group in OUTCOME_GROUPS
    }
    outcome_rank_counts = {
        group: {
            candidate_class: defaultdict(Counter)
            for candidate_class in OUTCOME_CANDIDATE_CLASSES
        }
        for group in OUTCOME_GROUPS
    }
    mapping_status_counts: Counter = Counter()
    output_queries = []

    for query_record in source_queries:
        query_text = str(query_record["query"])
        annotated_query_tokens = _attach_contextual_pos(
            query_text,
            query_record["qi_text_token_sequence"],
            pos_tagger,
        )
        mapping_status_counts.update(
            token["pos_mapping_status"] for token in annotated_query_tokens
        )

        candidate_outputs = []
        for role in ANALYZED_ROLES:
            candidate_record = _candidate_for_role(query_record, role)
            candidate_output = _candidate_pos_output(
                candidate_record,
                annotated_query_tokens,
                count_scope,
                top_k,
            )
            for top_row in candidate_output["top_query_tokens"]:
                pos = str(top_row.get("pos") or "UNKNOWN")
                rank = int(top_row["top_rank"])
                role_pos_counts[role][pos] += 1
                role_rank_counts[role][rank][pos] += 1
                combined_pos_counts[pos] += 1
                combined_rank_counts[rank][pos] += 1
            candidate_outputs.append(candidate_output)

        additional_incorrect_outputs = [
            _candidate_pos_output(
                candidate_record,
                annotated_query_tokens,
                count_scope,
                top_k,
            )
            for candidate_record in query_record.get(
                "additional_incorrect_candidates_before_correct",
                [],
            )
        ]
        base_candidate_outputs = {
            str(candidate["role"]): candidate for candidate in candidate_outputs
        }
        outcome_group, selected_correct, selected_incorrect = (
            _select_outcome_group_candidates(
                query_record,
                base_candidate_outputs,
                additional_incorrect_outputs,
            )
        )
        outcome_query_counts[outcome_group] += 1
        selected_by_class = {
            "correct_candidates": [selected_correct],
            "incorrect_candidates": selected_incorrect,
        }
        for candidate_class, selected_candidates in selected_by_class.items():
            outcome_candidate_counts[outcome_group][candidate_class] += len(
                selected_candidates
            )
            for selected_candidate in selected_candidates:
                _accumulate_top_token_pos(
                    selected_candidate,
                    outcome_pos_counts[outcome_group][candidate_class],
                    outcome_rank_counts[outcome_group][candidate_class],
                )

        output_queries.append({
            "qid": query_record.get("qid"),
            "doc_name": query_record.get("doc_name"),
            "domain": query_record.get("domain"),
            "q_idx": query_record.get("q_idx"),
            "query": query_text,
            "query_tokens_with_pos": annotated_query_tokens,
            "candidates": candidate_outputs,
            "top1_outcome_group": outcome_group,
            "outcome_group_candidate_selection": {
                "correct_candidate": {
                    "candidate_letter": selected_correct["candidate_letter"],
                    "final_rank": selected_correct["final_rank"],
                    "local_page_id": selected_correct["local_page_id"],
                },
                "incorrect_candidates": [
                    {
                        "candidate_letter": candidate["candidate_letter"],
                        "final_rank": candidate["final_rank"],
                        "local_page_id": candidate["local_page_id"],
                    }
                    for candidate in selected_incorrect
                ],
            },
            "additional_incorrect_candidates_before_correct": (
                additional_incorrect_outputs
            ),
        })

    num_queries = len(output_queries)
    role_statistics = {
        role: _pos_statistics(
            role_pos_counts[role],
            role_rank_counts[role],
            num_queries=num_queries,
            num_candidates=num_queries,
        )
        for role in ANALYZED_ROLES
    }
    combined_statistics = _pos_statistics(
        combined_pos_counts,
        combined_rank_counts,
        num_queries=num_queries,
        num_candidates=num_queries * len(ANALYZED_ROLES),
    )
    outcome_descriptions = {
        TOP1_CORRECT_GROUP: (
            "Top1 is correct: use that correct candidate and the highest-ranked "
            "incorrect candidate."
        ),
        TOP1_INCORRECT_GROUP: (
            "Top1 is incorrect: use the highest-ranked correct candidate and "
            "every incorrect candidate ranked ahead of it."
        ),
    }
    outcome_statistics = {}
    for group in OUTCOME_GROUPS:
        group_num_queries = int(outcome_query_counts[group])
        group_statistics: Dict[str, Any] = {
            "description": outcome_descriptions[group],
            "num_queries": group_num_queries,
        }
        for candidate_class in OUTCOME_CANDIDATE_CLASSES:
            group_statistics[candidate_class] = _pos_statistics(
                outcome_pos_counts[group][candidate_class],
                outcome_rank_counts[group][candidate_class],
                num_queries=group_num_queries,
                num_candidates=int(
                    outcome_candidate_counts[group][candidate_class]
                ),
            )
        incorrect_count = int(
            outcome_candidate_counts[group]["incorrect_candidates"]
        )
        group_statistics["incorrect_candidate_count"] = {
            "total": incorrect_count,
            "mean_per_query": (
                incorrect_count / group_num_queries if group_num_queries else 0.0
            ),
        }
        outcome_statistics[group] = group_statistics

    return {
        "schema_version": 1,
        "description": (
            "Part-of-speech distribution among the query tokens attracting the "
            "largest numbers of visual tokens in the highest-ranked correct and "
            "highest-ranked incorrect candidate for every input query."
        ),
        "source_alignment": {
            "schema_version": alignment_payload.get("schema_version"),
            "num_queries_exported": declared_queries,
            "run": alignment_payload.get("run", {}),
        },
        "configuration": {
            "top_k_query_tokens_per_candidate": top_k,
            "count_scope": count_scope,
            "ranking_count_field": COUNT_FIELDS[count_scope],
            "exclude_zero_count_tokens": True,
            "exclude_special_tokens_from_top_k": True,
            "candidate_roles": list(ANALYZED_ROLES),
            "subword_pos_semantics": (
                "Each model query token receives the contextual POS of the tagged "
                "word span with which it has the largest character overlap. "
                "Different subwords of one word can therefore contribute multiple "
                "Top-K occurrences of the same POS."
            ),
        },
        "pos_tagger": dict(pos_tagger.metadata),
        "coverage": {
            "queries_in_input": declared_queries,
            "queries_analyzed": num_queries,
            "expected_num_queries": expected_num_queries,
            "all_input_queries_analyzed": num_queries == declared_queries,
            "query_token_pos_mapping_status_counts": dict(
                sorted(mapping_status_counts.items())
            ),
        },
        "aggregate": {
            "by_candidate_role": role_statistics,
            "correct_and_incorrect_combined": combined_statistics,
        },
        "aggregate_by_top1_outcome": outcome_statistics,
        "queries": output_queries,
    }
