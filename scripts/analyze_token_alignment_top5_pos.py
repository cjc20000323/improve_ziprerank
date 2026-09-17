"""Create separate aggregate and per-query Top-K POS reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.token_alignment_pos_analysis import (  # noqa: E402
    SpacyPosTagger,
    analyze_token_alignment_pos,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count visual-token alignments for every query token, retain the "
            "Top-K tokens per GT/best-wrong candidate, and save aggregate POS "
            "statistics separately from per-query results."
        )
    )
    parser.add_argument(
        "--input_file",
        required=True,
        help="Existing token_alignment_analysis JSON; it is opened read-only.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help=(
            "Aggregate statistics JSON. Defaults to <input_stem>_top5_pos.json "
            "beside the input file. The input file is never overwritten."
        ),
    )
    parser.add_argument(
        "--query_output_file",
        default=None,
        help=(
            "Per-query results JSON. Defaults to <output_stem>_queries.json "
            "beside --output_file."
        ),
    )
    parser.add_argument(
        "--spacy_model",
        default="en_core_web_sm",
        help="Installed spaCy model name or local trained-pipeline directory.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Most frequently aligned positive-count query tokens per candidate.",
    )
    parser.add_argument(
        "--count_scope",
        choices=["all", "kept", "pruned"],
        default="all",
        help=(
            "Alignment count used to rank query tokens. all counts every visual "
            "token; kept/pruned restrict the count by pruning outcome."
        ),
    )
    parser.add_argument(
        "--expected_num_queries",
        type=int,
        default=None,
        help=(
            "Optional completeness check. The command fails without writing if "
            "the input JSON does not contain exactly this many queries."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = (
        Path(args.output_file)
        if args.output_file
        else input_path.with_name(f"{input_path.stem}_top5_pos.json")
    )
    query_output_path = (
        Path(args.query_output_file)
        if args.query_output_file
        else output_path.with_name(
            f"{output_path.stem}_queries{output_path.suffix}"
        )
    )
    resolved_paths = {
        input_path.resolve(),
        output_path.resolve(),
        query_output_path.resolve(),
    }
    if len(resolved_paths) != 3:
        raise ValueError(
            "--input_file, --output_file, and --query_output_file must be "
            "three different paths"
        )

    alignment_payload = json.loads(input_path.read_text(encoding="utf-8"))
    pos_tagger = SpacyPosTagger(args.spacy_model)
    aggregate_report, query_report = analyze_token_alignment_pos(
        alignment_payload,
        pos_tagger,
        top_k=args.top_k,
        count_scope=args.count_scope,
        expected_num_queries=args.expected_num_queries,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    query_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(aggregate_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    query_output_path.write_text(
        json.dumps(query_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "input_file_unchanged": str(input_path),
        "aggregate_output_file": str(output_path),
        "query_output_file": str(query_output_path),
        "queries_analyzed": aggregate_report["coverage"]["queries_analyzed"],
        "top_k": args.top_k,
        "count_scope": args.count_scope,
        "spacy_model": args.spacy_model,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
