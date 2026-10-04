#!/usr/bin/env python3
"""Extract PTB trees from JSONL and replace preterminal labels with gold POS."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from nltk.tree import Tree


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: object) -> str:
    value = "" if text is None else str(text).strip()
    match = ANSWER_RE.search(value)
    return (match.group(1) if match else value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="prediction JSONL")
    parser.add_argument("--pos", required=True, type=Path, help="one space-separated POS sequence per line")
    parser.add_argument("--output", required=True, type=Path, help="one prepared tree per line")
    parser.add_argument("--prediction-key", default="predict")
    parser.add_argument("--input-format", choices=("auto", "jsonl", "trees"), default="auto")
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.input.resolve() == args.output.resolve() or args.pos.resolve() == args.output.resolve():
        parser.error("output must differ from the input files")

    input_lines = [line for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    input_format = args.input_format
    if input_format == "auto":
        input_format = "jsonl" if args.input.suffix == ".jsonl" else "trees"
    if input_format == "jsonl":
        predictions = [json.loads(line)[args.prediction_key] for line in input_lines]
    else:
        predictions = input_lines
    pos_lines = [line.split() for line in args.pos.read_text(encoding="utf-8").splitlines()]
    if len(predictions) != len(pos_lines):
        parser.error(f"prediction/POS line mismatch: {len(predictions)} != {len(pos_lines)}")

    prepared = []
    for index, (prediction, tags) in enumerate(zip(predictions, pos_lines), 1):
        try:
            tree = Tree.fromstring(extract_answer(prediction))
        except Exception as error:
            raise ValueError(f"line {index}: invalid tree; run repair_with_llm.py first") from error
        preterminals = list(tree.subtrees(lambda subtree: subtree.height() == 2))
        if len(preterminals) != len(tags):
            raise ValueError(
                f"line {index}: POS/tree length mismatch: {len(tags)} != {len(preterminals)}"
            )
        for subtree, tag in zip(preterminals, tags):
            subtree.set_label(tag)
        prepared.append(" ".join(str(tree).split()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(prepared) + "\n", encoding="utf-8")
    print(f"wrote {len(prepared)} trees to {args.output}")


if __name__ == "__main__":
    main()
