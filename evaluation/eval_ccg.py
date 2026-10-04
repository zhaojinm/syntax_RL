#!/usr/bin/env python3
"""Evaluate CCG supertagging predictions stored as JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def paired_fields(text: str):
    fields = text.split()
    if not fields or len(fields) % 2:
        raise ValueError("expected category/token pairs")
    return [(fields[i], fields[i + 1]) for i in range(0, len(fields), 2)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--prediction-key", default="predict")
    args = parser.parse_args()

    correct = 0
    total = 0
    sentence_count = 0
    with args.predictions.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            gold = paired_fields(str(record[args.label_key]))
            pred = paired_fields(str(record[args.prediction_key]))
            if len(gold) != len(pred):
                raise ValueError(f"line {line_number}: prediction length mismatch")
            for (gold_cat, gold_tok), (pred_cat, pred_tok) in zip(gold, pred):
                if gold_tok != pred_tok:
                    raise ValueError(f"line {line_number}: token alignment mismatch")
                correct += pred_cat == gold_cat
                total += 1
            sentence_count += 1

    if not total:
        raise ValueError("no tokens found")
    print(f"sentences={sentence_count}")
    print(f"correct={correct}")
    print(f"tokens={total}")
    print(f"accuracy={100.0 * correct / total:.4f}")


if __name__ == "__main__":
    main()

