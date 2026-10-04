"""CCG token-accuracy reward used by the Syntax-RL GRPO run."""

from __future__ import annotations

import re
from typing import List, Tuple


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(text: object) -> str:
    value = "" if text is None else str(text)
    match = ANSWER_RE.search(value)
    return (match.group(1) if match else value).strip()


def _parse_historical_sequence(text: str) -> List[Tuple[str, str]]:
    """Parse the whitespace-paired representation used by the original code."""

    fields = text.split()
    if not fields or len(fields) % 2:
        raise ValueError("CCG output must contain category/token pairs")
    return [(fields[i], fields[i + 1]) for i in range(0, len(fields), 2)]


def score_ccg(prediction: str, reference: str) -> float:
    """Return aligned token category accuracy on the 0--1 scale.

    A length mismatch, token mismatch, or malformed sequence returns zero.
    Category comparison drops the first character exactly as in the retained
    reward (normally the opening ``{`` in the braced representation).
    """

    try:
        pred_pairs = _parse_historical_sequence(_extract_answer(prediction))
        gold_pairs = _parse_historical_sequence(_extract_answer(reference))
        if len(pred_pairs) != len(gold_pairs):
            return 0.0

        correct = 0
        for (pred_cat, pred_tok), (gold_cat, gold_tok) in zip(pred_pairs, gold_pairs):
            if pred_tok != gold_tok:
                return 0.0
            if pred_cat[1:] == gold_cat[1:]:
                correct += 1
        return correct / len(gold_pairs)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    **kwargs,
) -> float:
    """verl custom-reward entry point."""

    del data_source, extra_info, kwargs
    return score_ccg(solution_str, ground_truth)

