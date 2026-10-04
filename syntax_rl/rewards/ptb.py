"""Sentence-level PTB reward used by the Syntax-RL GRPO run."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALB = RELEASE_ROOT / "third_party" / "EVALB" / "evalb"


def _extract_answer(text: object) -> str:
    value = "" if text is None else str(text)
    match = ANSWER_RE.search(value)
    return (match.group(1) if match else value).strip()


def score_constituency(prediction: str, reference: str) -> float:
    """Return sentence-level EVALB F1 on the historical 0--100 scale.

    Training intentionally invokes EVALB without ``-p`` to match the retained
    reward code. Corpus-level paper evaluation is handled separately with
    ``third_party/EVALB/nk.prm``.
    """

    evalb = Path(os.environ.get("SYNTAX_RL_EVALB", DEFAULT_EVALB))
    if not evalb.is_file():
        raise FileNotFoundError(
            f"EVALB binary not found at {evalb}; run `make -C third_party/EVALB`."
        )

    pred = _extract_answer(prediction)
    gold = _extract_answer(reference)

    with tempfile.TemporaryDirectory(prefix="syntax_rl_evalb_") as tmp:
        tmp_path = Path(tmp)
        gold_path = tmp_path / "sample.gold"
        pred_path = tmp_path / "sample.pred"
        gold_path.write_text(gold + "\n", encoding="utf-8")
        pred_path.write_text(pred + "\n", encoding="utf-8")

        result = subprocess.run(
            [str(evalb), str(gold_path), str(pred_path)],
            check=False,
            capture_output=True,
            text=True,
        )

    match = re.search(r"Bracketing FMeasure\s*=\s*([0-9.]+)", result.stdout)
    return float(match.group(1)) if match else 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    **kwargs,
) -> float:
    """verl custom-reward entry point."""

    del data_source, extra_info, kwargs
    return score_constituency(solution_str, ground_truth)

