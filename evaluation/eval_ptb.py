#!/usr/bin/env python3
"""Evaluate PTB trees with the exact Syntax-RL paper parameter file."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALB = ROOT / "third_party" / "EVALB" / "evalb"
DEFAULT_PARAM = ROOT / "third_party" / "EVALB" / "nk.prm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--evalb", type=Path, default=DEFAULT_EVALB)
    parser.add_argument("--param", type=Path, default=DEFAULT_PARAM)
    args = parser.parse_args()

    for path in (args.gold, args.pred, args.evalb, args.param):
        if not path.is_file():
            parser.error(f"file not found: {path}")

    result = subprocess.run(
        [str(args.evalb), "-p", str(args.param), str(args.gold), str(args.pred)],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = {}
    for name in ("Bracketing Recall", "Bracketing Precision", "Bracketing FMeasure"):
        match = re.search(rf"{name}\s*=\s*([0-9.]+)", result.stdout)
        if not match:
            raise RuntimeError(f"EVALB output is missing {name}")
        metrics[name] = float(match.group(1))

    print(f"LR={metrics['Bracketing Recall']:.2f}")
    print(f"LP={metrics['Bracketing Precision']:.2f}")
    print(f"F1={metrics['Bracketing FMeasure']:.2f}")


if __name__ == "__main__":
    main()

