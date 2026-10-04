#!/usr/bin/env python3
"""Repair the small number of malformed PTB/CCG predictions with an LLM.

The script never exposes gold labels to the repair model. It sends only the
source sentence and the model prediction, validates the returned structure,
and keeps an audit JSONL with both the original and candidate text.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
TREE_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")
CCG_PAIR_RE = re.compile(r"\{(\S+)\s+([^{}\s]+)\}")


def extract_answer(text: object) -> str:
    value = "" if text is None else str(text).strip()
    match = ANSWER_RE.search(value)
    return (match.group(1) if match else value).strip()


def clean_candidate(text: str) -> str:
    value = extract_answer(text)
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    return " ".join(value.split())


def source_from_record(record: dict, task: str) -> str:
    for key in ("input", "sentence", "source"):
        if record.get(key):
            return str(record[key]).strip()

    prompt = record.get("prompt")
    if isinstance(prompt, list):
        user_messages = [item.get("content", "") for item in prompt if item.get("role") == "user"]
        if user_messages:
            return _strip_ptb_instruction(str(user_messages[-1]), task)
    if isinstance(prompt, str):
        value = prompt
        if "\nuser\n" in value:
            value = value.rsplit("\nuser\n", 1)[1]
        if "\nassistant\n" in value:
            value = value.split("\nassistant\n", 1)[0]
        return _strip_ptb_instruction(value, task)
    raise ValueError("record has no input, sentence, source, or recognizable prompt")


def _strip_ptb_instruction(value: str, task: str) -> str:
    text = value.strip()
    instruction = "Generate the constituency tree of the given sentence."
    if task == "ptb" and text.startswith(instruction):
        text = text[len(instruction) :].strip()
    return text


def ptb_leaves(tree: str) -> list[str]:
    tokens = TREE_TOKEN_RE.findall(extract_answer(tree))
    position = 0

    def parse_node() -> list[str]:
        nonlocal position
        if position >= len(tokens) or tokens[position] != "(":
            raise ValueError("expected '('")
        position += 1
        if position >= len(tokens) or tokens[position] in {"(", ")"}:
            raise ValueError("missing node label")
        position += 1  # node label
        leaves: list[str] = []
        child_count = 0
        while position < len(tokens) and tokens[position] != ")":
            child_count += 1
            if tokens[position] == "(":
                leaves.extend(parse_node())
            else:
                leaves.append(tokens[position])
                position += 1
        if position >= len(tokens):
            raise ValueError("unclosed node")
        if child_count == 0:
            raise ValueError("empty node")
        position += 1
        return leaves

    leaves = parse_node()
    if position != len(tokens):
        raise ValueError("content found outside the tree")
    return leaves


def ccg_pairs(sequence: str) -> list[tuple[str, str]]:
    value = extract_answer(sequence)
    pairs = CCG_PAIR_RE.findall(value)
    remainder = CCG_PAIR_RE.sub("", value)
    if not pairs or remainder.strip():
        raise ValueError("expected only braced CCG category/token pairs")
    return pairs


def validate_ptb(prediction: str, source: str) -> None:
    leaves = ptb_leaves(prediction)
    if leaves != source.split():
        raise ValueError(f"leaf mismatch: expected {len(source.split())}, got {len(leaves)} tokens")


def validate_ptb_historical(prediction: str, source: str) -> None:
    """Match the manual-repair trigger: valid tree and equal token count.

    The historical POS-replacement script checked the number of terminals but
    did not rewrite equal-length lexical substitutions made by the model.
    """

    leaves = ptb_leaves(prediction)
    if len(leaves) != len(source.split()):
        raise ValueError(f"leaf-count mismatch: expected {len(source.split())}, got {len(leaves)} tokens")


def validate_ccg(prediction: str, source: str) -> None:
    tokens = [token for _, token in ccg_pairs(prediction)]
    if tokens != source.split():
        raise ValueError(f"token mismatch: expected {len(source.split())}, got {len(tokens)} tokens")


def repair_prompt(task: str, source: str, prediction: str) -> tuple[str, str]:
    common = (
        "You repair serialization errors in syntactic predictions. Do not solve the task again. "
        "Never consult or infer a gold parse. Preserve the prediction's syntactic decisions and "
        "labels as much as possible. Return only the repaired prediction, with no explanation."
    )
    if task == "ptb":
        request = f"""Repair this PTB-style constituency tree.

Requirements:
- Return exactly one balanced bracketed tree on one line.
- Its terminal leaves must be exactly the SOURCE TOKENS in the same order.
- Make the smallest formatting/token-alignment changes possible.
- Preserve existing nonterminal and POS labels; do not improve the parse.

SOURCE TOKENS:
{source}

PREDICTION:
{prediction}"""
    else:
        request = f"""Repair this CCG supertag sequence.

Requirements:
- Return only whitespace-separated `{{CATEGORY TOKEN}}` pairs.
- The TOKEN fields must be exactly the SOURCE TOKENS in the same order.
- Reuse and align the predicted categories; do not improve them.
- If a source token has no recoverable predicted category, use `__INVALID__` so it scores wrong.
- Drop commentary and formatting artifacts.

SOURCE TOKENS:
{source}

PREDICTION:
{prediction}"""
    return common, request


class ChatCompletionsClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float, retries: int):
        self.endpoint = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.endpoint, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                return str(content)
            except (KeyError, IndexError, TypeError, ValueError, urllib.error.URLError) as error:
                last_error = error
                if attempt < self.retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"LLM request failed after {self.retries + 1} attempts: {last_error}")


def output_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("ptb", "ccg"))
    parser.add_argument("--input", required=True, type=output_path)
    parser.add_argument("--output", required=True, type=output_path)
    parser.add_argument("--audit", required=True, type=output_path)
    parser.add_argument("--prediction-key", default="predict")
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--max-repairs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--strict-token-match",
        action="store_true",
        help="also repair equal-length PTB lexical mismatches (not used for the reported result)",
    )
    args = parser.parse_args()

    if args.input == args.output or args.input == args.audit or args.output == args.audit:
        parser.error("input, output, and audit must be three different paths")
    if args.output.exists() or args.audit.exists():
        parser.error("output/audit already exists; choose new paths to avoid overwriting evidence")
    if not args.dry_run and not args.model:
        parser.error("set --model or LLM_MODEL")
    if args.max_repairs < 0:
        parser.error("--max-repairs must be non-negative")

    if args.task == "ptb":
        detection_validator: Callable[[str, str], None] = (
            validate_ptb if args.strict_token_match else validate_ptb_historical
        )
        candidate_validator: Callable[[str, str], None] = validate_ptb
    else:
        detection_validator = validate_ccg
        candidate_validator = validate_ccg
    client = None if args.dry_run else ChatCompletionsClient(
        args.base_url, args.api_key, args.model, args.timeout, args.retries
    )

    counts = {"valid": 0, "repaired": 0, "failed": 0, "skipped_limit": 0, "dry_run": 0}
    attempts = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)

    with (
        args.input.open(encoding="utf-8") as source_stream,
        args.output.open("x", encoding="utf-8") as output_stream,
        args.audit.open("x", encoding="utf-8") as audit_stream,
    ):
        for line_number, line in enumerate(source_stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            source = source_from_record(record, args.task)
            original = str(record.get(args.prediction_key, ""))
            try:
                detection_validator(original, source)
                counts["valid"] += 1
                output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            except ValueError as error:
                original_error = str(error)

            audit = {
                "line_number": line_number,
                "task": args.task,
                "source": source,
                "original": original,
                "original_error": original_error,
                "model": args.model,
            }
            if attempts >= args.max_repairs:
                audit["status"] = "skipped_limit"
                counts["skipped_limit"] += 1
            elif args.dry_run:
                attempts += 1
                audit["status"] = "dry_run"
                counts["dry_run"] += 1
            else:
                attempts += 1
                system, user = repair_prompt(args.task, source, original)
                try:
                    candidate = clean_candidate(client.complete(system, user))
                    audit["candidate"] = candidate
                    candidate_validator(candidate, source)
                    record[args.prediction_key] = candidate
                    audit["status"] = "repaired"
                    counts["repaired"] += 1
                except Exception as error:  # preserve original and record API/validation failure
                    audit["status"] = "failed"
                    audit["repair_error"] = str(error)
                    counts["failed"] += 1

            output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            audit_stream.write(json.dumps(audit, ensure_ascii=False) + "\n")

    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
