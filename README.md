# Syntax-RL

Code release for *Syntactic Prediction through Reinforcement Learning*,
accepted to **EMNLP 2026 Main Conference**.

This release currently covers Penn Treebank constituency parsing (PTB) and
CCGbank supertagging (CCG).

## What is included

- A clean snapshot of verl at commit
  `3a394c9bd09c8bd321ae2070edaa3b5e84af201a`, the checkout recorded by all
  retained RL runs.
- A clean snapshot of LLaMA-Factory at commit
  `92fa3df4c4e4e338ecc1cbdc6530b118337286e3`, the checkout recorded by the
  retained SFT runs.
- Two-epoch Qwen2.5-32B SFT configurations for PTB and CCG.
- GRPO launchers reconstructed from retained Hydra overrides.
- The EVALB source used for reported PTB evaluation.
- Three real examples per JSON file from the PTB and CCGbank training splits,
  illustrating the SFT and RL schemas with their original annotations.

Apart from these small excerpts, the full treebanks and derived datasets are
not included. Model weights and experiment logs are also excluded.


## Environment

Install the vendored projects in separate environments following their upstream documentation:

```bash
pip install -e vendor/LlamaFactory
pip install -e vendor/verl
make -C third_party/EVALB
```

## Data

Obtain PTB and CCGbank from their official distributors and prepare the
licensed annotations in the schemas illustrated by:

- `data/ptb_sft_examples.json` and `data/ccg_sft_examples.json`
- `data/ptb_rl_examples.json` and `data/ccg_rl_examples.json`

The examples are selected from the PTB and CCGbank training data used in our
experiments. Each file contains three real records with unchanged source
sentences and annotations: SFT examples come from the 32k SFT split, and RL
examples come from the remaining RL split. RL metadata retains its original
values, including `data_source` and `extra_info.index`. Data conversion
scripts are intentionally not distributed. The experimental split
was random: 32,000 examples for SFT and the disjoint remainder for RL (7,832
for PTB and 7,604 for CCGbank).

For SFT, replace the ignored `data/ptb_sft.json` or `data/ccg_sft.json` with
your licensed full dataset. For RL, provide verl-compatible Parquet files and
pass their paths to the launchers through `TRAIN_FILE` and `VAL_FILE`.

## SFT

Run from the repository root:

```bash
cd vendor/LlamaFactory
llamafactory-cli train ../../configs/sft/ptb_qwen25_32b.yaml
# or
llamafactory-cli train ../../configs/sft/ccg_qwen25_32b.yaml
```

Both configs train for exactly two epochs. The default output locations are
`outputs/sft/ptb` and `outputs/sft/ccg`.

## RL

Set the licensed data paths and, if needed, override the SFT model path:

```bash
TRAIN_FILE=/path/to/ptb_rl.parquet \
VAL_FILE=/path/to/ptb_test.parquet \
MODEL_PATH="$PWD/outputs/sft/ptb" \
bash configs/rl/train_ptb.sh

TRAIN_FILE=/path/to/ccg_rl.parquet \
VAL_FILE=/path/to/ccg_test.parquet \
MODEL_PATH="$PWD/outputs/sft/ccg" \
bash configs/rl/train_ccg.sh
```

The PTB reward uses EVALB without a parameter file, matching the historical
training code. Reported corpus-level evaluation uses the bundled `nk.prm`,
which excludes punctuation and empty elements. Training reward and final
evaluation are therefore deliberately separate commands.

## Evaluation

Compile EVALB, then evaluate one tree per line:

```bash
python evaluation/prepare_ptb_predictions.py \
  --input predictions.repaired.jsonl \
  --pos /path/to/test.pos \
  --output predictions.goldpos.mrg
python evaluation/eval_ptb.py --gold /path/to/test.mrg --pred predictions.goldpos.mrg
python evaluation/eval_ccg.py --predictions /path/to/predictions.jsonl
```

For the PTB command, use `predictions.goldpos.mrg` as `--pred`.
`prepare_ptb_predictions.py` reproduces the original evaluation step that
assigns the gold POS sequence before punctuation-excluding EVALB scoring.
CCG JSONL records must contain `label` and `predict`. Categories and tokens use
the flat braced format shown in the examples.

### Repairing the few malformed predictions

The original reported evaluation manually repaired fewer than ten malformed
outputs. `evaluation/repair_with_llm.py` provides an auditable replacement. It
calls any OpenAI-compatible chat-completions endpoint only for records that
fail the task-specific structural validation; valid records preserve their
prediction fields exactly. Gold labels are never sent to the API. The source sentence and
raw prediction are sent to the configured endpoint, so use a local endpoint
unless your treebank license and data policy permit an external service.

First inspect which records would be sent:

```bash
python evaluation/repair_with_llm.py \
  --task ptb \
  --input predictions.jsonl \
  --output predictions.dry-run.jsonl \
  --audit repairs.dry-run.jsonl \
  --dry-run
```

Then configure the endpoint and run the repair:

```bash
export LLM_BASE_URL=https://your-endpoint.example/v1
export LLM_API_KEY=your-key
export LLM_MODEL=your-model

python evaluation/repair_with_llm.py \
  --task ptb \
  --input predictions.jsonl \
  --output predictions.repaired.jsonl \
  --audit repairs.audit.jsonl
```

Use `--task ccg` for CCG token/category alignment. The script refuses to
overwrite an existing output or audit file, limits calls to 20 by default, and
retains the original prediction whenever the API fails or the returned tokens
do not exactly match the source sentence. Review the audit before evaluation.
For PTB, the default detection rule matches the historical manual workflow:
repair malformed trees and token-count mismatches, but do not repair a model's
equal-length lexical substitution. `--strict-token-match` opts into repairing
those additional cases and therefore should not be used to reproduce the
reported paper number.

## Citation

If you use Syntax-RL in your research, please cite:

```bibtex
@inproceedings{zhao-etal-2026-syntactic,
  title = {Syntactic Prediction through Reinforcement Learning},
  author = {Zhao, Jinman and
            Wang, Yining and
            Liu, Jiahe and
            Zhang, Xueyan and
            Cao, Linbo and
            Huang, Jiakang and
            Zhao, Yiren and
            Cai, Renyi and
            Ding, Yitian and
            Penn, Gerald},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year = {2026},
  publisher = {Association for Computational Linguistics}
}
```
