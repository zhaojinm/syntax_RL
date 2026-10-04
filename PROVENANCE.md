# Syntax-RL release provenance

This file records evidence for reconstructing the public release. It is not a
claim that every paper result has already been mapped to an exact run.

## Confirmed runtime source

- All 56 retained Hydra run records point to the same local `verl` worktree.
- All 37 retained verl W&B metadata files that include source metadata record:
  - upstream remote: `https://github.com/verl-project/verl.git`
  - Git commit: `3a394c9bd09c8bd321ae2070edaa3b5e84af201a`
- Retained tracebacks resolve source files under that worktree, rather than an
  installed `site-packages/verl` tree.
- An early 2026-02-14 attempt referenced a reward module in a different
  worktree. It lasted about 20 seconds and has no evidence of completed
  training. Subsequent retained runs use the confirmed local checkout.

These facts establish the source checkout for the retained runs. They do not
establish where unretained Qwen2.5-7B or Qwen2.5-14B paper runs were executed.

## Run reconstruction rule

Current shell scripts are not treated as immutable evidence. The source of
truth for a retained run is, in order:

1. Hydra `.hydra/overrides.yaml` and resolved `.hydra/config.yaml`.
2. W&B `files/config.yaml` and `wandb-metadata.json`.
3. Runtime logs and checkpoint metadata.
4. Current shell scripts, used only when their arguments exactly match the
   retained Hydra overrides.

Release scripts should be regenerated from the retained run configuration,
not copied blindly from the current `verl/runs/` tree.

## Confirmed and candidate mappings

### PTB: 32k SFT + 8k RL

- Candidate script: `verl/runs/seq_sft32k_eq2_rl8k.sh`
- Retained run configuration:
  `verl/outputs/2026-02-14/17-53-33/.hydra/overrides.yaml`
- Status: same experiment lineage, but not an exact current-file match.
- Difference: retained run has rollout GPU memory utilization `0.6`; current
  script has `0.4`.
- The current script was modified on 2026-05-02, after the retained run.
- The two-epoch SFT artifact is
  `prediction/sequence/qwen25-32b_full_sft-32k_LDC99T42-ckpt8000.goldpos`.
  With `self-attentive-parser/EVALB/nk.prm`, it gives LR 96.07, LP 95.98,
  and F1 96.02. This matches the start of the paper's Figure 3 curve.
- The paper's PTB F1 96.52 is exactly reproduced by
  `prediction/sequence/qwen32b-seq-sft32k-ep3-rl8k/`
  `qwen32b-seq-sft32k-ep3-rl8k_step_140.goldpos` with
  `self-attentive-parser/EVALB/nk.prm`: LR 96.29, LP 96.75, F1 96.52.
  The paper reports the same rounded F1 but LR 96.25 and LP 96.79, so the
  retained final prediction is not byte-identical to the paper-evaluation
  artifact or its post-processing.
- The artifact directory name contains `ep3`, but there is no retained Hydra
  or W&B configuration for that run. The author states that the effective
  checkpoint used for the reported comparison was epoch 2 and requested that
  the public SFT recipe use two epochs. The release follows that author record
  while preserving the artifact-name discrepancy here.
- The historical PTB reward invokes EVALB without `-p`; final paper evaluation
  uses the punctuation-excluding `self-attentive-parser/EVALB/nk.prm`.
- Comparing the raw step-140 JSONL predictions with the retained `.fixed` file
  identifies nine manually edited records (1-based lines 78, 116, 208, 748,
  794, 993, 1016, 1634, and 2028). The release's LLM-assisted repair script
  uses the same default trigger and writes a complete before/after audit.

### PTB: 32k SFT + 40k RL

- Script: `verl/runs/seq_sft32k_eq2_rl40k.sh`
- Retained run configuration:
  `verl/outputs/2026-02-15/18-32-37/.hydra/overrides.yaml`
- Status: all 38 Hydra overrides exactly match the current script.
- Paper-result checkpoint and external EVALB artifact: not yet confirmed.

### PTB: 20k SFT + 20k RL

- Multiple ep2 and ep3 runs exist.
- `verl/runs/seq_sft20k_eq3_rl20k.sh` exactly matches
  `verl/outputs/2026-03-05/03-29-21/.hydra/overrides.yaml`.
- The current ep2 script does not exactly match its earliest retained run.
- Status: the specific run used for Table 4 is unresolved.

### CCGbank: 32k SFT + remaining RL split

- Script: `verl/runs/ccg/seq_sft32k_eq2_rl8k.sh`
- Exact retained matches:
  - `verl/outputs/2026-03-10/15-28-40/.hydra/overrides.yaml`
  - `verl/outputs/2026-03-10/15-45-50/.hydra/overrides.yaml`
- Status: launch configuration confirmed. Re-evaluation of retained predictions
  gives a best token accuracy of 96.6806 at step 180, which rounds to 96.68,
  not the paper's 96.70. The precise 96.70 artifact remains unresolved.

### SFT runtime facts

- PTB run: `LlamaFactory/wandb/run-20260213_050911-ds2rp29t`.
- CCG run: `LlamaFactory/wandb/run-20260310_035700-zrripkgb`.
- Both record a machine with eight H200 GPUs, per-device train batch size 1,
  gradient accumulation 2, learning rate 1e-5, and an original three-epoch job.
  Machine GPU inventory alone does not establish the training world size;
  the effective global batch size is not confirmed here.
- Checkpoint 8000 is epoch 2; checkpoint 12000 is epoch 3. The release stops
  SFT at two epochs as requested by the author.

## Missing provenance

- No retained formal PTB Syntax-RL runs for Qwen2.5-7B or Qwen2.5-14B were
  found in the local Hydra/W&B records.
- Several paper numbers still need exact run/checkpoint mappings; PTB 96.02 and
  96.52 are now mapped and reproducible from retained predictions.
- The CCGrebank evaluation pipeline and the stated LCS alignment implementation
  have not yet been located.
- The paper describes a composite format/task reward. The author confirmed that
  the actual runs had only the task reward: malformed output naturally receives
  zero EVALB F1 or zero CCG accuracy. The release preserves that behavior and
  does not add a separate format term.

## Publication constraints

- Do not publish LDC corpora or transformed annotations until redistribution
  rights are confirmed.
- Do not publish checkpoints. Also do not copy W&B logs, predictions, absolute
  paths, or machine metadata into the public Git repository.
- The public source tree may vendor clean, pinned framework snapshots, but must
  not copy the dirty experimental worktrees. Third-party licenses must remain.
- The first release covers PTB and CCGbank only; CTB is intentionally excluded.
