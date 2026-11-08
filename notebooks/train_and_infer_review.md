# Review of the new training notebook

`train_and_infer.ipynb` targets one Colab GPU and clones the Git repository, following the original notebook's workflow. Implementations and tests live in `training_pipeline/`; the notebook contains configuration and stage calls. Commit and push the new directory and notebook before using Colab. The original `lifemate_colab.ipynb`, including its saved outputs, was not edited.

## What the supplied results establish

- The original baseline used untrained Qwen3-4B-Instruct-2507 and PhoWhisper-small. Your reported public score is 18.54.
- Its output logs contain 634 completed predictions, 27 unsuccessful tool outcomes, and 28 samples over 30 seconds. The attached submission contains 634 unique predictions, including 88 with no calls. These facts alone cannot attribute the score loss between ASR and planning.
- The completed Qwen3 QLoRA notebook reports 53.4010 validation on 143 samples and 59.8113 held-out test on 82 samples, using gold transcripts. Those internal scores are not public leaderboard scores.
- The supplied alternate-order, recovery, and helper-tool training notebooks have no executed outputs establishing their gains. They also use a different Qwen3.5 routing/training protocol. Their adapters cannot be mixed into the Qwen3 runtime unchanged.

## Recipe and differences

The new notebook adapts the completed Qwen3 recipe: NF4 QLoRA, response-only next-action supervision, explicit completion targets, sequential real tool feedback, BGE multilingual ranking, compact schemas for every installed tool, preservation of recent complete history blocks, retrieval of older blocks, and grouped train/validation/test separation.

It replaces the Kaggle two-GPU launch with one GPU and accumulation 8, retaining effective batch 8. Defaults are rank 8, alpha 16, learning rate 2e-4, and two epochs. Set `EPOCHS=1` to reproduce the source run's epoch count. Two epochs provide candidates for full-rollout validation selection; improvement from the second epoch is not assumed.

Additional Colab work includes:

- Drive mounting and persistent per-run outputs.
- Full resume checkpoints every 100 steps and at epoch boundaries; automatic selection of a complete resume checkpoint.
- Best-epoch selection by state-gated tool-execution score, followed by separate held-out evaluation.
- Best-model export before public ASR is required: adapter, tokenizer, exact base weights and reranker by default, source, configuration, revisions and reports.
- Strict architecture-independent ASR JSONL handoff covering the final request and every user-history turn. ASR output cannot overwrite context, tool results or assistant history. No fallback to gold text.
- Optional inference-only loader for a teammate's merged Whisper-compatible checkpoint. Other architectures use the same transcript handoff.
- SQLite reranking cache on local Colab disk, with checkpoints and execution journals on Drive.
- Full-coverage and official schema validation before copying the submission to the requested top-level Drive folder.

The notebook retains the source implementation's metric field `official_score_100`; it explicitly labels this as a local metric reconstruction. The organizer's scoring remains authoritative.

## Drive locations

Daily continuation is now supported explicitly. Set `TRAIN_START='fresh'` for a new base-model run, or `TRAIN_START='best'` with `PREVIOUS_RUN_NAME` and a new `RUN_NAME` to train additional epochs from the previous selected adapter. The latter snapshots the adapter, records its weight hash, and starts a new optimizer/scheduler. `RESUME_CHECKPOINT='auto'` separately restores an interrupted session's full training state. Previous best weights compete against new checkpoints during validation selection, including a tie-break in their favor. Use held-out test evaluation only after the final daily session.

The updated notebook passed the 106 CPU contracts again, plus checks of actual configuration execution (snapshot, rerun, same-name rejection) and model selection (previous best retained, new best selected, ties retain previous). GPU continuation has not been run locally.

With the default run name:

```text
MyDrive/Zalo-AI-Challenge-2026/
  inputs/public_asr_requests.jsonl
  inputs/public_asr.jsonl                 # teammate supplies this
  runs/qwen3_4b_compact_sft_v1/
    checkpoints/
    best_adapter/
    best_model/                          # selected adapter + base bundle
    selection.json
    test_metrics.json
    public_test_best/submission.jsonl
  submission.jsonl                       # copied only after validation
  submission_provenance.json
```

The bundle is base + LoRA, not a merged full-precision model. The repository's loader restores both. `EXPORT_BASE_WEIGHTS=False` reduces export size but requires the exact pinned Hub models later. Training and model export can finish before the ASR teammate delivers their checkpoint/transcripts.

## Checks performed locally

- All notebook code cells and extracted repository Python modules parsed successfully. The refactored notebook has no embedded source-writing cells.
- The 106 original CPU contract tests pass after extraction, including loss masks, schema preservation, label isolation, metric matching, AMP recovery logic, single-GPU launch accounting, and ASR handoff. Additional host-workflow tests cover daily continuation snapshots, configuration reuse, stage commands, ASR manifests, and preventing failed public runs from replacing an existing submission.
- After extraction, all 112 CPU tests passed. The actual notebook clone cell was also exercised against a temporary local Git repository, including reuse of the same checkout and persisted revision. No network or GPU was needed for that check.
- Replayed all 1,545 labeled training trajectories through the supplied training SDK successfully. No public inference environments were created during these checks.
- Reproduced the grouped split: 1,320 train / 143 validation / 82 test.
- Tested the local real Qwen tokenizer on 40 samples, including the 16 longest histories: 114 supervised targets, correct masked prefixes and EOS, and all installed schemas retained. These plumbing checks used a deterministic fake ranker; they do not measure BGE relevance or predictive accuracy.
- Original notebook SHA256 before and after: `06f8d303fd8c3132d40a9a810ab81d2baf65c0898274dd5c530ab030de796589`.

GPU training, actual BGE inference, model export downloads and public prediction generation were not run locally. No new accuracy or GPU-memory result is claimed. The notebook includes a required real GPU optimizer/memory preflight. Its staged planner timings exclude ASR and reranking and therefore do not certify the final end-to-end latency requirement.

Public inference journals resume completed samples, but deliberately refuse to reset an interrupted in-flight sample, consistent with the supplied SDK's single-session rule. The notebook explains this limit. This is a training/submission notebook, not a final private-round Docker package.

Upstream interfaces checked while preparing the notebook: [Qwen's Unsloth training guide](https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/unsloth.md) and [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer).
