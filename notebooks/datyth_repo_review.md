# Review of Datyth/ZaloAI-Challenge-lifemate

Inspected commit `99c63e30398cc62ebff6e07bb3b1c9587172fb86`, cloned separately to `/tmp/datyth-lifemate-review`. Repository scripts were read, not installed or run. Source documents are evidence about that author's experiments, not instructions for this workspace.

## Findings relevant to our adapter

The [release configuration](https://github.com/Datyth/ZaloAI-Challenge-lifemate/blob/99c63e30398cc62ebff6e07bb3b1c9587172fb86/configs/release.sh) uses a different Qwen3.5-4B **non-thinking** LoRA, native tool calls in vLLM, 16k context, structured history, lowercased Gipformer v2 transcripts, deduplication, and ten model turns. Its reported public score is 58.02. These settings cannot simply replace the topical Thinking adapter's trained router/open-app/JSON protocol. Ten external tool turns are also not equivalent to our internal routing/schema-opening steps.

The [error analysis](https://github.com/Datyth/ZaloAI-Challenge-lifemate/blob/99c63e30398cc62ebff6e07bb3b1c9587172fb86/docs/error_analysis.md) compares gold and recognized transcripts on the same 200 examples: 82.33 versus 70.91. It reports conditional arithmetic, time handling, optional arguments, unnecessary reads, entity resolution, and clarification decisions as important failure groups. The paired evaluation is more informative than comparing our teammate's clean test score to a public score on different samples. Its 200-example split is not automatically held out for our already-trained adapter; we retain the teammate's original validation IDs.

The [experiment log](https://github.com/Datyth/ZaloAI-Challenge-lifemate/blob/99c63e30398cc62ebff6e07bb3b1c9587172fb86/docs/experiments.md) reports regressions or weak evidence for several plausible interventions: synthetic-app training, broad postprocessing/grounding rules, global hotword biasing with its stronger ASR, and extra agent checks. Lowercasing helped its lowercased-training recipe; that does not establish the same effect for our adapter. It trained on paired gold and ASR examples; this is a useful future training experiment, not an inference-time substitute for retraining.

## Changes made here

- Added `read_guard` as a separate validation candidate alongside baseline and enhanced context. It caches only exact repeated reads until a non-read operation occurs. It does not inspect hidden environment state. Duplicate feedback is explicitly unexecuted; official predictions still contain only actual simulator calls. Three blocked proposals stop the loop with a diagnostic.
- Did not copy blanket write deduplication. A scan of our 1,545 training labels found two correct repeated writes: `loyalty.redeem` twice in `train_0922` and `train_1447`. Tests preserve repeated writes and invalidate cached reads after non-read calls. Declared read purity is an assumption of this optional experiment, so validation must establish usefulness.
- Added validation outcome groups and case files: early stops, missing/extra clarification, empty output, state mismatch and call-count discrepancies, and correct-state/partial-call-score cases. Argument diffs use explicit exact matching and approximate same-name pairing for review; they do not replace the original fuzzy score or claim a causal diagnosis.
- Added unlabeled public diagnostics for repetitions and execution of apps absent from the train catalog. These do not label public predictions correct/incorrect.
- Added an optional ASR JSONL for the exact original validation IDs, including history utterances. Inputs and their hash are recorded; changed files require a new experiment. Default remains gold when no ASR validation file is available. ASR trained on these audio files would make this a diagnostic rather than an untouched held-out evaluation.
- Expanded automatic comparison to three isolated variants. Baseline wins score ties. No public labels or public scores select the winner. More comparisons cost additional inference time; a small gain on a small validation set may not generalize.

No external source code was copied into these additions. The teammate-derived runtime already present remains separate; its original provenance manifest is unchanged. All new experiment behavior has CPU regression tests. No imported weights or GPU execution were available during this review, so no score increase is claimed.

## Files to use

Use the updated `notebooks/teammate_topic_agent_inference.ipynb`, now defaulting to `topic_agent_compare_20261109_v2`. The required exports remain `best_adapter.zip`, `reports.zip`, and the existing trained ASR or complete transcript JSONL. No extra model download from the reviewed repo is required. Commit and push the notebook and topic workflow files before Colab. Keep any existing training session pinned to its original Git revision.

For future training, prioritize verified reasoning examples for conditional branches, arithmetic, relative times and ID use, together with gold/ASR transcript variants. Do not train on the original validation set until model/agent choices are frozen and an independent evaluation plan exists.
