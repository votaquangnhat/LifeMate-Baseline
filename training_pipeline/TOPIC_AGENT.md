# Teammate topical agent: inference import and validation comparison

Use `notebooks/teammate_topic_agent_inference.ipynb`. This is a Git-backed inference notebook, not another training run. It loads the selected adapter from the supplied teammate notebook using its router/open-app/tool-call protocol.

## What to transfer

From the teammate's Step 12 output directory `/kaggle/working/lifemate_qwen35_4b_thinking_lora_bm25_3ep_v1/`, download `best_adapter.zip` and `reports.zip`.

Upload both ZIPs unchanged to:

```
MyDrive/Zalo-AI-Challenge-2026/imports/teammate_topic_20261109/
    best_adapter.zip
    reports.zip
```

Keep the existing trained ASR at `MyDrive/Zalo-AI-Challenge-2026/ASR-cp/best.pt`. Alternatively reuse a complete existing public ASR JSONL, including transcripts for all user-history turns.

The adapter archive includes adapter weights, tokenizer assets, `run_config.json`, model revisions and package metadata. A loose `adapter_model.safetensors` is insufficient to reproduce its agent. The importer ignores bundled Python source. `reports.zip` supplies validation reference states/calls and split assignments; the dataset source/catalog hashes must match. Neither archive needs manual extraction. For the closest public comparison, obtain the exact ASR JSONL used for the reported 56.7 and upload it to `inputs/teammate_public_asr.jsonl`; choose transcript mode. The supplied planner notebook does not establish which ASR checkpoint or transcripts produced that submission. A full `resume_state.zip` is unnecessary for inference; keep it only if future full-state training resume is wanted.

## Why this differs from our compact agent

The supplied notebook uses rank-8/alpha-16 language LoRA on unquantized Qwen3.5-4B, learning the initial app router, later `open_apps`, actual tool calls, and completion. It retrieves topical dialogue segments through raw/compressed BM25 candidates and BGE reranking. LLMLingua text is used as a retrieval key; original dialogue supplies argument values. A separate frozen Qwen3 model segments long histories. This protocol and preparation differ from the earlier compact-all-schemas rank-16 notebook. Importing its adapter into that older pipeline would be a prompt/protocol mismatch.

Its saved held-out result is 73.87630662020906 on 82 gold-transcript samples, with zero aborted rollouts. It reports 0.97317 final gold-call app coverage and 0.77528 micro call recall. These show remaining planning/call errors even on clean text. The public cell has no recorded output; 56.7 is the user's reported public score. Clean held-out and public scores differ in data/transcripts and cannot be subtracted to measure ASR error.

## Agent changes to evaluate

`baseline` retains the teammate's original routing, retrieval and context budget. Its weights/revisions/configuration are imported rather than guessed. Hardware and library portability fixes mean this is a compatible reference, not a guarantee of bitwise reproduction of Kaggle outputs.

`enhanced` uses the same adapter, with 8192 total context, a 7168 executor prompt ceiling, and a 2048-token history budget. It augments contextual BM25 candidates with candidates matching the final request alone and reranks against that request. It preserves selected dialogue as complete original turns. Under prompt pressure it can remove schema descriptions and older history; it never replaces complete parameter schemas with names, truncates the current request, clips live tool results, or removes executed call/result pairs. If protected context still cannot fit, it records an early stop. The original baseline can perform all those truncations, including an emergency token splice, which can lose IDs, conditions or earlier progress. Avoiding those paths is a testable engineering improvement; whether it improves task accuracy remains to be measured.

The notebook evaluates baseline, enhanced context, and the separate repeated-read guard on the exact original validation IDs with the same selected adapter. Higher local score wins; ties choose baseline. It does not tune on test/public labels. Reusing the checkpoint-selection validation set is not a new unbiased test. No increased reasoning allowance or extra generic self-reflection call is added: the supplied held-out logs showed no reasoning-cap hits.

## Running and resuming

Push all new files and the notebook, then use L4/A100. Steps 1–3 import and configure the runtime, Step 4 compares validation, and Step 5 independently runs public inference with the chosen mode. For inference without comparison, omit the reports path and explicitly choose baseline or enhanced. Auto mode requires a completed comparison.

A new experiment has separate baseline/enhanced/read_guard outputs and a source/weight fingerprint. Keep the same code, paths and settings to resume. Completed simulated samples are reused atomically; incomplete samples restart from the initial simulator state. Model stages run in isolated subprocesses. Early stopped predictions are reported; only complete and schema-valid public submissions publish to the Drive root. No fabricated tool calls are inserted. Old training runs must remain pinned to their original Git commit for exact optimizer resume.

## Local verification and limits

Run the two suites separately because the topic runtime intentionally has its own flat module names:

```
python -m unittest discover -s training_pipeline -p 'selftest_*.py'
python -m unittest discover -s training_pipeline/topic_agent -p 'selftest_*.py'
```

The importer, model/config compatibility, thinking delimiters, routing, protected context, baseline tie-breaking, and notebook variables have CPU checks. Real Qwen3.5 tokenizer checks cover native thinking/action boundaries and dataset-shaped prompts. GPU execution and any score gain require the user's exported weights and a Colab runtime; neither is claimed as completed locally.

`topic_agent/provenance.json` records the supplied notebook hash and extracted module hashes. The copied inference modules and relevant CPU tests are adapted for isolated Colab execution, strict transcript merging, and optional protected context. Training entry points are not included or invoked.

## Follow-up repository review

See `notebooks/datyth_repo_review.md` for the pinned Datyth repository review. The updated notebook uses experiment name `topic_agent_compare_20261109_v2`. The additional `read_guard` variant preserves baseline prompts and blocks only exact repeated reads before any intervening non-read call. It does not block repeated writes or inspect hidden state. Three duplicate proposals stop the loop with a diagnostic.

Each validation run writes `validation_error_analysis.json` and `validation_error_cases.jsonl`; public inference writes observational `public_error_diagnostics.json` without correctness labels. `compare_validation(session, validation_transcripts=...)` optionally replaces all final/history user text with ASR transcripts for exactly the saved validation IDs. It checks the file hash and keeps the original references. Do not substitute the external repo's split for the already-trained teammate model's split.

## Full menu / lazy complete schemas (2026-11-10)

Use `notebooks/menu_agent_inference.ipynb` to keep the same adapter and compare the previous winning setting against `menu` and `menu_recovery`. The complete installed-app menu retains Vietnamese names and tool names; requested apps load unshortened parameter/return schemas into a reversible schema window. The recovery variant adds sample-clock dates, empty-read guidance and a single optional completion review. Read `notebooks/menu_agent_review.md` for the 62-point notebook analysis, exact differences and test limits. The new notebook defaults to reusing the prior public ASR JSONL after verifying its fingerprint.
