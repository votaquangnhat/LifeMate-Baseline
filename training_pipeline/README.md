# LifeMate training pipeline

Implementation for `notebooks/train_and_infer.ipynb`. The notebook clones this repository and runs these files; it does not generate Python source.

Before using Colab, commit and push this directory and the notebook to the branch/ref configured in the notebook. The default repository is `votaquangnhat/LifeMate-Baseline`, branch `main`. The notebook retains an existing checkout instead of automatically pulling into an active run, and records the Git commit in Drive. Use a commit SHA as `REPO_REF` to pin a run.

## Layout

- `colab_workflow.py`: host-side setup, dataset download, configuration, model export, ASR handoff and submission export. Importing it performs no installation or model loading.
- `requirements.txt`: isolated agent environment dependencies; the setup helper preserves Colab's CUDA PyTorch and records installed versions.
- `prepare_agent.py`, `context_rerank.py`: grouped splits, compact schemas/history, tokenization and reference preparation.
- `modeling_agent.py`, `train_agent.py`, `amp_health.py`: Qwen3 NF4 QLoRA, checkpointing, preflight and numerical-health checks.
- `run_pipeline.py`, `rollout_agent.py`, `agent_core.py`: stage orchestration, validation-based selection, tool execution and scoring.
- `prepare_public.py`, `asr_contract.py`, `export_whisper_asr.py`: public transcript validation and optional inference-only Whisper ASR.
- `selftest_*.py`: CPU contracts. Run with the agent environment's Python before GPU stages.

These are standalone scripts with imports resolved from their script directory. GPU stages run in fresh subprocesses so the reranker, ASR and planner do not retain GPU allocations between stages.

```bash
python -m unittest discover -s training_pipeline -p 'selftest_*.py'
python training_pipeline/run_pipeline.py --config /path/to/run_config.json --stage prepare
```

Other stages: `smoke`, `baseline`, `train`, `select`, `test`, `public`. In Colab, the notebook supplies the isolated interpreter, configuration and Drive paths.

## Inference after training

Step 13 is independent: in a fresh GPU session, run its single code cell directly. It mounts Drive, refreshes a separate repository checkout, and calls `infer_standalone.run`. Set `TRAINED_RUN_NAME` to the saved run containing `best_adapter`, `tokenizer` and `selection.json`. It restores inference artifacts into a separate `inference/<INFERENCE_NAME>/` directory without changing training files. It does not invoke preparation for SFT, training, preflight, or selection.

Default ASR is pretrained `vinai/PhoWhisper-small`, allowing inference while a teammate trains their ASR. This still transcribes every final/history user utterance. To use teammate transcripts later, switch to `ASR_MODE='transcripts'`, set `TRANSCRIPTS_PATH`, and use a new `INFERENCE_NAME`. The existing package-version and public execution-journal checks remain active.

## Training across days

First day: `TRAIN_START='fresh'`, `RUN_NAME='day1'`, `EPOCHS=2`.

Next day: `TRAIN_START='best'`, `PREVIOUS_RUN_NAME='day1'`, `RUN_NAME='day2'`, `EPOCHS=2`. This snapshots the prior best adapter and trains for two additional epochs with a fresh optimizer/scheduler. The prior best competes with the new checkpoints in validation selection.

An interruption within the same run uses `RESUME_CHECKPOINT='auto'` to restore the full latest complete checkpoint. Keep code, configuration, run name and epoch count fixed for that resume. Moving from the older self-contained notebook changes code paths/fingerprints: use a new run from its previous best instead of attempting an exact interrupted-checkpoint resume across the refactor.

The model recipe and ASR interface are unchanged by extraction. No ASR training is performed. Outputs remain under `MyDrive/Zalo-AI-Challenge-2026`. See `notebooks/train_and_infer_review.md` for reference results and limitations.
