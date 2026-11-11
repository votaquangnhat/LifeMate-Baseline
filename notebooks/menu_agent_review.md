# Menu-agent revision — 2026-11-10

## What the supplied 62-point notebook actually runs

Inspected the embedded source dictionary in `lifemate-public-test-a-asr-itn-c-a-m-nh-ag.ipynb`, without executing its setup or inference cells. Its `aio_hybrid.py` explicitly loads component `llm-lora-p1` from `hugebenevolence/lifemate-final` at revision `02080970ccd59a2e1aeec4fd66a17640da2cddf1`. `rollout_aio.py` uses non-thinking E1P with all installed tool schemas supplied to the model, plus duplicate/retry controls. Its ASR path is Gipformer v4 followed by ITN. The older topical run export is used to recover ASR/ITN settings; it is not the planner adapter actually loaded by this crossover notebook.

This shares the Qwen3.5-4B base family with our model, but it is not a controlled agent-only comparison against our imported topical Thinking LoRA. The score of 62 is user-reported; no attempt was made to reproduce it locally. The user chose to keep their existing adapter, so no AIO checkpoint, native-tool protocol, ASR retraining, ITN preprocessing or hidden-state context was adopted.

## Why change our existing lazy loader?

Our previous topical agent already had an app router and `open_apps`. The senior's suggestion is not wholly new to that pipeline. Specific gaps remained:

- app metadata such as Vietnamese names and app-level descriptions was discarded;
- lazy schemas omitted return descriptions and shortened field descriptions;
- the original fitting policy could reduce loaded schemas to tool names;
- opening more apps grew the schema context without a reversible memory window;
- a premature `done` could leave an unselected app unused;
- coverage reports did not clearly separate initial omissions from apps never opened by the end.

The quoted 76/313 is a lead to investigate, not a measured failure count for this adapter. A lazy router should normally omit apps that are only needed later.

## New variants

`menu` keeps all installed app identities and tool names in a compact menu. App-level Vietnamese names and descriptions are retained. It loads complete schemas only for selected/requested apps, including argument constraints, parameter descriptions, and return descriptions. Under pressure, it unloads inactive app schemas instead of slicing them; unloaded apps remain in the menu and can be reopened. A proposed call whose schema is not currently loaded is not executed: the schema is loaded and the model must regenerate the call. A malformed router response falls back to a menu with system tools only, not a flood of every tool schema.

`menu_recovery` adds three bounded interventions and the existing conservative read guard:

- a date-reference table derived from the sample's current time and timezone, never the host clock;
- guidance after genuinely empty read results, without automatically changing arguments or blocking unrelated searches of the same tool;
- at most one completion review when some apps remain unopened, explicitly permitting `done` for completed/unsupported requests;
- exact repeated-read caching until a non-read call, without blanket write deduplication or inspecting hidden state.

Both retain the adapter's apps/open_apps/action/done JSON protocol and thinking budgets. No weights are updated. They use an 8192-token context and a 1536-token history budget; complete loaded schemas, current request and live tool observations are protected. An impossible protected prompt stops with an explicit diagnostic rather than executing a call from a damaged schema.

## Comparison and output

The new Git-backed `notebooks/menu_agent_inference.ipynb` imports the same existing ZIPs. It verifies the previous validation winner's adapter hash and uses that setting as the first/reference candidate. The candidates are reference, menu, and menu_recovery. Local validation score chooses the winner; ties retain the reference. A small validation improvement is not a guaranteed public improvement.

By default it resolves the exact previous public ASR JSONL and checks it against the saved preparation fingerprint. This keeps the recognition fixed for an agent-only public comparison. The user can explicitly select another transcript path or Gipformer decoding, but should then treat the score as a changed end-to-end pipeline.

`validation_app_coverage.json` reports required apps never opened, apps opened but unused, and full app coverage with other remaining errors. For menu variants, an app enters the ever-opened set only when its full schemas are placed in a generation prompt. Labels are used only after rollout to calculate this report. Initial routing coverage is not confused with final coverage. Public diagnostics do not infer correctness without hidden labels.

## Verification

- CPU regressions cover complete menus, wrong initial routing, schema loading before execution, full-schema eviction/reloading, malformed-router fallback, one-time completion review, date rollover/timezones, same-adapter reference selection and prior-ASR hash validation.
- Real Qwen3.5 tokenizer checks exercised all 1,545 local training examples in both menu modes. Initial router/executor preparation had no overflow; peak router prompt was 2,700 tokens and peak executor prompt was 7,166. Nine stress cases deferred some requested schemas instead of clipping them. History ranking was stubbed for this formatting/capacity check; it is not a model-accuracy evaluation.
- No GPU model inference or leaderboard scoring was run locally. Runtime, memory, recovery quality and score gains must be measured in Colab with the user's exported adapter.

Original notebooks are preserved. New experiment name: `menu_agent_20261110_v1`. Existing model/ASR downloads remain sufficient. Keep old training resumes pinned to their original Git commit.

Supplied notebook SHA-256: `205ad401c30f347a0fc93d689ce914d0c6ae16dc7ec8d2861ad050db5eb40a06`.
