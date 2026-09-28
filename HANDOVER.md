# edgefront — state of the project

Working notes for whoever picks this up next. Written 2026-09-22.

## What this is

A benchmark that answers one question: *for my decision task, do I need a hosted
typed-decision model, or does a small local model match it?* It runs the same
examples, label space and wording through every backend and reports accuracy,
calibration error, the latency distribution and cost per million calls, then
prints a verdict.

Repo: https://github.com/shivpratapsinghpanwar/edgefront · MIT · CI green on 3.10 + 3.12.

## What already works

- `bench` / `report` / `verify` / `tasks` / `merge` CLI, data-doctor house
  style, exit code is the verdict. `merge` combines result documents from
  different machines (e.g. local backends on Kaggle, hosted locally) into one
  document with a recomputed frontier and verdict.
- Backends: `rules` (keyword floor), `stub` (deterministic, CI), `hf[:model-id]`
  (torch zero-shot NLI), `onnx:<file>:<tokenizer>[:precision]`, `jev` (hosted).
- `quantize/`: HF → ONNX export and dynamic INT8. Measured on
  `typeform/distilbert-base-uncased-mnli`: **267.9 MB → 67.3 MB, 4.0× smaller**.
- 49 tests, all offline — no API key, no model, no network.
- `kaggle_job/`: standalone script, installs edgefront from GitHub, runs
  banking77 (77 labels) with `hf` on CUDA and `onnx` int8 on CPU. Pushed and
  run for real on Kaggle (`shivpratap0007/edgefront-banking77-local-backends`);
  result below.
- `cascade`: confidence-gated local→hosted router plus a free threshold
  sweep from one local pass + one hosted pass (`edgefront.cascade`,
  `edgefront.predictions`). `kaggle_job/cascade/` is the sibling Kaggle job
  that collects raw local predictions instead of an aggregate doc. Real
  banking77 result below - the cascade is not a win on this task.

## The measured result so far

60 examples of the bundled `synthetic` task, Windows CPU, from India:

| backend | acc | ECE | p50 ms | p99 ms | $/1M | offline |
|---|---:|---:|---:|---:|---:|:--:|
| rules | 0.450 | 0.099 | 0.013 | 0.368 | 0.0003 | yes |
| onnx fp32 | 0.383 | 0.165 | 481 | 3911 | 12.9 | yes |
| onnx int8 | 0.500 | 0.284 | 249 | 523 | 6.69 | yes |
| jev | 0.950 | 0.036 | 386 | 557 | 3.59 | no |

**The hosted model wins decisively** — 95% vs 50%, better calibrated, and
cheaper, because the local NLI approach scores one hypothesis per label and
burns 4× the passes. This is the opposite of the project's original hypothesis.
Keep it that way in the README: the finding is the product.

Three things worth carrying into the writeup:

1. **The vendor's published 70–500 ms does not hold from this network position.**
   Measured p50 386 ms, p99 557 ms, and p99 1254 ms on an earlier run. This is
   why `environment()` records host and UTC time in every result document.
2. **INT8 beat FP32 on accuracy** (0.500 vs 0.383) and was ~2× faster. On 60
   examples the accuracy difference is noise; the latency and size are real.
3. **Local cost scales with label count**, since it is one forward pass per
   label. At banking77's 77 labels it gets much worse, not better.

## Gotcha that cost an hour — do not regress it

`typeform/distilbert-base-uncased-mnli` orders its labels
**ENTAILMENT, NEUTRAL, CONTRADICTION** — entailment is index **0**, not the
conventional 2. The ONNX backend originally assumed `n_classes - 1` and was
scoring the *contradiction* logit, i.e. the opposite of the question. Accuracy
read 0.167 instead of 0.417 and looked plausible.

`_entailment_index()` in `backends/onnx_local.py` now reads it from the
checkpoint config, and is regression-tested (see item 1 below). It is the
single most dangerous class of bug in this project: silently wrong rather
than loudly wrong.

### Real cascade experiment: banking77 local→hosted (done)

`kaggle_job/cascade/run_banking77_cascade.py` ran on the same Kaggle T4,
loading banking77 with the same `n=300, split=test, seed=0` as the published
run, and wrote raw per-example predictions (`edgefront.predictions`) for both
torch/CUDA fp32 and ONNX int8/CPU instead of an aggregate doc. `jev` ran
locally (`edgefront bench --task banking77 --backends jev --limit 300
--save-predictions`) and its 300 uid/gold pairs were verified to line up
exactly with both Kaggle files before sweeping (`edgefront cascade
--local-predictions ... --hosted-predictions ...`, 22 thresholds, free).

**Verdict: the cascade is not a win here.** Local confidence is weakly
informative — bucketing each local backend's own predictions into quartiles
by its own confidence shows the top quartile is ~1.9-2.2x more accurate than
the bottom quartile (torch: 0.120→0.227; ONNX int8: 0.107→0.240) — but the
ceiling is low (even the most-confident quarter is only ~23-24% accurate) and
the model is rarely confident at all (top decile confidence tops out around
0.90-0.92). Getting within 3 points of jev's 0.790 needs escalating ~95% of
calls either way, at which point there is not enough local traffic left to
amortize the local pass's own cost:

- ONNX int8/CPU (slow local pass, 810ms p50): threshold=0.65 reaches 0.773
  (1.7-pt gap) at 94.7% escalation for **$76.1/1M — 33% more expensive than
  just calling jev** ($57.4/1M). The cascade is strictly worse than not
  bothering with the local pass at all.
- torch/CUDA (cheap local pass, $1.68/1M alone): threshold=0.70 reaches 0.767
  (2.3-pt gap) at 94.7% escalation for $56.1/1M, ~2% cheaper than jev — the
  best point found anywhere in either sweep, and still a marginal win.

Full sweeps: `docs/results/banking77_cascade_torch.json` and
`banking77_cascade_onnx_int8.json`; compact summary + confidence buckets in
`docs/results/banking77_cascade_summary.json`; charts in
`docs/charts/banking77_cascade_frontier*.png`. See README's "The real
experiment: banking77 local→hosted cascade" for the full writeup. The raw
per-example prediction files (~1MB each) were kept locally, not committed —
only the aggregate sweep documents and the summary are small enough to check
in.

## Next, in order

1. ~~**Regression test for the entailment index.**~~ Done —
   `test_entailment_index_resolved_from_checkpoint_config` and
   `test_entailment_index_returns_none_when_config_unreachable` in
   `tests/test_edgefront.py`, both faking `transformers` in `sys.modules` so
   they stay offline.
2. ~~**Commit and push the current working tree.**~~ Done, in stages; CI green
   on every push.
3. ~~**Update the README with the real table above.**~~ Done.
4. ~~**The Kaggle job**~~ (see below) — done, and it ran end to end: real
   banking77 numbers are below, not placeholders.
5. `measure/calibration.py` reliability curve already exists in the result JSON
   under `meta.reliability_curve` — render it in `report.py`. Still open.

### banking77 result (300 examples, test split, seed 0)

The Kaggle job (`kaggle_job/run_banking77.py`) ran clean on a T4 after two
real bugs surfaced and got fixed (see "Bugs found running the Kaggle job"
below). Hosted (`jev`) ran locally with the API key and was merged in with
`edgefront merge`:

| backend | acc | ECE | p50 ms | p99 ms | $/1M | offline |
|---|---:|---:|---:|---:|---:|:--:|
| rules | 0.293 | 0.218 | 0.037 | 0.074 | 0.001 | yes |
| distilbert-mnli-torch-cuda | 0.193 | 0.128 | 62.7 | 122 | 1.68 | yes |
| distilbert-mnli-onnx-int8-cpu | 0.167 | 0.126 | 774 | 1721 | 20.8 | yes |
| jev (hosted) | 0.790 | 0.108 | 391 | 556 | 57.4 | no |

**Hosted leads by 59.7 points here** - a bigger gap than the 45-point gap on
the 4-label synthetic task, exactly as predicted: local cost and latency
scale with label count. Two things worth noting that don't fit the synthetic
task's story:

- Both local backends land *below* the keyword baseline on banking77
  (0.193 and 0.167 vs rules' 0.293) - zero-shot NLI does badly on fine-grained
  intents phrased close together (e.g. `card_swallowed` vs
  `lost_or_stolen_card`), worse than matching plausible keywords.
- On $/1M calls, local is still cheaper than hosted here (torch-cuda at
  $1.68 vs jev's $57.4) - the local cost model amortises hardware you already
  paid for, so at 77 labels it still looks cheap in isolation. Cost isn't the
  argument against local here; the 59.7-point accuracy gap is.

### Bugs found running the Kaggle job (fixed)

- `PolyAI/banking77` ships a python loading script; `datasets>=3.0` refuses to
  run those and raises "Dataset scripts are no longer supported" on every
  load, not intermittently. Fixed by pointing `tasks/banking77.py` at
  `legacy-datasets/banking77`, HF's own parquet mirror of the same rows.
- `torch.onnx.export()` defaults to the "dynamo" exporter on the torch build
  Kaggle ships; it needs `onnxscript` and produced a graph that failed
  onnxruntime's shape inference during quantization. Fixed by passing
  `dynamo=False` explicitly in `quantize/export.py`.

## The Kaggle job

banking77 is 77 labels, so the local NLI backend needs 77 forward passes per
example: roughly **19 s per example on this laptop's CPU**, which is unusable.
A T4 makes it tractable.

- Kaggle auth already works here (`python -m kaggle kernels list --user
  shivpratap0007` succeeds; the refresh token is valid).
- There is a battle-tested runner to model it on at
  `C:\Users\hp\Synthetic_Data_Factory\kaggle_runner\` — preflight checks,
  push, poll with backoff, collect, and a compact machine-readable `result.json`.
  Treat it as a pattern, do not copy employer code.
- The job should: pip install edgefront from the public GitHub repo, load
  banking77 via `datasets`, run `hf` on CUDA plus `onnx` int8 on CPU, and write
  the result document to `/kaggle/working`.
- **Do not put the API key in a Kaggle kernel.** Run the hosted arm locally and
  merge the two result documents, or use a Kaggle Secret.

## Constraints that are not negotiable

- **The owner is employed.** No employer code, data or hardware in this repo.
  The `Synthetic_Data_Factory` and `surveillance_use_case` runners are
  conceptual precedent only — reimplement clean-room.
- **Commits are authored by the repo owner only.** No AI attribution trailers,
  no `Co-Authored-By`. This was asked for explicitly.
- **Honest reporting.** If a number is unflattering, publish it. The one thing
  that would kill this project is a benchmark caught flattering its own premise.
- The API key lives at `C:\Users\hp\Resumes\api_type.txt`, format `apikey_…`.
  Never commit it; never echo it.

## Known weaknesses to be honest about in the README

- The bundled `synthetic` task is generated from templates that share vocabulary
  with the label descriptions, so a keyword baseline does unusually well on it.
  It exists to exercise the pipeline offline in CI, not to rank models. Say so.
- The local contender is zero-shot NLI, chosen because the hosted model is also
  zero-shot. A task-specific fine-tune would score far higher — and that is a
  legitimate objection to answer in the README, not to hide.
- Question wording is an experimental variable. TypeSafe's own guidance says
  agents write poor questions and expect to refine them collaboratively; bad
  `criteria` wording would unfairly penalise the hosted model.
