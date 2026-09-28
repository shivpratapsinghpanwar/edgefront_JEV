# edgefront JEV

[![PyPI](https://img.shields.io/pypi/v/edgefront)](https://pypi.org/project/edgefront/)
[![CI](https://github.com/shivpratapsinghpanwar/edgefront_JEV/actions/workflows/ci.yml/badge.svg)](https://github.com/shivpratapsinghpanwar/edgefront_JEV/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Do you need a hosted decision model, or does a small local model match it?
Measure it on your own task instead of guessing.

```bash
pip install edgefront
edgefront bench --task synthetic --backends rules,stub
```

## Why

Hosted "typed decision" models return a label plus calibrated probabilities in
70–500 ms for a fraction of an LLM's price. A quantized classifier on your own
hardware returns the same shape of answer in single-digit milliseconds for
nothing, and works offline.

Which one you should deploy depends on your task, your latency budget and your
volume. This measures all three and prints a verdict.

## What it does

Runs the same examples, the same label space and the same wording through every
backend, then reports accuracy, calibration error, the latency distribution and
cost per million calls.

## Tasks

A task is a label space, an instruction string, and a list of labelled examples
— see `edgefront.types.TaskSpec`. Two ship with the package:

| task | labels | examples | source | what it's for |
|---|---:|---:|---|---|
| `synthetic` | 4 | up to 200, generated | rule-based templates, no download | exercises the whole pipeline offline in CI — **not a model discriminator**, see Known weaknesses |
| `banking77` | 77 | up to ~3,000, `test` split | [`legacy-datasets/banking77`](https://huggingface.co/datasets/legacy-datasets/banking77) (HF parquet mirror of PolyAI's banking-intent set) | a real, hard, fine-grained intent-classification benchmark |

`edgefront tasks` lists both from the CLI. Add your own by writing a
`TaskSpec` loader — `src/edgefront/tasks/synthetic.py` is the template, and
`src/edgefront/tasks/banking77.py` shows how to wrap a HuggingFace dataset.
**Your own labelled data is what actually answers the question this tool
asks** — the two bundled tasks exist so the pipeline has something to run
against out of the box, not as a substitute for benchmarking your task.

## Models compared

Every result below asks the identical zero-shot question, worded identically,
of four backends:

| backend | what it is | how it decides |
|---|---|---|
| `rules` | no model — keyword-overlap baseline | counts words shared with each label's description |
| local, fp32 | [`typeform/distilbert-base-uncased-mnli`](https://huggingface.co/typeform/distilbert-base-uncased-mnli), 67M params, run via torch or ONNX Runtime at full precision | zero-shot NLI: scores `"This text is about {label}."` as an entailment hypothesis per label, softmaxes the entailment logits |
| local, int8 | the same checkpoint, dynamically quantized to INT8 via `onnxruntime.quantization` | identical method, quantized weights |
| `jev` (hosted) | TypeSafe's `jev-latest` | a `Choice` question over the label set, called through `typesafe-sdk` |

The local backend is zero-shot on purpose, not fine-tuned — see **Known
weaknesses** for why, and what a fine-tuned classifier would change.

## Cascade: confidence-gated local→hosted routing

Everything above measures each backend in isolation — one point per backend.
`edgefront cascade` measures something different: **try the local model
first, and only call the hosted model when the local prediction's own
confidence is below a threshold.** Pick the threshold well and you get most
of the hosted model's accuracy while paying the hosted price on only the
fraction of calls that actually needed it.

The threshold is exactly one number, and choosing it well means seeing the
whole accuracy/latency/cost tradeoff it produces, not one point on it. Doing
that naively would mean calling the hosted backend once per threshold — for a
10-point curve, that is 10x the hosted cost and 10x the wait, and every point
after the first is measured under whatever the hosted network position
happens to be at that moment (see "Three things worth carrying forward
honestly" above). Instead, `edgefront cascade`:

1. runs the local backend once over every example,
2. runs the hosted backend once over every example,
3. caches both raw prediction lists, then
4. for each threshold, decides per example — in pure Python, no more model
   calls — whether that example's local confidence would have cleared the
   bar, and blends accuracy/ECE/latency/cost accordingly.

So a whole cost-accuracy frontier costs exactly one local pass plus one
hosted pass, however many thresholds you sweep. As far as we know, nothing in
the Jev/typed-decision-model ecosystem publishes this curve — everything else
here (and everywhere else we've seen) reports one point per backend, not a
frontier over a routing knob. A prediction with no confidence signal at all
(this can happen with `jev` — see `Prediction.confidence`) always escalates:
you cannot gate a decision on a signal you don't have.

The two passes don't need to happen together, either. `edgefront bench
--save-predictions DIR` writes each backend's raw per-example predictions to
disk; `edgefront cascade --local-predictions ... --hosted-predictions ...`
sweeps two such files with no model calls, no network, and no optional
dependency at all — so the local pass can run on a GPU (Kaggle, say) and the
hosted pass can run later, wherever the API key lives, exactly like `edgefront
merge` does one level up for aggregate results.

```bash
# live: local and hosted backends run once each, in this process
edgefront cascade --task synthetic --local rules --hosted stub \
  --thresholds 0.1,0.26,0.7,1.01 --plot cascade.png

# cross-machine: sweep two prediction files with no model and no network
edgefront bench --backends rules --save-predictions preds/       # e.g. on a GPU box
edgefront bench --backends stub  --save-predictions preds/       # e.g. wherever the API key lives
edgefront cascade --local-predictions preds/rules.predictions.json \
                   --hosted-predictions preds/stub.predictions.json \
                   --thresholds 0.1,0.26,0.7,1.01
```

**Mechanism demo on the toy task — not the real experiment.** `stub` is a
free, deterministic fake (see `backends/stub.py`), not a real hosted model, so
this is a demonstration that the routing and the free-sweep arithmetic work,
not a finding about `rules` vs. hosted. Run for real, on `synthetic`, with
`--local rules --hosted stub`:

![Accuracy vs. latency for a local→hosted cascade on the synthetic task: pure local, pure hosted, and the swept threshold curve between them.](docs/charts/cascade_mechanism_demo.png)

| backend | acc | ECE | p50 ms | p99 ms | $/1M | offline |
|---|---:|---:|---:|---:|---:|:--:|
| cascade@0.10 | 0.445 | 0.140 | 0.016 | 0.035 | 4.00e-04 | no |
| cascade@0.26 | 0.825 | 0.144 | 0.022 | 0.042 | 0.266 | no |
| cascade@0.70 | 0.820 | 0.130 | 0.023 | 0.042 | 0.289 | no |
| cascade@1.01 | 0.755 | 0.127 | 0.026 | 0.051 | 0.462 | no |

At `threshold=0.10` almost nothing escalates (this is essentially `rules`
alone, 0.445); at `threshold=1.01` everything escalates (essentially `stub`
alone, 0.755). The interesting point is `threshold=0.26`: `rules` on this
task is either very confident (1.0, when it is usually right) or unconfident
(0.25, on genuinely ambiguous tickets) — see `RulesBackend`'s flat-distribution
fallback — so escalating only the unconfident quarter of calls reaches 0.825,
*above either backend alone*, for a fraction of `stub`'s cost. `stub` is a
free, deterministic fake, though, so this only demonstrates that the
mechanism and the free-sweep arithmetic work — it says nothing about whether
a real hosted model behaves the same way. That pattern does **not** hold on
the real banking77/Jev cascade below; the mechanism demo's own optimism is
part of why that distinction matters. This does not contradict "backends run
sequentially" in Design rules below — that rule is about *different* backends
never running concurrently against each other; a cascade's local-then-hosted
calls happen one after another inside a single backend's own `predict()`, for
one example at a time.

### The real experiment: banking77 local→hosted cascade

Both local backends from the banking77 comparison above score *below* the
keyword baseline in isolation (0.193 torch/CUDA, 0.167 ONNX int8/CPU, vs
0.293 for `rules`), so the honest question going in was not "how good is the
cascade" but "does the local model's confidence carry any signal at all worth
gating on." It does, weakly — and that weak signal is most of the story.

**Is local confidence informative?** Bucket each local backend's own 300
predictions into quartiles by its own reported confidence and look at
accuracy within each bucket (computed directly from the raw predictions,
`docs/results/banking77_cascade_summary.json`):

| quartile (by local confidence) | torch/CUDA fp32: mean conf → acc | ONNX int8/CPU: mean conf → acc |
|---|---:|---:|
| Q1 (least confident) | 0.132 → 0.120 | 0.100 → 0.107 |
| Q2 | 0.239 → 0.213 | 0.207 → 0.133 |
| Q3 | 0.335 → 0.213 | 0.300 → 0.187 |
| Q4 (most confident) | 0.576 → 0.227 | 0.533 → 0.240 |

Confidence is directionally informative — the most-confident quarter of local
predictions is genuinely more accurate than the least-confident quarter
(roughly 1.9x for torch, 2.2x for ONNX int8) — but the ceiling is low: even
the *most* confident local predictions are only ~23-24% accurate, and the
model is rarely confident at all (its top decile of confidence tops out
around 0.90-0.92, not the 0.99+ a well-calibrated 77-way classifier would hit
on its easy cases). A threshold has almost nothing above ~0.4 to work with.

![Accuracy vs. cost for a local→hosted cascade on banking77: ONNX int8/CPU local model escalating to Jev, swept over 22 thresholds from one local pass and one hosted pass.](docs/charts/banking77_cascade_frontier.png)

| threshold | % escalated | accuracy | p50 ms | $/1M |
|---:|---:|---:|---:|---:|
| 0.00 (never) | 0.0% | 0.167 | 810 | 21.8 |
| 0.20 | 34.7% | 0.410 | 924 | 41.7 |
| 0.40 | 80.3% | 0.687 | 1174 | 67.9 |
| 0.60 | 92.0% | 0.753 | 1218 | 74.5 |
| **0.65** | **94.7%** | **0.773** | 1230 | **76.1** |
| 0.80 | 99.0% | 0.787 | 1240 | 78.6 |
| 1.01 (always) | 100.0% | 0.790 | 1241 | 79.1 |
| *pure jev (reference)* | *n/a* | *0.790* | *391* | *57.4* |

`threshold=0.65` is the first point within 3 accuracy points of pure `jev`
(0.773 vs 0.790, a 1.7-pt gap) — but it gets there by escalating **94.7%** of
calls, and the blended cost, **$76.1/1M, is 33% more expensive than just
calling jev on every request ($57.4/1M)**. The ONNX int8/CPU local pass is
slow (810 ms p50) enough that its own amortized cost is not negligible, so
paying for it on every call *and* paying jev on 95% of calls is a strictly
worse deal than skipping the local pass entirely.

The faster torch/CUDA arm tells a less bad but still unflattering story
(`docs/charts/banking77_cascade_frontier_torch.png`,
`docs/results/banking77_cascade_torch.json`): its local pass is cheap enough
($1.68/1M in isolation) that the blended cost can edge *below* pure jev —
`threshold=0.70` reaches 0.767 (a 2.3-pt gap) at 94.7% escalation for
$56.1/1M, about 2% cheaper than jev's $57.4/1M. That is the best this cascade
does anywhere in the sweep: a ~2% cost saving for a ~2-point accuracy
haircut, achieved by keeping only 5.3% of calls local.

**Honest verdict: on banking77, this cascade is not a win.** The mechanism
(free multi-point sweep from one local pass + one hosted pass) works exactly
as designed — see the numbers above and the full sweep in
`docs/results/banking77_cascade_*.json` — but the local model's confidence,
while weakly informative, never gets high or reliable enough on enough
examples to route a useful fraction of traffic away from the hosted model
without a real accuracy cost. Once escalation exceeds ~90% (needed for
anything close to jev's accuracy), there just isn't enough local traffic left
to amortize the local pass's own cost, and for the slower ONNX int8/CPU arm
that inversion makes the cascade actively *more* expensive than calling jev
outright. A cascade is a good idea when the local model is a *decent but
imperfect* answer that's confidently right most of the time; here the local
model is mostly wrong (rules baseline beats it) and only weakly aware of
when, which is close to the worst case for this routing strategy. This
matches the project's running theme on banking77: the local zero-shot NLI
approach simply does not work well at 77 labels, and gating on its confidence
doesn't rescue that.

## Results

### `synthetic` — 4 labels, 60 examples, Windows CPU, called from India

![Accuracy vs. latency frontier for the synthetic task: keyword baseline, local NLI model at fp32 and int8, and the hosted Jev model.](docs/charts/synthetic_frontier.png)

| backend | acc | ECE | p50 ms | p99 ms | $/1M | offline |
|---|---:|---:|---:|---:|---:|:--:|
| rules | 0.450 | 0.099 | 0.013 | 0.368 | 0.0003 | yes |
| distilbert-mnli, ONNX fp32 | 0.383 | 0.165 | 481 | 3911 | 12.9 | yes |
| distilbert-mnli, ONNX int8 | 0.500 | 0.284 | 249 | 523 | 6.69 | yes |
| jev-latest (hosted) | 0.950 | 0.036 | 386 | 557 | 3.59 | no |

**The hosted model wins, decisively — by 45 points — and it is also better
calibrated and cheaper.** This is the opposite of the project's original
hypothesis, and it is the finding, not an embarrassment to bury: the local
zero-shot NLI approach scores one entailment hypothesis per label, so it pays
for a full forward pass *per label, per example*. On this 4-label task that is
already 4x the inference cost of one hosted call; the hosted model answers all
labels in a single request. Quantizing the local model to INT8 clawed back
some of that but did not close a 45-point accuracy gap, and paying
per-forward-pass cost on a worse model is not a trade worth making.

### `banking77` — 77 labels, 300 examples, torch/CUDA on a Kaggle T4 + hosted locally

![Accuracy vs. latency frontier for banking77: keyword baseline, local NLI model on torch/CUDA and ONNX int8/CPU, and the hosted Jev model.](docs/charts/banking77_frontier.png)

| backend | acc | ECE | p50 ms | p99 ms | $/1M | offline |
|---|---:|---:|---:|---:|---:|:--:|
| rules | 0.293 | 0.218 | 0.037 | 0.074 | 0.001 | yes |
| distilbert-mnli, torch/CUDA fp32 | 0.193 | 0.128 | 62.7 | 122 | 1.68 | yes |
| distilbert-mnli, ONNX int8/CPU | 0.167 | 0.126 | 774 | 1721 | 20.8 | yes |
| jev-latest (hosted) | 0.790 | 0.108 | 391 | 556 | 57.4 | no |

**Hosted leads by 59.7 points here — a bigger gap than on the toy task, exactly
as the label-count theory predicts.** The uncomfortable part, reported as
measured: **both local backends score below the keyword baseline.**
Zero-shot NLI does badly telling apart fine-grained, closely worded intents
(`card_swallowed` vs. `lost_or_stolen_card`) — worse, on this run, than
matching plausible keywords. On cost per call local is still cheaper in
isolation (the local cost model amortises hardware you already own), but a
59.7-point accuracy gap is the actual argument against it here, not price.
Produced by `kaggle_job/run_banking77.py` (real Kaggle T4 run) merged with a
local `jev` run via `edgefront merge` — see `kaggle_job/README.md`.

### Three things worth carrying forward honestly

1. **The vendor's published 70-500 ms latency does not hold from this network
   position.** Measured p50 386 ms on the toy task, p99 557 ms (p99 hit
   1254 ms on an earlier run). Hosted latency is a function of where you call
   it from, not just the model — `environment()` records host and UTC time in
   every result document so this doesn't get generalized past the network it
   was measured on.
2. **INT8 beat FP32 on speed and size every time it was measured**, and
   matched or beat it on accuracy on the toy task (0.500 vs 0.383 — on 60
   examples that gap is noise, but the ~2x latency improvement and the 4.0x
   smaller checkpoint are real and repeatable: 267.9 MB → 67.3 MB). There was
   no accuracy tax for quantizing dynamically, at least here.
3. **Local cost and accuracy both scale badly with label count.** The local
   backend is one forward pass per label; the hosted backend is not. Going
   from 4 labels to 77, local p50 latency went from milliseconds to
   hundreds-to-thousands of milliseconds, and local accuracy fell to *below*
   the keyword baseline. This is why the banking77 run needed a GPU at all
   (roughly 19 s/example on a laptop CPU at 77 labels) — see `kaggle_job/`.

Both result documents (`bench_full.json`, and the merged banking77 run) and
their generated `BENCH.md`-style reports are what the tables above are
transcribed from — nothing here is hand-typed and then left to drift; see
`docs/make_charts.py` for exactly how the charts are drawn from those numbers.

## Design rules

These are the parts that decide whether a benchmark is worth reading:

- **A keyword baseline runs by default.** If a frontier model is only a few
  points above keyword matching, that is the most important number on the page,
  and a benchmark without a floor hides it.
- **Latency is a distribution, never a mean.** p50/p90/p99/max. A mean hides the
  tail, and the tail is what breaks a real-time budget.
- **The first 3 calls per backend are discarded** as warmup.
- **Backends run sequentially** so they cannot contend for CPU and corrupt each
  other's timings.
- **A failed call counts as wrong, not as missing.** A backend that errors on
  10% of inputs is not accurate on the rest.
- **Cost assumptions are printed next to the cost.** Local cost is amortised
  hardware plus power, which is arguable, so the arithmetic is shown.
- **The comparison is zero-shot on both sides.** A hosted model has not seen
  your labels, so the local contender is scored zero-shot too via NLI
  entailment. A fine-tuned classifier would win while answering a different
  question.

## Known weaknesses

Said plainly, because a benchmark that hides its own weak points is worse than
no benchmark:

- **The bundled `synthetic` task is generated from templates that share
  vocabulary with the label descriptions**, so a keyword baseline does
  unusually well on it (0.450 accuracy above, competitive with the local
  models). It exists to exercise the pipeline offline in CI, not to rank
  models — it is not a good discriminator, and the numbers above should not be
  read as "keyword matching nearly beats a neural model in general."
- **The local contender is zero-shot NLI, not a fine-tuned classifier**, chosen
  because the hosted model is also zero-shot and the comparison has to hold
  that variable constant. A task-specific fine-tune of the same small model
  would very likely score far higher than either backend above. That is a
  legitimate objection to this benchmark's framing, not one to hide: if you can
  afford to fine-tune on your own labels, do that comparison instead — this
  tool answers "zero-shot local vs. zero-shot hosted," not "the best local
  model you could build vs. hosted."
- **Question wording is an experimental variable, not a constant.** The
  hypothesis template (`"This text is about {}."`) and the task's `criteria`
  wording both affect both backends, but not necessarily equally — TypeSafe's
  own guidance says agents write poor questions on the first pass and expect to
  refine them collaboratively. Bad wording here would unfairly penalize the
  hosted model, which is one more reason to rerun this on your own task and
  wording rather than trust the numbers above at face value.

## Usage

### From install to your own verdict

```bash
pip install 'edgefront[local,tasks]'

# 1. Check the pipeline works - no key, no model, no network, a few seconds.
edgefront bench --task synthetic --backends rules,stub

# 2. Export a local model to ONNX fp32 + dynamic INT8 (downloads ~270 MB once).
#    It prints the exact --backends string to use next.
edgefront export --out models

# 3. Benchmark the local models against the keyword baseline.
edgefront bench --task banking77 --limit 100 --backends \
  "rules,onnx:models/model.int8.onnx:typeform/distilbert-base-uncased-mnli:int8" \
  --json local.json

# 4. Add the hosted model (needs a key) and let edgefront give the verdict.
pip install 'edgefront[jev]'
export TYPESAFE_API_KEY=...        # PowerShell: $env:TYPESAFE_API_KEY="..."
edgefront bench --task banking77 --limit 100 --backends \
  "rules,jev,onnx:models/model.int8.onnx:typeform/distilbert-base-uncased-mnli:int8" \
  --json bench.json --md BENCH.md
```

### Every command

```bash
edgefront tasks                     # list decision tasks
edgefront bench --task synthetic --backends rules,stub --json out.json
edgefront report out.json --out BENCH.md
edgefront verify out.json --min-acc 0.85 --max-p99 50
edgefront merge run_a.json run_b.json --json merged.json  # combine runs from different machines
edgefront cascade --task synthetic --local rules --hosted stub \
  --thresholds 0.1,0.3,0.5,0.7,0.9 --json cascade.json --plot cascade.png
edgefront export --model <hf-nli-model-id> --out models  # HF -> ONNX fp32 + INT8
```

`verify` exits 1 when a threshold is missed, so it drops straight into CI.
`merge` combines `bench` result documents produced on different machines into
one document with a single recomputed frontier and verdict — for example, a
local backend benchmarked on a GPU where there is no hosted-model API key,
plus a hosted backend benchmarked wherever that key lives (see
`kaggle_job/`). It refuses to merge documents that ran a different number of
examples, or that both contain the same backend name, since either would make
the resulting gap meaningless rather than just imprecise.

`cascade` sweeps a confidence threshold across a local→hosted cascade — see
**Cascade** above. `--local`/`--hosted` run live backends once each;
`--local-predictions`/`--hosted-predictions` instead sweep two files written
by `bench --save-predictions DIR`, with no model calls, no network, and (in
that mode) no optional dependency at all. `--plot PATH` additionally writes a
cost-accuracy frontier chart (needs `edgefront[plot]`).

### Backends

| token | what it is | needs |
|---|---|---|
| `rules` | keyword overlap baseline | nothing |
| `stub` | deterministic fake, for tests | nothing |
| `hf[:model-id]` | local zero-shot NLI classifier, torch | `edgefront[local]` |
| `onnx:<file>:<tokenizer>[:precision]` | local zero-shot NLI, ONNX Runtime (fp32/int8) | `edgefront[local]` |
| `jev` | TypeSafe Jev, hosted | `edgefront[jev]` + `TYPESAFE_API_KEY` |

`hf` and `onnx` default to
[`typeform/distilbert-base-uncased-mnli`](https://huggingface.co/typeform/distilbert-base-uncased-mnli)
if no model id is given — the checkpoint used for every local result above —
but take any HuggingFace sequence-classification (NLI) checkpoint. `jev`
defaults to TypeSafe's `jev-latest`. See **Models compared** above for what
each one actually is.

`edgefront export` (or `edgefront.quantize.export_onnx` / `quantize_int8` from
Python) turns any HF sequence-classification checkpoint into the
`onnx:model.onnx:tokenizer-id` and `onnx:model.int8.onnx:tokenizer-id:int8`
backends above. Absolute Windows paths (`onnx:C:\models\model.onnx:...`) work.

Adding a backend means implementing two methods — `predict()` and `meta()`.
See `src/edgefront/backends/stub.py`; it is the whole contract.

`edgefront.cascade.CascadeBackend` is itself a `Backend` — wrapping a local
and a hosted one behind a single confidence threshold — but it takes two
already-built `Backend` objects in Python, not a `--backends` token: nesting
two full backend specs inside one flat colon-delimited spec string is
genuinely ambiguous with the convention above, so the cascade has its own
subcommand (see **Cascade**) instead of extending this table's grammar.

### Install

```bash
pip install edgefront            # core: stdlib only, runs bench/report/verify/merge/cascade
pip install 'edgefront[local]'   # local models (torch, onnxruntime, transformers)
pip install 'edgefront[jev]'     # hosted backend (typesafe-sdk)
pip install 'edgefront[tasks]'   # banking77 and other HF-dataset-backed tasks
pip install 'edgefront[plot]'    # regenerate docs/charts/*.png, or `cascade --plot`
pip install 'edgefront[all]'
```

The core install has no dependencies, so `report`, `verify`, `merge` and
`cascade` (including its cross-machine `--local-predictions`/
`--hosted-predictions` mode) run anywhere, including in CI with no model and
no network. Only `cascade --plot` needs `edgefront[plot]`.

`cascade` and `export` arrive in 0.1.1. If `pip install edgefront` gives you
0.1.0, install from GitHub instead:
`pip install "edgefront[all] @ git+https://github.com/shivpratapsinghpanwar/edgefront_JEV"`.

### When something goes wrong

| you see | what it means |
|---|---|
| `TYPESAFE_API_KEY is not set` | hosted backend needs a key; use `--backends rules,stub` to test offline |
| `typesafe-sdk is not installed` | `pip install 'edgefront[jev]'` |
| `local backends need extra deps` | `pip install 'edgefront[local]'` |
| `banking77 needs the datasets package` | `pip install 'edgefront[tasks]'` |
| `matplotlib is required for --plot` | `pip install 'edgefront[plot]'` |
| `unknown backend: ...` | check the token against the Backends table below |
| local model scores below `rules` | not a bug - zero-shot NLI is weak on many-label tasks; see Results |

Exit codes: `0` ok, `1` a `verify` threshold failed, `2` bad usage or missing
dependency. Anything else, a traceback, or a number that looks wrong is a bug:
please [open an issue](https://github.com/shivpratapsinghpanwar/edgefront_JEV/issues)
with the command, the output, and `pip show edgefront`.

## Status

The core question now has a published answer on two tasks (see Results
above): hosted beats local zero-shot NLI by 45 points on a toy 4-label task
and by 59.7 points on the real 77-label banking77 benchmark, and is better
calibrated on both. All backends work — `rules`, `stub`, `hf`, `onnx` (fp32
and int8), `jev` — along with HF→ONNX→INT8 export (`edgefront.quantize`),
merging multi-machine runs (`edgefront merge`), and a Kaggle job for
label-heavy tasks that don't fit on a laptop CPU (`kaggle_job/`).

New in this version: `edgefront cascade`, a confidence-gated local→hosted
router with a free cost-accuracy threshold sweep (see **Cascade** above), and
`edgefront bench --save-predictions` / `cascade --local-predictions
--hosted-predictions` so the local and hosted passes can run on different
machines. The real experiment now has a published answer too: on banking77,
gating on the local model's confidence is not a win (see **The real
experiment: banking77 local→hosted cascade** above) — confidence is weakly
informative (top-quartile local predictions are ~2x more accurate than
bottom-quartile) but never reliably high enough that a threshold can route
away from `jev` without giving up several accuracy points, and for the slower
ONNX int8/CPU local arm the blended cost ends up *higher* than just calling
`jev` on every request. The torch/CUDA arm does edge out pure `jev` on cost
(~2%) at a ~2-point accuracy cost, which is the best case found in the sweep,
not a representative one.

Also next: render the per-bucket calibration reliability curve that's already
in every result JSON (`meta.reliability_curve`) into the markdown report, and
a third task with a different shape (not intent classification) to check the
label-count finding isn't an artifact of NLI specifically.

## License

MIT
