"""Confidence-gated local -> hosted cascade, and a free cost-accuracy sweep.

The idea: call the free/fast local backend first. Only escalate to the
hosted backend when the local prediction's confidence is below a threshold.
Deploying this well needs exactly one number - the threshold - and choosing
it well means seeing the whole accuracy/latency/cost tradeoff it produces,
not one point on it.

Naively getting that tradeoff curve would mean calling both backends once per
threshold, so the hosted (paid, slow, rate-limited) call gets re-paid N times
for an N-point curve, and every point after the first is measured under
whatever the hosted network position happens to be at that moment - not a
fixed condition (see `bench.py`'s `environment()` for how seriously this
project already takes that). Instead: run the local backend once, run the
hosted backend once, and cache both raw prediction lists (see
`predictions.py`). Deciding which prediction "wins" for a given threshold is
then pure arithmetic on those two cached lists - no model calls, no network,
no extra cost - so `sweep_from_predictions` can produce as many threshold
points as wanted for the price of one local pass plus one hosted pass. As far
as this project's authors know, nothing in the Jev/typed-decision-model
ecosystem publishes this curve; every other result in this project (and, as
far as we've seen, everywhere else) reports one point per backend, not a
frontier over a routing knob.

The two passes do not need to happen in the same process, or even on the same
machine: `sweep_from_predictions` takes plain prediction lists, not `Backend`
objects, so a local pass collected on a GPU (`edgefront bench --save-
predictions`) and a hosted pass collected later, wherever the API key lives,
can be combined by loading both files back (`predictions.load_predictions`)
with no model and no network involved in the sweep itself. `sweep_thresholds`
is a thin same-process convenience wrapper around the same function, for
when a live cascade is enough (e.g. exercising this in CI with `rules` and
`stub`, both free and offline).
"""

from __future__ import annotations

from .bench import BenchConfig
from .measure.accuracy import expected_calibration_error, score
from .measure.cost import HOSTED_PRICES_USD_PER_MTOK, CostModel, cascade_cost
from .measure.latency import DEFAULT_WARMUP, percentile, summarise
from .types import Backend, BackendResult, Example, Prediction, TaskSpec


class CascadeError(ValueError):
    """A cascade sweep was asked to combine data that cannot be combined
    honestly - mismatched prediction counts, or two thresholds that would
    silently collapse into the same named result. Same shape and spirit as
    `merge.MergeError`.
    """


def _should_escalate(confidence: float | None, threshold: float) -> bool:
    """Whether a local prediction is not confident enough to keep.

    `confidence` can legitimately be `None` (e.g. `JevBackend`, when the SDK
    response doesn't carry one) - and `None` always escalates, on the same
    reasoning `measure.accuracy.expected_calibration_error` uses for the same
    field: you cannot gate a decision on a signal you don't have. Treating a
    missing signal as "confident enough" would silently under-escalate
    exactly the predictions this mechanism exists to catch.
    """
    return confidence is None or confidence < threshold


class CascadeBackend:
    """A `Backend` that tries `local` first and escalates to `hosted` below
    `threshold`.

    Satisfies the `Backend` protocol, so a single instance drops straight
    into `run_backend()` for a single-threshold, single-point measurement
    (e.g. inside `edgefront bench`). It is deliberately not the primary way
    to explore many thresholds: N instances at N thresholds would call the
    hosted backend N times over, which is exactly the repeated hosted cost
    `sweep_from_predictions` exists to avoid - use that (or `sweep_thresholds`)
    for a sweep.
    """

    def __init__(
        self,
        local: Backend,
        hosted: Backend,
        threshold: float,
        name: str | None = None,
    ) -> None:
        self.local = local
        self.hosted = hosted
        self.threshold = threshold
        self.name = name or f"cascade@{threshold:.2f}"
        # Counters accumulated as a side effect of predict(), read once by
        # meta() after every call completes - the same stateful-backend
        # pattern ONNXLocalBackend uses for its cached hypotheses.
        # bench.run_backend() only calls meta() after the full predictions
        # list is built, so this is a supported, established shape.
        self._n_calls = 0
        self._n_escalated = 0
        self._local_latencies: list[float] = []
        self._escalated_hosted_tokens = 0

    def predict(self, ex: Example, task: TaskSpec) -> Prediction:
        self._n_calls += 1
        local_pred = self.local.predict(ex, task)
        self._local_latencies.append(local_pred.latency_ms)
        if not _should_escalate(local_pred.confidence, self.threshold):
            return local_pred

        self._n_escalated += 1
        hosted_pred = self.hosted.predict(ex, task)
        self._escalated_hosted_tokens += hosted_pred.est_input_tokens or 0
        # Sequential, not parallel: local ran, THEN hosted ran because local
        # wasn't confident enough. Total latency for an escalated example is
        # the sum of both calls, not just the hosted call's own latency. This
        # happens inside one backend's predict() for one example - it is not
        # the inter-backend concurrency the README's "backends run
        # sequentially" rule is about (see README's Design rules).
        return Prediction(
            label=hosted_pred.label,
            latency_ms=local_pred.latency_ms + hosted_pred.latency_ms,
            probabilities=hosted_pred.probabilities,
            confidence=hosted_pred.confidence,
            est_input_tokens=hosted_pred.est_input_tokens,
            error=hosted_pred.error,
        )

    def meta(self) -> dict:
        escalation_rate = self._n_escalated / max(1, self._n_calls)
        local_sorted = sorted(self._local_latencies)
        return {
            "kind": "cascade",
            # Never offline: it can call the hosted backend. bench.py's cost
            # branch checks `kind == "cascade"` before the offline/hosted
            # binary specifically so this doesn't fall into the "offline"
            # cost path just because it *sometimes* stays local.
            "offline": False,
            "threshold": self.threshold,
            "escalation_rate": round(escalation_rate, 4),
            "local_p50_ms": percentile(local_sorted, 0.5) if local_sorted else 0.0,
            "avg_escalated_hosted_tokens": (
                self._escalated_hosted_tokens / self._n_escalated
                if self._n_escalated
                else 0.0
            ),
            "local_meta": self.local.meta(),
            "hosted_meta": self.hosted.meta(),
        }


def sweep_from_predictions(
    local_examples: list[Example],
    local_preds: list[Prediction],
    hosted_preds: list[Prediction],
    thresholds: list[float],
    cost_model: CostModel,
    hosted_price_per_mtok: float,
    warmup: int = DEFAULT_WARMUP,
) -> list[BackendResult]:
    """The cheap, novel part: turn two already-collected prediction lists into
    N threshold points, with no further model calls.

    Works identically whether `local_preds`/`hosted_preds` came from live
    backends in this process (see `sweep_thresholds`) or from
    `predictions.load_predictions()` reading files produced on two different
    machines at two different times - this function only ever does arithmetic
    on plain `Prediction` objects.
    """
    n = len(local_examples)
    if not (len(local_preds) == n and len(hosted_preds) == n):
        raise CascadeError(
            f"mismatched counts: {n} examples, {len(local_preds)} local "
            f"predictions, {len(hosted_preds)} hosted predictions - a "
            f"threshold sweep needs exactly one of each per example"
        )

    local_latencies = [p.latency_ms for p in local_preds if p.ok]
    local_p50 = float(summarise(local_latencies, warmup=warmup).get("p50", 0.0))
    hosted_tokens = [p.est_input_tokens or 0 for p in hosted_preds]
    avg_hosted_tokens = (
        sum(hosted_tokens) / len(hosted_tokens) if hosted_tokens else 0.0
    )

    seen_names: dict[str, float] = {}
    results: list[BackendResult] = []
    for threshold in thresholds:
        name = f"cascade@{threshold:.2f}"
        if name in seen_names:
            raise CascadeError(
                f"threshold {threshold} rounds to the same name '{name}' as "
                f"threshold {seen_names[name]} - refusing to silently "
                f"collapse two distinct measurements into one result"
            )
        seen_names[name] = threshold

        chosen: list[Prediction] = []
        n_escalated = 0
        for lp, hp in zip(local_preds, hosted_preds, strict=True):
            if _should_escalate(lp.confidence, threshold):
                n_escalated += 1
                # Same blend CascadeBackend.predict() computes live: hosted's
                # answer, but latency is local-then-hosted, sequentially.
                chosen.append(
                    Prediction(
                        label=hp.label,
                        latency_ms=lp.latency_ms + hp.latency_ms,
                        probabilities=hp.probabilities,
                        confidence=hp.confidence,
                        est_input_tokens=hp.est_input_tokens,
                        error=hp.error,
                    )
                )
            else:
                chosen.append(lp)

        escalation_rate = n_escalated / n if n else 0.0
        latencies = [p.latency_ms for p in chosen if p.ok]
        latency = summarise(latencies, warmup=warmup)
        accuracy, macro_f1 = score(chosen, local_examples)
        ece, curve = expected_calibration_error(chosen, local_examples)
        cost = cascade_cost(
            local_p50_ms=local_p50,
            cost_model=cost_model,
            hosted_avg_tokens_per_call=avg_hosted_tokens,
            hosted_price_per_mtok=hosted_price_per_mtok,
            escalation_rate=escalation_rate,
        )
        results.append(
            BackendResult(
                backend=name,
                meta={
                    "kind": "cascade",
                    "offline": False,
                    "threshold": threshold,
                    "escalation_rate": round(escalation_rate, 4),
                    "reliability_curve": curve,
                },
                predictions=chosen,
                accuracy=accuracy,
                macro_f1=macro_f1,
                ece=ece,
                latency=latency,
                cost=cost.to_dict(),
                n_errors=sum(1 for p in chosen if not p.ok),
            )
        )
    return results


def sweep_thresholds(
    local: Backend,
    hosted: Backend,
    task: TaskSpec,
    thresholds: list[float],
    config: BenchConfig,
) -> list[BackendResult]:
    """Same-process convenience wrapper around `sweep_from_predictions`.

    Still exactly one pass each - that part of the design does not change
    just because the backends happen to be live objects in this process
    rather than loaded from disk. Use `sweep_from_predictions` directly (with
    `predictions.load_predictions`) when the two passes ran on different
    machines at different times - that is the case this split exists for.
    """
    examples = task.examples[: config.limit] if config.limit else task.examples
    local_preds = [local.predict(ex, task) for ex in examples]
    hosted_preds = [hosted.predict(ex, task) for ex in examples]

    hosted_meta = hosted.meta()
    price = HOSTED_PRICES_USD_PER_MTOK.get(
        hosted_meta.get("vendor", ""), HOSTED_PRICES_USD_PER_MTOK["jev"]
    )
    return sweep_from_predictions(
        examples,
        local_preds,
        hosted_preds,
        thresholds,
        cost_model=config.cost_model,
        hosted_price_per_mtok=price,
        warmup=config.warmup,
    )
