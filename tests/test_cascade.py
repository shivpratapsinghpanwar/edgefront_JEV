"""Cascade tests: confidence-gated escalation, the free threshold sweep, and
the raw-predictions round trip that makes the sweep work across machines.

Offline throughout - no API key, no model download, no network.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from edgefront.backends.rules import RulesBackend
from edgefront.backends.stub import StubBackend
from edgefront.bench import BenchConfig, run_backend
from edgefront.cascade import (
    CascadeBackend,
    CascadeError,
    sweep_from_predictions,
    sweep_thresholds,
)
from edgefront.frontier import pareto
from edgefront.measure.cost import CostModel, cascade_cost, hosted_cost, local_cost
from edgefront.predictions import load_predictions, save_predictions
from edgefront.tasks import load_task
from edgefront.types import Example, Prediction, TaskSpec


def _ex(uid: str, gold: str) -> Example:
    return Example(uid=uid, text="t", gold=gold)


class _FixedBackend:
    """A backend with a hand-picked (label, confidence) per example uid, so
    escalation decisions can be asserted exactly rather than statistically.
    """

    def __init__(self, name: str, answers: dict, offline: bool = True) -> None:
        self.name = name
        self._answers = answers
        self._offline = offline

    def predict(self, ex: Example, task: TaskSpec) -> Prediction:
        label, confidence = self._answers[ex.uid]
        return Prediction(
            label=label, latency_ms=1.0, confidence=confidence, est_input_tokens=10
        )

    def meta(self) -> dict:
        return {"kind": "fixed", "offline": self._offline}


_TASK = TaskSpec(
    name="fixture",
    instructions="pick one",
    criteria={"x": "is x", "y": "is y", "z": "is z"},
    examples=[_ex("a", "x"), _ex("b", "y"), _ex("c", "z")],
)


# --------------------------------------------------------- escalation logic --
def test_confident_local_prediction_stays_local():
    local = _FixedBackend("local", {"a": ("x", 0.9), "b": ("y", 0.4), "c": ("z", 0.2)})
    hosted = _FixedBackend(
        "hosted", {"a": ("x", 0.99), "b": ("y", 0.99), "c": ("z", 0.99)}, offline=False
    )
    cascade = CascadeBackend(local, hosted, threshold=0.5)

    pred_a = cascade.predict(_ex("a", "x"), _TASK)
    assert pred_a.label == "x"
    assert pred_a.latency_ms == 1.0  # local only - never escalated

    meta = cascade.meta()
    assert meta["escalation_rate"] == 0.0


def test_unconfident_local_prediction_escalates_to_hosted():
    local = _FixedBackend("local", {"b": ("y", 0.4)})
    hosted = _FixedBackend("hosted", {"b": ("z", 0.99)}, offline=False)
    cascade = CascadeBackend(local, hosted, threshold=0.5)

    pred = cascade.predict(_ex("b", "y"), _TASK)
    # hosted's answer wins, and latency is local-then-hosted, sequentially
    assert pred.label == "z"
    assert pred.latency_ms == 2.0

    meta = cascade.meta()
    assert meta["escalation_rate"] == 1.0


def test_none_confidence_always_escalates():
    # Mirrors JevBackend, which can legitimately report confidence=None.
    local = _FixedBackend("local", {"a": ("x", None)})
    hosted = _FixedBackend("hosted", {"a": ("x", 0.99)}, offline=False)
    cascade = CascadeBackend(local, hosted, threshold=0.01)  # a near-zero bar

    pred = cascade.predict(_ex("a", "x"), _TASK)
    assert pred.latency_ms == 2.0  # escalated despite a threshold almost anyone clears

    meta = cascade.meta()
    assert meta["escalation_rate"] == 1.0


def test_cascade_meta_reports_kind_and_never_offline():
    local = _FixedBackend("local", {"a": ("x", 0.9)})
    hosted = _FixedBackend("hosted", {"a": ("x", 0.99)}, offline=False)
    cascade = CascadeBackend(local, hosted, threshold=0.5)
    cascade.predict(_ex("a", "x"), _TASK)
    meta = cascade.meta()
    assert meta["kind"] == "cascade"
    assert meta["offline"] is False  # it CAN call hosted, so never claim offline


# ---------------------------------------------------------- cascade backend flows
# through run_backend() - the single-threshold, single-point path, kept alive
# by bench.py's dedicated "kind == cascade" cost branch.
def test_cascade_backend_runs_through_bench_pipeline():
    task = load_task("synthetic", n=30)
    config = BenchConfig(limit=30)
    cascade = CascadeBackend(RulesBackend(), StubBackend(accuracy=0.95, seed=7), 0.5)
    result = run_backend(cascade, task, config)

    assert result.accuracy is not None
    assert result.meta["kind"] == "cascade"
    assert "local always runs" in result.cost["basis"]
    assert "hosted on" in result.cost["basis"]


# ------------------------------------------------------------------- cost ---
def test_cascade_cost_arithmetic():
    model = CostModel()
    local_component = local_cost(10.0, model)
    hosted_component = hosted_cost(
        total_input_tokens=1000, n_calls=1, price_per_mtok=0.042
    )

    report = cascade_cost(
        local_p50_ms=10.0,
        cost_model=model,
        hosted_avg_tokens_per_call=1000,
        hosted_price_per_mtok=0.042,
        escalation_rate=0.3,
    )
    expected = local_component.usd_per_call + 0.3 * hosted_component.usd_per_call
    assert report.usd_per_call == pytest.approx(expected)
    assert "30.0%" in report.basis


def test_cascade_cost_at_zero_escalation_equals_local_cost():
    model = CostModel()
    report = cascade_cost(
        local_p50_ms=5.0,
        cost_model=model,
        hosted_avg_tokens_per_call=500,
        hosted_price_per_mtok=0.042,
        escalation_rate=0.0,
    )
    assert report.usd_per_call == pytest.approx(local_cost(5.0, model).usd_per_call)


# ------------------------------------------------------------------ sweep ---
def test_sweep_from_predictions_rejects_mismatched_counts():
    with pytest.raises(CascadeError, match="mismatched counts"):
        sweep_from_predictions(
            local_examples=[_ex("a", "x")],
            local_preds=[Prediction(label="x", latency_ms=1.0, confidence=0.9)],
            hosted_preds=[],
            thresholds=[0.5],
            cost_model=CostModel(),
            hosted_price_per_mtok=0.042,
        )


def test_sweep_from_predictions_rejects_colliding_threshold_names():
    examples = [_ex("a", "x")]
    preds = [Prediction(label="x", latency_ms=1.0, confidence=0.9)]
    with pytest.raises(CascadeError, match="same name"):
        sweep_from_predictions(
            examples, preds, preds, [0.501, 0.502],
            cost_model=CostModel(), hosted_price_per_mtok=0.042,
        )


def test_sweep_thresholds_produces_one_result_per_threshold():
    task = load_task("synthetic", n=40)
    config = BenchConfig(limit=40)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    results = sweep_thresholds(
        RulesBackend(), StubBackend(accuracy=0.9, seed=3), task, thresholds, config
    )
    assert len(results) == len(thresholds)
    for r, t in zip(results, thresholds, strict=True):
        assert r.backend == f"cascade@{t:.2f}"
        assert r.accuracy is not None
        assert r.latency  # non-empty latency summary
        assert r.meta["kind"] == "cascade"


def test_escalation_rate_is_monotonic_in_threshold():
    # A lower threshold gives local a lower bar to clear ("stay local more"),
    # so the escalation rate must never increase as the threshold decreases.
    # If this direction is wrong, _should_escalate's comparison is inverted.
    task = load_task("synthetic", n=60)
    config = BenchConfig(limit=60)
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    results = sweep_thresholds(
        RulesBackend(), StubBackend(accuracy=0.9, seed=5), task, thresholds, config
    )
    rates = [r.meta["escalation_rate"] for r in results]
    assert rates == sorted(rates)  # non-decreasing as threshold increases


def test_swept_results_flow_through_pareto():
    task = load_task("synthetic", n=50)
    config = BenchConfig(limit=50)
    thresholds = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95]
    results = sweep_thresholds(
        RulesBackend(), StubBackend(accuracy=0.92, seed=11), task, thresholds, config
    )
    frontier = pareto(results)
    assert frontier  # non-empty
    assert set(frontier) <= {r.backend for r in results}


# -------------------------------------------------------- raw predictions ---
def test_predictions_round_trip(tmp_path: Path):
    task = load_task("synthetic", n=10)
    backend = StubBackend(accuracy=0.8, seed=1)
    examples = task.examples[:10]
    preds = [backend.predict(ex, task) for ex in examples]

    out = tmp_path / "stub.predictions.json"
    save_predictions(out, backend.name, backend.meta(), task, preds)
    name, meta, loaded_examples, loaded_preds = load_predictions(out)

    assert name == "stub"
    assert meta["kind"] == "stub"
    assert [e.uid for e in loaded_examples] == [e.uid for e in examples]
    assert [e.gold for e in loaded_examples] == [e.gold for e in examples]
    assert [p.label for p in loaded_preds] == [p.label for p in preds]
    assert [p.confidence for p in loaded_preds] == [p.confidence for p in preds]


def test_sweep_from_predictions_works_on_loaded_files(tmp_path: Path):
    # The whole point: two files, saved independently, swept with no model
    # calls and no live Backend objects at all.
    task = load_task("synthetic", n=30)
    examples = task.examples[:30]
    local_backend = RulesBackend()
    hosted_backend = StubBackend(accuracy=0.95, seed=9)
    local_preds = [local_backend.predict(ex, task) for ex in examples]
    hosted_preds = [hosted_backend.predict(ex, task) for ex in examples]

    local_path = tmp_path / "local.predictions.json"
    hosted_path = tmp_path / "hosted.predictions.json"
    save_predictions(
        local_path, local_backend.name, local_backend.meta(), task, local_preds
    )
    save_predictions(
        hosted_path, hosted_backend.name, hosted_backend.meta(), task, hosted_preds
    )

    _, _, loaded_examples, loaded_local = load_predictions(local_path)
    _, hosted_meta, _, loaded_hosted = load_predictions(hosted_path)

    results = sweep_from_predictions(
        loaded_examples, loaded_local, loaded_hosted, [0.3, 0.6, 0.9],
        cost_model=CostModel(), hosted_price_per_mtok=0.042,
    )
    assert len(results) == 3
    assert all(r.accuracy is not None for r in results)
    assert hosted_meta["kind"] == "stub"


# --------------------------------------------------------- cli exit codes ---
def _run_cli(*args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "edgefront.cli", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_cli_cascade_live_mode(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "cascade.json"
    proc = _run_cli(
        "cascade", "--task", "synthetic",
        "--local", "rules", "--hosted", "stub",
        "--thresholds", "0.2,0.4,0.6,0.8",
        "--limit", "40",
        "--json", str(out),
        cwd=root,
    )
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["results"]) == 4
    assert doc["stage"] == "cascade"
    for r in doc["results"]:
        assert r["meta"]["kind"] == "cascade"


def test_cli_cascade_rejects_mixed_modes(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    proc = _run_cli(
        "cascade", "--local", "rules", "--hosted", "stub",
        "--local-predictions", "nope.json",
        cwd=root,
    )
    assert proc.returncode == 2


def test_cli_cascade_rejects_no_mode(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    proc = _run_cli("cascade", cwd=root)
    assert proc.returncode == 2


def test_cli_bench_save_predictions_then_cli_cascade_from_files(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    pred_dir = tmp_path / "preds"
    bench_proc = _run_cli(
        "bench", "--task", "synthetic", "--backends", "rules,stub",
        "--limit", "30", "--save-predictions", str(pred_dir),
        cwd=root,
    )
    assert bench_proc.returncode == 0, bench_proc.stderr
    local_file = pred_dir / "rules.predictions.json"
    hosted_file = pred_dir / "stub.predictions.json"
    assert local_file.exists()
    assert hosted_file.exists()

    out = tmp_path / "cascade_from_files.json"
    cascade_proc = _run_cli(
        "cascade",
        "--local-predictions", str(local_file),
        "--hosted-predictions", str(hosted_file),
        "--thresholds", "0.25,0.5,0.75",
        "--json", str(out),
        cwd=root,
    )
    assert cascade_proc.returncode == 0, cascade_proc.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc["results"]) == 3
