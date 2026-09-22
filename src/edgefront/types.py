"""Core data types.

Everything here is stdlib-only and vendor-neutral. A backend is anything that
can turn an Example into a Prediction; nothing in this module knows that Jev,
ONNX Runtime or transformers exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Example:
    """One labelled decision to make."""

    uid: str
    text: str
    gold: str
    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """A decision task: the label space plus the wording every backend sees.

    ``criteria`` is the per-label description handed to a hosted model. The same
    text is what a local model's label names are matched against, so the wording
    is part of the experiment and must not differ between backends.
    """

    name: str
    instructions: str
    criteria: dict[str, str]
    examples: list[Example]
    source: str = ""

    @property
    def labels(self) -> list[str]:
        return list(self.criteria)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "instructions": self.instructions,
            "labels": self.labels,
            "n_examples": len(self.examples),
            "source": self.source,
        }


@dataclass(frozen=True)
class Prediction:
    """One backend's answer for one example."""

    label: str
    latency_ms: float
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    est_input_tokens: int | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "latency_ms": self.latency_ms,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "est_input_tokens": self.est_input_tokens,
            "error": self.error,
        }


@dataclass
class BackendResult:
    """Everything measured for one backend on one task."""

    backend: str
    meta: dict
    predictions: list[Prediction]
    accuracy: float | None = None
    macro_f1: float | None = None
    ece: float | None = None
    latency: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    n_errors: int = 0

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "meta": self.meta,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "ece": self.ece,
            "latency_ms": self.latency,
            "cost": self.cost,
            "n_predictions": len(self.predictions),
            "n_errors": self.n_errors,
        }


@runtime_checkable
class Backend(Protocol):
    """The only interface a backend must satisfy.

    Keeping this narrow is deliberate: every vendor-specific line in this project
    lives behind it, so the measurement, quantization and reporting layers stay
    useful if any one vendor changes or disappears.
    """

    name: str

    def predict(self, ex: Example, task: TaskSpec) -> Prediction: ...

    def meta(self) -> dict: ...
