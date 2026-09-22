"""Raw per-example prediction documents.

Every other document in this project is an aggregate: `bench.py`'s
`BackendResult` throws away the raw predictions list once accuracy/ECE/
latency/cost are computed (see `BackendResult.to_dict()`, and
`merge.py`'s `_as_backend_result()`, which reconstructs with
`predictions=[]` even when reading a *full* result document back). That is
deliberate house style and this module does not fight it.

But a confidence-gated cascade needs the raw label+confidence for every
example from *both* the local and the hosted backend, so a threshold sweep
can be computed post-hoc (see `cascade.sweep_from_predictions`). And the two
arms of a cascade often cannot run in the same process at the same time: the
local arm typically wants a GPU (Kaggle, say, per `kaggle_job/`) and the
hosted arm needs an API key that lives somewhere else entirely - the exact
same problem `merge.py` solves one level up, at the aggregate level (see its
module docstring). This module is the raw-data equivalent of `merge.py`: it
lets the local pass and the hosted pass happen at different times, in
different processes, and be combined later.
"""

from __future__ import annotations

import json
from pathlib import Path

from .types import Example, Prediction, TaskSpec

PREDICTIONS_SCHEMA_VERSION = 1


def save_predictions(
    path: str | Path,
    backend_name: str,
    meta: dict,
    task: TaskSpec,
    predictions: list[Prediction],
) -> None:
    """Write one backend's raw predictions for one task run.

    Only `uid` and `gold` are kept per example, not the input text - nothing
    downstream of a cascade sweep (scoring, calibration, latency, cost) reads
    the text, only the label it was scored against, and dropping it keeps
    these documents from ballooning on a large task.
    """
    doc = {
        "schema_version": PREDICTIONS_SCHEMA_VERSION,
        "stage": "raw-predictions",
        "backend": backend_name,
        "meta": meta,
        "task": task.to_dict(),
        "examples": [
            {"uid": ex.uid, "gold": ex.gold}
            for ex in task.examples[: len(predictions)]
        ],
        "predictions": [p.to_dict() for p in predictions],
    }
    Path(path).write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def load_predictions(
    path: str | Path,
) -> tuple[str, dict, list[Example], list[Prediction]]:
    """Read a raw-predictions document back.

    Reconstructed `Example`s carry an empty `text` - see `save_predictions`
    for why it was never written in the first place.
    """
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = [
        Example(uid=e["uid"], text="", gold=e["gold"])
        for e in doc.get("examples", [])
    ]
    predictions = [
        Prediction(
            label=p["label"],
            latency_ms=p["latency_ms"],
            probabilities=p.get("probabilities"),
            confidence=p.get("confidence"),
            est_input_tokens=p.get("est_input_tokens"),
            error=p.get("error"),
        )
        for p in doc.get("predictions", [])
    ]
    return doc["backend"], doc.get("meta") or {}, examples, predictions
