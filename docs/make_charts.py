"""Regenerate the README's frontier charts from measured results.

Run this whenever a real (non-stub) benchmark produces new numbers, so the
charts in `docs/charts/` never drift from `README.md`'s tables the way a
hand-edited image would. Values below are transcribed from `bench_full.json`
(synthetic) and `HANDOVER.md`'s recorded Kaggle result (banking77) - this
script does not invent or re-measure anything, it only draws what was already
reported.

Drawing code lives in `edgefront.plot` (`plot_frontier`), promoted out of this
script so `edgefront cascade --plot`'s cost-accuracy sweep chart does not need
a second copy of the same matplotlib boilerplate - this file only holds the
two charts' data and calls into it.

Palette: validated categorical slots 1/2/3 (blue/orange/aqua) from the
project's dataviz reference palette, `node scripts/validate_palette.js
"#2a78d6,#eb6834,#1baf7a" --mode light --pairs all` -> ALL CHECKS PASS. Baseline
uses neutral gray, outside the categorical set, since `rules` is a reference
floor rather than a competing "kind" of backend.
"""

from __future__ import annotations

from pathlib import Path

from edgefront.plot import plot_frontier

OUT = Path(__file__).parent / "charts"
OUT.mkdir(exist_ok=True)

# (label, kind, accuracy, p50_ms, cost_per_million_usd, label_anchor)
# label_anchor = (ha, va, dx, dy) hand-placed per point so a 4-point chart
# never needs generic collision avoidance - there are only ever 4 of these.
SYNTHETIC = [
    ("rules (baseline)", "baseline", 0.450, 0.013, 0.0003, ("left", "center", 10, 0)),
    ("distilbert-mnli, ONNX fp32", "local-fp32", 0.383, 481.0, 12.92, ("right", "top", -12, -10)),
    ("distilbert-mnli, ONNX int8", "local-int8", 0.500, 249.1, 6.69, ("left", "bottom", 12, 12)),
    ("jev-latest (hosted)", "hosted", 0.950, 385.5, 3.59, ("left", "center", 12, 0)),
]
BANKING77 = [
    ("rules (baseline)", "baseline", 0.293, 0.037, 0.001, ("left", "center", 10, 0)),
    ("distilbert-mnli, torch/CUDA fp32", "local-fp32", 0.193, 62.7, 1.68, ("right", "bottom", -12, 12)),
    ("distilbert-mnli, ONNX int8/CPU", "local-int8", 0.167, 774.0, 20.8, ("left", "top", 12, -12)),
    ("jev-latest (hosted)", "hosted", 0.790, 391.0, 57.4, ("left", "center", 12, 0)),
]


if __name__ == "__main__":
    synthetic_out = OUT / "synthetic_frontier.png"
    plot_frontier(
        SYNTHETIC,
        "synthetic task — 4 labels, 60 examples",
        "keyword-vocabulary-overlapping task; a weak discriminator on its own, see Known weaknesses",
        synthetic_out,
    )
    print(f"wrote {synthetic_out}")

    banking77_out = OUT / "banking77_frontier.png"
    plot_frontier(
        BANKING77,
        "banking77 — 77 labels, 300 examples",
        "real intent-classification benchmark, Kaggle T4 (local) + hosted (local machine)",
        banking77_out,
    )
    print(f"wrote {banking77_out}")
