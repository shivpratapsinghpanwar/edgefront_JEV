"""Kaggle kernel: benchmark local zero-shot NLI backends on banking77.

Runs the local half of the hosted-vs-local comparison on a GPU, where it is
actually tractable — 77 labels means one forward pass per label per example,
and that is roughly 19 s/example on a laptop CPU. Does NOT call the hosted
backend: there is no TypeSafe API key on this kernel, by design. Merge this
kernel's `banking77_local.json` with a locally-produced hosted-only result
using `edgefront.merge` (see kaggle_job/README.md) to get the full comparison.

Self-contained on purpose: this file is the entire kernel. It installs
edgefront from the public GitHub repo, so pushing a new edgefront commit and
re-running this kernel always benchmarks the latest code.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

EDGEFRONT_REPO = "git+https://github.com/shivpratapsinghpanwar/edgefront_JEV.git"
MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
N_EXAMPLES = 300
OUT_DIR = Path("/kaggle/working")


def _pip_install(*args: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", *args], check=True
    )


def main() -> None:
    started = time.perf_counter()

    print("installing edgefront + datasets ...", flush=True)
    _pip_install(f"{EDGEFRONT_REPO}#egg=edgefront[local]")
    _pip_install("datasets")
    # Defensive: the torch build Kaggle ships defaults torch.onnx.export()
    # to the "dynamo" exporter, which needs onnxscript. edgefront now pins
    # dynamo=False explicitly (see edgefront/quantize/export.py) so this
    # should not be exercised, but installing it is cheap insurance against
    # a future torch release routing through that path some other way.
    _pip_install("onnxscript")

    from edgefront.backends.hf_local import HFLocalBackend
    from edgefront.backends.onnx_local import ONNXLocalBackend
    from edgefront.backends.rules import RulesBackend
    from edgefront.bench import BenchConfig, build_document, run_backend
    from edgefront.frontier import pareto
    from edgefront.quantize import export_onnx, quantize_int8, size_mb
    from edgefront.tasks import load_task

    print("loading banking77 ...", flush=True)
    task = load_task("banking77", n=N_EXAMPLES)
    print(f"{len(task.examples)} examples, {len(task.labels)} labels")

    config = BenchConfig(limit=N_EXAMPLES)
    results = []

    print("running rules baseline ...", flush=True)
    results.append(run_backend(RulesBackend(), task, config))

    import torch

    # Never assume the GPU is actually there: a GPU kernel with CUDA
    # unavailable for any reason should not silently benchmark "torch-gpu" on
    # CPU and label the numbers as if the accelerator were used.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running {MODEL_NAME} on torch/{device} ...", flush=True)
    if device != "cuda":
        print(
            "  WARNING: CUDA not available on this kernel - falling back to "
            "CPU. Check the kernel's accelerator setting before trusting "
            "this arm's latency numbers.",
            flush=True,
        )
    torch_backend = HFLocalBackend(
        MODEL_NAME, name=f"distilbert-mnli-torch-{device}", device=device
    )
    results.append(run_backend(torch_backend, task, config))

    print("exporting to ONNX and quantizing to INT8 ...", flush=True)
    onnx_dir = OUT_DIR / "onnx_export"
    fp32_path = export_onnx(MODEL_NAME, onnx_dir)
    int8_path = quantize_int8(fp32_path)
    print(f"  fp32 {size_mb(fp32_path)} MB -> int8 {size_mb(int8_path)} MB")

    print("running ONNX INT8 on CPU ...", flush=True)
    onnx_backend = ONNXLocalBackend(
        int8_path,
        MODEL_NAME,
        precision="int8",
        name="distilbert-mnli-onnx-int8-cpu",
        providers=["CPUExecutionProvider"],  # deliberately CPU: the point of
        # this arm is "can a laptop-class device do this," not "how fast is a
        # Kaggle GPU at ONNX INT8."
    )
    results.append(run_backend(onnx_backend, task, config))

    duration = time.perf_counter() - started
    doc = build_document(
        task,
        results,
        config,
        duration,
        verdict_dict={
            "headline": "local-only run: no hosted backend on this kernel",
            "detail": [
                "merge with a local hosted-only result via edgefront.merge "
                "for the full verdict"
            ],
            "recommended": None,
            "tolerance_pts": 0.0,
        },
        frontier_names=pareto(results),
    )

    out_path = OUT_DIR / "banking77_local.json"
    out_path.write_text(
        json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")
    for r in results:
        lat = r.latency
        print(
            f"  {r.backend:<28} acc={r.accuracy:.3f}  "
            f"p50={lat.get('p50', 0):.1f}ms  n_errors={r.n_errors}"
        )


if __name__ == "__main__":
    main()
