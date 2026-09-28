"""Kaggle kernel: collect raw local predictions for the real banking77 cascade.

Sibling to ``kaggle_job/run_banking77.py``, but writes raw per-example
predictions (``edgefront.predictions.save_predictions``) instead of an
aggregate result document, because ``edgefront cascade
--local-predictions/--hosted-predictions`` needs the raw label+confidence per
example to sweep thresholds - see ``src/edgefront/cascade.py`` and
``src/edgefront/predictions.py``.

Does NOT call the hosted (`jev`) backend - no API key on this kernel, on
purpose. The hosted pass runs locally, wherever the key lives, and the two
raw-prediction files get combined afterwards with `edgefront cascade
--local-predictions ... --hosted-predictions ...` (no model calls, no
network in that step).

Same task load as `run_banking77.py`: banking77, n=300, split=test, seed=0
(all `load_task` defaults except `n`) - so example order/uid must match
exactly whatever the hosted pass loads (`edgefront bench --task banking77
--limit 300`, which loads n=500 then truncates to the same first 300 rows of
the same seed-0 shuffle). This match is verified locally before sweeping, not
assumed.

Self-contained on purpose: this file is the entire kernel. It installs
edgefront from the public GitHub repo, so pushing a new edgefront commit and
re-running this kernel always benchmarks the latest code (the cascade module
landed on `main` at commit 8c61efe).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

EDGEFRONT_REPO = "git+https://github.com/shivpratapsinghpanwar/edgefront.git"
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
    # See run_banking77.py's note: some torch builds Kaggle ships default
    # torch.onnx.export() to the "dynamo" exporter, which needs onnxscript.
    # edgefront pins dynamo=False explicitly (quantize/export.py) so this
    # should not be exercised, but installing it is cheap insurance.
    _pip_install("onnxscript")

    from edgefront.backends.hf_local import HFLocalBackend
    from edgefront.backends.onnx_local import ONNXLocalBackend
    from edgefront.bench import BenchConfig, run_backend
    from edgefront.predictions import save_predictions
    from edgefront.quantize import export_onnx, quantize_int8, size_mb
    from edgefront.tasks import load_task

    print("loading banking77 ...", flush=True)
    task = load_task("banking77", n=N_EXAMPLES)
    print(f"{len(task.examples)} examples, {len(task.labels)} labels")
    print(f"first uid={task.examples[0].uid} last uid={task.examples[-1].uid}")

    config = BenchConfig(limit=N_EXAMPLES)

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
    torch_result = run_backend(torch_backend, task, config)
    torch_path = OUT_DIR / f"{torch_backend.name}.predictions.json"
    save_predictions(
        torch_path, torch_backend.name, torch_result.meta, task, torch_result.predictions
    )
    print(f"wrote {torch_path}")

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
        providers=["CPUExecutionProvider"],  # deliberately CPU - see
        # run_banking77.py's note: the point of this arm is "can a
        # laptop-class device do this," not GPU ONNX throughput.
    )
    onnx_result = run_backend(onnx_backend, task, config)
    onnx_path = OUT_DIR / f"{onnx_backend.name}.predictions.json"
    save_predictions(
        onnx_path, onnx_backend.name, onnx_result.meta, task, onnx_result.predictions
    )
    print(f"wrote {onnx_path}")

    duration = time.perf_counter() - started
    print(f"\ndone in {duration:.1f}s")
    for name, result in (
        (torch_backend.name, torch_result),
        (onnx_backend.name, onnx_result),
    ):
        lat = result.latency
        print(
            f"  {name:<28} acc={result.accuracy:.3f}  "
            f"p50={lat.get('p50', 0):.1f}ms  n_errors={result.n_errors}"
        )


if __name__ == "__main__":
    main()
