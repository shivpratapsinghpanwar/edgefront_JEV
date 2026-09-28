"""edgefront command line interface.

Follows the same contract as data-doctor: --json on every subcommand, and the
exit code is the verdict, so `edgefront verify` drops straight into CI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from .bench import BenchConfig, build_document, run_backend
from .cascade import CascadeError, sweep_from_predictions
from .frontier import DEFAULT_TOLERANCE_PTS, decide, pareto
from .measure.accuracy import score
from .measure.cost import HOSTED_PRICES_USD_PER_MTOK, CostModel, hosted_cost, local_cost
from .measure.latency import summarise
from .merge import MergeError, merge_documents
from .predictions import load_predictions, save_predictions
from .report import render_markdown, render_table
from .tasks import list_tasks, load_task
from .types import Backend, TaskSpec

EXIT_OK = 0
EXIT_FAILED_CHECK = 1
EXIT_BAD_USAGE = 2

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]+")
_ONNX_SPEC = re.compile(r"^onnx:(.+?\.onnx):([^:]+)(?::([^:]+))?$", re.IGNORECASE)


def _sanitize_filename(name: str) -> str:
    """Turn a backend name into a safe filename fragment.

    Backend names can contain `:` or `/` (e.g. an onnx spec's model path), and
    those are directory separators or reserved characters on at least one
    platform this project supports.
    """
    return _UNSAFE_FILENAME.sub("_", name)


def _write_json(data: dict, out: str | None) -> None:
    if out:
        Path(out).write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nreport written to {out}")


def _headline(ok: bool, message: str) -> None:
    print(("  OK   " if ok else "  FAIL ") + message)


def _build_backend(spec: str) -> Backend:
    """Turn a --backends token into a backend instance.

    Import is lazy and per-backend so that a missing optional dependency only
    breaks the backend that needs it.
    """
    name = spec.strip().lower()
    if name == "rules":
        from .backends.rules import RulesBackend

        return RulesBackend()
    if name == "stub":
        from .backends.stub import StubBackend

        return StubBackend()
    if name == "jev":
        from .backends.jev import JevBackend

        return JevBackend()
    if name.startswith("onnx:"):
        # onnx:<model.onnx>:<tokenizer-id>[:precision]
        # Anchored on ".onnx:" rather than split(":") so a Windows drive
        # letter (onnx:C:\models\model.onnx:...) is not read as a separator.
        m = _ONNX_SPEC.match(spec)
        if not m:
            raise ValueError(
                "onnx backend needs onnx:<model.onnx>:<tokenizer-id>[:precision]"
            )
        from .backends.onnx_local import ONNXLocalBackend

        path, tokenizer, precision = m.group(1), m.group(2), m.group(3) or "fp32"
        return ONNXLocalBackend(path, tokenizer, precision=precision)
    if name.startswith("hf:") or name == "hf":
        from .backends.hf_local import DEFAULT_MODEL, HFLocalBackend

        model = spec.split(":", 1)[1] if ":" in spec else DEFAULT_MODEL
        return HFLocalBackend(model)
    raise ValueError(
        f"unknown backend: {spec}. Known: rules, stub, jev, "
        f"hf[:model-id], onnx:<file>:<tokenizer>[:precision]"
    )


def cmd_tasks(args: argparse.Namespace) -> int:
    for name, description in list_tasks().items():
        print(f"  {name:<12} {description}")
    return EXIT_OK


def cmd_bench(args: argparse.Namespace) -> int:
    try:
        task = load_task(args.task)
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_USAGE

    backends = []
    for spec in args.backends.split(","):
        if not spec.strip():
            continue
        try:
            backends.append(_build_backend(spec))
        except (ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_BAD_USAGE
    if not backends:
        print("error: no backends selected", file=sys.stderr)
        return EXIT_BAD_USAGE

    config = BenchConfig(
        warmup=args.warmup,
        limit=args.limit,
        cost_model=CostModel(
            hardware_usd=args.hardware_usd,
            watts=args.watts,
            utilisation=args.utilisation,
        ),
    )
    n = len(task.examples[: args.limit] if args.limit else task.examples)
    print(f"task {task.name}: {n} examples, {len(task.labels)} labels\n")

    started = time.perf_counter()
    results = []
    for backend in backends:
        print(f"running {backend.name} ...", flush=True)
        results.append(run_backend(backend, task, config))
    duration = time.perf_counter() - started

    if args.save_predictions:
        # Predictions are still in memory here (BackendResult keeps them;
        # only .to_dict() and the JSON written below discard them) - this is
        # what lets a later `edgefront cascade --local-predictions ...
        # --hosted-predictions ...` combine a local run and a hosted run that
        # never happened in the same process, see predictions.py.
        out_dir = Path(args.save_predictions)
        out_dir.mkdir(parents=True, exist_ok=True)
        for backend, result in zip(backends, results, strict=True):
            fname = f"{_sanitize_filename(backend.name)}.predictions.json"
            out_path = out_dir / fname
            save_predictions(
                out_path, backend.name, result.meta, task, result.predictions
            )
            print(f"predictions written to {out_path}")

    verdict = decide(results, tolerance_pts=args.tolerance)
    frontier = pareto(results)
    doc = build_document(
        task, results, config, duration, verdict.to_dict(), frontier
    )

    print()
    print(render_table(doc))
    print()
    print(f"verdict: {verdict.headline}")
    for line in verdict.detail:
        print(f"         {line}")

    _write_json(doc, args.json)
    if args.md:
        Path(args.md).write_text(render_markdown(doc), encoding="utf-8")
        print(f"markdown written to {args.md}")
    return EXIT_OK


def cmd_cascade(args: argparse.Namespace) -> int:
    """Confidence-gated local->hosted cascade: sweep thresholds for free.

    Two input modes, mutually exclusive: --local/--hosted builds live
    backends and runs each exactly once in this process (see
    cascade.sweep_thresholds); --local-predictions/--hosted-predictions reads
    two files written by `edgefront bench --save-predictions` - possibly
    produced on different machines, at different times, by different people -
    and sweeps those instead, with no backend construction and no optional
    dependency needed at all. Either way, the sweep itself
    (cascade.sweep_from_predictions) is pure arithmetic on cached predictions:
    the hosted backend is never called more than once per example.
    """
    live = bool(args.local or args.hosted)
    from_files = bool(args.local_predictions or args.hosted_predictions)
    if live and from_files:
        print(
            "error: give --local/--hosted or --local-predictions/"
            "--hosted-predictions, not both",
            file=sys.stderr,
        )
        return EXIT_BAD_USAGE
    if not live and not from_files:
        print(
            "error: need --local/--hosted (live) or --local-predictions/"
            "--hosted-predictions (cross-machine)",
            file=sys.stderr,
        )
        return EXIT_BAD_USAGE
    if args.plot:
        # Checked up front: plot.py imports matplotlib lazily, so without this
        # a missing extra would only surface after every model has already run.
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print(
                "error: matplotlib is required for --plot. "
                "pip install 'edgefront[plot]'",
                file=sys.stderr,
            )
            return EXIT_BAD_USAGE

    try:
        thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    except ValueError:
        print(
            f"error: --thresholds must be comma-separated floats, got "
            f"{args.thresholds!r}",
            file=sys.stderr,
        )
        return EXIT_BAD_USAGE
    if not thresholds:
        print("error: no thresholds given", file=sys.stderr)
        return EXIT_BAD_USAGE

    cost_model = CostModel(
        hardware_usd=args.hardware_usd, watts=args.watts, utilisation=args.utilisation
    )

    started = time.perf_counter()
    if live:
        if not (args.local and args.hosted):
            print("error: --local and --hosted must both be given", file=sys.stderr)
            return EXIT_BAD_USAGE
        try:
            task = load_task(args.task)
        except (ValueError, ImportError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_BAD_USAGE
        try:
            local_backend = _build_backend(args.local)
            hosted_backend = _build_backend(args.hosted)
        except (ValueError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_BAD_USAGE

        examples = task.examples[: args.limit] if args.limit else task.examples
        print(
            f"task {task.name}: {len(examples)} examples, running "
            f"{local_backend.name} and {hosted_backend.name} once each ...",
            flush=True,
        )
        local_preds = [local_backend.predict(ex, task) for ex in examples]
        hosted_preds = [hosted_backend.predict(ex, task) for ex in examples]
        hosted_meta = hosted_backend.meta()
        task_doc: TaskSpec | dict = task
        local_name, hosted_name = local_backend.name, hosted_backend.name
    else:
        if not (args.local_predictions and args.hosted_predictions):
            print(
                "error: --local-predictions and --hosted-predictions must "
                "both be given",
                file=sys.stderr,
            )
            return EXIT_BAD_USAGE
        local_name, _local_meta, examples, local_preds = load_predictions(
            args.local_predictions
        )
        hosted_name, hosted_meta, hosted_examples, hosted_preds = load_predictions(
            args.hosted_predictions
        )
        if [e.uid for e in examples] != [e.uid for e in hosted_examples]:
            print(
                "error: local and hosted prediction files do not describe "
                "the same examples in the same order",
                file=sys.stderr,
            )
            return EXIT_BAD_USAGE
        print(
            f"loaded {len(examples)} examples from two prediction files "
            f"({local_name}, {hosted_name}) - no model calls, no network",
            flush=True,
        )
        # No live TaskSpec exists in this mode - both prediction files carry
        # their own task summary (written by predictions.save_predictions),
        # but only their labels/name, not the full TaskSpec.criteria this
        # command never needs. build_document() accepts a plain dict here.
        # The task name itself IS worth recovering from the raw file (purely
        # for a readable title on `--plot`'s chart and this doc's `task.name`)
        # - peeked directly rather than widening load_predictions' return
        # shape for every caller just for a label.
        try:
            task_name = (
                json.loads(
                    Path(args.local_predictions).read_text(encoding="utf-8")
                )
                .get("task", {})
                .get("name")
                or "cascade (loaded predictions)"
            )
        except Exception:
            task_name = "cascade (loaded predictions)"
        task_doc = {
            "name": task_name,
            "instructions": "",
            "labels": sorted({ex.gold for ex in examples}),
            "n_examples": len(examples),
            "source": f"{args.local_predictions} + {args.hosted_predictions}",
        }

    price = HOSTED_PRICES_USD_PER_MTOK.get(
        hosted_meta.get("vendor", ""), HOSTED_PRICES_USD_PER_MTOK["jev"]
    )
    try:
        results = sweep_from_predictions(
            examples,
            local_preds,
            hosted_preds,
            thresholds,
            cost_model=cost_model,
            hosted_price_per_mtok=price,
            warmup=args.warmup,
        )
    except CascadeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_USAGE
    duration = time.perf_counter() - started

    verdict = decide(results)
    frontier = pareto(results)
    doc = build_document(
        task_doc,
        results,
        BenchConfig(warmup=args.warmup, cost_model=cost_model),
        duration,
        verdict.to_dict(),
        frontier,
    )
    doc["stage"] = "cascade"

    print()
    print(render_table(doc))
    print()
    print(f"verdict: {verdict.headline}")
    for line in verdict.detail:
        print(f"         {line}")
    if frontier:
        print(f"cost-accuracy frontier: {', '.join(frontier)}")

    _write_json(doc, args.json)
    if args.md:
        Path(args.md).write_text(render_markdown(doc), encoding="utf-8")
        print(f"markdown written to {args.md}")

    if args.plot:
        from .plot import draw_cascade

        local_ok = [p for p in local_preds if p.ok]
        local_latency = summarise([p.latency_ms for p in local_ok], warmup=args.warmup)
        local_acc, _ = score(local_preds, examples)
        local_only_cost = local_cost(float(local_latency.get("p50", 0.0)), cost_model)

        hosted_ok = [p for p in hosted_preds if p.ok]
        hosted_latency = summarise(
            [p.latency_ms for p in hosted_ok], warmup=args.warmup
        )
        hosted_acc, _ = score(hosted_preds, examples)
        hosted_tokens = sum(p.est_input_tokens or 0 for p in hosted_ok)
        hosted_only_cost = hosted_cost(hosted_tokens, max(1, len(hosted_ok)), price)

        curve = [
            (
                r.accuracy or 0.0,
                float(r.latency.get("p50", 0.0)),
                float(r.cost.get("usd_per_million_calls", 0.0)),
            )
            for r in sorted(results, key=lambda r: r.meta.get("threshold", 0.0))
        ]
        task_name = task_doc.name if hasattr(task_doc, "name") else task_doc["name"]
        draw_cascade(
            local_label=f"{local_name} (local only)",
            local_acc=local_acc,
            local_p50=float(local_latency.get("p50", 0.0)),
            local_cost=local_only_cost.usd_per_million_calls,
            hosted_label=f"{hosted_name} (hosted only)",
            hosted_acc=hosted_acc,
            hosted_p50=float(hosted_latency.get("p50", 0.0)),
            hosted_cost=hosted_only_cost.usd_per_million_calls,
            curve=curve,
            title=f"{task_name} — local→hosted cascade",
            subtitle=(
                f"{local_name} → {hosted_name}, {len(thresholds)} thresholds "
                f"swept from one local pass + one hosted pass"
            ),
            out_path=args.plot,
        )
        print(f"chart written to {args.plot}")
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    doc = json.loads(Path(args.results).read_text(encoding="utf-8"))
    markdown = render_markdown(doc)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"markdown written to {args.out}")
    else:
        print(markdown)
    return EXIT_OK


def cmd_merge(args: argparse.Namespace) -> int:
    docs = []
    for path in args.results:
        docs.append(json.loads(Path(path).read_text(encoding="utf-8")))
    try:
        merged = merge_documents(docs, tolerance_pts=args.tolerance)
    except MergeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_USAGE

    print(render_table(merged))
    print()
    verdict = merged["verdict"]
    print(f"verdict: {verdict['headline']}")
    for line in verdict["detail"]:
        print(f"         {line}")

    _write_json(merged, args.json)
    if args.md:
        Path(args.md).write_text(render_markdown(merged), encoding="utf-8")
        print(f"markdown written to {args.md}")
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    doc = json.loads(Path(args.results).read_text(encoding="utf-8"))
    results = doc.get("results", [])
    if not results:
        print("  FAIL no results in document")
        return EXIT_FAILED_CHECK

    target = args.backend
    if target:
        results = [r for r in results if r["backend"] == target]
        if not results:
            print(f"  FAIL no backend named {target} in document")
            return EXIT_FAILED_CHECK

    failed = False
    for r in results:
        name = r["backend"]
        latency = r.get("latency_ms") or {}
        if args.min_acc is not None:
            acc = r.get("accuracy")
            ok = acc is not None and acc >= args.min_acc
            failed |= not ok
            _headline(ok, f"{name}: accuracy {acc:.3f} >= {args.min_acc:.3f}")
        if args.max_p99 is not None:
            p99 = latency.get("p99")
            ok = p99 is not None and p99 <= args.max_p99
            failed |= not ok
            _headline(ok, f"{name}: p99 {p99:.1f}ms <= {args.max_p99:.1f}ms")
        if args.max_ece is not None:
            ece = r.get("ece")
            ok = ece is not None and ece <= args.max_ece
            failed |= not ok
            _headline(ok, f"{name}: ECE {ece} <= {args.max_ece}")
        if r.get("n_errors"):
            failed = True
            _headline(False, f"{name}: {r['n_errors']} failed call(s)")

    return EXIT_FAILED_CHECK if failed else EXIT_OK


def cmd_export(args: argparse.Namespace) -> int:
    from .quantize import export_onnx, quantize_int8, size_mb

    try:
        fp32 = export_onnx(args.model, args.out, overwrite=args.overwrite)
        print(f"fp32 model: {fp32} ({size_mb(fp32)} MB)")
        specs = [f"onnx:{fp32}:{args.model}"]
        if not args.no_int8:
            int8 = quantize_int8(fp32)
            print(f"int8 model: {int8} ({size_mb(int8)} MB)")
            specs.append(f"onnx:{int8}:{args.model}:int8")
    except ImportError:
        print(
            "error: export needs extra deps: pip install 'edgefront[local]'",
            file=sys.stderr,
        )
        return EXIT_BAD_USAGE
    print()
    print("benchmark them with:")
    print(f'  edgefront bench --task synthetic --backends "rules,{",".join(specs)}"')
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edgefront",
        description=(
            "Do you need a hosted decision model, or does a quantized local "
            "model match it? Measure it."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("tasks", help="list available decision tasks")
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("bench", help="run a task across backends")
    p.add_argument("--task", default="synthetic")
    p.add_argument(
        "--backends",
        default="rules,stub",
        help="rules, stub, jev, hf[:model-id], onnx:<file>:<tok>[:prec]",
    )
    p.add_argument("--limit", type=int, default=None, help="cap examples")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_PTS)
    p.add_argument("--hardware-usd", type=float, default=600.0)
    p.add_argument("--watts", type=float, default=45.0)
    p.add_argument("--utilisation", type=float, default=0.25)
    p.add_argument("--json", default=None, help="write the result document here")
    p.add_argument("--md", default=None, help="write a markdown report here")
    p.add_argument(
        "--save-predictions",
        default=None,
        metavar="DIR",
        help=(
            "write each backend's raw per-example predictions to DIR "
            "(<backend>.predictions.json), for a later cross-machine "
            "`edgefront cascade --local-predictions/--hosted-predictions`"
        ),
    )
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser(
        "cascade",
        help=(
            "confidence-gated local->hosted cascade; sweep thresholds for "
            "the price of one local pass + one hosted pass"
        ),
    )
    p.add_argument("--task", default="synthetic")
    p.add_argument("--local", default=None, help="backend spec for the local arm")
    p.add_argument("--hosted", default=None, help="backend spec for the hosted arm")
    p.add_argument(
        "--local-predictions",
        default=None,
        metavar="PATH",
        help="a *.predictions.json file from `edgefront bench --save-predictions`",
    )
    p.add_argument(
        "--hosted-predictions",
        default=None,
        metavar="PATH",
        help="a *.predictions.json file from `edgefront bench --save-predictions`",
    )
    p.add_argument(
        "--thresholds",
        default="0.3,0.5,0.7,0.9,0.95",
        help="comma-separated confidence thresholds to sweep",
    )
    p.add_argument(
        "--limit", type=int, default=None, help="cap examples (live mode only)"
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--hardware-usd", type=float, default=600.0)
    p.add_argument("--watts", type=float, default=45.0)
    p.add_argument("--utilisation", type=float, default=0.25)
    p.add_argument("--json", default=None, help="write the result document here")
    p.add_argument("--md", default=None, help="write a markdown report here")
    p.add_argument(
        "--plot",
        default=None,
        metavar="PATH",
        help="write a cost-accuracy frontier chart (PNG) here; needs edgefront[plot]",
    )
    p.set_defaults(func=cmd_cascade)

    p = sub.add_parser(
        "merge",
        help="combine result documents from different machines into one verdict",
    )
    p.add_argument("results", nargs="+", help="two or more result JSON files")
    p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE_PTS)
    p.add_argument("--json", default=None)
    p.add_argument("--md", default=None)
    p.set_defaults(func=cmd_merge)

    p = sub.add_parser("report", help="render a result document as markdown")
    p.add_argument("results")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("verify", help="assert thresholds; exit 1 if any fail")
    p.add_argument("results")
    p.add_argument("--backend", default=None, help="check only this backend")
    p.add_argument("--min-acc", type=float, default=None)
    p.add_argument("--max-p99", type=float, default=None)
    p.add_argument("--max-ece", type=float, default=None)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser(
        "export",
        help="export a HuggingFace NLI model to ONNX fp32 + dynamic INT8",
    )
    p.add_argument(
        "--model",
        default="typeform/distilbert-base-uncased-mnli",
        help="HuggingFace model id (an NLI / zero-shot checkpoint)",
    )
    p.add_argument("--out", default="models", help="output directory")
    p.add_argument("--no-int8", action="store_true", help="skip INT8 quantization")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_export)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        code = args.func(args)
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        code = EXIT_BAD_USAGE
    except json.JSONDecodeError as exc:
        print(f"error: not a valid edgefront JSON document ({exc})", file=sys.stderr)
        code = EXIT_BAD_USAGE
    sys.exit(code)


if __name__ == "__main__":
    main()
