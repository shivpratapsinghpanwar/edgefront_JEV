"""The CLI as a first-time user meets it: clean errors, correct exit codes."""

from __future__ import annotations

import subprocess
import sys

import pytest

from edgefront import cli


def _run(*argv: str, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "edgefront.cli", *argv],
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def test_quickstart_bench_report_verify(tmp_path):
    out = tmp_path / "out.json"
    r = _run(
        "bench", "--task", "synthetic", "--backends", "rules,stub", "--json", str(out)
    )
    assert r.returncode == 0, r.stderr
    assert "verdict:" in r.stdout

    r = _run("report", str(out), "--out", str(tmp_path / "BENCH.md"))
    assert r.returncode == 0, r.stderr

    assert _run("verify", str(out), "--min-acc", "0.3").returncode == 0
    assert _run("verify", str(out), "--min-acc", "0.99").returncode == 1


@pytest.mark.parametrize("command", ["report", "verify"])
def test_missing_results_file_is_a_clean_error(tmp_path, command):
    r = _run(command, str(tmp_path / "nothere.json"))
    assert r.returncode == 2
    assert "file not found" in r.stderr
    assert "Traceback" not in r.stderr


def test_corrupt_results_file_is_a_clean_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    r = _run("report", str(bad))
    assert r.returncode == 2
    assert "Traceback" not in r.stderr


def test_unknown_backend_is_a_clean_error():
    r = _run("bench", "--task", "synthetic", "--backends", "nope")
    assert r.returncode == 2
    assert "unknown backend" in r.stderr
    assert "Traceback" not in r.stderr


def test_cascade_plot_without_matplotlib_fails_before_running(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["edgefront", "cascade", "--local", "rules", "--hosted", "stub",
         "--plot", "never.png"],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "edgefront[plot]" in captured.err
    assert "running" not in captured.out


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("onnx:m/model.onnx:org/tok", ("m/model.onnx", "org/tok", None)),
        ("onnx:m/q.int8.onnx:org/tok:int8", ("m/q.int8.onnx", "org/tok", "int8")),
        (r"onnx:C:\m\model.onnx:org/tok:int8", (r"C:\m\model.onnx", "org/tok", "int8")),
        ("onnx:C:/m/model.onnx:org/tok", ("C:/m/model.onnx", "org/tok", None)),
    ],
)
def test_onnx_spec_survives_windows_drive_letters(spec, expected):
    assert cli._ONNX_SPEC.match(spec).groups() == expected


def test_export_is_listed():
    r = _run("export", "--help")
    assert r.returncode == 0
    assert "--model" in r.stdout
