"""Tests for scripts/recon_busy_check.py — the STDLIB-ONLY gate helper that
reads a fused-memory /health JSON body from stdin and classifies whether a
full reconciliation cycle is in flight (task 2703 δ). Consumed by the
cycle-aware default path of restart-fused-memory.sh.

Mirrors test_drain_check.py: a pure classify() taxonomy plus a
subprocess-driven CLI. Fail-safe by design — an unreachable/malformed body
classifies 'unreachable' (NOT busy), so the restart proceeds rather than
wedging on an endpoint it cannot read.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from recon_busy_check import classify

SCRIPT = Path(__file__).parent.parent / "recon_busy_check.py"

BUSY_ENTRY = {
    "project_id": "dark_factory",
    "run_id": "run-xyz",
    "stage": "stage1_memory_consolidation",
    "started_at": "2026-07-18T06:00:00+00:00",
}
BUSY_ENTRY_2 = {
    "project_id": "other_proj",
    "run_id": "run-2",
    "stage": None,
    "started_at": "2026-07-18T06:05:00+00:00",
}


# ---------------------------------------------------------------------------
# Pure classify(health) taxonomy: busy / idle / unreachable
# ---------------------------------------------------------------------------

def test_classify_nonempty_recon_busy_is_busy():
    assert classify({"recon_busy": [BUSY_ENTRY]}) == "busy"


def test_classify_empty_recon_busy_is_idle():
    assert classify({"status": "ok", "recon_busy": []}) == "idle"


def test_classify_absent_recon_busy_is_idle():
    assert classify({"status": "ok"}) == "idle"


def test_classify_none_is_unreachable():
    assert classify(None) == "unreachable"


def test_classify_non_dict_is_unreachable():
    """A parsed-but-non-object body (e.g. a JSON array) is not a health body
    — fail-safe to 'unreachable' so the caller proceeds."""
    assert classify([1, 2, 3]) == "unreachable"


# ---------------------------------------------------------------------------
# CLI (reads /health body from stdin) — driven via subprocess.run
# ---------------------------------------------------------------------------

def _cli_timeout_from_env(default: float = 60.0) -> float:
    """Resolve the default wall-clock budget (seconds) for a CLI subprocess.

    10s was tight enough to flake under machine load: concurrent
    orchestrator agents can push interpreter startup + imports past 10s even
    though the CLI under test behaves correctly (returncode/stderr already
    correct at the moment the old budget expired). 60s gives real headroom
    without materially slowing an idle-machine run (measured baseline for
    the whole file is ~1.1s).

    RECON_BUSY_CHECK_TEST_TIMEOUT overrides the default for further tuning
    without a code change. An unset or blank value (e.g. a CI template that
    always exports the var) is treated as "not overridden" rather than an
    error — otherwise this escape hatch would itself fail the *entire*
    module's collection, including the pure-unit tests here that never spawn
    a subprocess. A present-but-malformed value (non-numeric or non-positive)
    still fails loudly, naming the offending value, rather than silently
    falling back and masking a typo'd override.
    """
    raw = os.environ.get("RECON_BUSY_CHECK_TEST_TIMEOUT", "").strip()
    if not raw:
        return default
    error = ValueError(
        f"RECON_BUSY_CHECK_TEST_TIMEOUT must be a positive number of seconds; got {raw!r}"
    )
    try:
        value = float(raw)
    except ValueError:
        raise error from None
    if value <= 0:
        raise error
    return value


_CLI_TIMEOUT = _cli_timeout_from_env()


def _run_cli(stdin_text: str, timeout: float = _CLI_TIMEOUT) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(SCRIPT)],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_cli_busy_prints_word_then_one_detail_line_per_cycle():
    result = _run_cli(json.dumps({"recon_busy": [BUSY_ENTRY]}))
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    lines = result.stdout.splitlines()
    assert lines[0] == "busy"
    detail = [line for line in lines[1:] if line.startswith("recon_busy_cycle")]
    assert len(detail) == 1
    assert "project_id=dark_factory" in detail[0]
    assert "run_id=run-xyz" in detail[0]
    assert "stage=stage1_memory_consolidation" in detail[0]
    assert "started_at=2026-07-18T06:00:00+00:00" in detail[0]


def test_cli_busy_multiple_cycles_one_detail_each():
    result = _run_cli(json.dumps({"recon_busy": [BUSY_ENTRY, BUSY_ENTRY_2]}))
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    lines = result.stdout.splitlines()
    assert lines[0] == "busy"
    detail = [line for line in lines if line.startswith("recon_busy_cycle")]
    assert len(detail) == 2
    assert any("run_id=run-xyz" in line for line in detail)
    assert any("run_id=run-2" in line for line in detail)


def test_cli_idle_prints_word_and_no_detail():
    result = _run_cli(json.dumps({"recon_busy": []}))
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    lines = result.stdout.splitlines()
    assert lines[0] == "idle"
    assert all(not line.startswith("recon_busy_cycle") for line in lines[1:])


def test_cli_blank_stdin_is_unreachable():
    result = _run_cli("")
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.splitlines()[0] == "unreachable"


def test_cli_whitespace_only_stdin_is_unreachable():
    result = _run_cli("   \n  \t\n")
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.splitlines()[0] == "unreachable"


def test_cli_malformed_json_is_unreachable():
    result = _run_cli("{not valid json")
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.splitlines()[0] == "unreachable"


def test_cli_non_object_json_is_unreachable():
    result = _run_cli("[1, 2, 3]")
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert result.stdout.splitlines()[0] == "unreachable"


# ---------------------------------------------------------------------------
# _cli_timeout_from_env(): default-resolution branches (unset/blank -> 60.0;
# a valid override parses through). Loud-rejection branches are in the next
# section, once the helper exists.
# ---------------------------------------------------------------------------

def test_cli_timeout_from_env_unset_returns_default(monkeypatch):
    monkeypatch.delenv("RECON_BUSY_CHECK_TEST_TIMEOUT", raising=False)
    assert _cli_timeout_from_env() == 60.0


def test_cli_timeout_from_env_blank_returns_default(monkeypatch):
    monkeypatch.setenv("RECON_BUSY_CHECK_TEST_TIMEOUT", "")
    assert _cli_timeout_from_env() == 60.0


def test_cli_timeout_from_env_whitespace_only_returns_default(monkeypatch):
    monkeypatch.setenv("RECON_BUSY_CHECK_TEST_TIMEOUT", "   \n\t")
    assert _cli_timeout_from_env() == 60.0


def test_cli_timeout_from_env_integral_override_parses(monkeypatch):
    monkeypatch.setenv("RECON_BUSY_CHECK_TEST_TIMEOUT", "5")
    assert _cli_timeout_from_env() == 5.0


def test_cli_timeout_from_env_fractional_override_parses(monkeypatch):
    monkeypatch.setenv("RECON_BUSY_CHECK_TEST_TIMEOUT", "2.5")
    assert _cli_timeout_from_env() == 2.5


# ---------------------------------------------------------------------------
# _cli_timeout_from_env(): loud-rejection branches — a present-but-malformed
# override must fail loudly, naming both the env var and the offending raw
# value, rather than silently falling back or misbehaving far from the cause.
# ---------------------------------------------------------------------------

def test_cli_timeout_from_env_non_numeric_raises(monkeypatch):
    monkeypatch.setenv("RECON_BUSY_CHECK_TEST_TIMEOUT", "abc")
    with pytest.raises(ValueError) as excinfo:
        _cli_timeout_from_env()
    assert "RECON_BUSY_CHECK_TEST_TIMEOUT" in str(excinfo.value)
    assert "abc" in str(excinfo.value)


def test_cli_timeout_from_env_zero_raises(monkeypatch):
    monkeypatch.setenv("RECON_BUSY_CHECK_TEST_TIMEOUT", "0")
    with pytest.raises(ValueError) as excinfo:
        _cli_timeout_from_env()
    assert "RECON_BUSY_CHECK_TEST_TIMEOUT" in str(excinfo.value)
    assert "0" in str(excinfo.value)


def test_cli_timeout_from_env_negative_raises(monkeypatch):
    monkeypatch.setenv("RECON_BUSY_CHECK_TEST_TIMEOUT", "-1")
    with pytest.raises(ValueError) as excinfo:
        _cli_timeout_from_env()
    assert "RECON_BUSY_CHECK_TEST_TIMEOUT" in str(excinfo.value)
    assert "-1" in str(excinfo.value)


# ---------------------------------------------------------------------------
# _run_cli() wiring: the resolved budget must actually reach subprocess.run
# — this is the regression this task exists to fix. Behavioural (spy on
# subprocess.run) rather than inspect.signature-based, so it survives a
# refactor that moves resolution out of the default argument, and it spawns
# no interpreter.
# ---------------------------------------------------------------------------

def test_run_cli_passes_resolved_timeout_to_subprocess_run(monkeypatch):
    captured = {}

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="idle\n", stderr="")

    monkeypatch.setattr(subprocess, "run", spy)
    _run_cli("{}")
    assert captured["timeout"] == _CLI_TIMEOUT
    assert captured["timeout"] >= 60
