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
import subprocess
from pathlib import Path

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

def _run_cli(stdin_text: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(SCRIPT)],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=10,
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
