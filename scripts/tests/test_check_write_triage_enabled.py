"""Tests for scripts/check_write_triage_enabled.py — gate φ's fail-closed flag sub-check.

Assertions are on the exit code only; the printed line is for the operator.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'check_write_triage_enabled.py'


def run(tmp_path: Path, text: str | None) -> int:
    config = tmp_path / 'config.yaml'
    if text is not None:
        config.write_text(text)
    return subprocess.run([sys.executable, str(SCRIPT), '--config', str(config)],
                          capture_output=True, text=True, cwd=tmp_path).returncode


def test_enabled_true_exits_zero(tmp_path):
    assert run(tmp_path, 'write_triage:\n  enabled: true\n') == 0


@pytest.mark.parametrize('text', [
    'write_triage:\n  enabled: false\n',
    "write_triage:\n  enabled: 'true'\n",
    'write_triage:\n  enabled: 1\n',
    'write_triage:\n  candidate_k: 20\n',
    'mem0:\n  enabled: true\n',
    'write_triage: [enabled]\n',
    '',
    'write_triage: {enabled: true\n',
    None,
], ids=['false', 'string', 'int', 'key-absent', 'section-absent', 'section-not-a-map',
        'empty-file', 'unparseable', 'file-absent'])
def test_anything_but_boolean_true_fails_closed(tmp_path, text):
    assert run(tmp_path, text) == 1


def test_the_executable_bit_is_set():
    assert SCRIPT.stat().st_mode & 0o111
