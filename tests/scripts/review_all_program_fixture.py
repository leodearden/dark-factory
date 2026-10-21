"""Shared helpers for the review-all program tooling tests.

``skills/review-all/scripts/program/`` holds stdlib scripts loaded by path, not
an importable package, and tests may not live under ``skills/``
(``test_skills_module_config_decision.py::test_skills_has_no_tests_of_its_own``).
This non-test module is importable because ``tests/scripts/conftest.py``
inserts this directory on ``sys.path``.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import select
import subprocess
import sys
import tempfile
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).parents[2]
PROGRAM_DIR = REPO_ROOT / 'skills' / 'review-all' / 'scripts' / 'program'
FIXTURE_DIR = PROGRAM_DIR / 'fixture'
STATE_PATH = PROGRAM_DIR / 'state.py'
STATE_MODULE_NAME = 'review_all_program_state'
FAKE_STORE_PATH = FIXTURE_DIR / 'fake_store.py'
FAKE_STORE_START_TIMEOUT_SECS = 30


def load_state() -> types.ModuleType:
    """Load ``state.py`` by path, source-authoritative.

    The cached bytecode is unlinked first: the stale-bytecode guard of
    ``test_find_escalation_port.py::_load_find_escalation_port``. The module is
    registered in ``sys.modules`` before it executes, as dataclasses resolving
    their own module expect.
    """
    assert STATE_PATH.is_file(), f'module under test is missing: {STATE_PATH}'
    Path(importlib.util.cache_from_source(str(STATE_PATH))).unlink(missing_ok=True)
    spec = importlib.util.spec_from_file_location(STATE_MODULE_NAME, STATE_PATH)
    assert spec is not None and spec.loader is not None, f'no spec for {STATE_PATH}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[STATE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FakeStore:
    """A running ``fixture/fake_store.py``: its endpoint URL and its request journal."""

    url: str
    journal: Path

    def requests(self) -> list[dict[str, Any]]:
        if not self.journal.exists():
            return []
        return [json.loads(line) for line in self.journal.read_text().splitlines()]


def _endpoint_line(proc: subprocess.Popen[str], stderr_path: Path) -> str:
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], FAKE_STORE_START_TIMEOUT_SECS)
    line = proc.stdout.readline().strip() if ready else ''
    if not line:
        proc.kill()
        proc.wait()
        raise AssertionError(
            f'fake store printed no endpoint (rc={proc.returncode}); stderr:\n'
            f'{stderr_path.read_text()}'
        )
    return line


@contextlib.contextmanager
def running_fake_store(
    *flags: str, store: Path = FIXTURE_DIR / 'store.json', journal: Path | None = None,
) -> Iterator[FakeStore]:
    """Run the fake store as ``-I -S`` from a temp cwd outside the repo, for the block's duration."""
    with tempfile.TemporaryDirectory(prefix='review-all-fake-store-') as scratch:
        cwd = Path(scratch)
        journal_path = journal if journal is not None else cwd / 'journal.jsonl'
        stderr_path = cwd / 'stderr.log'
        command = [
            sys.executable, '-I', '-S', str(FAKE_STORE_PATH), '--port', '0',
            '--store', str(store), '--journal', str(journal_path), *flags,
        ]
        with stderr_path.open('w') as stderr:
            proc = subprocess.Popen(
                command, cwd=cwd, stdout=subprocess.PIPE, stderr=stderr, text=True,
            )
        try:
            yield FakeStore(_endpoint_line(proc, stderr_path), journal_path)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
