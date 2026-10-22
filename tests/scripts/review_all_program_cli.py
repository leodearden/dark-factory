"""Drive the review-all program scripts the way a run does: ``-I -S`` subprocesses outside the repo.

A non-test helper (importable through ``tests/scripts/conftest.py``'s ``sys.path``
insert) shared by the program-script suites. ``ProgramEnv`` gives each test a
temp homes root and a temp cwd; every script sees them through
``REVIEW_ALL_HOME`` and ``HOME``, and the store through ``FUSED_MEMORY_URL``.
Fixture files are copied, never edited.
"""
from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from review_all_program_fixture import FIXTURE_DIR, PROGRAM_DIR

RUN_ID = 'review-all-fixture-20261019'
FIXTURE_RETURNS = FIXTURE_DIR / 'returns'
SCRIPT_TIMEOUT_SECS = 60


@dataclass(frozen=True)
class ProgramEnv:
    """One test's homes root, its cwd outside the repo, and the store the scripts read."""

    homes_root: Path
    cwd: Path
    store_url: str

    @classmethod
    def under(cls, scratch: Path, store_url: str) -> ProgramEnv:
        env = cls(scratch / 'homes', scratch / 'cwd', store_url)
        env.homes_root.mkdir(parents=True)
        env.cwd.mkdir(parents=True)
        return env

    @property
    def home(self) -> Path:
        return self.homes_root / RUN_ID

    def run(self, script: str, *args: str | Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, '-I', '-S', str(PROGRAM_DIR / script), *map(str, args)],
            cwd=self.cwd, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_SECS,
            env={
                'PATH': os.environ.get('PATH', ''),
                'HOME': str(self.cwd),
                'REVIEW_ALL_HOME': str(self.homes_root),
                'FUSED_MEMORY_URL': self.store_url,
            },
        )

    def put_inbox(self, fixture_return_name: str, as_stem: str | None = None) -> Path:
        """Copy a fixture return into the inbox as ``<S>.<pass>[-item].json`` (its variant dropped)."""
        stem = as_stem if as_stem is not None else '.'.join(fixture_return_name.split('.')[:2])
        target = self.home / 'inbox' / f'{stem}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURE_RETURNS / fixture_return_name, target)
        return target

    def snapshot(self) -> dict[str, bytes]:
        """Every file under the home, by home-relative path."""
        if not self.home.exists():
            return {}
        return {
            str(path.relative_to(self.home)): path.read_bytes()
            for path in sorted(self.home.rglob('*')) if path.is_file()
        }


def load_program_module(name: str) -> types.ModuleType:
    """Import a program module by name, for in-process tests of its public API."""
    if str(PROGRAM_DIR) not in sys.path:
        sys.path.insert(0, str(PROGRAM_DIR))
    return importlib.import_module(name)
