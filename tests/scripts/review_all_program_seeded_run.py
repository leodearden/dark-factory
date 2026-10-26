"""A seeded review-all run home and a writable task-store copy, and a program script run over both.

A non-test helper (importable through ``tests/scripts/conftest.py``'s ``sys.path``
insert) for the gate_check and seam_scan suites. The home carries the fixture
``program.json``, one ``stream_status authored`` ledger line per seeded stream,
and the fixture author returns asked for. The store is a temp copy of a fixture
store served by ``fixture/fake_store.py``, which re-reads it on every request,
so an edit between two runs is the store after a correction. Every script runs
``-I -S`` from a temp cwd outside the repo, seeing only ``PATH``, ``HOME``,
``REVIEW_ALL_HOME`` and ``FUSED_MEMORY_URL``. Fixture files are copied, never
edited.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_all_program_fixture import FIXTURE_DIR, PROGRAM_DIR, load_state, running_fake_store

state = load_state()

RUN_ID = 'review-all-fixture-20261019'
FIXTURE_HOME = state.Home(FIXTURE_DIR)
SCRIPT_TIMEOUT_SECS = 60
SEEDED_AT = '2026-10-19T12:00:00Z'
SEEDED_BY = 'lead'

TaskChange = Mapping[str, Any] | None
"""Field changes for one store task: ``metadata`` keys merge into its metadata, any other field
is replaced, and None removes the task."""


def _edited(task: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {**task['metadata'], **change.get('metadata', {})}
    return {**task, **change, 'metadata': metadata}


@dataclass(frozen=True)
class StoreCopy:
    """A writable copy of one fixture store."""

    path: Path

    @classmethod
    def of(cls, fixture_name: str, directory: Path) -> StoreCopy:
        path = directory / fixture_name
        shutil.copyfile(FIXTURE_DIR / fixture_name, path)
        return cls(path)

    def _read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text())

    def _write_tasks(self, tasks: Sequence[Mapping[str, Any]]) -> None:
        self.path.write_text(json.dumps({**self._read(), 'tasks': list(tasks)}, indent=2))

    def task_ids(self) -> frozenset[int]:
        return frozenset(task['id'] for task in self._read()['tasks'])

    def edit(self, changes: Mapping[int, TaskChange]) -> None:
        """Apply each task's change in place; naming a task the copy lacks is a test bug."""
        unknown = sorted(set(changes) - self.task_ids())
        assert not unknown, f'{self.path.name} has no task {unknown}'
        edited = []
        for task in self._read()['tasks']:
            change = changes.get(task['id'], {})
            if change is not None:
                edited.append(_edited(task, change))
        self._write_tasks(edited)

    def keep_only(self, task_ids: Collection[int]) -> None:
        self.edit({task_id: None for task_id in self.task_ids() - set(task_ids)})


def _authored_line(seq: int, stream: str) -> dict[str, Any]:
    fixture_return = FIXTURE_HOME.returns_file(stream, state.Pass.AUTHOR)
    return {
        'seq': seq, 'at': SEEDED_AT, 'by': SEEDED_BY, 'op': str(state.LedgerOp.STREAM_STATUS),
        'stream': stream, 'status': str(state.StreamStatus.AUTHORED),
        'prd_path': json.loads(fixture_return.read_text())['prd_path'],
    }


def _seed_home(homes_root: Path, authored: Sequence[str], returns: Sequence[str]) -> None:
    home = state.run_home(RUN_ID, {'REVIEW_ALL_HOME': str(homes_root)})
    home.root.mkdir(parents=True)
    shutil.copyfile(FIXTURE_HOME.program_json, home.program_json)
    state.append_ledger(home, [_authored_line(seq, s) for seq, s in enumerate(authored, 1)])
    for stream in returns:
        target = home.returns_file(stream, state.Pass.AUTHOR)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURE_HOME.returns_file(stream, state.Pass.AUTHOR), target)


@dataclass(frozen=True)
class SeededRun:
    """One test's homes root, its cwd outside the repo, its store copy and the fake store's URL."""

    homes_root: Path
    cwd: Path
    store: StoreCopy
    store_url: str

    @property
    def home(self) -> Any:
        """The seeded run's ``state.Home``."""
        return state.run_home(RUN_ID, {'REVIEW_ALL_HOME': str(self.homes_root)})

    def run(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, '-I', '-S', str(PROGRAM_DIR / script), *args],
            cwd=self.cwd, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_SECS,
            env={
                'PATH': os.environ.get('PATH', ''),
                'HOME': str(self.cwd),
                'REVIEW_ALL_HOME': str(self.homes_root),
                'FUSED_MEMORY_URL': self.store_url,
            },
        )


@contextlib.contextmanager
def seeded_run(
    scratch: Path, *, store: str = 'store.json', authored: Sequence[str] = ('S1', 'S2'),
    returns: Sequence[str] = (), store_flags: Sequence[str] = (),
) -> Iterator[SeededRun]:
    """A home seeded under *scratch* with *authored* streams and *returns*, over a copy of *store*.

    *store_flags* go to the fake store (``--no-include-rows`` for the INV-11 cases).
    """
    homes_root, cwd = scratch / 'homes', scratch / 'cwd'
    cwd.mkdir(parents=True)
    _seed_home(homes_root, authored, returns)
    copy = StoreCopy.of(store, scratch)
    with running_fake_store(*store_flags, store=copy.path) as fake:
        yield SeededRun(homes_root, cwd, copy, fake.url)
