#!/usr/bin/env python3
"""Walk the review-all program fixture end to end, every program script a ``-I -S`` subprocess outside the checkout.

Each scenario is one row of ``plans/review-all-program-tooling-prd.md`` §4
"Boundary-test sketch". The walk prints one ``B<n> ok|FAIL <title>: <detail>``
line per row and exits 0 only when every row holds, 1 when one fails, and 2
when it cannot run at all.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_DIR = REPO_ROOT / 'skills' / 'review-all' / 'scripts' / 'program'
FIXTURE_DIR = PROGRAM_DIR / 'fixture'
FIXTURE_RETURNS = FIXTURE_DIR / 'returns'
FIXTURE_PROGRAM = FIXTURE_DIR / 'program.json'
FAKE_STORE = FIXTURE_DIR / 'fake_store.py'
RUN_ID: str = json.loads(FIXTURE_PROGRAM.read_text(encoding='utf-8'))['run_id']

ISOLATION_FLAGS = ('-I', '-S')
ISOLATED_ENV_NAMES = frozenset({'PATH', 'HOME', 'REVIEW_ALL_HOME', 'FUSED_MEMORY_URL'})
SCRIPT_TIMEOUT_SECS = 60
STORE_START_TIMEOUT_SECS = 30
STORE_STOP_GRACE_SECS = 10
DIAGNOSTIC_TAIL_LINES = 20
SCRATCH_PREFIX = 'review-all-fixture-walk-'


class ScenarioFailed(Exception):
    """A scenario's invariant does not hold: the message names it, ``diagnostics`` shows what ran."""

    def __init__(self, invariant: str, diagnostics: str = '') -> None:
        super().__init__(invariant)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class Invocation:
    """One isolated run of a program script, or one fake-store launch (``returncode`` None)."""

    script: Path
    args: tuple[str, ...]
    argv: tuple[str, ...]
    cwd: Path
    env_names: frozenset[str]
    returncode: int | None
    stdout: str
    stderr: str

    def describe(self) -> str:
        return ' '.join((self.script.name, *self.args))

    def diagnostics(self) -> str:
        return '\n'.join((
            f'$ {self.describe()}  (cwd {self.cwd}, exit {self.returncode})',
            '--- stderr tail', _tail(self.stderr),
            '--- stdout tail', _tail(self.stdout),
        ))


def _tail(text: str) -> str:
    return '\n'.join(text.splitlines()[-DIAGNOSTIC_TAIL_LINES:])


class Journal:
    """Every isolated invocation the walk made, in order; B14 reads it back."""

    def __init__(self) -> None:
        self._entries: list[Invocation] = []

    def record(self, invocation: Invocation) -> None:
        self._entries.append(invocation)

    def entries(self) -> tuple[Invocation, ...]:
        return tuple(self._entries)


def expect(condition: bool, invariant: str, invocation: Invocation | None = None) -> None:
    if not condition:
        raise ScenarioFailed(invariant, invocation.diagnostics() if invocation else '')


def expect_exit(invocation: Invocation, code: int, invariant: str) -> Invocation:
    if invocation.returncode != code:
        raise ScenarioFailed(
            f'{invariant}: {invocation.describe()} exited {invocation.returncode}, expected {code}',
            invocation.diagnostics(),
        )
    return invocation


def isolated_env(cwd: Path, homes_root: Path, store_url: str) -> dict[str, str]:
    return {
        'PATH': os.environ.get('PATH', ''),
        'HOME': str(cwd),
        'REVIEW_ALL_HOME': str(homes_root),
        'FUSED_MEMORY_URL': store_url,
    }


def inside_checkout(path: Path) -> bool:
    return path.resolve().is_relative_to(REPO_ROOT)


def _isolated_argv(script: Path, args: Sequence[str]) -> tuple[str, ...]:
    return (sys.executable, *ISOLATION_FLAGS, str(script), *args)


def _refuse_cwd_inside_checkout(cwd: Path) -> None:
    if inside_checkout(cwd):
        raise ScenarioFailed(f'refused: cwd {cwd} is inside the checkout {REPO_ROOT}')


def run_script(
    journal: Journal, script: Path, args: Sequence[str], *, cwd: Path, env: Mapping[str, str],
) -> Invocation:
    """The one place the walk runs a program script: ``-I -S``, from *cwd*, with exactly *env*."""
    _refuse_cwd_inside_checkout(cwd)
    argv = _isolated_argv(script, args)
    try:
        proc = subprocess.run(
            argv, cwd=cwd, env=dict(env), capture_output=True, text=True,
            timeout=SCRIPT_TIMEOUT_SECS,
        )
    except subprocess.TimeoutExpired:
        raise ScenarioFailed(
            f'{script.name} {" ".join(args)} did not exit within {SCRIPT_TIMEOUT_SECS}s',
        ) from None
    invocation = Invocation(
        script, tuple(args), argv, cwd, frozenset(env), proc.returncode, proc.stdout, proc.stderr,
    )
    journal.record(invocation)
    return invocation


def _endpoint_line(proc: subprocess.Popen[str], stderr_path: Path) -> str:
    assert proc.stdout is not None
    ready, _, _ = select.select([proc.stdout], [], [], STORE_START_TIMEOUT_SECS)
    line = proc.stdout.readline().strip() if ready else ''
    if not line:
        raise ScenarioFailed(
            f'the fake store printed no endpoint within {STORE_START_TIMEOUT_SECS}s',
            stderr_path.read_text(encoding='utf-8', errors='replace'),
        )
    return line


def _stop(proc: subprocess.Popen[str]) -> None:
    """Terminate and reap *proc*; a SIGTERM arriving meanwhile is delivered once it is reaped."""
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        proc.terminate()
        try:
            proc.wait(timeout=STORE_STOP_GRACE_SECS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})


@contextlib.contextmanager
def serving(journal: Journal, store: Path, *flags: str, scratch: Path) -> Iterator[str]:
    """Serve *store* with ``fake_store.py`` launched like every script; yield its endpoint URL.

    The store is terminated and reaped on every exit path.
    """
    cwd = Path(tempfile.mkdtemp(prefix='store-', dir=scratch))
    _refuse_cwd_inside_checkout(cwd)
    args = ('--port', '0', '--store', str(store), *flags)
    argv = _isolated_argv(FAKE_STORE, args)
    env = {'PATH': os.environ.get('PATH', ''), 'HOME': str(cwd)}
    stderr_path = cwd / 'stderr.log'
    proc: subprocess.Popen[str] | None = None
    try:
        with stderr_path.open('w', encoding='utf-8') as stderr:
            proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=stderr, text=True)
        url = _endpoint_line(proc, stderr_path)
        journal.record(Invocation(FAKE_STORE, args, argv, cwd, frozenset(env), None, url, ''))
        yield url
    finally:
        if proc is not None:
            _stop(proc)


def edit_store(path: Path, changes: Mapping[int, Mapping[str, Any] | None]) -> None:
    """Edit a store copy per task id: ``metadata`` keys merge, other fields replace, None removes."""
    data = json.loads(path.read_text(encoding='utf-8'))
    unknown = sorted(set(changes) - {task['id'] for task in data['tasks']})
    if unknown:
        raise ValueError(f'{path.name} has no task {unknown}')
    edited = []
    for task in data['tasks']:
        change = changes.get(task['id'], {})
        if change is None:
            continue
        merged = {**task, **{field: value for field, value in change.items() if field != 'metadata'}}
        merged['metadata'] = {**task.get('metadata', {}), **change.get('metadata', {})}
        edited.append(merged)
    path.write_text(json.dumps({**data, 'tasks': edited}, indent=2), encoding='utf-8')


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


@dataclass(frozen=True)
class ProgramRun:
    """One scenario's homes root, cwd and project checkout, all under *root*, and the store it reads."""

    root: Path
    store_url: str
    journal: Journal

    @property
    def homes_root(self) -> Path:
        return self.root / 'homes'

    @property
    def cwd(self) -> Path:
        return self.root / 'cwd'

    @property
    def project(self) -> Path:
        return self.root / 'project'

    @property
    def home(self) -> Path:
        return self.homes_root / RUN_ID

    @property
    def own_store(self) -> Path:
        """The writable store copy an ``own_run`` serves."""
        return self.root / 'store.json'

    def run(self, script_name: str, *args: str | Path) -> Invocation:
        return run_script(
            self.journal, PROGRAM_DIR / script_name, [str(arg) for arg in args],
            cwd=self.cwd, env=isolated_env(self.cwd, self.homes_root, self.store_url),
        )

    def init(self, program_overrides: Mapping[str, Any] | None = None) -> None:
        """``record.py init`` a copy of the fixture program whose project_root is this run's project."""
        program = {**read_json(FIXTURE_PROGRAM), 'project_root': str(self.project), **(program_overrides or {})}
        (self.project / 'plans').mkdir(parents=True, exist_ok=True)
        copy = self.root / 'program.json'
        copy.write_text(json.dumps(program, indent=2), encoding='utf-8')
        expect_exit(self.run('record.py', 'init', copy), 0, 'record.py init accepts the fixture program')

    def put_inbox(self, fixture_return: str, stem: str | None = None) -> Path:
        """Copy a fixture return into the inbox as ``<S>.<pass>[-item].json``, its variant suffix dropped."""
        stem = stem if stem is not None else '.'.join(fixture_return.split('.')[:2])
        target = self.home / 'inbox' / f'{stem}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FIXTURE_RETURNS / fixture_return, target)
        return target

    def applied(self, *fixture_returns: str, amend: bool = False) -> None:
        for fixture_return in fixture_returns:
            inbox = self.put_inbox(fixture_return)
            args = (inbox, '--amend') if amend else (inbox,)
            expect_exit(self.run('apply_return.py', *args), 0, f'{fixture_return} applies')

    def ledger(self) -> list[dict[str, Any]]:
        path = self.home / 'ledger.jsonl'
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]

    def record_copy(self, extra_filings: Iterable[Mapping[str, Any]] = ()) -> Path:
        """The fixture run record, plus *extra_filings*, written at the project's record_path."""
        record = read_json(FIXTURE_DIR / 'record.json')
        record['filings'] = [*record['filings'], *extra_filings]
        target = self.project / read_json(FIXTURE_PROGRAM)['record_path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2), encoding='utf-8')
        return target


def _fresh_run_root(scratch: Path, name: str) -> Path:
    root = Path(tempfile.mkdtemp(prefix=f'{name}-', dir=scratch))
    (root / 'homes').mkdir()
    (root / 'cwd').mkdir()
    return root


@dataclass(frozen=True)
class Walk:
    """One walk: its journal, its scratch root and the shared store over the pristine fixture."""

    journal: Journal
    scratch: Path
    store_url: str

    def new_run(self, name: str) -> ProgramRun:
        return ProgramRun(_fresh_run_root(self.scratch, name), self.store_url, self.journal)

    @contextlib.contextmanager
    def own_run(self, name: str, store_fixture: str, *flags: str) -> Iterator[ProgramRun]:
        """A run on its own fake store, serving a writable copy of *store_fixture* with *flags*."""
        root = _fresh_run_root(self.scratch, name)
        store = root / 'store.json'
        shutil.copyfile(FIXTURE_DIR / store_fixture, store)
        with serving(self.journal, store, *flags, scratch=root) as url:
            yield ProgramRun(root, url, self.journal)


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    check: Callable[[Walk], str]


def _streams_sharing_a_key(program: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    owners: dict[str, list[str]] = {}
    for stream in program['streams']:
        for key in stream['keys']:
            owners.setdefault(key, []).append(stream['id'])
    shared = [(key, tuple(streams)) for key, streams in owners.items() if len(streams) > 1]
    if len(shared) != 1:
        raise ValueError(f'the dupkey fixture should share exactly one key, shares {shared}')
    return shared[0]


def b1_init(walk: Walk) -> str:
    plain = walk.new_run('b1-init')
    expect_exit(plain.run('record.py', 'init', FIXTURE_PROGRAM), 0, 'record.py init accepts the fixture program')
    expect((plain.home / 'program.json').is_file(), f'init writes {plain.home / "program.json"}')

    dupkey_program = FIXTURE_DIR / 'program-dupkey.json'
    key, streams = _streams_sharing_a_key(read_json(dupkey_program))
    dupkey = walk.new_run('b1-dupkey')
    refused = expect_exit(
        dupkey.run('record.py', 'init', dupkey_program), 1, 'record.py init refuses a key two streams share',
    )
    expect(key in refused.stderr, f'the refusal names the shared key {key}', refused)
    expect(all(stream in refused.stderr for stream in streams),
           f'the refusal names both streams {streams}', refused)
    expect(not (dupkey.home / 'program.json').exists(), 'a refused init writes no program.json', refused)
    return f'fixture program initialised; {key} in {"+".join(streams)} refused'


SCENARIOS = (
    Scenario('B1', 'init', b1_init),
)


def _one_line(text: str) -> str:
    return ' '.join(text.split())


def _report(scenario: Scenario, held: bool, detail: str) -> None:
    print(f'{scenario.id} {"ok" if held else "FAIL"} {scenario.title}: {_one_line(detail)}', flush=True)


def _holds(walk: Walk, scenario: Scenario) -> bool:
    try:
        detail = scenario.check(walk)
    except ScenarioFailed as failed:
        _report(scenario, False, f'{type(failed).__name__}: {failed}')
        print(f'--- {scenario.id} {failed}\n{failed.diagnostics}', file=sys.stderr, flush=True)
        return False
    except Exception as broken:
        _report(scenario, False, f'{type(broken).__name__}: {broken}')
        print(f'--- {scenario.id} raised', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return False
    _report(scenario, True, detail)
    return True


def _walk_scenarios(scratch: Path) -> int:
    journal = Journal()
    with contextlib.ExitStack() as stack:
        try:
            store_url = stack.enter_context(serving(journal, FIXTURE_DIR / 'store.json', scratch=scratch))
        except ScenarioFailed as broken:
            print(f'cannot run: the shared fake store did not start: {broken}\n{broken.diagnostics}',
                  file=sys.stderr)
            return 2
        walk = Walk(journal, scratch, store_url)
        started = time.monotonic()
        held = sum(_holds(walk, scenario) for scenario in SCENARIOS)
        elapsed = time.monotonic() - started
    print(f'fixture walk: {held}/{len(SCENARIOS)} scenarios hold in {elapsed:.1f}s', flush=True)
    return 0 if held == len(SCENARIOS) else 1


def _exit_on_sigterm(signum: int, _frame: object) -> None:
    """Unwind like an exception so every fake store and in-flight script is reaped; cleanup runs once."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description=(__doc__ or '').splitlines()[0]).parse_args(argv)
    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    with tempfile.TemporaryDirectory(prefix=SCRATCH_PREFIX) as scratch_text:
        scratch = Path(scratch_text).resolve()
        if inside_checkout(scratch):
            print(f'cannot run: the scratch root {scratch} is inside the checkout {REPO_ROOT}; '
                  'point TMPDIR outside it', file=sys.stderr)
            return 2
        return _walk_scenarios(scratch)


if __name__ == '__main__':
    sys.exit(main())
