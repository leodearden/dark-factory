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
import re
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


@contextlib.contextmanager
def _sigterm_held() -> Iterator[None]:
    signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})


def _release_sigterm() -> None:
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM})


def _stop(proc: subprocess.Popen[str]) -> None:
    with _sigterm_held():
        proc.terminate()
        try:
            proc.wait(timeout=STORE_STOP_GRACE_SECS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


@contextlib.contextmanager
def _owned_child(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], stderr: Any) -> Iterator[subprocess.Popen[str]]:
    """Spawn *argv*, then terminate (if still running) and reap it on every exit path.

    SIGTERM is held across the spawn, so the walk's SystemExit can never land inside
    ``Popen()`` and orphan the child it just forked; the child itself starts with SIGTERM
    released. A SIGTERM held meanwhile is delivered once the child is owned.
    """
    proc: subprocess.Popen[str] | None = None
    try:
        with _sigterm_held():
            proc = subprocess.Popen(
                argv, cwd=cwd, env=dict(env), stdout=subprocess.PIPE, stderr=stderr, text=True,
                preexec_fn=_release_sigterm,
            )
        yield proc
    finally:
        if proc is not None:
            _stop(proc)


def run_script(
    journal: Journal, script: Path, args: Sequence[str], *, cwd: Path, env: Mapping[str, str],
) -> Invocation:
    """The one place the walk runs a program script: ``-I -S``, from *cwd*, with exactly *env*."""
    _refuse_cwd_inside_checkout(cwd)
    argv = _isolated_argv(script, args)
    with _owned_child(argv, cwd=cwd, env=env, stderr=subprocess.PIPE) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=SCRIPT_TIMEOUT_SECS)
        except subprocess.TimeoutExpired:
            raise ScenarioFailed(
                f'{script.name} {" ".join(args)} did not exit within {SCRIPT_TIMEOUT_SECS}s',
            ) from None
    invocation = Invocation(script, tuple(args), argv, cwd, frozenset(env), proc.returncode, stdout, stderr)
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


@contextlib.contextmanager
def serving(journal: Journal, store: Path, *flags: str, scratch: Path) -> Iterator[str]:
    """Serve *store* with ``fake_store.py`` launched like every script; yield its endpoint URL."""
    cwd = Path(tempfile.mkdtemp(prefix='store-', dir=scratch))
    _refuse_cwd_inside_checkout(cwd)
    args = ('--port', '0', '--store', str(store), *flags)
    argv = _isolated_argv(FAKE_STORE, args)
    env = {'PATH': os.environ.get('PATH', ''), 'HOME': str(cwd)}
    stderr_path = cwd / 'stderr.log'
    with stderr_path.open('w', encoding='utf-8') as stderr, \
            _owned_child(argv, cwd=cwd, env=env, stderr=stderr) as proc:
        url = _endpoint_line(proc, stderr_path)
        journal.record(Invocation(FAKE_STORE, args, argv, cwd, frozenset(env), None, url, ''))
        yield url


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


def fixture_return(name: str) -> dict[str, Any]:
    return read_json(FIXTURE_RETURNS / name)


def store_task(task_id: int, store_fixture: str = 'store.json') -> dict[str, Any]:
    (task,) = [task for task in read_json(FIXTURE_DIR / store_fixture)['tasks'] if task['id'] == task_id]
    return task


def check_names(task: Mapping[str, Any]) -> list[str]:
    return [check['name'] for check in task['metadata'].get('delivered_checks', [])]


def snapshot(directory: Path) -> dict[str, bytes]:
    """Every file under *directory*, by relative path."""
    if not directory.exists():
        return {}
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in sorted(directory.rglob('*')) if path.is_file()}


def output(invocation: Invocation) -> str:
    return invocation.stdout + invocation.stderr


def doc_name() -> str:
    return Path(read_json(FIXTURE_PROGRAM)['doc_path']).name


def table_rows(lines: Sequence[str]) -> list[dict[str, str]]:
    """The first markdown table among *lines*, one mapping per body row keyed by its header cells."""
    rows = [[cell.strip() for cell in re.split(r'(?<!\\)\|', line.strip())[1:-1]]
            for line in lines if line.startswith('|')]
    return [dict(zip(rows[0], row, strict=False)) for row in rows[2:]] if len(rows) >= 2 else []


def section(text: str, heading_prefix: str) -> list[str]:
    """The lines after the first ``## `` heading opening with *heading_prefix*, up to the next one."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('## ') and line.startswith(heading_prefix)]
    if not starts:
        raise ScenarioFailed(f'no {heading_prefix!r} heading', text)
    body = lines[starts[0] + 1:]
    ends = [i for i, line in enumerate(body) if line.startswith('## ')]
    return body[:ends[0]] if ends else body


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
    def render(self) -> Path:
        return self.home / 'render'

    @property
    def render_doc(self) -> Path:
        return self.render / doc_name()

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


# store.json's tasks outside the program around S1's touched paths: 201 declares one in
# metadata.files, 202 anchors one in a delivered check, 203's check anchors the path S1.decompose
# moves, and 204's anchors the path it deletes.
SEAM_DECLARER, SEAM_ANCHOR, REPATHED_TASK, DELETED_PATH_TASK = 201, 202, 203, 204


def _candidate_rows(stdout: str) -> dict[int, str]:
    """seam_scan's ``| task <id>: …`` rows, by task id."""
    rows: dict[int, str] = {}
    for line in stdout.splitlines():
        if line.startswith('| task ') and ':' in line:
            task_id = line.removeprefix('| task ').partition(':')[0]
            if task_id.isdigit():
                rows[int(task_id)] = rows.get(int(task_id), '') + line
    return rows


def _expect_author_applied(run: ProgramRun, applied: Invocation) -> str:
    authored = fixture_return('S1.author.json')
    moved = run.home / 'returns' / 'S1.author.json'
    expect(moved.is_file() and moved.read_bytes() == (FIXTURE_RETURNS / 'S1.author.json').read_bytes(),
           'returns/S1.author.json is the fixture return, byte for byte', applied)
    expect(not any((run.home / 'inbox').iterdir()), 'inbox/ is empty after the apply', applied)
    expect(any(line.get('op') == 'stream_status' and line.get('stream') == 'S1'
               and line.get('status') == 'authored' and line.get('prd_path') == authored['prd_path']
               for line in run.ledger()),
           f'the ledger records S1 authored with prd_path {authored["prd_path"]}', applied)
    digest = next(iter(applied.stdout.splitlines()), '')
    expect(digest.startswith('S1 author: ') and digest.endswith('→ returns/S1.author.json'),
           f'the first stdout line is the S1 author digest, not {digest!r}', applied)
    return digest


def _expect_seam_candidates(run: ProgramRun) -> None:
    scanned = expect_exit(run.run('seam_scan.py', 'S1', '--run', RUN_ID), 0, 'seam_scan.py lists S1 candidates')
    rows = _candidate_rows(scanned.stdout)
    expect('metadata.files' in rows.get(SEAM_DECLARER, ''),
           f'seam_scan names task {SEAM_DECLARER}, declaring a touched path in metadata.files', scanned)
    (anchor_check,) = check_names(store_task(SEAM_ANCHOR))
    expect(anchor_check in rows.get(SEAM_ANCHOR, ''),
           f'seam_scan names task {SEAM_ANCHOR}, anchoring a touched path in check {anchor_check}', scanned)


def b2_author_applied(walk: Walk) -> str:
    run = walk.new_run('b2-author')
    run.init()
    applied = expect_exit(run.run('apply_return.py', run.put_inbox('S1.author.json')), 0,
                          'apply_return.py applies S1.author')
    digest = _expect_author_applied(run, applied)
    _expect_seam_candidates(run)
    return f'{digest}; seam_scan lists tasks {SEAM_DECLARER} and {SEAM_ANCHOR}'


B3_REJECTIONS = (
    ('S1.decompose.notes-over-cap.json', 'form: notes:', 'notes'),
    ('S1.decompose.absent-filed.json', 'contradiction: ', 'filed[0].task_id'),
)


def _rejection_lines(invocation: Invocation) -> list[str]:
    return [line for line in output(invocation).splitlines() if line.startswith(('form: ', 'contradiction: '))]


def _expect_rejected(run: ProgramRun, fixture: str, prefix: str, fragment: str) -> None:
    inbox = run.put_inbox(fixture, stem='S1.decompose')
    before = snapshot(run.home)
    checked = expect_exit(run.run('apply_return.py', '--check', inbox), 1, f'--check rejects {fixture}')
    applied = expect_exit(run.run('apply_return.py', inbox), 1, f'apply_return.py rejects {fixture}')
    for invocation in (checked, applied):
        expect(any(line.startswith(prefix) and fragment in line for line in _rejection_lines(invocation)),
               f'{invocation.describe()} names {fragment} on a {prefix!r} line', invocation)
    expect(_rejection_lines(checked) == _rejection_lines(applied),
           f'--check and apply reject {fixture} with the same lines', applied)
    expect(snapshot(run.home) == before, f'rejecting {fixture} moves and appends nothing', applied)


def b3_decompose_rejected(walk: Walk) -> str:
    run = walk.new_run('b3-rejected')
    run.init()
    run.applied('S1.author.json')
    for fixture, prefix, fragment in B3_REJECTIONS:
        _expect_rejected(run, fixture, prefix, fragment)
    return 'form error on notes and store contradiction on filed[0].task_id; --check agrees; home untouched'


def _expect_scan_repaths(run: ProgramRun) -> None:
    scan = expect_exit(run.run('clerk_batch.py', 'write', '--run', RUN_ID, '--scan'), 0,
                       'clerk_batch.py write --scan')
    ledger = run.ledger()
    repaths = {line['task']: line for line in ledger
               if line.get('op') == 'action' and line.get('lead_op') == 'repath'}
    (moved_from,) = [move['from'] for move in fixture_return('S1.decompose.json')['moves'] if move['to']]
    (check,) = check_names(store_task(REPATHED_TASK))
    text = repaths.get(REPATHED_TASK, {}).get('text', '')
    expect(check in text and moved_from in text,
           f'a repath action for task {REPATHED_TASK} names check {check} and {moved_from}', scan)
    deleted_id = repaths.get(DELETED_PATH_TASK, {}).get('id')
    expect(deleted_id is not None and any(
        line.get('op') == 'action_result' and line.get('id') == deleted_id
        and line.get('state') == 'kept-for-human' for line in ledger),
        f'the repath for task {DELETED_PATH_TASK}, whose path the to: null move deletes, is kept for a human',
        scan)


def _expect_held_on_roster(run: ProgramRun) -> str:
    (leaf,) = fixture_return('S1.decompose.json')['held']
    roster = expect_exit(run.run('program_doc.py', '--run', RUN_ID, '--roster'), 0, 'program_doc.py --roster')
    held = [line for line in roster.stdout.splitlines() if line.startswith(f'held {leaf["task_id"]} ')]
    expect(len(held) == 1 and f'release owner {leaf["release_owner"]}' in held[0],
           f'--roster lists held leaf {leaf["task_id"]} with release owner {leaf["release_owner"]}', roster)
    return held[0]


def b4_decompose_applied(walk: Walk) -> str:
    run = walk.new_run('b4-decompose')
    run.init()
    run.applied('S1.author.json')
    decomposed = expect_exit(run.run('apply_return.py', run.put_inbox('S1.decompose.json')), 0,
                             'apply_return.py applies S1.decompose')
    actions = [line for line in run.ledger()
               if line.get('op') == 'action' and line.get('return') == 'returns/S1.decompose.json']
    expect(len(actions) == len(fixture_return('S1.decompose.json')['lead_actions']),
           'the ledger records one action per S1.decompose lead_action', decomposed)
    _expect_scan_repaths(run)
    again = run.run('apply_return.py', run.put_inbox('S1.decompose.json'))
    expect_exit(again, 1, 'a second application of S1.decompose is refused')
    expect('already applied' in output(again), 'the second application is refused as already applied', again)
    held = _expect_held_on_roster(run)
    return f'{len(actions)} actions, scan repaths {REPATHED_TASK} and keeps {DELETED_PATH_TASK} for a human; {held}'


# store.json's S1 α task, moved to done between renders; and its keyless program task, the one
# FILED buckets as unstreamed.
MOVED_TO_DONE, KEYLESS_PROGRAM_TASK = 101, 107


def _decomposed_s1(run: ProgramRun) -> ProgramRun:
    run.init()
    run.applied('S1.author.json', 'S1.decompose.json')
    return run


def _render(run: ProgramRun, *args: str | Path) -> Invocation:
    return run.run('program_doc.py', '--run', RUN_ID, *args)


def _rendered_text(run: ProgramRun, *args: str | Path) -> str:
    expect_exit(_render(run, *args), 0, f'program_doc.py {" ".join(map(str, args))} renders'.strip())
    return run.render_doc.read_text(encoding='utf-8')


def b5_render(walk: Walk) -> str:
    with walk.own_run('b5-render', 'store.json') as run:
        _decomposed_s1(run)
        first = expect_exit(_render(run), 0, 'the first render')
        expect(run.render_doc.is_file() and (run.render / '.state').is_dir(),
               f'the render writes render/{doc_name()} and render/.state/', first)
        before = snapshot(run.render)
        edit_store(run.own_store, {MOVED_TO_DONE: {'status': 'done'}})
        second = expect_exit(_render(run), 0, f'the render after task {MOVED_TO_DONE} moved to done')
        expect(snapshot(run.render) == before, 'a render after the store moved a status is byte-identical', second)
        filed = table_rows(section(run.render_doc.read_text(encoding='utf-8'), '## FILED'))
        unstreamed = [row['Tasks'] for row in filed if row.get('Stream') == 'unstreamed']
        expect(len(unstreamed) == 1 and str(KEYLESS_PROGRAM_TASK) in unstreamed[0].replace(',', ' ').split(),
               f'FILED has an unstreamed row naming task {KEYLESS_PROGRAM_TASK}', second)
    return f'render/ byte-identical after task {MOVED_TO_DONE} moved to done; {KEYLESS_PROGRAM_TASK} unstreamed'


LEDGER_HEADING = '## Ledger'


def _expect_published(run: ProgramRun) -> list[str]:
    published = expect_exit(_render(run, '--publish'), 0, 'program_doc.py --publish')
    program = read_json(FIXTURE_PROGRAM)
    paths = [line.strip() for line in published.stdout.splitlines()
             if line.strip().startswith(f'{program["plans_dir"]}/')]
    snapshots = [path for path in paths if path != program['doc_path']]
    expect(len(paths) == 4 and program['doc_path'] in paths
           and all(Path(path).parent.name.endswith('.state') for path in snapshots),
           'publish prints the three .state snapshot paths and the doc path', published)
    for path in paths:
        counterpart = run.render_doc if path == program['doc_path'] else run.render / '.state' / Path(path).name
        copy = run.project / path
        expect(copy.is_file() and copy.read_bytes() == counterpart.read_bytes(),
               f'{path} is published byte-equal to {counterpart.relative_to(run.home)}', published)
    return paths


def _template_without(heading: str, directory: Path) -> Path:
    lines = (PROGRAM_DIR / 'program_doc_template.md').read_text(encoding='utf-8').splitlines(keepends=True)
    scratch = directory / 'template-without-heading.md'
    scratch.write_text(''.join(line for line in lines if line.rstrip('\n') != heading), encoding='utf-8')
    return scratch


def _expect_heading_self_check(run: ProgramRun) -> None:
    template = _template_without(LEDGER_HEADING, run.root)
    before = snapshot(run.render)
    refused = expect_exit(_render(run, '--template', template), 1,
                          f'a render dropping the published {LEDGER_HEADING} is refused')
    expect(LEDGER_HEADING in refused.stderr, f'the refusal names {LEDGER_HEADING}', refused)
    expect(snapshot(run.render) == before, 'a refused render writes nothing', refused)
    allowed = expect_exit(_render(run, '--template', template, '--allow-drop', LEDGER_HEADING), 0,
                          f'--allow-drop {LEDGER_HEADING!r} lets the render drop it')
    expect(LEDGER_HEADING not in run.render_doc.read_text(encoding='utf-8').splitlines(),
           f'the allowed render has no {LEDGER_HEADING} heading', allowed)


def b6_publish_and_heading_check(walk: Walk) -> str:
    run = _decomposed_s1(walk.new_run('b6-publish'))
    paths = _expect_published(run)
    _expect_heading_self_check(run)
    return f'published {len(paths)} paths; dropping {LEDGER_HEADING} refused, then allowed by --allow-drop'


HANDOFF_HEADINGS = ['## Sources of truth', '## State', '## Inherited duties', '## Remaining work']


def _expect_inherited_duties(text: str, handoff: Invocation) -> None:
    extra = read_json(FIXTURE_DIR / 'record.json')['method']['extra']
    duties = '\n'.join(section(text, '## Inherited duties'))
    expect(extra['launched_from'] in duties and all(seat in duties for seat in extra['fable_seats'])
           and 'launched_from' in duties and 'fable_seats' in duties and '--yes' in duties,
           "Inherited duties carry the record's launched_from, fable_seats and --yes", handoff)


def _expect_remaining_work(run: ProgramRun, text: str, handoff: Invocation) -> None:
    remaining = section(text, '## Remaining work')
    (leaf,) = fixture_return('S1.decompose.json')['held']
    expect(any(line.startswith(f'- held {leaf["task_id"]} ') and f'release owner {leaf["release_owner"]}' in line
               for line in remaining),
           f'Remaining work names held leaf {leaf["task_id"]} and its release owner', handoff)
    ledger = run.ledger()
    resulted = {line['id'] for line in ledger if line.get('op') == 'action_result'}
    for action in (line['id'] for line in ledger if line.get('op') == 'action' and line['id'] not in resulted):
        expect(any(line.startswith(f'- action {action} ') for line in remaining),
               f'Remaining work names unapplied action {action}', handoff)
    expect(any(line.split()[:3] == ['-', 'stream', 'S3'] for line in remaining),
           'Remaining work names stream S3', handoff)


def b8_handoff(walk: Walk) -> str:
    run = _decomposed_s1(walk.new_run('b8-handoff'))
    run.record_copy()
    handoff = expect_exit(_render(run, '--handoff'), 0, 'program_doc.py --handoff')
    text = (run.home / 'HANDOFF.md').read_text(encoding='utf-8')
    headings = [line for line in text.splitlines() if line.startswith('## ')]
    expect(headings == HANDOFF_HEADINGS, f'HANDOFF.md has exactly {HANDOFF_HEADINGS}, not {headings}', handoff)
    _expect_inherited_duties(text, handoff)
    _expect_remaining_work(run, text, handoff)
    return 'HANDOFF.md carries the four sections, the inherited duties and the remaining work'


def _rule(run: ProgramRun, ruling: str, *extra: str) -> Invocation:
    return run.run(
        'record.py', 'rule', '--run', RUN_ID, '--by', 'human', '--question', 'Does alpha or beta own C1?',
        '--ruling', ruling, '--source', 'fixture walk', '--option', 'a', '--option', 'b',
        '--consequence', 'c', *extra,
    )


def b12_rulings(walk: Walk) -> str:
    run = walk.new_run('b12-rulings')
    run.init()
    first = expect_exit(_rule(run, 'alpha owns C1'), 0, 'record.py rule records a ruling')
    expect('ruling D1' in first.stdout, 'the first human ruling is D1', first)
    rulings = run.home / 'rulings.jsonl'
    d1_line = rulings.read_bytes().splitlines()[0]
    second = expect_exit(_rule(run, 'beta owns C1', '--supersedes', 'D1'), 0, 'record.py rule --supersedes D1')
    expect('ruling D2' in second.stdout, 'the superseding ruling is D2', second)
    expect(rulings.read_bytes().splitlines()[0] == d1_line, "D1's line in rulings.jsonl is unchanged", second)
    decisions = section(_rendered_text(run), '## Resolved decisions')
    d2 = [line for line in decisions if line.startswith('- D2 ')]
    expect(len(d2) == 1 and 'supersedes D1' in d2[0], 'Resolved decisions shows D2 naming D1 as superseded', second)
    expect(not [line for line in decisions if line.startswith('- D1 ')], 'Resolved decisions shows no D1 entry', second)
    return 'D2 supersedes D1; D1 line unchanged; only D2 rendered'


SCENARIOS = (
    Scenario('B1', 'init', b1_init),
    Scenario('B2', 'author return applied', b2_author_applied),
    Scenario('B3', 'decompose return rejected, both classes', b3_decompose_rejected),
    Scenario('B4', 'decompose return applied', b4_decompose_applied),
    Scenario('B5', 'render', b5_render),
    Scenario('B6', 'heading self-check and publish', b6_publish_and_heading_check),
    Scenario('B8', 'handoff', b8_handoff),
    Scenario('B12', 'rulings', b12_rulings),
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
