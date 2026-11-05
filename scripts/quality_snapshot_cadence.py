#!/usr/bin/env python3
"""Measure a project's main, or given shas, from a pinned tree into ``<data-dir>/<project_id>/<sha>.json``.

A sha whose file already records the same measurement identity is skipped; nothing
is ever committed. Exit 0 when done; exit 2 on any failure, its cause named on
stderr. Rationale: plans/quality-dashboard-prd.md decisions 18-19.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import quality_metrics_snapshot as snapshot
import source_measures

FACTORY_ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT_SCRIPT = FACTORY_ROOT / 'scripts' / 'quality_metrics_snapshot.py'
DEFAULT_PYTHON = FACTORY_ROOT / '.venv' / 'bin' / 'python'
DEFAULT_DATA_DIR = FACTORY_ROOT / 'data' / 'quality-metrics'

_PROJECT_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')


def _run(command: Sequence[str], *, capture: bool) -> subprocess.CompletedProcess[str]:
    """*command*'s completed process, whatever its exit code; one that cannot start is refused naming it."""
    try:
        return subprocess.run(command, capture_output=capture, text=True, check=False)
    except OSError as exc:
        raise source_measures.MetricsError(
            f'could not run `{shlex.join(command)}`: {exc.__class__.__name__}: {exc}'
        ) from exc


def _stdout(command: Sequence[str]) -> str:
    """*command*'s stdout; a non-zero exit is refused naming the command, its code and its stderr."""
    completed = _run(command, capture=True)
    if completed.returncode != 0:
        raise source_measures.MetricsError(
            f'`{shlex.join(command)}` exited {completed.returncode}: '
            f'{completed.stderr.strip() or "no stderr"}'
        )
    return completed.stdout


def _git(root: Path, *args: str) -> str:
    return _stdout(['git', '-C', str(root), *args])


def _worktree(root: Path, *args: str) -> str:
    """``git worktree *args`` with the project's own hooks off: pinning a tree runs none of them."""
    return _git(root, '-c', 'core.hooksPath=/dev/null', 'worktree', *args)


def _full_sha(root: Path, rev: str) -> str:
    return _git(root, 'rev-parse', '--verify', '--end-of-options', f'{rev}^{{commit}}').strip()


def _shas(root: Path, requested: Sequence[str] | None) -> list[str]:
    return [_full_sha(root, rev) for rev in requested or ['main']]


def _state_home(project_id: str) -> Path:
    """Where pinned trees live: the factory's XDG state-home convention, read at call time."""
    base = os.environ.get('XDG_STATE_HOME')
    if not base:
        try:
            base = str(Path.home() / '.local' / 'state')
        except RuntimeError as exc:
            raise source_measures.MetricsError(
                f'no home directory to hold the pinned tree ({exc}); set XDG_STATE_HOME'
            ) from exc
    return Path(base) / 'dark-factory' / 'quality-snapshot' / project_id


def _predicted_params(python: Path) -> dict[str, Any]:
    """The params a measurement by *python* would record, asked of the instrument itself."""
    command = [str(python), str(INSTRUMENT_SCRIPT), '--params']
    printed = _stdout(command)
    try:
        params = json.loads(printed)
    except json.JSONDecodeError as exc:
        raise source_measures.MetricsError(
            f'`{shlex.join(command)}` printed no JSON params: {exc}'
        ) from exc
    if not isinstance(params, dict):
        raise source_measures.MetricsError(
            f'`{shlex.join(command)}` printed {printed.strip()}, not a JSON object'
        )
    return params


def _added_worktree(root: Path, sha: str, home: Path) -> Path:
    """A detached worktree of *sha* in a fresh directory under *home*; the directory goes if git refuses."""
    try:
        home.mkdir(parents=True, exist_ok=True)
        tree = Path(tempfile.mkdtemp(prefix=f'{sha[:12]}-', dir=home))
    except OSError as exc:
        raise source_measures.MetricsError(
            f'{home}: no directory could be made there for the pinned tree -- '
            f'{exc.__class__.__name__}: {exc}'
        ) from exc
    try:
        _worktree(root, 'add', '--detach', str(tree), sha)
    except source_measures.MetricsError as refusal:
        _discard_refused_tree(root, tree, refusal)
        raise
    return tree


def _discard_refused_tree(root: Path, tree: Path, refusal: source_measures.MetricsError) -> None:
    """Remove what a refused add left at *tree*, registered or not; a tree that stays is named, chained to *refusal*."""
    try:
        _worktree(root, 'remove', '--force', str(tree))
    except source_measures.MetricsError:
        try:
            shutil.rmtree(tree)
        except OSError as exc:
            raise source_measures.MetricsError(
                f'{tree}: the pinned tree a refused `git worktree add` left could not be removed -- '
                f'{exc.__class__.__name__}: {exc}'
            ) from refusal


def _remove_worktree(root: Path, tree: Path, *, after: BaseException | None) -> None:
    """Remove *tree*; a failure names the tree left behind, chained to the failure *after* which it ran."""
    try:
        _worktree(root, 'remove', '--force', str(tree))
    except source_measures.MetricsError as removal:
        if after is None:
            raise
        raise removal from after


@contextlib.contextmanager
def _pinned(root: Path, sha: str, home: Path) -> Iterator[Path]:
    """A detached worktree of *sha* under *home*, removed on exit however the body ends."""
    tree = _added_worktree(root, sha, home)
    try:
        yield tree
    except BaseException as failure:
        _remove_worktree(root, tree, after=failure)
        raise
    _remove_worktree(root, tree, after=None)


def _measure(python: Path, sha: str, tree: Path, out: Path, run_id: str) -> None:
    """Run the instrument on *tree*, its stdout and stderr this process's own."""
    command = [str(python), str(INSTRUMENT_SCRIPT), '--run-id', run_id, '--out', str(out), '--root', str(tree)]
    returncode = _run(command, capture=False).returncode
    if returncode != 0:
        raise source_measures.MetricsError(
            f'the instrument exited {returncode} measuring {sha} in the pinned tree {tree}; '
            'its stderr above names the cause'
        )


@dataclasses.dataclass(frozen=True)
class _Settings:
    """The command line, resolved: whose shas, measured by which interpreter, into where."""

    root: Path
    project_id: str
    python: Path
    data_dir: Path
    force: bool


def _utc_stamp() -> str:
    """Now to the microsecond, so two runs of one sha never share a run_id."""
    return datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S.%fZ')


def _already_measured(out: Path, wanted: snapshot.MeasurementIdentity) -> bool:
    """Whether *out* holds a measurement at *wanted*; one whose schema predates the identity holds none."""
    if not out.exists():
        return False
    existing = snapshot.load_snapshot(out)
    return snapshot.identity_known(existing) and snapshot.measurement_identity(existing) == wanted


def _cadence_one(settings: _Settings, sha: str, params: Mapping[str, Any]) -> str:
    """Measure *sha* unless its file already records this identity; the line to print."""
    out = settings.data_dir / settings.project_id / f'{sha}.json'
    wanted = snapshot.MeasurementIdentity.of(sha, params)
    if not settings.force and _already_measured(out, wanted):
        return f'already measured: {wanted}'
    run_id = f'quality-snapshot-{settings.project_id}-{sha[:12]}-{_utc_stamp()}'
    with _pinned(settings.root, sha, _state_home(settings.project_id)) as tree:
        _measure(settings.python, sha, tree, out, run_id)
    recorded = snapshot.measurement_identity(snapshot.load_snapshot(out))
    if recorded != wanted:
        raise source_measures.MetricsError(
            f'{out}: the measurement recorded the identity {recorded}, not the '
            f'predicted {wanted}; the instrument or its interpreter changed mid-run'
        )
    return f'measured: {sha} -> {out}'


def _project_id(value: str) -> str:
    if not _PROJECT_ID.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f'{value!r} is not a project id: it names a directory, so letters, digits and _ . - '
            'only, starting with a letter or digit'
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure a project's main, or the given shas, with the factory's snapshot "
            'instrument in a pinned worktree, skipping a sha already measured at the same '
            'identity (as_of_sha, params). Never commits.'
        ),
        epilog='Exit codes: 0 done; 2 any failure, its cause named on stderr.',
    )
    parser.add_argument('--root', required=True, type=Path, help='The project checkout to measure.')
    parser.add_argument(
        '--project-id', required=True, type=_project_id, help='The data directory to write under.'
    )
    parser.add_argument('--sha', nargs='+', help='Measure these commits, in order, instead of main.')
    parser.add_argument(
        '--force', action='store_true', help='Re-measure even a sha already measured at this identity.'
    )
    parser.add_argument(
        '--data-dir', type=Path, default=DEFAULT_DATA_DIR, help=f'Default: {DEFAULT_DATA_DIR}.'
    )
    parser.add_argument(
        '--python',
        type=Path,
        default=DEFAULT_PYTHON,
        help=f'The interpreter that measures (default: {DEFAULT_PYTHON}).',
    )
    return parser


def _require_executable(python: Path) -> None:
    if not (python.is_file() and os.access(python, os.X_OK)):
        raise source_measures.MetricsError(
            f'{python}: no executable interpreter to measure with; run `uv sync --all-packages` '
            f'in {FACTORY_ROOT}, or pass --python'
        )


def _failure_lines(failure: source_measures.MetricsError) -> list[str]:
    """*failure*, then each earlier failure it followed; a cause of another type is already in its text."""
    lines = [f'quality_snapshot_cadence: {failure}']
    cause = failure.__cause__
    while isinstance(cause, source_measures.MetricsError):
        lines.append(f'  following: {cause}')
        cause = cause.__cause__
    return lines


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; see the module docstring for the exit codes."""
    args = _build_parser().parse_args(argv)
    settings = _Settings(args.root, args.project_id, args.python, args.data_dir, args.force)
    try:
        _require_executable(settings.python)
        shas = _shas(settings.root, args.sha)
        params = _predicted_params(settings.python)
        for sha in shas:
            print(_cadence_one(settings, sha, params), flush=True)
    except source_measures.MetricsError as exc:
        print(*_failure_lines(exc), sep='\n', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
