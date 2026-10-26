"""Nightly sweep: redact live secret values from every agent transcript already on disk.

Archive-time redaction (``shared.transcript_archive``) covers what is archived
from now on; this sweep covers what is already on disk: each factory
project's agent-transcript archive and ``~/.claude/projects``, including
archive staging copies a hard kill stranded. It matches the values its OWN
environment holds (the wrapper sources the repo ``.env``), and replaces them
with ``shared.secret_redaction``'s same-length placeholders. It reads only
regular files, and skips those modified within ``min_age_secs``, which a live
session may still be appending to. A sweep that redacted or refused anything
files one ``risk_identified`` info escalation. Nothing it prints or files
carries a value. Operator view: ``OPERATIONS.md`` §12.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import enum
import functools
import json
import os
import stat
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

try:
    import orchestrator_projects
    from dashboard.config import DashboardConfig
    from legibility import census_trigger
    from shared.project_layout import agent_transcripts_dir, default_project_root
    from shared.secret_env import secret_env_values
    from shared.secret_redaction import (
        FileChangedDuringRedaction,
        FileRedaction,
        Needle,
        redact_in_place,
        redaction_needles,
    )
    from shared.transcript_archive import STAGING_SUFFIX
except ImportError as exc:
    print(
        f'transcript_secret_sweep.py: {exc} -- run it as '
        '`uv run --project shared python scripts/transcript_secret_sweep.py`',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

AGENT_ROLE = 'transcript-secret-sweep'
DEFAULT_MIN_AGE_SECS = 900.0
MAX_DETAIL_FILES = 200
NO_SECRETS_CAUSE = 'no secret-bearing values in the environment'
TRANSCRIPT_SUFFIXES = ('.jsonl', '.jsonl' + STAGING_SUFFIX)

PostEscalation = Callable[[dict[str, object]], object]


@dataclass(frozen=True)
class SweepConfig:
    roots: tuple[Path, ...]
    now: datetime.datetime
    min_age_secs: float = DEFAULT_MIN_AGE_SECS


@dataclass(frozen=True)
class RedactedFile:
    path: Path
    counts: Mapping[str, int]


@dataclass(frozen=True)
class RefusedTranscriptLine:
    path: Path
    var: str
    line_number: int


class _Verdict(enum.Enum):
    CLEAN = 'clean'
    DEFERRED = 'deferred'
    UNREADABLE = 'unreadable'
    NOT_A_FILE = 'not-a-file'


@dataclass(frozen=True)
class SweepResult:
    ok: bool
    cause: str | None = None
    redacted: tuple[RedactedFile, ...] = ()
    refused: tuple[RefusedTranscriptLine, ...] = ()
    deferred: tuple[Path, ...] = ()
    unreadable: tuple[Path, ...] = ()
    clean: int = 0
    escalation_error: str | None = None

    def record(self) -> dict[str, object]:
        return {
            'ok': self.ok,
            'cause': self.cause,
            'redacted': [{'path': str(f.path), 'counts': dict(f.counts)} for f in self.redacted],
            'refused': [{'path': str(f.path), 'var': f.var, 'line': f.line_number} for f in self.refused],
            'deferred': [str(path) for path in self.deferred],
            'unreadable': [str(path) for path in self.unreadable],
            'clean': self.clean,
            'escalation_error': self.escalation_error,
        }


def transcript_roots(project_roots: Iterable[Path], claude_projects_dir: Path) -> tuple[Path, ...]:
    return (*(agent_transcripts_dir(root) for root in project_roots), claude_projects_dir)


def run(
    config: SweepConfig,
    *,
    environ: Mapping[str, str],
    post_escalation: PostEscalation,
) -> SweepResult:
    needles = redaction_needles(secret_env_values(environ))
    if not needles:
        return SweepResult(ok=False, cause=NO_SECRETS_CAUSE)
    result = _scan(config, needles)
    if not (result.redacted or result.refused):
        return result
    try:
        post_escalation(_escalation_arguments(result, config.now))
    except Exception as exc:
        return dataclasses.replace(result, escalation_error=f'{type(exc).__name__}: {exc}')
    return result


def _scan(config: SweepConfig, needles: Sequence[Needle]) -> SweepResult:
    cutoff_ns = int((config.now.timestamp() - config.min_age_secs) * 1e9)
    redacted: list[RedactedFile] = []
    refused: list[RefusedTranscriptLine] = []
    deferred: list[Path] = []
    unreadable: list[Path] = []
    clean = 0
    for path in _transcripts(config.roots, unreadable):
        outcome = _sweep_file(path, needles, cutoff_ns)
        match outcome:
            case FileRedaction():
                if outcome.redacted:
                    redacted.append(RedactedFile(path, outcome.counts))
                refused.extend(RefusedTranscriptLine(path, line.var, line.line_number) for line in outcome.refused)
            case _Verdict.CLEAN:
                clean += 1
            case _Verdict.DEFERRED:
                deferred.append(path)
            case _Verdict.UNREADABLE:
                unreadable.append(path)
            case _Verdict.NOT_A_FILE:
                pass
    return SweepResult(
        ok=True,
        redacted=tuple(redacted),
        refused=tuple(refused),
        deferred=tuple(deferred),
        unreadable=tuple(unreadable),
        clean=clean,
    )


def _transcripts(roots: Iterable[Path], unreadable: list[Path]) -> Iterator[Path]:
    def note_unlistable(error: OSError) -> None:
        unreadable.append(Path(error.filename))

    for root in roots:
        if not root.is_dir():
            continue
        for directory, subdirs, files in os.walk(root, onerror=note_unlistable):
            subdirs.sort()
            for name in sorted(files):
                if name.endswith(TRANSCRIPT_SUFFIXES):
                    yield Path(directory) / name


def _sweep_file(path: Path, needles: Sequence[Needle], cutoff_ns: int) -> FileRedaction | _Verdict:
    try:
        st = os.lstat(path)
    except OSError:
        return _Verdict.UNREADABLE
    if not stat.S_ISREG(st.st_mode):
        return _Verdict.NOT_A_FILE
    if st.st_mtime_ns > cutoff_ns:
        return _Verdict.DEFERRED
    try:
        redaction = redact_in_place(path, needles)
    except FileChangedDuringRedaction:
        return _Verdict.DEFERRED
    except OSError:
        return _Verdict.UNREADABLE
    finally:
        _drop_page_cache(path)
    return redaction if redaction.redacted or redaction.refused else _Verdict.CLEAN


def _drop_page_cache(path: Path) -> None:
    if not hasattr(os, 'posix_fadvise'):
        return
    with contextlib.suppress(OSError):
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def _counts_label(counts: Mapping[str, int]) -> str:
    return ', '.join(f'{var}×{n}' for var, n in sorted(counts.items()))


def _bounded(lines: list[str]) -> list[str]:
    if len(lines) <= MAX_DETAIL_FILES:
        return lines
    overflow = len(lines) - MAX_DETAIL_FILES
    return [*lines[:MAX_DETAIL_FILES], f'... and {overflow} more file(s): see the sweep\'s journal record']


def _escalation_arguments(result: SweepResult, now: datetime.datetime) -> dict[str, object]:
    occurrences = sum(sum(f.counts.values()) for f in result.redacted)
    refused_files = len({line.path for line in result.refused})
    summary = (
        f'Transcript secret sweep redacted live secret values in {len(result.redacted)} file(s) '
        f'({occurrences} occurrence(s)); {len(result.refused)} line(s) in {refused_files} file(s) '
        'refused and left un-redacted'
    )
    lines = [
        *(f'{f.path}: {_counts_label(f.counts)}' for f in result.redacted),
        *(f'{f.path}: refused ({f.var} on line {f.line_number}), left un-redacted' for f in result.refused),
    ]
    return {
        'task_id': AGENT_ROLE,
        'agent_role': AGENT_ROLE,
        'category': 'risk_identified',
        'summary': summary,
        'detail': '\n'.join(_bounded(lines)),
        'suggested_action': (
            'These live credentials were on disk in agent transcripts: consider rotating them. '
            'A refused line still holds its value and needs redacting by hand.'
        ),
        'evidence': [{'observation': summary, 'measured_at': now.isoformat(), 'ref': AGENT_ROLE}],
    }


def escalation_url() -> str:
    config = DashboardConfig(project_root=default_project_root())
    return config.escalation_urls[config.project_root.name]


def _post_escalate_info(url: str | None, arguments: dict[str, object]) -> object:
    return census_trigger.post_mcp_tool_call(url or escalation_url(), 'escalate_info', arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Redact live secret values from every agent transcript already on disk.',
        epilog='Exit 0: the sweep ran. Exit 2: it could not run, or its escalation was not filed.',
    )
    orchestrator_projects.add_arguments(parser)
    parser.add_argument(
        '--claude-projects-dir',
        type=Path,
        default=None,
        help='the interactive sessions\' transcript dir (default: ~/.claude/projects)',
    )
    parser.add_argument(
        '--escalation-url',
        default=None,
        help="the escalation server's MCP URL (default: this checkout's configured escalation server)",
    )
    parser.add_argument(
        '--min-age-secs',
        type=float,
        default=DEFAULT_MIN_AGE_SECS,
        help=f'skip files modified more recently than this (default: {DEFAULT_MIN_AGE_SECS:g})',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    targets, unit_scan = orchestrator_projects.targets_from_args(args)
    for problem in unit_scan.problems:
        print(f'transcript_secret_sweep.py: {problem}', file=sys.stderr)
    claude_projects = args.claude_projects_dir or Path.home() / '.claude' / 'projects'
    config = SweepConfig(
        roots=transcript_roots([target.root for target in targets], claude_projects),
        now=datetime.datetime.now(datetime.UTC),
        min_age_secs=args.min_age_secs,
    )
    result = run(
        config,
        environ=os.environ,
        post_escalation=functools.partial(_post_escalate_info, args.escalation_url),
    )
    print(json.dumps(result.record(), sort_keys=True))
    if not result.ok:
        print(f'transcript_secret_sweep.py: could not run: {result.cause}', file=sys.stderr)
        return 2
    if result.escalation_error is not None:
        print(f'transcript_secret_sweep.py: the escalation was not filed: {result.escalation_error}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
