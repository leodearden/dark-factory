"""Nightly sweep: redact live secret values from agent transcripts and test logs on disk.

Archive-time redaction (``shared.transcript_archive``) covers transcripts
archived from now on; this sweep covers what is already on disk. For each
factory project it walks the agent-transcript archive (``*.jsonl``, including
archive staging copies a hard kill stranded), the verify logs and the
offline-lane logs (``LOG_SUFFIXES``: ``*.log``, ``*.json``, junit ``*.xml``
and gzipped junit ``*.xml.gz``), and it walks ``~/.claude/projects`` for
transcripts. A ``*.log`` is matched for the raw value as well as its
JSON-escaped form; transcripts and ``*.json`` for the JSON-escaped form only.
Gzipped junit is redacted by decompressing, replacing and
recompressing it, and XML gets an entity-escaped placeholder so a report still
parses; a ``.gz`` that does not decompress is reported unreadable and left as
found. It matches the values its OWN environment holds (the wrapper sources
the repo ``.env``), and replaces them with ``shared.secret_redaction``'s
same-length placeholders. It reads only regular files, and skips those
modified within ``min_age_secs``, which a live session or test run may still
be appending to. A sweep that redacted or refused anything files one
``risk_identified`` info escalation, and a sweep that could not run, found a
required root absent or could not read an orchestrator unit files one
``infra_issue`` info escalation naming each shortfall. Nothing it prints or
files carries a value. Operator view: ``OPERATIONS.md`` §12.
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
    from shared.mcp_post import mcp_server_base
    from shared.mcp_session import SyncMcpSession
    from shared.project_layout import (
        agent_transcripts_dir,
        default_project_root,
        offline_lane_dir,
        verify_logs_dir,
    )
    from shared.secret_env import secret_env_values
    from shared.secret_redaction import (
        Codec,
        CorruptCompressedFile,
        FileChangedDuringRedaction,
        FileRedaction,
        Markup,
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
ESCALATION_TIMEOUT_SECS = 30.0
MAX_DETAIL_FILES = 200
NO_SECRETS_CAUSE = 'no secret-bearing values in the environment'
SHORTFALL_TASK_ID = 'transcript-secret-sweep-shortfall'
TRANSCRIPT_SUFFIXES = ('.jsonl', '.jsonl' + STAGING_SUFFIX)
LOG_SUFFIXES = ('.log', '.json', '.xml', '.xml.gz')

_FORMATS: tuple[tuple[str, Markup, Codec], ...] = (
    ('.xml.gz', Markup.XML, Codec.GZIP),
    ('.xml', Markup.XML, Codec.PLAIN),
    ('.log', Markup.TEXT, Codec.PLAIN),
)

PostEscalation = Callable[[dict[str, object]], object]
NeedlesByMarkup = Mapping[Markup, Sequence[Needle]]


@dataclass(frozen=True)
class SweptTree:
    root: Path
    suffixes: tuple[str, ...]
    required: bool


@dataclass(frozen=True)
class SweepConfig:
    trees: tuple[SweptTree, ...]
    now: datetime.datetime
    min_age_secs: float = DEFAULT_MIN_AGE_SECS
    unswept: tuple[str, ...] = ()


@dataclass(frozen=True)
class RedactedFile:
    path: Path
    counts: Mapping[str, int]


@dataclass(frozen=True)
class RefusedFileLine:
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
    refused: tuple[RefusedFileLine, ...] = ()
    deferred: tuple[Path, ...] = ()
    unreadable: tuple[Path, ...] = ()
    clean: int = 0
    missing: tuple[Path, ...] = ()
    unswept: tuple[str, ...] = ()
    escalation_error: str | None = None

    @property
    def shortfall(self) -> tuple[str, ...]:
        return (
            *(() if self.ok else (f'could not run: {self.cause}',)),
            *(f'{path}: required root does not exist' for path in self.missing),
            *self.unswept,
        )

    def record(self) -> dict[str, object]:
        return {
            'ok': self.ok,
            'cause': self.cause,
            'redacted': [{'path': str(f.path), 'counts': dict(f.counts)} for f in self.redacted],
            'refused': [{'path': str(f.path), 'var': f.var, 'line': f.line_number} for f in self.refused],
            'deferred': [str(path) for path in self.deferred],
            'unreadable': [str(path) for path in self.unreadable],
            'clean': self.clean,
            'missing': [str(path) for path in self.missing],
            'unswept': list(self.unswept),
            'escalation_error': self.escalation_error,
        }


def swept_trees(project_roots: Iterable[Path], claude_projects_dir: Path) -> tuple[SweptTree, ...]:
    per_project = (
        tree
        for root in project_roots
        for tree in (
            SweptTree(agent_transcripts_dir(root), TRANSCRIPT_SUFFIXES, required=True),
            SweptTree(verify_logs_dir(root), LOG_SUFFIXES, required=False),
            SweptTree(offline_lane_dir(root), LOG_SUFFIXES, required=False),
        )
    )
    return (*per_project, SweptTree(claude_projects_dir, TRANSCRIPT_SUFFIXES, required=True))


def run(
    config: SweepConfig,
    *,
    environ: Mapping[str, str],
    post_escalation: PostEscalation,
) -> SweepResult:
    result = _sweep(config, environ)
    errors = [
        error
        for arguments in _escalations(result, config.now)
        if (error := _posting_error(post_escalation, arguments)) is not None
    ]
    return dataclasses.replace(result, escalation_error='; '.join(errors) or None)


def _sweep(config: SweepConfig, environ: Mapping[str, str]) -> SweepResult:
    secrets = secret_env_values(environ)
    needles = {markup: redaction_needles(secrets, markup=markup) for markup in Markup}
    result = _scan(config, needles) if needles[Markup.JSON] else SweepResult(ok=False, cause=NO_SECRETS_CAUSE)
    missing = tuple(tree.root for tree in config.trees if tree.required and not tree.root.is_dir())
    return dataclasses.replace(result, missing=missing, unswept=config.unswept)


def _escalations(result: SweepResult, now: datetime.datetime) -> list[dict[str, object]]:
    findings = [_escalation_arguments(result, now)] if result.redacted or result.refused else []
    shortfalls = [_shortfall_arguments(result, now)] if result.shortfall else []
    return [*findings, *shortfalls]


def _posting_error(post_escalation: PostEscalation, arguments: dict[str, object]) -> str | None:
    try:
        post_escalation(arguments)
    except Exception as exc:
        return f'{type(exc).__name__}: {exc}'
    return None


def _scan(config: SweepConfig, needles: NeedlesByMarkup) -> SweepResult:
    cutoff_ns = int((config.now.timestamp() - config.min_age_secs) * 1e9)
    redacted: list[RedactedFile] = []
    refused: list[RefusedFileLine] = []
    deferred: list[Path] = []
    unreadable: list[Path] = []
    clean = 0
    for path in _swept_files(config.trees, unreadable):
        outcome = _sweep_file(path, needles, cutoff_ns)
        match outcome:
            case FileRedaction():
                if outcome.redacted:
                    redacted.append(RedactedFile(path, outcome.counts))
                refused.extend(RefusedFileLine(path, line.var, line.line_number) for line in outcome.refused)
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


def _swept_files(trees: Iterable[SweptTree], unreadable: list[Path]) -> Iterator[Path]:
    def note_unlistable(error: OSError) -> None:
        unreadable.append(Path(error.filename))

    for tree in trees:
        if not tree.root.is_dir():
            continue
        for directory, subdirs, files in os.walk(tree.root, onerror=note_unlistable):
            subdirs.sort()
            for name in sorted(files):
                if name.endswith(tree.suffixes):
                    yield Path(directory) / name


def _format_of(name: str) -> tuple[Markup, Codec]:
    for suffix, markup, codec in _FORMATS:
        if name.endswith(suffix):
            return markup, codec
    return Markup.JSON, Codec.PLAIN


def _sweep_file(path: Path, needles: NeedlesByMarkup, cutoff_ns: int) -> FileRedaction | _Verdict:
    try:
        st = os.lstat(path)
    except OSError:
        return _Verdict.UNREADABLE
    if not stat.S_ISREG(st.st_mode):
        return _Verdict.NOT_A_FILE
    if st.st_mtime_ns > cutoff_ns:
        return _Verdict.DEFERRED
    markup, codec = _format_of(path.name)
    try:
        redaction = redact_in_place(path, needles[markup], codec=codec)
    except FileChangedDuringRedaction:
        return _Verdict.DEFERRED
    except (CorruptCompressedFile, OSError):
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
            'These live credentials were on disk in agent transcripts or verify logs: consider rotating them. '
            'A refused line still holds its value and needs redacting by hand.'
        ),
        'evidence': [{'observation': summary, 'measured_at': now.isoformat(), 'ref': AGENT_ROLE}],
    }


def _shortfall_arguments(result: SweepResult, now: datetime.datetime) -> dict[str, object]:
    summary = f'Transcript secret sweep did not cover everything: {len(result.shortfall)} shortfall(s)'
    return {
        'task_id': SHORTFALL_TASK_ID,
        'agent_role': AGENT_ROLE,
        'category': 'infra_issue',
        'summary': summary,
        'detail': '\n'.join(_bounded(list(result.shortfall))),
        'suggested_action': (
            'Check that $REPO/.env exists and still defines the secret-bearing variables, that each '
            "project's transcript archive exists at its default root, and that the orchestrator unit "
            'files are readable.'
        ),
        'evidence': [{'observation': summary, 'measured_at': now.isoformat(), 'ref': SHORTFALL_TASK_ID}],
    }


def escalation_url() -> str:
    config = DashboardConfig(project_root=default_project_root())
    return config.escalation_urls[config.project_root.name]


def _post_escalate_info(url: str | None, arguments: dict[str, object]) -> object:
    base_url = mcp_server_base(url or escalation_url())
    with SyncMcpSession(base_url, client_name='transcript-secret-sweep', timeout=ESCALATION_TIMEOUT_SECS) as session:
        return session.call_tool('escalate_info', arguments)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Redact live secret values from agent transcripts, verify logs and offline-lane logs '
            'already on disk.'
        ),
        epilog=(
            'Exit 0: the sweep covered every required root. Exit 2: it could not run or could not cover '
            'a required root (it files an infra_issue escalation), or an escalation was not filed.'
        ),
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
    claude_projects = args.claude_projects_dir or Path.home() / '.claude' / 'projects'
    config = SweepConfig(
        trees=swept_trees([target.root for target in targets], claude_projects),
        now=datetime.datetime.now(datetime.UTC),
        min_age_secs=args.min_age_secs,
        unswept=unit_scan.problems,
    )
    result = run(
        config,
        environ=os.environ,
        post_escalation=functools.partial(_post_escalate_info, args.escalation_url),
    )
    print(json.dumps(result.record(), sort_keys=True))
    for line in result.shortfall:
        print(f'transcript_secret_sweep.py: did not cover: {line}', file=sys.stderr)
    if result.escalation_error is not None:
        print(f'transcript_secret_sweep.py: the escalation was not filed: {result.escalation_error}', file=sys.stderr)
    return 2 if result.shortfall or result.escalation_error is not None else 0


if __name__ == '__main__':
    raise SystemExit(main())
