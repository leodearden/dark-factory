#!/usr/bin/env python3
"""Raise one escalation for a failed quality-snapshot@<project_id> run; systemd's OnFailure= runs it.

Units: scripts/quality-snapshot@.service, scripts/quality-snapshot-alarm@.service;
operator view: OPERATIONS.md §"Nightly quality snapshot (03:45)". Exit 0 when the
escalation landed, 2 when it was not posted.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import TypedDict

import _checkout as _checkout
from dashboard.config import DashboardConfig
from shared.mcp_post import mcp_server_base
from shared.mcp_session import SyncMcpSession
from shared.project_layout import default_project_root

AGENT_ROLE = 'quality-snapshot-alarm'
ENV_FILE_DISPLAY = '%E/dark-factory/quality-snapshot/{project_id}.env'
EXIT_POSTED = 0
EXIT_NOT_POSTED = 2
JOURNAL_LINES = 40
JOURNAL_TIMEOUT_SECS = 30.0
POST_TIMEOUT_SECS = 30.0
UNKNOWN = 'unknown'
_LOG_PREFIX = 'quality_snapshot_alarm:'

Poster = Callable[[str, Mapping[str, object]], object]
"""``poster(base_url, arguments)`` delivers one escalate_info and raises on failure."""


class Evidence(TypedDict):
    observation: str
    measured_at: str
    ref: str


class EscalationArguments(TypedDict):
    """The escalate_info arguments one failed run files."""

    task_id: str
    agent_role: str
    category: str
    severity: str
    summary: str
    detail: str
    evidence: list[Evidence]


class NoEscalationServer(LookupError):
    """The factory checkout configures no escalation server to post to."""


@dataclasses.dataclass(frozen=True)
class FailedRun:
    """What systemd told this handler about the snapshot run that failed."""

    project_id: str
    unit: str
    service_result: str
    exit_code: str
    exit_status: str
    invocation_id: str | None
    project_root: str | None

    @classmethod
    def from_environment(cls, project_id: str, environ: Mapping[str, str]) -> FailedRun:
        """The MONITOR_* facts systemd hands an OnFailure= unit, and the instance's root; absent ones read unknown."""
        return cls(
            project_id=project_id,
            unit=environ.get('MONITOR_UNIT') or f'quality-snapshot@{project_id}.service',
            service_result=environ.get('MONITOR_SERVICE_RESULT') or UNKNOWN,
            exit_code=environ.get('MONITOR_EXIT_CODE') or UNKNOWN,
            exit_status=environ.get('MONITOR_EXIT_STATUS') or UNKNOWN,
            invocation_id=environ.get('MONITOR_INVOCATION_ID') or None,
            project_root=environ.get('QUALITY_SNAPSHOT_ROOT') or None,
        )


def journal_tail(unit: str, *, lines: int = JOURNAL_LINES) -> str:
    """The newest *lines* of *unit*'s user journal; a journal that cannot be read is named, never raised.

    Matched by unit rather than by invocation id: the user journal's invocation
    field differs across systemd versions, and the failed run is the newest.
    """
    command = ['journalctl', '--user', '-u', unit, '-o', 'cat', '--no-pager', '-n', str(lines)]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=JOURNAL_TIMEOUT_SECS, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f'(journal unreadable: {exc.__class__.__name__}: {exc})'
    if completed.returncode != 0:
        return (
            f'(journal unreadable: `{shlex.join(command)}` exited {completed.returncode}: '
            f'{completed.stderr.strip() or "no stderr"})'
        )
    return completed.stdout


def _summary(run: FailedRun) -> str:
    return (
        f'quality snapshot of {run.project_id} failed: {run.unit} result {run.service_result}, '
        f'{run.exit_code} with status {run.exit_status}'
    )


def _root_line(run: FailedRun) -> str:
    if run.project_root is not None:
        return f'Project root: {run.project_root}'
    return (
        'Project root: unknown -- QUALITY_SNAPSHOT_ROOT was not in this handler\'s environment; '
        f'the unit reads it from {ENV_FILE_DISPLAY.format(project_id=run.project_id)}, '
        'which scripts/install-quality-snapshot-timer.sh writes'
    )


def _detail(run: FailedRun, journal: str) -> str:
    return '\n'.join([
        _root_line(run),
        f'Unit: {run.unit}, invocation {run.invocation_id or UNKNOWN}',
        f'Read the run: journalctl --user -u {run.unit}',
        f'Re-run it: systemctl --user start {run.unit}',
        f'Its exact command: systemctl --user cat {run.unit}',
        'Runbook: OPERATIONS.md §"Nightly quality snapshot (03:45)"',
        'Journal tail:',
        journal,
    ])


def escalation_arguments(run: FailedRun, journal: str, *, now: datetime.datetime) -> EscalationArguments:
    """The one escalate_info a failed *run* files, its *journal* tail quoted, measured at *now*."""
    return {
        'task_id': f'quality-snapshot-{run.project_id}',
        'agent_role': AGENT_ROLE,
        'category': 'infra_issue',
        'severity': 'info',
        'summary': _summary(run),
        'detail': _detail(run, journal),
        'evidence': [{
            'observation': (
                f'{run.unit}: result={run.service_result} exit_code={run.exit_code} '
                f'exit_status={run.exit_status}'
            ),
            'measured_at': now.isoformat(),
            'ref': f'invocation {run.invocation_id or UNKNOWN}',
        }],
    }


def default_escalation_url() -> str:
    """The factory checkout's configured escalation MCP URL, found as scripts/transcript_secret_sweep.py::escalation_url does."""
    config = DashboardConfig(project_root=default_project_root())
    try:
        return config.escalation_urls[config.project_root.name]
    except KeyError:
        raise NoEscalationServer(
            f'{config.project_root}: no escalation server is configured '
            '(escalation.port in its dark-factory-orchestrator.yaml)'
        ) from None


def _post(base_url: str, arguments: Mapping[str, object]) -> object:
    with SyncMcpSession(base_url, client_name=AGENT_ROLE, timeout=POST_TIMEOUT_SECS) as session:
        return session.call_tool('escalate_info', arguments)


def _not_posted(why: str, exc: Exception) -> int:
    print(f'{_LOG_PREFIX} the escalation was not posted: {why}: {exc.__class__.__name__}: {exc}', file=sys.stderr)
    return EXIT_NOT_POSTED


def _deliver(url: str | None, arguments: EscalationArguments, poster: Poster) -> int:
    try:
        target = url or default_escalation_url()
    except Exception as exc:
        return _not_posted('the factory escalation server could not be found', exc)
    try:
        reply = poster(mcp_server_base(target), arguments)
    except Exception as exc:
        return _not_posted(f'{target} was not reached', exc)
    if isinstance(reply, Mapping) and 'error' in reply:
        print(f'{_LOG_PREFIX} {target} refused the escalation: {reply["error"]}', file=sys.stderr)
        return EXIT_NOT_POSTED
    return EXIT_POSTED


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='File one escalation for a failed quality-snapshot@<project_id> run.',
        epilog='Exit codes: 0 the escalation landed; 2 it was not posted.',
    )
    parser.add_argument('--project-id', required=True, help='The instance (%%i) whose run failed.')
    parser.add_argument(
        '--escalation-url',
        default=None,
        help="The escalation server's MCP URL (default: the factory checkout's configured server).",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] = os.environ,
    poster: Poster | None = None,
) -> int:
    """CLI entry point: journal the failure's summary first, so it stands even when the post fails, then post it."""
    args = _parser().parse_args(argv)
    run = FailedRun.from_environment(args.project_id, environ)
    arguments = escalation_arguments(run, journal_tail(run.unit), now=datetime.datetime.now(datetime.UTC))
    print(f'{_LOG_PREFIX} {arguments["summary"]}', file=sys.stderr, flush=True)
    return _deliver(args.escalation_url, arguments, poster or _post)


if __name__ == '__main__':
    sys.exit(main())
