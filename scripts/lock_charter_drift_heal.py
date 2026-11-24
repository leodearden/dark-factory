#!/usr/bin/env python3
"""Route one cross-repo lock-charter reconciliation to the owner that can act on it.

``scripts/check_lock_charter_drift.py`` answers whether reify's and
dark-factory's lock-charter mirrors agree.  ``lock-charter-drift-check.timer``
runs this script nightly to deliver that answer, off the merge gate:

    ok           nothing to do.
    drift        exactly ONE open heal task, found by metadata
                 ``x_lock_charter_drift``.  An ACTIVE one is adopted, and a
                 half-filed ``deferred`` one is re-committed.  Otherwise one
                 is filed in planning mode and committed, so the next run's
                 lookup already sees it.
    unavailable  one ``escalate_info`` in slot ``lock-charter-drift:unavailable``,
                 skipped while one is pending there.

Exit 0 when the outcome reached its owner, 2 when a tool call failed.
Operations: ``OPERATIONS.md`` section 9a.
"""

from __future__ import annotations

import argparse
import enum
import functools
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import check_lock_charter_drift as drift
    from dashboard.config import DEFAULT_FUSED_MEMORY_URLS, DashboardConfig
    from shared.elision import elide_at_word
    from shared.mcp_post import mcp_server_base
    from shared.mcp_session import SyncMcpSession
    from shared.project_layout import default_project_root
    from shared.task_statuses import ACTIVE, TaskStatus
except ImportError as exc:
    print(
        f'lock_charter_drift_heal.py: {exc} -- run it as '
        '`uv run --project shared python scripts/lock_charter_drift_heal.py`',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

AGENT_ROLE = 'lock-charter-drift-check'
HEAL_KEY = 'x_lock_charter_drift'
HEAL_KEY_VALUE = 'reify-mirror'
UNAVAILABLE_SLOT = 'lock-charter-drift:unavailable'

HEAL_SITES = (
    'shared/src/shared/locking.py',
    'fused-memory/src/fused_memory/middleware/lock_charter_guard.py',
    'shared/tests/test_locking.py',
    'fused-memory/tests/test_lock_charter_guard.py',
)
"""Every file a mirror edit touches: the two vector copies and their two test vectors."""

CHECK_COMMAND = 'uv run --project shared python scripts/check_lock_charter_drift.py'
DRY_RUN_COMMAND = 'uv run --project shared python scripts/lock_charter_drift_heal.py --dry-run'

TITLE_CAP = 100
FUSED_MEMORY_URL_ENV = 'FUSED_MEMORY_MCP_URL'
FUSED_MEMORY_TIMEOUT_SECS = 120
ESCALATION_TIMEOUT_SECS = 30

Report = Mapping[str, Any]
ToolCall = Callable[[str, dict[str, object]], Mapping[str, object]]
"""One MCP tool call: tool name and arguments in, the tool's reply out."""


class RouteFailure(Exception):
    """A tool call that did not answer, or answered an error: the outcome did not reach its owner."""


class Outcome(enum.StrEnum):
    OK = 'ok'
    FILED = 'filed'
    ADOPTED = 'adopted'
    ESCALATED = 'escalated'
    ESCALATION_PENDING = 'escalation already pending'


@dataclass(frozen=True)
class RouteResult:
    outcome: Outcome
    task_id: str | None = None
    note: str | None = None

    def line(self) -> str:
        head = str(self.outcome) if self.task_id is None else f'{self.outcome} {self.task_id}'
        return head if self.note is None else f'{head} ({self.note})'


@dataclass(frozen=True)
class Ports:
    fused_memory: ToolCall
    escalation: ToolCall


def _reply(port: ToolCall, tool: str, args: dict[str, object]) -> Mapping[str, object]:
    try:
        result = port(tool, args)
    except Exception as exc:
        raise RouteFailure(f'{tool} failed: {type(exc).__name__}: {exc}') from exc
    if not isinstance(result, Mapping):
        raise RouteFailure(f'{tool} answered {result!r}, not a mapping')
    return result


def _call(port: ToolCall, tool: str, args: dict[str, object]) -> Mapping[str, object]:
    result = _reply(port, tool, args)
    if 'error' in result:
        raise RouteFailure(f'{tool} answered an error: {result!r}')
    return result


# ---------------------------------------------------------------------------
# The heal task's brief.


@dataclass(frozen=True)
class _Drifted:
    """One drifted name: its part's label, the name, and a witness path when the corpus supplied one."""

    label: str
    name: str
    witness: str | None = None

    def text(self) -> str:
        return self.name if self.witness is None else f'{self.name} (e.g. {self.witness})'


_VECTOR_LABELS = (('extensions', 'extensions'), ('extensionless', 'extension-less names'))


def _by_direction(report: Report) -> tuple[list[_Drifted], list[_Drifted]]:
    """``(only_in_reify, only_in_dark_factory)`` across every part of *report*."""
    parts = report.get('parts', {})
    vectors = parts.get('vectors', {}).get('vectors', {})
    extensionless = parts.get('extensionless', {})
    only_in_reify: list[_Drifted] = []
    only_in_dark_factory: list[_Drifted] = []
    for key, label in _VECTOR_LABELS:
        diff = vectors.get(key, {})
        only_in_reify += [_Drifted(f'{label} reify declares', n) for n in diff.get('only_in_reify', [])]
        only_in_dark_factory += [
            _Drifted(f'{label} reify no longer declares', n) for n in diff.get('only_in_dark_factory', [])
        ]
    only_in_reify += [
        _Drifted('tracked extensions the predicate rejects', ext, path)
        for ext, path in sorted(parts.get('corpus', {}).get('findings', {}).items())
    ]
    only_in_reify += [
        _Drifted('tracked extension-less names the predicate rejects', name, path)
        for name, path in sorted(extensionless.get('findings', {}).items())
    ]
    only_in_dark_factory += [
        _Drifted('allowlisted names that are directories in reify', name, path)
        for name, path in sorted(extensionless.get('collisions', {}).items())
    ]
    return only_in_reify, only_in_dark_factory


def _section(heading: str, drifted: list[_Drifted]) -> list[str]:
    if not drifted:
        return []
    labels = dict.fromkeys(d.label for d in drifted)
    return [heading] + [
        f'  - {label}: ' + ', '.join(d.text() for d in drifted if d.label == label) for label in labels
    ]


def heal_brief(report: Report) -> tuple[str, str]:
    """The heal task's ``(title, description)`` for a drift *report*."""
    only_in_reify, only_in_dark_factory = _by_direction(report)
    names = sorted({d.name for d in only_in_reify + only_in_dark_factory})
    title = elide_at_word(
        'Mirror reify lock-charter drift into dark-factory: ' + (', '.join(names) or 'see description'),
        TITLE_CAP,
    )
    lines = [
        "The nightly cross-repo check found reify's lock-charter declarations out of step with "
        "dark-factory's.",
        '',
        *_section('Only in reify, so mirror each into dark-factory:', only_in_reify),
        *_section(
            'Only in dark-factory: reify narrowed, or its tree contradicts the allowlist. Each needs a '
            'judgement (drop the name, or have reify rename the directory), not a blind mirror:',
            only_in_dark_factory,
        ),
        '',
        'Edit these together: the EXTENSIONLESS_FILENAMES / FILE_EXTENSIONS vectors in the two source '
        'copies, and the _CANONICAL_* / *_ACCEPT_PATHS vectors in their two test copies:',
        *[f'  - {site}' for site in HEAL_SITES],
        '',
        f'Re-run `{CHECK_COMMAND}` until it exits 0. Names that drift after this task was filed are '
        'healed here too: the nightly run adopts this task rather than filing another.',
        '',
        "This is reify's news, not a fault of any dark-factory branch, and it blocks no dark-factory merge.",
    ]
    return title, '\n'.join(lines)


# ---------------------------------------------------------------------------
# The three routes.


def _committed(ports: Ports, project_root: str, task_id: str) -> str | None:
    """Release *task_id* from its planning hold: ``None`` when released, else the refusal.

    A refusal is a hold someone else stamped (fused-memory releases only its
    own planning hold), so it is reported, never retried around.
    """
    args: dict[str, object] = {'project_root': project_root, 'task_ids': task_id, 'target_status': 'pending'}
    reply = _reply(ports.fused_memory, 'commit_planning', args)
    if reply.get('success') is True:
        return None
    per_id = reply.get('results')
    reasons = sorted({r for r in map(_refusal_reason, per_id if isinstance(per_id, list) else []) if r})
    return f'commit_planning refused: {", ".join(reasons) if reasons else reply.get("error", reply)}'


def _refusal_reason(one: object) -> str | None:
    """The ``reason_code`` (else ``error``) of one refused id in a ``commit_planning`` reply."""
    result = one.get('result') if isinstance(one, Mapping) else None
    if not isinstance(result, Mapping) or result.get('success'):
        return None
    return str(result.get('reason_code') or result.get('error'))


def _open_owner(ports: Ports, project_root: str) -> tuple[str, str] | None:
    args: dict[str, object] = {
        'project_root': project_root,
        'key': HEAL_KEY,
        'value': HEAL_KEY_VALUE,
        'statuses': sorted(ACTIVE),
    }
    rows = _call(ports.fused_memory, 'find_tasks_by_metadata', args).get('results')
    if not isinstance(rows, list):
        raise RouteFailure(f'find_tasks_by_metadata answered {rows!r}, not a results list')
    if not rows:
        return None
    first = rows[0]
    if not isinstance(first, Mapping) or 'task_id' not in first or 'status' not in first:
        raise RouteFailure(f'find_tasks_by_metadata answered a row with no task_id/status: {first!r}')
    return str(first['task_id']), str(first['status'])


def ensure_heal_task(report: Report, ports: Ports, *, project_root: Path) -> RouteResult:
    """Adopt the open heal task, or file and commit one: never a second open owner."""
    root = str(project_root)
    owner = _open_owner(ports, root)
    if owner is not None:
        task_id, status = owner
        note = _committed(ports, root, task_id) if status == TaskStatus.DEFERRED else None
        return RouteResult(Outcome.ADOPTED, task_id, note)
    title, description = heal_brief(report)
    args: dict[str, object] = {
        'project_root': root,
        'title': title,
        'description': description,
        'details': report['detail'],
        'priority': 'medium',
        'planning_mode': True,
        'agent_id': AGENT_ROLE,
        'metadata': {HEAL_KEY: HEAL_KEY_VALUE, 'source': AGENT_ROLE, 'files': list(HEAL_SITES)},
    }
    submitted = _call(ports.fused_memory, 'submit_task', args)
    task_id = submitted.get('task_id')
    if not isinstance(task_id, str) or not task_id:
        raise RouteFailure(f'submit_task answered {submitted!r}, naming no task')
    return RouteResult(Outcome.FILED, task_id, _committed(ports, root, task_id))


def _pending_slots(escalation: ToolCall) -> frozenset[str]:
    records = _call(escalation, 'get_pending_escalations', {'compact': True}).get('result')
    if not isinstance(records, list):
        raise RouteFailure(f'get_pending_escalations answered {records!r}, not a result list')
    slots = []
    for record in records:
        slot = record.get('task_id') if isinstance(record, Mapping) else None
        if not isinstance(slot, str):
            raise RouteFailure(f'get_pending_escalations answered a record with no str task_id: {record!r}')
        slots.append(slot)
    return frozenset(slots)


def _unavailable_summary() -> str:
    return 'Nightly reify lock-charter drift check could not look: nothing was compared'


def escalate_unavailable(report: Report, ports: Ports, *, now: datetime, reify_repo: str | None) -> RouteResult:
    """Raise "could not look" as loudly as drift: one ``escalate_info``, unless one is pending."""
    if UNAVAILABLE_SLOT in _pending_slots(ports.escalation):
        return RouteResult(Outcome.ESCALATION_PENDING)
    sibling = reify_repo or 'the reify checkout REIFY_ROOT or discovery resolves'
    args: dict[str, object] = {
        'task_id': UNAVAILABLE_SLOT,
        'agent_role': AGENT_ROLE,
        'category': 'infra_issue',
        'summary': _unavailable_summary(),
        'detail': f'{report["detail"]}\n\nUntil this is fixed, no reify drift is reported anywhere.',
        'suggested_action': (
            f'Restore {sibling} with its scripts/lock-charter-guard.sh, or point the timer at the right '
            f'checkout with --reify-repo (REIFY_REPO in scripts/lock-charter-drift-check.sh); then '
            f'confirm with `{DRY_RUN_COMMAND}`.'
        ),
        'evidence': [{
            'observation': str(report['detail']).splitlines()[0],
            'measured_at': now.isoformat(),
            'ref': 'scripts/check_lock_charter_drift.py',
        }],
    }
    _call(ports.escalation, 'escalate_info', args)
    return RouteResult(Outcome.ESCALATED)


def route(
    report: Report, ports: Ports, *, project_root: Path, now: datetime, reify_repo: str | None = None,
) -> RouteResult:
    """Deliver *report* to its owner.  Raises `RouteFailure` when it could not."""
    match report['exit_code']:
        case drift.EXIT_OK:
            return RouteResult(Outcome.OK)
        case drift.EXIT_DRIFT:
            return ensure_heal_task(report, ports, project_root=project_root)
        case drift.EXIT_UNAVAILABLE:
            return escalate_unavailable(report, ports, now=now, reify_repo=reify_repo)
    raise RouteFailure(f'the reconciliation reported exit code {report["exit_code"]!r}, which is not 0, 1 or 2')


def dry_run_text(report: Report) -> str:
    """What `route` would deliver for *report*, for an operator; calls nothing."""
    match report['exit_code']:
        case drift.EXIT_DRIFT:
            title, description = heal_brief(report)
            return (
                f'drift: would adopt the open task carrying {HEAL_KEY}={HEAL_KEY_VALUE}, or file:\n'
                f'title: {title}\n\n{description}'
            )
        case drift.EXIT_UNAVAILABLE:
            return (
                f'unavailable: would escalate_info in slot {UNAVAILABLE_SLOT} unless one is pending:\n'
                f'{_unavailable_summary()}\n\n{report["detail"]}'
            )
    return f'{report["status"]}: nothing to route\n{report["detail"]}'


# ---------------------------------------------------------------------------
# Ports over the real servers, and the CLI.


def configured_escalation_url(project_root: Path) -> str:
    config = DashboardConfig(project_root=project_root)
    url = config.escalation_urls.get(config.project_root.name)
    if url is None:
        raise RouteFailure(f'{config.project_root} configures no escalation server (escalation.port)')
    return url


def _call_tool(base_url: str, tool: str, arguments: dict[str, object], *, timeout: float) -> dict:
    with SyncMcpSession(base_url, client_name=AGENT_ROLE, timeout=timeout) as session:
        return session.call_tool(tool, arguments)


def build_ports(project_root: Path, fused_memory_url: str | None, escalation_url: str | None) -> Ports:
    """Both MCP servers over the real transport; a URL left None takes its default."""
    fused_memory = fused_memory_url or os.environ.get(FUSED_MEMORY_URL_ENV, DEFAULT_FUSED_MEMORY_URLS[0])
    escalation = escalation_url or configured_escalation_url(project_root)
    return Ports(
        fused_memory=functools.partial(_call_tool, mcp_server_base(fused_memory), timeout=FUSED_MEMORY_TIMEOUT_SECS),
        escalation=functools.partial(_call_tool, mcp_server_base(escalation), timeout=ESCALATION_TIMEOUT_SECS),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='lock_charter_drift_heal.py',
        description='Reconcile reify and dark-factory lock-charter declarations and route the outcome.',
        epilog='Exit 0: the outcome reached its owner (or a dry run printed it). Exit 2: a tool call failed.',
    )
    parser.add_argument('--project-root', type=Path, default=None, help="the main checkout (default: this checkout's)")
    parser.add_argument('--reify-repo', default=None, help='reify checkout to compare against (out-ranks $REIFY_ROOT)')
    parser.add_argument(
        '--fused-memory-url', default=None,
        help=f'fused-memory base URL (default: ${FUSED_MEMORY_URL_ENV} or {DEFAULT_FUSED_MEMORY_URLS[0]})',
    )
    parser.add_argument(
        '--escalation-url', default=None, help="escalation MCP URL (default: from the project's orchestrator config)",
    )
    parser.add_argument('--dry-run', action='store_true', help='reconcile and print what would be routed; call nothing')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = drift.reconcile(args.reify_repo)
    if args.dry_run:
        print(dry_run_text(report))
        return 0
    project_root = args.project_root or default_project_root()
    try:
        ports = build_ports(project_root, args.fused_memory_url, args.escalation_url)
        result = route(report, ports, project_root=project_root, now=datetime.now(UTC), reify_repo=args.reify_repo)
    except RouteFailure as exc:
        print(f'lock_charter_drift_heal.py: {report["status"]} not routed: {exc}', file=sys.stderr)
        return 2
    print(result.line())
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
