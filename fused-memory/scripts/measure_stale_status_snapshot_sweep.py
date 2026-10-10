"""Measure the stale status-snapshot sweep over the complete live corpus. (task 4851)

Answers three questions about
``fused_memory/reconciliation/stale_status_snapshot_edge_sweep.py`` with one
read per project, and writes the answers to ONE committed artifact pair —
the only place their figures live:

- #4387 COVERAGE. How much of what the sweep could retire does it retire?
- #4588 GENERAL RULE. Would retiring an id whenever its asserted status
  differs from the live one retire anything the shipped rules do not?
- #4260 CLAUSE BREAKS. How often does each clause-break character occur
  token-internally, as the shipped predicate classifies it?

READ-ONLY. Edges come from ``GraphitiBackend.enumerate_all_valid_edges``,
the same read the production sweep uses, on a backend initialised with
``skip_maintenance=True``. Statuses come from
``SqliteTaskBackend.get_statuses``, as
``memory_eval_staleness_sweep.py::fetch_terminal_task_ids`` reads them; a
project root whose task DB is missing or empty is refused
(``assert_task_store_populated``) rather than opened, because opening one
would create it. Nothing is written except the artifacts.

Regenerate (the root must hold ``.taskmaster/tasks/tasks.db``, so from a
task worktree pass the main checkout explicitly):

    cd fused-memory && uv run python scripts/measure_stale_status_snapshot_sweep.py \\
        --project dark_factory=/home/leo/src/dark-factory

DEFINITIONS, over the deduped valid edges of each measured project:

- gate_passing: the fact matches ``SNAPSHOT_STATUS_RE``, i.e. it carries a
  status marker anywhere.
- terminal_referencing: gate_passing, and the fact LEXICALLY names a task
  that is now done/cancelled (``lexically_referenced_task_ids``: every
  ``TASK_REF_RE`` id plus every digit run after a 'tasks' head). A superset
  probe, wider than extraction by construction.
- selected: the terminal_referencing edges the SHIPPED rules select now.
- unselected: terminal_referencing minus selected. The miss UPPER bound:
  most are not status snapshots at all, so it is triaged by eye from the
  samples, not read as a miss count.
- general rule: an id asserted under a marker that maps to a TaskStatus
  ('pending' -> pending, 'in-progress' -> in-progress) is contradicted by a
  positively-known, non-terminal live status other than the mapped one.
  'blocked' is already shipped rule 2. 'active' and 'stalled' map to no
  TaskStatus, so the rule has no contradiction test for them: they are
  counted, never simulated. newly_retired counts edges the general rule
  selects and the shipped rules do not.

Why the task-2613 miss 'rate' is not recomputed literally: it was never
deterministic. Its only figures were LLM-sampled (17 of an estimated 23+,
and task 3079's 4 of ~10), so there is nothing to recompute; the funnel
above replaces it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# The probe IMPORTS what it measures rather than re-spelling it, so a code
# change can never leave it measuring a stale spelling (the
# measure_plural_enum_guard_recall.py precedent).
_SRC = Path(__file__).resolve().parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from shared.task_statuses import TaskStatus  # noqa: E402

from fused_memory.backends.graphiti_client import PagedRead  # noqa: E402
from fused_memory.reconciliation.stale_status_snapshot_edge_sweep import (  # noqa: E402
    CLAUSE_BREAK_CHARS,
    SNAPSHOT_STATUS_RE,
    extract_marker_task_ids,
    flatten_dedup_edges,
    is_token_internal_break,
    select_stale_status_snapshot_edges,
)
from fused_memory.reconciliation.task_filter import (  # noqa: E402
    INACTIVE_TASK_STATUSES,
    TASK_REF_RE,
)
from fused_memory.utils.target_store_preflight import (  # noqa: E402
    TargetStoreMissing,
    assert_task_store_populated,
)

logger = logging.getLogger('measure_stale_status_snapshot_sweep')

SCHEMA_VERSION = 1

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_JSON_OUT = _REPO_ROOT / 'plans' / 'stale-status-snapshot-sweep-report.json'
DEFAULT_MD_OUT = _REPO_ROOT / 'plans' / 'stale-status-snapshot-sweep-report.md'
DEFAULT_MAX_SAMPLES = 25

DO_NOT_WIDEN_VERDICT = (
    'DO NOT WIDEN: the general rule would newly retire no edge in the measured '
    'corpus, so the selection stays terminal rule plus blocked-only rule 2.'
)

EdgeSource = Callable[[str], Awaitable[tuple[dict[str, list[dict]], PagedRead]]]
StatusSource = Callable[[str], Awaitable[dict[str, str]]]


def general_rule_verdict(newly_retired: int) -> str:
    if newly_retired == 0:
        return DO_NOT_WIDEN_VERDICT
    return (
        f'PROVISIONAL: the general rule would newly retire {newly_retired} '
        'edge(s). Validate each shape\'s precision in a follow-up task before '
        'widening; over-selection is unrecoverable.'
    )


def _fact(edge: dict) -> str:
    return edge.get('fact') or ''


def _by_uuid(edges: Iterable[dict]) -> list[dict]:
    return sorted(edges, key=lambda edge: edge['uuid'])


# --------------------------------------------------------------------------- #
# #4387 coverage
# --------------------------------------------------------------------------- #

_PLURAL_HEAD_RE = re.compile(r'\btasks\b', re.IGNORECASE)
_DIGIT_RUN_RE = re.compile(r'\d+')


def lexically_referenced_task_ids(fact: str) -> set[int]:
    """Every id the fact could possibly name: a SUPERSET of extraction."""
    ids = {int(match.group(1)) for match in TASK_REF_RE.finditer(fact)}
    head = _PLURAL_HEAD_RE.search(fact)
    if head:
        ids.update(int(run) for run in _DIGIT_RUN_RE.findall(fact, head.end()))
    return ids


@dataclass(frozen=True)
class Coverage:
    gate_passing: int
    terminal_referencing: int
    selected: int
    unselected: int
    unselected_samples: tuple[str, ...]


def measure_coverage(
    edges: list[dict], statuses: Mapping[str, str], *, max_samples: int,
) -> Coverage:
    gated = [edge for edge in edges if SNAPSHOT_STATUS_RE.search(_fact(edge))]
    referencing = [
        edge for edge in gated
        if any(
            statuses.get(str(task_id)) in INACTIVE_TASK_STATUSES
            for task_id in lexically_referenced_task_ids(_fact(edge))
        )
    ]
    selected = {
        edge['uuid'] for edge in select_stale_status_snapshot_edges(referencing, dict(statuses))
    }
    unselected = _by_uuid(edge for edge in referencing if edge['uuid'] not in selected)
    return Coverage(
        gate_passing=len(gated),
        terminal_referencing=len(referencing),
        selected=len(selected),
        unselected=len(unselected),
        unselected_samples=tuple(_fact(edge) for edge in unselected[:max_samples]),
    )


# --------------------------------------------------------------------------- #
# #4588 general-rule simulation
# --------------------------------------------------------------------------- #

# One (adjective, transitive) alternation pair per marker — this rule's own
# per-marker vocabulary — and the status a contradiction is judged against.
_MARKERS: dict[str, tuple[str | None, str | None, str | None]] = {
    'pending': (r'(?:pending)', None, TaskStatus.PENDING),
    'in-progress': (r'(?:in[-\s]?progress)', None, TaskStatus.IN_PROGRESS),
    'active': (r'(?:active)', None, None),
    'stalled': (None, r'(?:stalled)', None),
}
_KNOWN_STATUSES = frozenset(TaskStatus)


def ids_by_marker(fact: str) -> dict[str, set[int]]:
    """Per-marker ids, through the sweep's own per-marker seam."""
    return {
        marker: extract_marker_task_ids(fact, adjective, transitive)
        for marker, (adjective, transitive, _) in _MARKERS.items()
    }


def _general_rule_contradicts(marker: str, live: str | None) -> bool:
    mapped = _MARKERS[marker][2]
    return (
        mapped is not None
        and live in _KNOWN_STATUSES
        and live not in INACTIVE_TASK_STATUSES
        and live != mapped
    )


@dataclass(frozen=True)
class GeneralRule:
    asserted: dict[tuple[str, str | None], int]
    undefined_contradiction: tuple[str, ...]
    newly_retired: int
    newly_retired_samples: tuple[str, ...]
    verdict: str


def measure_general_rule(
    edges: list[dict], statuses: Mapping[str, str], *, max_samples: int,
) -> GeneralRule:
    shipped = {
        edge['uuid'] for edge in select_stale_status_snapshot_edges(edges, dict(statuses))
    }
    asserted: Counter[tuple[str, str | None]] = Counter()
    newly: list[dict] = []
    for edge in _by_uuid(edges):
        contradicted = False
        for marker, ids in ids_by_marker(_fact(edge)).items():
            for task_id in ids:
                live = statuses.get(str(task_id))
                asserted[(marker, live)] += 1
                contradicted |= _general_rule_contradicts(marker, live)
        if contradicted and edge['uuid'] not in shipped:
            newly.append(edge)
    return GeneralRule(
        asserted=dict(asserted),
        undefined_contradiction=tuple(
            sorted(marker for marker, (_, _, mapped) in _MARKERS.items() if mapped is None)
        ),
        newly_retired=len(newly),
        newly_retired_samples=tuple(_fact(edge) for edge in newly[:max_samples]),
        verdict=general_rule_verdict(len(newly)),
    )


# --------------------------------------------------------------------------- #
# #4260 token-internal clause breaks
# --------------------------------------------------------------------------- #

_BREAK_CHAR_RE = re.compile('[' + re.escape(CLAUSE_BREAK_CHARS) + ']')


@dataclass(frozen=True)
class TokenInternalBreaks:
    counts: dict[str, int]
    samples: dict[str, tuple[str, ...]]


def measure_token_internal_breaks(
    edges: list[dict], *, max_samples: int,
) -> TokenInternalBreaks:
    counts = dict.fromkeys(CLAUSE_BREAK_CHARS, 0)
    samples: dict[str, list[str]] = {char: [] for char in CLAUSE_BREAK_CHARS}
    for edge in _by_uuid(edges):
        fact = _fact(edge)
        for match in _BREAK_CHAR_RE.finditer(fact):
            if not is_token_internal_break(fact, match.start()):
                continue
            char = match.group()
            counts[char] += 1
            if len(samples[char]) < max_samples and fact not in samples[char]:
                samples[char].append(fact)
    return TokenInternalBreaks(
        counts=counts, samples={char: tuple(found) for char, found in samples.items()},
    )


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ProjectMeasurement:
    project_id: str
    project_root: str
    valid_edges: int
    complete: bool
    incomplete_kind: str | None
    coverage: Coverage
    general_rule: GeneralRule
    token_internal_breaks: TokenInternalBreaks


async def measure_project(
    project_id: str,
    project_root: str,
    *,
    edge_source: EdgeSource,
    status_source: StatusSource,
    max_samples: int,
) -> ProjectMeasurement:
    grouped, paged = await edge_source(project_id)
    edges = flatten_dedup_edges(grouped)
    statuses = await status_source(project_root)
    return ProjectMeasurement(
        project_id=project_id,
        project_root=project_root,
        valid_edges=len(edges),
        complete=paged.complete,
        incomplete_kind=paged.incomplete_kind,
        coverage=measure_coverage(edges, statuses, max_samples=max_samples),
        general_rule=measure_general_rule(edges, statuses, max_samples=max_samples),
        token_internal_breaks=measure_token_internal_breaks(edges, max_samples=max_samples),
    )


@dataclass(frozen=True)
class Report:
    measured_at: str
    projects: tuple[ProjectMeasurement, ...]
    triage_note: str | None
    schema_version: int = SCHEMA_VERSION

    @property
    def complete(self) -> bool:
        return bool(self.projects) and all(project.complete for project in self.projects)


def build_report(
    projects: Iterable[ProjectMeasurement], *, measured_at: str,
    triage_note: str | None = None,
) -> Report:
    return Report(measured_at=measured_at, projects=tuple(projects), triage_note=triage_note)


def exit_code(report: Report) -> int:
    return 0 if report.complete else 1


def _project_payload(project: ProjectMeasurement) -> dict[str, Any]:
    payload = asdict(project)
    payload['general_rule']['asserted'] = [
        {'marker': marker, 'status': status, 'count': count}
        for (marker, status), count in sorted(
            project.general_rule.asserted.items(),
            key=lambda item: (item[0][0], item[0][1] or ''),
        )
    ]
    return payload


def to_json(report: Report) -> str:
    payload = {
        'schema_version': report.schema_version,
        'measured_at': report.measured_at,
        'complete': report.complete,
        'triage_note': report.triage_note,
        'projects': [_project_payload(project) for project in report.projects],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + '\n'


def _table(rows: Iterable[tuple[str, object]]) -> list[str]:
    return ['| metric | value |', '|---|---|', *(f'| {name} | {value} |' for name, value in rows)]


def _samples(title: str, samples: Iterable[str]) -> list[str]:
    lines = [f'{title}:', '']
    lines += [f'- `{sample}`' for sample in samples] or ['- (none)']
    return [*lines, '']


def _render_project(project: ProjectMeasurement) -> list[str]:
    coverage = project.coverage
    rule = project.general_rule
    breaks = project.token_internal_breaks
    return [
        f'## {project.project_id}',
        '',
        f'- Project root: `{project.project_root}`',
        f'- Valid edges (deduped): {project.valid_edges}',
        f'- Enumeration complete: {project.complete}'
        + (f' ({project.incomplete_kind})' if project.incomplete_kind else ''),
        '',
        '### Coverage (#4387)',
        '',
        *_table([
            ('gate_passing', coverage.gate_passing),
            ('terminal_referencing', coverage.terminal_referencing),
            ('selected', coverage.selected),
            ('unselected', coverage.unselected),
        ]),
        '',
        *_samples('Unselected samples (miss upper bound, triage by eye)',
                  coverage.unselected_samples),
        '### General rule simulation (#4588)',
        '',
        f'**Verdict:** {rule.verdict}',
        '',
        *_table([
            ('newly_retired', rule.newly_retired),
            ('undefined_contradiction', ', '.join(rule.undefined_contradiction)),
        ]),
        '',
        '| marker | live status | assertions |',
        '|---|---|---|',
        *(
            f'| {marker} | {status or "(absent)"} | {count} |'
            for (marker, status), count in sorted(
                rule.asserted.items(), key=lambda item: (item[0][0], item[0][1] or ''),
            )
        ),
        '',
        *_samples('Newly-retired samples', rule.newly_retired_samples),
        '### Token-internal clause breaks (#4260)',
        '',
        '| char | token-internal occurrences |',
        '|---|---|',
        *(f'| `{char}` | {count} |' for char, count in breaks.counts.items()),
        '',
        *(
            line
            for char, samples in breaks.samples.items() if samples
            for line in _samples(f'`{char}` samples', samples)
        ),
    ]


def render_markdown(report: Report) -> str:
    lines = [
        '# Stale status-snapshot sweep measurement (task 4851)',
        '',
        f'- Measured at: `{report.measured_at}`',
        f'- Complete: {report.complete}',
        f'- Projects: {", ".join(p.project_id for p in report.projects) or "(none)"}',
        '- Generated by `fused-memory/scripts/measure_stale_status_snapshot_sweep.py`; '
        'its docstring defines every metric. Every figure below is a point-in-time '
        'fact about a growing corpus.',
        '',
    ]
    for project in report.projects:
        lines += _render_project(project)
    if report.triage_note:
        lines += ['## Triage (by eye)', '', report.triage_note.strip(), '']
    return '\n'.join(lines).rstrip() + '\n'


def write_artifacts(report: Report, json_out: Path, md_out: Path) -> tuple[Path, Path]:
    """Write the pair; an INCOMPLETE report goes to sidecars, never the canon."""
    if not report.complete:
        json_out = json_out.with_suffix('.incomplete.json')
        md_out = md_out.with_suffix('.incomplete.md')
    json_out.write_text(to_json(report))
    md_out.write_text(render_markdown(report))
    return json_out, md_out


# --------------------------------------------------------------------------- #
# Thin I/O shell
# --------------------------------------------------------------------------- #


def _parse_project(value: str) -> tuple[str, str]:
    graph, sep, root = value.partition('=')
    if not (sep and graph and root):
        raise argparse.ArgumentTypeError(f'expected GRAPH=ROOT, got {value!r}')
    return graph, root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Measure the stale status-snapshot sweep over the live corpus (read-only).',
    )
    parser.add_argument(
        '--project', dest='projects', action='append', type=_parse_project,
        metavar='GRAPH=ROOT',
        help=f'Graph and task-tree root to measure; repeatable '
             f'(default: dark_factory={_REPO_ROOT}).',
    )
    parser.add_argument('--json-out', type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument('--md-out', type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument('--max-samples', type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument(
        '--triage-note', type=Path, default=None,
        help='Markdown file holding a by-eye triage of the samples.',
    )
    return parser


async def _main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    args = _build_parser().parse_args(argv)
    projects: list[tuple[str, str]] = args.projects or [('dark_factory', str(_REPO_ROOT))]

    from datetime import UTC, datetime  # noqa: PLC0415

    from fused_memory.backends.graphiti_client import GraphitiBackend  # noqa: PLC0415
    from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend  # noqa: PLC0415
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    for _graph, root in projects:
        try:
            assert_task_store_populated(root, operation='measure_stale_status_snapshot_sweep')
        except TargetStoreMissing as refusal:
            logger.error('%s', refusal)
            return 2

    measured_at = datetime.now(UTC).isoformat()
    config = FusedMemoryConfig()
    graphiti = GraphitiBackend(config)
    tasks = SqliteTaskBackend(config.taskmaster)
    try:
        await graphiti.initialize(skip_maintenance=True)
        await tasks.start()

        async def edge_source(project_id: str) -> tuple[dict[str, list[dict]], PagedRead]:
            grouped, paged = await graphiti.enumerate_all_valid_edges(group_id=project_id)
            return {entity: [dict(edge) for edge in edges] for entity, edges in grouped.items()}, paged

        async def status_source(project_root: str) -> dict[str, str]:
            return await tasks.get_statuses(project_root)

        measured = [
            await measure_project(
                graph, root, edge_source=edge_source, status_source=status_source,
                max_samples=args.max_samples,
            )
            for graph, root in projects
        ]
    finally:
        await tasks.close()
        await graphiti.close()

    triage_note = args.triage_note.read_text() if args.triage_note else None
    report = build_report(measured, measured_at=measured_at, triage_note=triage_note)
    written = write_artifacts(report, args.json_out, args.md_out)
    logger.info('complete=%s written=%s', report.complete, [str(p) for p in written])
    for project in report.projects:
        if not project.complete:
            logger.error(
                'project=%s enumeration incomplete (%s)', project.project_id,
                project.incomplete_kind,
            )
    return exit_code(report)


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(_main(argv))


if __name__ == '__main__':
    sys.exit(main())
