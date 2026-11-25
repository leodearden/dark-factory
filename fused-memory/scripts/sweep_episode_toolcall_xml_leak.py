#!/usr/bin/env python3
"""Sweep one project's Graphiti episodes for leaked tool-call XML (task 3233).

The Graphiti counterpart of ``sweep_toolcall_xml_leak.py``. Discovery is
``MemoryService.scan_episode_toolcall_leaks``, and each leak is classified by
the shared ``fused_memory.utils.toolcall_xml_leak.classify_leak_shape``. A dry
run, the default, prints the report and mutates nothing. ``--apply`` redacts
the repairable shapes in place through ``MemoryService.redact_episode_content``,
never a delete, and leaves ``manual_review`` episodes for a human.

Rationale, scope and runbook: docs/mcp-toolcall-xml-leak.md. Run it BEFORE any
large consolidation pass, which can destroy the specimens before they are
classified.

Usage
-----
  python scripts/sweep_episode_toolcall_xml_leak.py                    # dry run
  python scripts/sweep_episode_toolcall_xml_leak.py --apply
  python scripts/sweep_episode_toolcall_xml_leak.py --project-id reify
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import asyncio
import json
import logging
from typing import Any

from shared.cli_boundary import LoudArgumentParser, run_cli

from fused_memory.backends.episode_toolcall_leak_scan import (
    EpisodeToolcallLeak,
    EpisodeToolcallLeakScan,
)
from fused_memory.utils.store_mutation_preflight import assert_store_mutation_allowed
from fused_memory.utils.toolcall_xml_leak import (
    LeakShape,
    classify_leak_shape,
    repair_leaked_text,
)

SCRIPT_ID = 'sweep_episode_toolcall_xml_leak'
logger = logging.getLogger(SCRIPT_ID)

EXIT_OK = 0
EXIT_NEEDS_HUMAN = 1
EXIT_ABORTED = 2

REPAIRABLE_SHAPES = frozenset({LeakShape.REPAIRABLE_TAIL, LeakShape.REPAIRABLE_DUPLICATE})

# Per-record outcomes of an --apply that leave the episode for a human. Every
# producer and resolve_exit_code use these names, so a spelling cannot drift.
RECORD_ERROR = 'record_error'
CONTENT_CHANGED_SINCE_SCAN = 'content_changed_since_scan'
HUMAN_ADJUDICATION_FLAGS: tuple[str, ...] = (RECORD_ERROR, CONTENT_CHANGED_SINCE_SCAN)

_UNSCANNED_SUMMARY: dict[str, Any] = {
    'episodes_in_group': None,
    'candidates_seen': 0,
    'candidates_expected': None,
    'complete': False,
    'incomplete_kind': None,
}


# ---------------------------------------------------------------------------
# Pure core
# ---------------------------------------------------------------------------


def summarize_scan(scan: EpisodeToolcallLeakScan) -> dict[str, Any]:
    """The scan's denominator and completeness, as the report carries them."""
    return {
        'episodes_in_group': scan.episodes_in_group,
        'candidates_seen': scan.candidates.rows_seen,
        'candidates_expected': scan.candidates.expected_rows,
        'complete': scan.candidates.complete,
        'incomplete_kind': scan.candidates.incomplete_kind,
    }


def leak_record(leak: EpisodeToolcallLeak) -> dict[str, Any]:
    """One report record: the episode, its leak, its shape, and its repair when it has one."""
    shape = classify_leak_shape(leak.content)
    record: dict[str, Any] = {
        'uuid': leak.uuid,
        'name': leak.name,
        'created_at': leak.created_at,
        'classification': shape,
        'fragment': leak.fragment,
        'index': leak.index,
        'content': leak.content,
    }
    if shape in REPAIRABLE_SHAPES:
        record['repaired_content'] = repair_leaked_text(leak.content)
    return record


def build_report(
    *,
    project_id: str,
    dry_run: bool,
    scan_summary: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the JSON-serializable report from already-computed fields."""
    counts = {shape.value: 0 for shape in LeakShape}
    for record in records:
        counts[record['classification']] += 1
    return {
        'project_id': project_id,
        'dry_run': dry_run,
        **scan_summary,
        'counts': counts,
        'redacted': sum(1 for record in records if record.get('redacted')),
        'records': records,
    }


def new_progress() -> dict[str, Any]:
    """A caller-owned container :func:`run` fills in, so an aborted run still reports."""
    return {'records': [], 'scan_summary': dict(_UNSCANNED_SUMMARY)}


def report_from_progress(args: Any, progress: dict[str, Any]) -> dict[str, Any]:
    """The report for whatever *progress* holds, in the same shape a finished run returns."""
    return build_report(
        project_id=args.project_id,
        dry_run=not args.apply,
        scan_summary=progress['scan_summary'],
        records=progress['records'],
    )


def resolve_exit_code(report: dict[str, Any]) -> int:
    """0 for any dry run: its report is the investigation, whatever it found.

    An --apply exits 1 when it left work for a human: a manual_review episode,
    an incomplete scan (an unknown share of the group went unswept), or a record
    carrying one of :data:`HUMAN_ADJUDICATION_FLAGS`.
    """
    if report['dry_run']:
        return EXIT_OK
    needs_human = (
        report['counts'][LeakShape.MANUAL_REVIEW] > 0
        or not report['complete']
        or any(
            record.get(flag)
            for record in report['records']
            for flag in HUMAN_ADJUDICATION_FLAGS
        )
    )
    return EXIT_NEEDS_HUMAN if needs_human else EXIT_OK


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


async def run(args: Any, memory_service: Any, progress: dict[str, Any] | None = None) -> dict:
    """Scan the project's episodes, classify every leak and, under --apply, redact.

    An --apply first proves the store is writable from here, before it scans.
    Each record joins *progress* before its redaction is attempted, so an
    abort still reports it. A scan TimeoutError propagates.
    """
    progress = new_progress() if progress is None else progress
    if args.apply:
        assert_store_mutation_allowed(
            operation=f'{SCRIPT_ID} --apply',
            logger=logger,
            extra_detail=(
                'a redaction writes a Graphiti episode content SET plus a '
                'write-journal row. Nothing was scanned and nothing was redacted.'
            ),
        )
    scan = await memory_service.scan_episode_toolcall_leaks(project_id=args.project_id)
    progress['scan_summary'] = summarize_scan(scan)
    for leak in scan.leaks:
        record = leak_record(leak)
        progress['records'].append(record)
        if args.apply:
            await _redact_record(record, leak, memory_service, project_id=args.project_id)
    return report_from_progress(args, progress)


async def _redact_record(
    record: dict[str, Any], leak: EpisodeToolcallLeak, memory_service: Any, *, project_id: str,
) -> None:
    """Redact one repairable episode in place, recording the outcome on *record*.

    Any other shape is left untouched. A failed redaction is recorded on its
    own record and never stops the sweep.
    """
    if record['classification'] not in REPAIRABLE_SHAPES:
        logger.warning(
            '%s: episode %s is %s; left untouched for a human',
            SCRIPT_ID, leak.uuid, record['classification'],
        )
        return
    try:
        result = await memory_service.redact_episode_content(
            leak.uuid,
            new_content=record['repaired_content'],
            project_id=project_id,
            agent_id=SCRIPT_ID,
            _source=SCRIPT_ID,
        )
    except Exception as exc:
        record[RECORD_ERROR] = True
        record['error'] = f'{type(exc).__name__}: {exc}'
        logger.exception('%s: redacting episode %s failed; the sweep continues', SCRIPT_ID, leak.uuid)
        return
    record['redacted'] = True
    if result.get('old_content') != leak.content:
        record[CONTENT_CHANGED_SINCE_SCAN] = True
        logger.warning(
            '%s: episode %s changed between the scan and its redaction; '
            'the write journal holds the content it replaced',
            SCRIPT_ID, leak.uuid,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> LoudArgumentParser:
    parser = LoudArgumentParser(
        description=(
            "Report which of a project's Graphiti episodes carry a leaked tool-call "
            'fragment and, with --apply, redact the shapes that can be repaired '
            'without losing content.'
        ),
    )
    parser.add_argument(
        '--project-id', dest='project_id', default='dark_factory',
        help='Project graph to sweep (default: dark_factory).',
    )
    parser.add_argument(
        '--apply', action='store_true', default=False,
        help=(
            'Redact the repairable episodes in place (default: dry run, which '
            'prints the report and exits 0).'
        ),
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (default: the environment).',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse *argv*, open the service, run the sweep and print its report."""
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s',
    )
    args = _build_parser().parse_args(argv)
    progress = new_progress()

    async def _run_live() -> dict:
        from fused_memory.maintenance._utils import ServiceOpen, maintenance_service

        open_as = ServiceOpen.FULL if args.apply else ServiceOpen.READS
        async with maintenance_service(args.config, open_as=open_as) as (_, memory):
            return await run(args, memory, progress)

    try:
        report = asyncio.run(_run_live())
    except Exception:
        report = report_from_progress(args, progress)
        report['aborted'] = True
        logger.exception(
            '%s: the sweep aborted; the partial report covers the %d record(s) '
            'processed so far',
            SCRIPT_ID, len(report['records']),
        )
        print(json.dumps(report, indent=2, default=str))
        return EXIT_ABORTED

    print(json.dumps(report, indent=2, default=str))
    return resolve_exit_code(report)


if __name__ == '__main__':
    sys.exit(run_cli(main))
