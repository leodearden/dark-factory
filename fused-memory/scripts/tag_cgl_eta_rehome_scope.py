#!/usr/bin/env python3
"""One-shot (idempotently re-runnable) maintenance script: prefix every
CGL-eta ``cross_target_rehome`` Mem0 entry's content with an explicit
source-project scope tag (task 2452).

Background
----------
The CGL-eta leak migration (task 2273) wrote Mem0 entries (metadata
``kind='cgl_eta_cross_target_rehome'``) whose true project scope
(``src_project``/``dst_project``/``src_entity``/``dst_entity``) lives only in
metadata, never inline in the human-readable content. Because a rehomed fact
physically lives in its ``dst_project``'s Mem0 collection but references
``src_project``'s task numbers, a plain/semantic search over the dst
collection cannot distinguish the stale foreign fact from the dst project's
own same-numbered task without inspecting metadata -- risking a future
Stage-2/Stage-3 reconciliation cycle misattributing it.

No deletion
-----------
Unlike ``scripts/prune_recon_cycle_summaries.py`` and
``scripts/sweep_orphan_flag_markers.py`` (which delete rather than retag
because their Mem0 path lacked an in-place payload-update primitive), this
script edits IN PLACE via a metadata-forwarding ``Mem0Backend.update`` --
preserving ``created_at`` and every custom metadata key (``src_project`` /
``dst_project`` / ``kind`` / ``source_migration`` / ...) by passing the
existing payload's custom-provenance subset back on update (mem0-owned keys
like ``data``/``hash``/``created_at``/``updated_at`` are stripped first --
mem0 re-derives them itself; see ``_MEM0_MANAGED_METADATA_KEYS``). See
``fused_memory.maintenance.rehome_scope_tag``
for the pure tagging helper (:func:`apply_scope_tag`) this script routes
content through -- the same helper any future rehome path should reuse so
new entries are gated from ever landing untagged.

Two-phase model
----------------
**Phase 1 -- Scan + report (default, dry-run)**: enumerate every known
project's ``cgl_eta_cross_target_rehome`` entries, classify each as
tag/skip, and print a structured JSON report + human-readable summary
table. No writes.

**Phase 2 -- Apply (--apply)**: update the classified tag-set in place via
``memory.mem0.update`` (best-effort; a failed update is logged and excluded,
other updates still proceed). Idempotent -- re-running after a successful
apply finds every entry already tagged and updates nothing.

Direct ``memory.mem0.update`` -- no ``MemoryService`` wrapper, no MCP tool,
no reconciliation event -- a cosmetic provenance tag must not trigger a
recon cycle over these very facts.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/tag_cgl_eta_rehome_scope.py

  # Commit the tags.
  python scripts/tag_cgl_eta_rehome_scope.py --apply

  # Limit to a single project.
  python scripts/tag_cgl_eta_rehome_scope.py --project-id dark_factory

  # Widen the scan window (a project has more than the default scan-limit
  # cgl_eta_cross_target_rehome points) -- this raises the scan only.
  python scripts/tag_cgl_eta_rehome_scope.py --apply --scan-limit 20000

  # Point at a specific fused-memory config file.
  python scripts/tag_cgl_eta_rehome_scope.py --config /path/to/config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from fused_memory.maintenance.project_selection import select_projects
from fused_memory.maintenance.rehome_scope_tag import (
    CGL_ETA_REHOME_KIND,
    apply_scope_tag,
    scope_tag_for,
)
from fused_memory.models.scope import Scope

logger = logging.getLogger('tag_cgl_eta_rehome_scope')

_DEFAULT_SCAN_LIMIT = 10000


# ---------------------------------------------------------------------------
# Pure core: classify_rehome_record
# ---------------------------------------------------------------------------

def classify_rehome_record(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one ``cgl_eta_cross_target_rehome`` record for tagging.

    Reads content from ``record['metadata']['data']`` (Mem0's payload
    content key) and computes ``apply_scope_tag(content, record['metadata'])``.

    Returns a decision dict ``{'id', 'action', 'new_content'}`` where
    ``action`` is one of:

    - ``'skip:no_src_project'`` -- :func:`~fused_memory.maintenance.
      rehome_scope_tag.scope_tag_for` returned ``None`` (no origin project
      to disambiguate against);
    - ``'skip:already_tagged'`` -- a tag was computable but *content*
      already carries it (``apply_scope_tag`` returned it unchanged);
    - ``'tag'`` -- content needs (and gets) the scope tag prepended.

    ``new_content`` always holds ``apply_scope_tag``'s return value, so a
    ``'tag'`` decision can be applied directly and re-applying a ``'skip'``
    decision (if ever done by mistake) is a no-op.

    Pure function -- no I/O.
    """
    metadata = record['metadata']
    content = metadata.get('data') or ''
    new_content = apply_scope_tag(content, metadata)

    if scope_tag_for(metadata) is None:
        action = 'skip:no_src_project'
    elif new_content == content:
        action = 'skip:already_tagged'
    else:
        action = 'tag'

    return {'id': record['id'], 'action': action, 'new_content': new_content}


# ---------------------------------------------------------------------------
# Pure core: audit report
# ---------------------------------------------------------------------------

def build_tag_report(
    decisions_by_project: dict[str, list[dict[str, Any]]],
    applied_ids: set[str],
    dry_run: bool,
    generated_at: str,
    truncated_projects: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the structured JSON-serialisable tag report.

    Mirrors ``prune_recon_cycle_summaries.build_prune_report``'s shape and
    deterministic sorting, adapted to per-record (not per-pool)
    classification.

    Args:
        decisions_by_project: ``{project_id: [classify_rehome_record(...),
            ...]}`` -- one list per project that was scanned.
        applied_ids: Set of memory ids actually updated (empty on dry-run).
        dry_run: True when no writes were made.
        generated_at: ISO timestamp string.
        truncated_projects: project_ids whose scroll came back at/over
            ``--scan-limit`` (possible truncation -- the true pool may be
            larger than what was scanned). Unlike
            ``prune_recon_cycle_summaries`` (which hard-aborts on this
            condition because its deletes are irreversible), this in-place
            tag is idempotent and non-destructive, so truncation is only
            ever surfaced as a report flag -- it never blocks scanning or
            ``--apply``.

    Returns:
        Dict with keys: dry_run, generated_at, projects, totals, changes,
        possibly_truncated, truncated_projects. ``projects`` nests
        ``{project_id: {scanned, taggable, tagged, skipped}}``. ``changes``
        is a deterministically-ordered list of ``{project_id, id, action,
        applied}`` dicts -- one entry per 'tag' decision (skips never
        appear, mirroring build_prune_report's deletions-only 'deletions'
        list). ``possibly_truncated`` is true iff ``truncated_projects`` is
        non-empty; ``truncated_projects`` is always a sorted list.
    """
    projects: dict[str, dict[str, Any]] = {}
    changes: list[dict[str, Any]] = []

    for project_id in sorted(decisions_by_project):
        decisions = decisions_by_project[project_id]
        scanned = len(decisions)
        taggable_ids = [d['id'] for d in decisions if d['action'] == 'tag']
        taggable = len(taggable_ids)
        tagged = sum(1 for mid in taggable_ids if mid in applied_ids)
        skipped = scanned - taggable

        projects[project_id] = {
            'scanned': scanned,
            'taggable': taggable,
            'tagged': tagged,
            'skipped': skipped,
        }

        for mid in sorted(taggable_ids):
            changes.append({
                'project_id': project_id,
                'id': mid,
                'action': 'tag',
                'applied': mid in applied_ids,
            })

    totals: dict[str, Any] = {
        'scanned': sum(p['scanned'] for p in projects.values()),
        'taggable': sum(p['taggable'] for p in projects.values()),
        'tagged': sum(p['tagged'] for p in projects.values()),
        'skipped': sum(p['skipped'] for p in projects.values()),
    }

    sorted_truncated = sorted(truncated_projects) if truncated_projects else []

    return {
        'dry_run': dry_run,
        'generated_at': generated_at,
        'projects': projects,
        'totals': totals,
        'changes': changes,
        'possibly_truncated': bool(sorted_truncated),
        'truncated_projects': sorted_truncated,
    }


# ---------------------------------------------------------------------------
# mem0-owned metadata keys
# ---------------------------------------------------------------------------

# Keys mem0's AsyncMemory._update_memory (site-packages/mem0/memory/main.py,
# ~line 2449) never trusts from a forwarded metadata dict: 'data'/'hash'/
# 'created_at'/'updated_at' are unconditionally recomputed from the update
# call's own arguments and the existing stored point, and 'user_id'/
# 'agent_id'/'run_id'/'actor_id'/'role' are restored from the *currently
# stored* payload (unconditionally for 'actor_id'; whenever absent from what
# was forwarded for the rest). Forwarding stale copies of these currently
# works only because mem0 keeps overwriting/re-deriving them -- an implicit
# coupling to mem0 internals. Stripping them here makes the intent explicit:
# preserve only this record's CUSTOM provenance keys (kind/src_project/
# dst_project/src_entity/dst_entity/source_migration/...).
_MEM0_MANAGED_METADATA_KEYS = frozenset({
    'data', 'hash', 'created_at', 'updated_at',
    'user_id', 'agent_id', 'run_id', 'actor_id', 'role',
})


# ---------------------------------------------------------------------------
# Live shell: apply_tags
# ---------------------------------------------------------------------------

async def apply_tags(
    memory: Any,
    decisions_by_project: dict[str, list[dict[str, Any]]],
    records_by_project: dict[str, list[dict[str, Any]]],
) -> set[str]:
    """Apply every ``'tag'`` decision via ``memory.mem0.update``, best-effort.

    For each project's decisions with ``action == 'tag'``, calls
    ``memory.mem0.update(record_id, new_content, scope, metadata=<custom
    provenance subset of record['metadata']>)`` -- passing the scrolled
    payload back with mem0-owned keys (see ``_MEM0_MANAGED_METADATA_KEYS``)
    stripped, so mem0's payload-overwriting ``_update_memory`` preserves the
    custom provenance keys (``src_project``/``dst_project``/``kind``/
    ``source_migration``/...) instead of wiping them; ``created_at`` and the
    other mem0-owned fields are (re)derived by mem0 itself regardless.

    Mirrors ``prune_recon_cycle_summaries.apply_prune``'s best-effort
    posture: each update is individually guarded by try/except so one bad
    entry cannot abort the batch -- a failure is logged (WARNING) and its id
    excluded from the returned applied set.

    Records classified ``'skip:*'`` (and any decision whose id has no
    matching scrolled record) are never touched.

    Parameters
    ----------
    memory:
        Live (or mock) MemoryService instance.
    decisions_by_project:
        ``{project_id: [classify_rehome_record(...), ...]}``.
    records_by_project:
        ``{project_id: [scrolled record, ...]}`` -- the raw
        ``scroll_by_metadata`` results the decisions were derived from, in
        the same order; used to recover each ``'tag'`` decision's full
        original metadata payload.

    Returns
    -------
    Set of memory ids actually updated.
    """
    applied: set[str] = set()
    for pid, decisions in decisions_by_project.items():
        scope = Scope(project_id=pid)
        records_by_id = {r['id']: r for r in records_by_project.get(pid, [])}
        for decision in decisions:
            if decision['action'] != 'tag':
                continue
            record = records_by_id.get(decision['id'])
            if record is None:
                continue
            provenance_metadata = {
                k: v for k, v in record['metadata'].items()
                if k not in _MEM0_MANAGED_METADATA_KEYS
            }
            try:
                await memory.mem0.update(
                    decision['id'], decision['new_content'], scope,
                    metadata=provenance_metadata,
                )
            except Exception as exc:  # noqa: BLE001 -- best-effort, mirrors apply_prune
                logger.warning(
                    'tag_cgl_eta_rehome_scope: failed to update memory %s '
                    '(project %s): %s',
                    decision['id'], pid, exc,
                )
            else:
                applied.add(decision['id'])
    return applied


# ---------------------------------------------------------------------------
# Live shell: run
# ---------------------------------------------------------------------------

async def run(
    args: argparse.Namespace,
    *,
    memory: Any,
    known_projects_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Scan every selected project's CGL-eta ``cross_target_rehome`` entries,
    classify each as tag/skip, and report.

    Enumerates each selected project's ``cgl_eta_cross_target_rehome``
    entries via ``memory.mem0.scroll_by_metadata(scope, {'kind':
    CGL_ETA_REHOME_KIND}, limit=args.scan_limit)`` -- a deterministic
    Qdrant payload-filtered scroll -- classifies each record via
    :func:`classify_rehome_record`, and assembles the report via
    :func:`build_tag_report`. See the module docstring for the two-phase
    (dry-run default / ``--apply``) model.

    Parameters
    ----------
    args:
        Parsed ``argparse.Namespace`` (apply, project_id, scan_limit).
    memory:
        Live (or mock) MemoryService instance.
    known_projects_map:
        Optional override for the project registry (used in tests to inject
        a fixture map without reading env vars / filesystem).

    Returns
    -------
    The tag report dict (see :func:`build_tag_report`), or an abort payload
    (``{'aborted': True, ...}``) if an unknown ``--project-id`` was given.
    """
    generated_at = datetime.now(UTC).isoformat()

    if known_projects_map is None:
        from fused_memory.config.schema import FusedMemoryConfig as _FMC  # noqa: PLC0415
        from fused_memory.models.scope import build_known_projects_map  # noqa: PLC0415
        cfg = _FMC()
        known_projects_map = build_known_projects_map(
            cfg.taskmaster.project_root if cfg.taskmaster else '',
        )
        if len(known_projects_map) <= 1:
            logger.warning(
                'tag_cgl_eta_rehome_scope: known-projects map resolved to only '
                '%s. If you expected the full project registry, run with the '
                'fused-memory service environment (PROJECT_ROOT + '
                'DASHBOARD_KNOWN_PROJECT_ROOTS) -- otherwise this scan '
                'silently skips every other project.',
                sorted(known_projects_map),
            )

    try:
        project_ids = select_projects(known_projects_map, args.project_id)
    except ValueError as exc:
        abort_payload: dict[str, Any] = {
            'error': str(exc),
            'aborted': True,
            'dry_run': not args.apply,
            'generated_at': generated_at,
        }
        print(json.dumps(abort_payload, indent=2, default=str))
        print(
            f'ABORT: unknown --project-id {args.project_id!r}; '
            f'known: {sorted(known_projects_map)}',
            file=sys.stderr,
        )
        return abort_payload

    # Scan + classify every selected project.
    decisions_by_project: dict[str, list[dict[str, Any]]] = {}
    records_by_project: dict[str, list[dict[str, Any]]] = {}
    truncated_projects: set[str] = set()
    for pid in project_ids:
        scope = Scope(project_id=pid)
        scrolled = await memory.mem0.scroll_by_metadata(
            scope, {'kind': CGL_ETA_REHOME_KIND}, limit=args.scan_limit,
        )
        if not isinstance(scrolled, list):
            logger.warning(
                'tag_cgl_eta_rehome_scope: mem0.scroll_by_metadata returned a '
                'non-list result for project %s; treating as empty. got: %s',
                pid, repr(scrolled)[:200],
            )
            scrolled = []

        # A scroll that comes back at/over scan_limit is ambiguous -- the
        # true pool may be larger than what was scanned (Qdrant's scroll
        # fills up to `limit` in one pass). Unlike prune_recon_cycle_summaries
        # (irreversible deletes -- hard-abort), this in-place tag is
        # idempotent and non-destructive, so surface it as a report flag
        # and keep going: a follow-up re-run with a larger --scan-limit
        # safely completes coverage.
        if args.scan_limit and len(scrolled) >= args.scan_limit:
            truncated_projects.add(pid)
            logger.warning(
                'tag_cgl_eta_rehome_scope: scan for project %s hit '
                '--scan-limit=%s; results may be incomplete -- re-run with '
                'a larger --scan-limit for full coverage.',
                pid, args.scan_limit,
            )

        records_by_project[pid] = scrolled
        decisions_by_project[pid] = [classify_rehome_record(r) for r in scrolled]

    # Apply or dry-run.
    applied_ids: set[str] = set()
    if args.apply:
        applied_ids = await apply_tags(memory, decisions_by_project, records_by_project)

    report = build_tag_report(
        decisions_by_project=decisions_by_project,
        applied_ids=applied_ids,
        dry_run=not args.apply,
        generated_at=generated_at,
        truncated_projects=truncated_projects,
    )

    # JSON report goes to stdout (machine-readable); summary to stderr.
    print(json.dumps(report, indent=2, default=str))
    print(
        f"CGL-eta rehome-scope tag report "
        f"({'DRY RUN' if not args.apply else 'APPLIED'}) at {generated_at}: "
        f"scanned={report['totals']['scanned']} "
        f"taggable={report['totals']['taggable']} "
        f"tagged={report['totals']['tagged']} "
        f"skipped={report['totals']['skipped']}",
        file=sys.stderr,
    )
    if report['possibly_truncated']:
        print(
            f"NOTE: scan possibly truncated for projects "
            f"{report['truncated_projects']} (hit --scan-limit="
            f"{args.scan_limit}); re-run with a larger --scan-limit for "
            f"full coverage. This is a coverage warning only -- tagging is "
            f"idempotent and non-destructive, so nothing was blocked.",
            file=sys.stderr,
        )

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for this script.

    Extracted from ``main()`` so tests can assert flag defaults without
    invoking the live ``main()`` entry point (mirrors
    ``prune_recon_cycle_summaries.build_parser``).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--apply', action='store_true', default=False,
        help='Commit the scope-tag updates (default: dry-run, report only)',
    )
    parser.add_argument(
        '--project-id', dest='project_id', default=None,
        help='Restrict the scan/tag to a single project_id',
    )
    parser.add_argument(
        '--scan-limit', dest='scan_limit', type=int, default=_DEFAULT_SCAN_LIMIT,
        help='Per-project Mem0 scan window: max cgl_eta_cross_target_rehome '
             'points enumerated via the metadata-filtered scroll '
             f'(default: {_DEFAULT_SCAN_LIMIT}).',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to a fused-memory config file (sets CONFIG_PATH before loading).',
    )
    return parser


def main() -> int:
    """CLI entry point: parse args, build a live MemoryService, run, report.

    Mirrors ``prune_recon_cycle_summaries.main`` (live-service construction,
    exit code derived from ``report.get('aborted')``), plus the ``--config``
    -> ``CONFIG_PATH`` env passthrough from ``migrate_cross_graph_leak.main``.
    """
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.config:
        import os  # noqa: PLC0415
        os.environ['CONFIG_PATH'] = str(args.config)

    async def _run_live() -> int:
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        config = FusedMemoryConfig()
        memory = MemoryService(config)
        await memory.initialize()
        try:
            report = await run(args, memory=memory)
        finally:
            if hasattr(memory, 'close'):
                await memory.close()
        return 1 if report.get('aborted') else 0

    return asyncio.run(_run_live())


if __name__ == '__main__':
    sys.exit(main())
