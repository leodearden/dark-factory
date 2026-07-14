"""fused_memory.server.manifest_stamping — commit_planning capability-manifest stamper.

Implements task γ of the capability-delivered-checks PRD
(``plans/capability-delivered-checks-prd.md``): at ``commit_planning``, for
each batch task whose metadata carries both ``prd_path`` and
``prd_task_label``, locate the capability-manifest sidecar
(``<prd-stem>.capability-manifest.yaml``), stamp the real ``task_id`` onto
the matching label entry (written back to disk — the decompose session
commits it, this module never does), and copy that label's MECHANICAL
``delivered_checks`` (``grep``/``script`` kinds only — never ``manual``)
into the producer task's ``metadata.delivered_checks`` via the interceptor's
per-project write lock.

Fail-soft, loud: no sidecar on disk is a complete no-op (``None`` — the
caller attaches no ``manifest_stamping`` key, keeping the legacy response
byte-identical); a malformed sidecar or absent labels populate
``report['errors']`` but never raise and never block the ``commit_planning``
status flip that called this helper.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from shared.capability_manifest import DeliveredCheckMeta, parse_capability_manifest

logger = logging.getLogger(__name__)

_SIDECAR_SUFFIX = '.capability-manifest.yaml'


async def stamp_capability_manifests(
    *,
    project_root: str,
    ids: list[str],
    tasks_data: list[Any],
    task_interceptor: Any,
    agent_id: str | None = None,
) -> dict[str, Any] | None:
    """Stamp task_ids and copy mechanical delivered_checks for a commit_planning batch.

    Args:
        project_root: Absolute project root (matches ``commit_planning``'s
            already-normalized value); sidecar paths resolve under this.
        ids: The batch's task ids, id-aligned with ``tasks_data`` (same
            ordering ``commit_planning`` used for its ``asyncio.gather``
            read).
        tasks_data: The already-fetched ``get_task``-shaped dicts for
            ``ids`` — reused as-is, no extra reads. Non-dict entries (e.g.
            unconfigured mocks in unit tests) are skipped.
        task_interceptor: The live ``TaskInterceptor`` — ``update_task`` is
            called per stamped label that carries mechanical checks.
        agent_id: Forwarded to ``task_interceptor.update_task`` for
            provenance.

    Returns:
        ``None`` when no batch task carries both a non-empty ``prd_path``
        and ``prd_task_label``, or when the derived sidecar file doesn't
        exist on disk — a complete no-op in both cases. Otherwise a
        structured report ``{path, stamped, missing_labels, errors}``
        (``path`` relative to ``project_root``).
    """
    # 1. Build the manifest-bearing set: (task_id, prd_task_label, derived
    #    sidecar rel path) for batch tasks whose metadata carries BOTH a
    #    non-empty prd_path and prd_task_label. The sidecar path is derived
    #    STRICTLY via regex substitution — drift-immune vs the historically
    #    drifting .md filename.
    manifest_tasks: list[tuple[str, str, str]] = []
    for tid, task in zip(ids, tasks_data, strict=False):
        if not isinstance(task, dict):
            continue
        meta = task.get('metadata')
        if not isinstance(meta, dict):
            continue
        prd_path = meta.get('prd_path')
        prd_task_label = meta.get('prd_task_label')
        if not prd_path or not prd_task_label:
            continue
        rel = re.sub(r'\.md$', '', prd_path) + _SIDECAR_SUFFIX
        manifest_tasks.append((tid, prd_task_label, rel))

    if not manifest_tasks:
        return None

    # 2. Keep only distinct rel paths (insertion order) whose file actually
    #    exists on disk under project_root.
    root = Path(project_root)
    seen_rel_paths: list[str] = []
    for _tid, _label, rel in manifest_tasks:
        if rel not in seen_rel_paths:
            seen_rel_paths.append(rel)
    existing_rel_paths = sorted(rel for rel in seen_rel_paths if (root / rel).is_file())
    if not existing_rel_paths:
        return None

    # One-sidecar-per-batch is the normal contract. An unexpected second
    # distinct sidecar path is processed as the lexicographically-first
    # (deterministic), with the rest named loudly in errors rather than
    # silently dropped.
    sidecar_rel = existing_rel_paths[0]
    report: dict[str, Any] = {
        'path': sidecar_rel,
        'stamped': [],
        'missing_labels': [],
        'errors': [],
    }
    if len(existing_rel_paths) > 1:
        extra = ', '.join(existing_rel_paths[1:])
        report['errors'].append(
            f'multiple capability-manifest sidecars matched this batch; '
            f'processing {sidecar_rel!r}, ignoring: {extra}'
        )

    # 3. Read + validate the sidecar via the shared α-loader. Fail-soft/loud:
    #    a malformed sidecar (bad YAML syntax, or a doc that fails α's
    #    pydantic schema) must never raise out of this helper — it's
    #    recorded in report['errors'] and stamping is skipped entirely for
    #    this sidecar. Validating BEFORE any mutation below guarantees a
    #    malformed doc leaves the on-disk file byte-identical.
    sidecar_abs = root / sidecar_rel
    try:
        raw_text = sidecar_abs.read_text(encoding='utf-8')
        raw = yaml.safe_load(raw_text)
        doc = parse_capability_manifest(raw)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        logger.warning(
            'stamp_capability_manifests: failed to load/validate %s', sidecar_rel, exc_info=True,
        )
        report['errors'].append(f'{sidecar_rel}: failed to load/validate sidecar — {exc}')
        return report
    except Exception as exc:  # pragma: no cover - defensive fallback, never raise
        logger.warning(
            'stamp_capability_manifests: unexpected error loading %s', sidecar_rel, exc_info=True,
        )
        report['errors'].append(f'{sidecar_rel}: unexpected error loading sidecar — {exc}')
        return report

    # 4. Stamp task_id onto each matching label entry and write back to disk.
    #    Scoped to batch tasks whose OWN derived rel path is this sidecar
    #    (relevant only for the unexpected multi-sidecar case above). Also
    #    guarded — an unexpected failure here (e.g. a disk write error)
    #    must not raise, and must leave report['stamped'] empty since the
    #    file was not confirmed written.
    label_to_task_id: dict[str, str] = {
        label: tid for tid, label, rel in manifest_tasks if rel == sidecar_rel
    }
    stamped_labels: list[str] = []
    try:
        for entry in raw.get('tasks', []):
            if not isinstance(entry, dict):
                continue
            label = entry.get('label')
            if label in label_to_task_id:
                entry['task_id'] = int(label_to_task_id[label])
                stamped_labels.append(label)
        if stamped_labels:
            sidecar_abs.write_text(
                yaml.safe_dump(raw, sort_keys=False, allow_unicode=True),
                encoding='utf-8',
            )
    except Exception as exc:
        logger.warning(
            'stamp_capability_manifests: failed to stamp/write %s', sidecar_rel, exc_info=True,
        )
        report['errors'].append(f'{sidecar_rel}: failed to stamp/write sidecar — {exc}')
        return report
    report['stamped'] = stamped_labels

    # 4b. Batch prd_task_labels that don't appear among this sidecar's own
    #     task labels are recorded loudly rather than silently dropped —
    #     e.g. the decompose session referenced a label that was renamed or
    #     removed from the manifest after authoring. Order follows the
    #     batch (manifest_tasks) order, deduped, first-occurrence.
    stamped_label_set = set(stamped_labels)
    missing_labels: list[str] = []
    for _tid, label, rel in manifest_tasks:
        if rel == sidecar_rel and label not in stamped_label_set and label not in missing_labels:
            missing_labels.append(label)
    report['missing_labels'] = missing_labels

    # 5. For each stamped label, copy only MECHANICAL (grep/script)
    #    delivered_checks into that producer's metadata.delivered_checks —
    #    a manual-only (or checkless) label collects an empty list and is
    #    skipped, leaving the δ gate a status-only no-op for it.
    for task in doc.tasks:
        if task.label not in stamped_label_set:
            continue
        mechanical: list[dict[str, Any]] = []
        for cap in task.capabilities:
            check = cap.delivered_check
            if check is None or check.kind not in ('grep', 'script'):
                continue
            mechanical.append(
                DeliveredCheckMeta(
                    name=cap.name,
                    kind=check.kind,
                    pattern=check.pattern,
                    expect=check.expect,
                    paths=check.paths,
                    script=check.script,
                    args=check.args,
                    timeout_secs=check.timeout_secs,
                ).model_dump()
            )
        if not mechanical:
            continue
        tid = label_to_task_id[task.label]
        # Independently guarded per label (mirrors index_committed_tasks'
        # per-task record_task guard): one label's metadata-write failure
        # is recorded and skipped without discarding the sidecar stamp
        # already committed to disk above, nor blocking sibling labels.
        try:
            await task_interceptor.update_task(
                tid,
                project_root,
                metadata=json.dumps({'delivered_checks': mechanical}),
                agent_id=agent_id,
            )
        except Exception as exc:
            logger.warning(
                'stamp_capability_manifests: update_task failed for label %s (task %s)',
                task.label,
                tid,
                exc_info=True,
            )
            report['errors'].append(
                f'{sidecar_rel}: failed to write delivered_checks for label '
                f'{task.label!r} (task {tid}) — {exc}'
            )

    return report
