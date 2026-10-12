"""fused_memory.server.manifest_stamping — commit_planning capability-manifest stamper.

Implements task γ of the capability-delivered-checks PRD
(``plans/capability-delivered-checks-prd.md``): at ``commit_planning``, for
each batch task whose metadata carries both ``prd_path`` and
``prd_task_label``, locate the capability-manifest sidecar
(``<prd-stem>.capability-manifest.yaml``), stamp the real ``task_id`` onto
the matching label entry (written back to disk — the decompose session
commits it, this module never does), and copy that label's MECHANICAL
``delivered_checks`` (every ``DeliveredCheckMeta`` kind — ``manual`` alone
is dropped; see :data:`shared.capability_manifest.MECHANICAL_CHECK_KINDS`)
into the producer task's ``metadata.delivered_checks`` via the interceptor's
per-project write lock.

Write-back contract: a stamp changes only ``task_id`` values. Every other
byte of the sidecar — comments, quoting, key order, line endings — is
preserved (:func:`fused_memory.server.manifest_sidecar_text.stamp_task_ids`),
and the result is re-parsed and schema-validated before the atomic replace.
A batch label naming a block whose producer is external (``external_task_id``)
is never stamped; it is reported on its own, and its siblings still stamp. Any
other stamp that cannot be verified, or that would leave an invalid sidecar,
is refused, reported in ``report['errors']``, and nothing is written.

Never raises, and NEVER blocks the ``commit_planning`` flip: stamping runs
after the flip, a sidecar is scoped and may legitimately omit a label, and an
out-of-plan follow-up legitimately carries ``prd_path`` without a
``prd_task_label`` — so a block would reject correct batches with no override.
Loud instead. No sidecar on disk is a complete no-op (``None``: the caller
attaches no ``manifest_stamping`` key). Otherwise every batch task whose
``prd_path`` derives the processed sidecar lands in exactly one bucket —
``stamped``, ``missing_labels``, ``external_labels``, ``near_miss_labels``,
``near_miss_keys`` or ``unlabeled_tasks``, the last four attached only when
non-empty — and a report with anything to repair renders
:func:`stamping_action_required` lines, which ``commit_planning`` attaches as
``manifest_stamping_action_required`` and this module logs once at WARNING.

I/O contract: every filesystem and YAML step runs in one ``asyncio.to_thread``
hop, and a process-wide lock serializes each sidecar read-modify-write.

DELIVERED-CHECK POLARITY REFUSAL (task 3500). Every mechanical check is
linted by :func:`shared.delivered_check_polarity.lint_delivered_checks`
against the authoring tree before it is copied, and a check the lint
REJECTS is dropped from the copy and named in ``report['errors']``.

That is the same lint ``commit_planning`` runs, with the OPPOSITE
contract, and the difference is forced by what each wire point is allowed
to do. ``commit_planning`` is a synchronous gate whose caller is a live
agent that can repair the descriptor and re-commit, so it REJECTS the
whole batch. This helper is contractually never-raising and must not
block the status flip that called it, so the worst it may do is refuse to
copy. Dropping degrades toward the SAFE direction — a dependent
dispatched ungated, exactly as before the delivered-check gate existed —
rather than toward the wedge this task exists to prevent, where a
dependent is blocked forever behind a check that can never go green.

Dispositions below ``reject`` are non-blocking by construction: a ``warn``
(an over-broad ``expect=absent`` pattern, undecidable at authoring time)
is copied and surfaces under the conditional ``report['polarity_warnings']``
key, and an ``errored`` (the check could not be EVALUATED at all — no git,
root not a repo, unresolvable ref) is copied and logged at WARNING. Both
new report keys are attached only when non-empty, so a clean batch's
report keeps its exact four-key legacy shape.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError
from shared.capability_manifest import (
    MECHANICAL_CHECK_KINDS,
    CapabilityManifestDoc,
    DeliveredCheckMeta,
    ManifestTask,
    parse_capability_manifest,
)
from shared.delivered_check_polarity import GATE_REF, lint_delivered_checks
from shared.safe_io import atomic_write_text

from fused_memory.middleware.task_interceptor import interceptor_write_succeeded
from fused_memory.server.manifest_sidecar_text import SidecarStampRefused, stamp_task_ids

logger = logging.getLogger(__name__)

_SIDECAR_SUFFIX = '.capability-manifest.yaml'

#: Serializes every sidecar read-modify-write in this process. The on-disk
#: steps run in worker threads, where two concurrent commit_planning calls on
#: one sidecar would otherwise each stamp a stale read and one stamp would be
#: lost. A threading.Lock, not an asyncio.Lock: it is taken in those threads,
#: and a module-level asyncio.Lock would bind to whichever loop first used it.
_SIDECAR_RMW_LOCK = threading.Lock()

#: Normalized metadata keys that look like a misspelt ``prd_task_label``.
_NEAR_MISS_LABEL_KEYS = frozenset({'prdtasklabel', 'prdlabel', 'tasklabel', 'label'})

#: Spelled out rather than taken from unicodedata, whose name for λ is LAMDA.
#: casefold already maps Σ and final ς to σ.
_GREEK_LETTER_NAMES = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
    'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
    'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi',
    'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi',
    'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
}


@dataclass(frozen=True)
class _PrdBoundTask:
    """A batch task carrying ``prd_path``, and how — if at all — it names its label.

    At most one of ``label`` (the exact ``prd_task_label``, which binds) and
    ``near_miss`` (the ``(key, value)`` of a misspelt label key) is set;
    neither means the task is unlabeled.
    """

    task_id: str
    sidecar_rel: str
    label: Any = None
    near_miss: tuple[str, Any] | None = None


@dataclass(frozen=True)
class _SidecarOutcome:
    """What the on-disk steps 2-4 hand back to the coroutine.

    ``bound`` is ``{prd_task_label: task_id}`` for the batch's labeled tasks
    whose sidecar this is. ``doc`` is None when the sidecar failed to load or
    its stamp failed to write: the report then ends at ``errors``, with
    nothing stamped.
    """

    root: Path
    sidecar_rel: str
    bound: Mapping[str, str]
    doc: CapabilityManifestDoc | None
    stamped: tuple[str, ...]
    errors: tuple[str, ...]


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
            provenance. ``commit_planning`` (the sole caller) has no
            ``agent_id``/``ctx`` parameter of its own — its ``set_task_status``
            call earlier in the same handler is likewise un-attributed — so
            that call site intentionally passes ``None`` here; that is
            parity with the existing handler, not a missed wiring.

    Returns:
        ``None`` when no batch task carries a non-empty ``prd_path``, or
        when no derived sidecar exists on disk — a complete no-op in both
        cases. Otherwise a structured report ``{path, stamped,
        missing_labels, errors}`` (``path`` relative to ``project_root``),
        plus ``external_labels``, ``near_miss_labels``, ``near_miss_keys``,
        ``unlabeled_tasks`` and ``polarity_warnings`` when non-empty; see the
        module docstring.

    This is a thin never-raising wrapper around
    :func:`_stamp_capability_manifests_impl`. The implementation's steps
    3/4/5 already convert their own anticipated failure modes into
    ``report['errors']`` without raising, but ``commit_planning`` (the sole
    caller) has no try/except of its own around this call — it relies
    entirely on this function never raising. This top-level try/except is
    the structural backstop that keeps that promise even for a failure mode
    the implementation didn't anticipate, rather than relying on every
    block inside it being audited by inspection.
    """
    try:
        return await _stamp_capability_manifests_impl(
            project_root=project_root,
            ids=ids,
            tasks_data=tasks_data,
            task_interceptor=task_interceptor,
            agent_id=agent_id,
        )
    except Exception as exc:  # pragma: no cover - structural backstop, see docstring above
        logger.error(
            'stamp_capability_manifests: unexpected error stamping batch ids=%r — '
            'never propagating into commit_planning',
            ids,
            exc_info=True,
        )
        return {
            'path': None,
            'stamped': [],
            'missing_labels': [],
            'errors': [f'unexpected error in stamp_capability_manifests: {exc}'],
        }


async def _stamp_capability_manifests_impl(
    *,
    project_root: str,
    ids: list[str],
    tasks_data: list[Any],
    task_interceptor: Any,
    agent_id: str | None = None,
) -> dict[str, Any] | None:
    """Implementation for :func:`stamp_capability_manifests` — see there for the contract.

    Wrapped by the public function's top-level try/except backstop above.
    Steps 3 (load/validate), 4 (stamp/write), and 5 (mechanical
    delivered_checks copy) below each additionally guard their own unit of
    work so anticipated failures are attributed precisely in
    ``report['errors']`` rather than falling through to the generic
    backstop message.

    Step 5 additionally REFUSES to copy any check the polarity lint rejects
    (task 3500 — see the module docstring for why refusal, not rejection, is
    the right degradation at this wire point). The refusal is per CHECK, not
    per label: dropping a whole label would strip a sound gate along with the
    unsound one. A refused check never rolls back the ``task_id`` stamp step 4
    already committed to disk — that stamp is the decompose session's record
    of which task owns the label, and is correct regardless of any descriptor
    defect on it.
    """
    # 1. Classify every PRD-bound batch task (non-empty prd_path; one
    #    without cannot derive a sidecar and stays out of the report) as
    #    labeled, near-miss key, or unlabeled. The BINDING admission —
    #    truthy prd_path AND truthy prd_task_label — is unchanged, because
    #    scripts/audit_manifest_descriptor_drift.py::_manifest_binding
    #    mirrors it. The sidecar path is derived STRICTLY via regex
    #    substitution — drift-immune vs the historically drifting .md
    #    filename.
    prd_bound = _prd_bound_tasks(ids, tasks_data)
    if not prd_bound:
        return None

    # 2-4 touch the filesystem and parse YAML, so they run off the event
    # loop in one hop — see _stamp_sidecar_on_disk.
    outcome = await asyncio.to_thread(_stamp_sidecar_on_disk, project_root, prd_bound)
    if outcome is None:
        return None
    sidecar_rel = outcome.sidecar_rel
    report: dict[str, Any] = {
        'path': sidecar_rel,
        'stamped': list(outcome.stamped),
        'missing_labels': [],
        'errors': list(outcome.errors),
    }
    doc = outcome.doc
    if doc is None:
        _warn_if_action_required(report)
        return report
    label_to_task_id = outcome.bound

    # 4b. Every one of this sidecar's tasks that was not stamped is recorded
    #     loudly in exactly one bucket rather than silently dropped — e.g.
    #     the decompose session referenced a label that was renamed or
    #     removed from the manifest after authoring (missing_labels: batch
    #     order, deduped, first-occurrence), named a foreign-produced block
    #     (external_labels), transliterated it (near_miss_labels), misspelt
    #     its key (near_miss_keys), or set none (unlabeled_tasks).
    stamped_label_set = set(outcome.stamped)
    report.update(
        _unstamped_buckets(
            [task for task in prd_bound if task.sidecar_rel == sidecar_rel],
            sidecar_tasks=doc.tasks,
            stamped=stamped_label_set,
        )
    )

    # 5. For each stamped label, copy its MECHANICAL delivered_checks into
    #    that producer's metadata.delivered_checks. Mechanical is not an
    #    enumeration kept here — it is every DeliveredCheckMeta kind, since
    #    a metadata entry IS a copied check; only 'manual' is dropped. The
    #    rule is imported as MECHANICAL_CHECK_KINDS rather than restated, so
    #    a new kind cannot be added to the schema and silently missed here
    #    (a dropped check is one the δ gate never sees, which reads as
    #    ungated rather than as failing). A manual-only (or checkless) label
    #    collects an empty list and is skipped, leaving the δ gate a
    #    status-only no-op for it.
    #
    #    Each copy is POLARITY-LINTED first (task 3500) and a rejected check
    #    is refused rather than persisted. The task's declared metadata.files
    #    feeds the lint's scope-sensitive rules, and comes from the batch
    #    tasks_data already in hand — no extra reads.
    files_by_task_id: dict[str, list[str]] = {}
    for tid, batch_task in zip(ids, tasks_data, strict=False):
        if not isinstance(batch_task, dict):
            continue
        batch_meta = batch_task.get('metadata')
        if not isinstance(batch_meta, dict):
            continue
        declared = batch_meta.get('files')
        if isinstance(declared, list):
            files_by_task_id[tid] = [f for f in declared if isinstance(f, str)]
    polarity_warnings: list[dict[str, Any]] = []

    for task in doc.tasks:
        if task.label not in stamped_label_set:
            continue
        # Independently guarded per label (mirrors index_committed_tasks'
        # per-task record_task guard): the WHOLE per-label body — building
        # the mechanical list, the label_to_task_id lookup, and the
        # update_task call itself — is one unit of work, so a failure
        # anywhere in it (including e.g. future schema drift making
        # DeliveredCheckMeta re-validation fail) is recorded and skipped
        # without discarding the sidecar stamp already committed to disk
        # above, nor blocking sibling labels.
        try:
            mechanical: list[dict[str, Any]] = []
            for cap in task.capabilities:
                check = cap.delivered_check
                if check is None or check.kind not in MECHANICAL_CHECK_KINDS:
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
            # Polarity lint (task 3500), AFTER the DeliveredCheckMeta
            # re-validation above so a schema-invalid check is still
            # diagnosed by the existing except-arm rather than relabelled by
            # this one, and BEFORE the write so a rejected check is never
            # persisted. `manifest_path` is threaded through only for the
            # lint (it is not a DeliveredCheckMeta field and never reaches
            # metadata) so the self-reference classifier can tell a match in
            # the descriptor's own PRD family from a real one. Synchronous
            # and git-shelling, hence asyncio.to_thread; never raises.
            findings = await asyncio.to_thread(
                lint_delivered_checks,
                [{**check_meta, 'manifest_path': sidecar_rel} for check_meta in mechanical],
                files=files_by_task_id.get(tid, []),
                repo_root=str(outcome.root),
                ref=GATE_REF,
            )
            refused = {f.check_name: f for f in findings if f.severity == 'reject'}
            for finding in findings:
                if finding.severity == 'reject':
                    continue
                if finding.severity == 'warn':
                    polarity_warnings.append(
                        {
                            'task_id': tid,
                            'label': task.label,
                            'name': finding.check_name,
                            'code': finding.code,
                            'severity': finding.severity,
                            'message': finding.message,
                        }
                    )
                else:
                    # 'errored' — the check could not be EVALUATED. Copied
                    # anyway (fail open on infrastructure) and logged rather
                    # than reported, because an unevaluable check is the
                    # DEFAULT state on any non-git root and a report entry
                    # there would break the exact-shape contract this
                    # report's callers assert (esc-3500-2).
                    logger.warning(
                        'stamp_capability_manifests: could not evaluate '
                        'delivered_check %r for label %s (task %s) — copying it '
                        'unvalidated: %s',
                        finding.check_name, task.label, tid, finding.message,
                    )
            if refused:
                for name, finding in refused.items():
                    logger.warning(
                        'stamp_capability_manifests: refusing to copy '
                        'delivered_check %r for label %s (task %s) — [%s] %s',
                        name, task.label, tid, finding.code, finding.message,
                    )
                    report['errors'].append(
                        f'{sidecar_rel}: refused to copy delivered_check {name!r} '
                        f'for label {task.label!r} (task {tid}) — [{finding.code}] '
                        f'{finding.message}'
                    )
                mechanical = [c for c in mechanical if c.get('name') not in refused]
                # Re-checked AFTER the drop so an all-refused label writes
                # nothing at all — the same arm a checkless label takes.
                if not mechanical:
                    continue
            resp = await task_interceptor.update_task(
                tid,
                project_root,
                metadata=json.dumps({'delivered_checks': mechanical}),
                agent_id=agent_id,
            )
        except Exception as exc:
            logger.warning(
                'stamp_capability_manifests: failed to build/write delivered_checks '
                'for label %s',
                task.label,
                exc_info=True,
            )
            report['errors'].append(
                f'{sidecar_rel}: failed to write delivered_checks for label '
                f'{task.label!r} — {exc}'
            )
            continue

        # update_task's own gates (write-authority floor, directory-lock
        # charter, reconciliation backlog) reject by RETURNING a structured
        # {'success': False, ...} / {'error': ...} dict — they do not raise,
        # so the except above would never see a rejection. Classify
        # explicitly (per update_task's own documented contract) so a
        # rejected write is never silently dropped.
        if not interceptor_write_succeeded(resp):
            logger.warning(
                'stamp_capability_manifests: update_task rejected delivered_checks '
                'write for label %s (task %s): %r',
                task.label,
                tid,
                resp,
            )
            report['errors'].append(
                f'{sidecar_rel}: update_task rejected delivered_checks write for '
                f'label {task.label!r} (task {tid}) — {resp!r}'
            )

    # Non-blocking polarity findings (task 3500), attached in the same
    # CONDITIONAL shape the caller uses for the whole report: only when
    # non-empty, so a clean batch's report keeps its exact four-key legacy
    # shape and every existing exact-dict assertion on it stays valid.
    if polarity_warnings:
        report['polarity_warnings'] = polarity_warnings

    _warn_if_action_required(report)
    return report


def stamping_action_required(report: Mapping[str, Any]) -> list[str]:
    """One remedy line per report entry that needs repair; ``[]`` for a clean report.

    Rendered from the report's buckets, never parsed back. ``polarity_warnings``
    are advisory and already copied, so they need no action here.
    """
    sidecar = report.get('path') or '<no sidecar>'
    lines = [
        f'{sidecar}: prd_task_label {label!r} matches no task block — add its block, '
        f'correct the spelling, or record the deliberate omission in the .md twin'
        for label in report.get('missing_labels', [])
    ]
    lines += [
        f"{sidecar}: task {entry['task_id']} has prd_task_label {entry['label']!r}, but that "
        f"block's producer is external ({entry['external_task_id']}) — a local task cannot "
        f'also own it; drop the prd_task_label, or bind the block by task_id if the '
        f'producer is local'
        for entry in report.get('external_labels', [])
    ]
    lines += [
        f"{sidecar}: task {entry['task_id']} has prd_task_label {entry['label']!r}, which "
        f"only resembles sidecar label {entry['sidecar_label']!r} — the value must equal "
        f'the sidecar label byte-for-byte'
        for entry in report.get('near_miss_labels', [])
    ]
    lines += [
        f"{sidecar}: task {entry['task_id']} names its label under {entry['key']!r} — "
        f'the key must be exactly prd_task_label, valued with the sidecar label byte-for-byte'
        + (f" ({entry['sidecar_label']!r})" if entry['sidecar_label'] is not None else '')
        for entry in report.get('near_miss_keys', [])
    ]
    lines += [
        f'{sidecar}: task {task_id} carries prd_path but no prd_task_label — fine only for '
        f'an out-of-plan follow-up; otherwise set prd_task_label to its sidecar label'
        for task_id in report.get('unlabeled_tasks', [])
    ]
    lines += [f'{sidecar}: stamping error — {error}' for error in report.get('errors', [])]
    return lines


def _warn_if_action_required(report: Mapping[str, Any]) -> None:
    action = stamping_action_required(report)
    if action:
        logger.warning(
            'stamp_capability_manifests: %s needs action:\n%s',
            report.get('path'), '\n'.join(action),
        )


def _prd_bound_tasks(ids: list[str], tasks_data: list[Any]) -> list[_PrdBoundTask]:
    """Step 1: every batch task whose metadata carries a non-empty ``prd_path``, classified."""
    prd_bound: list[_PrdBoundTask] = []
    for tid, task in zip(ids, tasks_data, strict=False):
        meta = task.get('metadata') if isinstance(task, dict) else None
        if not isinstance(meta, dict) or not meta.get('prd_path'):
            continue
        rel = re.sub(r'\.md$', '', meta['prd_path']) + _SIDECAR_SUFFIX
        label = meta.get('prd_task_label')
        if label:
            prd_bound.append(_PrdBoundTask(tid, rel, label=label))
        else:
            prd_bound.append(_PrdBoundTask(tid, rel, near_miss=_near_miss_label_key(meta)))
    return prd_bound


def _near_miss_label_key(meta: Mapping[str, Any]) -> tuple[str, Any] | None:
    for key, value in meta.items():
        normalized = re.sub(r'[^a-z0-9]', '', key.casefold())
        if key != 'prd_task_label' and value and normalized in _NEAR_MISS_LABEL_KEYS:
            return key, value
    return None


def _unstamped_buckets(
    tasks: Sequence[_PrdBoundTask], *, sidecar_tasks: Sequence[ManifestTask], stamped: set[str],
) -> dict[str, list[Any]]:
    """Step 4b: each unstamped task of one sidecar in exactly one bucket.

    ``missing_labels`` is always present (legacy shape); the other buckets
    only when non-empty.
    """
    sidecar_labels = [block.label for block in sidecar_tasks]
    external_producers = _external_producers(sidecar_tasks)
    missing_labels: list[Any] = []
    external_labels: list[dict[str, Any]] = []
    near_miss_labels: list[dict[str, Any]] = []
    near_miss_keys: list[dict[str, Any]] = []
    unlabeled_tasks: list[str] = []
    for task in tasks:
        if task.label is not None:
            if task.label in stamped:
                continue
            if task.label in external_producers:
                external_labels.append({
                    'task_id': task.task_id,
                    'label': task.label,
                    'external_task_id': external_producers[task.label],
                })
                continue
            sidecar_label = _sidecar_label_resembling(task.label, sidecar_labels)
            if sidecar_label is not None:
                near_miss_labels.append(
                    {'task_id': task.task_id, 'label': task.label, 'sidecar_label': sidecar_label}
                )
            elif task.label not in missing_labels:
                missing_labels.append(task.label)
        elif task.near_miss is not None:
            key, value = task.near_miss
            near_miss_keys.append({
                'task_id': task.task_id,
                'key': key,
                'value': value,
                'sidecar_label': _sidecar_label_resembling(value, sidecar_labels),
            })
        else:
            unlabeled_tasks.append(task.task_id)
    optional = {
        'external_labels': external_labels,
        'near_miss_labels': near_miss_labels,
        'near_miss_keys': near_miss_keys,
        'unlabeled_tasks': unlabeled_tasks,
    }
    return {'missing_labels': missing_labels, **{k: v for k, v in optional.items() if v}}


def _external_producers(sidecar_tasks: Sequence[ManifestTask]) -> dict[str, str]:
    """``{label: external_task_id}`` for the blocks another project's task produces."""
    return {
        block.label: block.external_task_id
        for block in sidecar_tasks
        if block.external_task_id is not None
    }


def _sidecar_label_resembling(value: Any, sidecar_labels: Sequence[str]) -> str | None:
    """The first sidecar label *value* equals after folding, if *value* is a string."""
    if not isinstance(value, str):
        return None
    folded = _fold_label(value)
    return next((label for label in sidecar_labels if _fold_label(label) == folded), None)


def _fold_label(label: str) -> str:
    """NFKC, casefold, whitespace removed, each Greek letter spelled out in English."""
    compact = ''.join(unicodedata.normalize('NFKC', label).casefold().split())
    return ''.join(_GREEK_LETTER_NAMES.get(char, char) for char in compact)


def _labels_bound_to(prd_bound: Sequence[_PrdBoundTask], sidecar_rel: str) -> dict[str, str]:
    """``{prd_task_label: task_id}`` for the labeled batch tasks whose sidecar is *sidecar_rel*."""
    return {
        task.label: task.task_id
        for task in prd_bound
        if task.label is not None and task.sidecar_rel == sidecar_rel
    }


def _stamp_sidecar_on_disk(
    project_root: str, prd_bound: Sequence[_PrdBoundTask]
) -> _SidecarOutcome | None:
    """Steps 2-4, synchronous: select the batch's sidecar, load it, stamp it.

    Returns None when no derived sidecar is both contained under
    *project_root* and present on disk — the coroutine's complete no-op.
    Run via ``asyncio.to_thread``; never call it on the event loop.
    """
    root = Path(project_root).resolve()
    sidecar_rel, selection_errors = _select_sidecar(root, prd_bound)
    if sidecar_rel is None:
        return None
    bound = _labels_bound_to(prd_bound, sidecar_rel)
    with _SIDECAR_RMW_LOCK:
        doc, stamped, error = _load_and_stamp(root / sidecar_rel, sidecar_rel, bound)
    errors = selection_errors if error is None else (*selection_errors, error)
    return _SidecarOutcome(root, sidecar_rel, bound, doc, stamped, errors)


def _select_sidecar(
    root: Path, prd_bound: Sequence[_PrdBoundTask]
) -> tuple[str | None, tuple[str, ...]]:
    """Step 2: choose the one sidecar this batch stamps, plus the errors naming the rest."""
    # Keep only distinct rel paths (insertion order) that stay CONTAINED
    # under project_root and whose file actually exists on disk. prd_path is
    # author-controlled task metadata (set by the /prd decompose skill, read
    # here without re-validation) — an absolute path or one containing '../'
    # would otherwise let both the read and the step-4 write-back escape
    # project_root, so every candidate is resolved and containment-checked
    # before it is ever stat-ed.
    seen_rel_paths = list(dict.fromkeys(task.sidecar_rel for task in prd_bound))
    unsafe_rel_paths: list[str] = []
    existing_rel_paths: list[str] = []
    for rel in seen_rel_paths:
        resolved = (root / rel).resolve()
        if not resolved.is_relative_to(root):
            unsafe_rel_paths.append(rel)
            continue
        if resolved.is_file():
            existing_rel_paths.append(rel)
    # A sidecar a labeled task names always outranks one only unlabeled
    # tasks name, so an out-of-plan follow-up cannot displace the stamp.
    claimed = {task.sidecar_rel for task in prd_bound if task.label is not None}
    existing_rel_paths.sort(key=lambda rel: (rel not in claimed, rel))
    if not existing_rel_paths:
        if unsafe_rel_paths:
            # Nothing safe to attach a report to (see the "no sidecar"
            # no-op contract), but a path-traversal attempt is worth a
            # server-side signal even so.
            logger.warning(
                'stamp_capability_manifests: derived sidecar path(s) resolve '
                'outside project_root, refusing to read/write: %r',
                unsafe_rel_paths,
            )
        return None, ()

    # One-sidecar-per-batch is the normal contract. An unexpected second
    # distinct sidecar path is processed as the first in that order
    # (deterministic), with the rest — and any containment-rejected
    # candidates — named loudly in errors rather than silently dropped.
    sidecar_rel = existing_rel_paths[0]
    errors: list[str] = []
    if len(existing_rel_paths) > 1:
        extra = ', '.join(existing_rel_paths[1:])
        errors.append(
            f'multiple capability-manifest sidecars matched this batch; '
            f'processing {sidecar_rel!r}, ignoring: {extra}'
        )
    if unsafe_rel_paths:
        extra = ', '.join(unsafe_rel_paths)
        errors.append(f'sidecar path(s) resolved outside project_root, refused: {extra}')
    return sidecar_rel, tuple(errors)


def _load_and_stamp(
    sidecar_abs: Path, sidecar_rel: str, bound: Mapping[str, str],
) -> tuple[CapabilityManifestDoc | None, tuple[str, ...], str | None]:
    """Steps 3-4: validate the sidecar, then stamp its bound labels' task_ids to disk.

    Returns ``(doc, stamped labels, error)``; ``doc`` is None exactly when
    ``error`` is set, and then nothing was stamped.
    """
    # 3. Read + validate the sidecar via the shared α-loader. Fail-soft/loud:
    #    a malformed sidecar (bad YAML syntax, or a doc that fails α's
    #    pydantic schema) must never raise out of this helper — it's
    #    recorded in errors and stamping is skipped entirely for this
    #    sidecar. Validating BEFORE any mutation below guarantees a
    #    malformed doc leaves the on-disk file byte-identical.
    try:
        # Bytes, then decode: read_text would translate CRLF, and the stamp
        # below must hand back every byte it did not mean to change.
        raw_text = sidecar_abs.read_bytes().decode('utf-8')
        doc = parse_capability_manifest(yaml.safe_load(raw_text))
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        logger.warning(
            'stamp_capability_manifests: failed to load/validate %s', sidecar_rel, exc_info=True,
        )
        return None, (), f'{sidecar_rel}: failed to load/validate sidecar — {exc}'
    except Exception as exc:  # pragma: no cover - defensive fallback, never raise
        logger.warning(
            'stamp_capability_manifests: unexpected error loading %s', sidecar_rel, exc_info=True,
        )
        return None, (), f'{sidecar_rel}: unexpected error loading sidecar — {exc}'

    # 4. Stamp task_id onto each bound label's block and write back to disk.
    #    bound holds only batch tasks whose OWN derived rel path is this
    #    sidecar (relevant only for the unexpected multi-sidecar case). A
    #    block another project produces is skipped here — step 4b reports
    #    it — so it cannot cost its siblings their stamp. A stamp the text
    #    surgery or the schema still refuses is reported and nothing is
    #    written; any other failure (e.g. a disk write error) must not raise
    #    either. Both leave nothing reported as stamped, since the file was
    #    not confirmed written.
    external = _external_producers(doc.tasks)
    stamped_labels = tuple(
        task.label for task in doc.tasks if task.label in bound and task.label not in external
    )
    try:
        stamp = stamp_task_ids(raw_text, {label: int(bound[label]) for label in stamped_labels})
        if stamp.text != raw_text:
            parse_capability_manifest(stamp.document)
            atomic_write_text(sidecar_abs, stamp.text, encoding='utf-8')
    except (SidecarStampRefused, ValidationError) as exc:
        logger.warning(
            'stamp_capability_manifests: refused to stamp %s', sidecar_rel, exc_info=True,
        )
        return None, (), f'{sidecar_rel}: refused to stamp sidecar — {exc}'
    except Exception as exc:
        logger.warning(
            'stamp_capability_manifests: failed to stamp/write %s', sidecar_rel, exc_info=True,
        )
        return None, (), f'{sidecar_rel}: failed to stamp/write sidecar — {exc}'
    return doc, stamped_labels, None
