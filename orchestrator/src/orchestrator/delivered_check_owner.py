"""orchestrator.delivered_check_owner — who declares a failing delivered check
(task 7514, absorbing 4934).

A ``DEP_CAPABILITY_NOT_DELIVERED`` escalation is filed against the blocked
DEPENDENT, but the failing ``metadata.delivered_checks`` entry belongs to the
DEPENDENCY. This module is the one home for "where is that check declared":

- :func:`derive_check_owner` is pure — the scheduler calls it at tick time, so
  no file is read and no YAML parsed there.
- :func:`verify_check_owner` reads the owner's sidecar from ``main`` (never the
  working tree, which the machine-operated checkout may have dirty) and
  resolves whether it declares the check. Best-effort: it never raises, and no
  gate verdict depends on it.
- :func:`render_check_owner` is pure and renders the detail block from the
  same value the record stores, so prose and data cannot disagree.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pydantic
import yaml
from escalation.models import CheckSidecarSite, DeliveredCheckOwner, SidecarOnMain
from shared.capability_manifest import parse_capability_manifest, sidecar_path
from shared.delivered_check_polarity import GATE_REF, CheckOutcome, evaluate_path_at_tree

_GIT_SHOW_TIMEOUT_SECS = 30

_SIDECAR_LINE: dict[SidecarOnMain, str] = {
    'declares': 'Sidecar: {path} declares it too on main; a correction edits both.',
    'does_not_declare': (
        "Sidecar: {path} (from task {o}'s prd_path) does not declare it on main; "
        'the task metadata is the only site.'
    ),
    'unverified': (
        "Sidecar: {path} (from task {o}'s prd_path) could not be read on main; "
        'check it before editing.'
    ),
}
_NO_SIDECAR_LINE = 'Sidecar: none (task {o} has no prd_path); the task metadata is the only site.'
_OWNER_LINES = (
    'Check owner: task {o}, the dependency. The blocked task does not declare this check.',
    "Declared in: task {o} metadata.delivered_checks, entry {check_name!r}.",
)
_REMEDY_LINE = (
    "Remedy: if the capability is already on main, the check is stale or mis-authored; "
    "correct it at task {o}'s sites above. If the capability is missing from main, "
    "the gap is task {o}'s."
)
_RE_PEND_LINE = (
    'Blocked dependents are NOT auto-re-pended — set task {dependent} back to pending '
    'once the check passes on main.'
)


def derive_check_owner(
    *,
    dep_id: str,
    dep_metadata: Mapping[str, Any],
    check_name: str,
) -> DeliveredCheckOwner:
    """The owner of *check_name*, with its sidecar path derived and unverified.

    ``prd_path`` is admitted exactly as the stamper admits it: a non-empty str.
    """
    prd_path = dep_metadata.get('prd_path')
    sidecar: CheckSidecarSite | None = None
    if isinstance(prd_path, str) and prd_path:
        sidecar = {'path': sidecar_path(prd_path), 'on_main': 'unverified'}
    return {'task_id': dep_id, 'check_name': check_name, 'sidecar': sidecar}


def verify_check_owner(
    owner: DeliveredCheckOwner,
    *,
    project_root: str | Path,
    ref: str = GATE_REF,
) -> DeliveredCheckOwner:
    """*owner* with ``sidecar.on_main`` resolved at *ref*.

    Synchronous and shelling out to git, so callers off-load it. An owner
    without a sidecar is returned unchanged. Git, YAML and schema failures
    resolve to ``'unverified'``; none raises.
    """
    site = owner['sidecar']
    if site is None:
        return owner
    on_main = _sidecar_on_main(owner, site['path'], project_root=project_root, ref=ref)
    return {**owner, 'sidecar': {'path': site['path'], 'on_main': on_main}}


def render_check_owner(owner: DeliveredCheckOwner | None, *, dependent_id: str) -> str:
    """The owner / declaration sites / remedy / re-pend block of the detail.

    The dependent's id appears only in the final re-pend line. With no owner
    (a caller that predates the owner keywords) only that line is rendered.
    """
    re_pend = _RE_PEND_LINE.format(dependent=dependent_id)
    if owner is None:
        return re_pend
    o = owner['task_id']
    site = owner['sidecar']
    sidecar_line = (
        _NO_SIDECAR_LINE.format(o=o)
        if site is None
        else _SIDECAR_LINE[site['on_main']].format(path=site['path'], o=o)
    )
    return '\n'.join(
        [
            *(line.format(o=o, check_name=owner['check_name']) for line in _OWNER_LINES),
            sidecar_line,
            _REMEDY_LINE.format(o=o),
            re_pend,
        ]
    )


def _sidecar_on_main(
    owner: DeliveredCheckOwner,
    path: str,
    *,
    project_root: str | Path,
    ref: str,
) -> SidecarOnMain:
    exists = evaluate_path_at_tree([path], expect='present', repo_root=project_root, ref=ref)
    if exists is CheckOutcome.ERRORED:
        return 'unverified'
    if exists is CheckOutcome.FAIL:
        return 'does_not_declare'
    text = _show_at_ref(path, project_root=project_root, ref=ref)
    if text is None:
        return 'unverified'
    try:
        doc = parse_capability_manifest(yaml.safe_load(text))
    except (yaml.YAMLError, pydantic.ValidationError):
        return 'unverified'
    declares = any(
        block.task_id is not None
        and str(block.task_id) == owner['task_id']
        and any(
            capability.name == owner['check_name'] and capability.delivered_check is not None
            for capability in block.capabilities
        )
        for block in doc.tasks
    )
    return 'declares' if declares else 'does_not_declare'


def _show_at_ref(path: str, *, project_root: str | Path, ref: str) -> str | None:
    try:
        completed = subprocess.run(
            ['git', '-C', str(project_root), 'show', f'{ref}:{path}'],
            capture_output=True,
            text=True,
            timeout=_GIT_SHOW_TIMEOUT_SECS,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return completed.stdout if completed.returncode == 0 else None
