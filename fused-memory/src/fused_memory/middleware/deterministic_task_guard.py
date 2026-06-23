"""Deterministic-task-kind guard (α).

Validates and injects ``task_kind`` at the fused-memory ``submit_task``
boundary, blocking ill-formed deterministic task submissions with a clear
diagnostic before any persistence happens.

## Contract (PRD §4 decisions 1-3, §11.1; boundary scenario B10)

A ``submit_task`` call is rejected (returns a structured ``ValidationError``
dict) when any of these invariants is violated:

1. ``task_kind`` not in ``{'normal', 'deterministic'}`` — enum reject.
2. ``task_kind='deterministic'`` ∧ ``before_done is None`` ∧ ``not
   always_escalates`` — ill-formed no-op: a deterministic task must run an
   action or always escalate.
3. ``task_kind='normal'`` ∧ ``before_done is not None`` — before_done is only
   valid on deterministic tasks.
4. When ``before_done`` is present on a deterministic task: must be a dict;
   ``script`` required string that resolves UNDER ``project_root``, exists,
   and is executable (``os.access X_OK``); ``timeout_secs`` required positive
   int.

``inject_task_kind`` normalises the metadata dict and sets
``metadata.task_kind`` so it is persisted alongside ``before_done`` /
``always_escalates`` by the existing metadata persistence path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

__all__ = [
    'deterministic_task_error',
    'inject_task_kind',
]

_VALID_KINDS: frozenset[str] = frozenset({'normal', 'deterministic'})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_metadata(metadata: Any) -> dict:
    """Return *metadata* as a dict (best-effort; unknown shapes → empty dict)."""
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            parsed = json.loads(metadata)
        except (json.JSONDecodeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _validation_error(message: str, *, hint: str | None = None) -> dict[str, Any]:
    """Return a structured ValidationError dict naming the violated invariant."""
    err: dict[str, Any] = {
        'error': message,
        'error_type': 'ValidationError',
    }
    if hint is not None:
        err['hint'] = hint
    return err


# ---------------------------------------------------------------------------
# Public guard functions
# ---------------------------------------------------------------------------


def deterministic_task_error(
    task_kind: str,
    metadata: str | dict[str, Any] | None,
    project_root: str,
) -> dict[str, Any] | None:
    """Validate the deterministic-task-kind invariants.

    Returns a structured error dict (``{'error': ..., 'error_type':
    'ValidationError', ...}``) when any invariant is violated, or ``None``
    when the submission is valid.

    Args:
        task_kind: The caller-supplied kind string ('normal' or 'deterministic').
        metadata: Task metadata (dict, JSON string, or None).
        project_root: Absolute path to the project root (already normalised by
            the tools.py wrapper before this function is called).
    """
    # Invariant 1: enum
    if task_kind not in _VALID_KINDS:
        valid_list = ', '.join(repr(k) for k in sorted(_VALID_KINDS))
        return _validation_error(
            f"task_kind={task_kind!r} is not valid; must be one of {valid_list}.",
            hint=(
                "Use task_kind='normal' (default) for ordinary tasks or "
                "task_kind='deterministic' for tasks with a pre-done action "
                "(before_done) or a mandatory escalation gate (always_escalates)."
            ),
        )

    meta = _parse_metadata(metadata)
    before_done = meta.get('before_done')
    always_escalates = bool(meta.get('always_escalates', False))

    # Invariant 2: deterministic no-op
    if task_kind == 'deterministic' and before_done is None and not always_escalates:
        return _validation_error(
            "ill-formed no-op: a deterministic task must run an action "
            "(before_done) or always escalate (always_escalates=True).",
            hint=(
                "Supply 'before_done': {'script': '<path>', 'timeout_secs': <n>} "
                "and/or set 'always_escalates': True in metadata."
            ),
        )

    # Invariant 3: before_done on a normal task
    if task_kind == 'normal' and before_done is not None:
        return _validation_error(
            "before_done is only valid on deterministic tasks "
            "(task_kind='deterministic').",
            hint=(
                "Either set task_kind='deterministic' to enable the before_done "
                "action, or remove before_done from metadata."
            ),
        )

    # Invariant 4: before_done structural + filesystem checks (only when present)
    if before_done is not None:
        err = _validate_before_done(before_done, project_root)
        if err is not None:
            return err

    return None


def _validate_before_done(before_done: Any, project_root: str) -> dict[str, Any] | None:
    """Validate the before_done payload structurally and against the filesystem.

    Returns a ValidationError dict on failure, or None when valid.
    """
    # Must be a dict
    if not isinstance(before_done, dict):
        return _validation_error(
            f"before_done must be an object (dict), got {type(before_done).__name__!r}.",
            hint="Supply before_done as a JSON object: {'script': '<path>', 'timeout_secs': <n>}.",
        )

    # script: required non-empty string
    script = before_done.get('script')
    if not isinstance(script, str) or not script.strip():
        return _validation_error(
            "before_done.script is required and must be a non-empty string path.",
            hint="Set 'script' to a relative path of an executable file under project_root.",
        )

    # Resolve the script path relative to project_root
    root = Path(project_root).resolve()
    resolved_script = (root / script).resolve()

    # Path containment: reject ../ escapes
    try:
        resolved_script.relative_to(root)
    except ValueError:
        return _validation_error(
            f"before_done.script {script!r} resolves outside project_root — "
            "path traversal is not allowed.",
            hint="Use a path relative to project_root without leading '../' segments.",
        )

    # File must exist and be executable
    if not resolved_script.exists():
        return _validation_error(
            f"before_done.script {script!r} does not exist under project_root.",
            hint=f"Expected an executable file at: {resolved_script}",
        )

    if not os.access(resolved_script, os.X_OK):
        return _validation_error(
            f"before_done.script {script!r} exists but is not executable "
            "(os.X_OK check failed).",
            hint=f"Run: chmod +x {resolved_script}",
        )

    # timeout_secs: required positive int
    timeout_secs = before_done.get('timeout_secs')
    if not isinstance(timeout_secs, int) or isinstance(timeout_secs, bool) or timeout_secs <= 0:
        return _validation_error(
            "before_done.timeout_secs is required and must be a positive integer "
            f"(got {timeout_secs!r}).",
            hint=(
                "Set 'timeout_secs' to a positive integer (e.g. 60). "
                "The runner kills the script and escalates on timeout."
            ),
        )

    return None


def inject_task_kind(
    metadata: str | dict[str, Any] | None,
    task_kind: str,
) -> dict[str, Any]:
    """Return a metadata dict with ``task_kind`` set to *task_kind*.

    Mirrors ``TaskInterceptor._inject_routing_override``: normalises *metadata*
    (None / JSON-string / dict / unparseable → fresh dict) then shallow-copies
    and sets ``metadata['task_kind']``, so the caller's object is not mutated
    and the result is always a plain dict ready for JSON serialisation.

    Args:
        metadata: Incoming task metadata (dict, JSON string, or None).
        task_kind: The validated kind string to persist ('normal' or
            'deterministic').
    """
    meta = _parse_metadata(metadata)
    if isinstance(metadata, dict):
        meta = dict(meta)  # shallow copy — don't mutate caller's dict
    else:
        meta = dict(meta)  # fresh copy from parsed result
    meta['task_kind'] = task_kind
    return meta
