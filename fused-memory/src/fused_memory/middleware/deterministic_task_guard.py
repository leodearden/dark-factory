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
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from shared.task_metadata import Milestone, parse_metadata

__all__ = [
    'deterministic_task_error',
    'inject_task_kind',
]

logger = logging.getLogger(__name__)

_VALID_KINDS: frozenset[str] = frozenset({'normal', 'deterministic'})

# SchemaWarning codes (shared.task_metadata) that mean the WHOLE metadata
# value was discarded — parse_metadata could not produce any dict at all.
# Only these are loud here; 'unknown_key'/'invalid_field'/'invalid_submodel'
# etc. are expected on legitimate curator-internal metadata (before_done,
# always_escalates) and belong to the write boundary (W3-β), not here.
_DISCARD_CODES = frozenset({'unparseable_json', 'not_an_object'})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_metadata_with_discard(metadata: Any) -> tuple[dict, bool]:
    """Parse *metadata* to a raw dict, delegating malformed-input policy to
    :func:`shared.task_metadata.parse_metadata`.

    Returns ``(parsed, discarded)``: ``discarded`` is True when *metadata* is
    non-None but could not be treated as a JSON object at all (parse_metadata's
    ``unparseable_json`` / ``not_an_object`` SchemaWarning codes) — the same
    "whole metadata replaced by a fresh dict" event
    ``TaskInterceptor._inject_routing_override`` already warns on.

    ``parsed`` is always the RAW dict — never ``parse_metadata(...).model_dump()``
    — so unknown keys (e.g. curator-internal ``spawned_from``) round-trip
    byte-for-value (I1) instead of gaining ``TaskMetadata``'s typed-field
    defaults. Only the parse/validate *policy* (what counts as malformed) is
    delegated to shared; the returned value is independently re-derived from
    the input so its shape is untouched.
    """
    if metadata is None:
        return {}, False
    if isinstance(metadata, dict):
        return metadata, False
    if isinstance(metadata, str) and not metadata:
        # Empty string is benign-absent, not a discard — mirrors the
        # pre-collapse `isinstance(raw, str) and raw` guard.
        return {}, False
    _, warnings = parse_metadata(metadata, direction='read')
    if any(w.code in _DISCARD_CODES for w in warnings):
        return {}, True
    # metadata is a JSON string parse_metadata accepted as an object — reparse
    # independently for the raw dict (see docstring: never model_dump()).
    return json.loads(metadata), False


def _parse_metadata(metadata: Any) -> dict:
    """Return *metadata* as a dict (best-effort; unknown shapes → empty dict)."""
    parsed, _ = _parse_metadata_with_discard(metadata)
    return parsed


def _validation_error(message: str, *, hint: str | None = None) -> dict[str, Any]:
    """Return a structured ValidationError dict naming the violated invariant."""
    err: dict[str, Any] = {
        'error': message,
        'error_type': 'ValidationError',
    }
    if hint is not None:
        err['hint'] = hint
    return err


def _validate_milestone(milestone: Any) -> dict[str, Any] | None:
    """Validate metadata.milestone by delegating to the shared Milestone model.

    Returns a ValidationError dict on failure, or None when valid. Delegates
    to :class:`shared.task_metadata.Milestone` (the single valid-shape
    authority for milestone's mode='dated'/'delayed' iff rules) rather than
    re-implementing its checks here, so guard and persistence-layer semantics
    can never drift. ``TypeError`` is caught alongside ``ValidationError`` to
    cover non-mapping milestone values (e.g. a bare string or list) that
    can't be splatted into the model's keyword arguments.
    """
    try:
        Milestone(**milestone)
    except (ValidationError, TypeError) as exc:
        return _validation_error(
            f'metadata.milestone is invalid: {exc}',
            hint=(
                "A milestone must be {'mode': 'dated', 'at': '<ISO-8601 datetime>'} "
                "or {'mode': 'delayed', 'after_secs': <positive int>}."
            ),
        )
    return None


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
    # Use identity check (is True) rather than bool() so that non-bool truthy values
    # (e.g. always_escalates='false', always_escalates=1) do NOT satisfy the gate.
    # bool('false') == True, which would silently accept the opposite of the caller's intent.
    always_escalates = meta.get('always_escalates', False)

    # metadata.milestone is orthogonal to task_kind (allowed on BOTH 'normal' and
    # 'deterministic' tasks) — validated by presence alone, independent of the
    # task_kind branches below (PRD §5 dec 1).
    milestone = meta.get('milestone')
    if milestone is not None:
        err = _validate_milestone(milestone)
        if err is not None:
            return err

    # Invariant 2: deterministic no-op
    if task_kind == 'deterministic' and before_done is None and always_escalates is not True:
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

    # Invariant 3b: always_escalates on a normal task (mirrors the before_done rule)
    if task_kind == 'normal' and always_escalates is True:
        return _validation_error(
            "always_escalates is only valid on deterministic tasks "
            "(task_kind='deterministic').",
            hint=(
                "Either set task_kind='deterministic' to enable the always_escalates "
                "gate, or remove always_escalates from metadata."
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
    parsed, discarded = _parse_metadata_with_discard(metadata)
    if discarded:
        logger.warning(
            'task_metadata.schema_warning: non-dict metadata discarded '
            '(type=%s); using fresh dict. Original value: %r',
            type(metadata).__name__,
            metadata,
        )
    meta = dict(parsed)  # shallow copy — don't mutate caller's dict
    meta['task_kind'] = task_kind
    return meta
