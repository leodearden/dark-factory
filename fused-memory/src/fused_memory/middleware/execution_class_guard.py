"""Execution-class guard (η).

Validates and injects ``metadata.execution_class`` at the fused-memory
``submit_task`` boundary for recon-stage callers, rejecting submissions that
omit — or misdeclare — the class with a structured ``ValidationError``
before any persistence happens.

## Contract (PRD §8.5; task 2225/W5-η)

Enforcement is scoped to recon-stage callers only: ``agent_id`` must satisfy
``isinstance(agent_id, str) and agent_id.startswith('recon-stage-')``. Every
other caller (human, ``claude-interactive``, an absent agent_id, ...) is
entirely unaffected — ``execution_class_error`` returns ``None``
immediately.

For a recon-stage caller, ``metadata.execution_class`` must be one of the
three canonical values from
:data:`fused_memory.reconciliation.recon_self_model.EXECUTION_CLASSES`
(currently ``'code_tdd'``, ``'operational'``, ``'decision'``) — imported,
not redefined, so the guard's accepted set can never drift from
``render_execution_class_section()``'s rendered-prompt text that instructs
recon agents what to declare. Absent or unrecognized → rejected.

## Non-goal: routing coercion is NOT this task's job

This module deliberately does NOT coerce ``operational``/``decision``
submissions to ``task_kind='deterministic'`` + ``always_escalates=True``.
That routing coercion is owned by task 2085 (MECH B: ``TaskCurator``'s
``route_deterministic`` at the recon Stage-2 emit path) — see the
2026-07-06 correction recorded against this task. η delivers only the
machine-checkable DECLARATION layer: reject-if-missing-or-invalid,
persist-if-valid. This is the explicit, lintable input that 2085's
``route_deterministic`` and W5-ξ's premise-lint consume.

``inject_execution_class`` normalises the metadata dict, providing the
persistence seam that later work (W3/2158) can type against.
"""

from __future__ import annotations

import json
from typing import Any

from fused_memory.reconciliation.recon_self_model import EXECUTION_CLASSES

__all__ = [
    'execution_class_error',
    'inject_execution_class',
]

_VALID_EXECUTION_CLASSES: frozenset[str] = frozenset(EXECUTION_CLASSES)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_metadata(metadata: Any) -> dict:
    """Return *metadata* as a dict (best-effort; unknown shapes → empty dict).

    None → {}; empty string → {}; dict → returned as-is (callers that need
    to mutate must copy first); JSON string → parsed dict, or {} if the
    parse fails or yields a non-dict value; anything else → {}.
    """
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        if not metadata:
            return {}
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
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


def execution_class_error(
    metadata: str | dict[str, Any] | None,
    agent_id: str | None,
    project_root: str,
) -> dict[str, Any] | None:
    """Validate the recon-stage execution_class invariant.

    Returns a structured error dict (``{'error': ..., 'error_type':
    'ValidationError', 'hint': ...}``) when a recon-stage caller's
    submission omits or misdeclares ``metadata.execution_class``, or
    ``None`` when the submission is valid — including for every non-recon
    caller, which is never enforced.

    Args:
        metadata: Task metadata (dict, JSON string, or None).
        agent_id: The resolved caller identity. Enforcement fires only when
            this is a string starting with ``'recon-stage-'``.
        project_root: Absolute path to the project root. Unused — no
            filesystem check is needed for a metadata-only field — but kept
            to mirror ``deterministic_task_error``'s signature (PRD §8.5) so
            the two guards read as a matched pair, and so a future
            project-scoped rule has a home without a signature change.
    """
    if not (isinstance(agent_id, str) and agent_id.startswith('recon-stage-')):
        return None

    execution_class = _parse_metadata(metadata).get('execution_class')
    if execution_class not in _VALID_EXECUTION_CLASSES:
        valid_list = ', '.join(repr(c) for c in sorted(_VALID_EXECUTION_CLASSES))
        return _validation_error(
            f'recon-stage submit_task requires metadata.execution_class in '
            f'{{{valid_list}}} (got {execution_class!r}).',
            hint=(
                "Set metadata.execution_class to 'code_tdd' (default "
                "engineering/TDD path), 'operational', or 'decision'."
            ),
        )
    return None


def inject_execution_class(
    metadata: str | dict[str, Any] | None,
) -> dict[str, Any]:
    """Return *metadata* normalised to a fresh dict (the persistence seam).

    Mirrors ``inject_task_kind``'s normalisation half: None / JSON-string /
    dict / unparseable → a fresh, shallow-copied ``dict`` so the caller's
    object is never mutated and the result is always a plain dict ready for
    JSON serialisation and for later typing (W3/2158).

    Unlike ``inject_task_kind`` this injects *no* default value: a declared
    ``execution_class`` is preserved verbatim, and an absent one stays absent.
    Validity is enforced separately — and earlier — by
    ``execution_class_error`` for recon-stage callers, so by the time a
    recon-stage submission reaches here the field is already present and
    valid; non-recon callers are unconstrained, so the field may legitimately
    be absent.

    Args:
        metadata: Incoming task metadata (dict, JSON string, or None).
    """
    return dict(_parse_metadata(metadata))
