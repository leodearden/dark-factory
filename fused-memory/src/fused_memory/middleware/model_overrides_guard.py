"""model_overrides shape guard (ζ).

Validates ``metadata.model_overrides`` shape at the fused-memory
``submit_task``/``update_task`` boundary, rejecting malformed submissions
with a structured ``ValidationError`` before any persistence happens.

## Contract (PRD plans/adaptive-model-routing-prd.md task ζ, decision 9)

fused-memory validates SHAPE only — known role names + string values — by
delegating to the shared ``validate_model_overrides`` authority
(``shared.task_metadata``). It deliberately does **not** validate model
*strings* against any allowlist: fused-memory does not know the
orchestrator's ``routing.allowed_models``. That check is the orchestrator
resolver's fail-safe job at resolve time — an override naming a model
outside the allowlist is skipped and recorded in
``RoutingDecision.rejected``, never rejected at submit time (a dispatch is
never blocked by a routing mis-config).

Presence-gated like ``deterministic_task_guard._validate_milestone``: an
absent, ``None``, or empty-dict ``model_overrides`` is valid, so every
existing caller that never sets it is unaffected.

CAVEAT — a role key accepted by this shape guard is not automatically
*effective*: ``shared.task_metadata.KNOWN_ROLE_NAMES`` also includes a
few collapsed ``ModelsConfig`` keys ('reviewer', 'triage', 'module_tagger')
that the orchestrator resolver never matches against (it keys strictly on
the full dispatch ``role_name``, e.g. 'reviewer_comprehensive'). An
override pinned under one of those collapsed keys passes here but is
silently inert at resolve time — see the ``KNOWN_ROLE_NAMES`` docstring in
``shared/task_metadata.py`` for the full explanation.
"""

from __future__ import annotations

import json
from typing import Any

from shared.task_metadata import validate_model_overrides

__all__ = [
    'model_overrides_error',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_metadata(metadata: Any) -> dict:
    """Return *metadata* as a dict (best-effort; unknown shapes -> {}).

    Mirrors ``execution_class_guard._parse_metadata`` /
    ``routing_intent_guard._parse_metadata``: None -> {}; empty string ->
    {}; dict -> returned as-is; JSON string -> parsed dict (or {} on parse
    failure / non-dict result); anything else -> {}.
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
# Public guard function
# ---------------------------------------------------------------------------


def model_overrides_error(metadata: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate the ``metadata.model_overrides`` shape invariant.

    Returns a structured error dict (``{'error': ..., 'error_type':
    'ValidationError', 'hint': ...}``) when ``model_overrides`` is present
    and malformed (unknown role name, non-string value, or a non-dict
    value), or ``None`` when it is absent, ``None``, an empty dict, or
    well-formed.

    Gated on *presence*, not truthiness: only a missing key, an explicit
    ``None``, or an explicit ``{}`` are treated as "nothing to validate".
    A wrong-typed-but-falsy scalar (e.g. ``0`` or ``False``) is NOT treated
    as absent -- it reaches ``validate_model_overrides`` below and is
    rejected as a non-dict, same as any other malformed value.

    Args:
        metadata: Task metadata (dict, JSON string, or None).
    """
    parsed = _parse_metadata(metadata)
    if 'model_overrides' not in parsed:
        # Key not present at all -- nothing to validate.
        return None
    model_overrides = parsed['model_overrides']
    if model_overrides is None or model_overrides == {}:
        # Explicit null or explicit empty dict -- benign "no overrides".
        return None
    try:
        validate_model_overrides(model_overrides)
    except ValueError as exc:
        return _validation_error(
            f'metadata.model_overrides is invalid: {exc}',
            hint=(
                'model_overrides must map a known role name to a non-empty '
                "model string, e.g. {'implementer': 'haiku'}."
            ),
        )
    return None
