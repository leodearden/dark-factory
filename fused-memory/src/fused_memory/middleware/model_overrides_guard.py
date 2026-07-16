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
absent or empty ``model_overrides`` is valid, so every existing caller that
never sets it is unaffected.
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
    value), or ``None`` when it is absent, empty, or well-formed.

    Args:
        metadata: Task metadata (dict, JSON string, or None).
    """
    model_overrides = _parse_metadata(metadata).get('model_overrides')
    if not model_overrides:
        # Absent, None, or empty -- nothing to validate (presence-gated).
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
