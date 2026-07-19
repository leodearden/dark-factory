"""Operational routing guard (β).

A deterministic, pure submit-boundary transform that maps the *declared*
``(execution_class, operational_mode)`` to routing metadata, making the
operational→deterministic coercion an **unbypassable boundary invariant**
(PRD ``plans/operational-ask-routing-prd.md``, task β). This deliberately
reverses the "declaration-only, no boundary coercion" scoping of tasks
2085/2563 (owner-greenlit 2026-07-19): the authoritative coercion moves out
of the fail-open curator and onto this boundary inject, which runs on every
submit path (normal, planning_mode, direct interceptor).

## Contract table (first match wins)

| execution_class    | operational_mode   | result                                    |
|--------------------|--------------------|-------------------------------------------|
| ``operational``    | ``gate`` / absent  | pure gate                                 |
| ``decision``       | (any)              | pure gate                                 |
| ``operational``    | ``llm``            | pure gate PLUS the x_ llm-gate marker     |
| ``code_tdd``       | (any)              | untouched (original object returned)      |
| absent / anything else | (any)          | untouched                                 |

"pure gate" = ``task_kind='deterministic'``, ``always_escalates=True``,
``before_done`` dropped — the shape enforced by
``deterministic_task_guard.deterministic_task_error`` (a deterministic task
with no action must always escalate, never be an ill-formed no-op). The
stamp itself is delegated verbatim to
:meth:`TaskInterceptor._inject_deterministic_pure_gate` (INV-5 — the
pure-gate stamp is reused, never re-implemented).

Only exact ``operational_mode == 'llm'`` takes the llm sub-branch;
``operational`` with ``gate``/absent/``None``/any-other mode takes the plain
pure-gate branch (the conservative default). This transform does NOT reject
invalid ``operational_mode`` values — that rule is owned by the shared
``TaskMetadata`` ``Literal`` + ``parse_metadata`` at the backend write
boundary (task ζ integration matrix), not β.

## Guarantees

- **Idempotent** — re-running on already-coerced metadata yields a
  by-value-identical result, so the tools.py→interceptor double-application
  is a no-op.
- **Override-safe** — a hand-supplied ``task_kind='normal'`` / ``before_done``
  on an ``operational``|``decision`` task is overwritten to the pure-gate
  shape.
- **Pure** — no I/O; the caller's object is never mutated on the coercion
  branch (the reused stamp shallow-copies), and the *original* object is
  returned verbatim on the non-coercion branch (a byte-identical no-op for
  every ``code_tdd``/absent/``None``/``str`` submission).

## Import-cycle note

:func:`inject_operational_routing` lazily imports ``TaskInterceptor`` *inside*
the function. ``task_interceptor.py`` imports this module at top level (for
its direct-path twin), so a top-level import of ``TaskInterceptor`` here would
be circular; the lazy import breaks the cycle (by call time ``task_interceptor``
is fully loaded). Lazy imports are an established pattern in this codebase.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    'OPERATIONAL_LLM_GATE_MARKER_KEY',
    'inject_operational_routing',
]

#: Dedicated, machine-checkable marker key stamped True only for
#: execution_class='operational' + operational_mode='llm'. The ``x_`` prefix
#: is census-clean (shared/task_metadata.py exempts ``x_``-prefixed keys from
#: the unknown_key schema-warning census), so no _BLESSED_METADATA_KEYS edit is
#: needed. Task γ/2803 consumes it (or the preserved operational_mode='llm').
OPERATIONAL_LLM_GATE_MARKER_KEY = 'x_operational_llm_gate'

_COERCED_EXECUTION_CLASSES: frozenset[str] = frozenset({'operational', 'decision'})


def _parse_metadata(metadata: Any) -> dict:
    """Return *metadata* as a dict (best-effort; unknown shapes → empty dict).

    Mirrors ``execution_class_guard._parse_metadata``. None → {}; empty string
    → {}; dict → returned as-is (used only to READ declared fields, never
    mutated here); JSON string → parsed dict, or {} if the parse fails or
    yields a non-dict; anything else → {}.
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


def inject_operational_routing(metadata: str | dict[str, Any] | None) -> Any:
    """Coerce a declared operational/decision submission to a deterministic pure gate.

    See the module docstring for the full contract table and guarantees.

    On the coercion branch (``execution_class`` ∈ {``operational``,
    ``decision``}) returns a fresh coerced dict — the caller's object is not
    mutated. On the non-coercion branch (``code_tdd`` / absent / anything
    else) returns the *original* ``metadata`` object verbatim (``None`` stays
    ``None``, a ``str`` stays a ``str``, a dict is the same object), so the
    transform is a byte-identical no-op for every non-operational submission.

    Args:
        metadata: Incoming task metadata (dict, JSON string, or None).
    """
    parsed = _parse_metadata(metadata)
    execution_class = parsed.get('execution_class')
    if execution_class not in _COERCED_EXECUTION_CLASSES:
        # code_tdd / absent / anything not operational|decision → untouched.
        return metadata

    # INV-5: reuse the pure-gate stamp verbatim (never re-implement). Lazy
    # import breaks the task_interceptor↔guard cycle — see module docstring.
    from fused_memory.middleware.task_interceptor import TaskInterceptor

    coerced = TaskInterceptor._inject_deterministic_pure_gate(metadata)
    if execution_class == 'operational' and parsed.get('operational_mode') == 'llm':
        # Dedicated llm-gate marker (task γ/2803 consumes it). operational_mode
        # itself is ALSO preserved by the pure-gate stamp, so γ may read either.
        coerced[OPERATIONAL_LLM_GATE_MARKER_KEY] = True
    return coerced
