"""InvocationOutcome sum type + classify_invocation — the W4 cap/error classification seam.

Consolidates the cap/near-cap/auth-failure/CLI-error/wedge classification
logic previously scattered across ``shared.usage_gate`` (Claude/Codex/Gemini
cap string tables, ``detect_cap_hit``) and ``shared.cli_invoke``
(``NON_CAP_CLI_ERROR_MARKERS``, ``is_zero_output_timeout``) into one total,
pure classifier: :func:`classify_invocation`.

This module is additive only — it does not modify ``usage_gate.py`` or
``cli_invoke.py``. Rewiring those consumers to call ``classify_invocation``
is a follow-up task; until then the string tables are intentionally
duplicated between the old and new homes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime

    from shared.cli_invoke import AgentResult

__all__ = [
    'InvocationOutcome',
    'OK',
    'CapHit',
    'NearCap',
    'AuthFailed',
    'CliLocalError',
    'ZeroOutputWedge',
    'Failure',
]


class InvocationOutcome:
    """Base class for the invocation-outcome tagged union.

    Every concrete variant is a frozen dataclass subclass — see ``__all__``
    for the full set. Use ``isinstance(outcome, <Variant>)`` to discriminate.
    """


@dataclass(frozen=True)
class OK(InvocationOutcome):
    """The invocation completed successfully."""


@dataclass(frozen=True)
class CapHit(InvocationOutcome):
    """The backend reported (or the CLI heuristically detected) a usage cap."""

    resets_at: datetime | None
    reason: str


@dataclass(frozen=True)
class NearCap(InvocationOutcome):
    """The backend warned that a usage cap is imminent (not yet blocking)."""

    reason: str


@dataclass(frozen=True)
class AuthFailed(InvocationOutcome):
    """The backend rejected the request as unauthorized (HTTP 401/403)."""

    status: int


@dataclass(frozen=True)
class CliLocalError(InvocationOutcome):
    """A local CLI/usage error occurred that must never be treated as a cap hit."""

    marker: str


@dataclass(frozen=True)
class ZeroOutputWedge(InvocationOutcome):
    """The invocation timed out having produced no transcript turns (a wedge)."""


@dataclass(frozen=True)
class Failure(InvocationOutcome):
    """The invocation failed for a reason not covered by the other variants."""

    kind: str
