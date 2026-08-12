"""Canonical PHANTOM reconciliation-verdict rule — the row-level detection contract.

A **phantom** verdict is a placeholder ``Judge._parse_verdict``'s except block
fabricates when the judge's raw output could not be parsed into a verdict at
all.  There is no judge opinion behind it — only a loud, fail-closed stand-in
so the run is not silently dropped.  It is deliberately NOT the same thing as a
genuine verdict whose *content* the judge rated ``severity=serious``: that is a
real review outcome and must never be treated as phantom.

Two shapes are recognised, in this order:

1. **Structured marker** (primary, :func:`has_unparseable_marker`) — every row
   written after task 2947 landed (2026-07-23) carries
   ``code == 'unparseable_judge_response'`` on its (sole) fabricated finding.
2. **Legacy unmarked shape** (fallback) — rows written BEFORE 2947 carry no
   ``code`` key at all, so they are identified by the shape ``_parse_verdict``
   has always fabricated: ``severity == 'serious'`` AND exactly one finding AND
   that finding's ``issue`` starts with :data:`UNPARSEABLE_ISSUE_PREFIX`.

All three conjuncts on the legacy branch matter.  The fabrication path only
ever emits ``serious`` (never ``moderate``); the sole genuine ``serious`` row
in the live DB (``bc9459b8``) has FIVE findings, so the single-finding conjunct
is what keeps this fallback from swallowing it; and ``startswith`` (not a
substring test) means a genuine finding that merely *mentions* parsing
somewhere in its prose is not mistaken for the fabricated one.  A future
variant shape that fails one of these conjuncts falls back to the status quo —
still counted as a genuine verdict — rather than silently under-counting.
Fail-closed in the safe direction: over-reporting a review is recoverable,
suppressing a real serious finding is not.

**Why this lives in ``shared`` rather than being imported from ``judge.py``.**
The rule was born in ``fused_memory.reconciliation.judge`` (task 3070), which
is still its model-typed home — but neither of the two consumers that need it
next CAN import that module:

* ``fused_memory.reconciliation.journal`` importing ``judge`` would be a
  CIRCULAR import — ``judge.py`` already does
  ``from fused_memory.reconciliation.journal import ReconciliationJournal`` at
  module scope.
* The ``dashboard`` package deliberately does not depend on ``fused-memory`` at
  all: ``dashboard/pyproject.toml`` declares the workspace deps ``escalation``
  and ``dark-factory-shared`` only, and adding ``fused-memory`` would drag
  graphiti / mem0 / qdrant into the dashboard for one boolean.

Both packages already depend on ``dark-factory-shared``, so ``shared`` is the
one place all three can reach.  This is precisely the situation
``shared/cap_markers.py`` was created for and documents: *"two sites, two
lists, the same blind spot found twice: the fix is one production module both
can import."*  ``judge.is_phantom_verdict`` remains the fused-memory-side
model-typed API and delegates here, so there is exactly ONE spelling of the
rule and 3070's regression net keeps proving it.

**Primitives, not the pydantic model.**  The signature takes
``severity: str`` + ``findings: Sequence[Mapping]`` rather than a
``JudgeVerdict`` so this module needs no ``fused_memory`` import — the
dependency direction that is the whole point.  ``VerdictSeverity`` is a
``StrEnum`` (``fused_memory/models/reconciliation.py``), so a member compares
equal to its value and the same primitive comparison serves both a validated
model field and a raw SQLite row with no conversion.

Not re-exported from ``shared/__init__.py``: ``shared.cap_markers`` and
``shared.invocation_outcome`` are not either, submodule imports are this
package's norm, and ``test_public_api.py`` pins the curated top-level surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

UNPARSEABLE_VERDICT_CODE = 'unparseable_judge_response'
"""Machine-readable marker on a fabricated finding, identifying its verdict as
phantom.

One spelling shared by every producer and consumer — the stamp in
``Judge._parse_verdict``, ``Judge.review_run``'s halt-reason branch,
:func:`is_phantom_verdict_row` below, and ``ReconciliationJournal.get_stats``'
SQL candidate prefilter — so a reword of the marker can never silently desync
the detector from the producer.
"""

UNPARSEABLE_ISSUE_PREFIX = 'Judge response could not be parsed'
"""Prefix of the ``issue`` text ``_parse_verdict`` stamps on its fabricated
finding (constructed there as ``f'{PREFIX}: {e}'``).

Hoisted for the same reason as :data:`UNPARSEABLE_VERDICT_CODE`, plus one of
its own: it is the ONLY signal available on a pre-2947 row, which was written
before the ``code`` marker existed and so cannot be identified any other way.
"""


def has_unparseable_marker(findings: Sequence[Mapping[str, Any]]) -> bool:
    """True iff any entry in *findings* carries the structured
    ``code == 'unparseable_judge_response'`` marker.

    The machine-readable half of phantom detection, and the only branch that
    applies to a post-2947 row.  Deliberately SEVERITY-AGNOSTIC: the marker is
    ground truth about how the row was produced, so a future producer that
    stamps it at some severity other than ``serious`` is still writing a
    phantom.

    Callers pass either a pydantic-validated ``JudgeVerdict.findings``
    (guaranteed ``list[dict]``) or a ``json.loads``-ed SQLite ``findings``
    column.  ``Mapping.get`` is the only operation used, so a raw parsed row
    needs no conversion.
    """
    return any(f.get('code') == UNPARSEABLE_VERDICT_CODE for f in findings)


def is_phantom_verdict_row(
    severity: str, findings: Sequence[Mapping[str, Any]]
) -> bool:
    """True iff this verdict row is a fabricated stand-in for a review that
    never happened, rather than a genuine judge finding.

    See the module docstring for the two recognised shapes and the reasoning
    behind each conjunct.

    Args:
        severity: The verdict's severity, as a plain ``str`` or any ``StrEnum``
            member equal to one (e.g. ``VerdictSeverity.serious``).  Compared
            against the literal ``'serious'`` — never against
            ``VerdictSeverity``, which lives in ``fused_memory`` and must not
            be imported here.
        findings: The verdict's findings, as validated model dicts or a
            ``json.loads``-ed SQLite column.

    Returns:
        ``False`` for any shape not positively recognised, including a
        malformed or empty findings list — an unrecognised row stays counted as
        a genuine verdict rather than silently vanishing from the totals.
    """
    if has_unparseable_marker(findings):
        return True
    if severity == 'serious' and len(findings) == 1:
        issue = findings[0].get('issue')
        if isinstance(issue, str) and issue.startswith(UNPARSEABLE_ISSUE_PREFIX):
            return True
    return False
