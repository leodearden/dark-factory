"""VerifyResult doubles that the merge lane's disposition classifier can
really classify, plus the log predicate that detects when it could not.

Imported by bare module name (``from _verify_result_doubles import ...``),
like ``_merge_lane_fakes`` -- ``orchestrator/tests/`` has no ``__init__.py``.
The two-sided proof lives in ``orchestrator/tests/test_verify_result_doubles.py``
(the doubles' own contract) and
``orchestrator/tests/test_merge_speculation.py::TestDispositionDoubleFidelity``
(the predicate firing on a bare double and staying silent on these).
"""
from __future__ import annotations

import dataclasses
import logging
import traceback
from typing import Any
from unittest.mock import MagicMock

from orchestrator.verify import VerifyResult


def real_verify_result(passed: bool) -> VerifyResult:
    """Return a real VerifyResult with the given pass/fail status."""
    return VerifyResult(
        passed=passed,
        test_output='ok' if passed else 'FAILED',
        lint_output='',
        type_output='',
        summary='ok' if passed else 'fail',
        category='' if passed else 'test_failure',
    )


_VERIFY_RESULT_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(VerifyResult))


def fake_verify_result(*, passed: bool, **overrides: Any) -> MagicMock:
    """Return a MagicMock(spec=VerifyResult) seeded from a REAL VerifyResult's
    field defaults, with **overrides layered on top (task 3477).

    Why not a bare ``MagicMock(passed=..., summary=..., ...)``: an
    unconfigured attribute READ on a bare MagicMock auto-vivifies a truthy
    child Mock rather than returning a real default. The inline
    VerifyResult-shaped doubles this replaced all omitted ``cause_hint``, so
    ``orchestrator/src/orchestrator/merge_lane/disposition.py::_extract_failing_tests_and_candidate_files``
    joins ``verify_result.cause_hint`` and ``verify_result.test_output`` via
    ``str.join`` — which raised
    ``TypeError: sequence item 0: expected str instance, MagicMock found``
    on every FAILING double.
    ``orchestrator/src/orchestrator/merge_lane/disposition.py::classify_merge_failure_disposition``
    swallows that into a silent fail-open (WARNING + INDETERMINATE), so the
    affected tests only APPEARED to exercise disposition classification.

    Why ``spec=VerifyResult``: it makes a future unknown-attribute READ
    fail loudly (``AttributeError``) instead of auto-vivifying — the
    root-cause fix, not just a ``cause_hint=''`` patch for today's one
    known field. Seeding from ``dataclasses.fields(VerifyResult)`` (rather
    than a hardcoded attr list) means a field added to VerifyResult in
    future is picked up automatically, so this factory does not itself
    drift out of sync the way the inline doubles did.

    Why this stays a MagicMock and not a real VerifyResult instance: a real
    dataclass would also make ``.failure_report()`` return a real string
    instead of a Mock. The ``isinstance(s, str)`` filter in
    ``orchestrator/src/orchestrator/merge_lane/worker.py::_verify_hit_enospc``
    currently discards the Mock; a real string would pass through and
    append a ``## Test Failures`` block to the reason
    ``orchestrator/src/orchestrator/merge_lane/worker.py::_run_post_merge_verify``
    renders — assertion churn well beyond this factory's fidelity-only
    scope. Seeding a MagicMock from real defaults confines the behaviour
    delta to exactly the fields callers read.

    Deliberately does NOT special-case a ``verify_skipped`` kwarg: that
    field is
    ``orchestrator/src/orchestrator/merge_lane/types.py::MergeOutcome.verify_skipped``,
    not a VerifyResult one, and is never read off a VerifyResult anywhere in
    src/ — callers migrating off the old inline doubles should simply drop
    it rather than pass it through **overrides.

    **overrides are validated against VerifyResult's real field names
    before being applied (task 3477 amend). ``spec=`` alone only makes
    unconfigured *reads* fail loudly; a ``spec=`` (not ``spec_set=``) mock
    still accepts an unknown attribute WRITE via plain ``setattr`` — so a
    typo'd override (or a leftover ``verify_skipped=``) would silently
    produce a double whose real field keeps its seeded default instead of
    raising, which is exactly the class of silent drift this factory
    exists to eliminate.
    """
    if unknown := set(overrides) - _VERIFY_RESULT_FIELD_NAMES:
        raise TypeError(
            f'fake_verify_result: not VerifyResult fields: {sorted(unknown)}'
        )
    real = real_verify_result(passed)
    fake = MagicMock(spec=VerifyResult)
    for f in dataclasses.fields(VerifyResult):
        setattr(fake, f.name, getattr(real, f.name))
    for key, value in overrides.items():
        setattr(fake, key, value)
    return fake


# ---------------------------------------------------------------------------
# Disposition fail-open predicate (task 3980 amendment) — the companion guard
# to fake_verify_result above, and deliberately colocated with it.
#
# The classifier has TWO fail-open sites, both emitting the same substring on
# the lane's one logger, 'orchestrator.merge_lane':
#   1. merge_lane/disposition.py::classify_merge_failure_disposition — the
#      classifier's own internal fail-open.
#   2. merge_lane/worker.py::_classify_disposition_for_outcome — the
#      belt-and-suspenders catch for anything site 1 re-raises past.
# Keying on the shared substring catches both; filtering by any narrower
# signal than that silently excludes one of them.
#
# Lives here, beside the factory, rather than in test_merge_speculation.py so
# there is exactly one predicate for one production invariant. Task 3980 first
# wrote a second copy over there, and the two had already diverged before the
# amendment pass: the older inline version keyed on a longer substring that
# matched only site 1. Two copies of a drift-detector is how the drift gets
# back in.
_FAIL_OPEN_SUBSTRING = 'degrading to INDETERMINATE (fail-open, I3)'
FAIL_OPEN_LOGGERS = ('orchestrator.merge_lane',)


def fail_open_records(
    records: list[logging.LogRecord],
) -> list[logging.LogRecord]:
    """Return the disposition fail-open WARNINGs among *records* (both sites).

    NOT symmetric with "the classifier ran": a classifier that SUCCEEDS emits
    NOTHING at WARNING (it logs only on
    ``classify_merge_failure_disposition``'s two degrade paths), so an empty
    return here means "no fail-open", never "the classifier was reached".
    Proving the classifier was reached needs a delegating SPY on the
    callable — see test_merge_speculation.py's ``TestLateArrivalFailCascade``
    for the live one and ``TestDispositionDoubleFidelity`` for the isolated
    two-sided proof. Do not read log silence here as evidence the classifier
    never ran: a successful classification is precisely the silent case.

    Deliberately does NOT match the classifier's *other* degrade
    ('degrading implicated landings to INDETERMINATE (...)'), which is a
    legitimate evidence-absent verdict rather than a swallowed exception.
    """
    return [
        r for r in records
        if r.name in FAIL_OPEN_LOGGERS and _FAIL_OPEN_SUBSTRING in r.getMessage()
    ]


def format_fail_open_records(records: list[logging.LogRecord]) -> str:
    """Render fail-open records for an assertion message, naming the underlying
    exception (the actionable part — the WARNING text alone never says WHY)."""
    return '\n'.join(
        f'  - [{r.name}] {r.getMessage()}'
        + (
            f'\n    underlying: {r.exc_info[1]!r}\n'
            + ''.join(traceback.format_exception(*r.exc_info))
            if r.exc_info
            else ''
        )
        for r in records
    )
