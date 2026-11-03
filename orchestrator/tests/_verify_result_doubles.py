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
    """Return a ``MagicMock(spec=VerifyResult)`` whose every field is seeded
    from ``real_verify_result(passed)``, with *overrides* layered on top.

    Reading an attribute VerifyResult lacks raises AttributeError; an
    *overrides* key that is not a VerifyResult field raises TypeError. Methods
    such as ``failure_report()`` stay mocked. The reasons behind each choice
    are pinned in ``orchestrator/tests/test_verify_result_doubles.py``.
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
# Disposition fail-open predicate. The classifier fails open at two sites, and
# both log the same substring on the lane's one logger, 'orchestrator.merge_lane':
#   1. orchestrator/src/orchestrator/merge_lane/disposition.py::classify_merge_failure_disposition
#   2. orchestrator/src/orchestrator/merge_lane/worker.py::_classify_disposition_for_outcome
# Keying on that shared substring catches both. Keep this the only copy of the
# predicate, so one production invariant has one detector.
_FAIL_OPEN_SUBSTRING = 'degrading to INDETERMINATE (fail-open, I3)'
FAIL_OPEN_LOGGERS = ('orchestrator.merge_lane',)


def fail_open_records(
    records: list[logging.LogRecord],
) -> list[logging.LogRecord]:
    """Return the disposition fail-open WARNINGs among *records* (both sites).

    An empty return means "no fail-open", never "the classifier ran": a
    successful classification logs nothing at WARNING. Proving the classifier
    was reached needs a delegating spy on it, as in
    ``orchestrator/tests/test_merge_speculation.py::TestLateArrivalFailCascade``.

    The classifier's other degrade ('degrading implicated landings to
    INDETERMINATE (...)') is a legitimate evidence-absent verdict and is not
    matched.
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
