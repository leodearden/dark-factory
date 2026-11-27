"""The shared VerifyResult doubles' own contract (``_verify_result_doubles``).

The merge-lane tests lean on ``fake_verify_result`` to reach the real
disposition classifier. That a failing double classifies without failing open
is proven two-sidedly, against the classifier itself, by
``orchestrator/tests/test_merge_speculation.py::TestDispositionDoubleFidelity``.
This module pins the factory's own surface.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from _verify_result_doubles import fake_verify_result, real_verify_result

from orchestrator.verify import VerifyResult


class TestFakeVerifyResultFidelity:
    """`fake_verify_result` is a ``MagicMock(spec=VerifyResult)`` seeded from a
    real VerifyResult, with its overrides validated against VerifyResult's fields.

    It stays a MagicMock rather than a real VerifyResult, so its methods stay
    mocked. On a real one, ``failure_report()`` returns a real report, and
    ``orchestrator/src/orchestrator/merge_lane/worker.py`` appends that report
    to the blocked-merge reasons it renders. That would change assertions in
    tests that only needed a double the classifier can read.
    """

    def test_failing_double_carries_a_real_str_cause_hint(self) -> None:
        """The task-3477 root cause, pinned on its own.

        A bare ``MagicMock(passed=False, ...)`` that omits ``cause_hint``
        auto-vivifies it as a truthy child Mock. The classifier's evidence
        extraction ``str.join``s ``(cause_hint, test_output)`` and raises
        ``TypeError``. ``classify_merge_failure_disposition`` swallows that into
        a silent fail-open (WARNING + INDETERMINATE), so every disposition
        assertion built on such a double is vacuous.
        """
        double = fake_verify_result(
            passed=False,
            summary='test_failure',
            test_output='FAILED',
            category='test_failure',
        )

        assert isinstance(double.cause_hint, str), (
            f'Expected double.cause_hint to be a real str (the VerifyResult '
            f'default is \'\'), not an auto-vivified MagicMock child; got '
            f'{double.cause_hint!r} ({type(double.cause_hint)!r}).'
        )

    def test_rejects_unknown_override_field(self) -> None:
        """An override key that is not a real VerifyResult field must raise
        immediately, not silently no-op (task 3477 amend).

        ``spec=VerifyResult`` alone only guards unconfigured *reads*; a
        plain ``setattr`` on a ``spec=`` (not ``spec_set=``) mock still
        accepts an unknown attribute WRITE. Without this validation, a
        typo'd override (e.g. ``categoy=`` for ``category=``) or a
        leftover ``verify_skipped=`` would silently produce a double whose
        real field keeps its seeded default instead of raising -- the same
        class of silent drift this factory exists to eliminate.
        ``verify_skipped`` is
        ``orchestrator/src/orchestrator/merge_lane/types.py::MergeOutcome.verify_skipped``,
        never a VerifyResult field: a caller migrating off an old inline
        double drops it.
        """
        with pytest.raises(TypeError, match='verify_skipped'):
            fake_verify_result(passed=False, verify_skipped=True)

    @pytest.mark.parametrize('passed', [True, False])
    def test_every_field_is_seeded_from_real_verify_result(self, passed: bool) -> None:
        """Seeding walks ``dataclasses.fields(VerifyResult)``, so a field added
        to VerifyResult is seeded with its real default with no edit to the
        factory."""
        real = real_verify_result(passed)
        assert isinstance(real, VerifyResult)
        assert real.passed == passed

        fake = fake_verify_result(passed=passed)
        for field in dataclasses.fields(VerifyResult):
            assert getattr(fake, field.name) == getattr(real, field.name), (
                f'fake_verify_result(passed={passed}).{field.name} must be '
                f'seeded from real_verify_result({passed}); got '
                f'{getattr(fake, field.name)!r}, expected '
                f'{getattr(real, field.name)!r}.'
            )

    @pytest.mark.parametrize('overrides', [{}, {'trivial': True}])
    def test_derived_properties_answer_from_the_doubles_own_fields(self, overrides: dict[str, Any]) -> None:
        """A VerifyResult property read on the double gives what the real class gives for the
        same fields, never a truthy auto-mock: a mocked ``ran_narrow_gate`` once made the
        LocalRunner skip the unscoped type-check gate and land a red tip (task 5411)."""
        real = dataclasses.replace(real_verify_result(True), **overrides)
        fake = fake_verify_result(passed=True, **overrides)

        for name in ('ran_narrow_gate', 'is_whole_tree_evidence', 'partial_evidence_label'):
            assert getattr(fake, name) == getattr(real, name), name

    def test_derived_property_follows_a_field_set_after_construction(self) -> None:
        fake = fake_verify_result(passed=True)
        fake.trivial = True

        assert fake.is_whole_tree_evidence is False

    def test_override_replaces_the_seeded_value(self) -> None:
        fake = fake_verify_result(passed=False, summary='x', category='lint_failure')

        assert (fake.summary, fake.category) == ('x', 'lint_failure')

    def test_unknown_attribute_read_raises(self) -> None:
        """``spec=VerifyResult`` makes a read of an attribute VerifyResult lacks
        fail loudly instead of auto-vivifying. That covers the next missing
        field as well as today's ``cause_hint``."""
        fake = fake_verify_result(passed=True)

        with pytest.raises(AttributeError):
            _ = fake.verify_skipped
