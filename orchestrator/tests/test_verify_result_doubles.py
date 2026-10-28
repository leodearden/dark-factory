"""The shared VerifyResult doubles' own contract (``_verify_result_doubles``).

The merge-lane tests lean on these doubles to reach the real disposition
classifier, so what each one returns is pinned here through their public
surface.
"""
from __future__ import annotations

import dataclasses

import pytest
from _verify_result_doubles import fake_verify_result, real_verify_result

from orchestrator.merge_lane.disposition import (
    _extract_failing_tests_and_candidate_files,
)
from orchestrator.verify import VerifyResult


class TestFakeVerifyResultFidelity:
    """`fake_verify_result` doubles must survive the real disposition-
    classification extraction path, unlike a bare `MagicMock(passed=...)`.
    """

    def test_failing_double_survives_disposition_extraction(self) -> None:
        """A failing `fake_verify_result` double must not TypeError inside
        `_extract_failing_tests_and_candidate_files`.
        """
        double = fake_verify_result(
            passed=False,
            summary='test_failure',
            test_output='FAILED',
            category='test_failure',
        )

        failing_tests, candidate_files = _extract_failing_tests_and_candidate_files(
            double,
        )
        assert (failing_tests, candidate_files) == ((), ()), (
            f'Expected an uninformative failing double to degrade to the '
            f'honest empty-candidate-set return, not raise or manufacture a '
            f'false candidate; got {(failing_tests, candidate_files)!r}.'
        )

        # Root-cause pin: a bare MagicMock auto-vivifies `.cause_hint` as a
        # truthy child Mock (not a str) — that is the whole bug. The factory
        # must seed a real string default (VerifyResult's own default: '')
        # instead.
        assert isinstance(double.cause_hint, str), (
            f'Expected double.cause_hint to be a real str (the VerifyResult '
            f'default is \'\'), not an auto-vivified MagicMock child; got '
            f'{double.cause_hint!r} ({type(double.cause_hint)!r}). This is '
            f'the root cause of the file-wide fail-open: '
            f'_extract_failing_tests_and_candidate_files joins '
            f'(cause_hint, test_output) and str.join rejects a non-str item.'
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
        """
        with pytest.raises(TypeError, match='verify_skipped'):
            fake_verify_result(passed=False, verify_skipped=True)

    @pytest.mark.parametrize('passed', [True, False])
    def test_every_field_is_seeded_from_real_verify_result(self, passed: bool) -> None:
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

    def test_override_replaces_the_seeded_value(self) -> None:
        fake = fake_verify_result(passed=False, summary='x', category='lint_failure')

        assert (fake.summary, fake.category) == ('x', 'lint_failure')

    def test_unknown_attribute_read_raises(self) -> None:
        fake = fake_verify_result(passed=True)

        with pytest.raises(AttributeError):
            fake.verify_skipped  # noqa: B018
