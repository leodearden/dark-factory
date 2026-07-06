"""Tests for orchestrator.verify_categories — the declarative FailureCategory
+ CATEGORY_POLICY table (PRD: plans/verify-plan-prd.md task α).

Replaces verify.py's 4 hand-synced classifier-category registries (
``_ARCHIVE_DENY_LIST``, ``_CATEGORY_PRIORITY``, ``PREEXISTING_BREAK_SKIP_CATEGORIES``,
the sweep infra-sentinel set) plus the ``endswith('_error')`` archive heuristic
with ONE table: ``CATEGORY_POLICY: dict[FailureCategory, CategoryPolicy]``.

Test coverage:
  step-1: core surface — FailureCategory member set, StrEnum wire byte-identity,
          RetryKind, CategoryPolicy shape, CATEGORY_POLICY exhaustiveness, spot rows
  step-3: _validate_exhaustive reusable guard (synthetic enum/policy mismatch)
  step-5: derived registries (CATEGORY_PRIORITY / ARCHIVE_DENY_LIST /
          PREEXISTING_BREAK_SKIP_CATEGORIES / INFRA_TRANSIENT_CATEGORIES) match
          today's verify.py literals byte-for-byte
  step-7: should_archive(category) reproduces the legacy per-category archive
          decision without the endswith('_error') heuristic
  step-9: verify.py sources categories from this table (classifier returns
          FailureCategory instances; re-exported registries are the derived objects)
  step-11: cross-module consumers (workflow.py, merge_queue.py) single-source
           the derived registries and the unscoped-gate sentinel namespace is
           preserved separately
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# step-1: core surface
# ---------------------------------------------------------------------------

# The exact 12-member closed output domain of verify._classify_failure today,
# including the '' (NONE) empty-string sentinel (default of _worst_category on
# empty input; a member of both _CATEGORY_PRIORITY and _ARCHIVE_DENY_LIST).
_EXPECTED_CATEGORY_VALUES = {
    'infra_timeout',
    'cargo_cli_error',
    'compile_error',
    'tree_sitter_generate_error',
    'flock_error',
    'npm_error',
    'pytest_internalerror',
    'env_transient',
    'test_failure',
    'unknown_test_failure',
    'passed',
    '',
}


class TestFailureCategoryMemberSet:
    """FailureCategory is a StrEnum with exactly the 12 legacy category strings."""

    def test_member_values_match_legacy_category_set_exactly(self):
        from orchestrator.verify_categories import FailureCategory
        assert {c.value for c in FailureCategory} == _EXPECTED_CATEGORY_VALUES

    def test_member_count_is_twelve(self):
        from orchestrator.verify_categories import FailureCategory
        assert len(list(FailureCategory)) == 12

    def test_is_strenum_subclass(self):
        from enum import StrEnum

        from orchestrator.verify_categories import FailureCategory
        assert issubclass(FailureCategory, StrEnum)

    def test_none_member_is_empty_string(self):
        from orchestrator.verify_categories import FailureCategory
        assert FailureCategory.NONE == ''
        assert FailureCategory.NONE.value == ''


class TestFailureCategoryWireByteIdentity:
    """F2: every member IS its str value — json.dumps output is unchanged.

    StrEnum membership must not perturb any on-the-wire category string
    (verify_runner canonical-JSON Invariant 1).
    """

    def test_str_of_every_member_equals_its_value(self):
        from orchestrator.verify_categories import FailureCategory
        for member in FailureCategory:
            assert str(member) == member.value

    def test_json_dumps_of_every_member_matches_plain_str(self):
        from orchestrator.verify_categories import FailureCategory
        for member in FailureCategory:
            assert json.dumps({'category': member}) == json.dumps({'category': member.value})

    def test_member_equals_and_hashes_as_plain_str(self):
        from orchestrator.verify_categories import FailureCategory
        for member in FailureCategory:
            assert member == member.value
            assert hash(member) == hash(member.value)


class TestRetryKind:
    """RetryKind enumerates the three verify-retry strategies."""

    def test_has_expected_members(self):
        from orchestrator.verify_categories import RetryKind
        assert {m.name for m in RetryKind} == {'NONE', 'TIMEOUT', 'ENV_SERIAL'}


class TestCategoryPolicyShape:
    """CategoryPolicy is a dataclass with the 5 documented fields."""

    def test_has_expected_fields(self):
        import dataclasses

        from orchestrator.verify_categories import CategoryPolicy
        assert dataclasses.is_dataclass(CategoryPolicy)
        field_names = {f.name for f in dataclasses.fields(CategoryPolicy)}
        assert field_names == {
            'severity_rank', 'archive', 'preexisting_probe',
            'is_infra_transient', 'retry_kind',
        }


class TestCategoryPolicyExhaustive:
    """F1: CATEGORY_POLICY has exactly one row per FailureCategory member."""

    def test_keyed_by_failure_category(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
        assert set(CATEGORY_POLICY) == set(FailureCategory)

    def test_row_count_is_twelve(self):
        from orchestrator.verify_categories import CATEGORY_POLICY
        assert len(CATEGORY_POLICY) == 12


class TestCategoryPolicyGoldenRows:
    """Spot-check exact rows against the golden table (full table covered by
    the derived-registry byte-identity tests in step-5)."""

    def test_infra_timeout_row(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory, RetryKind
        row = CATEGORY_POLICY[FailureCategory.INFRA_TIMEOUT]
        assert row.severity_rank == 0
        assert row.archive is False
        assert row.preexisting_probe is False
        assert row.is_infra_transient is False
        assert row.retry_kind == RetryKind.TIMEOUT

    def test_env_transient_row(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory, RetryKind
        row = CATEGORY_POLICY[FailureCategory.ENV_TRANSIENT]
        assert row.retry_kind == RetryKind.ENV_SERIAL
        assert row.is_infra_transient is True
        assert row.archive is False
        assert row.preexisting_probe is False

    def test_none_row(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
        row = CATEGORY_POLICY[FailureCategory.NONE]
        assert row.severity_rank == 11


# ---------------------------------------------------------------------------
# step-3: _validate_exhaustive reusable guard
# ---------------------------------------------------------------------------


class TestValidateExhaustive:
    """F1 exhaustiveness as a reusable, unit-testable function.

    RED today: ``_validate_exhaustive`` is not a named/exported function yet
    (the real module only has an inline ``assert`` at import time).
    """

    def _make_synth_policy(self):
        from enum import StrEnum

        from orchestrator.verify_categories import (
            CATEGORY_POLICY,
            CategoryPolicy,
            FailureCategory,
        )

        class _Synth(StrEnum):
            A = 'a'
            B = 'b'

        any_row = next(iter(CATEGORY_POLICY.values()))
        return _Synth, any_row

    def test_missing_member_raises_and_names_it(self):
        from orchestrator.verify_categories import _validate_exhaustive

        _Synth, any_row = self._make_synth_policy()
        with pytest.raises(AssertionError, match='B'):
            _validate_exhaustive(_Synth, {_Synth.A: any_row})

    def test_stray_policy_key_raises(self):
        from orchestrator.verify_categories import _validate_exhaustive

        _Synth, any_row = self._make_synth_policy()
        policy = {_Synth.A: any_row, _Synth.B: any_row, 'stray': any_row}
        with pytest.raises(AssertionError, match='stray'):
            _validate_exhaustive(_Synth, policy)

    def test_matching_sets_do_not_raise(self):
        from orchestrator.verify_categories import _validate_exhaustive

        _Synth, any_row = self._make_synth_policy()
        _validate_exhaustive(_Synth, {_Synth.A: any_row, _Synth.B: any_row})

    def test_real_module_import_satisfies_its_own_guard(self):
        # Importing the real module must succeed — its shipped table already
        # satisfies _validate_exhaustive (this is F1 firing at import time).
        import orchestrator.verify_categories as vc

        vc._validate_exhaustive(vc.FailureCategory, vc.CATEGORY_POLICY)


# ---------------------------------------------------------------------------
# step-5: derived registries match today's verify.py literals byte-for-byte
# ---------------------------------------------------------------------------


class TestDerivedRegistriesByteIdentity:
    """The four registries verify.py hand-syncs today must be DERIVED from
    CATEGORY_POLICY and reproduce the current literals exactly (plain-str
    goldens — StrEnum == str makes equality hold across the enum/str boundary).
    """

    def test_category_priority_matches_legacy_order(self):
        from orchestrator.verify_categories import CATEGORY_PRIORITY
        assert CATEGORY_PRIORITY == [
            'infra_timeout',
            'cargo_cli_error',
            'compile_error',
            'tree_sitter_generate_error',
            'flock_error',
            'npm_error',
            'pytest_internalerror',
            'env_transient',
            'test_failure',
            'unknown_test_failure',
            'passed',
            '',
        ]

    def test_archive_deny_list_matches_legacy_set(self):
        from orchestrator.verify_categories import ARCHIVE_DENY_LIST
        assert ARCHIVE_DENY_LIST == frozenset({
            'compile_error', 'test_failure', 'infra_timeout', 'passed', '',
            'pytest_internalerror', 'env_transient',
        })

    def test_preexisting_break_skip_categories_matches_legacy_set(self):
        from orchestrator.verify_categories import PREEXISTING_BREAK_SKIP_CATEGORIES
        assert PREEXISTING_BREAK_SKIP_CATEGORIES == frozenset({
            'infra_timeout', 'flock_error', 'pytest_internalerror', 'env_transient',
        })

    def test_infra_transient_categories_matches_legacy_sweep_set(self):
        from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
        assert INFRA_TRANSIENT_CATEGORIES == frozenset({
            'pytest_internalerror', 'env_transient',
        })


# ---------------------------------------------------------------------------
# step-7: should_archive(category) reproduces the legacy per-category decision
# ---------------------------------------------------------------------------


class TestShouldArchive:
    """should_archive(category) is a pure CATEGORY_POLICY table lookup that
    reproduces the legacy _should_archive_category decision for every
    category — proving the endswith('_error') heuristic can be deleted
    without changing behavior for any of the 12 known categories.
    """

    @pytest.mark.parametrize(
        ('category', 'expected'),
        [
            ('infra_timeout', False),
            ('cargo_cli_error', True),
            ('compile_error', False),
            ('tree_sitter_generate_error', True),
            ('flock_error', True),
            ('npm_error', True),
            ('pytest_internalerror', False),
            ('env_transient', False),
            ('test_failure', False),
            ('unknown_test_failure', True),
            ('passed', False),
            ('', False),
        ],
    )
    def test_matches_legacy_decision_for_every_category(self, category, expected):
        from orchestrator.verify_categories import should_archive
        assert should_archive(category) is expected

    def test_unscoped_sentinel_categories_default_false(self):
        # Out-of-band verify_runner sentinels are not CATEGORY_POLICY members —
        # should_archive must not raise and must default to False for them.
        from orchestrator.verify_categories import should_archive
        assert should_archive('unscoped_typecheck_failed') is False
        assert should_archive('unscoped_typecheck_timeout') is False

    def test_unknown_category_defaults_false_endswith_heuristic_gone(self):
        # Before this refactor, a bare '..._error' suffix implied archival.
        # That heuristic is deleted: an unrecognized category must NOT
        # auto-archive just because it ends with '_error'.
        from orchestrator.verify_categories import should_archive
        assert should_archive('made_up_error') is False
