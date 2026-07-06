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
