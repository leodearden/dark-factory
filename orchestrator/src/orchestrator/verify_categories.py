"""Declarative failure-category table (PRD: plans/verify-plan-prd.md task α).

Single source of truth for every category ``verify._classify_failure`` can
return. Replaces 4 hand-synced registries in verify.py (``_ARCHIVE_DENY_LIST``,
``_CATEGORY_PRIORITY``, ``PREEXISTING_BREAK_SKIP_CATEGORIES``, the sweep
infra-sentinel set) plus the ``endswith('_error')`` archive heuristic with one
table: ``CATEGORY_POLICY: dict[FailureCategory, CategoryPolicy]``.

``FailureCategory`` is a ``StrEnum`` so every member IS its string value —
``json.dumps`` output and every existing str-keyed comparison/membership check
stay byte-identical to today (verify_runner canonical-JSON Invariant 1).

The two ``verify_runner`` ``UNSCOPED_TYPECHECK_*`` sentinels are a separate,
out-of-band gate namespace injected into ``VerifyResult.category`` — never
produced by the classifier — and are deliberately NOT members here (see
plans/verify-plan-prd.md task α design decisions).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum


class FailureCategory(StrEnum):
    """The closed 12-value output domain of ``verify._classify_failure``."""

    INFRA_TIMEOUT = 'infra_timeout'
    CARGO_CLI_ERROR = 'cargo_cli_error'
    COMPILE_ERROR = 'compile_error'
    TREE_SITTER_GENERATE_ERROR = 'tree_sitter_generate_error'
    FLOCK_ERROR = 'flock_error'
    NPM_ERROR = 'npm_error'
    PYTEST_INTERNALERROR = 'pytest_internalerror'
    ENV_TRANSIENT = 'env_transient'
    TEST_FAILURE = 'test_failure'
    UNKNOWN_TEST_FAILURE = 'unknown_test_failure'
    PASSED = 'passed'
    NONE = ''


class RetryKind(Enum):
    """How ``run_verification`` recovers from a given category, if at all."""

    NONE = 'none'
    TIMEOUT = 'timeout'
    ENV_SERIAL = 'env_serial'


@dataclass(frozen=True)
class CategoryPolicy:
    """Everything the rest of verify.py needs to know about one category."""

    severity_rank: int
    archive: bool
    preexisting_probe: bool
    is_infra_transient: bool
    retry_kind: RetryKind


CATEGORY_POLICY: dict[FailureCategory, CategoryPolicy] = {
    FailureCategory.INFRA_TIMEOUT: CategoryPolicy(
        severity_rank=0, archive=False, preexisting_probe=False,
        is_infra_transient=False, retry_kind=RetryKind.TIMEOUT,
    ),
    FailureCategory.CARGO_CLI_ERROR: CategoryPolicy(
        severity_rank=1, archive=True, preexisting_probe=True,
        is_infra_transient=False, retry_kind=RetryKind.NONE,
    ),
    FailureCategory.COMPILE_ERROR: CategoryPolicy(
        severity_rank=2, archive=False, preexisting_probe=True,
        is_infra_transient=False, retry_kind=RetryKind.NONE,
    ),
    FailureCategory.TREE_SITTER_GENERATE_ERROR: CategoryPolicy(
        severity_rank=3, archive=True, preexisting_probe=True,
        is_infra_transient=False, retry_kind=RetryKind.NONE,
    ),
    FailureCategory.FLOCK_ERROR: CategoryPolicy(
        severity_rank=4, archive=True, preexisting_probe=False,
        is_infra_transient=False, retry_kind=RetryKind.NONE,
    ),
    FailureCategory.NPM_ERROR: CategoryPolicy(
        severity_rank=5, archive=True, preexisting_probe=True,
        is_infra_transient=False, retry_kind=RetryKind.NONE,
    ),
    FailureCategory.PYTEST_INTERNALERROR: CategoryPolicy(
        severity_rank=6, archive=False, preexisting_probe=False,
        is_infra_transient=True, retry_kind=RetryKind.NONE,
    ),
    FailureCategory.ENV_TRANSIENT: CategoryPolicy(
        severity_rank=7, archive=False, preexisting_probe=False,
        is_infra_transient=True, retry_kind=RetryKind.ENV_SERIAL,
    ),
    FailureCategory.TEST_FAILURE: CategoryPolicy(
        severity_rank=8, archive=False, preexisting_probe=True,
        is_infra_transient=False, retry_kind=RetryKind.NONE,
    ),
    FailureCategory.UNKNOWN_TEST_FAILURE: CategoryPolicy(
        severity_rank=9, archive=True, preexisting_probe=True,
        is_infra_transient=False, retry_kind=RetryKind.NONE,
    ),
    FailureCategory.PASSED: CategoryPolicy(
        severity_rank=10, archive=False, preexisting_probe=True,
        is_infra_transient=False, retry_kind=RetryKind.NONE,
    ),
    FailureCategory.NONE: CategoryPolicy(
        severity_rank=11, archive=False, preexisting_probe=True,
        is_infra_transient=False, retry_kind=RetryKind.NONE,
    ),
}

assert set(CATEGORY_POLICY) == set(FailureCategory), (
    'CATEGORY_POLICY out of sync with FailureCategory: '
    f'missing rows for {sorted(set(FailureCategory) - set(CATEGORY_POLICY))}, '
    f'stray rows {sorted(set(CATEGORY_POLICY) - set(FailureCategory))}'
)
