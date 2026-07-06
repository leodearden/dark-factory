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


def _validate_exhaustive(enum_cls, policy) -> None:
    """Raise if ``policy`` doesn't have exactly one row per ``enum_cls`` member.

    Reusable, unit-testable form of the F1 exhaustiveness guard: a synthetic
    enum/policy pair can be checked directly instead of only observing the
    failure via a real import crash.
    """
    missing = set(enum_cls) - set(policy)
    extra = set(policy) - set(enum_cls)
    if missing or extra:
        raise AssertionError(
            'CATEGORY_POLICY out of sync with FailureCategory: '
            f'missing rows for {sorted(missing)}, stray rows {sorted(extra)}'
        )


_validate_exhaustive(FailureCategory, CATEGORY_POLICY)


# Ordered from highest to lowest severity; used by ``_worst_category``. Derived
# (not hand-written) so a new category can never drift out of sync with its
# CATEGORY_POLICY row — see bug_history: task 2048 required 4 registry edits
# + 2 inline sets for a single category change.
CATEGORY_PRIORITY: list[FailureCategory] = sorted(
    FailureCategory, key=lambda c: CATEGORY_POLICY[c].severity_rank,
)

# Categories that must NOT be auto-archived. Derived from CategoryPolicy.archive.
ARCHIVE_DENY_LIST: frozenset[FailureCategory] = frozenset(
    c for c, p in CATEGORY_POLICY.items() if not p.archive
)

# Categories skipped by the preexisting-main-break probe. Derived from
# CategoryPolicy.preexisting_probe.
PREEXISTING_BREAK_SKIP_CATEGORIES: frozenset[FailureCategory] = frozenset(
    c for c, p in CATEGORY_POLICY.items() if not p.preexisting_probe
)

# Categories treated as infra-transient by the main-tip sweep (retried rather
# than reported as drift). Derived from CategoryPolicy.is_infra_transient.
INFRA_TRANSIENT_CATEGORIES: frozenset[FailureCategory] = frozenset(
    c for c, p in CATEGORY_POLICY.items() if p.is_infra_transient
)


def should_archive(category: str) -> bool:
    """Return True when *category* warrants durable human-triage archival.

    Pure CATEGORY_POLICY table lookup — no ``endswith('_error')`` heuristic.
    A category outside the known 12 (e.g. a verify_runner UNSCOPED_TYPECHECK_*
    sentinel, or any other unrecognized string) defaults to False.
    """
    try:
        member = FailureCategory(category)
    except ValueError:
        return False
    return CATEGORY_POLICY[member].archive
