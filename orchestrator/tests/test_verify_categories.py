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

# The exact 15-member closed output domain of verify._classify_failure today
# (the 12 legacy category strings, task 2549's DISK_FULL/SEMAPHORE_TIMEOUT
# host-infrastructure categories, and task 3173's INFRA_KILL external-signal
# category), including the '' (NONE) empty-string sentinel (default of
# _worst_category on empty input; a member of both _CATEGORY_PRIORITY and
# _ARCHIVE_DENY_LIST).
_EXPECTED_CATEGORY_VALUES = {
    'infra_timeout',
    'infra_kill',
    'disk_full',
    'semaphore_timeout',
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
    """FailureCategory is a StrEnum with exactly the 15 category strings."""

    def test_member_values_match_legacy_category_set_exactly(self):
        from orchestrator.verify_categories import FailureCategory
        assert {c.value for c in FailureCategory} == _EXPECTED_CATEGORY_VALUES

    def test_member_count_is_fifteen(self):
        from orchestrator.verify_categories import FailureCategory
        assert len(list(FailureCategory)) == 15

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
    """CategoryPolicy is a dataclass with the 6 documented fields."""

    def test_has_expected_fields(self):
        import dataclasses

        from orchestrator.verify_categories import CategoryPolicy
        assert dataclasses.is_dataclass(CategoryPolicy)
        field_names = {f.name for f in dataclasses.fields(CategoryPolicy)}
        assert field_names == {
            'severity_rank', 'archive', 'preexisting_probe',
            'is_infra_transient', 'verdict_indeterminate', 'retry_kind',
        }

    def test_verdict_indeterminate_has_no_default(self):
        """A new category must ADJUDICATE whether its non-completion is
        branch-causable, not inherit an answer: getting it wrong in the True
        direction is a false-GREEN (task 3173)."""
        import dataclasses

        from orchestrator.verify_categories import CategoryPolicy
        field = next(
            f for f in dataclasses.fields(CategoryPolicy)
            if f.name == 'verdict_indeterminate'
        )
        assert field.default is dataclasses.MISSING
        assert field.default_factory is dataclasses.MISSING


class TestCategoryPolicyExhaustive:
    """F1: CATEGORY_POLICY has exactly one row per FailureCategory member."""

    def test_keyed_by_failure_category(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
        assert set(CATEGORY_POLICY) == set(FailureCategory)

    def test_row_count_is_fifteen(self):
        from orchestrator.verify_categories import CATEGORY_POLICY
        assert len(CATEGORY_POLICY) == 15


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
        assert row.verdict_indeterminate is False
        assert row.retry_kind == RetryKind.TIMEOUT

    def test_env_transient_row(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory, RetryKind
        row = CATEGORY_POLICY[FailureCategory.ENV_TRANSIENT]
        assert row.retry_kind == RetryKind.ENV_SERIAL
        assert row.is_infra_transient is True
        assert row.verdict_indeterminate is True
        assert row.archive is False
        assert row.preexisting_probe is False

    def test_none_row(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
        row = CATEGORY_POLICY[FailureCategory.NONE]
        assert row.severity_rank == 14


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
            'infra_kill',
            'disk_full',
            'semaphore_timeout',
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
        """The legacy set MINUS semaphore_timeout, removed deliberately by
        task 3679 (that category terminates in a blocking human escalation,
        so its log is the only triage artifact — see
        TestSemaphoreTimeoutArchivesForHumanTriage). Every other member is
        still the legacy value; this stays a byte-for-byte pin so an
        UNINTENDED archive-policy change still reds here."""
        from orchestrator.verify_categories import ARCHIVE_DENY_LIST
        assert frozenset({
            'compile_error', 'test_failure', 'infra_timeout', 'passed', '',
            'pytest_internalerror', 'env_transient', 'disk_full',
        }) == ARCHIVE_DENY_LIST

    def test_preexisting_break_skip_categories_matches_legacy_set(self):
        from orchestrator.verify_categories import PREEXISTING_BREAK_SKIP_CATEGORIES
        assert frozenset({
            'infra_timeout', 'flock_error', 'pytest_internalerror', 'env_transient',
            'disk_full', 'semaphore_timeout', 'infra_kill',
        }) == PREEXISTING_BREAK_SKIP_CATEGORIES

    def test_infra_transient_categories_matches_legacy_sweep_set(self):
        from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
        assert frozenset({
            'pytest_internalerror', 'env_transient', 'disk_full', 'semaphore_timeout',
            'infra_kill',
        }) == INFRA_TRANSIENT_CATEGORIES


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
            ('infra_kill', True),
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


# ---------------------------------------------------------------------------
# step-9: verify.py single-sources categories from the table
# ---------------------------------------------------------------------------


class TestVerifyRegistriesSingleSourced:
    """verify.py's four category registries must BE (identity, not just
    value-equality) the verify_categories objects — the mechanism that
    eliminates the hand-sync bug_history (task 2048: a single category
    change required 4 registry edits + 2 inline sets).

    RED today: verify.py still holds hand-written literals for
    _CATEGORY_PRIORITY / _ARCHIVE_DENY_LIST / PREEXISTING_BREAK_SKIP_CATEGORIES,
    and INFRA_TRANSIENT_CATEGORIES does not exist on verify.py yet.
    """

    def test_category_priority_is_the_derived_object(self):
        from orchestrator import verify, verify_categories
        assert verify._CATEGORY_PRIORITY is verify_categories.CATEGORY_PRIORITY

    def test_archive_deny_list_is_the_derived_object(self):
        from orchestrator import verify, verify_categories
        assert verify._ARCHIVE_DENY_LIST is verify_categories.ARCHIVE_DENY_LIST

    def test_preexisting_break_skip_categories_is_the_derived_object(self):
        from orchestrator import verify, verify_categories
        assert (
            verify.PREEXISTING_BREAK_SKIP_CATEGORIES
            is verify_categories.PREEXISTING_BREAK_SKIP_CATEGORIES
        )

    def test_infra_transient_categories_is_the_derived_object(self):
        from orchestrator import verify, verify_categories
        assert verify.INFRA_TRANSIENT_CATEGORIES is verify_categories.INFRA_TRANSIENT_CATEGORIES


class TestShouldArchiveCategoryDelegatesToTable:
    """verify._should_archive_category must delegate to
    verify_categories.should_archive — no endswith('_error') heuristic.

    RED today: the endswith heuristic makes an unrecognized '..._error'
    string archive (True) instead of defaulting to False.
    """

    @pytest.mark.parametrize('category', sorted(_EXPECTED_CATEGORY_VALUES))
    def test_matches_table_lookup_for_every_known_category(self, category):
        from orchestrator import verify, verify_categories
        assert verify._should_archive_category(category) == verify_categories.should_archive(category)

    def test_unknown_error_suffixed_category_no_longer_auto_archives(self):
        from orchestrator import verify
        assert verify._should_archive_category('made_up_error') is False


class TestClassifierByteIdentityGolden:
    """Defensive golden: byte-identical ``json.dumps`` output survives the
    α→δ classifier evolution.

    ``_worst_category`` is untouched by task δ and still returns a plain
    ``str``. ``classify_failure`` — δ's tool-dispatched replacement for the
    pre-δ ``_classify_failure`` this test originally exercised (task 2131
    step-11) — returns a REAL ``FailureCategory`` instance rather than a
    plain ``str``: the α-era ``type(category) is str`` characterization
    migrates to ``isinstance(category, str)`` (``FailureCategory`` IS its
    string value, a ``StrEnum``), but ``json.dumps`` output stays
    byte-identical either way (F2 / verify_runner canonical-JSON Invariant 1).
    """

    @pytest.mark.parametrize(
        ('rc', 'output', 'timed_out', 'expected'),
        [
            (0, '', False, 'passed'),
            (1, '', True, 'infra_timeout'),
            (1, 'error[E0432]: unresolved import', False, 'compile_error'),
            (1, 'error: --exclude can only be used together with --workspace', False, 'cargo_cli_error'),
            (1, 'INTERNALERROR> Traceback', False, 'pytest_internalerror'),
            (1, 'test_foo FAILED', False, 'test_failure'),
            (1, 'npm ERR! code E404', False, 'npm_error'),
            (1, 'flock: failed to get lock', False, 'flock_error'),
            (1, 'tree-sitter generate failed', False, 'tree_sitter_generate_error'),
            (1, 'nothing matched', False, 'unknown_test_failure'),
        ],
    )
    def test_classify_failure_returns_legacy_str_with_byte_identical_json(
        self, rc, output, timed_out, expected,
    ):
        from orchestrator.verify_classify import classify_failure  # noqa: PLC0415
        from orchestrator.verify_cmd import ToolKind  # noqa: PLC0415

        category = classify_failure(ToolKind.OPAQUE, rc, output, timed_out)
        assert category == expected
        assert isinstance(category, str)
        assert json.dumps({'category': category}) == json.dumps({'category': expected})

    def test_worst_category_returns_legacy_str_with_byte_identical_json(self):
        from orchestrator.verify import _worst_category
        worst = _worst_category(['test_failure', 'infra_timeout', 'npm_error'])
        assert worst == 'infra_timeout'
        assert type(worst) is str
        assert json.dumps({'category': worst}) == json.dumps({'category': 'infra_timeout'})


# ---------------------------------------------------------------------------
# step-11: cross-module single-source + sentinel-namespace separation
#
# NOTE on the F2 wire golden (json.dumps({'category': member}) ==
# json.dumps({'category': member.value}) for every FailureCategory member):
# already covered by TestFailureCategoryWireByteIdentity.
# test_json_dumps_of_every_member_matches_plain_str (step-1) — not duplicated
# here.
# ---------------------------------------------------------------------------


class TestAssertSentinelsDisjoint:
    """_assert_sentinels_disjoint(sentinels, enum_cls) is the reusable,
    unit-testable guard step-12 wires into verify_runner.py's import to keep
    the UNSCOPED_TYPECHECK_* sentinel namespace provably separate from
    FailureCategory (mirrors _validate_exhaustive's synthetic-input design).

    RED today: the helper does not exist yet.
    """

    def test_colliding_sentinel_raises_and_names_it(self):
        from enum import StrEnum

        from orchestrator.verify_categories import _assert_sentinels_disjoint

        class _Synth(StrEnum):
            KNOWN = 'synthetic_known_value'

        with pytest.raises(AssertionError, match='synthetic_known_value'):
            _assert_sentinels_disjoint(
                frozenset({'synthetic_known_value', 'totally_unrelated_sentinel'}), _Synth,
            )

    def test_disjoint_sentinels_do_not_raise(self):
        from enum import StrEnum

        from orchestrator.verify_categories import _assert_sentinels_disjoint

        class _Synth(StrEnum):
            KNOWN = 'synthetic_known_value'

        _assert_sentinels_disjoint(frozenset({'totally_unrelated_sentinel'}), _Synth)

    def test_real_unscoped_sentinels_are_disjoint_from_failure_category(self):
        # Exercises the helper against verify_runner's real production
        # sentinels — the exact call step-12 wires in at verify_runner's
        # import time.
        from orchestrator.verify_categories import FailureCategory, _assert_sentinels_disjoint
        from orchestrator.verify_runner import _UNSCOPED_SENTINEL_CATEGORIES

        _assert_sentinels_disjoint(_UNSCOPED_SENTINEL_CATEGORIES, FailureCategory)

    def test_importing_verify_runner_does_not_raise(self):
        # Defensive golden for step-12: once the guard is wired into
        # verify_runner.py's module body, a real collision would raise at
        # import time. Import succeeding here proves the production
        # sentinel/category data stays disjoint.
        import orchestrator.verify_runner  # noqa: F401


class TestCrossModulePreexistingSingleSourced:
    """merge_queue.py and workflow.py must single-source
    PREEXISTING_BREAK_SKIP_CATEGORIES from verify_categories (identity, not
    just equality) — step-12 switches both imports directly to
    orchestrator.verify_categories instead of via orchestrator.verify.

    Already GREEN as a side effect of step-10 (verify.py re-exports the
    verify_categories object unchanged, and `from X import Y` binds by
    reference — no copy happens across either import hop). Kept as a
    regression guard that stays true once step-12 switches the import
    source directly.
    """

    def test_merge_queue_preexisting_is_the_derived_object(self):
        from orchestrator import merge_queue, verify_categories
        assert (
            merge_queue.PREEXISTING_BREAK_SKIP_CATEGORIES
            is verify_categories.PREEXISTING_BREAK_SKIP_CATEGORIES
        )

    def test_workflow_preexisting_is_the_derived_object(self):
        from orchestrator import verify_categories, workflow
        assert (
            workflow.PREEXISTING_BREAK_SKIP_CATEGORIES
            is verify_categories.PREEXISTING_BREAK_SKIP_CATEGORIES
        )


# ---------------------------------------------------------------------------
# PART 2 (task 2549): two new host-infrastructure FailureCategory members —
# DISK_FULL (ENOSPC / linker SIGBUS-on-full-disk) and SEMAPHORE_TIMEOUT
# (flock/semaphore slot-acquisition timeout) — that previously had no
# category at all and fell through to whichever code-fault category the
# classifier matched first. Both are env_transient-family policy rows
# (is_infra_transient=True, preexisting_probe=False, retry_kind=NONE — a
# serial re-run cannot fix a host condition) ranked just below INFRA_TIMEOUT
# and above every code-fault category.
#
# The pair no longer shares an ARCHIVE policy (task 3679): DISK_FULL keeps
# archive=False, SEMAPHORE_TIMEOUT is now archive=True. See
# TestSemaphoreTimeoutArchivesForHumanTriage for the grounding — with
# retry_kind=NONE, SEMAPHORE_TIMEOUT surfaces to a human rather than
# self-clearing, and the archived log is that human's only triage artifact.
#
# RED today: neither member/row exists yet, so every test below fails on
# the local ``from orchestrator.verify_categories import FailureCategory``
# attribute access (``FailureCategory.DISK_FULL`` / ``.SEMAPHORE_TIMEOUT``
# don't exist) or on the corresponding _worst_category priority check.
# ---------------------------------------------------------------------------


class TestEnvironmentalCategoriesExistWithInfraTransientPolicy:
    """DISK_FULL / SEMAPHORE_TIMEOUT are new FailureCategory members with an
    env_transient-family CATEGORY_POLICY row.

    The two rows are identical EXCEPT for ``archive`` (task 3679):
    SEMAPHORE_TIMEOUT is archive=True, DISK_FULL remains archive=False.
    """

    def test_disk_full_value_and_policy(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory, RetryKind
        assert FailureCategory.DISK_FULL.value == 'disk_full'
        row = CATEGORY_POLICY[FailureCategory.DISK_FULL]
        assert row.is_infra_transient is True
        assert row.archive is False
        assert row.preexisting_probe is False
        assert row.retry_kind == RetryKind.NONE

    def test_semaphore_timeout_value_and_policy(self):
        """archive=True since task 3679 — the one field where this row no
        longer matches its DISK_FULL sibling (see the class docstring)."""
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory, RetryKind
        assert FailureCategory.SEMAPHORE_TIMEOUT.value == 'semaphore_timeout'
        row = CATEGORY_POLICY[FailureCategory.SEMAPHORE_TIMEOUT]
        assert row.is_infra_transient is True
        assert row.archive is True
        assert row.preexisting_probe is False
        assert row.retry_kind == RetryKind.NONE


class TestSemaphoreTimeoutArchivesForHumanTriage:
    """task 3679 / esc-5848-2 / esc-5893-3: SEMAPHORE_TIMEOUT must archive
    its verify log.

    A SEMAPHORE_TIMEOUT is infra-transient but NOT self-clearing:
    ``retry_kind=NONE`` means no serial re-run is attempted, so the category
    can and does terminate in a BLOCKING human escalation — both live
    incidents did, each holding its lane ~5h. ``archive=False`` meant the one
    artifact a human needs to triage that block was discarded by the same
    wrong label that caused the block: reify ``data/verify-logs/5848`` and
    ``/5893`` do not exist, while 553 sibling task dirs do.

    Also pins the invariant that should survive a future policy edit: a
    category that is infra-transient but can still surface to a human must
    not be in ``ARCHIVE_DENY_LIST``. Asserted for SEMAPHORE_TIMEOUT
    specifically and deliberately NOT broadened to every infra-transient
    category — DISK_FULL, PYTEST_INTERNALERROR and ENV_TRANSIENT keep their
    current rows, which this task does not adjudicate.

    RED today: the row is ``archive=False``, so all four assertions fail.
    """

    def test_policy_row_archives(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
        assert CATEGORY_POLICY[FailureCategory.SEMAPHORE_TIMEOUT].archive is True

    def test_not_in_archive_deny_list(self):
        """ARCHIVE_DENY_LIST is DERIVED from the row, so this pins that the
        derivation actually followed the row rather than a stale hand-synced
        copy — the drift hazard verify_categories was built to close."""
        from orchestrator.verify_categories import ARCHIVE_DENY_LIST, FailureCategory
        assert FailureCategory.SEMAPHORE_TIMEOUT not in ARCHIVE_DENY_LIST

    def test_should_archive_returns_true(self):
        from orchestrator.verify_categories import should_archive
        assert should_archive('semaphore_timeout') is True

    def test_verify_delegation_path_archives(self):
        """The call site that actually decides whether the log survives —
        mirrors TestShouldArchiveCategoryDelegatesToTable's style."""
        from orchestrator import verify
        assert verify._should_archive_category('semaphore_timeout') is True


class TestEnvironmentalCategoriesOutrankCodeFaults:
    """_worst_category must pick the infra root cause over a co-occurring
    downstream code-fault category — DISK_FULL/SEMAPHORE_TIMEOUT are ranked
    just below INFRA_TIMEOUT and above every code-fault category."""

    def test_disk_full_outranks_compile_error(self):
        from orchestrator.verify import _worst_category
        assert _worst_category(['disk_full', 'compile_error']) == 'disk_full'

    def test_semaphore_timeout_outranks_test_failure(self):
        from orchestrator.verify import _worst_category
        assert _worst_category(['semaphore_timeout', 'test_failure']) == 'semaphore_timeout'


# ---------------------------------------------------------------------------
# PART 3 (task 3173): INFRA_KILL — a leg terminated by an EXTERNAL signal
# (SIGKILL/SIGTERM/SIGINT/SIGHUP) never produced an exit verdict at all, so it
# must not be reported as a branch fault.
#
# The measured incident: a merge-verify lint leg was SIGKILLed at 0.31s under
# host load; ``classify_failure`` fell through to UNKNOWN_TEST_FAILURE (a
# blocking, branch-blaming verdict) and the leg went on to discard the OTHER
# host's completed 1097s PASS behind merge commit b1ac2c7f.
#
# RED today: FailureCategory.INFRA_KILL and INDETERMINATE_VERDICT_CATEGORIES
# do not exist, so every test below fails on attribute access / ImportError.
# ---------------------------------------------------------------------------


class TestInfraKillCategory:
    """INFRA_KILL is a new FailureCategory member with an infra-transient
    policy row that — uniquely among the infra categories — ARCHIVES.
    """

    def test_member_exists_with_expected_value(self):
        from orchestrator.verify_categories import FailureCategory
        assert FailureCategory.INFRA_KILL == 'infra_kill'
        assert FailureCategory.INFRA_KILL.value == 'infra_kill'

    def test_policy_row(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory, RetryKind
        row = CATEGORY_POLICY[FailureCategory.INFRA_KILL]
        assert row.severity_rank == 1
        assert row.archive is True
        assert row.preexisting_probe is False
        assert row.is_infra_transient is True
        assert row.verdict_indeterminate is True
        assert row.retry_kind == RetryKind.NONE

    def test_is_infra_transient_registry_member(self):
        from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES
        assert 'infra_kill' in INFRA_TRANSIENT_CATEGORIES

    def test_is_preexisting_break_skip_member(self):
        from orchestrator.verify_categories import PREEXISTING_BREAK_SKIP_CATEGORIES
        assert 'infra_kill' in PREEXISTING_BREAK_SKIP_CATEGORIES

    def test_archives_unlike_every_other_infra_category(self):
        # The ONLY forensic handle for an off-box, un-reproducible kill: the
        # incident was diagnosable solely because the archive existed.
        from orchestrator.verify_categories import ARCHIVE_DENY_LIST, should_archive
        assert 'infra_kill' not in ARCHIVE_DENY_LIST
        assert should_archive('infra_kill') is True

    def test_outranks_every_output_derived_category_but_not_a_timeout(self):
        # A kill means "no verdict was produced", so it must dominate any
        # co-occurring output-derived category — but INFRA_TIMEOUT keeps rank 0.
        from orchestrator.verify_categories import CATEGORY_PRIORITY
        assert CATEGORY_PRIORITY[0:3] == ['infra_timeout', 'infra_kill', 'disk_full']


class TestIndeterminateVerdictCategories:
    """The derived registry naming the categories whose leg produced NO
    completed verdict AND whose non-completion the branch could not have
    caused — which therefore may not veto another host's completed PASS
    (merge_queue's per-land cross-check).
    """

    def test_registry_exists_and_contains_infra_kill(self):
        from orchestrator.verify_categories import INDETERMINATE_VERDICT_CATEGORIES
        assert 'infra_kill' in INDETERMINATE_VERDICT_CATEGORIES

    def test_membership_is_exactly_the_adjudicated_set(self):
        # The whole golden set, spelled out: every member is a HOST condition
        # the diff cannot reach (an external kill, a full disk, a lost
        # semaphore, a mutated shared venv). Adding a row here widens the
        # set of local failures that may be overruled by a remote PASS, so
        # it is pinned exactly rather than by spot-checks.
        from orchestrator.verify_categories import INDETERMINATE_VERDICT_CATEGORIES
        assert set(INDETERMINATE_VERDICT_CATEGORIES) == {
            'infra_kill', 'disk_full', 'semaphore_timeout', 'env_transient',
        }

    def test_is_derived_from_the_policy_rows_not_the_infra_transient_set(self):
        # REGRESSION GUARD (amendment, task 3173): the registry was first
        # spelled `INFRA_TRANSIENT_CATEGORIES - {INFRA_TIMEOUT}` — a no-op
        # subtraction, since INFRA_TIMEOUT is is_infra_transient=False. It
        # was therefore a second NAME for the infra-transient set, and the
        # exclusion tests below passed vacuously. The two sets answer
        # different questions and must not collapse again.
        from orchestrator.verify_categories import (
            CATEGORY_POLICY,
            INDETERMINATE_VERDICT_CATEGORIES,
            INFRA_TRANSIENT_CATEGORIES,
        )
        assert INDETERMINATE_VERDICT_CATEGORIES != INFRA_TRANSIENT_CATEGORIES
        assert INDETERMINATE_VERDICT_CATEGORIES < INFRA_TRANSIENT_CATEGORIES
        assert frozenset(
            c for c, p in CATEGORY_POLICY.items() if p.verdict_indeterminate
        ) == INDETERMINATE_VERDICT_CATEGORIES

    def test_excludes_infra_timeout(self):
        # DESIGN DECISION (plan.json, task 3173): a hang is one of the few
        # non-completions a branch can genuinely CAUSE (an infinite loop or a
        # deadlock introduced by the diff), so a local timeout must keep
        # vetoing a remote PASS — fail CLOSED there. Landing on a local
        # timeout would be the false-GREEN class tasks 2822/1700 hardened.
        from orchestrator.verify_categories import (
            CATEGORY_POLICY,
            INDETERMINATE_VERDICT_CATEGORIES,
            FailureCategory,
        )
        assert 'infra_timeout' not in INDETERMINATE_VERDICT_CATEGORIES
        # Non-vacuous: the exclusion is an adjudicated row, not a side effect
        # of infra_timeout being absent from some other set.
        assert CATEGORY_POLICY[FailureCategory.INFRA_TIMEOUT].verdict_indeterminate is False

    def test_excludes_pytest_internalerror(self):
        # DESIGN DECISION (amendment, task 3173): pytest_internalerror is
        # is_infra_transient=True — retrying it is usually worthwhile — but
        # `^INTERNALERROR>` is raised at COLLECTION time, which a conftest.py
        # or plugin added by the diff can trigger, and may trigger only on
        # this host's interpreter/plugin set. So it produced no verdict, but
        # the branch may well have caused that: fail CLOSED, keep the veto.
        from orchestrator.verify_categories import (
            CATEGORY_POLICY,
            INDETERMINATE_VERDICT_CATEGORIES,
            INFRA_TRANSIENT_CATEGORIES,
            FailureCategory,
        )
        assert 'pytest_internalerror' in INFRA_TRANSIENT_CATEGORIES
        assert 'pytest_internalerror' not in INDETERMINATE_VERDICT_CATEGORIES
        assert CATEGORY_POLICY[FailureCategory.PYTEST_INTERNALERROR].verdict_indeterminate is False

    def test_no_completed_verdict_category_is_ever_indeterminate(self):
        """A category that DID reach an exit decision about the branch can
        never be overruled — the registry only ever names non-completions."""
        from orchestrator.verify_categories import INDETERMINATE_VERDICT_CATEGORIES
        for category in (
            'test_failure', 'unknown_test_failure', 'compile_error',
            'cargo_cli_error', 'npm_error', 'tree_sitter_generate_error',
            'flock_error', 'passed', '',
        ):
            assert category not in INDETERMINATE_VERDICT_CATEGORIES, category

    def test_is_a_frozenset_of_categories(self):
        from orchestrator.verify_categories import (
            INDETERMINATE_VERDICT_CATEGORIES,
            FailureCategory,
        )
        assert isinstance(INDETERMINATE_VERDICT_CATEGORIES, frozenset)
        assert INDETERMINATE_VERDICT_CATEGORIES.issubset(set(FailureCategory))
