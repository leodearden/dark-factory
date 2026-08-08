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
        # REVIEW AMENDMENT (task 3173): flipped True -> False. Retrying an
        # env_transient is still worthwhile (is_infra_transient stays True and
        # RetryKind.ENV_SERIAL is untouched), but it may no longer overrule
        # another host's PASS — see TestIndeterminateExclusionsAreAdjudicated
        # .test_excludes_env_transient for the residual that forced this.
        assert row.verdict_indeterminate is False
        # task 3683: flipped False -> True. A separate field from the
        # verdict_indeterminate amendment above — see
        # TestEnvTransientArchivesForHumanTriage.
        assert row.archive is True
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
        """The legacy set MINUS the members removed deliberately, in order:

          * semaphore_timeout (task 3679) — that category terminates in a
            blocking human escalation, so its log is the only triage artifact
            (see TestSemaphoreTimeoutArchivesForHumanTriage).
          * disk_full (task 3683) — same terminus, reached through the
            identical category-agnostic retry windows (see
            TestDiskFullArchivesForHumanTriage).
          * pytest_internalerror (task 3683) — likewise; the sweep retry that
            justified its exclusion is the one fail-OPEN arm and does not
            cover the three that escalate (see
            TestPytestInternalerrorArchivesForHumanTriage).
          * env_transient (task 3683) — likewise; its ENV_SERIAL retry is a
            single bounded re-run that fires only for a failing TEST leg (see
            TestEnvTransientArchivesForHumanTriage).

        No is_infra_transient member remains — the invariant
        ``INFRA_TRANSIENT_CATEGORIES & ARCHIVE_DENY_LIST == frozenset()``,
        enforced at import time by ``_assert_infra_transient_rows_archive``.

        Every other member is still the legacy value; this stays a
        byte-for-byte pin so an UNINTENDED archive-policy change still reds
        here."""
        from orchestrator.verify_categories import ARCHIVE_DENY_LIST
        assert frozenset({
            'compile_error', 'test_failure', 'infra_timeout', 'passed', '',
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

    pytest_internalerror and env_transient have since been ADJUDICATED away
    from the legacy value rather than drifting from it (task 3683): both are
    infra-transient categories that terminate in a blocking human escalation,
    so the archived log is that human's only triage artifact — see
    TestPytestInternalerrorArchivesForHumanTriage and
    TestEnvTransientArchivesForHumanTriage.
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
            ('pytest_internalerror', True),
            ('env_transient', True),
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
# The pair shares an ARCHIVE policy again: task 3679 flipped
# SEMAPHORE_TIMEOUT to archive=True, and task 3683's audit found the same
# grounding applies verbatim to DISK_FULL and flipped it too. See
# TestSemaphoreTimeoutArchivesForHumanTriage and
# TestDiskFullArchivesForHumanTriage — with retry_kind=NONE both surface to a
# human rather than self-clearing, and the archived log is that human's only
# triage artifact.
#
# RED today: neither member/row exists yet, so every test below fails on
# the local ``from orchestrator.verify_categories import FailureCategory``
# attribute access (``FailureCategory.DISK_FULL`` / ``.SEMAPHORE_TIMEOUT``
# don't exist) or on the corresponding _worst_category priority check.
# ---------------------------------------------------------------------------


class TestEnvironmentalCategoriesExistWithInfraTransientPolicy:
    """DISK_FULL / SEMAPHORE_TIMEOUT are new FailureCategory members with an
    env_transient-family CATEGORY_POLICY row.

    The two rows are identical again, ``archive`` included: task 3679 flipped
    SEMAPHORE_TIMEOUT to archive=True, and task 3683 found the same grounding
    holds for DISK_FULL — both are retry_kind=NONE infra-transient rows whose
    bounded retry windows terminate in a blocking human escalation.
    """

    def test_disk_full_value_and_policy(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory, RetryKind
        assert FailureCategory.DISK_FULL.value == 'disk_full'
        row = CATEGORY_POLICY[FailureCategory.DISK_FULL]
        assert row.is_infra_transient is True
        assert row.archive is True
        assert row.preexisting_probe is False
        assert row.retry_kind == RetryKind.NONE

    def test_semaphore_timeout_value_and_policy(self):
        """archive=True since task 3679 — matched by its DISK_FULL sibling
        since task 3683 (see the class docstring)."""
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


class TestDiskFullArchivesForHumanTriage:
    """task 3683: DISK_FULL must archive its verify log.

    Same grounding as its SEMAPHORE_TIMEOUT sibling above, reached through the
    identical code path. DISK_FULL is ``retry_kind=NONE`` — no in-verify serial
    re-run is attempted — and every consumer that retries it does so on FLAT
    ``in INFRA_TRANSIENT_CATEGORIES`` set membership with no per-category
    branch, so it cannot be structurally exempt from any of them:

      * workflow.py:9020 retries any infra-transient VerifyResult for
        ``config.verify_infra_retry_max_attempts`` (default 5) attempts. On
        exhaustion workflow.py:9060-9067 stamps ``escalate_to_human=True`` /
        ``category='infra_issue'``, which short-circuits the steward L0 at
        workflow.py:14436 and files ``Escalation(severity='blocking', level=1,
        suggested_action='manual_intervention')`` (:14791).
      * merge_queue.py:3134 terminates post-merge verify with a
        TRANSIENT_INFRA_REASON_PREFIX blocked outcome → workflow.py:10305 →
        the same blocking L1.
      * verify.py:7255's isolated-confirm gate consumes one of
        ``_SWEEP_CONFIRM_MAX_ATTEMPTS`` per infra-transient hit → a red-main
        blocking L1 at harness.py:11325.

    Archival is decided PER ATTEMPT from the category alone
    (verify.py:1902), inside the retry loop, with no knowledge of whether this
    is the exhausting attempt — so ``archive=False`` discards the log on the
    attempt that hands the incident to a human too, leaving that human with
    only a truncated ``failure_report()``.

    RED today: the row is ``archive=False`` (verify_categories.py:174), so all
    four assertions fail.
    """

    def test_policy_row_archives(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
        assert CATEGORY_POLICY[FailureCategory.DISK_FULL].archive is True

    def test_not_in_archive_deny_list(self):
        """ARCHIVE_DENY_LIST is DERIVED from the row, so this pins that the
        derivation actually followed the row rather than a stale hand-synced
        copy — the drift hazard verify_categories was built to close."""
        from orchestrator.verify_categories import ARCHIVE_DENY_LIST, FailureCategory
        assert FailureCategory.DISK_FULL not in ARCHIVE_DENY_LIST

    def test_should_archive_returns_true(self):
        from orchestrator.verify_categories import should_archive
        assert should_archive('disk_full') is True

    def test_verify_delegation_path_archives(self):
        """The call site that actually decides whether the log survives."""
        from orchestrator import verify
        assert verify._should_archive_category('disk_full') is True


class TestPytestInternalerrorArchivesForHumanTriage:
    """task 3683: PYTEST_INTERNALERROR must archive its verify log.

    This class exists to REBUT the rationale previously pinned in
    ``test_verify.py::TestShouldArchiveCategory`` — "The sweep already retries
    on this category (returns None sentinel); archiving it would create
    spurious human-triage artifacts for transient crashes." That is true of
    exactly ONE arm: the FIRST-PASS main-tip sweep (verify.py:7062/:7141),
    which returns a ``None`` sentinel, retries next tick indefinitely and never
    escalates. It is the only fail-OPEN consumer, and it is an ADDITIONAL path,
    not an exemption from the three that terminate in front of a human:

      * workflow.py:9020 — retries any infra-transient VerifyResult for
        ``config.verify_infra_retry_max_attempts`` (default 5) attempts, then
        stamps ``escalate_to_human=True`` / ``category='infra_issue'``
        (:9060-9067), which short-circuits the steward L0 at :14436 and files
        ``Escalation(severity='blocking', level=1)`` at :14791.
      * merge_queue.py:2761/:3134 — post-merge verify, bounded retry budget,
        then a TRANSIENT_INFRA_REASON_PREFIX blocked outcome →
        workflow.py:10305 → the same blocking L1.
      * verify.py:7255 — the isolated-confirm gate, where an infra-transient
        hit CONSUMES one of ``_SWEEP_CONFIRM_MAX_ATTEMPTS = 2``; two of them
        convert the result into a blocking red-main L1 at harness.py:11325.

    Each of those tests flat ``in INFRA_TRANSIENT_CATEGORIES`` membership with
    no per-category branch, so PYTEST_INTERNALERROR cannot be structurally
    exempt from any of them. "Spurious artifacts for transient crashes" is the
    cost; a human holding a blocking infra_issue with nothing but a truncated
    ``failure_report()`` is what archive=False bought instead. Archive growth
    is bounded independently by ``_prune_archive``'s retention.

    RED today: the row is ``archive=False`` (verify_categories.py:237), so all
    four assertions fail.
    """

    def test_policy_row_archives(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
        assert CATEGORY_POLICY[FailureCategory.PYTEST_INTERNALERROR].archive is True

    def test_not_in_archive_deny_list(self):
        """ARCHIVE_DENY_LIST is DERIVED from the row, so this pins that the
        derivation actually followed the row rather than a stale hand-synced
        copy — the drift hazard verify_categories was built to close."""
        from orchestrator.verify_categories import ARCHIVE_DENY_LIST, FailureCategory
        assert FailureCategory.PYTEST_INTERNALERROR not in ARCHIVE_DENY_LIST

    def test_should_archive_returns_true(self):
        from orchestrator.verify_categories import should_archive
        assert should_archive('pytest_internalerror') is True

    def test_verify_delegation_path_archives(self):
        """The call site that actually decides whether the log survives."""
        from orchestrator import verify
        assert verify._should_archive_category('pytest_internalerror') is True


class TestEnvTransientArchivesForHumanTriage:
    """task 3683: ENV_TRANSIENT must archive its verify log.

    ENV_TRANSIENT is the ONLY one of the three categories this task
    adjudicates with a non-NONE ``retry_kind`` (``RetryKind.ENV_SERIAL``), so
    it is the only one where "it self-clears before it reaches anyone" is even
    arguable. It does not hold, for two independent reasons:

      1. The ENV_SERIAL retry is a SINGLE bounded re-run — "retrying test
         command once, forced serial" (verify.py:5221-5233). Once it is spent,
         a still-ENV_TRANSIENT result surfaces exactly like a retry_kind=NONE
         row.
      2. Per task 3367 the gate is ``category == ENV_TRANSIENT and
         attempt.test.cmd is not None and attempt.test.rc != 0``
         (verify.py:5220-5223), so a LINT or TYPE leg classified ENV_TRANSIENT
         gets ZERO retries and is "reported directly ... and infra-transient at
         the merge lane". Since 3367 there are three ToolKind-INDEPENDENT
         ENV_TRANSIENT producers (a broken ``_merge-verify`` worktree, restart
         collateral, a mis-resolved pyright interpreter — esc-3359-1), so this
         is a live path, not a corner case.

    Past the retry it shares the family terminus: workflow.py:9020 retries it
    on flat ``in INFRA_TRANSIENT_CATEGORIES`` membership for
    ``config.verify_infra_retry_max_attempts`` (default 5) attempts and, on
    exhaustion, stamps ``escalate_to_human=True`` / ``category='infra_issue'``
    (:9060-9067) → blocking level-1 ``Escalation`` at :14791. Corroborated by
    merge_queue.py:3134 → workflow.py:10305 and by verify.py:7255 →
    harness.py:11325.

    RED today: the row is ``archive=False`` (verify_categories.py:258), so all
    four assertions fail.
    """

    def test_policy_row_archives(self):
        from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory
        assert CATEGORY_POLICY[FailureCategory.ENV_TRANSIENT].archive is True

    def test_not_in_archive_deny_list(self):
        """ARCHIVE_DENY_LIST is DERIVED from the row, so this pins that the
        derivation actually followed the row rather than a stale hand-synced
        copy — the drift hazard verify_categories was built to close."""
        from orchestrator.verify_categories import ARCHIVE_DENY_LIST, FailureCategory
        assert FailureCategory.ENV_TRANSIENT not in ARCHIVE_DENY_LIST

    def test_should_archive_returns_true(self):
        from orchestrator.verify_categories import should_archive
        assert should_archive('env_transient') is True

    def test_verify_delegation_path_archives(self):
        """The call site that actually decides whether the log survives."""
        from orchestrator import verify
        assert verify._should_archive_category('env_transient') is True


class TestAssertInfraTransientRowsArchive:
    """task 3683: ``is_infra_transient`` ⇒ ``archive``, enforced at import
    time as a reusable, unit-testable guard.

    The three classes above adjudicate three specific rows. This is the
    durable part: the rule holds UNCONDITIONALLY, so it needs no
    ``human_reachable`` field and no per-category branch. Every consumer that
    retries an infra-transient result does so on flat ``in
    INFRA_TRANSIENT_CATEGORIES`` set membership (workflow.py:9020,
    merge_queue.py:2761, verify.py:7255), so no member can be structurally
    exempt from any of them, and each of those windows is BOUNDED and
    terminates in a blocking level-1 ``infra_issue`` escalation on exhaustion.
    An infra-transient row that does not archive therefore discards the only
    triage artifact its human gets beyond a truncated ``failure_report()``.

    Modelled on ``_validate_exhaustive`` and ``_assert_sentinels_disjoint``:
    the guard takes ``policy`` as a PARAMETER, so a synthetic table exercises
    it directly instead of requiring a real import crash.

    RED today: ``_assert_infra_transient_rows_archive`` does not exist yet.
    """

    def _row(self, *, archive: bool, is_infra_transient: bool):
        """Build a row from the REAL CategoryPolicy dataclass.

        All six fields are required (the dataclass has no defaults on
        purpose), so a synthetic row cannot drift from the real row shape.
        """
        from orchestrator.verify_categories import CategoryPolicy, RetryKind
        return CategoryPolicy(
            severity_rank=0,
            archive=archive,
            preexisting_probe=False,
            is_infra_transient=is_infra_transient,
            verdict_indeterminate=False,
            retry_kind=RetryKind.NONE,
        )

    def test_infra_transient_row_without_archive_raises_and_names_it(self):
        from enum import StrEnum

        from orchestrator.verify_categories import _assert_infra_transient_rows_archive

        class _Synth(StrEnum):
            BAD = 'synthetic_bad_row'
            GOOD = 'synthetic_good_row'

        policy = {
            _Synth.BAD: self._row(archive=False, is_infra_transient=True),
            _Synth.GOOD: self._row(archive=True, is_infra_transient=True),
        }
        # Naming the offender matters: a future author hitting this at import
        # time must learn WHICH row to fix, not just that something is wrong.
        with pytest.raises(AssertionError, match='synthetic_bad_row'):
            _assert_infra_transient_rows_archive(policy)

    def test_ordinary_archive_false_rows_do_not_trip_the_guard(self):
        """The guard must not over-fire on non-infra archive=False rows —
        compile_error/test_failure/passed are legitimately not archived."""
        from enum import StrEnum

        from orchestrator.verify_categories import _assert_infra_transient_rows_archive

        class _Synth(StrEnum):
            INFRA = 'synthetic_infra_row'
            CODE_FAULT = 'synthetic_code_fault_row'

        policy = {
            _Synth.INFRA: self._row(archive=True, is_infra_transient=True),
            _Synth.CODE_FAULT: self._row(archive=False, is_infra_transient=False),
        }
        _assert_infra_transient_rows_archive(policy)

    def test_real_module_import_satisfies_its_own_guard(self):
        # Importing the real module must succeed — its shipped table already
        # satisfies the guard (this is it firing at import time).
        import orchestrator.verify_categories as vc

        vc._assert_infra_transient_rows_archive(vc.CATEGORY_POLICY)

    def test_no_infra_transient_category_is_archive_denied(self):
        """The human-legible form of the same invariant, asserted against the
        DERIVED registries rather than the rows — after task 3683 the two sets
        are disjoint, and the guard is what keeps them that way."""
        from orchestrator.verify_categories import (
            ARCHIVE_DENY_LIST,
            INFRA_TRANSIENT_CATEGORIES,
        )
        assert INFRA_TRANSIENT_CATEGORIES & ARCHIVE_DENY_LIST == frozenset()


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
        # The whole golden set, spelled out. Adding a row here widens the set
        # of local failures that may be overruled by a remote PASS, so it is
        # pinned exactly rather than by spot-checks.
        #
        # REVIEW AMENDMENT (task 3173): narrowed from four members to one.
        # The predicate gained a part (3): the classifier's EVIDENCE for the
        # row must be a STRUCTURAL, out-of-band signal the branch cannot forge
        # in text. merge_queue never sees ground truth — it only ever sees
        # `classify_failure`'s OUTPUT — so a category is only safe here if a
        # branch-caused failure cannot be MISCLASSIFIED into it. disk_full,
        # semaphore_timeout and env_transient each fail (2) or (3) and are
        # adjudicated out below; infra_kill alone is grounded in a waitpid
        # status rather than a text match.
        from orchestrator.verify_categories import INDETERMINATE_VERDICT_CATEGORIES
        assert set(INDETERMINATE_VERDICT_CATEGORIES) == {'infra_kill'}

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


class TestIndeterminateExclusionsAreAdjudicated:
    """REVIEW AMENDMENT (task 3173, blocking finding 2 — ENV_TRANSIENT
    false-GREEN): the three rows that were originally waved in on "it's a host
    condition the diff cannot reach", re-adjudicated against the SHARPENED
    three-part predicate.

    The added part (3) is the one that matters here: merge_queue never observes
    ground truth about WHY a leg failed — it only ever sees the STRING
    ``classify_failure`` returned. So a row whose IDEAL condition is host-only
    but whose PATTERN can swallow a branch-caused failure is still a
    false-GREEN, and part (2) alone does not catch that. Every exclusion below
    asserts both the registry membership AND the underlying policy field, so
    none of them can pass vacuously if the registry is ever re-derived from
    some other set.
    """

    def test_excludes_env_transient(self):
        # Fails part (3). `_classify_environmental`'s OWN docstring
        # (verify_classify.py:498-508) documents the accepted residual: shape-1
        # `_VERIFY_WORKTREE_COLLATERAL_READ_FAILURE_RE` matches a bare
        # "<read verb> ... No such file or directory", and the rustc-span veto
        # only disambiguates rustc's OWN phrasing. A non-rustc guard or build
        # script tripping over a file THE DIFF DELETED OR RENAMED emits text
        # that is indistinguishable from worktree-removal collateral, and still
        # classifies ENV_TRANSIENT. That is a branch-caused failure wearing a
        # host-condition label, so it must keep its veto — fail CLOSED. The row
        # flips back only once shape-1 carries a POSITIVE worktree-removal
        # anchor. `is_infra_transient` is untouched: retrying is still right.
        from orchestrator.verify_categories import (
            CATEGORY_POLICY,
            INDETERMINATE_VERDICT_CATEGORIES,
            INFRA_TRANSIENT_CATEGORIES,
            FailureCategory,
        )
        assert 'env_transient' in INFRA_TRANSIENT_CATEGORIES
        assert 'env_transient' not in INDETERMINATE_VERDICT_CATEGORIES
        assert CATEGORY_POLICY[FailureCategory.ENV_TRANSIENT].verdict_indeterminate is False

    def test_excludes_semaphore_timeout(self):
        # Fails part (3), on the IDENTICAL residual shape as env_transient
        # above — leaving it True after fixing env_transient on these grounds
        # would repeat the very inconsistency the review flagged.
        # verify_classify.py:438-453 records the task-2748/2821 "Known gap"
        # verbatim: a deterministic SHELL-script gate assertion (the docstring's
        # own example is a manifest-drift check — precisely a BRANCH-caused
        # failure) that quotes a lock token plus a timeout token but emits no
        # grounded verdict marker still satisfies the loose `_LOCK_TOKEN_RE` +
        # `_TIMEOUT_TOKEN_RE` co-occurrence and is still classified
        # SEMAPHORE_TIMEOUT.
        from orchestrator.verify_categories import (
            CATEGORY_POLICY,
            INDETERMINATE_VERDICT_CATEGORIES,
            INFRA_TRANSIENT_CATEGORIES,
            FailureCategory,
        )
        assert 'semaphore_timeout' in INFRA_TRANSIENT_CATEGORIES
        assert 'semaphore_timeout' not in INDETERMINATE_VERDICT_CATEGORIES
        assert CATEGORY_POLICY[FailureCategory.SEMAPHORE_TIMEOUT].verdict_indeterminate is False

    def test_excludes_disk_full(self):
        # Fails part (2), NOT part (3) — the distinction is worth keeping
        # explicit. Unlike the two rows above, DISK_FULL's ENOSPC markers ARE
        # solid evidence that the disk really was full; the classification is
        # not in doubt. What is in doubt is CAUSATION: a diff that generates
        # very large build artifacts, or whose new test emits a runaway log,
        # can genuinely cause the ENOSPC itself. So the branch COULD have
        # caused this non-completion, and a local disk_full keeps its veto.
        from orchestrator.verify_categories import (
            CATEGORY_POLICY,
            INDETERMINATE_VERDICT_CATEGORIES,
            INFRA_TRANSIENT_CATEGORIES,
            FailureCategory,
        )
        assert 'disk_full' in INFRA_TRANSIENT_CATEGORIES
        assert 'disk_full' not in INDETERMINATE_VERDICT_CATEGORIES
        assert CATEGORY_POLICY[FailureCategory.DISK_FULL].verdict_indeterminate is False

    def test_infra_kill_is_the_only_member_and_its_evidence_is_structural(self):
        # The positive half of the adjudication: infra_kill is the sole member
        # because it is the ONLY category whose evidence is a waitpid status
        # rather than a text match — part (3) holds by construction, since a
        # branch cannot make the kernel report a negative returncode by
        # printing anything.
        #
        # The non-vacuous grounding for that claim is `is_external_kill_rc`
        # reading the RAW asyncio returncode and nothing else: the ordinary
        # branch-caused OOM kills a memory-hungry GRANDCHILD, and the
        # `start_new_session=True` shell wrapper (verify.py:3423-3443) reports
        # that as POSITIVE 137, which the predicate rejects. Only a signal
        # delivered to our OWN direct child yields the negative rc.
        from orchestrator.verify_categories import (
            INDETERMINATE_VERDICT_CATEGORIES,
            FailureCategory,
        )
        from orchestrator.verify_classify import (
            ToolKind,
            classify_failure,
            is_external_kill_rc,
        )
        assert frozenset({FailureCategory.INFRA_KILL}) == INDETERMINATE_VERDICT_CATEGORIES
        assert is_external_kill_rc(-9) is True
        assert is_external_kill_rc(137) is False
        assert classify_failure(
            ToolKind.PYTEST, 137, 'Killed\n', False
        ) is not FailureCategory.INFRA_KILL

    def test_registry_stays_a_strict_subset_of_infra_transient(self):
        # Keeps `test_is_derived_from_the_policy_rows_not_the_infra_transient_set`
        # non-vacuous at the NEW size: with one member left, the two sets must
        # still be distinct objects with a proper-subset relation, not a
        # coincidental equality that would let the registry silently collapse
        # back into "another name for the infra-transient set".
        from orchestrator.verify_categories import (
            INDETERMINATE_VERDICT_CATEGORIES,
            INFRA_TRANSIENT_CATEGORIES,
        )
        assert INDETERMINATE_VERDICT_CATEGORIES < INFRA_TRANSIENT_CATEGORIES
        assert len(INDETERMINATE_VERDICT_CATEGORIES) == 1
