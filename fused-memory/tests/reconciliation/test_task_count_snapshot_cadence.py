"""Tests for task_count_snapshot_cadence — task 2278.

Hardens the Mem0 ``task_count_snapshot`` write cadence written by Stage 2
(task_knowledge_sync) as its final action each cycle: a Stage-2 freshness
stat plus a harness consecutive-full-cycle-miss escalation.

Covers (grown step-by-step per plan.json):
- TestConstants                        (step-1/2, step-3/4)
- TestExtractSnapshotWritten            (step-1/2)
- TestComputeSnapshotMissStreak         (step-1/2)
- TestEvaluateSnapshotCadence           (step-3/4)
- TestBuildStaleSnapshotFinding         (step-3/4)

Task 2325 follow-up (exempt never-snapshotted projects + make the Stage-2
task_count_snapshot write structural):
- TestBuildTaskCountSnapshotContent     (task 2325 step-3/4)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from fused_memory.models.reconciliation import (
    StageId,
    StageReport,
)
from fused_memory.reconciliation.task_count_snapshot_cadence import (
    ESCALATION_CATEGORY,
    SNAPSHOT_PRUNE_ENUMERATED_STAT_KEY,
    SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY,
    SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY,
    SNAPSHOT_PRUNED_STAT_KEY,
    SNAPSHOT_WRITTEN_STAT_KEY,
    TASK_COUNT_SNAPSHOT_CATEGORY,
    TASK_COUNT_SNAPSHOT_KIND,
    TASK_COUNT_SNAPSHOT_MISS_THRESHOLD,
    build_stale_snapshot_finding,
    build_task_count_snapshot_content,
    build_task_count_snapshot_unavailable_content,
    compute_snapshot_miss_streak,
    evaluate_snapshot_cadence,
    extract_snapshot_written,
)
from fused_memory.reconciliation.task_filter import is_count_snapshot


def _stage_report(stats: dict) -> StageReport:
    """Build a minimal real StageReport carrying *stats*."""
    now = datetime.now(UTC)
    return StageReport(
        stage=StageId.task_knowledge_sync,
        started_at=now,
        completed_at=now,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Assert module-level constants have expected values."""

    def test_kind_value(self):
        assert TASK_COUNT_SNAPSHOT_KIND == 'task_count_snapshot'

    def test_stat_key_value(self):
        """Exact-value pin — also the anti-regression guard for the task-3045
        rename: the un-namespaced spelling read as a persistence claim about
        Graphiti, which is what drove Stage 3 / the judge to report a
        "rejected write" for a snapshot that is Mem0-only BY DESIGN (see
        Snapshot Discipline, prompts/stage1.py). Pinning the exact value is
        what stops a future edit from quietly restoring a Graphiti-implying
        spelling.
        """
        assert SNAPSHOT_WRITTEN_STAT_KEY == 'task_count_snapshot_mem0_written'

    def test_prune_enumerated_stat_key_value(self):
        assert SNAPSHOT_PRUNE_ENUMERATED_STAT_KEY == 'task_count_snapshot_prune_enumerated'

    def test_pruned_stat_key_value(self):
        assert SNAPSHOT_PRUNED_STAT_KEY == 'task_count_snapshot_mem0_pruned'

    def test_prune_enumeration_ok_stat_key_value(self):
        assert SNAPSHOT_PRUNE_ENUMERATION_OK_STAT_KEY == 'task_count_snapshot_prune_enumeration_ok'

    def test_prune_truncated_stat_key_value(self):
        assert SNAPSHOT_PRUNE_TRUNCATED_STAT_KEY == 'task_count_snapshot_prune_truncated'

    def test_miss_threshold_value(self):
        assert TASK_COUNT_SNAPSHOT_MISS_THRESHOLD == 2

    def test_escalation_category_value(self):
        assert ESCALATION_CATEGORY == 'recon_stale_task_count_snapshot'


# ---------------------------------------------------------------------------
# extract_snapshot_written
# ---------------------------------------------------------------------------


class TestExtractSnapshotWritten:
    """extract_snapshot_written(stage_report) -> bool | None.

    Must handle a real StageReport AND a raw dict shape identically (mirrors
    the isinstance(x, dict) guard convention used elsewhere in reconciliation,
    e.g. journal.get_run's stage_reports reconstruction).
    """

    def test_stats_1_is_true_on_stage_report(self):
        report = _stage_report({SNAPSHOT_WRITTEN_STAT_KEY: 1})
        assert extract_snapshot_written(report) is True

    def test_stats_0_is_false_on_stage_report(self):
        report = _stage_report({SNAPSHOT_WRITTEN_STAT_KEY: 0})
        assert extract_snapshot_written(report) is False

    def test_missing_key_is_none_on_stage_report(self):
        report = _stage_report({})
        assert extract_snapshot_written(report) is None

    def test_stats_1_is_true_on_raw_dict(self):
        report = {'stats': {SNAPSHOT_WRITTEN_STAT_KEY: 1}}
        assert extract_snapshot_written(report) is True

    def test_stats_0_is_false_on_raw_dict(self):
        report = {'stats': {SNAPSHOT_WRITTEN_STAT_KEY: 0}}
        assert extract_snapshot_written(report) is False

    def test_missing_key_is_none_on_raw_dict(self):
        report = {'stats': {}}
        assert extract_snapshot_written(report) is None

    def test_missing_stats_key_is_none_on_raw_dict(self):
        assert extract_snapshot_written({}) is None

    def test_none_report_is_none(self):
        assert extract_snapshot_written(None) is None

    def test_neither_key_present_is_still_none(self):
        report = _stage_report({'some_unrelated_stat': 1})
        assert extract_snapshot_written(report) is None

    # --- pre-rename spelling is retired, not honored (task 3488) ------------
    #
    # Task 3045's read-only legacy alias was deleted once measurement showed
    # no in-window journal row could still change the computed miss streak:
    # all 24 in-window full+completed pre-rename rows carried value 1, and
    # compute_snapshot_miss_streak breaks on anything that is not False, so
    # True (fallback present) and None (fallback gone) are indistinguishable
    # to it. Value 0 is the only value the fallback could have changed, and
    # none was in-window.
    #
    # The absent-vs-legitimate-zero distinction that the deleted `in stats`
    # membership test used to protect is guarded by the EXISTING
    # test_stats_0_is_false_on_stage_report / test_missing_key_is_none_on_stage_report
    # above -- those two are load-bearing for the `.get(...)` collapse and
    # must not be removed: collapsing 0 (confirmed miss) into None (unknown)
    # would be a real fail-quiet regression, since unknown STOPS the streak.
    #
    # Raw literals below, not a constant: the constant is gone, and the
    # literal is what real journal blobs contain.
    #
    # These two are the PRIMARY regression guard on the retirement: they fail
    # for any behavioural resurrection of the fallback -- under the old
    # constant name, a new one, an inline literal, or a `.get(...) or ...`
    # chain. (An earlier symbol-introspection assertion on the constant's
    # absence was dropped as strictly weaker: it missed every spelling but
    # one, and fired on a harmless unused re-declaration.)

    def test_pre_rename_key_0_is_unknown_on_stage_report(self):
        """The important direction: a pre-rename CONFIRMED MISS must now read
        as unknown, never as False.

        Unknown stops the miss streak, so retirement can under-escalate by at
        most one cycle and can never over-escalate -- the fail-safe side.
        """
        report = _stage_report({'task_count_snapshot_written': 0})
        assert extract_snapshot_written(report) is None

    def test_pre_rename_key_1_is_unknown_on_raw_dict(self):
        report = {'stats': {'task_count_snapshot_written': 1}}
        assert extract_snapshot_written(report) is None


# ---------------------------------------------------------------------------
# compute_snapshot_miss_streak
# ---------------------------------------------------------------------------


class TestComputeSnapshotMissStreak:
    """compute_snapshot_miss_streak(recent_flags) -> int.

    recent_flags is most-recent-first list of bool|None; counts the leading
    run of consecutive False, stopping at the first True (a written cycle
    resets the streak) or None (unknown -> stop, fail-safe).
    """

    @pytest.mark.parametrize(
        ('recent_flags', 'expected'),
        [
            ([], 0),
            ([True], 0),
            ([False], 1),
            ([False, False], 2),
            ([False, True, False], 1),
            ([False, None, False], 1),
        ],
    )
    def test_streak(self, recent_flags, expected):
        assert compute_snapshot_miss_streak(recent_flags) == expected


# ---------------------------------------------------------------------------
# evaluate_snapshot_cadence
# ---------------------------------------------------------------------------


class TestEvaluateSnapshotCadence:
    """evaluate_snapshot_cadence(current_written, prior_flags, *, blocked, threshold) -> dict.

    Returns {'streak': int, 'escalate': bool}.
    """

    def test_current_written_true_never_escalates_regardless_of_priors(self):
        result = evaluate_snapshot_cadence(True, [False, False, False], blocked=False)
        assert result['escalate'] is False

    def test_current_written_none_never_escalates(self):
        """Unknown current cycle -> never escalate (fail-safe)."""
        result = evaluate_snapshot_cadence(None, [False, False, False], blocked=False)
        assert result['escalate'] is False

    def test_blocked_project_never_escalates_even_with_long_streak(self):
        result = evaluate_snapshot_cadence(False, [False] * 5, blocked=True)
        assert result['escalate'] is False

    def test_current_false_empty_priors_streak_1_below_threshold(self):
        result = evaluate_snapshot_cadence(False, [], blocked=False)
        assert result == {'streak': 1, 'escalate': False}

    def test_current_false_one_prior_miss_streak_2_meets_threshold(self):
        result = evaluate_snapshot_cadence(False, [False], blocked=False)
        assert result == {'streak': 2, 'escalate': True}

    def test_current_false_prior_write_resets_streak_to_1(self):
        """A prior successful write resets the streak; current miss alone is 1."""
        result = evaluate_snapshot_cadence(False, [True], blocked=False)
        assert result == {'streak': 1, 'escalate': False}


# ---------------------------------------------------------------------------
# build_stale_snapshot_finding
# ---------------------------------------------------------------------------


class TestBuildStaleSnapshotFinding:
    """build_stale_snapshot_finding(project_id) -> dict.

    Stable identity ({category, affected_ids, description}) so _escalate's
    content-fingerprint dedup folds repeats into a single pending escalation
    (mirrors _DEAD_OWNER_STORM_FINDING).
    """

    def test_category_is_escalation_category(self):
        finding = build_stale_snapshot_finding('reify')
        assert finding['category'] == 'recon_stale_task_count_snapshot'

    def test_affected_ids_scoped_to_project(self):
        finding = build_stale_snapshot_finding('reify')
        assert finding['affected_ids'] == ['task_count_snapshot:reify']

    def test_description_is_stable_and_non_empty(self):
        first = build_stale_snapshot_finding('reify')
        second = build_stale_snapshot_finding('reify')
        assert first['description']
        assert first['description'] == second['description']


# ---------------------------------------------------------------------------
# build_task_count_snapshot_content (task 2325 step-3/4)
# ---------------------------------------------------------------------------


class TestBuildTaskCountSnapshotContent:
    """build_task_count_snapshot_content(...) -> str (task 2325).

    Pure, dependency-free renderer for the deterministic task_count_snapshot
    Mem0 write content -- no I/O, no mocks needed. Mirrors the module's
    "pure compute helper" contract (see module docstring).
    """

    def test_category_constant_value(self):
        assert TASK_COUNT_SNAPSHOT_CATEGORY == 'observations_and_summaries'

    def test_content_contains_project_counts_highest_id_and_date(self):
        content = build_task_count_snapshot_content(
            'reify',
            total=42, done=18, cancelled=3, active=20, other=1,
            highest_task_id=99, as_of='2026-07-08',
        )

        assert isinstance(content, str)
        assert 'reify' in content
        assert '2026-07-08' in content
        # Labeled substrings (not bare digits) so the assertion actually pins
        # each count to its own label -- a swap of e.g. done/cancelled in the
        # renderer, or an incidental digit match (e.g. '1' inside '18'),
        # would fail here (reviewer finding, amendment round).
        assert '42 total' in content
        assert '18 done' in content
        assert '3 cancelled' in content
        assert '20 active' in content
        assert '1 other' in content
        assert 'highest task id 99' in content

    def test_content_without_as_of_is_non_empty_and_contains_counts(self):
        content = build_task_count_snapshot_content(
            'reify',
            total=42, done=18, cancelled=3, active=20, other=1,
            highest_task_id=99, as_of=None,
        )

        assert isinstance(content, str)
        assert content
        assert 'reify' in content
        assert '42' in content
        assert '99' in content


class TestBuildTaskCountSnapshotUnavailableContent:
    """build_task_count_snapshot_unavailable_content(...) -> str (task 2738).

    Pure, dependency-free renderer for the UNKNOWN sentinel written by
    ``_write_task_count_snapshot`` when a zero-count project_root is not a
    readable git working tree -- i.e. the zero count is a false census from
    SqliteTaskBackend.get_tasks auto-creating an empty tasks.db, not a
    genuinely empty project. Sibling of TestBuildTaskCountSnapshotContent
    above: no I/O, no mocks needed.
    """

    def test_content_is_non_empty_string(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert isinstance(out, str)
        assert out

    def test_content_is_not_mistaken_for_a_numeric_count_snapshot(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert is_count_snapshot(out) is False

    def test_content_names_project_id_and_root(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert 'my_solar_challenge' in out
        assert '/home/leo/src/my-solar-challenge' in out

    def test_content_signals_unavailability_and_reason(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert 'UNAVAILABLE' in out or 'unavailable' in out
        # A stable, semantically-load-bearing token rather than the full
        # prose sentence -- pins "this is about git working tree
        # readability" without locking the exact wording (amendment,
        # task 2738 review).
        assert 'git working tree' in out

    def test_content_carries_no_zeroed_numeric_census(self):
        out = build_task_count_snapshot_unavailable_content(
            'my_solar_challenge', '/home/leo/src/my-solar-challenge',
        )

        assert '0 total' not in out
        assert '0 done' not in out


# ---------------------------------------------------------------------------
# Stage-3 prompt coupling — the stat-vs-edge guidance names the LIVE keys
# (task 3045 step-7/8)
# ---------------------------------------------------------------------------


class TestStage3PromptNamesMem0SnapshotStats:
    """STAGE3_SYSTEM_PROMPT must name the stat keys Stage 3 actually receives.

    _format_report dumps the WHOLE stats dict verbatim
    (``json.dumps(report.stats)``) into Stage 3's payload, so the renamed
    keys land in front of the model unfiltered. The rename fixes the NAME;
    the model still has to be told the invariant, or it can reconstruct the
    same "stat claims a write but there is no Graphiti edge" false positive
    from a differently-spelled key.

    Deliberately a COUPLING test, not a prose pin, following the established
    assembled-prompt drift-guard convention (tests/test_recon_report_guidance_drift.py,
    tests/test_standing_decision_prompt_drift.py): every assertion is keyed
    off the imported constants, so a future rename cannot orphan the
    guidance — leaving the prompt warning about a key the model never sees
    while the false positive quietly returns.

    The contract is PRESENCE-ONLY. No assertion is made about wording,
    sentence structure, section ordering, or what the prompt must NOT say —
    the prompt stays free to mention the legacy spelling if a future cycle
    wants to explain the rename to the model during the back-compat window
    that extract_snapshot_written's legacy fallback keeps open.
    """

    def test_prompt_names_the_written_stat_key(self):
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        assert SNAPSHOT_WRITTEN_STAT_KEY in STAGE3_SYSTEM_PROMPT

    def test_prompt_names_the_pruned_stat_key(self):
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        assert SNAPSHOT_PRUNED_STAT_KEY in STAGE3_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Judge prompt coupling — same guidance, same drift guard (task 3045,
# amendment round: architecture-coherence finding)
# ---------------------------------------------------------------------------


class TestJudgePromptNamesMem0SnapshotStats:
    """JUDGE_SYSTEM_PROMPT must name the stat keys the judge actually receives.

    Same contract and same rationale as TestStage3PromptNamesMem0SnapshotStats
    above, for the OTHER reader of Stage 2's stats: judge.py puts report.stats
    in the judge prompt, and JUDGE_SYSTEM_PROMPT explicitly instructs the model
    to "cross-reference stage report stats against MCP Actions to verify
    consistency" -- which is the instruction that manufactures the false
    positive this task exists to kill. Renaming the keys without telling the
    judge the invariant leaves it free to read
    ``task_count_snapshot_mem0_written=1`` with no matching Graphiti edge as a
    stats-vs-reality mismatch.

    PRESENCE-ONLY, keyed off the imported constants -- no assertion about
    wording, ordering, or what the prompt must NOT say.
    """

    def test_prompt_names_the_written_stat_key(self):
        from fused_memory.reconciliation.prompts.judge import JUDGE_SYSTEM_PROMPT

        assert SNAPSHOT_WRITTEN_STAT_KEY in JUDGE_SYSTEM_PROMPT

    def test_prompt_names_the_pruned_stat_key(self):
        from fused_memory.reconciliation.prompts.judge import JUDGE_SYSTEM_PROMPT

        assert SNAPSHOT_PRUNED_STAT_KEY in JUDGE_SYSTEM_PROMPT
