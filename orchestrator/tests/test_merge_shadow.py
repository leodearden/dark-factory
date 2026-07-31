"""Tests for orchestrator.merge_shadow: extracted warm-vs-cold shadow-compare
subsystem (MQ-refactor task γ).

These tests encode the behavior-preserving contracts of the module split,
mirroring task β's test_merge_gates.py:

1. Module-existence — ``orchestrator.merge_shadow`` exists and exports the
   full closure of moved symbols (state/diff types, per-test parsers,
   sentinels, and the shadow-compare functions).
2. Logger-name — the module logs under the ``orchestrator.merge_queue``
   logger name (not ``orchestrator.merge_shadow``) so existing ``caplog``
   assertions filtered to the merge_queue logger keep capturing the moved
   shadow-compare's WARNING/INFO lines.
3. Reach-back / string-path monkeypatch routing — the existing test suite
   monkeypatches shadow-compare dependencies by STRING PATH
   ``orchestrator.merge_queue.<name>``.  A moved function must resolve a
   monkeypatched-or-staying sibling via a function-local deferred import so
   those patches stay effective even though the function body now lives in
   this module.  Each ``TestReachBackRouting`` test below patches BOTH
   namespaces (merge_shadow-local naive vs. merge_queue reach-back target)
   with CONTRASTING behaviour — or, for a dependency with no merge_shadow
   local copy at all (``_run_unscoped_typechecks``), patches only the
   merge_queue side — so the assertion is unambiguous about which one
   governed.  Includes ``run_scoped_verification``, which the step-3 plan
   prose didn't enumerate by name but which
   :class:`~orchestrator.verify_runner.LocalRunner`'s own docstring and the
   existing ``TestColdShadowVerifyLocalOnly`` tests in
   ``test_merge_queue_multihost_wiring.py`` require: those tests construct a
   real ``LocalRunner`` and patch ``orchestrator.merge_queue.run_scoped_verification``
   to control it, which only works if the injected callable is resolved via
   the merge_queue reach-back rather than a bare local import.
4. Shim re-export identity (added in a later step, once merge_queue.py's
   shim swap lands).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.verify import VerifyResult


def test_merge_shadow_exports_moved_public_symbols() -> None:
    from orchestrator.merge_shadow import (
        _LIBTEST_TEST_LINE_RE,
        _NEXTEST_SUMMARY_LINE_RE,
        _NEXTEST_TEST_LINE_RE,
        _WARM_COLD_SHADOW_SENTINEL,
        _WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL,
        ShadowCompareDiff,
        ShadowCompareState,
        _alarm_warm_shadow_unparseable,
        _classify_test_status,
        _load_shadow_compare_state,
        _maybe_schedule_shadow_compare,
        _nextest_reported_test_count,
        _persistent_alarm_tests,
        _run_cold_shadow_verify,
        _run_shadow_compare,
        _save_shadow_compare_state,
        _shadow_compare_due,
        _submit_shadow_divergence_escalation,
        diff_per_test_results,
        parse_per_test_results,
    )

    for name, obj in {
        'ShadowCompareState': ShadowCompareState,
        'ShadowCompareDiff': ShadowCompareDiff,
        'parse_per_test_results': parse_per_test_results,
        '_classify_test_status': _classify_test_status,
        '_nextest_reported_test_count': _nextest_reported_test_count,
        '_NEXTEST_TEST_LINE_RE': _NEXTEST_TEST_LINE_RE,
        '_LIBTEST_TEST_LINE_RE': _LIBTEST_TEST_LINE_RE,
        '_NEXTEST_SUMMARY_LINE_RE': _NEXTEST_SUMMARY_LINE_RE,
        'diff_per_test_results': diff_per_test_results,
        '_persistent_alarm_tests': _persistent_alarm_tests,
        '_load_shadow_compare_state': _load_shadow_compare_state,
        '_save_shadow_compare_state': _save_shadow_compare_state,
        '_shadow_compare_due': _shadow_compare_due,
        '_WARM_COLD_SHADOW_SENTINEL': _WARM_COLD_SHADOW_SENTINEL,
        '_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL': _WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL,
        '_submit_shadow_divergence_escalation': _submit_shadow_divergence_escalation,
        '_alarm_warm_shadow_unparseable': _alarm_warm_shadow_unparseable,
        '_run_cold_shadow_verify': _run_cold_shadow_verify,
        '_run_shadow_compare': _run_shadow_compare,
        '_maybe_schedule_shadow_compare': _maybe_schedule_shadow_compare,
    }.items():
        assert obj is not None, f'{name} must not be None'


def test_merge_shadow_logger_name_is_merge_queue() -> None:
    """merge_shadow emits under the 'orchestrator.merge_queue' logger name.

    RED (pre-module): ``orchestrator.merge_shadow`` does not exist yet.

    Required so existing ``caplog.at_level(..., logger='orchestrator.merge_queue')``
    assertions in the warm/cold shadow-compare test files keep capturing the
    moved functions' WARNING/INFO-level messages after relocation.
    """
    import orchestrator.merge_shadow as merge_shadow

    assert merge_shadow.logger.name == 'orchestrator.merge_queue'


@pytest.mark.asyncio
class TestReachBackRouting:
    """Reach-back / string-path monkeypatch routing contract.

    Each test patches the SAME logical dependency in both namespaces with
    CONTRASTING behaviour: the merge_shadow-local (naive) patch — where one
    exists — is engineered to raise/diverge so an accidental direct
    reference is caught unambiguously; the orchestrator.merge_queue
    (reach-back target) patch supplies the value that must actually govern
    the outcome.  A dependency with no merge_shadow-local copy at all
    (``_run_unscoped_typechecks``) is patched only on the merge_queue side,
    mirroring task β's analogous ``_check_post_merge_pyright`` test.
    """

    async def test_run_shadow_compare_reachback_to_run_cold_shadow_verify(self) -> None:
        """(1) _run_shadow_compare must resolve _run_cold_shadow_verify via
        orchestrator.merge_queue, not the co-located merge_shadow copy."""
        from orchestrator.merge_shadow import _run_shadow_compare

        git_ops = MagicMock()
        req = MagicMock()
        req.task_id = 'task-shadow-compare-reachback'

        warm_results = {'pkg t': 'pass'}
        naive_cold = AsyncMock(side_effect=AssertionError(
            'naive merge_shadow._run_cold_shadow_verify must not be called'
        ))
        # Reach-back target: identical to warm → no divergence → parity-ok event.
        reachback_cold = AsyncMock(return_value=dict(warm_results))

        event_store = MagicMock()

        with (
            patch('orchestrator.merge_shadow._run_cold_shadow_verify', naive_cold),
            patch('orchestrator.merge_queue._run_cold_shadow_verify', reachback_cold),
        ):
            await _run_shadow_compare(
                git_ops, req, 'commit-sha', warm_results, None, event_store,
            )

        naive_cold.assert_not_called()
        reachback_cold.assert_awaited_once()
        event_store.emit.assert_called_once()
        from orchestrator.event_store import EventType
        emitted_type = event_store.emit.call_args.args[0]
        assert emitted_type == EventType.verdict_parity_ok, (
            f'expected the orchestrator.merge_queue-patched cold results (agreeing '
            f'with warm) to govern and emit verdict_parity_ok, got emit call '
            f'{event_store.emit.call_args!r}'
        )

    async def test_maybe_schedule_shadow_compare_reachback_to_run_shadow_compare(
        self, tmp_path: Path,
    ) -> None:
        """(2) _maybe_schedule_shadow_compare must resolve _run_shadow_compare via
        orchestrator.merge_queue, not the co-located merge_shadow copy."""
        from orchestrator.merge_shadow import _maybe_schedule_shadow_compare

        git_ops = MagicMock()
        config = OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(
                warm_verify_shadow_compare=True,
                warm_verify_shadow_compare_every_n_merges=1,
            ),
        )
        req = MagicMock()
        req.config = config

        worker = MagicMock()
        worker._shadow_state_path = tmp_path / 'warm_verify_shadow.json'
        worker._shadow_compare_tasks = set()

        naive = AsyncMock(side_effect=AssertionError(
            'naive merge_shadow._run_shadow_compare must not be called'
        ))
        reachback = AsyncMock(return_value=None)

        with (
            patch('orchestrator.merge_shadow._run_shadow_compare', naive),
            patch('orchestrator.merge_queue._run_shadow_compare', reachback),
        ):
            await _maybe_schedule_shadow_compare(
                worker, git_ops, req, 'commit-sha', {'pkg t': 'pass'}, None, None,
            )
            assert len(worker._shadow_compare_tasks) == 1, (
                'expected exactly one shadow-compare task to be scheduled'
            )
            task = next(iter(worker._shadow_compare_tasks))
            await task

        naive.assert_not_called()
        reachback.assert_awaited_once()

    async def test_run_cold_shadow_verify_reachback_to_pool_construction_deps(
        self, tmp_path: Path,
    ) -> None:
        """(3) _run_cold_shadow_verify must resolve build_merge_verify_spec,
        VerifyRunnerPool, LocalRunner, and run_scoped_verification via
        orchestrator.merge_queue, not the co-located merge_shadow imports.

        orchestrator.merge_queue's own bindings of build_merge_verify_spec /
        VerifyRunnerPool / LocalRunner are left unpatched (genuine, working
        objects — merge_queue.py has not been touched yet) so this exercises
        the real construction/dispatch chain; only run_scoped_verification is
        overridden (on the merge_queue side) to a fast, distinctive result so
        the test never shells out to a real verify command.
        """
        from orchestrator.merge_shadow import _run_cold_shadow_verify

        git_ops = MagicMock()
        git_ops.create_throwaway_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-cold-reachback'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(project_root=tmp_path)

        reachback_result = VerifyResult(
            passed=True,
            test_output='PASS [0.01s] pkg reachback::marker',
            lint_output='', type_output='', summary='reachback',
        )

        with (
            patch(
                'orchestrator.merge_shadow.build_merge_verify_spec',
                MagicMock(side_effect=AssertionError('naive build_merge_verify_spec used')),
            ),
            patch(
                'orchestrator.merge_shadow.VerifyRunnerPool',
                MagicMock(side_effect=AssertionError('naive VerifyRunnerPool used')),
            ),
            patch(
                'orchestrator.merge_shadow.LocalRunner',
                MagicMock(side_effect=AssertionError('naive LocalRunner used')),
            ),
            patch(
                'orchestrator.merge_shadow.run_scoped_verification',
                AsyncMock(side_effect=AssertionError('naive run_scoped_verification used')),
            ),
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                AsyncMock(return_value=reachback_result),
            ),
        ):
            result = await _run_cold_shadow_verify(git_ops, req, 'commit-sha', None)

        assert result == {'pkg reachback::marker': 'pass'}, (
            f'expected the orchestrator.merge_queue-patched dependency chain to '
            f'govern _run_cold_shadow_verify, got {result!r}'
        )

    async def test_run_cold_shadow_verify_reachback_to_run_unscoped_typechecks(
        self, tmp_path: Path,
    ) -> None:
        """(3 cont'd) _run_cold_shadow_verify must resolve _run_unscoped_typechecks
        via orchestrator.merge_queue.  It has no merge_shadow-local copy at all
        (mirrors task β's analogous _check_post_merge_pyright ↔
        _run_unscoped_typechecks test) so only the merge_queue side is patched;
        with module_configs=[] the REAL _run_unscoped_typechecks always returns
        broken=False, so a broken=True result can only appear in the outcome if
        the merge_queue-patched mock actually governed.
        """
        from orchestrator.merge_shadow import _run_cold_shadow_verify

        git_ops = MagicMock()
        git_ops.create_throwaway_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-cold-unscoped-reachback'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(project_root=tmp_path)

        scoped_result = VerifyResult(
            passed=True, test_output='PASS [0.01s] pkg t',
            lint_output='', type_output='', summary='ok',
        )
        broken_gate = MagicMock(
            broken=True, failing_subprojects=['pkg'], timed_out_subprojects=[],
            detail='mocked-detail',
        )

        with (
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                AsyncMock(return_value=scoped_result),
            ),
            patch(
                'orchestrator.merge_queue._run_unscoped_typechecks',
                AsyncMock(return_value=broken_gate),
            ),
        ):
            result = await _run_cold_shadow_verify(git_ops, req, 'commit-sha', None)

        assert result == {}, (
            f'expected the orchestrator.merge_queue-patched (broken) '
            f'_run_unscoped_typechecks result to short-circuit the scoped '
            f'test_output to empty, got {result!r}'
        )


def test_merge_queue_reexports_identical_objects() -> None:
    """merge_queue re-exports the SAME objects from merge_shadow (shim identity).

    Covers every one of the 20 originally-moved names, plus the two later
    co-located-and-re-exported pure helpers (``build_fail_fast_map`` /
    ``did_not_pass_subset``, PRD verify-retry-failed-only D2) so the identity
    guard extends to them — a future refactor cannot silently break their
    re-export without this test going red.

    RED (pre-shim): merge_queue.py still defines its own independent copies
    of these names (the duplicate definitions left in place by the EXPAND
    step), so ``getattr(merge_queue, name) is getattr(merge_shadow, name)``
    fails for every name — two distinct objects that merely share a name.
    (The two D2 helpers were born re-exported, never duplicated, so the RED
    narrative applies only to the original 20; they are guarded here purely
    forward-looking.)
    """
    import orchestrator.merge_queue as merge_queue
    import orchestrator.merge_shadow as merge_shadow

    moved_names = [
        'ShadowCompareState',
        'ShadowCompareDiff',
        'parse_per_test_results',
        '_classify_test_status',
        '_nextest_reported_test_count',
        '_NEXTEST_TEST_LINE_RE',
        '_LIBTEST_TEST_LINE_RE',
        '_NEXTEST_SUMMARY_LINE_RE',
        'diff_per_test_results',
        '_persistent_alarm_tests',
        '_load_shadow_compare_state',
        '_save_shadow_compare_state',
        '_shadow_compare_due',
        '_WARM_COLD_SHADOW_SENTINEL',
        '_WARM_COLD_SHADOW_UNPARSEABLE_SENTINEL',
        '_submit_shadow_divergence_escalation',
        '_alarm_warm_shadow_unparseable',
        '_run_cold_shadow_verify',
        '_run_shadow_compare',
        '_maybe_schedule_shadow_compare',
        # PRD verify-retry-failed-only D2: born re-exported through the same
        # shim (not part of the original EXPAND-then-shim move), guarded here
        # so their re-export identity cannot silently regress.
        'build_fail_fast_map',
        'did_not_pass_subset',
    ]

    for name in moved_names:
        mq_obj = getattr(merge_queue, name)
        ms_obj = getattr(merge_shadow, name)
        assert mq_obj is ms_obj, (
            f'{name}: orchestrator.merge_queue.{name} and '
            f'orchestrator.merge_shadow.{name} must be the identical object'
        )


# ---------------------------------------------------------------------------
# {did-not-pass} retry-subset construction (PRD verify-retry-failed-only D2).
#
# The core user-observable signal of the failed-only merge-verify retry: under
# nextest fail-fast a failing attempt-0 CANCELS the not-yet-started tests, which
# are ABSENT from parse_per_test_results output.  The sound retry subset is
# therefore {did-not-pass} = failed ∪ not-started ∪ inconclusive, NOT {failed}.
# ---------------------------------------------------------------------------


def test_did_not_pass_subset_includes_non_pass_verdicts() -> None:
    """did_not_pass_subset selects every non-'pass' verdict, sorted.

    The subset MUST include not-started (fail-fast-cancelled) and inconclusive
    tests, not only the 'fail' entry — a failed-only filter would be unsound
    under nextest fail-fast.
    """
    from orchestrator.merge_shadow import did_not_pass_subset

    fail_fast_map = {
        'crate a::x': 'pass',
        'crate a::y': 'fail',
        'crate a::z': 'not-started',
        'crate a::w': 'inconclusive',
    }
    # sorted; failed ∪ not-started ∪ inconclusive — NOT just the 'fail' entry.
    assert did_not_pass_subset(fail_fast_map) == [
        'crate a::w',
        'crate a::y',
        'crate a::z',
    ]


def test_did_not_pass_subset_all_pass_is_empty() -> None:
    """An all-'pass' fail-fast map yields the empty subset (nothing to retry)."""
    from orchestrator.merge_shadow import did_not_pass_subset

    assert did_not_pass_subset({'a::x': 'pass', 'a::y': 'pass'}) == []


# ---------------------------------------------------------------------------
# nextest filter-id mapping (task 3059).
#
# DF's internal key space ("<binary-id> <test-name>", the parse_per_test_results
# key) is NOT the domain cargo-nextest's `test(=...)` matcher accepts.  Resolved
# EMPIRICALLY against cargo-nextest 0.9.136 — the exact version reify's merge
# gate runs — not from documentation:
#
#     cargo nextest list -E 'test(=mymod::mytest)'          -> MATCHES
#     cargo nextest list -E 'test(=nxprobe mymod::mytest)'  -> MATCHES NOTHING
#
# reify wraps each filter-file line as `test(=<line>)` (verify.sh
# emit_nextest_pass), so shipping full parse keys yields a file that is
# non-empty — reify's "retry refused: no subset" loud fallback therefore never
# fires — and that matches ZERO tests: a narrowed retry that runs nothing and
# reports PASS.  A latent FALSE GREEN, not mere inertness.
# ---------------------------------------------------------------------------


def test_nextest_filter_ids_strips_binary_id_prefix() -> None:
    """A "<binary-id> <test-name>" parse key maps to the bare test name.

    Empirical basis (cargo-nextest 0.9.136): `test(=some::mod::test_a)` matches,
    `test(=reify-core::lib some::mod::test_a)` matches nothing.
    """
    from orchestrator.merge_shadow import nextest_filter_ids

    assert nextest_filter_ids(['reify-core::lib some::mod::test_a']) == [
        'some::mod::test_a'
    ]


def test_nextest_filter_ids_passes_through_unqualified_keys() -> None:
    """A key with NO space (parse_per_test_results' libtest branch) is unchanged.

    The libtest branch of parse_per_test_results keys on the bare test path
    already, so there is no binary-id prefix to strip.
    """
    from orchestrator.merge_shadow import nextest_filter_ids

    assert nextest_filter_ids(['some::mod::test_b']) == ['some::mod::test_b']


def test_nextest_filter_ids_splits_on_first_space_only() -> None:
    """The split is on the FIRST space, so a spaced test name keeps its remainder.

    nextest permits spaces in test names for some harnesses; splitting on every
    space would truncate such a name and silently drop it from the retry subset.
    """
    from orchestrator.merge_shadow import nextest_filter_ids

    assert nextest_filter_ids(['pkg::bin my test with spaces']) == [
        'my test with spaces'
    ]


def test_nextest_filter_ids_preserves_order_and_collapses_duplicates() -> None:
    """Input order is preserved; exact duplicates collapse to one entry.

    Two binaries running the same test name yield ONE `test(=name)` term — the
    unqualified term already matches the name in every binary.
    """
    from orchestrator.merge_shadow import nextest_filter_ids

    assert nextest_filter_ids(
        [
            'crate-b::lib zeta::test_last',
            'crate-a::lib alpha::test_first',
            'crate-c::lib alpha::test_first',  # same bare name, different binary
            'crate-b::lib zeta::test_last',  # exact duplicate key
        ]
    ) == ['zeta::test_last', 'alpha::test_first']


def test_nextest_filter_ids_empty_input_is_empty_list() -> None:
    """Empty input yields [] — the caller decides what an empty subset means."""
    from orchestrator.merge_shadow import nextest_filter_ids

    assert nextest_filter_ids([]) == []


# ---------------------------------------------------------------------------
# `cargo nextest list --message-format json` planned-set parsing (task 3059).
#
# The happy path is driven from CHECKED-IN REAL BYTES
# (fixtures/reify_verify_retry/nextest-list.json — unmodified cargo-nextest
# 0.9.136 output), never a hand-written inline copy.  If one of these fails the
# producer's JSON shape has drifted: RE-CAPTURE the fixture and fix the parser.
# Do NOT edit the fixture to make the test pass.  See the fixture dir's
# PROVENANCE.md.
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'reify_verify_retry'


def test_parse_nextest_list_planned_real_bytes() -> None:
    """Real cargo-nextest 0.9.136 JSON parses to "<binary-id> <test-name>" ids.

    Asserts against the fixture's OWN self-declared totals rather than a
    transcribed constant, so a re-capture with a different crate still holds.
    """
    import json

    from orchestrator.merge_shadow import parse_nextest_list_planned

    raw = (_FIXTURE_DIR / 'nextest-list.json').read_text()
    doc = json.loads(raw)

    planned = parse_nextest_list_planned(raw)
    assert planned is not None

    expected = sorted(
        f'{suite.get("binary-id")} {case}'
        for suite in doc['rust-suites'].values()
        for case in suite['testcases']
    )
    assert planned == expected
    # The document's own test-count corroborates that nothing was dropped.
    assert len(planned) == doc['test-count']
    # Ids are in parse_per_test_results' key space: "<binary-id> <test-name>".
    assert all(' ' in test_id for test_id in planned)


def test_parse_nextest_list_planned_unparseable_stdout_is_none() -> None:
    """Non-JSON stdout -> None (a probe FAILURE, routing to a full verify)."""
    from orchestrator.merge_shadow import parse_nextest_list_planned

    assert parse_nextest_list_planned('error: no such command: `nextest`') is None
    assert parse_nextest_list_planned('') is None


def test_parse_nextest_list_planned_wrong_shape_is_none() -> None:
    """Well-formed JSON of the wrong shape -> None, never a silent empty plan."""
    from orchestrator.merge_shadow import parse_nextest_list_planned

    assert parse_nextest_list_planned('[]') is None  # not an object
    assert parse_nextest_list_planned('"a string"') is None
    assert parse_nextest_list_planned('{"test-count": 0}') is None  # no rust-suites
    assert parse_nextest_list_planned('{"rust-suites": []}') is None  # not an object


def test_parse_nextest_list_planned_zero_testcases_is_empty_list() -> None:
    """A well-formed doc whose suites carry zero testcases -> [], NOT None.

    The None-vs-[] distinction is load-bearing: [] is a genuinely test-free
    workspace (a real, if unusual, answer), whereas None means "the probe
    failed, do not narrow anything".
    """
    from orchestrator.merge_shadow import parse_nextest_list_planned

    doc = '{"rust-suites": {"crate-a": {"binary-id": "crate-a", "testcases": {}}}}'
    assert parse_nextest_list_planned(doc) == []


def test_parse_nextest_list_planned_missing_binary_id_falls_back_to_key() -> None:
    """A suite entry missing `binary-id` falls back to the rust-suites map key.

    nextest keys the rust-suites map by the binary id, so the key is the same
    value and a schema change that drops the field stays parseable.
    """
    from orchestrator.merge_shadow import parse_nextest_list_planned

    doc = '{"rust-suites": {"crate-z::lib": {"testcases": {"m::t": {}}}}}'
    assert parse_nextest_list_planned(doc) == ['crate-z::lib m::t']


def test_parse_nextest_list_planned_excludes_ignored_and_filtered_out() -> None:
    """`#[ignore]`d and filterset-excluded testcases are NOT planned.

    Driven by REAL cargo-nextest 0.9.136 bytes
    (``nextest-list-ignored.json``, captured with a `-E` expression so BOTH
    exclusion shapes appear — ``reason: "ignored"`` and ``reason:
    "expression"``).  See PROVENANCE.md.

    Why it matters: `parse_per_test_results` deliberately drops SKIP/ignored
    result lines, so a skipped test NEVER gets a verdict, is annotated
    'not-started' by `build_fail_fast_map`, and therefore lands in the
    {did-not-pass} subset of EVERY narrowed retry.  Never unsafe (nextest still
    refuses to run it) — but it inflates every filter file toward reify's
    REIFY_VERIFY_RETRY_MAX_SUBSET ceiling, and tripping that ceiling makes reify
    refuse narrowing for the whole profile.  An ignore-heavy workspace would
    silently lose the capability.
    """
    import json

    from orchestrator.merge_shadow import parse_nextest_list_planned

    raw = (_FIXTURE_DIR / 'nextest-list-ignored.json').read_text()
    doc = json.loads(raw)

    planned = parse_nextest_list_planned(raw)
    assert planned is not None
    assert planned == ['crate-a alpha::test_one', 'crate-a beta::test_three']

    # The fixture really does carry all three shapes — otherwise this test
    # would pass vacuously against a re-capture that lost them.
    cases = {
        f'{suite["binary-id"]} {case}': meta
        for suite in doc['rust-suites'].values()
        for case, meta in suite['testcases'].items()
    }
    assert cases['crate-a alpha::test_ignored']['ignored'] is True
    assert cases['crate-a alpha::test_ignored']['filter-match'] == {
        'status': 'mismatch', 'reason': 'ignored',
    }
    assert cases['crate-b gamma::test_one']['ignored'] is False
    assert cases['crate-b gamma::test_one']['filter-match'] == {
        'status': 'mismatch', 'reason': 'expression',
    }
    # The document's own `test-count` counts the EXCLUDED cases too, so it is
    # NOT the planned count — pinned here so nobody re-derives the plan from it.
    assert doc['test-count'] == 5
    assert len(planned) == 2


def test_parse_nextest_list_planned_unknown_case_shape_is_included() -> None:
    """An unrecognised testcase shape is treated as PLANNED (superset bias).

    The whole module errs toward re-running MORE tests, never fewer: a probe
    that cannot be understood returns None (full verify), and a testcase that
    cannot be understood stays in the plan.  A future nextest schema change must
    not silently start SKIPPING tests on the retry.
    """
    from orchestrator.merge_shadow import parse_nextest_list_planned

    doc = (
        '{"rust-suites": {"crate-a": {"binary-id": "crate-a", "testcases": {'
        '"m::no_meta": {},'                                     # no fields at all
        '"m::not_a_dict": "surprise",'                          # wrong type
        '"m::odd_status": {"filter-match": {"status": 7}},'     # non-string status
        '"m::odd_filter_match": {"filter-match": "matches"},'   # non-dict filter-match
        '"m::ignored_false": {"ignored": false}'
        '}}}}'
    )
    assert parse_nextest_list_planned(doc) == [
        'crate-a m::ignored_false',
        'crate-a m::no_meta',
        'crate-a m::not_a_dict',
        'crate-a m::odd_filter_match',
        'crate-a m::odd_status',
    ]


def test_parse_nextest_list_planned_all_cases_excluded_is_empty_not_none() -> None:
    """A doc whose every testcase is ignored -> [], not None.

    [] is a real answer the caller turns into "nothing to narrow" (and therefore
    a full verify via the material-narrowing gate); None means "the probe
    failed".  Conflating them would be the false-green this module guards.
    """
    from orchestrator.merge_shadow import parse_nextest_list_planned

    doc = (
        '{"rust-suites": {"crate-a": {"binary-id": "crate-a", "testcases": {'
        '"m::t": {"ignored": true, "filter-match": {"status": "mismatch", '
        '"reason": "ignored"}}}}}}'
    )
    assert parse_nextest_list_planned(doc) == []


# ---------------------------------------------------------------------------
# run_all {failed}-member extraction (task 3059).
#
# Producer contract: reify tests/infra/run_all.sh:26-36 (documented) and
# :1839-1841 (emitted).  Two lines follow the Summary line when anything failed:
#
#   "=== FAILED: <space-separated names> ==="   human-readable summary
#   "FAILED <space-separated names>"            bare classifier marker
#
# The bare marker is the one DF's own verify.py already regexes as `^FAILED\\s`
# (pattern #7b), so this parser consumes an established, source-verified line —
# not an invented format.  Driven from the checked-in fixture; see PROVENANCE.md
# (that fixture is contract-derived-from-source, weaker grounding than a
# captured run).
# ---------------------------------------------------------------------------


def test_parse_failed_run_all_members_real_contract_bytes() -> None:
    """The bare `FAILED <names>` marker yields exactly those names, in order.

    Reads the checked-in fixture whole — including the verbatim contract
    excerpt, whose lines are '#'-commented and so must NOT match.
    """
    from orchestrator.merge_shadow import parse_failed_run_all_members

    text = (_FIXTURE_DIR / 'run_all-failed-marker.txt').read_text()
    assert parse_failed_run_all_members(text) == [
        'test_worktree_lifecycle.sh',
        'test_skip_ledger.sh',
    ]


def test_parse_failed_run_all_members_human_summary_not_double_counted() -> None:
    """The `=== FAILED: ... ===` human summary is not counted alongside the marker.

    Both lines are emitted together by run_all.sh:1839-1841; counting both would
    double every member and (with the '===' token) inject a bogus name.
    """
    from orchestrator.merge_shadow import parse_failed_run_all_members

    log = (
        '=== Summary: 2 discovered, 1 failed ===\n'
        '=== FAILED: test_a.sh ===\n'
        'FAILED test_a.sh\n'
    )
    assert parse_failed_run_all_members(log) == ['test_a.sh']


def test_parse_failed_run_all_members_no_marker_is_empty() -> None:
    """No marker line -> [] — NOT an error.

    An empty subset means reify runs the FULL run_all suite
    (verify.sh:2545 gates on `[ -n "${REIFY_RUN_ALL_MEMBER_SUBSET:-}" ]`), which
    is the safe direction.
    """
    from orchestrator.merge_shadow import parse_failed_run_all_members

    assert parse_failed_run_all_members('=== Summary: 3 discovered, 0 failed ===') == []
    assert parse_failed_run_all_members('') == []


def test_parse_failed_run_all_members_marker_without_names_is_empty() -> None:
    """A marker with no names after it -> [] (full run_all, the safe direction)."""
    from orchestrator.merge_shadow import parse_failed_run_all_members

    assert parse_failed_run_all_members('FAILED \nnext line\n') == []


def test_parse_failed_run_all_members_last_marker_wins() -> None:
    """If the log somehow carries two markers, only the LAST one governs.

    A merge-gate log can concatenate output from more than one run_all
    invocation; the most recent marker is the one describing the final state.
    """
    from orchestrator.merge_shadow import parse_failed_run_all_members

    log = 'FAILED test_old.sh\nmore output\nFAILED test_new_a.sh test_new_b.sh\n'
    assert parse_failed_run_all_members(log) == ['test_new_a.sh', 'test_new_b.sh']


def test_parse_failed_run_all_members_deduplicates_preserving_order() -> None:
    """Repeated names within one marker collapse, first-seen order preserved."""
    from orchestrator.merge_shadow import parse_failed_run_all_members

    assert parse_failed_run_all_members('FAILED b.sh a.sh b.sh\n') == ['b.sh', 'a.sh']


# ---------------------------------------------------------------------------
# The SECOND marker producer: `_ra_on_term`'s `(partial)` outer-timeout line
# (reify tests/infra/run_all.sh:683-685).
#
# Both forms below are matched by the same `^FAILED\s` marker regex as the
# clean producer, so the parser sees them whether or not it was written with
# them in mind.  Every line fed to the parser here is READ OUT OF THE CHECKED-IN
# FIXTURE, where it was recorded by executing the producer statements (see
# PROVENANCE.md) — never hand-authored from prose, which is the drift class this
# leaf exists to correct (PRD §12 root cause (a)).
# ---------------------------------------------------------------------------


def _run_all_fixture_line(exact: str) -> str:
    """Return `exact` as a one-line log, sourced from the checked-in fixture.

    Asserts the fixture carries the line EXACTLY once, so a future fixture edit
    that drops or reworders these producer bytes fails loudly here instead of
    silently turning the assertions below into no-ops against a literal that
    nothing grounds.
    """
    lines = (_FIXTURE_DIR / 'run_all-failed-marker.txt').read_text().splitlines()
    hits = [ln for ln in lines if ln == exact]
    assert len(hits) == 1, (
        f'fixture run_all-failed-marker.txt must carry exactly one {exact!r} '
        f'line (producer bytes); found {len(hits)}'
    )
    return hits[0] + '\n'


def test_parse_failed_run_all_members_partial_marker_without_names_refuses() -> None:
    """`FAILED (partial)` -> [] — the empty-`_names` outer-timeout marker.

    This is the FALSE-GREEN case, and the whole chain runs on a non-empty result:

    1. `_ra_on_term` (run_all.sh:683-685) fires on an outer-timeout SIGTERM
       before any member recorded a nonzero exit, so `${_names:+$_names }`
       expands to nothing and the bare line is `FAILED (partial)`.
    2. A parser that returns the sentinel as a member name yields
       `['(partial)']` — NON-EMPTY.
    3. DF then sets `REIFY_RUN_ALL_MEMBER_SUBSET='(partial)'`, so verify.sh's
       `[ -n "${REIFY_RUN_ALL_MEMBER_SUBSET:-}" ]` gate PASSES and the safe
       full-suite fallback never fires.
    4. run_all.sh:1327 warns `REIFY_RUN_ALL_MEMBER_SUBSET member '(partial)' not
       found in $INFRA_DIR (ignored)` and runs ZERO members — reporting green
       with no coverage at all.

    So `[]` here is not a nicety: it is the difference between a full retry and
    a green that tested nothing.
    """
    from orchestrator.merge_shadow import parse_failed_run_all_members

    log = _run_all_fixture_line('FAILED (partial)')
    got = parse_failed_run_all_members(log)
    assert got != ['(partial)'], (
        'the (partial) sentinel must never be emitted as a member name — '
        'a non-empty subset suppresses the full-suite fallback and runs zero members'
    )
    assert got == []


def test_parse_failed_run_all_members_partial_marker_with_names_refuses() -> None:
    """`FAILED a.sh b.sh (partial)` -> [] as well.

    An INTERRUPTED run's failed-set is not a complete failed-set: members that
    had not yet executed when the SIGTERM landed are neither passed nor failed,
    so narrowing to the named failures would silently SKIP them on the retry —
    the same coverage hole as the empty-names case, merely smaller.  Never
    narrow on an incomplete plan.
    """
    from orchestrator.merge_shadow import parse_failed_run_all_members

    log = _run_all_fixture_line('FAILED a.sh b.sh (partial)')
    assert parse_failed_run_all_members(log) == []


def test_parse_failed_run_all_members_clean_marker_unaffected_by_partial_rule() -> None:
    """No regression: a clean marker still narrows to exactly its names."""
    from orchestrator.merge_shadow import parse_failed_run_all_members

    log = _run_all_fixture_line('FAILED test_worktree_lifecycle.sh test_skip_ledger.sh')
    assert parse_failed_run_all_members(log) == [
        'test_worktree_lifecycle.sh',
        'test_skip_ledger.sh',
    ]


def test_parse_failed_run_all_members_last_marker_wins_across_producers() -> None:
    """Last-marker-wins is unchanged when the two producers are interleaved.

    A merge-gate log can concatenate more than one run_all invocation, and the
    two producers can therefore appear in either order.  The refusal is a
    property of the LAST marker only — it neither poisons an earlier clean
    marker's result nor rescues a later partial one.
    """
    from orchestrator.merge_shadow import parse_failed_run_all_members

    clean = _run_all_fixture_line('FAILED test_worktree_lifecycle.sh test_skip_ledger.sh')
    partial = _run_all_fixture_line('FAILED a.sh b.sh (partial)')

    # clean -> partial: the interrupted run is the final state, so refuse.
    assert parse_failed_run_all_members(clean + 'more output\n' + partial) == []
    # partial -> clean: a completed run superseded it, so its names govern.
    assert parse_failed_run_all_members(partial + 'more output\n' + clean) == [
        'test_worktree_lifecycle.sh',
        'test_skip_ledger.sh',
    ]


def test_parse_failed_run_all_members_human_partial_summary_alone_is_empty() -> None:
    """The `=== FAILED: ... (partial) ===` human summary alone still yields [].

    `_ra_on_term` emits it on the line before the bare marker.  The `^FAILED`
    anchor excludes it, so it is neither matched on its own nor double-counted
    beside the bare marker.  Both rendered forms are covered — note the empty-
    `_names` one carries the producer's DOUBLE space after the colon.
    """
    from orchestrator.merge_shadow import parse_failed_run_all_members

    assert parse_failed_run_all_members(_run_all_fixture_line('=== FAILED:  (partial) ===')) == []
    assert (
        parse_failed_run_all_members(
            _run_all_fixture_line('=== FAILED: a.sh b.sh (partial) ===')
        )
        == []
    )


def test_build_fail_fast_map_marks_cancelled_tests_not_started() -> None:
    """build_fail_fast_map annotates the authoritative plan with attempt-0 verdicts.

    A test present in ``planned`` but absent from ``verdicts`` was cancelled by
    nextest fail-fast and is annotated ``'not-started'`` — the crux of the
    soundness core.  ``planned`` is authoritative: a verdict key not in
    ``planned`` is ignored.
    """
    from orchestrator.merge_shadow import build_fail_fast_map

    planned = ['crate a::x', 'crate a::y', 'crate a::z']
    # As produced by parse_per_test_results on attempt-0 output: 'crate a::z'
    # was cancelled by fail-fast and is ABSENT.  'crate a::stale' is a verdict
    # for a test NOT in the plan and must be ignored.
    verdicts = {
        'crate a::x': 'pass',
        'crate a::y': 'fail',
        'crate a::stale': 'pass',
    }
    assert build_fail_fast_map(planned, verdicts) == {
        'crate a::x': 'pass',
        'crate a::y': 'fail',
        'crate a::z': 'not-started',
    }


def test_build_fail_fast_map_then_subset_retains_not_started() -> None:
    """End-to-end: the {did-not-pass} subset retains the fail-fast-cancelled test.

    A raw failed-only view would drop 'crate a::z'; feeding the fail-fast map
    through did_not_pass_subset keeps it (soundness).
    """
    from orchestrator.merge_shadow import build_fail_fast_map, did_not_pass_subset

    planned = ['crate a::x', 'crate a::y', 'crate a::z']
    verdicts = {'crate a::x': 'pass', 'crate a::y': 'fail'}
    assert did_not_pass_subset(build_fail_fast_map(planned, verdicts)) == [
        'crate a::y',
        'crate a::z',
    ]


# ---------------------------------------------------------------------------
# shadow-baseline map merge (PRD verify-retry-failed-only D4, §5.4).
#
# A narrowed {did-not-pass} merge-verify retry re-runs ONLY the tests that did
# not pass in attempt-0, so its per-test map is PARTIAL — it omits every test
# that already passed.  Storing that partial map as the warm shadow baseline
# makes the next FULL cold shadow compare classify every attempt-0-passed test
# as only_cold → phantom born-at-L2 divergence alarm.  merge_retry_shadow_baseline
# unions attempt-0's passes with the fresh retry map so the baseline is whole.
# ---------------------------------------------------------------------------


def test_merge_retry_shadow_baseline_carries_forward_and_retry_overwrites() -> None:
    """attempt-0 passes carry forward; retry wins on overlap (did-not-pass→pass).

    A,B passed in attempt-0 and are absent from the narrowed retry map — they
    must reappear in the merged baseline (else the full cold suite flags them
    only_cold → phantom divergence).  C did-not-pass in attempt-0 and the fresh
    retry pass is the latest verdict, so it overwrites C's attempt-0 'fail'.
    """
    from orchestrator.merge_shadow import merge_retry_shadow_baseline

    attempt0 = {'A': 'pass', 'B': 'pass', 'C': 'fail'}
    retry = {'C': 'pass'}
    assert merge_retry_shadow_baseline(attempt0, retry) == {
        'A': 'pass',
        'B': 'pass',
        'C': 'pass',
    }


def test_merge_retry_shadow_baseline_retry_precedence_over_inconclusive() -> None:
    """retry verdict wins on overlap even over an attempt-0 'inconclusive'."""
    from orchestrator.merge_shadow import merge_retry_shadow_baseline

    attempt0 = {'A': 'pass', 'C': 'inconclusive'}
    retry = {'C': 'pass'}
    assert merge_retry_shadow_baseline(attempt0, retry) == {'A': 'pass', 'C': 'pass'}


def test_merge_retry_shadow_baseline_drops_attempt0_non_pass_absent_from_retry() -> None:
    """attempt-0 non-pass verdicts NOT re-run by the retry are dropped, not carried.

    A stray 'fail'/'not-started' surviving into the baseline would be flipped by
    the full cold suite into a genuine-looking divergence — the exact phantom
    this helper exists to remove.  Only attempt-0 PASSES are carried forward.
    """
    from orchestrator.merge_shadow import merge_retry_shadow_baseline

    assert merge_retry_shadow_baseline({'A': 'pass', 'C': 'fail'}, {}) == {'A': 'pass'}


def test_merge_retry_shadow_baseline_does_not_mutate_inputs() -> None:
    """The input maps are treated as read-only (no in-place mutation)."""
    from orchestrator.merge_shadow import merge_retry_shadow_baseline

    attempt0 = {'A': 'pass', 'C': 'fail'}
    retry = {'C': 'pass'}
    merge_retry_shadow_baseline(attempt0, retry)
    assert attempt0 == {'A': 'pass', 'C': 'fail'}
    assert retry == {'C': 'pass'}


def test_build_warm_shadow_results_narrowed_merges_attempt0_passes() -> None:
    """NARROWED warm retry: partial retry output ∪ attempt-0 passes → whole map.

    ``test_output`` is the PARTIAL narrowed-retry output containing ONLY the
    re-run test; ``attempt0_verdicts`` supplies the attempt-0 pass that the
    narrowed run omitted.  The result must be the merged full-suite baseline.
    """
    from orchestrator.merge_shadow import build_warm_shadow_results

    test_output = 'PASS [   0.05s] reify-spec test_retried\n'
    attempt0 = {'reify-spec test_passed': 'pass'}
    assert build_warm_shadow_results(test_output, attempt0) == {
        'reify-spec test_passed': 'pass',
        'reify-spec test_retried': 'pass',
    }


def test_build_warm_shadow_results_non_narrowed_is_parse_only() -> None:
    """NON-narrowed (no attempt-0 map): byte-identical to a plain parse.

    ``attempt0_verdicts=None`` (the default) and an explicit empty ``{}`` must
    both return exactly ``parse_per_test_results(test_output)`` — the
    non-narrowed warm path stays byte-identical.
    """
    from orchestrator.merge_shadow import (
        build_warm_shadow_results,
        parse_per_test_results,
    )

    test_output = (
        'PASS [   0.05s] reify-spec test_a\n'
        'PASS [   0.06s] reify-spec test_b\n'
    )
    expected = parse_per_test_results(test_output)
    assert expected == {'reify-spec test_a': 'pass', 'reify-spec test_b': 'pass'}
    # default None
    assert build_warm_shadow_results(test_output) == expected
    # explicit empty attempt0 behaves like None (parse-only)
    assert build_warm_shadow_results(test_output, {}) == expected


def test_build_warm_shadow_results_empty_parse_returned_unchanged() -> None:
    """EMPTY/unparseable retry: return ``{}`` verbatim, NEVER merged.

    An empty retry parse must NOT be unioned with attempt-0 (that would turn an
    unparseable narrowed retry into a non-empty PARTIAL baseline, masking the
    fail-closed ``_alarm_warm_shadow_unparseable`` and feeding the shadow
    compare the exact partial map D4 exists to eliminate).
    """
    from orchestrator.merge_shadow import build_warm_shadow_results

    assert build_warm_shadow_results('', {'reify-spec test_passed': 'pass'}) == {}


# ---------------------------------------------------------------------------
# task 3018 (steps 9-10): a LIVE cold-shadow throwaway worktree must survive a
# concurrent periodic reap.
#
# Task 3018 promoted `reap_orphaned_merge_worktrees` from a startup-only sweep
# to a steady-state one fired from the merge worker's `_heartbeat_loop`, which
# silently turned RESOURCE_AUDIT_WORKTREE_GRACE_SECS from a *detection*
# threshold into a *destruction* deadline.  The cold-shadow throwaway
# `_merge-<uuid>` worktree is neither registered in `_owned_merge_worktrees`
# nor touched by `_touch_owned_merge_worktrees`, so its measured age is real
# elapsed time and the reap's only remaining liveness gate is the per-lane
# merge-verify flock consulted by `remove_merge_worktree_guarded` (the C1
# primitive the reaper routes through).
#
# These fixtures deliberately use a REAL git repo + REAL throwaway worktree
# (not the MagicMock git_ops the reach-back tests above use) so that
# `lane_lock_path(wt)` names a real file and the flock contention is genuine —
# a mocked git_ops cannot exercise a real lock.  Adapted from the fixture
# pattern in test_remove_merge_worktree_guarded.py:46-84.
# ---------------------------------------------------------------------------


async def _setup_shadow_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _shadow_head_sha(repo: Path) -> str:
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


@pytest.fixture
def shadow_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_shadow_repo(repo))
    return repo


@pytest.fixture
def shadow_git_ops(shadow_git_repo: Path) -> GitOps:
    git_config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(git_config, shadow_git_repo)


@pytest.mark.asyncio
class TestColdShadowVerifyHoldsLaneLease:
    """_run_cold_shadow_verify must hold merge_verify_lease(lane_dir=wt) (3018).

    The lane flock is the codebase's canonical "this tree is live" signal —
    it is exactly what `remove_merge_worktree_guarded`'s acquire-then-remove
    C1 primitive consults (git_ops.py:8486).  Holding it across the cold
    verify makes the throwaway tree's protection independent of how long the
    verify runs, rather than resting on an age heuristic.
    """

    async def test_live_cold_shadow_worktree_survives_concurrent_reap(
        self, shadow_git_ops: GitOps, shadow_git_repo: Path,
    ) -> None:
        """A periodic reap firing mid-verify must be REFUSED, and the tree must
        still be cleaned up once the verify returns.

        Three assertions, in order of load-bearingness:

        (a) the simulated sweep's outcome is ``'skipped_lease_held'`` — a
            concurrent periodic reap CANNOT delete the checkout out from under
            a running cold verify.  This is the assertion that goes RED today:
            with no lease held, the sweep acquires the uncontended
            ``<wt>.lock`` and returns ``'removed'``.
        (b) the worktree still existed at that moment (the skip was real, not
            an artefact of the tree already being gone).
        (c) AFTER `_run_cold_shadow_verify` returns the worktree is GONE —
            pinning that the lease is released BEFORE the existing
            ``finally: cleanup_merge_worktree(wt)``.  If the lease wrapped the
            finally too, the guarded removal's NON-BLOCKING acquire would fail
            against OURSELVES and return ``'skipped_lease_held'``, leaking the
            very tree the finally exists to remove — a worse leak than the one
            being fixed.  (c) passes vacuously in the RED state, which is why
            (a) leads.
        """
        from orchestrator.merge_shadow import _run_cold_shadow_verify

        head = await _shadow_head_sha(shadow_git_ops.project_root)
        recorded: list[tuple[str, bool, Path]] = []

        async def _reaping_scoped(worktree: Path, *args, **kwargs) -> VerifyResult:
            # Simulate the task-3018 periodic sweep firing WHILE the cold
            # verify is running, via the very primitive
            # reap_orphaned_merge_worktrees routes its removals through.
            outcome = await shadow_git_ops.remove_merge_worktree_guarded(
                worktree, reason='periodic-reap-sim',
            )
            recorded.append((outcome, worktree.exists(), worktree))
            return VerifyResult(
                passed=True,
                test_output='PASS [0.01s] pkg cold::marker',
                lint_output='', type_output='', summary='cold',
            )

        req = MagicMock()
        req.task_id = 'task-3018-cold-lease'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(project_root=shadow_git_repo)

        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _reaping_scoped,
        ):
            result = await _run_cold_shadow_verify(shadow_git_ops, req, head, None)

        assert recorded, (
            'the patched run_scoped_verification never ran, so the concurrent '
            'reap was never simulated — the test proves nothing'
        )
        outcome, existed_mid_verify, wt = recorded[0]
        assert outcome == 'skipped_lease_held', (
            f'a periodic reap firing during a LIVE cold shadow verify must be '
            f'refused by the lane flock, got {outcome!r} — the throwaway '
            f'worktree at {wt} would have been deleted out from under the '
            f'running verify'
        )
        assert existed_mid_verify is True, (
            f'the throwaway worktree {wt} must still exist at the moment the '
            f'reap is refused (otherwise the skip is vacuous)'
        )
        assert not wt.exists(), (
            f'the lease must be released BEFORE the finally-block '
            f'cleanup_merge_worktree, else the cleanup skips on our own lease '
            f'and leaks {wt}'
        )
        assert result == {'pkg cold::marker': 'pass'}, (
            f'the cold verify must still return its parsed per-test results '
            f'unchanged, got {result!r}'
        )
