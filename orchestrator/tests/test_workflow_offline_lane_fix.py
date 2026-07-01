"""Tests for the offline-lane β3 workflow helpers (task 1954).

Covers the two pure helpers this task adds to ``orchestrator.workflow``:

* ``compute_failing_test_set_fingerprint`` — dedup key over the CONFIRMED
  failing-test SET (never ``main_sha``; see module docstring for the
  per-advance flood hazard this avoids). Modeled on
  ``compute_preexisting_main_break_fingerprint``
  (test_workflow_main_health_routing.py's fingerprint tests).
* ``build_offline_lane_fix_task_arguments`` — the ``submit_task`` argument
  block for a filed offline-lane fix task. Modeled on
  ``_spawn_main_health_fix_task``'s argument block, but routed through the
  standard merge gate (``merge_lane='normal'``, ``status='pending'``).

See ``test_offline_lane.py`` for the ``OfflineLaneWorker._handle_red_run``
behavioral tests that consume these helpers.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# compute_failing_test_set_fingerprint (step-3/4)
# ---------------------------------------------------------------------------


class TestComputeFailingTestSetFingerprint:
    """compute_failing_test_set_fingerprint must delegate to
    compute_content_fingerprint('offline_lane_red', '', sorted(ids)), be
    order-independent, differ across distinct sets, and fail-safe to ''."""

    def test_matches_inline_composition(self) -> None:
        from escalation.dedupe import compute_content_fingerprint

        from orchestrator.workflow import compute_failing_test_set_fingerprint

        ids = ['tests::test_b', 'tests::test_a']

        expected = compute_content_fingerprint('offline_lane_red', '', sorted(ids))
        actual = compute_failing_test_set_fingerprint(ids)
        assert actual == expected, (
            f'Fingerprint mismatch: expected {expected!r}, got {actual!r}'
        )
        assert actual, 'Fingerprint must be non-empty'

    def test_order_independent(self) -> None:
        from orchestrator.workflow import compute_failing_test_set_fingerprint

        fp_ab = compute_failing_test_set_fingerprint(['a', 'b'])
        fp_ba = compute_failing_test_set_fingerprint(['b', 'a'])
        assert fp_ab == fp_ba, 'Fingerprint must be order-independent over the test-ID set'

    def test_different_sets_produce_different_fingerprints(self) -> None:
        from orchestrator.workflow import compute_failing_test_set_fingerprint

        fp1 = compute_failing_test_set_fingerprint(['tests::test_a', 'tests::test_b'])
        fp2 = compute_failing_test_set_fingerprint(['tests::test_a', 'tests::test_c'])
        assert fp1 != fp2, 'A different failing-test SET must produce a different fingerprint'
        assert fp1 and fp2, 'Both fingerprints must be non-empty'

    def test_returns_empty_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When compute_content_fingerprint raises, helper returns '' (fail-safe)."""
        import escalation.dedupe as dedupe_mod
        monkeypatch.setattr(
            dedupe_mod, 'compute_content_fingerprint',
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('injected')),
        )
        from orchestrator.workflow import compute_failing_test_set_fingerprint
        result = compute_failing_test_set_fingerprint(['tests::test_a'])
        assert result == '', f'Expected empty string on failure; got {result!r}'


# ---------------------------------------------------------------------------
# build_offline_lane_fix_task_arguments (step-5/6)
# ---------------------------------------------------------------------------


class TestBuildOfflineLaneFixTaskArguments:
    """build_offline_lane_fix_task_arguments must build a submit_task argument
    block that routes THROUGH the standard gate (status='pending',
    merge_lane='normal') — never the B3 red-main fix-forward path — carrying
    the failing-test IDs + suspect range + fingerprint in metadata."""

    def test_builds_expected_argument_block(self, tmp_path) -> None:
        from orchestrator.workflow import build_offline_lane_fix_task_arguments

        failing_test_ids = ['tests::test_b', 'tests::test_a']
        suspect_range = 'GREEN..HEAD1'
        fingerprint = 'fp-abc-123'
        head = 'HEAD1'

        arguments = build_offline_lane_fix_task_arguments(
            failing_test_ids, suspect_range, fingerprint, tmp_path, head,
        )

        assert arguments['status'] == 'pending'
        assert arguments['priority'] == 'high'
        assert arguments['project_root'] == str(tmp_path)

        metadata = arguments['metadata']
        assert metadata['spawn_context'] == 'offline_lane_red'
        assert metadata['offline_lane_fingerprint'] == fingerprint
        assert metadata['failing_tests'] == sorted(failing_test_ids)
        assert metadata['suspect_ranges'] == [suspect_range]
        assert metadata['merge_lane'] == 'normal'

        for test_id in failing_test_ids:
            assert test_id in arguments['title'] + arguments['description'], (
                f'{test_id!r} must be mentioned in the title or description'
            )
        assert suspect_range in arguments['title'] + arguments['description'], (
            'suspect_range must be mentioned in the title or description'
        )
