"""δ B+H boundary gate — shared-contract half.

This module is the shared-side half of the δ end-to-end liveness boundary gate
(PRD plans/agent-liveness-telemetry-resume-prd.md §8, Appendix B).  It locks the
cli_invoke ↔ orchestrator/workflow seam contract from the shared package's perspective:

  - Real-transcript classification via count_transcript_turns + predicates (B1/B2).
  - Appendix-A mutual-exclusivity invariant (exactly one predicate True when
    timed_out and transcript_turns is not None).
  - B4 cap-retry guard-gating (is_zero_output_timeout-keyed ~822 guard dormant for
    progress; successful resume preserves resume_session_id).
  - B7 predicate legacy fallback when transcript is unreadable (None).

See orchestrator/tests/test_liveness_boundary_gate.py for the orchestrator-consumer
half (B1/B2/B3/B5/B6/B7 consumer assertions).

(shared/tests/conftest.py already puts shared/src on sys.path.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.cli_invoke import (
    AgentResult,
    count_transcript_turns,
    invoke_with_cap_retry,
    is_timed_out_with_progress,
    is_zero_output_timeout,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _write_transcript(
    config_dir: Path,
    session_id: str,
    *,
    n_assistant: int,
    last_tool: str | None = None,
    cwd_slug: str = 'proj',
) -> Path:
    """Write a real nested-schema JSONL transcript and return its path.

    Creates ``<config_dir>/projects/<cwd_slug>/<session_id>.jsonl`` (mkdir parents).

    Schema: the REAL nested Claude CLI transcript format —

      assistant records have NO top-level 'content' key; the content-block list
      lives at rec['message']['content'].

    Layout:
    - One leading ``{"type":"user", ...}`` record so 0-assistant cases still have
      a non-empty file (count_transcript_turns skips type!="assistant").
    - ``n_assistant`` assistant records.  When ``last_tool`` is set, the **last**
      assistant record carries a ``{"type":"tool_use","name":last_tool}`` block in
      its ``message.content`` list; all others carry a plain text block.

    .. warning::
        This helper is duplicated verbatim in
        ``orchestrator/tests/test_liveness_boundary_gate.py``.  Both copies MUST
        stay byte-identical — divergence would silently weaken one half of the
        cli_invoke ↔ orchestrator seam contract.  Extraction into a shared fixture
        module was deferred (cross-package conftest complexity; see review comment).
    """
    transcript_dir = config_dir / 'projects' / cwd_slug
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f'{session_id}.jsonl'

    records: list[dict] = [
        {
            'type': 'user',
            'message': {
                'role': 'user',
                'content': [{'type': 'text', 'text': 'Start task.'}],
            },
            'uuid': 'u-0',
            'parentUuid': None,
        },
    ]

    for i in range(n_assistant):
        is_last = (i == n_assistant - 1)
        if is_last and last_tool is not None:
            content = [{'type': 'tool_use', 'name': last_tool, 'id': f'tu-{i}', 'input': {}}]
        else:
            content = [{'type': 'text', 'text': f'Step {i}.'}]
        records.append({
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': content},
            'uuid': f'a-{i}',
            'parentUuid': f'a-{i - 1}' if i > 0 else 'u-0',
        })

    transcript_path.write_text('\n'.join(json.dumps(r) for r in records) + '\n')
    return transcript_path


# ---------------------------------------------------------------------------
# Step-1: B1/B2 — classification from a REAL transcript
# ---------------------------------------------------------------------------


class TestBoundaryClassificationFromTranscript:
    """Shared half of B1/B2: real-transcript reader + predicate classification.

    Drives count_transcript_turns (the real glob-by-session-id reader) and the
    real predicates against on-disk JSONL transcripts — the genuine seam artifact.
    """

    def test_b1_progress_real_transcript(self, tmp_path: Path) -> None:
        """B1: 3 assistant turns → is_timed_out_with_progress True, is_zero_output_timeout False."""
        cfg = tmp_path / 'cfg'
        cfg.mkdir()
        sid = 'session-b1-progress'

        _write_transcript(cfg, sid, n_assistant=3, last_tool='Bash')
        turns = count_transcript_turns(cfg, sid)
        assert turns == 3, f'Expected 3 assistant turns; got {turns!r}'

        r = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=1_200_000,
            transcript_turns=turns,
        )
        assert is_timed_out_with_progress(r) is True, (
            'Expected is_timed_out_with_progress=True for transcript_turns=3'
        )
        assert is_zero_output_timeout(r) is False, (
            'Expected is_zero_output_timeout=False for transcript_turns=3'
        )

    def test_b2_wedge_real_transcript(self, tmp_path: Path) -> None:
        """B2: 0 assistant turns → is_zero_output_timeout True, is_timed_out_with_progress False."""
        cfg = tmp_path / 'cfg'
        cfg.mkdir()
        sid = 'session-b2-wedge'

        _write_transcript(cfg, sid, n_assistant=0)
        turns = count_transcript_turns(cfg, sid)
        assert turns == 0, f'Expected 0 assistant turns; got {turns!r}'

        r = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=1_200_000,
            transcript_turns=turns,
        )
        assert is_zero_output_timeout(r) is True, (
            'Expected is_zero_output_timeout=True for transcript_turns=0'
        )
        assert is_timed_out_with_progress(r) is False, (
            'Expected is_timed_out_with_progress=False for transcript_turns=0'
        )


# ---------------------------------------------------------------------------
# Step-2: Appendix-A mutual-exclusivity invariant
# ---------------------------------------------------------------------------


class TestPredicateMutualExclusivity:
    """Appendix-A invariant: exactly one predicate holds when timed_out and turns known."""

    @pytest.mark.parametrize('turns', [0, 1, 5])
    def test_exactly_one_holds_when_timed_out_and_turns_known(self, turns: int) -> None:
        """Exactly one of {is_zero_output_timeout, is_timed_out_with_progress} is True."""
        r = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=1_200_000,
            transcript_turns=turns,
        )
        zero = is_zero_output_timeout(r)
        prog = is_timed_out_with_progress(r)

        assert zero != prog, (
            f'Expected exactly one predicate True for transcript_turns={turns}; '
            f'got is_zero_output_timeout={zero}, is_timed_out_with_progress={prog}'
        )
        if turns == 0:
            assert zero is True, f'Expected zero_output True for turns=0; got {zero!r}'
        else:
            assert prog is True, f'Expected progress True for turns={turns}; got {prog!r}'

    def test_both_false_when_not_timed_out_with_turns(self) -> None:
        """When timed_out=False, both predicates are False regardless of transcript_turns."""
        for turns in (5, 0):
            r = AgentResult(
                success=True,
                output='done',
                timed_out=False,
                turns=3,
                cost_usd=0.25,
                duration_ms=5_000,
                transcript_turns=turns,
            )
            assert is_zero_output_timeout(r) is False, (
                f'Expected is_zero_output_timeout=False when timed_out=False (turns={turns})'
            )
            assert is_timed_out_with_progress(r) is False, (
                f'Expected is_timed_out_with_progress=False when timed_out=False (turns={turns})'
            )


# ---------------------------------------------------------------------------
# Step-3: B4 — cap-retry resume-clear guard gated for progress
# ---------------------------------------------------------------------------


class TestB4CapRetryGuardGated:
    """B4 (PRD §3/§5/Appendix A): the is_zero_output_timeout-keyed guard at
    cli_invoke.py:~822 is DORMANT for a progress result.

    Scope clarification (design_decisions[2]): this class verifies ONLY the ~822
    guard.  The generic 'resume failed → fresh' fallback at ~947 (fires for any
    non-success resume) is a separate mechanism explicitly OUT of B4's scope.
    """

    def test_zero_output_guard_dormant_for_progress(self) -> None:
        """is_zero_output_timeout(progress) is False → ~822 guard condition is False → dormant.

        Asserts ONLY the narrow is_zero_output_timeout-keyed guard (~line 822):
          ``if is_zero_output_timeout(result) and resume_session_id``
        Because is_zero_output_timeout(progress) is False, the whole compound is
        False and _reset_for_fresh_retry at that site does NOT fire.

        NOTE: the generic fallback at ~947 (``not result.success and resume_session_id``)
        is a separate mechanism out of B4's scope — a progress timeout has success=False
        so that guard CAN fire when the loop is entered with a progress result directly;
        B4 does not claim it never fires, only that the ~822 site is gated by construction.
        """
        progress_result = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=1_200_000,
            transcript_turns=5,
        )
        # The is_zero_output_timeout-keyed guard at ~822 keys on this:
        assert is_zero_output_timeout(progress_result) is False, (
            'is_zero_output_timeout(progress) must be False — if True, ~822 would fire '
            'and clear resume_session_id, violating B4'
        )
        # Therefore: ``is_zero_output_timeout(result) and resume_session_id`` is False.
        # _reset_for_fresh_retry is NOT called at ~822 for a progress result.

    def test_resume_session_id_preserved_on_successful_resume(self, tmp_path: Path) -> None:
        """A successful resume: neither ~822 nor ~947 guard fires; resume_session_id flows through.

        Drives invoke_with_cap_retry with a mock gate, resume_session_id='killed-sid',
        and an invoke_fn that returns success=True on the first call.  Asserts that:
          - _reset_for_fresh_retry is NOT called (neither guard fires on the success path).
          - The invoke_fn received resume_session_id='killed-sid' (session preserved).
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        # Build a minimal mock gate (no real accounts needed — success on first call)
        gate = MagicMock()
        gate.account_count = 1
        gate.before_invoke = AsyncMock(return_value='tok')
        gate.detect_cap_hit = MagicMock(return_value=False)
        gate.active_account_name = 'acct'
        gate.on_agent_complete = MagicMock()
        gate.confirm_account_ok = MagicMock()
        gate.release_probe_slot = MagicMock()
        gate.soonest_resets_at = None
        gate._handle_auth_failure = MagicMock(return_value=False)

        # Wire gate.invoke_slot() as an async context manager yielding a slot proxy.
        def _make_slot_cm():
            async def _aenter(*_a, **_kw):
                token = await gate.before_invoke()
                slot = MagicMock()
                slot.token = token
                slot.account_name = gate.active_account_name
                slot._settled = False

                def _detect(stderr, output, backend='claude'):
                    hit = gate.detect_cap_hit(stderr, output, backend, oauth_token=slot.token)
                    if hit:
                        slot._settled = True
                    return hit

                def _confirm(cost_usd=0.0):
                    gate.confirm_account_ok(slot.token)
                    gate.on_agent_complete(cost_usd)
                    slot._settled = True

                def _settle():
                    slot._settled = True

                slot.detect_cap_hit = _detect
                slot.confirm = _confirm
                slot.settle = _settle
                return slot

            async def _aexit(slot, *_args):
                if not slot._settled:
                    gate.release_probe_slot(slot.token)

            cm = MagicMock()
            cm.__aenter__ = _aenter
            cm.__aexit__ = _aexit
            return cm

        gate.invoke_slot = MagicMock(side_effect=lambda: _make_slot_cm())

        # invoke_fn returns success on the first call
        success_result = AgentResult(
            success=True,
            output='Done!',
            timed_out=False,
            turns=3,
            cost_usd=0.25,
            duration_ms=5_000,
            session_id='killed-sid',
        )
        mock_invoke_fn = AsyncMock(return_value=success_result)

        import asyncio

        with patch('shared.cli_invoke._reset_for_fresh_retry') as mock_reset:
            result = asyncio.run(
                invoke_with_cap_retry(
                    gate,
                    'test-label',
                    invoke_fn=mock_invoke_fn,
                    prompt='test-prompt',
                    resume_session_id='killed-sid',
                    cap_wait_sanity_secs=10.0,
                )
            )

        # Neither ~822 nor ~947 fires on the success path.
        # assert_not_called() raises AssertionError with an informative message if violated.
        mock_reset.assert_not_called()
        # invoke_fn received resume_session_id='killed-sid'
        call_kwargs = mock_invoke_fn.call_args.kwargs
        assert call_kwargs.get('resume_session_id') == 'killed-sid', (
            f"Expected invoke_fn to receive resume_session_id='killed-sid'; "
            f"got {call_kwargs.get('resume_session_id')!r}"
        )
        assert result.success is True


# ---------------------------------------------------------------------------
# Step-4: B7 — predicate legacy fallback when transcript unreadable
# ---------------------------------------------------------------------------


class TestB7PredicateLegacyFallback:
    """B7: transcript_turns=None degrades to legacy heuristic (turns==0 and cost_usd==0.0).

    Proves the None case never silently upgrades a wedge to progress and always
    degrades to today's conservative behavior (PRD decision #3).
    """

    def test_none_degrades_to_legacy_wedge(self) -> None:
        """transcript_turns=None + legacy signals (turns=0, cost=0.0) → zero_output True."""
        r = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=0,
            cost_usd=0.0,
            duration_ms=1_200_000,
            transcript_turns=None,
        )
        assert is_zero_output_timeout(r) is True, (
            'Expected is_zero_output_timeout=True via legacy fallback (turns=0, cost=0.0)'
        )
        assert is_timed_out_with_progress(r) is False, (
            'Expected is_timed_out_with_progress=False when transcript_turns=None'
        )

    def test_none_legacy_not_wedge_when_work_signals(self) -> None:
        """transcript_turns=None + work signals (turns>0, cost>0) → zero_output False."""
        r = AgentResult(
            success=False,
            output='',
            timed_out=True,
            turns=2,
            cost_usd=0.10,
            duration_ms=1_200_000,
            transcript_turns=None,
        )
        assert is_zero_output_timeout(r) is False, (
            'Expected is_zero_output_timeout=False: legacy heuristic sees work (turns=2, cost=0.10)'
        )

    def test_not_timed_out_none_is_false(self) -> None:
        """When timed_out=False and transcript_turns=None, both predicates are False."""
        r = AgentResult(
            success=True,
            output='done',
            timed_out=False,
            turns=0,
            cost_usd=0.0,
            duration_ms=5_000,
            transcript_turns=None,
        )
        assert is_zero_output_timeout(r) is False, (
            'Expected is_zero_output_timeout=False when timed_out=False'
        )
        assert is_timed_out_with_progress(r) is False, (
            'Expected is_timed_out_with_progress=False when timed_out=False'
        )
