"""The None-transcript storm escape — INV-4 ``storm-escape-required`` (task 4003).

``count_transcript_turns`` returns None when the transcript cannot be read, and
both consumers absorb that None by design: the liveness watchdog refuses to kill
on it ("NEVER kill on None — conservative degrade") and the cap-retry veto
force-freshes instead of resuming.  Both fail-softs were COUNTERLESS.

That is how a real defect ran silently for three weeks.  From 2026-07-18 (task
2744) to 2026-08-11 (task 4003) every reconciliation stage had its
``CLAUDE_CONFIG_DIR`` outside the sandbox writable set, so no transcript was
ever written; every read returned None; the watchdog went inert and every
cap-retry force-freshed.  Nothing logged, nothing counted, nothing escalated.

``note_unreadable_transcript`` is the escape: a counted, once-per-crossing
WARNING for an invocation that was configured WITH both ``config_dir`` and
``session_id`` — i.e. a role that is SUPPOSED to have a transcript.  It counts
rather than kills, deliberately: the conservative degrade stays, it just stops
being silent.
"""

from __future__ import annotations

import logging

import pytest

from shared.cli_invoke import (
    _WATCHDOG_UNREADABLE_STREAK_THRESHOLD,
    get_unreadable_transcript_escapes,
    note_unreadable_transcript,
    reset_unreadable_transcript_escapes,
)

_THRESHOLD = _WATCHDOG_UNREADABLE_STREAK_THRESHOLD


@pytest.fixture(autouse=True)
def _reset_counter():
    """The counter is process-wide, so every test must start from a known zero."""
    reset_unreadable_transcript_escapes()
    yield
    reset_unreadable_transcript_escapes()


def _note(streak: int, *, session_id: str = 'sid') -> bool:
    return note_unreadable_transcript(
        streak,
        config_dir='/tmp/cfg-x',
        session_id=session_id,
        label='Reconciliation stage (test)',
    )


def test_below_threshold_is_silent(caplog):
    """Streaks below the threshold neither fire nor log.

    A transcript that is briefly unreadable is normal — the file does not exist
    until the CLI's first write. Firing on streak 1 would make this a log storm
    on every healthy invocation, which is how a warning gets tuned out.
    """
    with caplog.at_level(logging.WARNING, logger='shared.cli_invoke'):
        for streak in range(1, _THRESHOLD):
            assert _note(streak) is False, (
                f'streak {streak} < threshold {_THRESHOLD} must not fire'
            )

    assert get_unreadable_transcript_escapes() == 0, (
        f'counter must stay 0 below threshold; got {get_unreadable_transcript_escapes()}'
    )
    assert not caplog.records, (
        f'no WARNING may be emitted below threshold; got {[r.message for r in caplog.records]}'
    )


def test_crossing_threshold_fires_once(caplog):
    """Crossing the threshold fires exactly one actionable WARNING."""
    with caplog.at_level(logging.WARNING, logger='shared.cli_invoke'):
        fired = note_unreadable_transcript(
            _THRESHOLD,
            config_dir='/tmp/cfg-x',
            session_id='sid-abc',
            label='Reconciliation stage (test)',
        )

    assert fired is True, f'streak == threshold ({_THRESHOLD}) must fire'
    assert get_unreadable_transcript_escapes() == 1, (
        f'counter must be 1 after one crossing; got {get_unreadable_transcript_escapes()}'
    )

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f'exactly one WARNING per crossing; got {[r.message for r in warnings]}'
    )
    msg = warnings[0].getMessage()
    # The escape must be actionable without a code read: which config dir,
    # which session, which invocation.
    for needle in ('/tmp/cfg-x', 'sid-abc', 'Reconciliation stage (test)'):
        assert needle in msg, f'WARNING must name {needle!r}; got {msg!r}'


def test_above_threshold_does_not_re_fire(caplog):
    """One fire per CROSSING, not per poll — this is what keeps it out of storm class.

    A wedged invocation polls its transcript for the whole of a long run. Firing
    on every poll above the threshold would emit hundreds of identical records
    and bury the signal; that is the failure mode this helper exists to avoid,
    so re-firing would be self-defeating.
    """
    assert _note(_THRESHOLD) is True
    assert get_unreadable_transcript_escapes() == 1

    with caplog.at_level(logging.WARNING, logger='shared.cli_invoke'):
        for streak in range(_THRESHOLD + 1, _THRESHOLD + 6):
            assert _note(streak) is False, (
                f'streak {streak} > threshold must not re-fire'
            )

    assert get_unreadable_transcript_escapes() == 1, (
        f'counter must stay at 1 above threshold; got {get_unreadable_transcript_escapes()}'
    )
    assert not caplog.records, (
        f'no further WARNING above threshold; got {[r.message for r in caplog.records]}'
    )

    # A streak that resets to 0 and climbs again IS a new crossing, and fires.
    assert _note(0) is False, 'a reset streak must not fire'
    assert _note(_THRESHOLD) is True, 'a second crossing must fire again'
    assert get_unreadable_transcript_escapes() == 2, (
        f'a second crossing must count; got {get_unreadable_transcript_escapes()}'
    )


def test_counter_is_process_wide_and_resettable():
    """The counter aggregates across invocations and resets cleanly for tests."""
    assert _note(_THRESHOLD, session_id='sid-1') is True
    assert _note(_THRESHOLD, session_id='sid-2') is True
    assert get_unreadable_transcript_escapes() == 2, (
        f'two crossings on distinct sessions must accumulate; '
        f'got {get_unreadable_transcript_escapes()}'
    )

    reset_unreadable_transcript_escapes()
    assert get_unreadable_transcript_escapes() == 0, (
        f'reset must return the counter to 0; got {get_unreadable_transcript_escapes()}'
    )


def test_threshold_constant_is_small_and_positive():
    """The threshold must be a small positive int — a guard against a silent retune.

    A future edit setting this to 0 (fires on every healthy poll) or to something
    huge (never fires within a real invocation's lifetime) would restore the
    silence this task removed, in one line and without touching a test.
    """
    assert isinstance(_WATCHDOG_UNREADABLE_STREAK_THRESHOLD, int), (
        f'threshold must be an int; got {type(_WATCHDOG_UNREADABLE_STREAK_THRESHOLD)}'
    )
    assert 1 <= _WATCHDOG_UNREADABLE_STREAK_THRESHOLD <= 10, (
        f'threshold must be in [1, 10] to fire within a real invocation without '
        f'storming; got {_WATCHDOG_UNREADABLE_STREAK_THRESHOLD}'
    )
