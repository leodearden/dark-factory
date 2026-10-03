"""The None-transcript storm escape — INV-4 ``storm-escape-required`` (task 4003).

``count_transcript_turns`` returns None when the transcript cannot be read, and
the liveness watchdog absorbs that None by design: it refuses to kill on it
("NEVER kill on None — conservative degrade"). That fail-soft was SILENT.

That is how a real defect ran for three weeks. From 2026-07-18 (task 2744) to
2026-08-11 (task 4003) every reconciliation stage had its ``CLAUDE_CONFIG_DIR``
outside the sandbox writable set, so no transcript was ever written; every read
returned None; the watchdog went inert. Nothing logged, nothing escalated.

``note_unreadable_transcript`` is the escape: an actionable WARNING for an
invocation that was configured WITH both ``config_dir`` and ``session_id`` —
i.e. a role that is SUPPOSED to have a transcript. It logs rather than kills,
deliberately: the conservative degrade stays, it just stops being silent.

SCOPE. This covers the WATCHDOG path. The cap-retry force-fresh is the other
consumer of the same unreadable transcript, and it is NOT routed through here:
it already emits its own WARNING at the point of decision ("capped session ...
has no transcript under ... — retrying FRESH"). It is a one-shot decision
rather than a poll loop, so it has no streak to latch and needs no escape.

The gate is WALL-CLOCK, not a poll count: it fires at the caller's
``startup_grace_secs``, the budget already defined as "how long before we may
conclude something is wrong". A poll count would mean two different durations in
the watchdog's two regimes, and ~15s in the startup regime — inside the MCP-init
window a healthy stage routinely spends before its first record lands.
"""

from __future__ import annotations

import logging

import pytest

from shared.cli_invoke import note_unreadable_transcript

_GRACE = 120.0


def _note(elapsed, *, grace = _GRACE, session_id = 'sid'):
    return note_unreadable_transcript(
        elapsed,
        grace_secs=grace,
        config_dir='/tmp/cfg-x',
        session_id=session_id,
        label='Reconciliation stage (test)',
    )


def test_inside_grace_is_silent(caplog):
    """An unreadable transcript inside the grace window neither fires nor logs.

    A transcript that does not exist yet is NORMAL: the file appears only on the
    CLI's first write, and a recon stage spawns with fused-memory + escalation
    MCP servers to initialise first. Firing during that window would emit the
    WARNING once on every healthy invocation, which is how a warning gets tuned
    out — the precise failure mode this escape exists to avoid.
    """
    with caplog.at_level(logging.WARNING, logger='shared.cli_invoke'):
        for elapsed in (0.0, 1.0, _GRACE / 2, _GRACE - 0.001):
            assert _note(elapsed) is False, (
                f'elapsed {elapsed} < grace {_GRACE} must not fire'
            )

    assert not caplog.records, (
        f'no WARNING may be emitted inside grace; got {[r.message for r in caplog.records]}'
    )


def test_at_grace_fires_one_actionable_warning(caplog):
    """Reaching the grace bound fires exactly one actionable WARNING.

    The bound is inclusive: it is the same instant at which the watchdog would
    have killed on an explicit 0-turn read, so an unreadable transcript there is
    a defect rather than patience.
    """
    with caplog.at_level(logging.WARNING, logger='shared.cli_invoke'):
        fired = note_unreadable_transcript(
            _GRACE,
            grace_secs=_GRACE,
            config_dir='/tmp/cfg-x',
            session_id='sid-abc',
            label='Reconciliation stage (test)',
        )

    assert fired is True, f'elapsed == grace ({_GRACE}) must fire'

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, (
        f'exactly one WARNING per call past the bound; got {[r.message for r in warnings]}'
    )
    msg = warnings[0].getMessage()
    # The escape must be actionable without a code read: which config dir,
    # which session, which invocation, and how long it has been wrong.
    for needle in ('/tmp/cfg-x', 'sid-abc', 'Reconciliation stage (test)', '120.0'):
        assert needle in msg, f'WARNING must name {needle!r}; got {msg!r}'


def test_past_grace_fires_because_the_latch_is_the_callers(caplog):
    """The helper is STATELESS — every call past the bound fires.

    This is a contract, not an oversight: the once-per-crossing latch lives in
    the caller (``_run_subprocess``'s ``unreadable_escape_fired``) so that two
    concurrent invocations in one process cannot silence each other, which a
    module-global latch would allow. The wiring half
    (``test_cli_invoke.py::TestUnreadableTranscriptEscapeWiring``) pins that the
    real caller does latch, so a wedged run emits ONE record and not hundreds.
    """
    with caplog.at_level(logging.WARNING, logger='shared.cli_invoke'):
        for elapsed in (_GRACE, _GRACE + 1.0, _GRACE * 10):
            assert _note(elapsed) is True, f'elapsed {elapsed} >= grace must fire'

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 3, (
        f'the helper itself does not latch — every call past the bound logs; '
        f'got {[r.message for r in warnings]}'
    )


@pytest.mark.parametrize('grace', [0.0, 5.0, 600.0])
def test_bound_is_the_callers_grace_not_a_constant_of_its_own(grace, caplog):
    """The escape has no threshold of its own — it tracks whatever grace it is given.

    A per-role ``startup_grace_secs`` is the whole point: a role with a long
    grace is one we are willing to wait longer for, and the escape must inherit
    that patience rather than second-guess it from a poll count that means a
    different duration in each watchdog regime.
    """
    with caplog.at_level(logging.WARNING, logger='shared.cli_invoke'):
        just_under = _note(max(grace - 0.001, 0.0) if grace else -1.0, grace=grace)
        at_bound = _note(grace, grace=grace)

    assert just_under is False, f'below grace={grace} must not fire'
    assert at_bound is True, f'at grace={grace} must fire'
