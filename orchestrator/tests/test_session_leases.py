"""Tests for orchestrator.session_leases: role leases (Attention Rail T7).

The lease-* CLI verbs are covered in test_session_registry.py.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _session_record_fixtures import make_record, set_mtime

from orchestrator import session_identity, session_leases, session_records

# A pid virtually guaranteed dead on any host (matches the convention in
# shared/tests/test_proc_liveness.py: 2**31 - 1).
_DEAD_PID = 2**31 - 1

_NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Step-10: role leases (Attention Rail T7)
# ---------------------------------------------------------------------------

# --- paths / names / contract types -----------------------------------------


def test_leases_dir_is_fleet_root_slash_leases(tmp_path: Path) -> None:
    assert session_leases.leases_dir(root=tmp_path) == tmp_path / 'leases'


def test_lease_path_for_name_joins_leases_dir_and_lease_suffix(tmp_path: Path) -> None:
    path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    assert path == tmp_path / 'leases' / 'watcher-df.lease'


def test_lease_path_for_name_preserves_hash_for_task_scoped_names(tmp_path: Path) -> None:
    path = session_leases.lease_path_for_name('unblock-df#2085', root=tmp_path)
    assert path == tmp_path / 'leases' / 'unblock-df#2085.lease'


def test_lease_path_for_name_sanitizes_path_separators(tmp_path: Path) -> None:
    # A name containing '/' must not escape leases_dir: it still resolves to
    # a single file directly inside leases_dir (no nested directory, no
    # traversal), unlike _SLUG_SANITIZE_RE this sanitizer PRESERVES '#'.
    path = session_leases.lease_path_for_name('../../etc/passwd', root=tmp_path)
    assert path.parent == session_leases.leases_dir(root=tmp_path)
    assert '/' not in path.name


def test_build_lease_name_watcher() -> None:
    assert session_leases.build_lease_name('watcher', 'df') == 'watcher-df'


def test_build_lease_name_recon_watcher() -> None:
    assert session_leases.build_lease_name('recon-watcher', 'df') == 'recon-watcher-df'


def test_build_lease_name_unblock_is_task_scoped_with_hash() -> None:
    assert session_leases.build_lease_name('unblock', 'df', '2085') == 'unblock-df#2085'


def test_lease_policy_has_exactly_stand_down_and_warn_and_proceed() -> None:
    values = {member.value for member in session_leases.LeasePolicy}
    assert values == {'stand-down', 'warn-and-proceed'}
    assert session_leases.LeasePolicy.STAND_DOWN.value == 'stand-down'
    assert session_leases.LeasePolicy.WARN_AND_PROCEED.value == 'warn-and-proceed'


def test_lease_decision_has_exactly_acquired_stand_down_proceed() -> None:
    values = {member.value for member in session_leases.LeaseDecision}
    assert values == {'acquired', 'stand-down', 'proceed'}
    assert session_leases.LeaseDecision.ACQUIRED.value == 'acquired'
    assert session_leases.LeaseDecision.STAND_DOWN.value == 'stand-down'
    assert session_leases.LeaseDecision.PROCEED.value == 'proceed'


def test_lease_mutation_has_exactly_applied_forced_absent_refused_faulted() -> None:
    values = {member.value for member in session_leases.LeaseMutation}
    assert values == {'applied', 'forced', 'absent', 'refused', 'faulted'}
    assert session_leases.LeaseMutation.APPLIED.value == 'applied'
    assert session_leases.LeaseMutation.FORCED.value == 'forced'
    assert session_leases.LeaseMutation.ABSENT.value == 'absent'
    assert session_leases.LeaseMutation.REFUSED.value == 'refused'
    assert session_leases.LeaseMutation.FAULTED.value == 'faulted'


def test_lease_mutation_is_a_str_enum_like_its_siblings() -> None:
    # Mirrors LeasePolicy/LeaseDecision: the CLI prints `.value` directly and
    # a StrEnum member compares equal to its own string form.
    assert issubclass(session_leases.LeaseMutation, str)
    assert session_leases.LeaseMutation.REFUSED == 'refused'


def test_lease_heartbeat_ttl_is_a_timedelta() -> None:
    assert isinstance(session_leases.LEASE_HEARTBEAT_TTL, timedelta)


def test_lease_holder_json_round_trip_is_lossless() -> None:
    holder = session_leases.LeaseHolder(
        session_slug='watcher-df-100', pid=4242, start_ts='2026-07-07T12:00:00+00:00'
    )
    assert session_leases.LeaseHolder.from_json(holder.to_json()) == holder


def test_lease_holder_dict_round_trip_is_lossless() -> None:
    holder = session_leases.LeaseHolder(
        session_slug='watcher-df-100', pid=4242, start_ts='2026-07-07T12:00:00+00:00'
    )
    assert session_leases.LeaseHolder.from_dict(holder.to_dict()) == holder


# --- _render_contention_message (task 3994 defect 3) ----------------------
#
# The old form was `lease held by <slug> ({alive|dead}, heartbeat <n>s ago)`,
# which reads as SELF-CONTRADICTORY on its most important branch: "dead,
# heartbeat 42s ago" invites the reader to conclude the holder is gone and
# the lease is reclaimable, when a fresh heartbeat means precisely the
# opposite. That exact string misled the 2026-08-08 rotation into
# force-releasing a live holder's lease. Liveness and freshness are two
# INDEPENDENT axes and each must now name itself.

_LIVE_HOLDER = session_leases.LeaseHolder(
    session_slug='watcher-df-100', pid=4242, start_ts='2026-07-07T12:00:00+00:00'
)


def test_contention_message_live_holder_names_the_slug_and_the_pid() -> None:
    msg = session_leases._render_contention_message(
        _LIVE_HOLDER, holder_alive=True, age_secs=42.0, policy=session_leases.LeasePolicy.STAND_DOWN
    )

    assert 'watcher-df-100' in msg
    assert '4242' in msg
    assert 'pid 4242 alive' in msg
    assert msg.endswith('— standing down')


def test_contention_message_dead_pid_but_fresh_heartbeat_says_not_reclaimable() -> None:
    ttl = timedelta(seconds=7200)
    msg = session_leases._render_contention_message(
        _LIVE_HOLDER,
        holder_alive=False,
        age_secs=1200.0,
        policy=session_leases.LeasePolicy.STAND_DOWN,
        ttl=ttl,
    )

    # The withdrawn string, pinned as an ANTI-assertion.
    assert '(dead, heartbeat' not in msg
    lowered = msg.lower()
    assert 'not running' in lowered  # the pid axis, named
    assert 'fresh' in lowered  # the heartbeat axis, named
    assert 'not reclaimable' in lowered  # the decision, explained
    assert '6000' in msg  # the remaining TTL, in seconds


def test_contention_message_remaining_ttl_tracks_the_injected_ttl() -> None:
    msg = session_leases._render_contention_message(
        _LIVE_HOLDER,
        holder_alive=False,
        age_secs=100.0,
        policy=session_leases.LeasePolicy.STAND_DOWN,
        ttl=timedelta(seconds=900),
    )
    assert '800' in msg


def test_contention_message_dead_pid_past_ttl_is_phrased_as_reclaim_eligible() -> None:
    msg = session_leases._render_contention_message(
        _LIVE_HOLDER,
        holder_alive=False,
        age_secs=9000.0,
        policy=session_leases.LeasePolicy.STAND_DOWN,
        ttl=timedelta(seconds=7200),
    )

    assert '(dead, heartbeat' not in msg
    lowered = msg.lower()
    assert 'not running' in lowered
    # Past the TTL the lease WAS reclaim-eligible; we are here only because
    # the reclaim race was lost to someone else, and the message must say so
    # rather than implying the holder is still healthy.
    assert 'reclaim' in lowered
    assert 'not reclaimable' not in lowered


def test_contention_message_corrupt_body_does_not_invent_a_holder() -> None:
    msg = session_leases._render_contention_message(
        None, holder_alive=False, age_secs=42.0, policy=session_leases.LeasePolicy.STAND_DOWN
    )

    assert 'unreadable' in msg.lower()
    assert '42s ago' in msg  # freshness is still reported
    assert msg.endswith('— standing down')


@pytest.mark.parametrize(
    ('policy', 'suffix'),
    [
        (session_leases.LeasePolicy.STAND_DOWN, '— standing down'),
        (session_leases.LeasePolicy.WARN_AND_PROCEED, '— proceeding anyway'),
    ],
)
@pytest.mark.parametrize(
    ('holder', 'alive', 'age'),
    [
        (_LIVE_HOLDER, True, 42.0),
        (_LIVE_HOLDER, False, 1200.0),
        (_LIVE_HOLDER, False, 9000.0),
        (None, False, 42.0),
    ],
    ids=['live', 'dead-fresh', 'dead-expired', 'corrupt'],
)
def test_contention_message_is_always_one_greppable_line(
    policy: session_leases.LeasePolicy,
    suffix: str,
    holder: session_leases.LeaseHolder | None,
    alive: bool,
    age: float,
) -> None:
    msg = session_leases._render_contention_message(holder, holder_alive=alive, age_secs=age, policy=policy)

    assert '\n' not in msg
    assert msg.endswith(suffix)


def test_claim_lease_dead_holder_within_ttl_reports_the_non_contradictory_message(
    tmp_path: Path,
) -> None:
    # End-to-end: the branch the 2026-08-08 rotation actually hit.
    dead = session_leases.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=dead, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    set_mtime(lease_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL - timedelta(minutes=10))

    contender = session_leases.LeaseHolder(session_slug='watcher-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease('watcher-df', holder=contender, root=tmp_path, now=_NOW)

    assert claim.decision == session_leases.LeaseDecision.STAND_DOWN
    assert '(dead, heartbeat' not in claim.message
    assert 'not reclaimable' in claim.message.lower()
    assert str(_DEAD_PID) in claim.message


# --- claim_lease: free lease --------------------------------------------


def test_claim_lease_acquires_a_free_lease(tmp_path: Path) -> None:
    holder = session_leases.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())

    claim = session_leases.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)

    assert claim.decision == session_leases.LeaseDecision.ACQUIRED
    assert claim.acquired is True
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-100'

    assert session_leases.leases_dir(root=tmp_path).is_dir()
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    assert lease_path.is_file()
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()) == holder


# --- claim_lease: held by a live holder ----------------------------------


def test_claim_lease_held_by_live_holder_stand_down_policy(tmp_path: Path) -> None:
    original = session_leases.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=original, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    set_mtime(lease_path, _NOW, timedelta(seconds=42))

    contender = session_leases.LeaseHolder(session_slug='watcher-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease(
        'watcher-df', holder=contender, policy=session_leases.LeasePolicy.STAND_DOWN, root=tmp_path, now=_NOW
    )

    assert claim.acquired is False
    assert claim.decision == session_leases.LeaseDecision.STAND_DOWN
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-100'
    assert claim.holder_alive is True
    assert claim.heartbeat_age_secs == 42
    assert claim.message == (
        f'lease held by watcher-df-100 (pid {os.getpid()} alive, heartbeat 42s ago) '
        '— standing down'
    )
    # No clobber: the on-disk body still names the ORIGINAL holder.
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()) == original


def test_claim_lease_held_by_live_holder_warn_and_proceed_policy(tmp_path: Path) -> None:
    original = session_leases.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    session_leases.claim_lease('unblock-df#2085', holder=original, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('unblock-df#2085', root=tmp_path)
    set_mtime(lease_path, _NOW, timedelta(seconds=42))

    contender = session_leases.LeaseHolder(session_slug='unblock-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease(
        'unblock-df#2085',
        holder=contender,
        policy=session_leases.LeasePolicy.WARN_AND_PROCEED,
        root=tmp_path,
        now=_NOW,
    )

    assert claim.acquired is False
    assert claim.decision == session_leases.LeaseDecision.PROCEED
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-100'
    assert 'lease held by watcher-df-100' in claim.message
    assert 'proceeding' in claim.message
    # No clobber: the on-disk body still names the ORIGINAL holder.
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()) == original


# --- claim_lease: stale-lease reap-and-reclaim + the AND boundary --------


def test_claim_lease_reaps_and_reclaims_a_stale_lease(tmp_path: Path) -> None:
    stale_holder = session_leases.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=stale_holder, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    set_mtime(lease_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    new_holder = session_leases.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is True
    assert claim.decision == session_leases.LeaseDecision.ACQUIRED
    assert claim.holder == new_holder
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()) == new_holder


def test_claim_lease_reap_and_reclaim_emits_structured_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # THREAD 1 hardened detection (task 2796): a reap-and-reclaim displaces
    # whoever currently held the lease. That holder is SUPPOSED to be dead,
    # but if the TTL is ever mis-tuned again this same path would silently
    # steal a live-but-quiet watcher's lease. Make every reap-and-reclaim
    # loud/greppable so any residual displacement of a supposedly-live holder
    # is surfaced rather than silent.
    stale_holder = session_leases.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=stale_holder, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    # Backdate past the (new) TTL so the lease IS reap-eligible.
    age = session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1)
    set_mtime(lease_path, _NOW, age)

    new_holder = session_leases.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    with caplog.at_level(logging.WARNING, logger=session_leases.logger.name):
        claim = session_leases.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    # Sanity: the reclaim actually happened (the WARNING is on that path).
    assert claim.acquired is True

    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and r.name == session_leases.logger.name
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    # Names the displaced holder (slug + pid), the observed heartbeat age, and
    # the new holder's slug.
    assert 'watcher-df-dead' in message
    assert str(_DEAD_PID) in message
    assert f'{age.total_seconds():.0f}' in message
    assert 'watcher-df-new' in message


def test_claim_lease_does_not_reap_a_dead_holder_within_ttl(tmp_path: Path) -> None:
    # Proves the reap rule is (age > TTL) AND (pid dead), not either alone:
    # a dead-pid holder with a still-fresh heartbeat is NOT reaped.
    stale_holder = session_leases.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=stale_holder, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    set_mtime(lease_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL - timedelta(minutes=1))

    new_holder = session_leases.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is False
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-dead'
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()) == stale_holder


def test_claim_lease_keeps_a_quiet_watchers_lease_across_a_full_slice(tmp_path: Path) -> None:
    # The required "duplicate-through" regression (task 2796 THREAD 1): a live
    # interactive watcher holds its lease with `--pid $$` (the ephemeral
    # Bash-tool shell, dead moments after the claim), so pid_alive is
    # ~always False for a watcher lease and staleness collapses to a pure
    # heartbeat-TTL check. That watcher heartbeats only once per Main Loop
    # cycle, bounded by ONE canonical watcher-rearm.sh `--timeout` slice
    # (3600s). If LEASE_HEARTBEAT_TTL is below that slice, a live-but-quiet
    # holder's lease goes stale->reap-eligible during any quiet slice and a
    # duplicate reaps+reclaims it -- the exact code path that let a
    # live-lease-holder's duplicate spawn through. With TTL raised above the
    # slice, the duplicate must instead STAND DOWN.
    original = session_leases.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=original, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    # Backdate the heartbeat by one full canonical wait slice (3600s).
    set_mtime(lease_path, _NOW, timedelta(seconds=3600))

    duplicate = session_leases.LeaseHolder(session_slug='watcher-df-dup', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease(
        'watcher-df', holder=duplicate, policy=session_leases.LeasePolicy.STAND_DOWN, root=tmp_path, now=_NOW
    )

    # RED today: TTL=300s < 3600s slice age, so the live-but-quiet holder's
    # lease is reaped-and-reclaimed and the duplicate ACQUIRES (claim.acquired
    # is True) -- the duplicate-through bug. GREEN after TTL is raised: the
    # duplicate stands down.
    assert claim.acquired is False
    assert claim.decision == session_leases.LeaseDecision.STAND_DOWN
    assert claim.holder is not None
    assert claim.holder.session_slug == 'watcher-df-dead'
    # No clobber / no reap: the on-disk body still names the ORIGINAL holder.
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()) == original

    # Invariant guard: the whole regression hinges on TTL exceeding one full
    # canonical slice. If a future edit lowers LEASE_HEARTBEAT_TTL back below
    # 3600s, this assertion fails loudly rather than silently re-opening the
    # gap.
    assert timedelta(seconds=3600) < session_leases.LEASE_HEARTBEAT_TTL


def test_claim_lease_survives_lease_vanishing_between_create_and_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates a concurrent release/reap landing in the narrow window
    # between our failed O_EXCL create and _read_lease_holder_state's
    # path.stat() -- this must reclaim the (now-free) lease rather than
    # propagate an uncaught FileNotFoundError.
    original = session_leases.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=original, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)

    real_stat = Path.stat

    def _vanished_stat(self: Path, *args: Any, **kwargs: Any) -> os.stat_result:
        if self == lease_path:
            raise FileNotFoundError('simulated concurrent release')
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'stat', _vanished_stat)

    new_holder = session_leases.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is True
    assert claim.decision == session_leases.LeaseDecision.ACQUIRED
    assert claim.holder == new_holder


def test_claim_lease_does_not_clobber_a_lease_reclaimed_between_stale_check_and_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulates a competitor's reap-and-reclaim landing in the gap between
    # claim_lease's first staleness read and its re-verify read immediately
    # before unlinking: the re-verify must see the competitor's fresh, live
    # holder and skip the unlink rather than clobbering it.
    stale_holder = session_leases.LeaseHolder(session_slug='watcher-df-dead', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=stale_holder, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    set_mtime(lease_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    competitor_holder = session_leases.LeaseHolder(
        session_slug='watcher-df-competitor', pid=os.getpid(), start_ts=_NOW.isoformat()
    )
    # First read (top of claim_lease) observes the true, stale state; the
    # re-verify read (immediately before unlinking) reports that a
    # competitor has already reaped-and-reclaimed it in the gap.
    responses = iter(
        [
            (stale_holder, False, session_leases.LEASE_HEARTBEAT_TTL.total_seconds() + 60),
            (competitor_holder, True, 1.0),
        ]
    )
    monkeypatch.setattr(session_leases, '_read_lease_holder_state', lambda path, *, now: next(responses))

    new_holder = session_leases.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is False
    assert claim.holder == competitor_holder
    # unlink() must never have been called: the on-disk body (still the
    # pre-mock stale_holder written at setup) is untouched, proving the
    # competitor's (simulated) fresh lease was not clobbered.
    assert session_leases.LeaseHolder.from_json(lease_path.read_text()) == stale_holder


# --- claim_lease: corrupt/unreadable lease body ---------------------------


def test_claim_lease_reaps_and_reclaims_a_corrupt_and_aged_lease(tmp_path: Path) -> None:
    corrupt_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text('{not valid json')
    set_mtime(corrupt_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    new_holder = session_leases.LeaseHolder(session_slug='watcher-df-new', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease('watcher-df', holder=new_holder, root=tmp_path, now=_NOW)

    assert claim.acquired is True
    assert claim.decision == session_leases.LeaseDecision.ACQUIRED
    assert claim.holder == new_holder
    assert session_leases.LeaseHolder.from_json(corrupt_path.read_text()) == new_holder


def test_claim_lease_contention_on_corrupt_body_within_ttl_reports_unknown_holder(
    tmp_path: Path,
) -> None:
    corrupt_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text('{not valid json')
    set_mtime(corrupt_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL - timedelta(minutes=1))

    contender = session_leases.LeaseHolder(session_slug='watcher-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())
    claim = session_leases.claim_lease(
        'watcher-df', holder=contender, policy=session_leases.LeasePolicy.STAND_DOWN, root=tmp_path, now=_NOW
    )

    assert claim.acquired is False
    assert claim.decision == session_leases.LeaseDecision.STAND_DOWN
    assert claim.holder is None
    assert claim.holder_alive is False
    assert '<unknown>' in claim.message
    assert 'standing down' in claim.message
    # Not reaped (within TTL): the corrupt body is left exactly as-is.
    assert corrupt_path.read_text() == '{not valid json'


# --- heartbeat_lease / release_lease --------------------------------------


def test_heartbeat_lease_advances_mtime(tmp_path: Path) -> None:
    holder = session_leases.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    set_mtime(lease_path, _NOW, timedelta(hours=1))
    old_mtime = lease_path.stat().st_mtime

    result = session_leases.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is session_leases.LeaseMutation.APPLIED
    assert lease_path.stat().st_mtime > old_mtime


def test_release_lease_removes_the_lease_file(tmp_path: Path) -> None:
    holder = session_leases.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)

    result = session_leases.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is session_leases.LeaseMutation.APPLIED
    assert not lease_path.exists()


def test_release_lease_is_idempotent_on_a_second_call(tmp_path: Path) -> None:
    holder = session_leases.LeaseHolder(session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat())
    session_leases.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)

    first = session_leases.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)
    second = session_leases.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert first is session_leases.LeaseMutation.APPLIED
    assert second is session_leases.LeaseMutation.ABSENT


# --- heartbeat_lease: ownership (task 3994 defect 1) -----------------------


def _seed_lease(tmp_path: Path, *, slug: str = 'watcher-df-100', pid: int | None = None) -> Path:
    """Claim `watcher-df` for *slug*/*pid* under tmp_path; return its lease path.

    Shared by the three task-3994 sections below (heartbeat ownership, release
    ownership, lease_status) -- defined once, here, at its first use.
    """
    holder = session_leases.LeaseHolder(
        session_slug=slug, pid=os.getpid() if pid is None else pid, start_ts=_NOW.isoformat()
    )
    session_leases.claim_lease('watcher-df', holder=holder, root=tmp_path, now=_NOW)
    return session_leases.lease_path_for_name('watcher-df', root=tmp_path)


def test_heartbeat_lease_by_a_stranger_is_refused_and_the_mtime_is_untouched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)
    set_mtime(lease_path, _NOW, timedelta(hours=1))
    backdated_mtime = lease_path.stat().st_mtime

    with caplog.at_level(logging.ERROR):
        result = session_leases.heartbeat_lease('watcher-df', slug='watcher-df-STRANGER', root=tmp_path)

    assert result is session_leases.LeaseMutation.REFUSED
    # THE assertion: an unrelated caller can no longer keep someone else's
    # lease structurally unreapable by refreshing its heartbeat clock.
    assert lease_path.stat().st_mtime == backdated_mtime
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors
    blob = ' '.join(r.getMessage() for r in errors)
    assert 'watcher-df-100' in blob
    assert 'watcher-df-STRANGER' in blob


def test_heartbeat_lease_on_an_absent_lease_is_absent_and_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        result = session_leases.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is session_leases.LeaseMutation.ABSENT
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_heartbeat_lease_on_a_reaped_lease_is_absent_and_the_owner_can_re_claim(
    tmp_path: Path,
) -> None:
    """ABSENT on a HEARTBEAT is not the harmless idempotence it is on a release.

    Code backing for the skill contract: the holder's own per-cycle heartbeat
    reporting ABSENT means its lease is GONE (reaped here; an operator
    `--force` release does the same), so the session is now running un-leased
    and any duplicate can take the name freely. The prescribed response is to
    re-claim, which must actually work — a watcher that kept beating into the
    void would sit at `result=absent` forever, which is the duplicate-watcher
    condition the lease exists to prevent.
    """
    lease_path = _seed_lease(tmp_path, pid=_DEAD_PID)
    set_mtime(lease_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))
    assert [r.reason for r in session_leases.reap_stale_leases(root=tmp_path, now=_NOW)] == ['stale_pid']

    result = session_leases.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is session_leases.LeaseMutation.ABSENT
    # ...and re-claiming is the documented recovery, so it must succeed.
    reclaim = session_leases.claim_lease(
        'watcher-df',
        holder=session_leases.LeaseHolder(
            session_slug='watcher-df-100', pid=os.getpid(), start_ts=_NOW.isoformat()
        ),
        root=tmp_path,
        now=_NOW,
    )
    assert reclaim.acquired
    assert session_leases.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path) is (
        session_leases.LeaseMutation.APPLIED
    )


def test_heartbeat_lease_on_a_corrupt_body_is_refused_with_mtime_untouched(
    tmp_path: Path,
) -> None:
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')
    set_mtime(lease_path, _NOW, timedelta(hours=1))
    backdated_mtime = lease_path.stat().st_mtime

    result = session_leases.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is session_leases.LeaseMutation.REFUSED
    assert lease_path.stat().st_mtime == backdated_mtime


def test_heartbeat_lease_with_force_overrides_a_stranger_refusal_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)
    set_mtime(lease_path, _NOW, timedelta(hours=1))
    backdated_mtime = lease_path.stat().st_mtime

    with caplog.at_level(logging.WARNING):
        result = session_leases.heartbeat_lease(
            'watcher-df', slug='watcher-df-STRANGER', force=True, root=tmp_path
        )

    assert result is session_leases.LeaseMutation.FORCED
    assert lease_path.stat().st_mtime > backdated_mtime
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings
    blob = ' '.join(r.getMessage() for r in warnings)
    assert 'watcher-df-100' in blob
    assert 'watcher-df-STRANGER' in blob


def test_heartbeat_lease_faults_soft_when_utime_itself_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _seed_lease(tmp_path)

    def _boom_utime(*_args: object, **_kwargs: object) -> None:
        raise OSError('simulated permission error')

    monkeypatch.setattr(session_leases.os, 'utime', _boom_utime)

    with caplog.at_level(logging.ERROR):
        result = session_leases.heartbeat_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    # Fail-soft: a lease-substrate fault never raises into a watcher's loop.
    assert result is session_leases.LeaseMutation.FAULTED
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_a_strangers_refused_heartbeat_leaves_a_dead_holders_lease_reapable(
    tmp_path: Path,
) -> None:
    # The structural-unreapability defect, pinned end to end: before task 3994
    # any caller could bump a dead holder's lease mtime forever, so the
    # (dead pid AND aged heartbeat) staleness AND-guard could never fire and
    # the lease outlived its holder indefinitely.
    lease_path = _seed_lease(tmp_path, slug='watcher-df-dead', pid=_DEAD_PID)
    set_mtime(lease_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    assert session_leases.heartbeat_lease('watcher-df', slug='watcher-df-STRANGER', root=tmp_path) is (
        session_leases.LeaseMutation.REFUSED
    )

    reaped = session_leases.reap_stale_leases(root=tmp_path, now=_NOW)

    assert {r.lease_name: r.reason for r in reaped} == {'watcher-df': 'stale_pid'}
    assert not lease_path.exists()


# --- release_lease: ownership (task 3994 defect 1) -------------------------


def test_release_lease_by_a_stranger_is_refused_and_the_lease_survives(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)
    original_body = lease_path.read_text()

    with caplog.at_level(logging.ERROR):
        result = session_leases.release_lease('watcher-df', slug='watcher-df-STRANGER', root=tmp_path)

    assert result is session_leases.LeaseMutation.REFUSED
    assert lease_path.is_file()
    # Byte-identical: a refused release must not touch the holder's body.
    assert lease_path.read_text() == original_body
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, 'a refused release must be loud (INV-2 structured-facts-at-failure)'
    blob = ' '.join(r.getMessage() for r in errors)
    # Both parties named: the real holder AND the would-be releaser.
    assert 'watcher-df-100' in blob
    assert 'watcher-df-STRANGER' in blob


def test_release_lease_on_an_absent_lease_is_absent_and_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR):
        result = session_leases.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    assert result is session_leases.LeaseMutation.ABSENT
    # An idempotent release is not a failure -- it must not cry ERROR.
    assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


def test_release_lease_on_a_corrupt_body_is_refused_and_the_file_survives(
    tmp_path: Path,
) -> None:
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')

    result = session_leases.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    # Fail TOWARD held, mirroring _read_lease_holder_state's documented rule.
    assert result is session_leases.LeaseMutation.REFUSED
    assert lease_path.read_text() == '{not valid json'


def test_release_lease_with_force_overrides_a_stranger_refusal_loudly(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)

    with caplog.at_level(logging.WARNING):
        result = session_leases.release_lease(
            'watcher-df', slug='watcher-df-STRANGER', force=True, root=tmp_path
        )

    assert result is session_leases.LeaseMutation.FORCED
    assert not lease_path.exists()
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, 'an operator force must be attributable'
    blob = ' '.join(r.getMessage() for r in warnings)
    assert 'watcher-df-100' in blob  # the displaced holder
    assert 'watcher-df-STRANGER' in blob  # who forced it


def test_release_lease_with_force_on_a_corrupt_body_removes_it(tmp_path: Path) -> None:
    lease_path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text('{not valid json')

    result = session_leases.release_lease('watcher-df', slug='watcher-df-100', force=True, root=tmp_path)

    assert result is session_leases.LeaseMutation.FORCED
    assert not lease_path.exists()


def test_release_lease_faults_soft_when_the_unlink_itself_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    lease_path = _seed_lease(tmp_path)
    real_unlink = Path.unlink

    def _flaky_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == lease_path:
            raise OSError('simulated permission error')
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', _flaky_unlink)

    with caplog.at_level(logging.ERROR):
        result = session_leases.release_lease('watcher-df', slug='watcher-df-100', root=tmp_path)

    # Fail-soft: a lease-substrate fault never raises into a watcher's loop.
    assert result is session_leases.LeaseMutation.FAULTED
    assert lease_path.is_file()
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# reap_stale_leases matrix
# ---------------------------------------------------------------------------


def test_reap_stale_leases_matrix(tmp_path: Path) -> None:
    live_holder = session_leases.LeaseHolder(session_slug='kept-live', pid=os.getpid(), start_ts=_NOW.isoformat())
    session_leases.claim_lease('kept-live', holder=live_holder, root=tmp_path, now=_NOW)
    set_mtime(session_leases.lease_path_for_name('kept-live', root=tmp_path), _NOW, timedelta(days=30))

    dead_within_ttl = session_leases.LeaseHolder(
        session_slug='kept-dead-fresh', pid=_DEAD_PID, start_ts=_NOW.isoformat()
    )
    session_leases.claim_lease('kept-dead-fresh', holder=dead_within_ttl, root=tmp_path, now=_NOW)
    set_mtime(
        session_leases.lease_path_for_name('kept-dead-fresh', root=tmp_path),
        _NOW,
        session_leases.LEASE_HEARTBEAT_TTL - timedelta(minutes=1),
    )

    dead_past_ttl = session_leases.LeaseHolder(
        session_slug='reap-dead-stale', pid=_DEAD_PID, start_ts=_NOW.isoformat()
    )
    session_leases.claim_lease('reap-dead-stale', holder=dead_past_ttl, root=tmp_path, now=_NOW)
    set_mtime(
        session_leases.lease_path_for_name('reap-dead-stale', root=tmp_path),
        _NOW,
        session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1),
    )

    corrupt_path = session_leases.lease_path_for_name('reap-corrupt', root=tmp_path)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_text('{not valid json')
    set_mtime(corrupt_path, _NOW, session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1))

    reaped = session_leases.reap_stale_leases(root=tmp_path, now=_NOW)

    reaped_by_name = {r.lease_name: r.reason for r in reaped}
    assert reaped_by_name == {
        'reap-dead-stale': 'stale_pid',
        'reap-corrupt': 'corrupt',
    }
    remaining = {p.stem for p in session_leases.leases_dir(root=tmp_path).iterdir()}
    assert remaining == {'kept-live', 'kept-dead-fresh'}


def test_reap_stale_leases_continues_sweep_when_one_removal_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    poison_holder = session_leases.LeaseHolder(session_slug='poison', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    session_leases.claim_lease('poison', holder=poison_holder, root=tmp_path, now=_NOW)
    victim_holder = session_leases.LeaseHolder(session_slug='victim', pid=_DEAD_PID, start_ts=_NOW.isoformat())
    session_leases.claim_lease('victim', holder=victim_holder, root=tmp_path, now=_NOW)
    for lease_name in ('poison', 'victim'):
        set_mtime(
            session_leases.lease_path_for_name(lease_name, root=tmp_path),
            _NOW,
            session_leases.LEASE_HEARTBEAT_TTL + timedelta(minutes=1),
        )

    poison_path = session_leases.lease_path_for_name('poison', root=tmp_path)
    real_unlink = Path.unlink

    def _flaky_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == poison_path:
            raise OSError('simulated permission error')
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', _flaky_unlink)

    with caplog.at_level(logging.ERROR):
        reaped = session_leases.reap_stale_leases(root=tmp_path, now=_NOW)

    assert {r.lease_name for r in reaped} == {'victim'}
    assert poison_path.is_file()  # left behind, not silently claimed as reaped
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# lease_status / lease-show (task 3994 defect 3)
#
# `cat <lease>` CANNOT show freshness: the body carries an immutable
# `start_ts` and the heartbeat lives only in the file mtime, so the archaeology
# a human actually needs is `cat` + `stat -c %y` + a TTL comparison done by
# hand. That is what misled the 2026-08-08 rotation. lease_status computes all
# of it through the same _read_lease_holder_state claim_lease decides on --
# one reader, one clock -- and lease-show prints it.
# ---------------------------------------------------------------------------


def test_lease_status_reports_the_holder_and_the_heartbeat_clock(tmp_path: Path) -> None:
    path = _seed_lease(tmp_path)
    set_mtime(path, _NOW, timedelta(minutes=30))

    status = session_leases.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.name == 'watcher-df'
    assert status.state == 'held'
    assert status.holder_slug == 'watcher-df-100'
    assert status.holder_pid == os.getpid()
    assert status.holder_pid_alive is True
    # Freshness is derived from the MTIME, the axis `cat` cannot show.
    assert int(status.heartbeat_age_secs) == 1800
    # heartbeat_ts is None only for an absent lease; a held one always has one.
    assert status.heartbeat_ts is not None
    assert datetime.fromisoformat(status.heartbeat_ts).replace(microsecond=0) == _NOW - timedelta(
        minutes=30
    )
    assert status.reclaimable is False


def test_lease_status_heartbeat_age_tracks_the_mtime(tmp_path: Path) -> None:
    path = _seed_lease(tmp_path)

    for age in (timedelta(seconds=5), timedelta(minutes=17), timedelta(hours=3)):
        set_mtime(path, _NOW, age)
        status = session_leases.lease_status('watcher-df', root=tmp_path, now=_NOW)
        assert int(status.heartbeat_age_secs) == int(age.total_seconds())


@pytest.mark.parametrize(
    ('age', 'expected'),
    [
        (session_leases.LEASE_HEARTBEAT_TTL - timedelta(seconds=1), False),
        (session_leases.LEASE_HEARTBEAT_TTL, False),
        (session_leases.LEASE_HEARTBEAT_TTL + timedelta(seconds=1), True),
    ],
)
def test_lease_status_reclaimable_flips_exactly_at_the_ttl_for_a_dead_pid(
    tmp_path: Path, age: timedelta, expected: bool
) -> None:
    path = _seed_lease(tmp_path, pid=_DEAD_PID)
    set_mtime(path, _NOW, age)

    status = session_leases.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.holder_pid_alive is False
    # Exactly claim_lease/reap_stale_leases' own predicate: staleness needs
    # BOTH a dead pid AND a heartbeat STRICTLY past the TTL. The display and
    # the decision cannot disagree, because they are the same computation.
    assert status.reclaimable is expected


def test_lease_status_never_reports_a_live_holder_reclaimable_at_any_age(tmp_path: Path) -> None:
    path = _seed_lease(tmp_path)  # our own, provably-live, pid
    set_mtime(path, _NOW, timedelta(days=30))

    status = session_leases.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.holder_pid_alive is True
    assert int(status.heartbeat_age_secs) == 30 * 24 * 3600
    # A live holder is never reclaimable however quiet it has been -- the
    # inverse reading is the duplicate-spawn incident.
    assert status.reclaimable is False


def test_lease_status_on_an_absent_lease_reports_state_absent(tmp_path: Path) -> None:
    status = session_leases.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.name == 'watcher-df'
    assert status.state == 'absent'
    assert status.holder_slug is None
    assert status.holder_pid is None
    assert status.holder_pid_alive is False
    assert status.reclaimable is False


def test_lease_status_on_a_corrupt_body_is_unreadable_but_still_dates_the_heartbeat(
    tmp_path: Path,
) -> None:
    path = session_leases.lease_path_for_name('watcher-df', root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{not valid json')
    set_mtime(path, _NOW, timedelta(minutes=42))

    status = session_leases.lease_status('watcher-df', root=tmp_path, now=_NOW)

    # Fail-soft: an unreadable body is REPORTED, never raised, and never
    # rendered as a fake holder -- but its freshness is still real, which is
    # exactly what decides whether it is reapable.
    assert status.state == 'unreadable'
    assert status.holder_slug is None
    assert status.holder_pid is None
    assert int(status.heartbeat_age_secs) == 2520


def test_lease_status_is_read_only(tmp_path: Path) -> None:
    path = _seed_lease(tmp_path)
    set_mtime(path, _NOW, timedelta(hours=1))
    mtime_before = path.stat().st_mtime
    body_before = path.read_text()

    session_leases.lease_status('watcher-df', root=tmp_path, now=_NOW)

    # Inspecting a lease must never be a heartbeat: an operator running
    # lease-show cannot be allowed to keep a dead holder's lease unreapable.
    assert path.stat().st_mtime == mtime_before
    assert path.read_text() == body_before


class TestLeaseSlugIsNotASessionRecordKey:
    """`read_record(<a lease slug>)` raises — pinned so the withdrawn corroboration cannot return.

    Task 3994 originally corroborated the orphan signal with the holder's own
    session_registry record, via ``read_record(holder.session_slug)``. That
    lookup is STRUCTURALLY IMPOSSIBLE, and the tests that appeared to exercise
    its branches only did so by hand-writing a record under a lease-shaped
    slug — a state no production path can produce. They were deleted; this
    class replaces them with the reason.

    MEASURED EVIDENCE (2026-08-15, live fleet root): the real
    ``~/.claude/fleet/leases/watcher-df.lease`` holds
    ``session_slug=watcher-df-1894895``, while 0 of the 46,624
    ``~/.claude/fleet/sessions/`` record dirs begin with ``watcher-``. Real
    record slugs look like ``architect-dark_factory-3133-25737ee4-...``.

    The namespaces differ in all three segments:
    - the LEASE slug is ``<lease name>-<session pid>``, built by
      ``default_lease_slug`` (task 4248 moved it out of shell in the SKILLs and
      into the CLI, for the reason 3994 moved the pid there);
    - the RECORD slug comes from ``build_session_slug(role, project, task,
      <uniqueness token>)``, where the token is a session UUID for a
      hand-launched session (``session_hooks.hook_session_slug``) and the
      project token is the registry's project id (``dark_factory``), not the
      lease's short name (``df``).

    No test here compares a literal this file wrote against
    ``build_session_slug``'s output, on purpose: the disjointness RATIONALE
    belongs to (and is recorded in) ``LeaseHolder.session_slug``'s docstring,
    and such a test cannot fail as a result of any production change. The
    first test has real coupling — it would break if ``read_record`` ever
    started resolving a lease slug, which is exactly the reintroduction worth
    catching. The second pins the pid pointer (task 4237) as the join that
    does reach the record, side by side with the lookup that cannot.
    """

    # A hand-launched session's uniqueness token: hook_session_slug passes the
    # hook's `session_id` (a UUID string) to build_session_slug as that token.
    # Kept literal so the test is deterministic.
    SESSION_UUID = '25737ee4-3b1e-4a7f-9f0e-6a1c2d3e4f50'

    def _lease_slug(self, project: str = 'df', pid: int = 1894895) -> str:
        """The lease slug exactly as production builds it (task 4248).

        Routed through ``default_lease_slug`` rather than hand-building the
        f-string: the disjointness this class pins is only interesting if the
        left-hand side is the REAL lease slug, so the helper must break when
        production's builder changes shape.
        """
        slug = session_leases.default_lease_slug(f'watcher-{project}', {session_identity.SESSION_PID_ENV: str(pid)})
        assert slug is not None  # pid > 0, so the builder always derives here
        return slug

    def _record_slug(self, project: str = 'dark_factory') -> str:
        """The record slug exactly as session_hooks.hook_session_slug builds it."""
        return session_records.build_session_slug(
            'session',
            project,
            None,
            self.SESSION_UUID,
        )

    def test_a_real_record_provably_does_not_live_under_its_lease_slug(
        self, tmp_path: Path
    ) -> None:
        # A well-formed record IS written for this notional session, under the
        # slug the registry actually keys on...
        record_slug = self._record_slug()
        session_records.write_record(make_record(session_slug=record_slug), root=tmp_path)
        assert session_records.read_record(record_slug, root=tmp_path).session_slug == record_slug

        # ...and read_record(<lease slug>) STILL raises FileNotFoundError. That
        # is why holder_session_state was a constant 'absent' in production.
        with pytest.raises(FileNotFoundError):
            session_records.read_record(self._lease_slug(), root=tmp_path)

    def test_the_pid_pointer_reaches_the_record_the_lease_slug_cannot(
        self, tmp_path: Path
    ) -> None:
        pid = 1894895
        record_slug = self._record_slug()
        session_records.write_record(
            make_record(session_slug=record_slug, claude_owner_pid=pid), root=tmp_path
        )
        session_records.write_session_pointer(pid, record_slug, root=tmp_path)

        with pytest.raises(FileNotFoundError):
            session_records.read_record(self._lease_slug(pid=pid), root=tmp_path)
        linked = session_records.resolve_session_slug_for_pid(pid, root=tmp_path)
        assert linked is not None
        assert session_records.read_record(linked, root=tmp_path).session_slug == record_slug


# --- default_lease_slug (task 4248)----------------------------------------
#
# 3994 moved the lease PID into code because it depended on every skill doc
# getting one shell token right. The SLUG stayed in shell -- and 3994 then made
# it load-bearing, since a mismatch on `lease-heartbeat`/`lease-release` is
# REFUSED. These tests pin the builder that moves it into code too.


def test_default_lease_slug_reproduces_the_measured_production_slug() -> None:
    """The defaulted slug is BYTE-IDENTICAL to what the WATCHER skills built.

    ``watcher-df-1894895`` is the slug measured in the live fleet root on
    2026-08-15 and recorded in ``TestLeaseSlugAndRecordSlugAreDisjoint``. This
    equality is the back-compat oracle FOR THE TWO WATCHER LEASES ONLY --
    ``watcher-<project>`` and ``recon-watcher-<project>``, whose pre-4248 skill
    prescription was already ``<name>-${CLAUDE_PID:-$PPID}``. For those, an
    IN-FLIGHT session that stops passing ``--slug`` keeps its lease with no
    restart, because the CLI re-derives exactly the token it claimed with.

    It does NOT generalise to ``/unblock`` -- see
    ``test_the_unblock_lease_slug_shape_changed_at_4248``, which records that
    asymmetry rather than leaving this docstring to imply it away.
    """
    assert session_leases.default_lease_slug('watcher-df', {session_identity.SESSION_PID_ENV: '1894895'}) == 'watcher-df-1894895'
    assert (
        session_leases.default_lease_slug('recon-watcher-df', {session_identity.SESSION_PID_ENV: '1894895'})
        == 'recon-watcher-df-1894895'
    )


def test_the_unblock_lease_slug_shape_changed_at_4248() -> None:
    """The `/unblock` lease slug is NOT back-compatible, and that is recorded here.

    The two watcher skills built ``<lease name>-<pid>``, so deriving from the
    name reproduces their token exactly. ``/unblock`` did not: its lease NAME
    is ``unblock-<project>#<TASK_ID>`` (``build_lease_name``'s ``#``) while its
    skill built the slug with a ``-`` separator and no ``#`` at all. Deriving
    from the name therefore yields a DIFFERENT token.

    ROLLOUT CONSEQUENCE, stated so nobody rediscovers it from a lingering
    lease: an `/unblock` session that CLAIMED under the pre-4248 prescription
    and then releases slug-less gets ``result=refused`` -- non-destructive, but
    its lease lingers and falsely reports a holder to the next `/unblock` on
    that task until the 2h ``LEASE_HEARTBEAT_TTL`` ages it out. Such a session
    must either keep passing its ORIGINAL ``--slug`` on release or accept the
    TTL wait; ``skills/unblock/SKILL.md`` says so at the release step. Sessions
    that claim after 4248 are unaffected -- both ends derive the same token.
    """
    name = session_leases.build_lease_name('unblock', 'df', '2085')
    derived = session_leases.default_lease_slug(name, {session_identity.SESSION_PID_ENV: '1894895'})

    assert derived == 'unblock-df#2085-1894895'
    # The exact string the pre-4248 SKILL.md prescribed, for contrast.
    assert derived != 'unblock-df-2085-1894895'


@pytest.mark.parametrize(
    'env',
    [
        {},
        {'CLAUDE_PID': ''},
        {'CLAUDE_PID': '   '},
        {'CLAUDE_PID': 'not-a-pid'},
        {'CLAUDE_PID': '0'},
        {'CLAUDE_PID': '-1'},
    ],
    ids=['unset', 'empty', 'blank', 'non-numeric', 'zero', 'negative'],
)
def test_default_lease_slug_refuses_to_derive_when_the_session_pid_is_unresolvable(
    env: dict[str, str],
) -> None:
    """An underivable slug is None -- and specifically NOT ``f'{name}-0'``.

    Same parametrize set as ``resolve_session_pid``'s degradation sweep, on
    purpose: these are exactly the envs where that function degrades to pid 0.

    THE assertion is the second one. ``resolve_session_pid`` may degrade to 0
    because a pid is a LIVENESS probe and 0 is provably dead, which keeps a
    crashed holder's lease reapable. A slug is an IDENTITY, and ``{name}-0``
    is not unique: two concurrently-degraded sessions contending the same lease
    name would compute an IDENTICAL slug, so each would pass the other's
    ``_is_lease_owner`` check and could heartbeat or release the other's lease
    -- 3994 defect 1 ("any caller may evict/refresh any lease") reopened on the
    degraded path. Refusing to synthesize a token it cannot make unique is the
    only answer that keeps the ownership guard honest; the CLI then exits 2 and
    names ``--slug`` as the escape hatch.
    """
    assert session_leases.default_lease_slug('watcher-df', env) is None
    assert session_leases.default_lease_slug('watcher-df', env) != 'watcher-df-0'


def test_default_lease_slug_defaults_to_os_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CLAUDE_PID', '424242')
    assert session_leases.default_lease_slug('watcher-df') == 'watcher-df-424242'


def test_default_lease_slug_passes_a_task_scoped_name_through_verbatim() -> None:
    """``build_lease_name``'s ``#`` separator survives into the slug untouched.

    A lease slug is never a filesystem path: ``lease_path_for_name`` sanitizes
    only the NAME, and ``sanitize_slug`` applies only to session-RECORD slugs.
    A lease slug lives inside the lease's JSON body and is compared for
    equality alone, so the task-scoped ``unblock-df#2085`` needs no escaping.
    """
    name = session_leases.build_lease_name('unblock', 'df', '2085')
    assert name == 'unblock-df#2085'
    assert session_leases.default_lease_slug(name, {session_identity.SESSION_PID_ENV: '55'}) == 'unblock-df#2085-55'
