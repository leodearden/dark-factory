"""Tests for a spawn-created LAUNCHING record before its owner's SessionStart binds it: an inheritor cannot claim it, the CLAUDE_SPAWN_OWNER_PPID probe decides, and withholding expires (task 4193)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _process_tree_fakes import claude_process_tree
from _session_hook_fixtures import (
    assert_hook_env_isolated,
    isolate_hook_env,
    launching_record,
    owner_ppid_env,
)

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr


@pytest.fixture(autouse=True)
def _isolate_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_hook_env(monkeypatch, tmp_path)


def test_hook_env_is_isolated_for_this_module(tmp_path: Path) -> None:
    assert_hook_env_isolated(tmp_path)


# ---------------------------------------------------------------------------
# Task 4193 amendment: an inheritor cannot claim an unsighted launching record
# ---------------------------------------------------------------------------


def test_refresh_path_does_not_bind_a_still_launching_record(tmp_path: Path) -> None:
    # spawn-claude.sh's `launching` write has not been sighted by its
    # owner's SessionStart yet (delayed, timed out under the 10s hook
    # timeout, or failed). A nested claude's Stop landing first must not be
    # allowed to bind ITS id into that record -- that would invert ownership
    # permanently, since the launching record is the one holding
    # role/project/prompt/result_file and the one finish() writes `exited`
    # to.
    slug = 'session-cockpit-3215100'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            role='session',
            project='cockpit',
            prompt='/spawn cockpit',
            launcher_pid=3215100,
            start_ts=datetime.now(UTC).isoformat(),
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id is None
    # ...and the STATUS is withheld too, not just the binding. Asserting only
    # the binding once let the real defect through: refresh_record is a
    # read-modify-WRITE, so a guard sitting after it still left the nested
    # Stop advertising a still-launching spawn as `idle` in the cockpit.
    assert record.status == sr.Status.LAUNCHING
    # Blast radius, stated explicitly: withholding is not forking. The probe
    # ADOPTS an unbound record (that is the legacy-migration path), so the
    # event lands on this slug and is then dropped -- it must not also mint a
    # second, nested-owned row for the cockpit to show alongside the spawn.
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_refresh_path_withholds_a_question_during_the_unbound_launch_window(
    tmp_path: Path,
) -> None:
    # The question is the attention rail's payload: a nested claude's prompt
    # landing on the spawner's record would make the cockpit claim the SPAWN
    # is the thing awaiting input. Withheld on the same unknowable-provenance
    # grounds as the status and the binding.
    slug = 'session-cockpit-3215103'
    launching_record(slug, tmp_path, pid=3215103)
    hook_input = {
        'session_id': 'uuid-nested',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'Do you want to proceed?',
    }

    sh.run_notification(hook_input, {'CLAUDE_SPAWN_SESSION_ID': slug}, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.question is None
    assert record.status == sr.Status.LAUNCHING
    assert record.claude_session_id is None


def test_refresh_path_writes_normally_to_a_bound_launching_record(
    tmp_path: Path,
) -> None:
    # The window is LAUNCHING *and unbound*. A LAUNCHING record that already
    # carries a binding HAS a discriminator, so _env_slug_is_owned has proved
    # this event belongs to the owner -- withholding there would strand a
    # legitimately-owned record at `launching`.
    slug = 'session-cockpit-3215104'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            launcher_pid=3215104,
            claude_session_id='uuid-owner',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'}

    sh.run_stop(hook_input, {'CLAUDE_SPAWN_SESSION_ID': slug}, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.IDLE
    assert record.claude_session_id == 'uuid-owner'


def test_owning_session_start_still_wins_the_launching_record_after_a_nested_stop(
    tmp_path: Path,
) -> None:
    # The whole inversion story end to end: the nested event refrains, then
    # the owner's SessionStart binds, and from there the nested claude forks
    # onto its own record exactly as it should.
    slug = 'session-cockpit-3215101'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            launcher_pid=3215101,
            result_file='/tmp/result-3215101.md',
        ),
        root=tmp_path,
    )
    env = owner_ppid_env(slug, 3215510)
    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    owner = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    # The owning claude is the direct child of the payload bash whose pid
    # spawn-claude.sh exported; the nested one's parent is its agent's
    # Bash-tool shell, so the same env var yields opposite verdicts.
    nested_tree = claude_process_tree(3215777, 3215776)
    owner_tree = claude_process_tree(3215511, 3215510)

    sh.run_stop(nested, env, root=tmp_path, process_tree=nested_tree)
    # Assert HERE, before run_session_start's unconditional `status = RUNNING`
    # masks it: that reset is what hid the original defect from this test.
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.LAUNCHING
    sh.run_session_start(owner, env, root=tmp_path, process_tree=owner_tree)

    parent = sr.read_record(slug, root=tmp_path)
    assert parent.claude_session_id == 'uuid-parent'
    assert parent.result_file == '/tmp/result-3215101.md'

    # The nested claude's next event now forks instead of flipping it.
    sh.run_stop(nested, env, root=tmp_path, process_tree=nested_tree)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING
    forked_slug = sh.hook_session_slug(nested, env, root=tmp_path, process_tree=nested_tree)
    forked = sr.read_record(forked_slug, root=tmp_path)
    assert forked.status == sr.Status.IDLE
    assert forked.claude_session_id == 'uuid-nested'


def test_refresh_path_still_binds_a_legacy_record_that_is_past_launching(
    tmp_path: Path,
) -> None:
    # The LAUNCHING guard must not cost the migration case it sits next to:
    # a pre-task-4193 in-flight record is RUNNING/AWAITING_INPUT/IDLE, never
    # LAUNCHING, so its proven owner still binds it on the very next event.
    slug = 'session-cockpit-3215102'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.AWAITING_INPUT, launcher_pid=3215102
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    tree = claude_process_tree(3215521, 3215520)

    sh.run_stop(hook_input, owner_ppid_env(slug, 3215520), root=tmp_path, process_tree=tree)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-parent'


# ---------------------------------------------------------------------------
# Task 4193 L2 ruling item 4: the UNBOUND launch window.
#
# Before the owner's own SessionStart lands, the spawn-created record carries
# no claude_session_id, so the stdin-session_id discriminator does not exist
# yet and the FIRST event to arrive captures the record -- owner or inheritor.
# The stateless CLAUDE_SPAWN_OWNER_PPID probe decides there instead.
# ---------------------------------------------------------------------------


def test_nested_session_start_cannot_capture_an_unbound_launching_record(
    tmp_path: Path,
) -> None:
    # THE REVIEWER'S BLOCKING CASE (esc-4193-9). SessionStart is the event
    # that reaches the launch window FIRST -- a nested claude always fires
    # its own SessionStart before any Notification/Stop -- so a guard wired
    # only into the refresh path never gets a chance to fire. Here the
    # nested claude's SessionStart arrives at a LAUNCHING+unbound record and
    # must NOT bind it: it forks onto its own slug, leaving the record that
    # holds role/prompt/result_file (the one spawn-claude.sh writes 'exited'
    # to) untouched and still available to its true owner.
    slug = 'session-cockpit-4193001'
    launching_record(slug, tmp_path)
    env = owner_ppid_env(slug, 4193500)
    # The nested claude's parent is its agent's Bash-tool shell, not the
    # payload bash spawn-claude.sh exported.
    tree = claude_process_tree(4193777, 4193776)

    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(nested, env, root=tmp_path, process_tree=tree)

    spawn_record = sr.read_record(slug, root=tmp_path)
    assert spawn_record.claude_session_id is None
    assert spawn_record.status is sr.Status.LAUNCHING
    assert spawn_record.role == 'cockpit'
    forked_slug = sh.hook_session_slug(nested, env, root=tmp_path, process_tree=tree)
    forked = sr.read_record(forked_slug, root=tmp_path)
    assert forked.claude_session_id == 'uuid-nested'
    assert forked.parent_session_id == slug


def test_owner_session_start_binds_the_unbound_launching_record(
    tmp_path: Path,
) -> None:
    # The other half: the session spawn-claude.sh actually launched is the
    # direct child of the payload bash whose pid it exported, so it adopts
    # and binds the spawn-created record -- exactly one row, converged.
    slug = 'session-cockpit-4193002'
    launching_record(slug, tmp_path, pid=4193002)
    env = owner_ppid_env(slug, 4193500)
    tree = claude_process_tree(4193501, 4193500)

    owner = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(owner, env, root=tmp_path, process_tree=tree)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'uuid-owner'
    assert record.status is sr.Status.RUNNING
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_nested_then_owner_session_start_leaves_the_spawn_record_with_the_owner(
    tmp_path: Path,
) -> None:
    # The end-to-end ordering the reviewer reproduced against the branch:
    # nested SessionStart -> nested Stop -> owner SessionStart. Previously
    # the nested SessionStart inverted ownership permanently. Now the spawn
    # record survives untouched until its real owner arrives.
    slug = 'session-cockpit-4193003'
    launching_record(slug, tmp_path, pid=4193003)
    env = owner_ppid_env(slug, 4193500)

    nested_tree = claude_process_tree(4193777, 4193776)
    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(nested, env, root=tmp_path, process_tree=nested_tree)
    sh.run_stop(nested, env, root=tmp_path, process_tree=nested_tree)
    assert sr.read_record(slug, root=tmp_path).status is sr.Status.LAUNCHING

    owner_tree = claude_process_tree(4193501, 4193500)
    owner = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(owner, env, root=tmp_path, process_tree=owner_tree)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'uuid-owner'
    assert record.status is sr.Status.RUNNING


def test_no_verdict_nested_session_start_leaves_the_spawn_record_for_its_owner(
    tmp_path: Path,
) -> None:
    # esc-4193-10, the reviewer's end-to-end repro. Two routine shapes yield
    # NO ownership verdict at all: macOS (no /proc, so _owning_claude_pid()
    # is always None -- "a first-class lane, cf. task 4058") and the deploy
    # window on Linux (hook scripts are read at event time, but a session's
    # env is fixed at launch, so every session already running when this
    # lands has no CLAUDE_SPAWN_OWNER_PPID). The unbound arm ADOPTS there by
    # design -- that is the fail-soft, pre-task-4193 collapse -- but it must
    # not also BIND: binding is permanent, so the nested claude would own the
    # spawn record (the one holding role/prompt/result_file, the one
    # spawn-claude.sh's finish() writes 'exited' to) and the true owner would
    # be exiled onto a degraded session-x-<uuid> row one event later.
    slug = 'session-cockpit-4193007'
    launching_record(slug, tmp_path, pid=4193007)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}  # spawned before OWNER_PPID existed
    tree = claude_process_tree(None)

    nested = {'session_id': 'uuid-NESTED', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(nested, env, root=tmp_path, process_tree=tree)
    assert sr.read_record(slug, root=tmp_path).claude_session_id is None

    owner = {'session_id': 'uuid-OWNER', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(owner, env, root=tmp_path, process_tree=tree)

    # The owner still CONVERGES on the spawn record instead of forking, and
    # the record is still open (unbound) rather than claimed by the nested.
    assert sh.hook_session_slug(owner, env, root=tmp_path, process_tree=tree) == slug
    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id is None
    assert record.status is sr.Status.RUNNING
    assert record.role == 'cockpit'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_owner_ppid_proof_still_binds_where_no_verdict_would_not(
    tmp_path: Path,
) -> None:
    # The other half of the asymmetry: withholding the bind is scoped to
    # UNPROVEN ownership only, so a modern spawn (spawn-claude.sh exports
    # CLAUDE_SPAWN_OWNER_PPID inside $inner) still binds on its first
    # SessionStart and every later nested event is discriminable by the
    # session_id binding alone.
    slug = 'session-cockpit-4193008'
    launching_record(slug, tmp_path, pid=4193008)
    env = owner_ppid_env(slug, 4193600)

    owner = {'session_id': 'uuid-OWNER', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(owner, env, root=tmp_path, process_tree=claude_process_tree(4193601, 4193600))
    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-OWNER'

    # Now a nested claude, with the probe gone dark (no /proc): the BINDING
    # is the discriminator from here on, so it forks on its own.
    nested = {'session_id': 'uuid-NESTED', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_stop(nested, env, root=tmp_path, process_tree=claude_process_tree(None))

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'uuid-OWNER'
    assert record.status is sr.Status.RUNNING


def test_owner_stop_is_not_withheld_from_a_still_launching_record(
    tmp_path: Path,
) -> None:
    # Ruling item 4-iii. The launch-window withholding suppressed status AND
    # the pending question for EVERY event, including the legitimate
    # owner's. A SessionStart killed by _HOOK_TIMEOUT_SECS (reachable:
    # _resolve_wm_window_id's own docstring prices the pathological wmctrl
    # case at ~10.8s against a 10s budget) writes nothing, so the record
    # stays LAUNCHING+unbound and the session went invisible for its whole
    # life. An event whose OWNER_PPID matches must write.
    slug = 'session-cockpit-4193004'
    launching_record(slug, tmp_path, pid=4193004)
    env = owner_ppid_env(slug, 4193500)
    tree = claude_process_tree(4193501, 4193500)

    owner = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_notification({**owner, 'message': 'may I proceed?'}, env, root=tmp_path, process_tree=tree)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.AWAITING_INPUT
    assert record.question is not None and record.question.text == 'may I proceed?'
    assert record.claude_session_id == 'uuid-owner'


def test_unknown_provenance_event_is_still_withheld_inside_the_window(
    tmp_path: Path,
) -> None:
    # No verdict at all (a session spawned by a pre-task-4193
    # spawn-claude.sh, or a platform with no /proc): provenance is genuinely
    # unknowable, so the conservative withholding stands.
    slug = 'session-cockpit-4193005'
    launching_record(slug, tmp_path, pid=4193005)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(
        {'session_id': 'uuid-anon', 'cwd': '/home/leo/src/dark-factory'},
        env,
        root=tmp_path,
        process_tree=claude_process_tree(None),
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.LAUNCHING
    assert record.claude_session_id is None


def test_withholding_expires_so_a_stuck_record_is_never_blind_forever(
    tmp_path: Path,
) -> None:
    # Nothing else bounds the window: no LAUNCHING reaper, and
    # reap_stale_records needs a DEAD launcher_pid, which is
    # spawn-claude.sh's own $$ -- alive for the whole session. Past the
    # bound, a possibly-wrong status beats a permanently blind record.
    slug = 'session-cockpit-4193006'
    stale = datetime.now(UTC) - timedelta(seconds=sh._LAUNCH_WINDOW_WITHHOLD_MAX_SECS + 60)
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=4193006,
            start_ts=stale.isoformat(),
        ),
        root=tmp_path,
    )
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(
        {'session_id': 'uuid-late', 'cwd': '/home/leo/src/dark-factory'},
        env,
        root=tmp_path,
        process_tree=claude_process_tree(None),
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.IDLE
    # The expiry buys VISIBILITY, not ownership: provenance is still
    # unproven, so the record stays unbound and open for its true owner
    # rather than being claimed by whoever broke the deadlock (esc-4193-10).
    assert record.claude_session_id is None


@pytest.mark.parametrize(
    'start_ts',
    [
        pytest.param('', id='missing'),
        pytest.param('not-a-timestamp', id='unparseable'),
        pytest.param(
            (
                datetime.now(UTC) - timedelta(seconds=sh._LAUNCH_WINDOW_WITHHOLD_MAX_SECS + 60)
            ).replace(tzinfo=None).isoformat(),
            id='stale-naive',
        ),
    ],
)
def test_unreadable_or_stale_start_ts_expires_immediately_instead_of_withholding_forever(
    tmp_path: Path,
    start_ts: str,
) -> None:
    # start_ts defaults to '' on SessionRecord, and may also arrive corrupt
    # or simply stale. None of these is evidence the spawn is healthy:
    # treating any of them as "still within the window" (the old behaviour
    # for the first two) would let a record withhold FOREVER -- the exact
    # permanently-blind failure _LAUNCH_WINDOW_WITHHOLD_MAX_SECS exists to
    # bound (esc-4193-8 item 4-iii). A record with no readable -- or no
    # in-window -- clock must be treated as expired and become visible
    # immediately. The third (tz-naive) case also confirms a naive
    # timestamp is parsed (assumed UTC) rather than rejected as
    # unparseable; see test_naive_start_ts_inside_the_window_still_withholds
    # for the discriminating in-window counterpart.
    slug = 'session-cockpit-4661001'
    launching_record(slug, tmp_path, pid=4661001, start_ts=start_ts)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(
        {'session_id': 'uuid-expired', 'cwd': '/home/leo/src/dark-factory'},
        env,
        root=tmp_path,
        process_tree=claude_process_tree(None),
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.IDLE
    assert record.claude_session_id is None


def test_naive_start_ts_inside_the_window_still_withholds(
    tmp_path: Path,
) -> None:
    # The discriminating case for the tz-naive-assumed-UTC branch. A stale
    # naive timestamp (above) would also expire under a WRONG assumption --
    # e.g. local time at a non-negative UTC offset, or a dropped comparison
    # -- so it cannot tell "assumed UTC" apart from those. Only an in-window
    # naive timestamp discriminates: under a correct UTC assumption the
    # record is still within bound and must stay withheld (LAUNCHING).
    slug = 'session-cockpit-4661004'
    recent_naive = (datetime.now(UTC) - timedelta(seconds=5)).replace(tzinfo=None).isoformat()
    launching_record(slug, tmp_path, pid=4661004, start_ts=recent_naive)
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(
        {'session_id': 'uuid-naive-recent', 'cwd': '/home/leo/src/dark-factory'},
        env,
        root=tmp_path,
        process_tree=claude_process_tree(None),
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.status is sr.Status.LAUNCHING
    assert record.claude_session_id is None


def test_sibling_mode_running_unbound_record_is_still_protected(
    tmp_path: Path,
) -> None:
    # Ruling item 4-i: resolve_sibling() runs `refresh --status running`
    # immediately at launch on EVERY backend branch, so a
    # CLAUDE_SPAWN_MODE=sibling record is RUNNING-and-unbound and a
    # `status is LAUNCHING` predicate never matches it at all. Keying the
    # ownership probe on UNBOUND-ness instead covers it.
    slug = 'session-cockpit-4193007'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug, status=sr.Status.RUNNING, launcher_pid=4193007, role='cockpit'
        ),
        root=tmp_path,
    )
    env = owner_ppid_env(slug, 4193500)
    tree = claude_process_tree(4193777, 4193776)

    nested = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory', 'source': 'startup'}
    sh.run_session_start(nested, env, root=tmp_path, process_tree=tree)

    assert sr.read_record(slug, root=tmp_path).claude_session_id is None
    assert sr.read_record(
        sh.hook_session_slug(nested, env, root=tmp_path, process_tree=tree), root=tmp_path
    ).claude_session_id == 'uuid-nested'
