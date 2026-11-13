"""Tests that hook events bind a spawn record to its owning claude session_id, and that mismatched inheritors fork a record of their own (task 4193; model in docs/session-hook-ownership.md)."""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from _process_identity_variants import IdentityVariant, own_identity
from _process_tree_fakes import claude_process_tree
from _session_hook_fixtures import (
    assert_hook_env_isolated,
    isolate_hook_env,
    owner_ppid_env,
    stdin_json,
    write_bound_parent,
)
from shared.process_identity import ProcessIdentity

from orchestrator import session_hooks as sh
from orchestrator import session_registry as sr
from orchestrator.session_ownership_probes import EventProbes


@pytest.fixture(autouse=True)
def _isolate_hook_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate_hook_env(monkeypatch, tmp_path)


def test_hook_env_is_isolated_for_this_module(tmp_path: Path) -> None:
    assert_hook_env_isolated(tmp_path)


# ---------------------------------------------------------------------------
# Task 4193 step-1: SessionStart binds the owning claude session_id
# ---------------------------------------------------------------------------


def test_run_session_start_binds_claude_session_id_on_first_adoption(
    tmp_path: Path,
) -> None:
    # The first hook event to adopt a spawn-claude.sh `launching` record
    # stamps its own Claude Code session_id onto it -- the token that later
    # tells this session apart from a nested claude that merely inherited
    # CLAUDE_SPAWN_SESSION_ID. Binding needs the ownership PROOF
    # spawn-claude.sh exports (CLAUDE_SPAWN_OWNER_PPID): adopting is
    # fail-soft, claiming the record permanently is not (esc-4193-10).
    slug = 'session-cockpit-3215040'
    result_file = str(sr.record_path_for_slug(slug, root=tmp_path).parent / 'result.md')
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            role='session',
            project='cockpit',
            prompt='/spawn cockpit',
            cwd='/home/leo/src/dark-factory',
            launcher_pid=3215040,
            result_file=result_file,
        ),
        root=tmp_path,
    )

    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = owner_ppid_env(slug, 3215500)
    tree = claude_process_tree(3215501, 3215500)

    sh.run_session_start(hook_input, env, root=tmp_path, process_tree=tree)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'uuid-parent'
    assert record.status == sr.Status.RUNNING
    # The rich launching-vintage fields still survive the binding write.
    assert record.role == 'session'
    assert record.project == 'cockpit'
    assert record.prompt == '/spawn cockpit'
    assert record.launcher_pid == 3215040
    assert record.result_file == result_file
    dirs = list(sr.sessions_dir(root=tmp_path).iterdir())
    assert len(dirs) == 1
    assert dirs[0].name == slug


def test_run_session_start_binds_claude_session_id_on_hand_launched_record(
    tmp_path: Path,
) -> None:
    # No CLAUDE_SPAWN_SESSION_ID and no pre-existing record: the freshly
    # synthesized record is bound too. Harmless (that slug already embeds
    # the session_id, so it can never mismatch) and it keeps every record
    # self-describing.
    hook_input = {'session_id': 'sess-hl', 'cwd': '/home/leo/src/dark-factory'}

    sh.run_session_start(hook_input, env={}, root=tmp_path)

    slug = sh.hook_session_slug(hook_input, env={})
    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'sess-hl'


def test_run_session_start_does_not_rebind_existing_claude_session_id(
    tmp_path: Path,
) -> None:
    # Bind-once: a re-fired SessionStart against an already-bound record
    # must not churn the binding.
    slug = 'session-cockpit-3215042'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.LAUNCHING,
            launcher_pid=3215042,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-parent'


def test_run_session_start_leaves_claude_session_id_unbound_when_stdin_has_no_session_id(
    tmp_path: Path,
) -> None:
    # hook_session_slug falls back to the literal 'unknown' when stdin
    # carries no session_id; that must NEVER be bound as an owner token, or
    # every discriminator-less session would collide on one bogus binding.
    slug = 'session-cockpit-3215043'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=3215043),
        root=tmp_path,
    )
    hook_input = {'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id is None
    assert record.status == sr.Status.RUNNING


# ---------------------------------------------------------------------------
# Task 4193 step-2: mismatched inheritors fork their own record
# ---------------------------------------------------------------------------


def test_hook_session_slug_adopts_env_slug_when_record_absent(tmp_path: Path) -> None:
    # First sight: no record at the env slug yet, so this hook event is the
    # one that will create/claim it.
    slug = 'session-cockpit-3215050'
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


def test_hook_session_slug_adopts_env_slug_when_record_unbound(tmp_path: Path) -> None:
    # A spawn-claude.sh `launching`-vintage (or pre-task-4193) record carries
    # no binding; the first matching hook adopts and binds it.
    slug = 'session-cockpit-3215051'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.LAUNCHING, launcher_pid=3215051),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


def test_hook_session_slug_adopts_env_slug_when_binding_matches(tmp_path: Path) -> None:
    slug = 'session-cockpit-3215052'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215052,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


def test_hook_session_slug_forks_when_binding_mismatches(tmp_path: Path) -> None:
    # A nested claude inherits the whole CLAUDE_SPAWN_* namespace but arrives
    # with its OWN stdin session_id -- the only token that can tell it from
    # the session spawn-claude.sh launched. It must fall through to the
    # unchanged hand-launched keying instead of adopting the parent's slug.
    slug = 'session-cockpit-3215053'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215053,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'CLAUDE_SPAWN_SESSION_ID': slug,
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '2085',
    }

    forked = sh.hook_session_slug(hook_input, env, root=tmp_path)
    assert forked == sr.build_session_slug(
        'unblock',
        'df',
        '2085',
        'uuid-nested',  # type: ignore[arg-type]
    )
    assert forked != slug


def test_run_session_start_nested_inheritor_forks_leaving_parent_untouched(
    tmp_path: Path,
) -> None:
    # The blast radius: run_session_start overwrites status/parent_session_id/
    # display on whatever record the slug resolves to. A nested claude's
    # SessionStart must land on its OWN record, leaving the parent's byte-
    # identical.
    slug = 'session-cockpit-3215041'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            title='parent-title-marker',
            launcher_pid=3215041,
            parent_session_id='root-df-1-1',
            display=sr.Display(kind='wm', wm_title='parent-marker', wm_window_id='0x1a'),
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    snapshot = sr.read_record(slug, root=tmp_path).to_dict()

    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    # CLAUDE_SPAWN_PARENT_ID/CLAUDE_SPAWN_TITLE/WINDOWID are the concrete
    # overwrite vectors: under the pre-task-4193 code they all landed on the
    # parent's record.
    env = {
        'CLAUDE_SPAWN_SESSION_ID': slug,
        'CLAUDE_SPAWN_PARENT_ID': 'some-other-root',
        'CLAUDE_SPAWN_TITLE': 'nested-title',
        'WINDOWID': '0x99',
    }

    sh.run_session_start(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).to_dict() == snapshot

    forked_slug = sh.hook_session_slug(hook_input, env, root=tmp_path)
    assert forked_slug != slug
    forked = sr.read_record(forked_slug, root=tmp_path)
    assert forked.status == sr.Status.RUNNING
    assert forked.claude_session_id == 'uuid-nested'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 2


# ---------------------------------------------------------------------------
# Task 4193 step-3: Stop/Notification honour the record binding
# ---------------------------------------------------------------------------


def test_run_notification_and_stop_keep_writing_to_the_bound_record(
    tmp_path: Path,
) -> None:
    # The owning session's own events still converge on its record, question
    # capture included -- the ownership probe must not disturb the task-2511
    # convergence it gates.
    slug = 'session-cockpit-3215060'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215060,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-parent',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'approve rollout?',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_notification(hook_input, env, root=tmp_path)
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.AWAITING_INPUT
    assert record.question is not None
    assert record.question.text == 'approve rollout?'

    sh.run_stop(hook_input, env, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_run_stop_from_nested_inheritor_does_not_idle_the_parent(tmp_path: Path) -> None:
    # The sharpest statement of the bug: a nested claude finishing its turn
    # must not advertise the spawning session as idle mid-turn.
    slug = 'session-cockpit-3215061'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215061,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING
    forked_slug = sh.hook_session_slug(hook_input, env, root=tmp_path)
    assert forked_slug != slug
    forked = sr.read_record(forked_slug, root=tmp_path)
    assert forked.status == sr.Status.IDLE
    assert forked.claude_session_id == 'uuid-nested'


def test_run_notification_from_nested_inheritor_does_not_flip_the_parent(
    tmp_path: Path,
) -> None:
    slug = 'session-cockpit-3215062'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215062,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'cwd': '/home/leo/src/dark-factory',
        'message': 'nested asks something',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_notification(hook_input, env, root=tmp_path)

    parent = sr.read_record(slug, root=tmp_path)
    assert parent.status == sr.Status.RUNNING
    assert parent.question is None
    forked = sr.read_record(sh.hook_session_slug(hook_input, env, root=tmp_path), root=tmp_path)
    assert forked.status == sr.Status.AWAITING_INPUT
    assert forked.question is not None
    assert forked.question.text == 'nested asks something'


def test_refresh_path_binds_a_legacy_unbound_record(tmp_path: Path) -> None:
    # Migration safety: an in-flight session whose record predates task 4193
    # (or was written by spawn-claude.sh's `launching`) is bound by the very
    # next Notification/Stop from its PROVEN owner, without waiting for a
    # SessionStart that may never come again -- so the next nested event is
    # already discriminable.
    slug = 'session-cockpit-3215063'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.RUNNING, launcher_pid=3215063),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = owner_ppid_env(slug, 3215560)
    tree = claude_process_tree(3215561, 3215560)

    sh.run_stop(hook_input, env, root=tmp_path, process_tree=tree)

    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.IDLE
    assert record.claude_session_id == 'uuid-parent'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_refresh_path_leaves_an_unproven_legacy_record_unbound(tmp_path: Path) -> None:
    # esc-4193-10 / L2 ruling item 8, the deploy-day shape: every record
    # already live when this lands is unbound AND its session's env predates
    # CLAUDE_SPAWN_OWNER_PPID, so no event under that slug can prove
    # ownership. Adoption still happens (fail-soft, the pre-task-4193
    # collapse), but the record is left OPEN -- a nested `claude -p` must not
    # be able to claim it and exile the owner onto a degraded session-x row.
    slug = 'session-cockpit-3215065'
    sr.write_record(
        sr.SessionRecord(session_slug=slug, status=sr.Status.RUNNING, launcher_pid=3215065),
        root=tmp_path,
    )
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(
        {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'},
        env,
        root=tmp_path,
        process_tree=claude_process_tree(None),
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id is None
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_refresh_path_does_not_rebind_or_write_when_already_bound(tmp_path: Path) -> None:
    slug = 'session-cockpit-3215064'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215064,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-parent'


# ---------------------------------------------------------------------------
# Task 4193 step-4: ownership probe is fail-soft
# ---------------------------------------------------------------------------


def test_hook_session_slug_adopts_when_record_is_corrupt(tmp_path: Path) -> None:
    # A corrupt body must degrade to the pre-task-4193 behaviour (adopt), not
    # fork a spurious record; the reaper already has its own 'corrupt' rule.
    slug = 'session-cockpit-3215070'
    record_path = sr.record_path_for_slug(slug, root=tmp_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text('not-json{{{')
    with pytest.raises(sr.CorruptSessionRecord):
        sr.read_record(slug, root=tmp_path)

    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


def test_hook_session_slug_adopts_when_read_record_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    slug = 'session-cockpit-3215071'
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError('disk on fire')

    monkeypatch.setattr(sr, 'read_record', _boom)

    with caplog.at_level(logging.WARNING):
        assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug

    # Degradation is never silent (repo's no-silent-fail-soft norm).
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


def test_run_stop_fail_soft_when_probe_read_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end: a fault in the OWNERSHIP PROBE's read must never be what
    # breaks Stop. Scoped to the probe's own read (the first of the two
    # read_record calls run_stop makes: the probe reads, then
    # refresh_record does) because refresh_record deliberately PROPAGATES a
    # corrupt/unreadable body rather than overwriting it -- see its
    # docstring, "A *corrupt* existing body is NOT treated as absent" -- and
    # main()'s blanket except is the backstop for that pre-existing case.
    slug = 'session-cockpit-3215072'
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    real_read_record = sr.read_record
    calls: list[int] = []

    def _boom_once(*args: object, **kwargs: object) -> sr.SessionRecord:
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk on fire')
        return real_read_record(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _boom_once)

    assert sh.run_stop(hook_input, env, root=tmp_path).startswith('\033]0;')
    # The probe was consulted (and degraded) rather than skipped.
    assert len(calls) >= 1
    # Degrading to adopt means the env slug still owns the write.
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE


def test_main_session_start_fail_soft_when_probe_read_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The CLI backstop, scoped to the PROBE's own read (the first of the two
    # read_record calls session-start makes). Booby-trapping every read
    # instead would prove nothing: main()'s pre-existing blanket except
    # would return 0 even if the probe propagated, and the hook would be a
    # no-op rather than fail-soft. So the assertion here is the OUTCOME --
    # the degraded probe still let SessionStart complete its write.
    slug = 'session-cockpit-3215073'
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    monkeypatch.setenv('CLAUDE_SPAWN_SESSION_ID', slug)
    stdin_json(monkeypatch, {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'})

    real_read_record = sr.read_record
    calls: list[int] = []

    def _boom_once(*args: object, **kwargs: object) -> sr.SessionRecord:
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk on fire')
        return real_read_record(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(sr, 'read_record', _boom_once)

    assert sh.main(['session-start']) == 0

    # The probe was consulted (and degraded), and adopting the env slug
    # unchecked is exactly the pre-task-4193 behaviour: one record, written.
    assert len(calls) >= 1
    record = sr.read_record(slug, root=tmp_path)
    assert record.status == sr.Status.RUNNING
    # ...but a probe that could not answer must not CLAIM the record either:
    # adopting is fail-soft, binding is permanent (esc-4193-10).
    assert record.claude_session_id is None
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_hand_launched_session_unaffected_by_ownership_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A hand-launched session (no CLAUDE_SPAWN_SESSION_ID) has nothing to
    # probe and must pay no probe cost at all -- the must-not-be-called
    # idiom, not merely an assertion that the outcome is unchanged. Note
    # run_session_start legitimately calls sr.read_record itself, so it is
    # sh._env_slug_is_owned that is booby-trapped here, not the registry.
    def _boom(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError('ownership probe must not run for a hand-launched session')

    monkeypatch.setattr(sh, '_env_slug_is_owned', _boom)

    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}
    slug = sh.hook_session_slug(hook_input, env={}, root=tmp_path)

    sh.run_session_start(hook_input, env={}, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.RUNNING

    sh.run_notification(hook_input, env={}, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.AWAITING_INPUT

    sh.run_stop(hook_input, env={}, root=tmp_path)
    assert sr.read_record(slug, root=tmp_path).status == sr.Status.IDLE

    assert slug == sr.build_session_slug(
        'session',
        'dark-factory',
        None,
        'sess-hand',  # type: ignore[arg-type]
    )
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_hook_session_slug_adopts_when_stdin_has_no_session_id(tmp_path: Path) -> None:
    # With no discriminator the only safe answer is today's behaviour --
    # forking here would key every discriminator-less session on the one
    # 'unknown' literal hook_session_slug falls back to.
    slug = 'session-cockpit-3215074'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215074,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path) == slug


# ---------------------------------------------------------------------------
# Task 4193 amendment: one definition of a usable stdin session_id
# ---------------------------------------------------------------------------


def test_hook_session_id_normalizes_every_missing_shape() -> None:
    # The probe, the binding stamp and the slug builder must agree on what
    # counts as a discriminator, so all the "no session_id" shapes collapse
    # to the same empty string.
    assert sh._hook_session_id({}) == ''
    assert sh._hook_session_id({'session_id': None}) == ''
    assert sh._hook_session_id({'session_id': ''}) == ''
    assert sh._hook_session_id({'session_id': '   '}) == ''
    assert sh._hook_session_id({'session_id': ' uuid-x '}) == 'uuid-x'


def test_hook_session_slug_treats_a_blank_session_id_as_unknown(tmp_path: Path) -> None:
    # A whitespace-only stdin session_id used to reach build_session_slug
    # verbatim (no .strip()), yielding a slug with a blank token while the
    # two 4193 helpers already read it as "no discriminator at all".
    blank = sh.hook_session_slug({'session_id': '   ', 'cwd': '/home/leo/src/df'}, {}, root=tmp_path)
    absent = sh.hook_session_slug({'cwd': '/home/leo/src/df'}, {}, root=tmp_path)
    assert blank == absent
    assert blank.endswith('-unknown')


def test_bind_claude_session_id_ignores_a_blank_stdin_session_id() -> None:
    record = sr.SessionRecord(session_slug='s', status=sr.Status.LAUNCHING)
    probes = EventProbes({}, claude_process_tree(None))
    assert sh._bind_claude_session_id(record, {'session_id': '   '}, probes=probes) is False
    assert record.claude_session_id is None


# ---------------------------------------------------------------------------
# Task 4193 amendment: /clear re-mints session_id inside the OWNING process
# ---------------------------------------------------------------------------


def test_hook_session_slug_adopts_env_slug_when_a_clear_remints_the_session_id(
    tmp_path: Path,
) -> None:
    # /clear mints a new session_id in the SAME process, so the owning
    # session's own SessionStart legitimately mismatches its binding.
    # Reading that as an inheritor would re-introduce the task-2511 split.
    # "SAME process" is load-bearing and now checked: the bound
    # claude_owner_pid must match the pid resolved for this event.
    tree = claude_process_tree(424_242)
    slug = 'session-cockpit-3215080'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215080,
            claude_session_id='uuid-before-clear',
            claude_owner_pid=424_242,
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-after-clear',
        'source': 'clear',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert (
        sh.hook_session_slug(hook_input, env, root=tmp_path, allow_remint=True, process_tree=tree)
        == slug
    )
    # ... but only where a `source` field means anything. Without the
    # SessionStart-only opt-in the same event is a plain mismatch and forks.
    assert sh.hook_session_slug(hook_input, env, root=tmp_path, process_tree=tree) != slug


def test_hook_session_slug_does_not_forgive_a_resume_remint(tmp_path: Path) -> None:
    # `allow_remint` is not blanket forgiveness -- it is scoped to the two
    # sources with no CLI spelling. `--resume`/`--continue` make a brand-new
    # nested process report 'resume' too, so honouring it here would let any
    # inheritor claim its spawner's record even with the opt-in withheld.
    slug = 'session-cockpit-3215087'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215087,
            claude_session_id='uuid-before',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'source': 'resume',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    assert sh.hook_session_slug(hook_input, env, root=tmp_path, allow_remint=True) != slug
    assert sh.hook_session_slug(hook_input, env, root=tmp_path) != slug


@pytest.mark.parametrize('source', ['clear', 'compact'])
def test_run_session_start_rebinds_on_an_owner_only_source(
    tmp_path: Path,
    source: str,
) -> None:
    # ... and the record is RE-bound to the new id, so the session stays
    # discriminable from a nested claude on every subsequent event.
    #
    # claude_owner_pid is stamped to the SAME process this hook resolves,
    # which is what makes the re-mint provably the owner's. Source alone is
    # not enough: auto-compaction fires source='compact' unprompted, so a
    # nested claude reaches this path too and must NOT be forgiven -- see
    # orchestrator/tests/test_session_hooks_bash.py::test_nested_compact_cannot_invert_ownership.
    tree = claude_process_tree(424_242)
    slug = f'session-cockpit-321508{["clear", "compact"].index(source) + 1}'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215081,
            claude_session_id='uuid-before',
            claude_owner_pid=424_242,
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-after',
        'source': source,
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path, process_tree=tree)

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'uuid-after'
    assert record.status == sr.Status.RUNNING
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 1


def test_run_session_start_startup_source_with_mismatched_id_still_forks(
    tmp_path: Path,
) -> None:
    # The other direction: a nested claude is always a fresh process and so
    # always reports source='startup'. It must still fork.
    slug = 'session-cockpit-3215084'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215084,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'source': 'startup',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    assert sr.read_record(slug, root=tmp_path).claude_session_id == 'uuid-parent'
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 2


def test_run_session_start_resume_source_with_mismatched_id_still_forks(
    tmp_path: Path,
) -> None:
    # `resume` is NOT owner-only -- `--resume`/`--continue` make a brand-new
    # process report it, so a nested `claude -c -p ...` would otherwise
    # rebind the spawner's record to itself. It is deliberately excluded from
    # _OWNER_ONLY_SESSION_START_SOURCES; a spawned session /resume'd in place
    # simply forks, which is the fail-safe direction.
    slug = 'session-cockpit-3215086'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215086,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'source': 'resume',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_session_start(hook_input, env, root=tmp_path)

    parent = sr.read_record(slug, root=tmp_path)
    assert parent.claude_session_id == 'uuid-parent'
    assert parent.status == sr.Status.RUNNING
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 2


def test_refresh_path_ignores_a_forged_owner_only_source(tmp_path: Path) -> None:
    # Notification/Stop carry no `source` field, so nothing there can vouch
    # for the caller: the re-mint escape hatch is SessionStart-only. A
    # forged one on a Stop must not be honoured -- otherwise any inheritor
    # gets a one-word bypass of the whole ownership check.
    slug = 'session-cockpit-3215085'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215085,
            claude_session_id='uuid-parent',
        ),
        root=tmp_path,
    )
    hook_input = {
        'session_id': 'uuid-nested',
        'source': 'clear',
        'cwd': '/home/leo/src/dark-factory',
    }
    env = {'CLAUDE_SPAWN_SESSION_ID': slug}

    sh.run_stop(hook_input, env, root=tmp_path)

    parent = sr.read_record(slug, root=tmp_path)
    assert parent.claude_session_id == 'uuid-parent'
    assert parent.status == sr.Status.RUNNING
    assert len(list(sr.sessions_dir(root=tmp_path).iterdir())) == 2


# ---------------------------------------------------------------------------
# Task 4193 amendment: a forked record describes ITSELF, not its spawner
# ---------------------------------------------------------------------------


def test_forked_inheritor_records_its_spawner_as_parent(tmp_path: Path) -> None:
    # CLAUDE_SPAWN_PARENT_ID names the SPAWNER's OWN parent, so copying it
    # onto the fork would render the nested session as its spawner's
    # sibling. The fork path is the one place the true parent is known
    # exactly: it is the env slug that was just rejected.
    slug = 'session-cockpit-3215090'
    write_bound_parent(slug, tmp_path, 3215090)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_PARENT_ID': 'some-other-root'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    forked = sr.read_record(sh.hook_session_slug(hook_input, env, root=tmp_path), root=tmp_path)
    assert forked.parent_session_id == slug
    # And the spawning session's own record still points where it did.
    assert sr.read_record(slug, root=tmp_path).parent_session_id is None


def test_forked_inheritor_does_not_claim_the_spawners_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CLAUDE_SPAWN_WM_TITLE marks the SPAWNING session's window: resolving
    # it here would give the forked row the parent's window id, so cockpit
    # focus would raise the parent's terminal and
    # mark_windowless_wm_sessions_exited would treat both rows as one
    # window. The resolver must not even be consulted.
    def _boom(*_args: object, **_kwargs: object) -> str:
        raise AssertionError('a forked inheritor must not resolve the spawner window marker')

    monkeypatch.setattr(sh, '_resolve_wm_window_id', _boom)

    slug = 'session-cockpit-3215091'
    write_bound_parent(slug, tmp_path, 3215091)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_WM_TITLE': 'parent-marker'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    forked = sr.read_record(sh.hook_session_slug(hook_input, env, root=tmp_path), root=tmp_path)
    assert forked.display is None


def test_forked_inheritor_does_not_inherit_the_spawners_role_task_or_title(
    tmp_path: Path,
) -> None:
    # Task 4663 (task-4193-review residual): parent_session_id/display/
    # launcher_pid were already guarded above, but resolve_hook_identity
    # still read CLAUDE_SPAWN_ROLE/TASK_ID/ESCALATION_ID/TITLE straight out
    # of the inherited env, so a forked record got the SPAWNER's role,
    # task_id, escalation_id and display title -- two cockpit rows
    # identical except for slug. These must resolve to this session's own
    # non-spawn defaults instead.
    slug = 'session-cockpit-3215093'
    write_bound_parent(slug, tmp_path, 3215093)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'CLAUDE_SPAWN_SESSION_ID': slug,
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '9999',
        'CLAUDE_SPAWN_ESCALATION_ID': 'esc-9999-1',
        'CLAUDE_SPAWN_TITLE': 'unblock:dark-factory#9999 spawner-title',
    }

    sh.run_session_start(hook_input, env, root=tmp_path)

    forked = sr.read_record(sh.hook_session_slug(hook_input, env, root=tmp_path), root=tmp_path)
    assert forked.role == 'session'
    # 'dark-factory' is the cwd basename, NOT the inherited 'df': every
    # load-bearing key of _SPAWNER_IDENTITY_ENV_KEYS is set here with a value
    # that differs from the fall-through default, so dropping any one of them
    # from the frozenset reddens this test rather than silently putting the
    # spawner's identity back on the forked row.
    assert forked.project == 'dark-factory'
    assert forked.task_id is None
    assert forked.escalation_id is None
    assert forked.title == 'session:dark-factory'
    # And the spawning session's own record is untouched -- role/title
    # never carried spawn env in the first place (see write_bound_parent).
    spawner = sr.read_record(slug, root=tmp_path)
    assert spawner.role == ''
    assert spawner.title == ''


@pytest.mark.parametrize(
    ('handler', 'status'),
    [
        (sh.run_stop, sr.Status.IDLE),
        (sh.run_notification, sr.Status.AWAITING_INPUT),
    ],
    ids=['stop', 'notification'],
)
def test_forked_inheritor_retitle_describes_its_own_record(
    tmp_path: Path,
    handler: Callable[..., str],
    status: sr.Status,
) -> None:
    # The escape goes to the hook process's OWN controlling terminal, so it
    # describes the session whose hook fired; a Bash-tool-launched nested
    # claude cannot reach its spawner's tab at all (task 5421).
    slug = 'session-cockpit-3215094'
    write_bound_parent(slug, tmp_path, 3215094)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'CLAUDE_SPAWN_SESSION_ID': slug,
        'CLAUDE_SPAWN_TITLE': 'unblock:dark-factory#9999 spawner-title',
    }

    sh.run_session_start(hook_input, env, root=tmp_path)
    osc = handler(hook_input, env, root=tmp_path)

    forked = sr.read_record(sh.hook_session_slug(hook_input, env, root=tmp_path), root=tmp_path)
    assert forked.title == 'session:dark-factory'
    assert osc == sh.osc_retitle_sequence(status, forked.title)
    assert forked.status == status


def test_forked_inheritor_retitle_falls_back_to_its_own_identity(
    tmp_path: Path,
) -> None:
    # No SessionStart, so the forked record is synthesized with a blank
    # title and the retitle takes hook_display_title's identity fallback:
    # that identity must be this session's own, not its spawner's.
    slug = 'session-cockpit-3215095'
    write_bound_parent(slug, tmp_path, 3215095)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'CLAUDE_SPAWN_SESSION_ID': slug,
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '9999',
    }

    osc = sh.run_stop(hook_input, env, root=tmp_path)

    assert osc == sh.osc_retitle_sequence(sr.Status.IDLE, 'session:dark-factory')


def test_adopted_session_retitle_keeps_its_spawn_identity(tmp_path: Path) -> None:
    # The strip is fork-only: the owning session's blank-titled record still
    # retitles from its own spawn identity.
    slug = 'session-cockpit-3215096'
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.RUNNING,
            launcher_pid=3215096,
            claude_session_id='uuid-owner',
        ),
        root=tmp_path,
    )
    hook_input = {'session_id': 'uuid-owner', 'cwd': '/home/leo/src/dark-factory'}
    env = {
        'CLAUDE_SPAWN_SESSION_ID': slug,
        'CLAUDE_SPAWN_ROLE': 'unblock',
        'CLAUDE_SPAWN_PROJECT': 'df',
        'CLAUDE_SPAWN_TASK_ID': '2085',
    }

    osc = sh.run_stop(hook_input, env, root=tmp_path)

    assert osc == sh.osc_retitle_sequence(sr.Status.IDLE, 'unblock:df#2085')


def test_adopted_session_still_resolves_the_spawn_window_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The gate is fork-only: the owning session's own SessionStart keeps the
    # task-2510 / Cockpit C10 marker resolution untouched.
    monkeypatch.setattr(sh, '_resolve_wm_window_id', lambda _marker: '0x1a')

    slug = 'session-cockpit-3215092'
    write_bound_parent(slug, tmp_path, 3215092)
    hook_input = {'session_id': 'uuid-parent', 'cwd': '/home/leo/src/dark-factory'}
    env = {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_WM_TITLE': 'parent-marker'}

    sh.run_session_start(hook_input, env, root=tmp_path)

    display = sr.read_record(slug, root=tmp_path).display
    assert display is not None
    assert display.wm_title == 'parent-marker'
    assert display.wm_window_id == '0x1a'


# ---------------------------------------------------------------------------
# Task 4193 amendment: a forked record must stay reapable
# ---------------------------------------------------------------------------


def test_forked_inheritor_record_gets_a_pid_that_dies_with_it(tmp_path: Path) -> None:
    # The hand-launched liveness pid (the session leader) outlives every nested
    # claude, so a fork stamped with it would never satisfy
    # reap_stale_records' stale_pid rule -- one permanent extra row per
    # nested `claude -p` an agent shells out to. The injected tree's nested
    # claude IS the pid the forked record must carry.
    tree = claude_process_tree(424242)

    slug = 'session-cockpit-3215093'
    write_bound_parent(slug, tmp_path, 3215093)
    hook_input = {'session_id': 'uuid-nested', 'cwd': '/home/leo/src/dark-factory'}
    # An inherited CLAUDE_SPAWN_LAUNCHER_PID is the SPAWNER's, so it must
    # not win here either.
    env = {'CLAUDE_SPAWN_SESSION_ID': slug, 'CLAUDE_SPAWN_LAUNCHER_PID': '3215093'}

    sh.run_session_start(hook_input, env, root=tmp_path, process_tree=tree)

    forked_slug = sh.hook_session_slug(hook_input, env, root=tmp_path, process_tree=tree)
    forked = sr.read_record(forked_slug, root=tmp_path)
    assert forked.launcher_pid == 424242


def test_hand_launched_record_keeps_the_durable_session_leader_pid(tmp_path: Path) -> None:
    # The nested-claude pid is fork-only: a hand-launched session gets the
    # terminal-lifetime session leader. A 424242 here would mean the
    # fork-only resolver ran.
    tree = dataclasses.replace(claude_process_tree(424242), session_leader_pid=lambda: 515151)

    hook_input = {'session_id': 'sess-hand', 'cwd': '/home/leo/src/dark-factory'}
    sh.run_session_start(hook_input, env={}, root=tmp_path, process_tree=tree)

    slug = sh.hook_session_slug(hook_input, env={}, root=tmp_path, process_tree=tree)
    assert sr.read_record(slug, root=tmp_path).launcher_pid == 515151


# ---------------------------------------------------------------------------
# Task 8125: hooks stamp and judge process identities
# ---------------------------------------------------------------------------

_IDENTITY_CWD = '/home/leo/src/dark-factory'

# A pid virtually guaranteed dead on any host (the suite-wide idiom; see
# test_session_registry.py's own _DEAD_PID -- deliberately not imported).
_DEAD_PID = 2**31 - 1


@pytest.mark.parametrize('owner_resolved', [True, False], ids=['owner-resolved', 'owner-unresolved'])
def test_the_bind_stamps_the_owner_identity_with_the_owner_pid(
    tmp_path: Path, owner_resolved: bool
) -> None:
    owner_pid = os.getpid() if owner_resolved else None
    hook_input = {'session_id': 'sess-stamp', 'cwd': _IDENTITY_CWD}
    slug = sh.hook_session_slug(
        hook_input, env={}, root=tmp_path, process_tree=claude_process_tree(None)
    )

    sh.run_session_start(
        hook_input, env={}, root=tmp_path, process_tree=claude_process_tree(owner_pid)
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_owner_pid == owner_pid
    assert record.claude_owner_identity == (own_identity() if owner_resolved else None)


def test_a_new_record_stamps_the_launcher_identity(tmp_path: Path) -> None:
    tree = dataclasses.replace(claude_process_tree(424242), session_leader_pid=os.getpid)
    hook_input = {'session_id': 'sess-launcher', 'cwd': _IDENTITY_CWD}

    sh.run_session_start(hook_input, env={}, root=tmp_path, process_tree=tree)

    slug = sh.hook_session_slug(hook_input, env={}, root=tmp_path, process_tree=tree)
    record = sr.read_record(slug, root=tmp_path)
    assert record.launcher_pid == os.getpid()
    assert record.launcher_identity == own_identity()


_RECLAIM_EVENT = {'session_id': 'sess-reclaim', 'cwd': _IDENTITY_CWD}


def _write_owned_by_this_process(tmp_path: Path, recorded: ProcessIdentity) -> str:
    """Seed the record ``_RECLAIM_EVENT`` keys on, owned by this test process as *recorded* names it."""
    slug = sh.hook_session_slug(
        _RECLAIM_EVENT, env={}, root=tmp_path, process_tree=claude_process_tree(None)
    )
    sr.write_record(
        sr.SessionRecord(
            session_slug=slug,
            status=sr.Status.IDLE,
            cwd=_IDENTITY_CWD,
            claude_session_id=_RECLAIM_EVENT['session_id'],
            claude_owner_pid=os.getpid(),
            claude_owner_identity=recorded,
        ),
        root=tmp_path,
    )
    return slug


@pytest.mark.parametrize('variant', list(IdentityVariant))
def test_the_reclaim_judges_the_recorded_owner_by_its_identity(
    tmp_path: Path, variant: IdentityVariant
) -> None:
    recorded = own_identity(variant)
    slug = _write_owned_by_this_process(tmp_path, recorded)

    sh.run_stop(_RECLAIM_EVENT, {}, root=tmp_path, process_tree=claude_process_tree(_DEAD_PID))

    record = sr.read_record(slug, root=tmp_path)
    if variant.judged_dead:
        assert record.claude_owner_pid == _DEAD_PID
    else:
        assert record.claude_owner_pid == os.getpid()
        assert record.claude_owner_identity == recorded


@pytest.mark.parametrize('variant', list(IdentityVariant))
def test_an_owner_recorded_with_this_events_pid_is_re_stamped_iff_its_identity_is_judged_dead(
    tmp_path: Path, variant: IdentityVariant
) -> None:
    recorded = own_identity(variant)
    slug = _write_owned_by_this_process(tmp_path, recorded)

    sh.run_stop(_RECLAIM_EVENT, {}, root=tmp_path, process_tree=claude_process_tree(os.getpid()))

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_owner_pid == os.getpid()
    assert record.claude_owner_identity == (own_identity() if variant.judged_dead else recorded)


# ---------------------------------------------------------------------------
# Task 8124: T3CODE_THREAD_ID binding
# ---------------------------------------------------------------------------

_THREAD = '9b2d4f6a-8124-4c3e-9a5b-7d1f3e5a7c90'
_T3_ENV = {'T3CODE_THREAD_ID': _THREAD}
_T3_SPAWN_SLUG = 'unblock-df-8124-4242'
_T3_CLAUDE_PID = 424242


def _t3_spawn_record(
    tmp_path: Path,
    *,
    session_id: str | None = None,
    owner_pid: int | None = None,
    owner_identity: ProcessIdentity | None = None,
) -> str:
    """A running t3 spawn record of ``_THREAD`` (result_file set), indexed by its thread pointer."""
    record_dir = sr.record_path_for_slug(_T3_SPAWN_SLUG, root=tmp_path).parent
    sr.write_record(
        sr.SessionRecord(
            session_slug=_T3_SPAWN_SLUG,
            status=sr.Status.RUNNING,
            title='unblock:df#8124',
            role='unblock',
            project='df',
            task_id='8124',
            cwd=_IDENTITY_CWD,
            launcher_pid=4242,
            result_file=str(record_dir / 'result.md'),
            display=sr.Display(
                kind=sr.DisplayKind.T3.value, wm_title='unblock:df#8124', t3_thread_id=_THREAD
            ),
            claude_session_id=session_id,
            claude_launch_session_id=session_id,
            claude_owner_pid=owner_pid,
            claude_owner_identity=owner_identity,
        ),
        root=tmp_path,
    )
    assert sr.write_t3_thread_pointer(_THREAD, _T3_SPAWN_SLUG, root=tmp_path)
    return _T3_SPAWN_SLUG


def _hand_slug(hook_input: dict[str, str], tmp_path: Path) -> str:
    """The hand-launched slug *hook_input*'s session_id keys on."""
    return sh.hook_session_slug(
        hook_input, env={}, root=tmp_path, process_tree=claude_process_tree(None)
    )


def _record_slugs(tmp_path: Path) -> list[str]:
    return sorted(path.name for path in sr.sessions_dir(tmp_path).iterdir())


def test_a_thread_pointer_adopts_and_binds_an_unbound_spawn_record(tmp_path: Path) -> None:
    slug = _t3_spawn_record(tmp_path)
    hook_input = {'session_id': 'sess-t3', 'cwd': _IDENTITY_CWD}

    sh.run_session_start(
        hook_input, _T3_ENV, root=tmp_path, process_tree=claude_process_tree(_T3_CLAUDE_PID)
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'sess-t3'
    assert record.claude_owner_pid == _T3_CLAUDE_PID
    assert sr.resolve_session_slug_for_claude_session_id('sess-t3', root=tmp_path) == slug
    assert sr.resolve_session_slug_for_pid(_T3_CLAUDE_PID, root=tmp_path) == slug
    assert _record_slugs(tmp_path) == [slug]
    assert record.display is not None
    assert (record.display.kind, record.display.t3_thread_id) == (sr.DisplayKind.T3, _THREAD)


@pytest.mark.parametrize(
    ('recorded_session_id', 'event_pid', 'source', 'adopted', 'bound_after'),
    [
        pytest.param(None, _T3_CLAUDE_PID, 'startup', False, None, id='unbound-another-live-owner-forks'),
        pytest.param(None, None, 'startup', True, None, id='unbound-unresolvable-pid-adopts-unbound'),
        pytest.param('sess-owner', os.getpid(), 'clear', True, 'sess-event', id='owner-re-mint-rebinds'),
    ],
)
def test_the_thread_lane_decides_adopt_or_fork_by_the_claude_pids(
    tmp_path: Path,
    recorded_session_id: str | None,
    event_pid: int | None,
    source: str,
    adopted: bool,
    bound_after: str | None,
) -> None:
    slug = _t3_spawn_record(
        tmp_path,
        session_id=recorded_session_id,
        owner_pid=os.getpid(),
        owner_identity=own_identity(),
    )
    event = {'session_id': 'sess-event', 'cwd': _IDENTITY_CWD, 'source': source}

    sh.run_session_start(
        event, _T3_ENV, root=tmp_path, process_tree=claude_process_tree(event_pid)
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == bound_after
    assert record.claude_launch_session_id == recorded_session_id
    assert (_record_slugs(tmp_path) == [slug]) is adopted
    assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) == slug


def test_a_resume_reaching_its_bound_record_by_the_thread_pointer_reclaims_a_dead_owner(
    tmp_path: Path,
) -> None:
    """Without the reclaim, the next nested claude would find a dead owner and take the record over."""
    slug = _t3_spawn_record(tmp_path, session_id='sess-owner', owner_pid=_DEAD_PID)
    resumed = {'session_id': 'sess-owner', 'cwd': _IDENTITY_CWD, 'source': 'resume'}
    nested = {'session_id': 'sess-nested', 'cwd': _IDENTITY_CWD, 'source': 'startup'}

    sh.run_session_start(
        resumed, _T3_ENV, root=tmp_path, process_tree=claude_process_tree(os.getpid())
    )
    sh.run_session_start(
        nested, _T3_ENV, root=tmp_path, process_tree=claude_process_tree(_T3_CLAUDE_PID)
    )

    record = sr.read_record(slug, root=tmp_path)
    assert (record.claude_session_id, record.claude_owner_pid) == ('sess-owner', os.getpid())
    assert record.claude_owner_identity == own_identity()
    assert sr.resolve_session_slug_for_claude_session_id('sess-owner', root=tmp_path) == slug
    assert _hand_slug(nested, tmp_path) in _record_slugs(tmp_path)


@pytest.mark.parametrize(
    'thread_id',
    [_THREAD, f'mcp:{_THREAD}', 'mcp:launch:' + 'a1b2c3d4' * 8],
    ids=['ui-uuid', 'mcp-uuid', 'mcp-launch'],
)
def test_an_unheld_thread_is_claimed_by_the_hand_launched_record(
    tmp_path: Path, thread_id: str
) -> None:
    hook_input = {'session_id': 'sess-hand', 'cwd': _IDENTITY_CWD}
    leaked_terminal = {'TMUX': '/tmp/tmux-1000/default,1,0', 'TMUX_PANE': '%3'}

    sh.run_session_start(
        hook_input,
        {'T3CODE_THREAD_ID': thread_id, **leaked_terminal},
        root=tmp_path,
        process_tree=claude_process_tree(_T3_CLAUDE_PID),
    )

    slug = _hand_slug(hook_input, tmp_path)
    record = sr.read_record(slug, root=tmp_path)
    assert sr.is_t3_record(record)
    assert record.display is not None and record.display.t3_thread_id == thread_id
    assert sr.resolve_session_slug_for_t3_thread(thread_id, root=tmp_path) == slug


def test_a_thread_id_that_cannot_name_a_pointer_is_never_claimed(tmp_path: Path) -> None:
    hook_input = {'session_id': 'sess-hand', 'cwd': _IDENTITY_CWD}

    sh.run_session_start(
        hook_input,
        {'T3CODE_THREAD_ID': '../escape'},
        root=tmp_path,
        process_tree=claude_process_tree(_T3_CLAUDE_PID),
    )

    record = sr.read_record(_hand_slug(hook_input, tmp_path), root=tmp_path)
    assert not sr.is_t3_record(record)
    assert not sr.t3_thread_pointers_dir(tmp_path).exists()


def test_a_nested_claude_in_a_bound_thread_forks_without_claiming_it(tmp_path: Path) -> None:
    slug = _t3_spawn_record(
        tmp_path, session_id='sess-owner', owner_pid=os.getpid(), owner_identity=own_identity()
    )
    nested = {'session_id': 'sess-nested', 'cwd': _IDENTITY_CWD}

    sh.run_session_start(
        nested, _T3_ENV, root=tmp_path, process_tree=claude_process_tree(_T3_CLAUDE_PID)
    )

    spawn = sr.read_record(slug, root=tmp_path)
    assert (spawn.claude_session_id, spawn.claude_owner_pid) == ('sess-owner', os.getpid())
    nested_slug = _hand_slug(nested, tmp_path)
    assert _record_slugs(tmp_path) == sorted([slug, nested_slug])
    assert not sr.is_t3_record(sr.read_record(nested_slug, root=tmp_path))
    assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) == slug


def _dead_owner(kind: str) -> tuple[int, ProcessIdentity | None]:
    if kind == 'dead-pid':
        return _DEAD_PID, None
    return os.getpid(), own_identity(IdentityVariant.REUSED_PID)


_DEAD_OWNER_KINDS = ['dead-pid', 'reused-pid']


@pytest.mark.parametrize('dead_owner', _DEAD_OWNER_KINDS)
def test_a_fresh_session_start_after_the_owner_died_rebinds_the_thread_record(
    tmp_path: Path, dead_owner: str
) -> None:
    owner_pid, owner_identity = _dead_owner(dead_owner)
    slug = _t3_spawn_record(
        tmp_path, session_id='sess-owner', owner_pid=owner_pid, owner_identity=owner_identity
    )
    resumed = {'session_id': 'sess-resumed', 'cwd': _IDENTITY_CWD, 'source': 'startup'}

    sh.run_session_start(
        resumed, _T3_ENV, root=tmp_path, process_tree=claude_process_tree(_T3_CLAUDE_PID)
    )

    record = sr.read_record(slug, root=tmp_path)
    assert record.claude_session_id == 'sess-resumed'
    assert record.claude_launch_session_id == 'sess-owner'
    assert record.claude_owner_pid == _T3_CLAUDE_PID
    assert sr.resolve_session_slug_for_claude_session_id('sess-resumed', root=tmp_path) == slug
    assert _record_slugs(tmp_path) == [slug]


@pytest.mark.parametrize('dead_owner', _DEAD_OWNER_KINDS)
def test_only_session_start_rebinds_a_thread_record_whose_owner_died(
    tmp_path: Path, dead_owner: str
) -> None:
    owner_pid, owner_identity = _dead_owner(dead_owner)
    slug = _t3_spawn_record(
        tmp_path, session_id='sess-owner', owner_pid=owner_pid, owner_identity=owner_identity
    )
    resumed = {'session_id': 'sess-resumed', 'cwd': _IDENTITY_CWD}

    sh.run_notification(
        resumed, _T3_ENV, root=tmp_path, process_tree=claude_process_tree(_T3_CLAUDE_PID)
    )

    record = sr.read_record(slug, root=tmp_path)
    assert (record.claude_session_id, record.claude_owner_pid) == ('sess-owner', owner_pid)
    assert record.status == sr.Status.RUNNING
    assert _hand_slug(resumed, tmp_path) in _record_slugs(tmp_path)
    assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) == slug


@pytest.mark.parametrize('owned', [True, False], ids=['owned-spawn', 'nested-in-a-spawn'])
def test_an_event_keyed_by_a_spawn_session_id_never_touches_the_thread_pointer(
    tmp_path: Path, owned: bool
) -> None:
    tmux_slug = 'worker-df-8124-77'
    if owned:
        sr.write_record(
            sr.SessionRecord(session_slug=tmux_slug, status=sr.Status.LAUNCHING, launcher_pid=77),
            root=tmp_path,
        )
    else:
        write_bound_parent(tmux_slug, tmp_path, 77)
    hook_input = {'session_id': 'sess-tmux', 'cwd': _IDENTITY_CWD}
    env = {**owner_ppid_env(tmux_slug, 4300), **_T3_ENV}

    sh.run_session_start(
        hook_input, env, root=tmp_path, process_tree=claude_process_tree(_T3_CLAUDE_PID, 4300)
    )

    written = tmux_slug if owned else _hand_slug(hook_input, tmp_path)
    assert sr.read_record(written, root=tmp_path).claude_session_id == 'sess-tmux'
    assert not sr.is_t3_record(sr.read_record(written, root=tmp_path))
    assert not sr.t3_thread_pointers_dir(tmp_path).exists()


def test_a_hand_t3_record_regains_its_lost_thread_pointer_on_its_next_event(
    tmp_path: Path,
) -> None:
    hook_input = {'session_id': 'sess-hand', 'cwd': _IDENTITY_CWD}
    tree = claude_process_tree(_T3_CLAUDE_PID)
    sh.run_session_start(hook_input, _T3_ENV, root=tmp_path, process_tree=tree)
    sr.t3_thread_pointer_path(_THREAD, tmp_path).unlink()

    sh.run_stop(hook_input, _T3_ENV, root=tmp_path, process_tree=tree)

    assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) == _hand_slug(
        hook_input, tmp_path
    )


def test_a_hook_event_never_moves_a_thread_pointer_another_live_record_holds(
    tmp_path: Path,
) -> None:
    hand_input = {'session_id': 'sess-hand', 'cwd': _IDENTITY_CWD}
    tree = claude_process_tree(_T3_CLAUDE_PID)
    sh.run_session_start(hand_input, _T3_ENV, root=tmp_path, process_tree=tree)
    spawn_slug = _t3_spawn_record(tmp_path)

    sh.run_notification(hand_input, _T3_ENV, root=tmp_path, process_tree=tree)

    assert sr.is_t3_record(sr.read_record(_hand_slug(hand_input, tmp_path), root=tmp_path))
    assert sr.resolve_session_slug_for_t3_thread(_THREAD, root=tmp_path) == spawn_slug
