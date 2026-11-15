"""Tests for the write-decision verb, driven where the watcher SKILLs call it: ``session_registry.py write-decision``."""

from __future__ import annotations

import argparse
import contextlib
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from _decision_fixtures import (
    EVIDENCE,
    claims_zero_matches,
    file_decision,
    file_from_a_linked_session,
    make_decision,
    names_the_destination_token,
    qid,
    two_queues,
)

from orchestrator import decision_cli
from orchestrator import decision_registry as dr
from orchestrator import session_registry as sr

# ---------------------------------------------------------------------------
# Step-6b: update_decision_state / set_manual_boost serialize their RMW via
# decision_id_lock (task 2427)
# ---------------------------------------------------------------------------


class TestDecisionHelpersAdoptLock:
    """Spy tests (mirror TestSubmitResolveAdoptLock, test_queue.py:2913):
    update_decision_state, set_manual_boost, set_decision_escalations_dir and
    the write-decision verb must EACH acquire decision_id_lock for the
    correct decision id.

    One case per public caller, deliberately, even though the three setters
    now share _mutate_decision: the lock is a per-caller CONTRACT, and a
    setter that grows its own body (or a wrapper that mutates before
    delegating) would slip past a single shared-implementation test. The
    write-decision verb is precisely such a caller -- it does NOT go through
    _mutate_decision at all (it is an upsert, not a strict RMW), so nothing
    but this spy would notice its lock disappearing. TestDecisionIdLock
    covers the lock primitive itself; these cover its ADOPTION.
    """

    def test_main_write_decision_acquires_lock_for_decision_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The write-decision verb is the FOURTH writer on a decision id --
        and the only one that may CREATE the record rather than only mutate
        an existing one.

        Asserted on the CREATE path deliberately: two watchers can file the
        same id concurrently with nothing on disk yet, so the span needs the
        lock even when there is no existing record to merge with. It is also
        the path a reader is most likely to assume needs no locking.
        """
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        real_lock = dr.decision_id_lock
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            with real_lock(decision_id, root=root):
                yield

        monkeypatch.setattr(dr, 'decision_id_lock', recording_lock)

        rc = sr.main(
            [
                'write-decision',
                '--id',
                'dec-spy-4',
                '--project',
                'df',
                '--text',
                'q',
                '--escalations-dir',
                str(tmp_path / 'escalations'),
            ]
        )

        assert rc == 0
        assert qid('dec-spy-4') in acquired, (
            f'Expected lock acquisition for {qid("dec-spy-4")}; got {acquired}'
        )

    def test_main_write_decision_refusal_takes_no_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A REFUSED invocation must not take the lock.

        decision_id_lock's own docstring flags ORPHAN SIDECARS: the
        ``<id>.json.lock`` file is created on acquisition and never cleaned
        up, so locking before the stamp guards would litter the decisions
        dir with sidecars for ids that are never written -- one per
        mis-invocation, forever. The guards therefore run FIRST.
        """
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            yield

        monkeypatch.setattr(dr, 'decision_id_lock', recording_lock)

        rc = sr.main(
            ['write-decision', '--id', 'dec-spy-5', '--project', 'df', '--text', 'q',
             '--escalations-dir', '']
        )

        assert rc == 0
        assert acquired == []


@pytest.mark.timeout(30)
def test_main_write_decision_enrichment_span_is_serialized_per_decision_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write-decision verb's read->merge->write span must serialize too.

    Task 3559 turned _run_write_decision from a blind single write into a
    read-then-write, which is exactly the hazard decision_id_lock exists for
    and which update_decision_state / set_manual_boost /
    set_decision_escalations_dir already name in their Concurrency NOTEs. It
    is a REAL race, not a theoretical one: the whole point of the enrichment
    branch is that two watchers on two queues file the same id, and nothing
    stops them doing so at the same moment.

    Same deterministic in-process idiom as
    test_concurrent_state_and_boost_updates_do_not_lose_a_field: a fixed
    delay is injected into the module-level write_decision seam -- AFTER
    each call's read, BEFORE its write -- so both threads read the same
    pre-merge snapshot before either writes.

    Without the lock, the later os.replace() clobbers the earlier call's
    enrichment: each thread's merged record carries only ITS OWN
    contribution (task_id or session_id, never both), so whichever writes
    last silently drops the other watcher's information -- the very
    lost-update this task exists to stop, reintroduced one level up. With
    the lock, the second span re-reads the first's persisted record and
    merges on top, so both contributions survive and the severity is the max
    across all three filings rather than whichever landed last.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)
    third = tmp_path / 'third'
    third.mkdir()
    dr.write_decision(
        make_decision(
            id=qid('esc-race-1'),
            project='df',
            text='Adopt the reify plan?',
            state=dr.DecisionState.OPEN,
            severity='info',
            task_id=None,
            session_id=None,
            escalations_dir=dr.normalize_escalations_dir(orch),
        ),
        root=tmp_path,
    )

    real_write_decision = dr.write_decision

    def delayed_write_decision(record: dr.DecisionRecord, root: Path | str | None = None) -> bool:
        time.sleep(0.3)
        return real_write_decision(record, root=root)

    monkeypatch.setattr(dr, 'write_decision', delayed_write_decision)

    def file_from(queue: Path, **extra: str) -> None:
        file_decision(id='esc-race-1', project='df', text='reify?', escalations_dir=str(queue), **extra)

    t1 = threading.Thread(target=file_from, args=(recon,), kwargs={'task_id': '5914', 'severity': 'critical'})
    t2 = threading.Thread(target=file_from, args=(third,), kwargs={'session_id': 'watcher-3', 'severity': 'urgent'})

    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not t1.is_alive(), 'recon-queue write-decision thread did not finish in time'
    assert not t2.is_alive(), 'third-queue write-decision thread did not finish in time'

    [reread] = [d for d in dr.list_decisions(root=tmp_path) if d.id == qid('esc-race-1')]
    # Neither filer's contribution was dropped...
    assert reread.task_id == '5914'
    assert reread.session_id == 'watcher-3'
    assert reread.severity == 'urgent'
    # ...and the first filer's fields are still intact.
    assert reread.text == 'Adopt the reify plan?'
    assert reread.escalations_dir == dr.normalize_escalations_dir(orch)


# ---------------------------------------------------------------------------
# CLI write-decision verb (Fleet Cockpit C8: park-to-registry for watchers)
# ---------------------------------------------------------------------------


def test_main_write_decision_files_open_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'dec-park-1',
            '--project',
            'df',
            '--text',
            'Approve risky merge?',
            '--task-id',
            '2085',
            '--escalation-id',
            'esc-1',
            '--session-id',
            'watcher-df-99',
            '--escalations-dir',
            str(tmp_path / 'escalations'),
        ]
    )

    assert rc == 0
    listed = dr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    rec = listed[0]
    assert rec.id == qid('dec-park-1')
    # Stored CANONICAL, not verbatim: --project is normalized at the CLI
    # boundary (task 3807, see test_main_write_decision_canonicalizes_project).
    assert rec.project == 'dark_factory'
    assert rec.text == 'Approve risky merge?'
    assert rec.task_id == '2085'
    assert rec.escalation_id == 'esc-1'
    assert rec.session_id == 'watcher-df-99'
    assert rec.state == dr.DecisionState.OPEN
    assert rec.filed_at != ''


def test_main_write_decision_files_under_the_project_qualified_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--id is project-LOCAL: the stored, printed id is ``<canonical project>-<--id>``
    (task 4835), so one escalation id in two projects can never share a row.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'esc-42-1',
            '--project',
            'df',
            '--text',
            'q?',
            '--escalations-dir',
            str(tmp_path / 'escalations'),
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out.strip() == 'dark_factory-esc-42-1'
    assert [d.id for d in dr.list_decisions(root=tmp_path)] == ['dark_factory-esc-42-1']
    assert dr.decision_path_for_id('dark_factory-esc-42-1', root=tmp_path).is_file()
    assert not dr.decision_path_for_id('esc-42-1', root=tmp_path).exists()


@pytest.mark.parametrize(
    'raw_project',
    ['df', 'DF', 'dark-factory', 'Dark-Factory', '  DARK_FACTORY '],
)
def test_main_write_decision_canonicalizes_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raw_project: str,
) -> None:
    """--project is stored NORMALIZED, not verbatim (task 3807).

    Mirrors how --escalations-dir is already stamped normalized one line
    below in _run_write_decision. Storing the raw argv string is what let one
    project's decisions accumulate in three separate partitions, each
    invisible to a reap scoped to either of the others.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'd-alias',
            '--project',
            raw_project,
            '--text',
            'q?',
            '--escalations-dir',
            str(tmp_path / 'escalations'),
        ]
    )

    assert rc == 0
    listed = dr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].project == 'dark_factory'


def test_main_write_decision_canonicalizes_an_unaliased_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The fold applies to EVERY project, not just the one aliased token: a
    genuinely new project must be able to file without a code change, so the
    verb normalizes rather than rejecting an unrecognized spelling.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'd-av',
            '--project',
            'autopilot-video',
            '--text',
            'q?',
            '--escalations-dir',
            str(tmp_path / 'escalations'),
        ]
    )

    assert rc == 0
    listed = dr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].project == 'autopilot_video'


def test_main_write_decision_qualifies_the_id_by_the_canonical_project_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The stored id is ``<canonical project>-<--id>`` with --id carried
    byte-for-byte (task 4835): the prefix comes from the FOLDED --project, so
    a raw ``df`` never leaks into the id, and --id is never parsed or
    rewritten -- not even a legacy hand-typed ``df-`` prefix on it.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'df-esc-3524-1',
            '--project',
            'df',
            '--text',
            'q?',
            '--escalations-dir',
            str(tmp_path / 'escalations'),
        ]
    )

    assert rc == 0
    qualified = 'dark_factory-df-esc-3524-1'
    assert (tmp_path / 'decisions' / f'{qualified}.json').is_file()
    assert capsys.readouterr().out.strip() == qualified
    listed = dr.list_decisions(root=tmp_path)
    assert [(d.id, d.project) for d in listed] == [(qualified, 'dark_factory')]


def test_main_write_decision_logs_a_project_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A rewrite is LOUD, not silent (project norm: loud-over-silent
    degradation) -- so an operator can see that what they typed is not what
    was stored, and fix the spelling at the source.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    with caplog.at_level(logging.WARNING):
        assert (
            sr.main(
                [
                    'write-decision',
                    '--id',
                    'd-w',
                    '--project',
                    'df',
                    '--text',
                    'q?',
                    '--escalations-dir',
                    str(tmp_path / 'escalations'),
                ]
            )
            == 0
        )

    assert any(
        r.levelno >= logging.WARNING and 'df' in r.getMessage() and 'dark_factory' in r.getMessage()
        for r in caplog.records
    )


def test_main_write_decision_already_canonical_project_logs_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning fires only on an actual REWRITE. A watcher already passing
    the canonical token -- which is what both SKILL.md files now tell it to
    do -- must not emit a warning on every park, or the signal is noise.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    file_from_a_linked_session(monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING):
        rc = sr.main(
            [
                'write-decision',
                '--id',
                'd-ok',
                '--project',
                'dark_factory',
                '--text',
                'q?',
                '--escalations-dir',
                str(tmp_path / 'escalations'),
            ]
        )

    assert rc == 0
    assert dr.list_decisions(root=tmp_path)[0].project == 'dark_factory'
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_main_write_decision_warns_on_a_declined_alias_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Passing a DECLINED alias target warns, and changes nothing else.

    This is the load-bearing pair of assertions for task 3813: the record is
    still filed under EXACTLY what the caller passed (we warn, we do not
    silently move another project's rows -- rewriting here would be the very
    cross-project behaviour change the task declined), and the return code
    is unaffected (advisory, never a refusal -- contrast the two hard
    refusals in _run_write_decision, which return without filing).
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    with caplog.at_level(logging.WARNING):
        rc = sr.main(
            [
                'write-decision',
                '--id',
                'd-declined',
                '--project',
                'my_solar_challenge',
                '--text',
                'q?',
                '--escalations-dir',
                str(tmp_path / 'escalations'),
            ]
        )

    assert rc == 0
    # NOT rewritten: filed under exactly the token the caller passed.
    assert dr.list_decisions(root=tmp_path)[0].project == 'my_solar_challenge'
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    # Non-vacuous on BOTH tokens: see names_the_destination_token.
    assert any('my_solar_challenge' in m and names_the_destination_token(m) for m in warnings)
    # ...and the message must be TRUE on this path. write-decision CREATES;
    # it matches nothing by definition, and one line after this warning it
    # files a row under this very token. A "matches no decisions" line here
    # would be false the moment it is acted on.
    assert not any(claims_zero_matches(m) for m in warnings)


def test_main_write_decision_recommended_solar_token_warns_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The token the skills recommend must stay silent.

    Same strict no-noise assertion as
    test_main_write_decision_already_canonical_project_logs_nothing: a
    watcher following the documented ``--project solar_challenge`` guidance
    must not be warned on every park, or the new hint is noise rather than
    signal.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    file_from_a_linked_session(monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING):
        rc = sr.main(
            [
                'write-decision',
                '--id',
                'd-solar-ok',
                '--project',
                'solar_challenge',
                '--text',
                'q?',
                '--escalations-dir',
                str(tmp_path / 'escalations'),
            ]
        )

    assert rc == 0
    assert dr.list_decisions(root=tmp_path)[0].project == 'solar_challenge'
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_main_write_decision_stamps_severity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A watcher's park moment supplies the escalation's severity so the
    cockpit decision queue can weight this ask (Fleet Cockpit F7 fix 1).
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'dec-sev',
            '--project',
            'df',
            '--text',
            'q',
            '--severity',
            'critical',
            '--escalations-dir',
            str(tmp_path / 'escalations'),
        ]
    )

    assert rc == 0
    listed = dr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].severity == 'critical'


def test_main_write_decision_severity_defaults_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omitting --severity yields severity='' on the filed record (a watcher
    that doesn't know/supply a severity still files a well-formed decision).
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'dec-nosev',
            '--project',
            'df',
            '--text',
            'q',
            '--escalations-dir',
            str(tmp_path / 'escalations'),
        ]
    )

    assert rc == 0
    listed = dr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].severity == ''


def test_main_write_decision_stamps_escalations_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A watcher stamps the queue its --escalation-id belongs to, so the
    fleet-global decision can later be joined back to the right per-queue
    escalation-id namespace (task 3528). The verb must store the NORMALIZED
    form, not the raw argv string: a watcher invoking with a relative/dotted
    spelling must still match a reaper passing the absolute one.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    recon = tmp_path / 'recon'
    recon.mkdir()
    (tmp_path / 'sub').mkdir()
    dotted = str(tmp_path / 'sub' / '..' / 'recon') + '/'

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'dec-q',
            '--project',
            'df',
            '--text',
            'q',
            '--escalation-id',
            'esc-1',
            '--escalations-dir',
            dotted,
        ]
    )

    assert rc == 0
    listed = dr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].escalations_dir == dr.normalize_escalations_dir(recon)
    assert Path(listed[0].escalations_dir).is_absolute()


def test_main_write_decision_refuses_when_escalations_dir_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Omitting --escalations-dir is an INVOCATION error, not a default.

    DecisionRecords are fleet-global (``~/.claude/fleet/decisions/``) but an
    ``esc-<taskid>-<n>`` id is unique only WITHIN one queue, and a project
    may run several (task 3528). A record filed without its queue stamp is
    therefore cross-queue-ambiguous -- exactly the legacy population task
    3640 had to back-fill out of, and which must not be allowed to regrow
    through this verb. So the flag is ``required=True``, joining --id /
    --project / --text on this same subparser (and mirroring the sibling
    ``reap-decisions``, whose --escalations-dir has always been required):
    argparse exits 2 and NOTHING is filed.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    with pytest.raises(SystemExit) as excinfo:
        sr.main(['write-decision', '--id', 'dec-noq', '--project', 'df', '--text', 'q'])

    assert excinfo.value.code == 2
    assert '--escalations-dir' in capsys.readouterr().err
    assert dr.list_decisions(root=tmp_path) == []


@pytest.mark.parametrize('blank', ['', '   '])
def test_main_write_decision_refuses_an_explicitly_empty_escalations_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    blank: str,
) -> None:
    """argparse's required=True cannot see the loophole this closes.

    ``--escalations-dir ''`` (or whitespace-only) SATISFIES a required flag
    and would still produce the unstamped, cross-queue-ambiguous record the
    verb now exists to refuse -- so the legacy population task 3640
    back-filled would simply regrow through the loophole. The verb
    therefore guards on the NORMALIZED stamp too, using the same
    normalize_escalations_dir the reaper compares with (it already collapses
    empty and whitespace-only to '').

    Shape is this module's loud-but-fail-soft CLI idiom, matching
    test_main_write_decision_fail_soft_when_fleet_root_under_a_file: rc 0
    (filing a decision can never crash a watcher's watch loop), an ERROR
    log, NOTHING written, and the id NOT echoed on stdout -- the absent id
    is itself the agent-visible failure signal, since both SKILL.md files
    tell the watcher to cross-link that printed id into its digest line.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    with caplog.at_level(logging.ERROR):
        rc = sr.main(
            [
                'write-decision',
                '--id',
                'dec-blankq',
                '--project',
                'df',
                '--text',
                'q',
                '--escalations-dir',
                blank,
            ]
        )

    assert rc == 0
    assert dr.list_decisions(root=tmp_path) == []
    assert 'dec-blankq' not in capsys.readouterr().out
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors
    assert any('dec-blankq' in r.getMessage() for r in errors)


def test_main_write_decision_refuses_the_unknown_queue_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``<unknown>`` is a BACK-FILL-only sentinel and is refused at this verb.

    skills/escalation-watcher/SKILL.md states the contract outright of the
    sentinel: "You never write it; task 3640's back-fill did". The reason it
    must be ENFORCED here rather than merely documented is that an
    UNKNOWN_QUEUE-stamped record is refused by EVERY reaper (the by-name
    guard in _run_reap_decisions._status), so a watcher filing one creates a
    record that only a human can ever close -- a regrowth strictly WORSE
    than the unstamped '' population this task outlaws, which at least still
    closes under project-only scoping.

    There is no legitimate write-path caller: a watcher always knows its own
    queue by construction, since it is the same dir it passes to
    reap-decisions. The sentinel stays fully valid on the BACK-FILL path via
    set_decision_escalations_dir, which is where task 3640 put it; this
    refusal is scoped to the CLI verb alone.

    Gets its OWN error message, distinct from the empty-stamp refusal, so
    the log tells a watcher author which of the two mistakes they made.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    with caplog.at_level(logging.ERROR):
        rc = sr.main(
            [
                'write-decision',
                '--id',
                'dec-unkq',
                '--project',
                'df',
                '--text',
                'q',
                '--escalations-dir',
                dr.UNKNOWN_QUEUE,
            ]
        )

    assert rc == 0
    assert dr.list_decisions(root=tmp_path) == []
    assert 'dec-unkq' not in capsys.readouterr().out
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors
    assert any(dr.UNKNOWN_QUEUE in r.getMessage() for r in errors)
    assert any('dec-unkq' in r.getMessage() for r in errors)


def test_main_write_decision_prints_filed_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The verb echoes the filed decision id on stdout so a watcher can
    cross-link it into its in-session note / afk-digest line.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    rc = sr.main(
        [
            'write-decision',
            '--id',
            'dec-park-2',
            '--project',
            'df',
            '--text',
            'q',
            '--escalations-dir',
            str(tmp_path / 'escalations'),
        ]
    )

    assert rc == 0
    assert 'dec-park-2' in capsys.readouterr().out


def test_main_write_decision_fail_soft_when_fleet_root_under_a_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mirrors test_write_decision_fail_soft_on_unwritable_root, at the CLI
    boundary: write_decision's own fail-soft guard (not main()'s outer
    try/except) must absorb the fault, so _run_write_decision must not print
    a false confirmation when the underlying write never happened.
    """
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory')
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(blocker / 'fleet'))

    with caplog.at_level(logging.ERROR):
        # The queue dir deliberately lives under tmp_path, NOT under the
        # broken fleet root: normalize_escalations_dir is non-strict and
        # resolves a non-existent path fine, so supplying the (now
        # mandatory) stamp does not itself need the unwritable root.
        rc = sr.main(
            [
                'write-decision',
                '--id',
                'dec-park-3',
                '--project',
                'df',
                '--text',
                'q',
                '--escalations-dir',
                str(tmp_path / 'escalations'),
            ]
        )

    assert rc == 0
    assert 'dec-park-3' not in capsys.readouterr().out
    assert any(r.levelno >= logging.ERROR for r in caplog.records)
    assert not (blocker / 'fleet').exists()


def test_main_write_decision_refiling_same_id_overwrites_not_duplicates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SKILL.md promises watchers can safely re-file the same stable id
    across restarts: the second write must overwrite the first record in
    place rather than accumulating a duplicate.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))

    queue = str(tmp_path / 'escalations')
    rc1 = sr.main(
        [
            'write-decision',
            '--id',
            'dec-park-4',
            '--project',
            'df',
            '--text',
            'first?',
            '--escalations-dir',
            queue,
        ]
    )
    rc2 = sr.main(
        [
            'write-decision',
            '--id',
            'dec-park-4',
            '--project',
            'df',
            '--text',
            'second?',
            '--escalations-dir',
            queue,
        ]
    )

    assert rc1 == 0
    assert rc2 == 0
    listed = dr.list_decisions(root=tmp_path)
    assert len(listed) == 1
    assert listed[0].id == qid('dec-park-4')
    assert listed[0].text == 'second?'


def test_main_write_decision_same_id_from_two_queues_stays_one_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (f): MODE-2 same-subject collapse (task 3528 ADDENDUM, req. (b)).

    Two collision modes must be kept apart. MODE 1 (esc-3036-1, the cases
    above) is two UNRELATED escalations sharing an id -- cross-queue closing
    there is a straight bug. MODE 2 (observed: esc-5914-1) is both queues
    surfacing the SAME underlying human gate; those must collapse to ONE
    cockpit decision, because a human asked the same question twice is a
    regression of its own.

    This design satisfies MODE 2 BY CONSTRUCTION: the queue is recorded as a
    FIELD on the record and the decision id is qualified by PROJECT only
    (task 4835), never by queue, so a second watcher filing the same question
    lands on the same id. It is the guard that a future refactor to per-queue
    decision ids ('recon:esc-5914-1' vs 'orch:esc-5914-1') would double-file
    the same question and must not be adopted. Complements
    test_main_write_decision_refiling_same_id_overwrites_not_duplicates,
    which pins the same-queue restart case.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)

    rc1 = sr.main(
        [
            'write-decision',
            '--id',
            'esc-5914-1',
            '--project',
            'df',
            '--text',
            'Adopt the reify plan?',
            '--escalation-id',
            'esc-5914-1',
            '--escalations-dir',
            str(orch),
        ]
    )
    rc2 = sr.main(
        [
            'write-decision',
            '--id',
            'esc-5914-1',
            '--project',
            'df',
            '--text',
            'Adopt the reify plan?',
            '--escalation-id',
            'esc-5914-1',
            '--escalations-dir',
            str(recon),
        ]
    )

    assert rc1 == 0
    assert rc2 == 0
    listed = dr.list_decisions(root=tmp_path)
    assert [d.id for d in listed] == ['dark_factory-esc-5914-1']
    # The discriminator is a FIELD holding a normalized queue path, never a
    # queue prefix baked into the id. The field is scalar, so only ONE of
    # the two queues can survive: the FIRST filer's (task 3559 -- the second
    # filing enriches rather than overwrites), which makes the outcome
    # deterministic instead of "whichever watcher happened to write last".
    assert listed[0].escalations_dir == dr.normalize_escalations_dir(orch)


def test_main_write_decision_mode2_second_queue_enriches_and_never_downgrades(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """THE HEADLINE REGRESSION (task 3559), at the CLI boundary.

    The observed esc-5914-1 MODE-2 shape: both dark_factory queues surface
    the SAME reify gate. The first watcher files a rich record (full text,
    critical, task + session ids); the second files a poorer view of the
    same gate through the OTHER queue.

    Before this change _run_write_decision blind-overwrote the whole file,
    so the second filing won every field: the cockpit row silently lost the
    first watcher's text and was DOWNGRADED critical -> info. Now the second
    filing enriches instead -- still exactly ONE row (the collapse-to-one-
    cockpit-row requirement, task 3528 (b)), but carrying the best
    information either watcher has, with custody fields untouched.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)

    rc1 = file_decision(
        id='esc-5914-1',
        project='df',
        text='Adopt the reify plan?',
        severity='critical',
        task_id='5914',
        session_id='watcher-df-1',
        escalations_dir=str(orch),
    )
    filed_at = dr.list_decisions(root=tmp_path)[0].filed_at
    rc2 = file_decision(
        id='esc-5914-1',
        project='df',
        text='reify?',
        severity='info',
        escalations_dir=str(recon),
    )

    assert rc1 == 0
    assert rc2 == 0
    listed = dr.list_decisions(root=tmp_path)
    assert [d.id for d in listed] == ['dark_factory-esc-5914-1']
    survivor = listed[0]
    assert survivor.text == 'Adopt the reify plan?'  # never clobbered
    assert survivor.severity == 'critical'  # never downgraded
    assert survivor.task_id == '5914'  # never clobbered
    assert survivor.session_id == 'watcher-df-1'  # never clobbered
    assert survivor.escalations_dir == dr.normalize_escalations_dir(orch)  # first wins
    assert survivor.state == dr.DecisionState.OPEN
    assert survivor.filed_at == filed_at  # queue age not restamped


def test_main_write_decision_same_queue_refile_still_fully_overwrites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The enrichment branch must not OVER-trigger on a same-queue re-file.

    Both SKILL.md files promise a watcher can re-file its own stable id
    across a restart and have its UPDATED view land -- and that includes
    fields going DOWN or empty, since the watcher is the sole authority on
    its own escalation. Freezing the first values would strand stale prose
    and a stale severity in the cockpit queue forever.

    The queue stamp is precisely the axis separating "the same watcher
    re-filing" from "a second watcher on the same gate", so it is the
    discriminator, and it needs no new field. Complements
    test_main_write_decision_refiling_same_id_overwrites_not_duplicates,
    which pins the no-duplicate half; this pins that even a DOWNGRADE lands
    when it comes from the same queue.

    ...with the CUSTODY fields excepted, which is the other half of this
    test. ``filed_at`` and ``manual_boost`` are NOT the watcher's to revise:
    a restart is not news about when the gate was first surfaced (filed_at
    drives cockpit ordering), and the boost belongs to the OPERATOR via
    set_manual_boost. merge_decision_enrichment already holds both back on
    the cross-queue path; custody cannot depend on which queue re-filed, or
    an operator who boosts a row to the top of the queue silently loses it
    on the watcher's next restart -- the very "second write downgrades an
    open record" shape task 3559 removes.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)

    file_decision(
        id='esc-5914-1',
        project='df',
        text='Adopt the reify plan?',
        severity='critical',
        task_id='5914',
        escalations_dir=str(orch),
    )
    # An operator triages the cockpit between the two filings: boosts the row
    # to the top of the queue. The watcher then restarts and re-files.
    filed_at = dr.list_decisions(root=tmp_path)[0].filed_at
    assert dr.set_manual_boost(qid('esc-5914-1'), 9, root=tmp_path) is not None
    file_decision(
        id='esc-5914-1',
        project='df',
        text='reify? (rephrased)',
        severity='info',
        escalations_dir=str(orch),
    )

    listed = dr.list_decisions(root=tmp_path)
    assert [d.id for d in listed] == [qid('esc-5914-1')]
    assert listed[0].text == 'reify? (rephrased)'
    assert listed[0].severity == 'info'
    assert listed[0].task_id is None
    assert listed[0].manual_boost == 9  # the operator's boost survives
    assert listed[0].filed_at == filed_at  # queue age not restamped

@pytest.mark.parametrize(
    'closed_state',
    [
        dr.DecisionState.DROPPED,
        dr.DecisionState.ANSWERED,
        'deferred-by-hand',
    ],
)
def test_main_write_decision_same_queue_refile_does_not_resurrect_a_closed_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    closed_state: str,
) -> None:
    """THE HEADLINE REGRESSION (task 3872), at the CLI boundary.

    An operator dismisses a still-PARKED row in the cockpit decision queue
    (C5b), and the watcher that filed it restarts. Both watcher SKILLs tell
    an agent to re-file the same stable id on every restart while an item
    stays parked -- so before this change the dismissal was silently undone
    on the very next restart, and again, and again: the guarded upsert block
    was scoped to an OPEN record, so a non-open one fell through to
    ``record = incoming``, a freshly constructed record carrying state=open,
    manual_boost=0 and a restamped filed_at. reap_answered_decisions likewise
    skips a non-open decision, so nothing downstream re-closed it either.
    C5b's drop action was therefore INERT for exactly the class of row it
    exists for, and the operator's dismissal could never stick.

    Arranged through the REAL cross-subsystem sequence rather than a
    hand-built record: filed through the verb, then triaged via the same
    set_manual_boost / update_decision_state helpers cockpit/app.py calls,
    then re-filed through the verb from the SAME queue.

    Parametrized over an unrecognized state as well as the two DecisionState
    members, since DecisionRecord.state is a plain str with no from_dict
    coercion -- a disposition a future writer adds must be held back too,
    not silently reset to 'open' by a module that has not been taught it.

    The final two asserts are what keep this from over-firing into "a closed
    row is frozen": the watcher's OWN fields still land, so the row's prose
    and severity stay current even while it stays closed.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)

    rc1 = file_decision(
        id='esc-5914-1',
        project='df',
        text='Adopt the reify plan?',
        severity='critical',
        task_id='5914',
        escalations_dir=str(orch),
    )
    filed_at = dr.list_decisions(root=tmp_path)[0].filed_at
    # The operator triages the row in the cockpit: boosts it, then dismisses
    # it. Same two helpers cockpit/app.py's C5b drop action calls.
    assert dr.set_manual_boost(qid('esc-5914-1'), 9, root=tmp_path) is not None
    assert dr.update_decision_state(qid('esc-5914-1'), closed_state, root=tmp_path) is not None

    # ...and the watcher restarts, re-filing its own id from its own queue.
    rc2 = file_decision(
        id='esc-5914-1',
        project='df',
        text='reify? (rephrased)',
        severity='info',
        escalations_dir=str(orch),
    )

    assert rc1 == 0
    assert rc2 == 0
    listed = dr.list_decisions(root=tmp_path)
    assert [d.id for d in listed] == [qid('esc-5914-1')]
    survivor = listed[0]
    assert survivor.state == closed_state  # the operator's disposition STICKS
    assert survivor.manual_boost == 9  # ...as does their boost
    assert survivor.filed_at == filed_at  # ...and queue age is not restamped
    assert survivor.text == 'reify? (rephrased)'  # but the row is not FROZEN
    assert survivor.severity == 'info'  # ...a downgrade still lands


@pytest.mark.parametrize(
    'closed_state', [dr.DecisionState.DROPPED, dr.DecisionState.ANSWERED]
)
def test_main_write_decision_cross_queue_refile_of_a_closed_record_still_overwrites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    closed_state: str,
) -> None:
    """The deliberately-UNCHANGED neighbour: the QUEUE is the discriminator.

    Stated adjacently to the test above so a future reader cannot mistake
    task 3872's change for "non-open records are now always protected". The
    axis is the QUEUE, not the state: within ONE queue an ``esc-<taskid>-<n>``
    id is unique (task 3528's premise), so a same-queue re-file is
    definitively the same gate the human already dealt with. ACROSS queues
    the id namespaces genuinely collide -- dark_factory runs
    ``data/escalations`` and ``data/reconciliation/escalations`` over one
    namespace -- so a non-open cross-queue filing may be an unrelated NEW
    ask, and holding it closed would make a live gate invisible, which is
    the fail-CLOSED direction _run_reap_decisions' docstring rules out.

    Absorbs (task 3872's amendment pass) the older
    test_main_write_decision_non_open_record_is_still_overwritten, which
    pinned this same orch-seed -> recon-file shape for an ANSWERED record and
    is now parametrized in here instead of kept as a near-clone whose
    docstring ("protection is scoped to an OPEN record") had gone false on
    the same-queue axis. This version files through the CLI verb on both
    sides and carries an operator boost as well, so the contrast with the
    same-queue case is visible in every custody field rather than only in
    `state`.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)

    file_decision(
        id='esc-5914-1',
        project='df',
        text='Adopt the reify plan?',
        severity='critical',
        escalations_dir=str(orch),
    )
    assert dr.set_manual_boost(qid('esc-5914-1'), 9, root=tmp_path) is not None
    assert dr.update_decision_state(qid('esc-5914-1'), closed_state, root=tmp_path) is not None

    # A DIFFERENT queue files the same id -- possibly an unrelated new ask.
    rc = file_decision(
        id='esc-5914-1',
        project='df',
        text='a brand new question that merely shares the id',
        severity='info',
        escalations_dir=str(recon),
    )

    assert rc == 0
    listed = dr.list_decisions(root=tmp_path)
    assert [d.id for d in listed] == [qid('esc-5914-1')]
    survivor = listed[0]
    assert survivor.state == dr.DecisionState.OPEN  # re-opened: a new ask
    assert survivor.manual_boost == 0
    assert survivor.text == 'a brand new question that merely shares the id'
    assert survivor.escalations_dir == dr.normalize_escalations_dir(recon)

def test_main_write_decision_warns_when_a_same_queue_refile_is_held_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Holding a closed row closed must be LOUD, not silent.

    This is the ONE place the verb deliberately declines to do what the filer
    asked: the watcher's filing carries ``state=open`` (the verb's own
    default) and the row stays closed. The repo's loud-over-silent-degradation
    norm applies, and the watcher SKILLs' own discipline is to ADJUDICATE such
    a divergence rather than assume the re-file landed -- which it can only do
    if the divergence is visible.

    The message must name BOTH the decision id and the held state, so an
    operator or agent reading the log can tell WHICH row and WHAT disposition
    was preserved; a bare "held back" line would send them to read the file.

    stdout is asserted UNCHANGED on purpose: both SKILLs document "if the id
    doesn't come back on stdout, your filing did not land" as the
    did-it-work signal, and this filing DID land (its text and severity were
    written) -- so repurposing that channel as a failure indicator here would
    break a contract agents already rely on. The divergence goes to the log.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    file_from_a_linked_session(monkeypatch, tmp_path)
    orch, _recon = two_queues(tmp_path)

    file_decision(
        id='esc-5914-1',
        project='df',
        text='Adopt the reify plan?',
        severity='critical',
        escalations_dir=str(orch),
    )
    assert (
        dr.update_decision_state(qid('esc-5914-1'), dr.DecisionState.DROPPED, root=tmp_path)
        is not None
    )
    capsys.readouterr()  # discard the first filing's stdout

    with caplog.at_level(logging.WARNING):
        rc = file_decision(
            id='esc-5914-1',
            project='df',
            text='reify? (rephrased)',
            severity='info',
            escalations_dir=str(orch),
        )

    assert rc == 0
    assert capsys.readouterr().out.strip() == qid('esc-5914-1')  # the id still lands
    held = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and 'esc-5914-1' in r.getMessage()
    ]
    assert held, 'holding a closed row closed must be logged, not silent'
    assert 'dropped' in held[0].getMessage()  # names the PRESERVED disposition
    assert 'reopen-decision' in held[0].getMessage()  # ...and a remedy that exists


def test_main_write_decision_same_queue_refile_of_an_open_record_is_quiet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """THE NOISE GUARD: the ordinary restart re-file must say nothing.

    A same-queue re-file against an OPEN record is by far the common path --
    every watcher restart, for every still-parked item. Nothing the filer
    asked for was declined there (it files ``state=open`` and the row IS
    open), so a warning would be pure noise, and a warning on every restart
    is how the genuinely-actionable one above gets tuned out.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    file_from_a_linked_session(monkeypatch, tmp_path)
    orch, _recon = two_queues(tmp_path)

    file_decision(
        id='esc-5914-1',
        project='df',
        text='Adopt the reify plan?',
        severity='critical',
        escalations_dir=str(orch),
    )

    with caplog.at_level(logging.WARNING):
        rc = file_decision(
            id='esc-5914-1',
            project='df',
            text='reify? (rephrased)',
            severity='info',
            escalations_dir=str(orch),
        )

    assert rc == 0
    assert dr.list_decisions(root=tmp_path)[0].state == dr.DecisionState.OPEN
    noise = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING and 'esc-5914-1' in r.getMessage()
    ]
    assert not noise, f'the common restart path must be quiet, got: {noise}'


def test_main_write_decision_same_local_id_in_two_projects_files_two_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A shared escalation id across PROJECTS is two asks, filed as two rows (task 4835).

    ``esc-<taskid>-<n>`` numbering restarts per project, so 'esc-42-1' in
    dark_factory and 'esc-42-1' in reify are unrelated human gates. Qualifying
    the stored id by project gives each its own row, so neither is refused,
    merged into the other, or overwritten.

    Arranged so a merge would be DETECTABLE: the df incumbent is the POORER
    record and reify's filing is richer, so any leak of reify's severity or
    ids onto df's row shows.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)

    rc1 = file_decision(
        id='esc-42-1',
        project='df',
        text='Adopt the reify plan?',
        severity='info',
        escalation_id='esc-42-1',
        escalations_dir=str(orch),
    )
    with caplog.at_level(logging.ERROR):
        rc2 = file_decision(
            id='esc-42-1',
            project='reify',
            text='an unrelated reify gate that merely shares the id',
            severity='critical',
            task_id='42',
            session_id='watcher-reify-1',
            escalation_id='esc-42-1',
            escalations_dir=str(recon),
        )

    assert rc1 == 0
    assert rc2 == 0
    rows = {d.id: d for d in dr.list_decisions(root=tmp_path)}
    assert set(rows) == {'dark_factory-esc-42-1', 'reify-esc-42-1'}
    df_row = rows['dark_factory-esc-42-1']
    assert df_row.project == 'dark_factory'
    assert df_row.text == 'Adopt the reify plan?'
    assert df_row.severity == 'info'
    assert df_row.task_id is None
    assert df_row.session_id is None
    assert df_row.escalations_dir == dr.normalize_escalations_dir(orch)
    reify_row = rows['reify-esc-42-1']
    assert reify_row.project == 'reify'
    assert reify_row.text == 'an unrelated reify gate that merely shares the id'
    assert reify_row.severity == 'critical'
    assert reify_row.task_id == '42'
    assert reify_row.session_id == 'watcher-reify-1'
    assert reify_row.escalations_dir == dr.normalize_escalations_dir(recon)
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_main_write_decision_cross_project_filing_leaves_a_closed_row_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Another project's filing of the same local id never touches this project's
    dismissed row: it lands on its own qualified id as a fresh OPEN ask (task 4835).
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)

    file_decision(
        id='esc-42-1',
        project='df',
        text='Adopt the reify plan?',
        severity='info',
        escalations_dir=str(orch),
    )
    assert dr.set_manual_boost('dark_factory-esc-42-1', 9, root=tmp_path) is not None
    assert (
        dr.update_decision_state(
            'dark_factory-esc-42-1', dr.DecisionState.DROPPED, root=tmp_path
        )
        is not None
    )

    rc = file_decision(
        id='esc-42-1',
        project='reify',
        text='an unrelated reify gate that merely shares the id',
        severity='critical',
        escalations_dir=str(recon),
    )

    assert rc == 0
    rows = {d.id: d for d in dr.list_decisions(root=tmp_path)}
    assert set(rows) == {'dark_factory-esc-42-1', 'reify-esc-42-1'}
    assert rows['reify-esc-42-1'].state == dr.DecisionState.OPEN
    assert rows['reify-esc-42-1'].manual_boost == 0
    assert rows['dark_factory-esc-42-1'].state == dr.DecisionState.DROPPED
    assert rows['dark_factory-esc-42-1'].manual_boost == 9


def test_main_write_decision_refuses_a_qualified_id_held_open_by_another_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The cross-PROJECT refusal arm survives qualification as defense in depth.

    It is reachable only when the QUALIFIED id is itself held by another
    project's record: a legacy hand-prefixed id, filed before task 4835, whose
    prefix spells a different project than the record carries (the live
    ``recon-esc-7459-1`` held by project reify is the shape). Merging would hide
    this ask inside that row and overwriting would delete a live row, so the
    filing is refused, loudly and fail-soft.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)
    dr.write_decision(
        make_decision(
            id='reify-esc-42-1',
            project='dark_factory',
            text='Adopt the reify plan?',
            state=dr.DecisionState.OPEN,
            escalations_dir=str(orch),
        ),
        root=tmp_path,
    )
    path = dr.decision_path_for_id('reify-esc-42-1', root=tmp_path)
    before = path.read_bytes()

    with caplog.at_level(logging.ERROR):
        rc = file_decision(
            id='esc-42-1',
            project='reify',
            text='an unrelated reify gate that merely shares the id',
            severity='critical',
            escalations_dir=str(recon),
        )

    assert rc == 0
    assert path.read_bytes() == before
    refusals = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert refusals, 'the refusal must be logged, not silent'
    assert 'reify' in refusals[0].getMessage()
    assert 'dark_factory' in refusals[0].getMessage()


@pytest.mark.parametrize(
    'closed_state', [dr.DecisionState.DROPPED, dr.DecisionState.ANSWERED]
)
def test_main_write_decision_overwrites_a_qualified_id_held_closed_by_another_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    closed_state: str,
) -> None:
    """The refusal arm keeps its OPEN scoping (task 3872).

    A closed incumbent is a question already dealt with, and a cross-project
    filing is by construction a cross-queue one, where a closed row cannot be
    shown to be the same gate. So the filing starts a new ask with a full
    overwrite, and no ERROR: a refusal here would be a silent drop.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)
    dr.write_decision(
        make_decision(
            id='reify-esc-42-1',
            project='dark_factory',
            text='Adopt the reify plan?',
            state=closed_state,
            manual_boost=9,
            escalations_dir=str(orch),
        ),
        root=tmp_path,
    )

    with caplog.at_level(logging.ERROR):
        rc = file_decision(
            id='esc-42-1',
            project='reify',
            text='an unrelated reify gate that merely shares the id',
            severity='critical',
            escalations_dir=str(recon),
        )

    assert rc == 0
    assert [d.id for d in dr.list_decisions(root=tmp_path)] == ['reify-esc-42-1']
    survivor = dr.list_decisions(root=tmp_path)[0]
    assert survivor.project == 'reify'
    assert survivor.text == 'an unrelated reify gate that merely shares the id'
    assert survivor.severity == 'critical'
    assert survivor.state == dr.DecisionState.OPEN
    assert survivor.manual_boost == 0
    assert survivor.escalations_dir == dr.normalize_escalations_dir(recon)
    refusals = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not refusals, f'a closed incumbent must not be defended, got: {refusals}'


def test_main_write_decision_enriches_a_legacy_unstamped_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """End-to-end enrichment of the legacy population, through the CLI.

    The unstamped record has to be seeded via write_decision directly,
    because the verb now makes it impossible to CREATE one -- which is the
    point of the whole task. A watcher then files that id with its real
    queue and the record becomes queue-scoped: this is what makes 3640's
    back-fill terminal rather than a recurring chore, since the residual
    population can now only shrink.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)
    dr.write_decision(
        make_decision(
            id=qid('esc-5914-1'),
            project='df',
            text='Adopt the reify plan?',
            state=dr.DecisionState.OPEN,
            escalations_dir='',
        ),
        root=tmp_path,
    )

    # --escalation-id is passed because a real watcher passes it by
    # construction: the stamp names the queue THAT id belongs to (see the
    # verb's own --escalations-dir help). It matters here because the queue
    # and the id move as a PAIR -- a filing that supplies the queue but not
    # the id cannot vouch for the seed's own 'esc-1', so the upgrade would be
    # (correctly) refused as ambiguous rather than forging a pair.
    file_decision(
        id='esc-5914-1',
        project='df',
        text='reify?',
        escalation_id='esc-1',
        escalations_dir=str(orch),
    )

    listed = dr.list_decisions(root=tmp_path)
    assert [d.id for d in listed] == [qid('esc-5914-1')]
    assert listed[0].escalations_dir == dr.normalize_escalations_dir(orch)
    assert listed[0].text == 'Adopt the reify plan?'  # enriched, not clobbered


# ---------------------------------------------------------------------------
# Lazy adoption of bare-keyed decisions filed before task 4835
# ---------------------------------------------------------------------------


def _seed_legacy_decision(root: Path, **overrides: object) -> Path:
    """Write a bare-keyed record, as write-decision filed it before task 4835."""
    record = make_decision(
        **{
            'id': 'esc-5914-1',
            'project': 'df',
            'text': 'Adopt the reify plan?',
            'state': dr.DecisionState.OPEN,
            **overrides,
        }
    )
    dr.write_decision(record, root=root)
    return dr.decision_path_for_id(record.id, root=root)


def _decision_files(root: Path) -> list[str]:
    return sorted(path.name for path in dr.decisions_dir(root).glob('*.json'))


def test_main_write_decision_continues_a_same_project_legacy_record_in_place(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A same-queue re-file of a pre-4835 bare-keyed record lands ON that record,
    keeping its custody, rather than opening a second, qualified row.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)
    _seed_legacy_decision(tmp_path, manual_boost=4, escalations_dir=str(orch))
    capsys.readouterr()

    rc = file_decision(
        id='esc-5914-1',
        project='dark_factory',
        text='reify? (rephrased)',
        escalations_dir=str(orch),
    )

    assert rc == 0
    assert _decision_files(tmp_path) == ['esc-5914-1.json']
    assert capsys.readouterr().out.strip() == 'esc-5914-1'
    [survivor] = dr.list_decisions(root=tmp_path)
    assert survivor.text == 'reify? (rephrased)'
    assert survivor.manual_boost == 4
    assert survivor.filed_at == make_decision().filed_at


def test_main_write_decision_holds_a_dropped_legacy_record_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Task 3872's guarantee survives the id-format change: the first
    post-deploy restart of a watcher must not undo an operator's dismissal of
    a row filed under the old bare id.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)
    _seed_legacy_decision(
        tmp_path, state=dr.DecisionState.DROPPED, escalations_dir=str(orch)
    )

    with caplog.at_level(logging.WARNING):
        rc = file_decision(
            id='esc-5914-1',
            project='dark_factory',
            text='reify? (rephrased)',
            escalations_dir=str(orch),
        )

    assert rc == 0
    assert _decision_files(tmp_path) == ['esc-5914-1.json']
    [survivor] = dr.list_decisions(root=tmp_path)
    assert survivor.state == dr.DecisionState.DROPPED
    held = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and 'dropped' in r.getMessage()
    ]
    assert held and 'esc-5914-1' in held[0]


def test_main_write_decision_enriches_a_same_project_legacy_record_from_the_other_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MODE-2 holds across the transition: the second queue's filing folds into
    the legacy row instead of opening a qualified twin beside it.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)
    _seed_legacy_decision(tmp_path, severity='critical', escalations_dir=str(orch))

    rc = file_decision(
        id='esc-5914-1',
        project='dark_factory',
        text='reify?',
        severity='info',
        escalations_dir=str(recon),
    )

    assert rc == 0
    assert _decision_files(tmp_path) == ['esc-5914-1.json']
    [survivor] = dr.list_decisions(root=tmp_path)
    assert survivor.severity == 'critical'
    assert survivor.text == 'Adopt the reify plan?'
    assert survivor.escalations_dir == dr.normalize_escalations_dir(orch)


def test_main_write_decision_ignores_another_projects_legacy_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)
    legacy = _seed_legacy_decision(
        tmp_path, id='esc-42-1', project='dark_factory', escalations_dir=str(orch)
    )
    before = legacy.read_bytes()

    with caplog.at_level(logging.ERROR):
        rc = file_decision(
            id='esc-42-1',
            project='reify',
            text='an unrelated reify gate that merely shares the id',
            escalations_dir=str(recon),
        )

    assert rc == 0
    assert _decision_files(tmp_path) == ['esc-42-1.json', 'reify-esc-42-1.json']
    assert legacy.read_bytes() == before
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_main_write_decision_prefers_the_qualified_record_over_a_legacy_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)
    legacy = _seed_legacy_decision(
        tmp_path, id='esc-7-1', project='dark_factory', escalations_dir=str(orch)
    )
    _seed_legacy_decision(
        tmp_path, id='dark_factory-esc-7-1', project='dark_factory', escalations_dir=str(orch)
    )
    before = legacy.read_bytes()

    rc = file_decision(
        id='esc-7-1',
        project='dark_factory',
        text='reify? (rephrased)',
        escalations_dir=str(orch),
    )

    assert rc == 0
    rows = {d.id: d for d in dr.list_decisions(root=tmp_path)}
    assert rows['dark_factory-esc-7-1'].text == 'reify? (rephrased)'
    assert legacy.read_bytes() == before


class TestLegacyAdoptionLocks:
    @staticmethod
    def _spy_on_locks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        real_lock = dr.decision_id_lock
        acquired: list[str] = []

        @contextlib.contextmanager
        def recording_lock(decision_id: str, root: Path | str | None = None):
            acquired.append(decision_id)
            with real_lock(decision_id, root=root):
                yield

        monkeypatch.setattr(dr, 'decision_id_lock', recording_lock)
        return acquired

    def test_adoption_locks_the_qualified_key_then_the_legacy_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        orch, _recon = two_queues(tmp_path)
        _seed_legacy_decision(tmp_path, escalations_dir=str(orch))
        acquired = self._spy_on_locks(monkeypatch)

        file_decision(
            id='esc-5914-1', project='dark_factory', text='q', escalations_dir=str(orch)
        )

        assert acquired == ['dark_factory-esc-5914-1', 'esc-5914-1']

    def test_a_fresh_filing_locks_only_the_qualified_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        orch, _recon = two_queues(tmp_path)
        acquired = self._spy_on_locks(monkeypatch)

        file_decision(id='esc-9-1', project='dark_factory', text='q', escalations_dir=str(orch))

        assert acquired == ['dark_factory-esc-9-1']
        assert not (dr.decisions_dir(tmp_path) / 'esc-9-1.json.lock').exists()


# ---------------------------------------------------------------------------
# decision_cli's seam: the filer's record slug is injected, never looked up
# ---------------------------------------------------------------------------


def _decision_verb_args(argv: list[str]) -> argparse.Namespace:
    """Parse *argv* with a parser holding only decision_cli's verbs."""
    parser = argparse.ArgumentParser()
    decision_cli.add_verb_parsers(parser.add_subparsers(dest='verb', required=True))
    return parser.parse_args(argv)


class TestRunVerbFilingRecordSlug:
    """run_verb resolves the filer's record slug through the injected callable, and only for write-decision."""

    @staticmethod
    def _recording_resolver(calls: list[tuple[str, str]]) -> Callable[[str, str], str]:
        def filing_record_slug(local_id: str, project: str) -> str:
            calls.append((local_id, project))
            return 'injected-record-slug'

        return filing_record_slug

    def test_write_decision_files_the_slug_the_resolver_returns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        calls: list[tuple[str, str]] = []
        args = _decision_verb_args([
            'write-decision', '--id', 'esc-9-1', '--project', 'Dark-Factory', '--text', 'approve?',
            '--escalations-dir', str(tmp_path / 'esc'),
        ])

        rc = decision_cli.run_verb(args, filing_record_slug=self._recording_resolver(calls))

        assert rc == 0
        assert calls == [('esc-9-1', 'Dark-Factory')]
        [filed] = dr.list_decisions()
        assert filed.record_slug == 'injected-record-slug'

    @pytest.mark.parametrize(
        'argv',
        [
            ['reap-decisions', '--project', 'df', '--escalations-dir', '/a/data/escalations'],
            ['migrate-decision-projects', '--dry-run'],
            ['close-decision', '--id', 'esc-9-1', '--state', 'answered', '--evidence', EVIDENCE,
             '--project', 'df', '--escalations-dir', '/a/data/escalations'],
        ],
        ids=lambda argv: argv[0],
    )
    def test_the_other_verbs_never_resolve_a_filing_slug(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, argv: list[str]
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        calls: list[tuple[str, str]] = []

        decision_cli.run_verb(_decision_verb_args(argv), filing_record_slug=self._recording_resolver(calls))

        assert calls == []
