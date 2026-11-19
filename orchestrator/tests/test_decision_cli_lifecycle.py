"""Tests for the reap-decisions, close-decision, reopen-decision and migrate-decision-projects verbs, driven at ``session_registry.py <verb>``."""

from __future__ import annotations

import json
import logging
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
    seed_mixed_spellings,
    two_queues,
)

from orchestrator import decision_registry as dr
from orchestrator import session_registry as sr

# ---------------------------------------------------------------------------
# CLI reap-decisions verb (Fleet Cockpit C8: close-on-resolve driver)
# ---------------------------------------------------------------------------


def test_main_reap_decisions_closes_answered_from_archived_escalation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An escalation archived (resolved) closes its OPEN decision to
    ANSWERED, and the closed id is echoed on stdout.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    archive_dir = escalations_dir / 'archive' / '2026-07-16'
    archive_dir.mkdir(parents=True)
    (archive_dir / 'esc-resolved.json').write_text(json.dumps({'status': 'resolved'}))
    dr.write_decision(
        make_decision(
            id='dec-cli-resolved',
            project='df',
            escalation_id='esc-resolved',
            state=dr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(escalations_dir)])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-cli-resolved'] == dr.DecisionState.ANSWERED
    assert 'dec-cli-resolved' in capsys.readouterr().out


def test_main_reap_decisions_leaves_pending_escalation_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A still-pending (queue-root) escalation leaves its decision OPEN."""
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    escalations_dir.mkdir(parents=True)
    (escalations_dir / 'esc-pending.json').write_text(json.dumps({'status': 'pending'}))
    dr.write_decision(
        make_decision(
            id='dec-cli-pending',
            project='df',
            escalation_id='esc-pending',
            state=dr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(escalations_dir)])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-cli-pending'] == dr.DecisionState.OPEN


def test_main_reap_decisions_warns_on_a_declined_alias_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The operationally important half of task 3813's hint.

    Filing under the wrong token is recoverable -- the record still exists
    and is still visible. REAPING under it is the silent zero-row no-op that
    started this whole thread: it looks exactly like "nothing to reap", and
    the only way to discover otherwise today is the hand-run Counter
    one-liner both SKILL.md files tell humans to paste.

    A decision is seeded under ``solar_challenge`` so there IS a populated
    bucket to miss, exactly as the live population has it.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    archive_dir = escalations_dir / 'archive' / '2026-07-16'
    archive_dir.mkdir(parents=True)
    (archive_dir / 'esc-solar.json').write_text(json.dumps({'status': 'resolved'}))
    dr.write_decision(
        make_decision(
            id='dec-solar',
            project='solar_challenge',
            escalation_id='esc-solar',
            escalations_dir=str(escalations_dir),
            state=dr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    with caplog.at_level(logging.WARNING):
        rc = sr.main(
            [
                'reap-decisions',
                '--project',
                'my_solar_challenge',
                '--escalations-dir',
                str(escalations_dir),
            ]
        )

    assert rc == 0
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    # Non-vacuous on BOTH tokens: see names_the_destination_token.
    assert any('my_solar_challenge' in m and names_the_destination_token(m) for m in warnings)
    # On THIS path the zero-match claim is the true one -- the reap really is
    # matching, and really does match none of the seeded rows.
    assert any(claims_zero_matches(m) for m in warnings)
    # The warning changes NO reaping behaviour: the seeded decision would
    # have been closed under the right token, and is still OPEN under this
    # one. All the hint does is make the zero-match visible.
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-solar'] == dr.DecisionState.OPEN


class TestReapExpectMatches:
    """``reap-decisions --expect-matches`` names a --project no registry record carries (task 4835)."""

    @staticmethod
    def _seed(root: Path) -> Path:
        for decision_id, project, state in [
            ('dec-df-1', 'dark_factory', dr.DecisionState.ANSWERED),
            ('dec-df-2', 'df', dr.DecisionState.DROPPED),
            ('dec-reify', 'reify', dr.DecisionState.ANSWERED),
        ]:
            dr.write_decision(make_decision(id=decision_id, project=project, state=state), root=root)
        return root / 'esc'

    @staticmethod
    def _reap(queue: Path, project: str, *flags: str) -> int:
        return sr.main(['reap-decisions', '--project', project, '--escalations-dir', str(queue), *flags])

    @staticmethod
    def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]

    def test_a_token_no_record_carries_warns_once_naming_the_tokens_that_exist(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        queue = self._seed(tmp_path)

        with caplog.at_level(logging.WARNING):
            rc = self._reap(queue, 'autopilot_video', '--expect-matches')

        assert rc == 0
        assert capsys.readouterr().out == ''
        [warning] = self._warnings(caplog)
        for named in ('autopilot_video', 'ZERO', 'dark_factory=2', 'reify=1'):
            assert named in warning

    def test_a_token_with_records_but_nothing_to_reap_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        queue = self._seed(tmp_path)

        with caplog.at_level(logging.WARNING):
            assert self._reap(queue, 'reify', '--expect-matches') == 0

        assert self._warnings(caplog) == []

    def test_without_the_flag_an_unmatched_token_is_quiet(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The per-cycle Main Loop reap: a never-filed project must not warn every cycle."""
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        queue = self._seed(tmp_path)

        with caplog.at_level(logging.WARNING):
            assert self._reap(queue, 'autopilot_video') == 0

        assert self._warnings(caplog) == []

    def test_a_declined_alias_target_warns_once_with_the_declined_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        queue = self._seed(tmp_path)

        with caplog.at_level(logging.WARNING):
            assert self._reap(queue, 'my_solar_challenge', '--expect-matches') == 0

        [warning] = self._warnings(caplog)
        assert 'DECLINED' in warning


def test_main_reap_decisions_recommended_solar_token_warns_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Same fixture with the DOCUMENTED token: reaps, and stays silent.

    A watcher reaping with the token both SKILL.md files recommend runs this
    every Main Loop cycle, so a warning here would accumulate indefinitely
    and drown the one case the hint exists to surface.

    The ANSWERED assertion below also keeps its sibling honest: it proves
    this fixture's decision really is reapable, so the sibling's "still
    OPEN under my_solar_challenge" is a genuine contrast rather than a
    vacuous pass for some unrelated reason.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    archive_dir = escalations_dir / 'archive' / '2026-07-16'
    archive_dir.mkdir(parents=True)
    (archive_dir / 'esc-solar.json').write_text(json.dumps({'status': 'resolved'}))
    dr.write_decision(
        make_decision(
            id='dec-solar',
            project='solar_challenge',
            escalation_id='esc-solar',
            escalations_dir=str(escalations_dir),
            state=dr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    with caplog.at_level(logging.WARNING):
        rc = sr.main(
            [
                'reap-decisions',
                '--project',
                'solar_challenge',
                '--escalations-dir',
                str(escalations_dir),
            ]
        )

    assert rc == 0
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-solar'] == dr.DecisionState.ANSWERED


def test_main_reap_decisions_scopes_to_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A decision from a DIFFERENT project is left OPEN even though its
    escalation has resolved -- decisions are fleet-global but escalations
    are per-project, so reap-decisions is a per-project driver.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    archive_dir = escalations_dir / 'archive' / '2026-07-16'
    archive_dir.mkdir(parents=True)
    (archive_dir / 'esc-other-project.json').write_text(json.dumps({'status': 'resolved'}))
    dr.write_decision(
        make_decision(
            id='dec-other-project',
            project='not-df',
            escalation_id='esc-other-project',
            state=dr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(escalations_dir)])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-other-project'] == dr.DecisionState.OPEN


def test_main_reap_decisions_closes_every_spelling_of_one_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """THE REGRESSION (task 3807): ONE run closes EVERY spelling of ONE project.

    Measured live 2026-08-06: 41 OPEN dark-factory decisions split 22/17/2
    across ``dark_factory``/``df``/``dark-factory``. The reaper compared
    ``decision.project`` to ``--project`` as raw strings, so each partition
    was invisible to a reap scoped to either of the others and an L2 session
    had to run the verb once per token to close them all.

    All four decisions share ONE queue and each pins its own resolved
    escalation, so the axis-2 queue guard is satisfied and cannot short-
    circuit this into a vacuous pass -- ``escalations_dir=`` is therefore
    passed explicitly at each call site, per make_decision's documented rule.

    One test, both directions: the three dark-factory spellings must ALL
    close, and the unrelated ``reify`` decision must stay OPEN -- so widening
    the axis-1 match to spellings of the SAME project provably does not widen
    it ACROSS projects.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    archive_dir = escalations_dir / 'archive' / '2026-08-06'
    archive_dir.mkdir(parents=True)
    for esc_id in ('esc-a', 'esc-b', 'esc-c', 'esc-other'):
        (archive_dir / f'{esc_id}.json').write_text(json.dumps({'status': 'resolved'}))
    seeded = {
        'dec-df': ('df', 'esc-a'),
        'dec-hyphen': ('dark-factory', 'esc-b'),
        'dec-canonical': ('dark_factory', 'esc-c'),
        'dec-reify': ('reify', 'esc-other'),
    }
    for decision_id, (project, escalation_id) in seeded.items():
        dr.write_decision(
            make_decision(
                id=decision_id,
                project=project,
                escalation_id=escalation_id,
                state=dr.DecisionState.OPEN,
                escalations_dir=str(escalations_dir),
            ),
            root=tmp_path,
        )

    rc = sr.main(
        ['reap-decisions', '--project', 'dark_factory', '--escalations-dir', str(escalations_dir)]
    )

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-df'] == dr.DecisionState.ANSWERED
    assert listed['dec-hyphen'] == dr.DecisionState.ANSWERED
    assert listed['dec-canonical'] == dr.DecisionState.ANSWERED
    assert listed['dec-reify'] == dr.DecisionState.OPEN


@pytest.mark.parametrize('reaper_project', ['DF', 'df', 'dark-factory', ' Dark_Factory '])
def test_main_reap_decisions_normalizes_the_reapers_own_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reaper_project: str,
) -> None:
    """BOTH sides of the join normalize, mirroring the axis-2 queue guard.

    A watcher invoking with a stale ``--project`` spelling must still close a
    decision stored under the canonical token -- otherwise the fix would be
    half-applied and the caller's own spelling would become a new way to
    silently miss rows.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    escalations_dir = tmp_path / 'esc'
    archive_dir = escalations_dir / 'archive' / '2026-08-06'
    archive_dir.mkdir(parents=True)
    (archive_dir / 'esc-r.json').write_text(json.dumps({'status': 'resolved'}))
    dr.write_decision(
        make_decision(
            id='dec-canon',
            project='dark_factory',
            escalation_id='esc-r',
            state=dr.DecisionState.OPEN,
            escalations_dir=str(escalations_dir),
        ),
        root=tmp_path,
    )

    rc = sr.main(
        ['reap-decisions', '--project', reaper_project, '--escalations-dir', str(escalations_dir)]
    )

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-canon'] == dr.DecisionState.ANSWERED


def test_main_reap_decisions_does_not_close_across_queues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """THE REGRESSION (task 3528), cases (a) + (d) in ONE reaper run.

    (a) A decision stamped with the `recon` queue must NOT be closed by a
    reaper scanning the `orch` queue, even though `orch` holds a RESOLVED
    escalation with the same id -- they are unrelated escalations that merely
    collide in the shared id namespace. Before this fix the join was scoped
    on project alone, so this decision closed to ANSWERED and vanished from
    the cockpit queue while its own escalation was still PENDING.

    (d) In the SAME run, a queue-less (pre-change, escalations_dir='')
    decision on the SAME escalation id still closes exactly as before -- so
    one invocation proves the new guard is both blocking (a) and backward-
    compatible (d), and that the only thing distinguishing them is the queue
    stamp.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)
    dr.write_decision(
        make_decision(
            id='dec-recon-gate',
            project='df',
            escalation_id='esc-3036-1',
            state=dr.DecisionState.OPEN,
            escalations_dir=dr.normalize_escalations_dir(recon),
        ),
        root=tmp_path,
    )
    dr.write_decision(
        make_decision(
            id='dec-legacy-queueless',
            project='df',
            escalation_id='esc-3036-1',
            state=dr.DecisionState.OPEN,
            escalations_dir='',
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-recon-gate'] == dr.DecisionState.OPEN
    assert listed['dec-legacy-queueless'] == dr.DecisionState.ANSWERED


def test_main_reap_decisions_does_not_close_across_queues_mirrored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (b): the guard is symmetric. A decision stamped with the `orch`
    queue is equally protected from the recon watcher's reaper, which scans
    `recon` and finds a RESOLVED escalation of the same id there. Neither
    watcher is privileged; either one reaping the other's decisions is the
    same bug.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)
    recon_archive = recon / 'archive' / '2026-07-26'
    recon_archive.mkdir(parents=True)
    (recon_archive / 'esc-mirror.json').write_text(json.dumps({'status': 'resolved'}))
    dr.write_decision(
        make_decision(
            id='dec-orch-gate',
            project='df',
            escalation_id='esc-mirror',
            state=dr.DecisionState.OPEN,
            escalations_dir=dr.normalize_escalations_dir(orch),
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(recon)])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-orch-gate'] == dr.DecisionState.OPEN


def test_main_reap_decisions_same_queue_still_closes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (c): the guard must not over-block. A decision stamped with the
    SAME queue the reaper is scanning still closes on its escalation's
    terminal status -- otherwise queue-scoping would quietly turn the reaper
    into a permanent no-op and every decision would need manual closure.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)
    dr.write_decision(
        make_decision(
            id='dec-orch-same-queue',
            project='df',
            escalation_id='esc-3036-1',
            state=dr.DecisionState.OPEN,
            escalations_dir=dr.normalize_escalations_dir(orch),
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-orch-same-queue'] == dr.DecisionState.ANSWERED


def test_main_reap_decisions_queue_match_is_spelling_insensitive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (e): the comparison normalizes BOTH sides, rather than doing a raw
    string compare of whatever each side happened to store.

    The decision here carries a dotted, trailing-slash spelling of the `orch`
    queue -- what a hand-repaired record, a future writer, or a record
    migrated between checkouts can hold, since it bypassed write-decision's
    write-time normalization. A raw compare would treat it as a foreign
    queue and fail OPEN forever; worse, the same laxness in reverse is how a
    false NON-match would reintroduce silent divergence between writer and
    reaper.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)
    (tmp_path / 'x').mkdir()
    dotted = str(orch.parent / 'x' / '..' / orch.name) + '/'
    assert dotted != str(orch)
    dr.write_decision(
        make_decision(
            id='dec-dotted-queue',
            project='df',
            escalation_id='esc-3036-1',
            state=dr.DecisionState.OPEN,
            escalations_dir=dotted,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-dotted-queue'] == dr.DecisionState.ANSWERED


def test_main_reap_decisions_normalizes_the_reapers_own_escalations_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Case (e) MIRRORED: the REAPER's side of the compare is normalized too.

    ..._queue_match_is_spelling_insensitive only exercises the DECISION side
    -- it stores a dotted spelling and passes the already-canonical
    ``str(orch)`` to the CLI, so the reaper-side normalize is a no-op there.
    Here it is the other way round: the record carries the canonical form and
    the CLI is handed a relative, dotted spelling of the same queue.

    This is the real invocation shape, not a contrivance. Both SKILL.md files
    now promise "stored normalized, so any spelling of the same directory
    works", and the recon watcher's documented command is
    ``--escalations-dir $DARK_FACTORY_ROOT/data/reconciliation/escalations``
    where ``$DARK_FACTORY_ROOT`` may legitimately be relative or symlinked.
    Drop the reaper-side normalize and EVERY stamped decision becomes a
    permanent no-close -- fail-open, so invisible: nothing errors, decisions
    just quietly stop closing. Running under monkeypatch.chdir also pins that
    the reaper side resolves against the cwd, matching
    normalize_escalations_dir's documented expanduser/resolve contract.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)
    (tmp_path / 'x').mkdir()
    monkeypatch.chdir(tmp_path)
    relative = f'x/../{orch.name}/'
    assert not Path(relative).is_absolute()
    dr.write_decision(
        make_decision(
            id='dec-canonical-queue',
            project='df',
            escalation_id='esc-3036-1',
            state=dr.DecisionState.OPEN,
            escalations_dir=dr.normalize_escalations_dir(orch),
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', relative])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-canonical-queue'] == dr.DecisionState.ANSWERED


def test_main_reap_decisions_mode2_collapsed_decision_is_reapable_only_by_its_stamped_queue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Pins the acknowledged MODE-2 tradeoff (task 3528, raised in review).

    A MODE-2 same-subject duplicate (esc-5914-1) collapses to ONE record by
    design -- but ``escalations_dir`` is single-valued, so the survivor can
    carry only ONE of the two queues: the FIRST filer's, since task 3559
    made a cross-queue re-file enrich rather than overwrite (pinned by
    ..._same_id_from_two_queues_stays_one_decision). The axis-2 guard then
    makes the OTHER queue's reaper skip it outright. Task 3559 changed only
    WHICH queue survives -- deterministically the first filer's, rather than
    racily whichever watcher wrote last -- not the gap's existence.

    So if the escalation that actually reaches a terminal status is the one
    in the NON-stamped queue -- entirely possible for two independently-filed
    escalations covering the same gate -- the decision now stays OPEN and
    needs human closure, where before this change either reaper would have
    closed it. That is a real behaviour change for the MODE-2 population,
    accepted because it is the fail-OPEN direction: an over-held decision is
    a visible, human-triageable cockpit row, while a falsely closed one is
    invisible (the ~7-day loss this task exists to prevent). Pinned here so
    the tradeoff is explicit rather than latent; documented on
    _run_reap_decisions and in the recon watcher's MODE-2 bullet.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, recon = two_queues(tmp_path)
    # The same gate is filed as a separate escalation in each queue; only the
    # orchestrator's copy has resolved.
    (orch / 'archive' / '2026-07-26' / 'esc-5914-1.json').write_text(
        json.dumps({'status': 'resolved'})
    )
    (recon / 'esc-5914-1.json').write_text(json.dumps({'status': 'pending'}))
    # Two watchers, one collapsed record. RECON FILES FIRST, so under task
    # 3559's first-writer-wins its queue is the one stamped -- which keeps
    # this case non-vacuous: the reaper below scans `orch`, the NON-stamped
    # queue, and `orch` is precisely where the RESOLVED copy lives, so an
    # absent axis-2 guard really would close this decision.
    for queue in (recon, orch):
        assert (
            sr.main(
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
                    str(queue),
                ]
            )
            == 0
        )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed[qid('esc-5914-1')] == dr.DecisionState.OPEN


def test_main_reap_decisions_fail_soft_on_bad_escalations_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A nonexistent --escalations-dir must not raise or close anything:
    read_escalation_status returns None for every lookup, and
    reap_answered_decisions treats None as "leave OPEN".
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    bad_escalations_dir = tmp_path / 'does-not-exist'
    dr.write_decision(
        make_decision(
            id='dec-cli-badescdir',
            project='df',
            escalation_id='esc-whatever',
            state=dr.DecisionState.OPEN,
        ),
        root=tmp_path,
    )

    rc = sr.main(
        ['reap-decisions', '--project', 'df', '--escalations-dir', str(bad_escalations_dir)]
    )

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-cli-badescdir'] == dr.DecisionState.OPEN


def test_main_reap_decisions_refuses_unknown_queue_but_still_closes_legacy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """REGRESSION GUARD, not a RED driver (task 3640) -- and deliberately so.

    Probed against the merged 3528 code before this test was written: a
    decision stamped with ANY truthy non-matching value already survives the
    axis-2 `if decision_dir and decision_dir != reaper_dir` compare, so an
    UNKNOWN_QUEUE stamp is ALREADY safe today. A test asserting "the reaper
    closes unknown-stamped records" would therefore be doomed-green and could
    never drive an implementation. What this pins instead is that the safety
    SURVIVES: it is currently an accident of string inequality, and a future
    simplification of that compare (or of the explicit by-name guard step-2
    adds) would silently make every back-filled undeterminable record closable
    again, restoring the exact ~7-day invisible-close failure 3528 removed.

    Both arms run in ONE reaper invocation against a queue holding a RESOLVED
    escalation with the shared id, so the ONLY thing distinguishing them is
    the queue stamp:
      - the UNKNOWN_QUEUE-stamped record stays OPEN (refuse, never default to
        close -- it stays a visible cockpit row for human closure);
      - the legacy `escalations_dir=''` record on the SAME id still closes to
        ANSWERED. That second assertion is the load-bearing one: task 3640
        must NOT redefine '' under the human. '' keeps its 3528 meaning
        (fall back to project-only scoping); the back-fill DRAINS that
        population instead of changing what it means.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)
    dr.write_decision(
        make_decision(
            id='dec-unknown-queue',
            project='df',
            escalation_id='esc-3036-1',
            state=dr.DecisionState.OPEN,
            escalations_dir=dr.UNKNOWN_QUEUE,
        ),
        root=tmp_path,
    )
    dr.write_decision(
        make_decision(
            id='dec-legacy-unset',
            project='df',
            escalation_id='esc-3036-1',
            state=dr.DecisionState.OPEN,
            escalations_dir='',
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', str(orch)])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-unknown-queue'] == dr.DecisionState.OPEN
    assert listed['dec-legacy-unset'] == dr.DecisionState.ANSWERED


def test_main_reap_decisions_refuses_unknown_queue_even_when_reaper_passes_the_sentinel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The degenerate case the string-inequality compare alone does NOT cover.

    If a reaper is invoked with the sentinel as its own ``--escalations-dir``
    (an operator copy-pasting a stamped value out of a record, or a wrapper
    threading the field straight through), then `decision_dir == reaper_dir`
    and the axis-2 guard does not fire at all. Today the record still survives
    only because read_escalation_status finds no queue at that bogus path and
    returns None -- safety by lucky accident, one directory named
    ``<unknown>`` away from failing. Step-2's explicit by-name guard is what
    makes the refusal intentional, and this pins it.

    Note the sentinel must reach the record VERBATIM for this to be a real
    test, which is exactly what step-2's normalize_escalations_dir case
    guarantees.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    orch, _recon = two_queues(tmp_path)
    # A real directory literally named '<unknown>', holding a RESOLVED
    # escalation for the shared id -- so a reaper that resolved the sentinel
    # as a relative path WOULD find a close-worthy status there.
    monkeypatch.chdir(tmp_path)
    bogus = tmp_path / dr.UNKNOWN_QUEUE
    bogus.mkdir()
    (bogus / 'esc-3036-1.json').write_text(json.dumps({'status': 'resolved'}))
    assert (orch / 'archive' / '2026-07-26' / 'esc-3036-1.json').is_file()
    dr.write_decision(
        make_decision(
            id='dec-unknown-selfmatch',
            project='df',
            escalation_id='esc-3036-1',
            state=dr.DecisionState.OPEN,
            escalations_dir=dr.UNKNOWN_QUEUE,
        ),
        root=tmp_path,
    )

    rc = sr.main(['reap-decisions', '--project', 'df', '--escalations-dir', dr.UNKNOWN_QUEUE])

    assert rc == 0
    listed = {d.id: d.state for d in dr.list_decisions(root=tmp_path)}
    assert listed['dec-unknown-selfmatch'] == dr.DecisionState.OPEN


# ---------------------------------------------------------------------------
# CLI migrate-decision-projects verb (task 3807: one-shot project backfill)
# ---------------------------------------------------------------------------


def test_main_migrate_decision_projects_prints_one_line_per_migrated_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """One ``<id> <old> -> <new>`` line per migrated record, mirroring
    reap-decisions' one-line-per-closed-decision convention -- so the verb is
    grep-able and diffable the same way and an operator's ``| wc -l`` habits
    transfer unchanged. Asserted as a SET: ordering is list_decisions' sorted
    glob, not a promise this verb makes.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    seed_mixed_spellings(tmp_path)

    rc = sr.main(['migrate-decision-projects'])

    assert rc == 0
    printed = {line for line in capsys.readouterr().out.splitlines() if line.strip()}
    assert printed == {
        'd-df df -> dark_factory',
        'd-hyphen dark-factory -> dark_factory',
        'd-mixed Dark_Factory -> dark_factory',
    }
    assert {d.id: d.project for d in dr.list_decisions(root=tmp_path)} == {
        'd-df': 'dark_factory',
        'd-hyphen': 'dark_factory',
        'd-mixed': 'dark_factory',
        'd-canon': 'dark_factory',
        'd-reify': 'reify',
    }


def test_main_migrate_decision_projects_dry_run_prints_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run prints the same preview lines and leaves every record's
    project unchanged, so an operator can read the plan before running it
    against the live fleet root.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    seed_mixed_spellings(tmp_path)

    rc = sr.main(['migrate-decision-projects', '--dry-run'])

    assert rc == 0
    printed = {line for line in capsys.readouterr().out.splitlines() if line.strip()}
    assert printed == {
        'd-df df -> dark_factory',
        'd-hyphen dark-factory -> dark_factory',
        'd-mixed Dark_Factory -> dark_factory',
    }
    assert {d.id: d.project for d in dr.list_decisions(root=tmp_path)} == {
        'd-df': 'df',
        'd-hyphen': 'dark-factory',
        'd-mixed': 'Dark_Factory',
        'd-canon': 'dark_factory',
        'd-reify': 'reify',
    }


def test_main_migrate_decision_projects_is_a_no_op_when_already_clean(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Re-running after the fleet is clean prints NOTHING.

    That silence is what makes the verb safe to keep permanently: it doubles
    as the repair tool for a hand-edited record without an operator having to
    reason about whether it already ran.
    """
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    seed_mixed_spellings(tmp_path)
    assert sr.main(['migrate-decision-projects']) == 0
    capsys.readouterr()

    rc = sr.main(['migrate-decision-projects'])

    assert rc == 0
    assert capsys.readouterr().out.strip() == ''


def test_main_migrate_decision_projects_fail_soft_when_fleet_root_under_a_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mirrors test_main_write_decision_fail_soft_when_fleet_root_under_a_file:
    an unusable fleet root must return 0 and raise nothing, preserving the
    module's documented boundary that a registry fault can never change a
    bash caller's exit code.
    """
    blocker = tmp_path / 'blocker'
    blocker.write_text('not a directory')
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(blocker / 'fleet'))

    rc = sr.main(['migrate-decision-projects'])

    assert rc == 0
    assert capsys.readouterr().out.strip() == ''
    assert not (blocker / 'fleet').exists()


# ---------------------------------------------------------------------------
# DecisionRecord.closing_evidence and the close-decision verb (task 5376)
# ---------------------------------------------------------------------------


class TestClosedAt:
    """When the evidence-carrying close happened: custody, stamped by close_decision_with_evidence alone."""

    def test_a_same_queue_write_decision_refile_keeps_it(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        queue = str(tmp_path / 'data' / 'escalations')
        filing = {'id': 'dec-close', 'project': 'dark_factory', 'text': 'approve?', 'escalations_dir': queue}
        file_decision(**filing)
        closed = dr.close_decision_with_evidence(
            qid('dec-close'), dr.DecisionState.ANSWERED, EVIDENCE, root=tmp_path,
            expected_project='dark_factory', expected_escalations_dir=queue,
        )
        assert closed is not None and closed.closed_at

        file_decision(**{**filing, 'text': 'approve? (rephrased)'})

        [reread] = dr.list_decisions(root=tmp_path)
        assert reread.text == 'approve? (rephrased)'
        assert (reread.state, reread.closing_evidence, reread.closed_at) == (
            dr.DecisionState.ANSWERED, EVIDENCE, closed.closed_at,
        )


class TestCloseDecisionVerb:
    def _argv(self, root: Path, *, state: str = 'answered', evidence: str = EVIDENCE) -> list[str]:
        seeded = make_decision()
        return [
            'close-decision', '--id', 'dec-close', '--state', state, '--evidence', evidence,
            '--project', seeded.project, '--escalations-dir', seeded.escalations_dir, '--root', str(root),
        ]

    def test_is_registered_in_the_parser(self, tmp_path: Path) -> None:
        args = sr._build_parser().parse_args(self._argv(tmp_path))
        seeded = make_decision()

        assert (args.verb, args.id, args.state, args.evidence, args.root) == (
            'close-decision', 'dec-close', 'answered', EVIDENCE, str(tmp_path),
        )
        assert (args.project, args.escalations_dir) == (seeded.project, seeded.escalations_dir)

    def test_success_prints_the_id_and_exits_0(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        dr.write_decision(make_decision(id='dec-close', state=dr.DecisionState.OPEN), root=tmp_path)

        rc = sr.main(self._argv(tmp_path))

        assert rc == 0
        assert capsys.readouterr().out.strip() == 'dec-close'
        [reread] = dr.list_decisions(root=tmp_path)
        assert (reread.state, reread.closing_evidence) == (dr.DecisionState.ANSWERED, EVIDENCE)

    def test_a_refusal_exits_nonzero_naming_it(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        dr.write_decision(make_decision(id='dec-close', state=dr.DecisionState.OPEN), root=tmp_path)

        rc = sr.main(self._argv(tmp_path, evidence=' '))

        assert rc != 0
        assert 'evidence is empty' in capsys.readouterr().err

    def test_an_absent_record_exits_nonzero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = sr.main(self._argv(tmp_path))

        assert rc != 0
        assert 'dec-close' in capsys.readouterr().err


class TestCloseDecisionIdentity:
    """A close lands only on the record the caller names: decision ids are fleet-global, esc numbering is per project."""

    _A_PROJECT = 'know_live'
    _A_QUEUE = '/a/data/escalations'
    _B_PROJECT = 'reify'
    _B_QUEUE = '/b/data/escalations'

    def _seed(self, root: Path, **overrides: object) -> Path:
        fields = {'id': 'esc-42-1', 'project': self._A_PROJECT, 'escalations_dir': self._A_QUEUE,
                  'state': dr.DecisionState.OPEN, 'closing_evidence': '', **overrides}
        dr.write_decision(make_decision(**fields), root=root)
        return dr.decision_path_for_id('esc-42-1', root=root)

    def _close(self, root: Path, project: str, queue: str) -> dr.DecisionRecord | None:
        return dr.close_decision_with_evidence(
            'esc-42-1', dr.DecisionState.ANSWERED, EVIDENCE, root=root,
            expected_project=project, expected_escalations_dir=queue,
        )

    @pytest.mark.parametrize(('project', 'queue'), [
        (_A_PROJECT, _B_QUEUE),
        (_B_PROJECT, _A_QUEUE),
        (_B_PROJECT, _B_QUEUE),
    ], ids=['queue-only', 'project-only', 'both'])
    def test_another_projects_record_at_the_same_id_is_refused_and_untouched(
        self, tmp_path: Path, project: str, queue: str
    ) -> None:
        """The reviewer's scenario: project B's close payload names an id project A already holds OPEN."""
        path = self._seed(tmp_path)
        before = path.read_bytes()

        with pytest.raises(dr.DecisionCloseRefused) as refused:
            self._close(tmp_path, project, queue)

        message = str(refused.value)
        for named in ('esc-42-1', self._A_PROJECT, self._A_QUEUE, project, queue):
            assert named in message
        assert path.read_bytes() == before
        [reread] = dr.list_decisions(root=tmp_path)
        assert (reread.state, reread.closing_evidence, reread.closed_at) == (dr.DecisionState.OPEN, '', '')

    @pytest.mark.parametrize('queue', ['/q/data/escalations/', '/q/./data/escalations'], ids=['slash', 'dot'])
    def test_both_sides_are_folded_before_the_compare(self, tmp_path: Path, queue: str) -> None:
        self._seed(tmp_path, project='df', escalations_dir=dr.normalize_escalations_dir('/q/data/escalations'))

        updated = self._close(tmp_path, 'dark_factory', queue)

        assert updated is not None
        assert (updated.state, updated.closing_evidence) == (dr.DecisionState.ANSWERED, EVIDENCE)

    @pytest.mark.parametrize('stamp', ['', dr.UNKNOWN_QUEUE], ids=['legacy', 'unknown'])
    def test_a_sentinel_stamp_closes_under_the_same_sentinel(self, tmp_path: Path, stamp: str) -> None:
        self._seed(tmp_path, escalations_dir=stamp)

        updated = self._close(tmp_path, self._A_PROJECT, stamp)

        assert updated is not None
        assert updated.closing_evidence == EVIDENCE

    @pytest.mark.parametrize(('stamp', 'queue'), [
        ('', '/q/data/escalations'),
        (dr.UNKNOWN_QUEUE, ''),
        (dr.UNKNOWN_QUEUE, '/q/data/escalations'),
    ], ids=['legacy-vs-real', 'unknown-vs-legacy', 'unknown-vs-real'])
    def test_a_sentinel_stamp_is_refused_under_any_other_expectation(
        self, tmp_path: Path, stamp: str, queue: str
    ) -> None:
        path = self._seed(tmp_path, escalations_dir=stamp)
        before = path.read_bytes()

        with pytest.raises(dr.DecisionCloseRefused, match='esc-42-1'):
            self._close(tmp_path, self._A_PROJECT, queue)

        assert path.read_bytes() == before

    def test_the_identity_check_runs_before_the_overwrite_check(self, tmp_path: Path) -> None:
        path = self._seed(tmp_path, state=dr.DecisionState.ANSWERED, closing_evidence='project A evidence')
        before = path.read_bytes()

        with pytest.raises(dr.DecisionCloseRefused) as refused:
            self._close(tmp_path, self._B_PROJECT, self._B_QUEUE)

        assert self._B_PROJECT in str(refused.value)
        assert 'overwrite' not in str(refused.value)
        assert path.read_bytes() == before

    def _argv(self, root: Path, *, project: str, queue: str) -> list[str]:
        return [
            'close-decision', '--id', 'esc-42-1', '--state', 'answered', '--evidence', EVIDENCE,
            '--project', project, '--escalations-dir', queue, '--root', str(root),
        ]

    @pytest.mark.parametrize('dropped', ['--project', '--escalations-dir'])
    def test_the_verb_requires_both_expectations(self, tmp_path: Path, dropped: str) -> None:
        argv = self._argv(tmp_path, project=self._A_PROJECT, queue=self._A_QUEUE)
        at = argv.index(dropped)
        del argv[at:at + 2]

        with pytest.raises(SystemExit) as exited:
            sr.main(argv)

        assert exited.value.code == 2

    def test_the_verb_exits_nonzero_on_a_mismatch_and_leaves_the_record(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = self._seed(tmp_path)
        before = path.read_bytes()

        rc = sr.main(self._argv(tmp_path, project=self._B_PROJECT, queue=self._B_QUEUE))

        assert rc != 0
        err = capsys.readouterr().err
        assert 'close-decision refused' in err
        assert self._B_PROJECT in err
        assert path.read_bytes() == before

    def test_the_verb_closes_the_record_it_names(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self._seed(tmp_path)

        rc = sr.main(self._argv(tmp_path, project=self._A_PROJECT, queue=self._A_QUEUE))

        assert rc == 0
        assert capsys.readouterr().out.strip() == 'esc-42-1'
        [reread] = dr.list_decisions(root=tmp_path)
        assert (reread.state, reread.closing_evidence) == (dr.DecisionState.ANSWERED, EVIDENCE)


class TestReopenDecision:
    """reopen_decision re-opens a closed record IN PLACE, under the same compare-and-swap as a close (task 4835)."""

    def test_a_same_queue_refile_after_a_reopen_stays_open_and_quiet(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        file_from_a_linked_session(monkeypatch, tmp_path)
        queue = str(tmp_path / 'data' / 'escalations')
        filing = {
            'id': 'esc-5914-1', 'project': 'dark_factory', 'text': 'Adopt the reify plan?', 'escalations_dir': queue,
        }
        file_decision(**filing)
        filed_id = qid('esc-5914-1')
        assert dr.update_decision_state(filed_id, dr.DecisionState.DROPPED, root=tmp_path) is not None
        assert dr.reopen_decision(
            filed_id, root=tmp_path, expected_project='dark_factory', expected_escalations_dir=queue
        ) is not None

        with caplog.at_level(logging.WARNING):
            file_decision(**{**filing, 'text': 'reify? (rephrased)'})

        [reread] = dr.list_decisions(root=tmp_path)
        assert reread.state == dr.DecisionState.OPEN
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestReopenDecisionVerb:
    _PROJECT = 'know_live'
    _QUEUE = '/a/data/escalations'

    def _argv(self, root: Path, *, project: str = _PROJECT, queue: str = _QUEUE) -> list[str]:
        return [
            'reopen-decision', '--id', 'esc-42-1', '--project', project, '--escalations-dir', queue,
            '--root', str(root),
        ]

    def _seed(self, root: Path) -> Path:
        dr.write_decision(
            make_decision(
                id='esc-42-1', project=self._PROJECT, escalations_dir=self._QUEUE,
                state=dr.DecisionState.DROPPED, closing_evidence=EVIDENCE,
            ),
            root=root,
        )
        return dr.decision_path_for_id('esc-42-1', root=root)

    @pytest.mark.parametrize('dropped', ['--project', '--escalations-dir'])
    def test_the_verb_requires_both_expectations(self, tmp_path: Path, dropped: str) -> None:
        argv = self._argv(tmp_path)
        at = argv.index(dropped)
        del argv[at:at + 2]

        with pytest.raises(SystemExit) as exited:
            sr.main(argv)

        assert exited.value.code == 2

    def test_success_prints_the_id_and_exits_0(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self._seed(tmp_path)

        rc = sr.main(self._argv(tmp_path))

        assert rc == 0
        assert capsys.readouterr().out == 'esc-42-1\n'
        [reread] = dr.list_decisions(root=tmp_path)
        assert reread.state == dr.DecisionState.OPEN

    def test_a_mismatch_exits_nonzero_and_leaves_the_record(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        before = self._seed(tmp_path).read_bytes()

        rc = sr.main(self._argv(tmp_path, project='reify'))

        assert rc != 0
        err = capsys.readouterr().err
        assert 'reopen-decision refused' in err
        assert 'reify' in err
        assert dr.decision_path_for_id('esc-42-1', root=tmp_path).read_bytes() == before

    def test_an_absent_record_exits_nonzero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = sr.main(self._argv(tmp_path))

        assert rc != 0
        assert 'esc-42-1' in capsys.readouterr().err

    @pytest.mark.parametrize(
        ('status', 'warned'), [('resolved', True), ('dismissed', True), ('pending', False)]
    )
    def test_reopening_a_row_the_next_reap_would_close_again_warns(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], status: str, warned: bool
    ) -> None:
        queue = tmp_path / 'queue'
        queue.mkdir()
        (queue / 'esc-42-1.json').write_text(json.dumps({'status': status}))
        dr.write_decision(
            make_decision(
                id='esc-42-1', project=self._PROJECT, escalations_dir=str(queue), escalation_id='esc-42-1',
                state=dr.DecisionState.DROPPED, closing_evidence=EVIDENCE,
            ),
            root=tmp_path,
        )

        rc = sr.main(self._argv(tmp_path, queue=str(queue)))

        assert rc == 0
        captured = capsys.readouterr()
        assert captured.out == 'esc-42-1\n'
        assert ('reap-decisions' in captured.err) is warned
        assert (status in captured.err) is warned
        assert dr.list_decisions(root=tmp_path)[0].state == dr.DecisionState.OPEN

    def test_a_held_closed_row_is_reopened_in_place_by_its_printed_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
        orch, _recon = two_queues(tmp_path)
        filing = {'id': 'esc-5914-1', 'project': 'df', 'text': 'Adopt the reify plan?', 'escalations_dir': str(orch)}
        capsys.readouterr()
        assert file_decision(**filing) == 0
        printed = capsys.readouterr().out.strip()
        assert dr.update_decision_state(printed, dr.DecisionState.DROPPED, root=tmp_path) is not None
        file_decision(**{**filing, 'text': 'reify? (rephrased)'})
        assert dr.list_decisions(root=tmp_path)[0].state == dr.DecisionState.DROPPED

        rc = sr.main(
            ['reopen-decision', '--id', printed, '--project', 'dark_factory', '--escalations-dir', str(orch)]
        )

        assert rc == 0
        [reread] = dr.list_decisions(root=tmp_path)
        assert (reread.id, reread.state) == (printed, dr.DecisionState.OPEN)
