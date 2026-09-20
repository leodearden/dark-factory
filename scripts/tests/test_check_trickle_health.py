"""Tests for scripts/legibility/check_trickle_health.py — the thing that
finally RUNS the two trickle probes.

The defect this module closes is not a wrong verdict, it is an ABSENT
CALLER: no systemd unit, cron entry or config bound either
``check_trickle_progress.py`` or ``check_trickle_liveness.sh``, and the
only bindings either ever had were the one-shot ``before_done`` milestone
predicates on tasks 2587/2615 — both ``done``, and a completed milestone
predicate never runs again. So the tests that matter most here are the
ones pinning that this module EXECUTES the sibling scripts by path.

Every test injects BOTH probe runners. None may shell out to a real
``systemctl`` or run the real progress probe against the operator's live
state; ``scripts/tests/conftest.py::_isolate_legibility_trickle_state`` is
the backstop if one ever slips.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from legibility import check_trickle_health, trickle_state

SCRIPT = Path(__file__).parent.parent / 'legibility' / 'check_trickle_health.py'


def _ok(*_args, **_kwargs):
    return 0, 'OK: probe passed'


def _fails(message):
    def _runner(*_args, **_kwargs):
        return 1, message
    return _runner


def _raises(exc):
    def _runner(*_args, **_kwargs):
        raise exc
    return _runner


def _check(**kwargs):
    """Run the aggregate with both runners injected and escalation off."""
    kwargs.setdefault('project_id', 'dark_factory')
    kwargs.setdefault('progress_runner', _ok)
    kwargs.setdefault('liveness_runner', _ok)
    kwargs.setdefault('poster', lambda url, envelope: None)
    return check_trickle_health.run_health_check(**kwargs)


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); it is not. A systemd '
        f'ExecStart and every sibling probe in this directory rely on it. '
        f'Run: chmod +x {SCRIPT}'
    )


class TestAggregateVerdict:
    """ONE verdict for the whole invocation, never one per probe — the same
    rule ``run_nightly`` states as "ONE escalation for the whole night"."""

    def test_both_probes_passing_is_healthy(self):
        result = _check()

        assert result.exit_code == 0
        assert result.progress_ok is True
        assert result.liveness_ok is True
        assert 'progress' in result.reason.lower()
        assert 'liveness' in result.reason.lower()

    def test_a_failing_progress_probe_fails_the_aggregate(self):
        result = _check(progress_runner=_fails('ERROR: barren streak'))

        assert result.exit_code == 1
        assert result.progress_ok is False
        assert result.liveness_ok is True

    def test_a_failing_liveness_probe_fails_the_aggregate(self):
        """The mitigation ``nightly::_record_trickle_progress`` claims is
        only real if this fails LOUD."""
        result = _check(liveness_runner=_fails('ERROR: Result=failed'))

        assert result.exit_code == 1
        assert result.progress_ok is True
        assert result.liveness_ok is False

    def test_both_probes_run_even_when_the_first_fails(self):
        """Never short-circuits: the operator needs BOTH answers in one
        place, and a progress failure is often explained by the liveness
        one."""
        result = _check(
            progress_runner=_fails('ERROR: progress detail here'),
            liveness_runner=_fails('ERROR: liveness detail here'),
        )

        assert result.exit_code == 1
        assert result.progress_ok is False
        assert result.liveness_ok is False
        assert 'progress detail here' in result.progress_output
        assert 'liveness detail here' in result.liveness_output

    @pytest.mark.parametrize('exc', [
        pytest.param(FileNotFoundError('no such probe'), id='FileNotFoundError'),
        pytest.param(
            subprocess.TimeoutExpired(cmd=['probe'], timeout=1.0),
            id='TimeoutExpired',
        ),
    ])
    def test_a_probe_that_cannot_RUN_is_a_failed_probe(self, exc):
        """A probe that cannot run is not evidence of health. Never a pass,
        never an uncaught traceback."""
        result = _check(progress_runner=_raises(exc))

        assert result.exit_code == 1
        assert result.progress_ok is False
        assert type(exc).__name__ in result.progress_output, (
            'the reason a probe could not run must be named, not swallowed'
        )


class TestTheProbesItActuallyTargets:
    """THE GAP-2 regression tests. The defect was that nothing referenced
    either probe, so pinning the argv BY CONSTRUCTION is what keeps a
    rename from silently orphaning one again."""

    def test_progress_argv_targets_the_sibling_script(self):
        argv = check_trickle_health.progress_argv(
            'dark_factory', max_barren_runs=3, max_age_hours=72,
            max_failed_runs=2,
        )

        assert argv == [
            sys.executable,
            str(SCRIPT.parent / 'check_trickle_progress.py'),
            'dark_factory', '3', '72', '2',
        ]

    def test_liveness_argv_targets_the_sibling_script(self):
        argv = check_trickle_health.liveness_argv('dark_factory', hours=72)

        assert argv == [
            'bash',
            str(SCRIPT.parent / 'check_trickle_liveness.sh'),
            'dark_factory', '72',
        ]

    def test_both_probe_paths_exist(self):
        assert check_trickle_health.PROGRESS_SCRIPT.is_file()
        assert check_trickle_health.LIVENESS_SCRIPT.is_file()


class TestCli:
    """``main(argv) -> exit-code``, the shape a systemd ExecStart runs."""

    def test_exactly_one_of_config_or_project_id_is_required(self, capsys):
        with pytest.raises(SystemExit):
            check_trickle_health.main([])
        assert 'project-id' in capsys.readouterr().err

    def test_defaults_are_read_from_the_module_not_hardcoded(self):
        """The two halves cannot drift: the probe's default thresholds ARE
        ``trickle_state``'s constants, by reference."""
        parser = check_trickle_health.build_parser()
        defaults = vars(parser.parse_args(['--project-id', 'dark_factory']))

        assert defaults['max_barren_runs'] == trickle_state.DEFAULT_MAX_BARREN_RUNS
        assert defaults['max_failed_runs'] == trickle_state.DEFAULT_MAX_FAILED_RUNS
        assert defaults['max_age_hours'] == 72

    def test_config_only_is_accepted_by_main(self, tmp_path, monkeypatch):
        """``--config`` alone is a valid arity and stays one: the REVIEW
        FIX 1 change is in project-id RESOLUTION, not in argument
        validation."""
        seen = {}
        monkeypatch.setattr(
            check_trickle_health, 'run_health_check',
            lambda **kw: seen.update(kw) or check_trickle_health.HealthResult(
                exit_code=0, progress_ok=True, liveness_ok=True,
                progress_output='', liveness_output='', escalated=False,
                reason='synthetic healthy reason',
            ),
        )
        config_path = _write_project_config(tmp_path, project_id='proj_a')

        assert check_trickle_health.main(['--config', str(config_path)]) == 0

        assert seen['project_id'] is None
        assert seen['config_path'] == str(config_path)

    def test_the_reason_goes_to_stderr_on_a_non_zero_exit(
        self, capsys, monkeypatch
    ):
        monkeypatch.setattr(
            check_trickle_health, 'run_health_check',
            lambda **kw: check_trickle_health.HealthResult(
                exit_code=1, progress_ok=False, liveness_ok=True,
                progress_output='', liveness_output='', escalated=False,
                reason='synthetic failure reason',
            ),
        )

        assert check_trickle_health.main(['--project-id', 'dark_factory']) == 1

        captured = capsys.readouterr()
        assert 'synthetic failure reason' in captured.err
        assert 'synthetic failure reason' not in captured.out

    def test_the_reason_goes_to_stdout_when_healthy(self, capsys, monkeypatch):
        monkeypatch.setattr(
            check_trickle_health, 'run_health_check',
            lambda **kw: check_trickle_health.HealthResult(
                exit_code=0, progress_ok=True, liveness_ok=True,
                progress_output='', liveness_output='', escalated=False,
                reason='synthetic healthy reason',
            ),
        )

        assert check_trickle_health.main(['--project-id', 'dark_factory']) == 0

        captured = capsys.readouterr()
        assert 'synthetic healthy reason' in captured.out
        assert 'synthetic healthy reason' not in captured.err


# ---------------------------------------------------------------------------
# Escalation, and the anti-double-alarm rule
# ---------------------------------------------------------------------------

def _write_project_config(tmp_path, *, project_id, escalation_port=8199):
    """Write a minimal valid legibility.yaml for *project_id* under a fresh
    search root, returning the CONFIG PATH.

    Copied in shape from ``test_install_trickle_timer.py::
    _write_project_config``, per this directory's deliberate
    copy-not-share convention, so ``cfg.escalation_port`` is a real
    resolved value rather than a mock."""
    project_root = tmp_path / 'search-root' / project_id
    legibility_dir = project_root / 'docs' / 'legibility'
    legibility_dir.mkdir(parents=True, exist_ok=True)
    config_path = legibility_dir / 'legibility.yaml'
    config_path.write_text(
        f'project_id: {project_id}\n'
        f'project_root: {project_root}\n'
        f'escalation_port: {escalation_port}\n'
        f'cwd_prefixes:\n'
        f'  - {project_root}\n',
        encoding='utf-8',
    )
    return config_path


def _seed_state(project_id, *, outcomes, last_recorded_at=None):
    """Seed real state through the REAL writer, one night per entry, so the
    suppression rule is exercised against genuine recorder output.

    *last_recorded_at* is the stamp the LAST seeded night carries,
    defaulting to now — which is every pre-existing caller's behaviour,
    byte for byte. Passing an older stamp is how a FROZEN document (a
    recorder that stopped) is seeded without hand-writing one: keeping the
    real writer in the loop is why these tests cannot drift from recorder
    output."""
    from datetime import date

    counters = {
        'productive': (0, dict(selected_count=2)),
        'barren': (0, dict(budget_skipped=4)),
        'failed': (1, dict(selected_count=1)),
    }
    now = last_recorded_at if last_recorded_at is not None else datetime.now(UTC)
    doc = None
    for i, outcome in enumerate(outcomes):
        exit_code, extra = counters[outcome]
        full = dict(
            zero_signal_dropped=0, dedupe_collapsed=0, below_sampling_cut=0,
            budget_skipped=0, selected_count=0,
        )
        full.update(extra)
        doc = trickle_state.record_run(
            project_id,
            target_date=date(2026, 7, 1),
            recorded_at=now - timedelta(days=(len(outcomes) - 1 - i)),
            exit_code=exit_code,
            total_records=sum(full.values()),
            # Passed explicitly, not left to record_run's defaults: without
            # them pyright matches this `**full` spread (a dict[str, int])
            # against the bool-typed `commit_made`/`budget_suppressed`
            # parameters and errors. Mirrors
            # `test_trickle_state.py::_record`, which spreads a counter dict
            # the same way and stays type-clean for exactly this reason.
            commit_made=False,
            budget_suppressed=False,
            **full,
        )
    return doc


class TestEscalation:
    """POST iff the PROGRESS probe failed — except on the exact run where
    the nightly's own edge-triggered barren-streak escalation already
    fired.

    The escalation predicate is deliberately NARROWER than the exit-code
    one: the exit code is a verdict an operator or predicate reads on
    demand, while an escalation is an INTERRUPT.
    """

    def _posted(self, tmp_path, **kwargs):
        envelopes = []
        config_path = _write_project_config(tmp_path, project_id='dark_factory')
        kwargs.setdefault('progress_runner', _ok)
        kwargs.setdefault('liveness_runner', _ok)
        result = check_trickle_health.run_health_check(
            project_id='dark_factory',
            config_path=config_path,
            poster=lambda url, envelope: envelopes.append((url, envelope)),
            **kwargs,
        )
        return result, envelopes

    def test_a_failing_progress_probe_posts_exactly_one_envelope(self, tmp_path):
        result, envelopes = self._posted(
            tmp_path, progress_runner=_fails('ERROR: barren streak'),
        )

        assert result.exit_code == 1
        assert result.escalated is True
        assert len(envelopes) == 1, 'ONE envelope for the whole invocation'

        _url, envelope = envelopes[0]
        assert envelope['params']['name'] == 'escalate_info'
        arguments = envelope['params']['arguments']
        assert arguments['category'] == 'infra_issue'
        assert arguments['severity'] == 'info'
        assert arguments['agent_role'] == 'legibility-trickle-health'
        assert arguments['task_id'] == 'legibility-trickle-health-dark_factory', (
            "distinct from nightly's legibility-trickle-<project_id>, so the "
            'two escalation histories stay separately readable'
        )

    def test_the_detail_carries_both_probes_output(self, tmp_path):
        _result, envelopes = self._posted(
            tmp_path,
            progress_runner=_fails('PROGRESS-MARKER barren streak'),
            liveness_runner=_fails('LIVENESS-MARKER Result=failed'),
        )

        detail = envelopes[0][1]['params']['arguments']['detail']
        assert 'PROGRESS-MARKER' in detail
        assert 'LIVENESS-MARKER' in detail
        assert 'check_trickle_progress.py' in detail, 'name the re-run command'
        assert 'check_trickle_liveness.sh' in detail

    def test_a_liveness_only_failure_posts_nothing(self, tmp_path):
        """A unit that RAN and FAILED is already owned by the nightly's own
        decision-8 escalation for that same run, and by
        ``check_trickle_liveness.sh``'s ``Result != success`` gate now that
        something finally runs it. Posting here would be exactly the
        double-alarm this task forbids.

        NO COVERAGE IS LOST, and the enumeration is what makes that
        checkable: a unit that never ran leaves the state ``missing``; one
        that stopped firing leaves ``recorded_at`` stale; one that fails
        repeatedly becomes a ``consecutive_failed_runs`` streak — all three
        fail the PROGRESS probe and do post."""
        result, envelopes = self._posted(
            tmp_path, liveness_runner=_fails('ERROR: Result=failed'),
        )

        assert result.exit_code == 1, 'still loud'
        assert result.escalated is False
        assert envelopes == []

    def test_two_green_probes_post_nothing(self, tmp_path):
        result, envelopes = self._posted(tmp_path)

        assert result.exit_code == 0
        assert result.escalated is False
        assert envelopes == []

    def test_silent_on_the_exact_run_the_nightly_escalated(self, tmp_path):
        """``nightly::_escalate_barren_streak`` is EDGE-triggered by exact
        equality and fired for this very run; posting here duplicates it."""
        _seed_state('dark_factory', outcomes=['barren'] * 3)

        result, envelopes = self._posted(
            tmp_path,
            progress_runner=_fails('ERROR: barren streak'),
            max_barren_runs=3,
        )

        assert result.exit_code == 1
        assert result.escalated is False
        assert envelopes == []

    def test_a_frozen_threshold_document_does_not_suppress_forever(
        self, tmp_path
    ):
        """THE regression test.

        The suppression used to be a predicate over the recorded document
        ALONE, with no freshness condition. Measured in this worktree: a
        ``['barren'] * 3`` document stamped 30 days ago gives
        ``exit_code=1, escalated=False, 0 envelopes`` — and since the
        document is frozen, that repeats EVERY NIGHT indefinitely. A
        recorder that dies on a night landing exactly on the threshold
        (timer disabled, unit 203/EXEC, recorder crash) then fails the
        progress probe's STALENESS branch forever while never once
        posting: the journal stays loud, and the only interrupt an
        operator actually receives is permanently suppressed."""
        _seed_state(
            'dark_factory', outcomes=['barren'] * 3,
            last_recorded_at=datetime.now(UTC) - timedelta(days=30),
        )

        result, envelopes = self._posted(
            tmp_path,
            progress_runner=_fails('ERROR: stale recorder'),
            max_barren_runs=3,
        )

        assert result.exit_code == 1
        assert len(envelopes) == 1

    def test_the_suppression_still_holds_for_a_fresh_threshold_document(
        self, tmp_path
    ):
        """The behaviour the suppression EXISTS for, pinned alongside the
        fix so the fix cannot be over-applied into the very double-alarm
        with ``nightly::_escalate_barren_streak`` it was built to avoid."""
        _seed_state(
            'dark_factory', outcomes=['barren'] * 3,
            last_recorded_at=datetime.now(UTC),
        )

        result, envelopes = self._posted(
            tmp_path,
            progress_runner=_fails('ERROR: barren streak'),
            max_barren_runs=3,
        )

        assert result.exit_code == 1
        assert envelopes == []

    @pytest.mark.parametrize('offset_hours, expected_envelopes', [
        pytest.param(71, 0, id='inside-the-window-suppressed'),
        pytest.param(73, 1, id='outside-the-window-posts'),
    ])
    def test_the_suppression_window_is_the_freshness_window(
        self, tmp_path, offset_hours, expected_envelopes
    ):
        """The boundary, asserted as a RELATION to ``max_age_hours`` rather
        than as a literal.

        The suppression can never outlive the nightly's edge-trigger: the
        moment a record is stale enough for the progress probe to call it
        stale, it is also too stale to be evidence that anyone alarmed."""
        _seed_state(
            'dark_factory', outcomes=['barren'] * 3,
            last_recorded_at=datetime.now(UTC) - timedelta(hours=offset_hours),
        )

        _result, envelopes = self._posted(
            tmp_path,
            progress_runner=_fails('ERROR: barren streak'),
            max_barren_runs=3,
            max_age_hours=72,
        )

        assert len(envelopes) == expected_envelopes

    def test_an_unparseable_recorded_at_is_post_worthy_not_suppression_grounds(
        self, tmp_path
    ):
        """Consistent with the existing posture that ``missing`` and
        ``malformed`` are VERDICTS rather than suppression grounds: a
        document whose freshness cannot be assessed must never be treated
        as fresh."""
        _seed_state('dark_factory', outcomes=['barren'] * 3)
        path = trickle_state.trickle_state_path('dark_factory')
        doc = json.loads(path.read_text(encoding='utf-8'))
        doc['recorded_at'] = 'not-a-timestamp'
        path.write_text(json.dumps(doc), encoding='utf-8')

        _result, envelopes = self._posted(
            tmp_path,
            progress_runner=_fails('ERROR: unparseable recorded_at'),
            max_barren_runs=3,
        )

        assert len(envelopes) == 1

    def test_posts_once_the_nightly_has_gone_silent(self, tmp_path):
        """At ``> max_barren_runs`` the nightly is edge-triggered and
        silent by design, so this probe takes over."""
        _seed_state('dark_factory', outcomes=['barren'] * 4)

        result, envelopes = self._posted(
            tmp_path,
            progress_runner=_fails('ERROR: barren streak'),
            max_barren_runs=3,
        )

        assert result.exit_code == 1
        assert len(envelopes) == 1

    def test_the_barren_edge_suppression_does_not_apply_to_a_failed_streak(
        self, tmp_path
    ):
        """A failed streak is a different alarm with a different remedy;
        the nightly never fires for it at all."""
        _seed_state('dark_factory', outcomes=['failed', 'failed'])

        result, envelopes = self._posted(
            tmp_path,
            progress_runner=_fails('ERROR: failed streak'),
            max_barren_runs=3,
        )

        assert result.exit_code == 1
        assert len(envelopes) == 1

    def test_a_missing_state_file_is_post_worthy_not_suppression_grounds(
        self, tmp_path
    ):
        result, envelopes = self._posted(
            tmp_path, progress_runner=_fails('ERROR: never recorded a run'),
        )

        assert len(envelopes) == 1

    def test_a_raising_poster_is_swallowed_best_effort(self, tmp_path, caplog):
        """Mirrors ``nightly::post_escalation``'s contract: a down
        escalation server must never mask the verdict."""
        def _boom(url, envelope):
            raise RuntimeError('escalation server is down')

        config_path = _write_project_config(tmp_path, project_id='dark_factory')

        with caplog.at_level('WARNING'):
            result = check_trickle_health.run_health_check(
                project_id='dark_factory',
                config_path=config_path,
                progress_runner=_fails('ERROR: barren streak'),
                liveness_runner=_ok,
                poster=_boom,
            )

        assert result.exit_code == 1, 'the verdict is unchanged'
        assert result.escalated is False
        assert len([r for r in caplog.records if r.levelname == 'WARNING']) == 1

    def test_an_absent_config_degrades_to_a_loud_verdict_not_a_traceback(
        self, tmp_path
    ):
        result = check_trickle_health.run_health_check(
            project_id='dark_factory',
            config_path=tmp_path / 'nope' / 'legibility.yaml',
            progress_runner=_fails('ERROR: barren streak'),
            liveness_runner=_ok,
            poster=lambda url, envelope: None,
        )

        assert result.exit_code == 1
        assert result.escalated is False

    def test_the_default_poster_is_not_reached_when_one_is_injected(
        self, tmp_path, install_fake_httpx
    ):
        """Belt to the injection braces: no real POST to localhost may
        escape even if the default poster is somehow reached."""
        posts = []
        install_fake_httpx(lambda *a, **kw: posts.append((a, kw)))

        result, envelopes = self._posted(
            tmp_path, progress_runner=_fails('ERROR: barren streak'),
        )

        assert len(envelopes) == 1
        assert posts == [], 'the injected poster must be the only one used'
        assert result.escalated is True


class TestProjectIdResolution:
    """Which project id actually REACHES the two probes.

    THE MEASURED DEFECT, reproduced in this worktree before these tests
    were written: ``main()`` accepts ``--config`` without ``--project-id``,
    and ``run_health_check`` then handed ``project_id or ''`` to both
    probes. With a valid ``legibility.yaml`` for a HEALTHY project,
    ``check_trickle_health.py --config <yaml>`` exits 1 with ``legibility
    trickle health DEGRADED for : progress=FAILED liveness=FAILED`` — the
    progress probe resolving ``.../legibility//trickle-state.json`` and
    reporting ``missing``, the liveness probe interrogating the literal
    TEMPLATE unit ``legibility-trickle@.service``. With a reachable server
    it also POSTS one envelope stamped
    ``task_id=legibility-trickle-health-<project>``: a fabricated alarm
    attributed to a healthy project.

    This is the same divergence class GAP 3 closed at the path layer — a
    probe that guesses at the project id reads a different state file than
    the writer wrote.
    """

    def _recorder(self, calls, *, returncode=0, output='OK: probe passed'):
        """A runner that records the ``project_id`` positional it is handed."""
        def _runner(project_id, **_kwargs):
            calls.append(project_id)
            return returncode, output
        return _runner

    def test_config_only_sends_the_configs_project_id_to_both_probes(
        self, tmp_path
    ):
        """THE regression test."""
        progress_calls, liveness_calls = [], []
        config_path = _write_project_config(tmp_path, project_id='proj_a')

        check_trickle_health.run_health_check(
            project_id=None,
            config_path=config_path,
            progress_runner=self._recorder(progress_calls),
            liveness_runner=self._recorder(liveness_calls),
            poster=lambda url, envelope: None,
        )

        assert progress_calls == ['proj_a']
        assert liveness_calls == ['proj_a']
        # Asserted separately and deliberately: '' is the sentinel this fix
        # deletes, and asserting only the positive would still pass if a
        # future change reintroduced a DIFFERENT falsy default.
        assert '' not in progress_calls
        assert '' not in liveness_calls

    def test_config_only_on_a_healthy_project_is_quiet(self, tmp_path):
        """The operator-visible consequence, end to end."""
        envelopes = []
        config_path = _write_project_config(tmp_path, project_id='proj_a')

        result = check_trickle_health.run_health_check(
            project_id=None,
            config_path=config_path,
            progress_runner=_ok,
            liveness_runner=_ok,
            poster=lambda url, envelope: envelopes.append(envelope),
        )

        assert result.exit_code == 0
        assert result.escalated is False
        assert envelopes == []
        assert 'proj_a' in result.reason
        assert 'for :' not in result.reason, (
            'the exact legibility artefact the empty project id produced'
        )

    def test_project_id_takes_precedence_over_the_configs(self, tmp_path):
        """Pins the precedence DIRECTION.

        ``scripts/legibility/check_transcript_persistence.py::_load_config``
        resolves ``--config`` FIRST for the CONFIG, but never had to
        arbitrate a project id — so the rule is stated here rather than
        assumed from the sibling."""
        progress_calls, liveness_calls = [], []
        config_path = _write_project_config(tmp_path, project_id='proj_a')

        check_trickle_health.run_health_check(
            project_id='dark_factory',
            config_path=config_path,
            progress_runner=self._recorder(progress_calls),
            liveness_runner=self._recorder(liveness_calls),
            poster=lambda url, envelope: None,
        )

        assert progress_calls == ['dark_factory']
        assert liveness_calls == ['dark_factory']

    def test_an_unresolvable_config_without_a_project_id_runs_neither_probe(
        self, tmp_path
    ):
        """There is no project id to probe and no escalation port to post
        to, so running the probes could only manufacture the same false
        DEGRADED this fix removes.

        Non-zero rather than 0 because ``_invoke``'s own rule — a probe
        that cannot run is not evidence of health — applies a fortiori to a
        probe that was never invoked."""
        envelopes = []
        progress_calls, liveness_calls = [], []
        config_path = tmp_path / 'nope' / 'legibility.yaml'

        result = check_trickle_health.run_health_check(
            project_id=None,
            config_path=config_path,
            progress_runner=self._recorder(progress_calls),
            liveness_runner=self._recorder(liveness_calls),
            poster=lambda url, envelope: envelopes.append(envelope),
        )

        assert result.exit_code == 1
        assert result.escalated is False
        assert envelopes == []
        assert progress_calls == [], 'the progress probe must never be invoked'
        assert liveness_calls == [], 'the liveness probe must never be invoked'
        assert str(config_path) in result.reason, (
            'the journal line must distinguish a misconfigured invocation '
            'from a genuine probe failure'
        )

    def test_the_escalation_path_never_sees_an_empty_project_id(self, tmp_path):
        """The half of the defect that reaches an operator's inbox rather
        than a journal."""
        envelopes = []
        config_path = _write_project_config(tmp_path, project_id='proj_a')

        result = check_trickle_health.run_health_check(
            project_id=None,
            config_path=config_path,
            progress_runner=_fails('ERROR: never recorded a run'),
            liveness_runner=_ok,
            poster=lambda url, envelope: envelopes.append(envelope),
        )

        assert result.exit_code == 1
        assert len(envelopes) == 1
        arguments = envelopes[0]['params']['arguments']
        assert arguments['task_id'] == 'legibility-trickle-health-proj_a'
        assert 'for :' not in arguments['summary']
        assert 'for :' not in arguments['detail']
