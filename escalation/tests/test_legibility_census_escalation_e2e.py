"""END-TO-END: every legibility alarm lands a retrievable escalation.

`scripts/legibility/alarm.py` is the one poster for all four alarm sources
(census, trickle, trickle-health, transcript-check). These tests drive it,
and each source's own entrypoint, against a REAL escalation MCP server (the
`serve_escalation_mcp` fixture), then read the escalation back out of the
queue BY ID. A test that only observed the poster being called would stay
green while the queue stayed empty.

WHY THIS FILE LIVES IN `escalation/tests/` rather than `scripts/tests/`:
`scripts/tests/` runs under `uv run --project shared`, and `shared` does not
depend on fastmcp, so an in-process real-server test there would fail to
import or be `importorskip`-ed into a silent skip. Adding fastmcp+escalation
to `shared`'s deps would invert the package dependency direction. Under
`cd escalation && uv run pytest tests/` both sides import.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import sys
from pathlib import Path

import pytest

# --- scoped legibility-scripts path setup ---
#
# Deliberately HERE and not in this directory's conftest.py: this module is the
# only consumer.
#
# APPEND, never insert(0, ...): repo and stdlib names must keep winning even
# once this module has been imported (sys.path edits are process-global and
# outlive collection of this file).
#
# `scripts/` provides the `legibility` PACKAGE. `scripts/legibility/` is needed
# only for a flat sibling name a transitively imported module still uses
# (census -> coder -> `import codebook`). The `shared` stub installed by
# conftest.py is compatible: its `__path__` points at the real
# `shared/src/shared`.
if (_SCRIPTS_DIR := str(Path(__file__).resolve().parents[2] / 'scripts')) not in sys.path:
    sys.path.append(_SCRIPTS_DIR)
if (_LEGIBILITY_DIR := str(Path(_SCRIPTS_DIR) / 'legibility')) not in sys.path:
    sys.path.append(_LEGIBILITY_DIR)

# The package form keeps ONE `legibility.alarm` module object, the one that
# holds `own_endpoint`'s grant.
from legibility import alarm, census, check_transcript_persistence, check_trickle_health, nightly

# WHY `with alarm.own_endpoint():` AROUND EVERY REAL POST BELOW.
#
# alarm refuses every MCP post made from inside a pytest process, because a
# test-minted escalation carries a real job's task_id and agent_role and is
# indistinguishable at triage from a genuine failure. `serve_escalation_mcp`
# is this suite's OWN server, on an ephemeral port over a tmp_path queue, so
# nothing reaches the operator's queue. Nothing distinguishes the two cases
# automatically, so each test declares the grant around exactly the call that
# posts to its server, and no wider.

_RAISED_MESSAGE = 'codebook merge produced an invalid codebook'
_PROJECT_ID = 'e2e_census_project'


def _write_legibility_yaml(project_root, *, escalation_port):
    """Write a minimal valid legibility.yaml at the default location.

    Same shape (and the same deliberate plain-text-lines approach, not a
    `yaml.safe_dump` round trip) as `_write_legibility_yaml` in
    `scripts/tests/test_legibility_census.py` -- kept independent of the
    module under test's own YAML writer. The only difference that matters
    here is `escalation_port`, pointed at the live test server's ephemeral
    port instead of the fleet's :8103.
    """
    config_path = project_root / 'docs' / 'legibility' / 'legibility.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        '\n'.join([
            f'project_id: {_PROJECT_ID}',
            f'project_root: {project_root}',
            f'escalation_port: {escalation_port}',
            'cwd_prefixes:',
            f'  - {project_root}',
        ]) + '\n',
        encoding='utf-8',
    )
    return config_path


def _commit_all(repo):
    """Make *repo* a git repo with one commit: census.main resolves as_of_sha
    from the censused project's HEAD and exits 1 before run_census otherwise."""
    for args in (
        ('init', '-q', '-b', 'main'),
        ('config', 'user.email', 'test@example.com'),
        ('config', 'user.name', 'Test'),
        ('add', '-A'),
        ('commit', '-q', '-m', 'fixture'),
    ):
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True)


def _live_config_path(tmp_path, port):
    """A tmp project whose legibility.yaml points at *port*; its config path."""
    project_root = tmp_path / 'project'
    project_root.mkdir()
    return _write_legibility_yaml(project_root, escalation_port=port)


def _live_cfg(tmp_path, port):
    """A LegibilityConfig pointed at *port*."""
    return census.config.load_config(_live_config_path(tmp_path, port))


def _the_one_escalation(queue):
    """The queue's only pending escalation, resolved back through the queue's
    own lookup by id rather than trusted from the pending listing."""
    pending = queue.get_pending()
    assert len(pending) == 1, (
        f'expected exactly one escalation in the queue, got {len(pending)}: '
        f'{[e.id for e in pending]}'
    )
    fetched = queue.get(pending[0].id)
    assert fetched is not None, f'escalation {pending[0].id} is not retrievable by id'
    return fetched


def _assert_raised_by(fetched, source):
    assert fetched.task_id == f'legibility-{source}-{_PROJECT_ID}'
    assert fetched.agent_role == f'legibility-{source}'
    assert (fetched.category, fetched.severity) == ('infra_issue', 'info')


@pytest.mark.parametrize('source', list(alarm.AlarmSource), ids=str)
def test_every_alarm_source_lands_a_retrievable_escalation(
    source, tmp_path, serve_escalation_mcp,
):
    _base_url, port, queue = serve_escalation_mcp(tmp_path / 'queue')
    project = alarm.AlarmProject.of(_live_cfg(tmp_path, port))

    with alarm.own_endpoint():
        landed = alarm.raise_alarm(
            project, source, f'legibility {source} e2e ({_PROJECT_ID})', 'detail',
            level=logging.WARNING,
        )

    assert landed is True, 'raise_alarm reported failure against a live server'
    fetched = _the_one_escalation(queue)
    _assert_raised_by(fetched, source)
    assert fetched.summary == f'legibility {source} e2e ({_PROJECT_ID})'
    assert fetched.detail == 'detail'


def test_an_alarm_to_a_down_server_reports_not_landed_without_raising(tmp_path, caplog):
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    project = alarm.AlarmProject.of(_live_cfg(tmp_path, port))

    with caplog.at_level(logging.WARNING, logger='legibility.alarm'), alarm.own_endpoint():
        landed = alarm.raise_alarm(
            project, alarm.AlarmSource.CENSUS, 'summary', 'detail', level=logging.WARNING,
        )

    assert landed is False
    assert any(
        r.name == 'legibility.alarm' and r.levelno == logging.WARNING for r in caplog.records
    ), 'a post that did not land must say so in the journal'


def test_census_hard_failure_lands_a_retrievable_escalation(
    tmp_path, serve_escalation_mcp, monkeypatch,
):
    """A forced census failure exits 1 and leaves its fail-loud boundary's
    alarm in a real queue, retrievable by id."""
    _base_url, port, queue = serve_escalation_mcp(tmp_path / 'queue')
    project_root = _live_config_path(tmp_path, port).parents[2]
    _commit_all(project_root)

    def _raising_run_census(**kwargs):
        raise RuntimeError(_RAISED_MESSAGE)

    monkeypatch.setattr(census, 'run_census', _raising_run_census)

    with alarm.own_endpoint():
        exit_code = census.main(['--project-root', str(project_root), '--force'])

    assert exit_code == 1, 'the alarm never masks the authoritative exit code'
    fetched = _the_one_escalation(queue)
    _assert_raised_by(fetched, alarm.AlarmSource.CENSUS)
    assert fetched.summary.startswith(
        f'legibility census run failed ({_PROJECT_ID}): RuntimeError: {_RAISED_MESSAGE}'
    )
    assert _RAISED_MESSAGE in fetched.detail


def test_nightly_crash_lands_a_retrievable_escalation(tmp_path, serve_escalation_mcp):
    """A stage that raises ends the night with one CRASHED alarm in a real
    queue. The recorder raises because the default one writes the real
    passwd-resolved trickle state; nightly swallows a recorder failure."""
    _base_url, port, queue = serve_escalation_mcp(tmp_path / 'queue')
    config_path = _live_config_path(tmp_path, port)
    launched = []

    def _raising_selector(*args, **kwargs):
        raise RuntimeError(_RAISED_MESSAGE)

    def _raising_recorder(*args, **kwargs):
        raise RuntimeError('this test writes no trickle state')

    with alarm.own_endpoint():
        result = nightly.run_nightly(
            config_path=config_path,
            pass_kind=nightly.PassKind.MANUAL,
            selector=_raising_selector,
            recorder=_raising_recorder,
            census_launcher=lambda *args, **kwargs: launched.append(args),
        )

    assert result.exit_code == 1
    assert result.escalated is True
    assert [a.kind for a in result.alarms] == [nightly.AlarmKind.CRASHED]
    assert launched == [], 'a crashed night launches no census'
    fetched = _the_one_escalation(queue)
    _assert_raised_by(fetched, alarm.AlarmSource.TRICKLE)
    assert fetched.summary == f'legibility trickle run crashed: RuntimeError: {_RAISED_MESSAGE}'
    assert _RAISED_MESSAGE in fetched.detail


def test_trickle_health_finding_lands_a_retrievable_escalation(tmp_path, serve_escalation_mcp):
    _base_url, port, queue = serve_escalation_mcp(tmp_path / 'queue')
    cfg = _live_cfg(tmp_path, port)
    result = check_trickle_health.HealthResult(
        exit_code=1,
        progress_ok=False,
        liveness_ok=True,
        progress_output='last run FAILED',
        liveness_output='',
        census_ok=True,
        census_output='',
        escalated=False,
        reason='the progress probe failed',
    )
    rerun = (['check_trickle_progress.py'], ['check_trickle_liveness.sh'], ['check_census_progress.py'])

    with alarm.own_endpoint():
        landed = check_trickle_health.post_health_finding(
            cfg, result, rerun, project_id=_PROJECT_ID,
        )

    assert landed is True, 'post_health_finding reported failure against a live server'
    fetched = _the_one_escalation(queue)
    _assert_raised_by(fetched, alarm.AlarmSource.TRICKLE_HEALTH)
    assert 'progress probe(s) FAILED' in fetched.summary
    assert 'last run FAILED' in fetched.detail


def test_transcript_loss_alarm_lands_a_retrievable_escalation(tmp_path, serve_escalation_mcp):
    """`post_findings` takes a `Sequence[MissingTranscript]`, so a minimal
    non-empty findings list is what reaches the post."""
    _base_url, port, queue = serve_escalation_mcp(tmp_path / 'queue')
    cfg = _live_cfg(tmp_path, port)
    finding = check_transcript_persistence.MissingTranscript(
        session_slug='sess-lost-e2e',
        cwd=str(tmp_path / 'project'),
        prompt_prefix='Diagnose the stuck reconciliation.',
        start_ts='2026-08-10T00:00:00+00:00',
        exit_code=0,
        expected_dir=tmp_path / 'projects' / 'enc',
    )

    with alarm.own_endpoint():
        landed = check_transcript_persistence.post_findings(cfg, [finding])

    assert landed is True, 'post_findings reported failure against a live server'
    fetched = _the_one_escalation(queue)
    _assert_raised_by(fetched, alarm.AlarmSource.TRANSCRIPT_CHECK)
    assert 'sess-lost-e2e' in fetched.detail
