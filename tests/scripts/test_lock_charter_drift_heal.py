"""Contract for ``scripts/lock_charter_drift_heal.py`` — the router of the nightly drift reconciliation.

The drift checker answers "do reify's and dark-factory's lock-charter mirrors
agree?" with three exit codes.  This router delivers each answer to the one
owner that can act on it: DRIFT to exactly one open heal task, deduped on a
stable metadata key; UNAVAILABLE to one ``escalate_info`` in a fixed slot; OK
to nobody.

Both MCP servers are real sockets (``shared.testing_mcp_stub.StubMcpServer``)
answered by the small stateful fakes below, so the router is driven over its
real transport.  Every report comes from the real
``check_lock_charter_drift`` reconciliation over a reify sibling planted under
``tmp_path`` by ``lock_charter_sibling``; the autouse fixture scrubs the
ambient ``REIFY_ROOT`` and ``FUSED_MEMORY_MCP_URL`` so the host cannot steer a
result.
"""

from __future__ import annotations

import copy
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import check_lock_charter_drift as drift
import lock_charter_drift_heal as heal
import lock_charter_sibling
import pytest
from shared.locking import EXTENSIONLESS_FILENAMES, FILE_EXTENSIONS
from shared.task_statuses import ACTIVE
from shared.testing_mcp_stub import StubMcpServer

NOW = datetime(2026, 11, 24, 1, 45, tzinfo=UTC)
REPO_ROOT = Path(__file__).resolve().parents[2]
DROPPED_EXTENSION = 'toml'


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    monkeypatch.delenv('REIFY_ROOT', raising=False)
    monkeypatch.delenv('FUSED_MEMORY_MCP_URL', raising=False)


class FakeTaskStore:
    """The four fused-memory tools the router may call, over one in-memory task table."""

    def __init__(self, tasks: Sequence[Mapping[str, Any]] = ()) -> None:
        self.tasks: dict[str, dict[str, Any]] = {str(t['id']): copy.deepcopy(dict(t)) for t in tasks}
        self.held: set[str] = set()
        self.fail_submit = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._next_id = 1000

    def calls_to(self, tool: str) -> list[dict[str, Any]]:
        return [arguments for name, arguments in self.calls if name == tool]

    def answer(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, copy.deepcopy(dict(arguments))))
        return getattr(self, f'_{tool}')(arguments)

    def _find_tasks_by_metadata(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        statuses = arguments.get('statuses')
        rows = [
            {'task_id': task_id, 'status': task['status']}
            for task_id, task in self.tasks.items()
            if task.get('metadata', {}).get(arguments['key']) == arguments.get('value')
            and (statuses is None or task['status'] in statuses)
        ]
        return {'results': rows, 'total': len(rows), 'has_more': False}

    def _submit_task(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.fail_submit:
            return {'error': 'backlog is full', 'error_type': 'BacklogFull'}
        task_id = str(self._next_id)
        self._next_id += 1
        status = 'deferred' if arguments.get('planning_mode') else 'pending'
        self.tasks[task_id] = {'id': task_id, 'status': status, 'metadata': dict(arguments.get('metadata', {}))}
        return {'task_id': task_id, 'status': status, 'planning_mode': bool(arguments.get('planning_mode'))}

    def _commit_planning(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        results = []
        for task_id in str(arguments['task_ids']).split(','):
            if task_id in self.held:
                results.append({'task_id': task_id, 'result': {
                    'success': False, 'error': 'deferral_invalid', 'task_id': task_id,
                    'reason_code': 'not_planning_hold', 'field': 'deferral.kind',
                    'value': 'until_condition', 'hint': 'a human hold is not released by commit_planning',
                }})
            else:
                self.tasks[task_id]['status'] = 'pending'
                results.append({'task_id': task_id, 'result': {'success': True}})
        return {'success': all(r['result']['success'] for r in results), 'results': results}

    def _get_task(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(self.tasks[str(arguments['id'])])


class FakeEscalationServer:
    """``get_pending_escalations`` and ``escalate_info`` over one in-memory queue."""

    def __init__(self, pending_slots: Sequence[str] = ()) -> None:
        self.pending = [{'id': f'esc-{n}', 'task_id': slot} for n, slot in enumerate(pending_slots)]
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def calls_to(self, tool: str) -> list[dict[str, Any]]:
        return [arguments for name, arguments in self.calls if name == tool]

    def answer(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append((tool, copy.deepcopy(dict(arguments))))
        if tool == 'get_pending_escalations':
            return {'result': copy.deepcopy(self.pending)}
        if tool == 'escalate_info':
            record = {'id': f'esc-{len(self.pending)}', 'task_id': arguments['task_id']}
            self.pending.append(record)
            return {'escalation_id': record['id'], 'status': 'pending'}
        return {'error': f'unexpected tool {tool}'}


@pytest.fixture
def store():
    return FakeTaskStore()


@pytest.fixture
def queue():
    return FakeEscalationServer()


@pytest.fixture
def servers(store, queue):
    with (
        StubMcpServer(stateful=False, tools=store.answer) as fused_memory,
        StubMcpServer(stateful=True, tools=queue.answer) as escalation,
    ):
        yield fused_memory, escalation


@pytest.fixture
def ports(servers, tmp_path):
    fused_memory, escalation = servers
    return heal.build_ports(tmp_path / 'df', fused_memory.url, escalation.url)


def _project_root(tmp_path: Path) -> Path:
    return tmp_path / 'df'


def _main_argv(tmp_path: Path, servers, *extra: str) -> list[str]:
    fused_memory, escalation = servers
    return [
        '--project-root', str(_project_root(tmp_path)),
        '--fused-memory-url', fused_memory.url,
        '--escalation-url', escalation.url,
        *extra,
    ]


def _owner(task_id: str, status: str) -> dict[str, Any]:
    return {'id': task_id, 'status': status, 'metadata': {heal.HEAL_KEY: heal.HEAL_KEY_VALUE}}


def _drift_report(tmp_path: Path) -> dict[str, Any]:
    """A real reconciliation drifting in every part and in both directions."""
    assert DROPPED_EXTENSION in FILE_EXTENSIONS
    repo = lock_charter_sibling.plant_tracked_repo(
        tmp_path / 'reify', ['a/b.py', 'x/y.proto', 'scripts/canon-bin/guard/newtool', 'tools/cargo/x.py'],
    )
    lock_charter_sibling.plant_reify_guard(
        repo,
        extensions=sorted(FILE_EXTENSIONS - {DROPPED_EXTENSION}),
        extensionless=sorted([*EXTENSIONLESS_FILENAMES, 'newtool']),
    )
    report = drift.check(repo)
    assert report['exit_code'] == drift.EXIT_DRIFT
    return report


def _simple_drift_report(tmp_path: Path) -> dict[str, Any]:
    report = drift.check(lock_charter_sibling.plant_divergent_reify(tmp_path / 'reify'))
    assert report['exit_code'] == drift.EXIT_DRIFT
    return report


def _unavailable_report(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    monkeypatch.setenv('REIFY_ROOT', str(tmp_path / 'no-such-reify'))
    report = drift.reconcile()
    assert report['exit_code'] == drift.EXIT_UNAVAILABLE
    return report


class TestDriftFilesOneHealTask:
    def test_drift_with_no_open_owner_files_and_commits_one_task(self, tmp_path, store, ports):
        report = _drift_report(tmp_path)

        result = heal.route(report, ports, project_root=_project_root(tmp_path), now=NOW)

        (submitted,) = store.calls_to('submit_task')
        assert submitted['planning_mode'] is True
        assert submitted['priority'] == 'medium'
        assert submitted['metadata'][heal.HEAL_KEY] == heal.HEAL_KEY_VALUE
        assert submitted['metadata']['source'] == heal.AGENT_ROLE
        assert submitted['metadata']['files'] == list(heal.HEAL_SITES)
        (committed,) = store.calls_to('commit_planning')
        assert committed['task_ids'] == '1000'
        assert result.outcome is heal.Outcome.FILED
        assert result.task_id == '1000'
        assert 'filed 1000' in result.line()

    def test_description_names_every_drifted_name_by_direction_and_every_site(self, tmp_path, store, ports):
        heal.route(_drift_report(tmp_path), ports, project_root=_project_root(tmp_path), now=NOW)

        (submitted,) = store.calls_to('submit_task')
        description = submitted['description']
        only_in_reify, _, only_in_dark_factory = description.partition('Only in dark-factory')
        assert 'Only in reify' in only_in_reify
        for reify_name in ('newtool', 'proto', 'scripts/canon-bin/guard/newtool'):
            assert reify_name in only_in_reify, reify_name
        for dark_factory_name in (DROPPED_EXTENSION, 'cargo'):
            assert dark_factory_name in only_in_dark_factory, dark_factory_name
        for site in heal.HEAL_SITES:
            assert site in description, site
        assert 'scripts/check_lock_charter_drift.py' in description
        assert len(submitted['title']) <= 100
        assert submitted['title'].startswith('Mirror reify lock-charter drift into dark-factory: ')

    def test_open_owner_is_adopted_without_filing(self, tmp_path, store, ports):
        store.tasks['900'] = _owner('900', 'pending')

        result = heal.route(_simple_drift_report(tmp_path), ports, project_root=_project_root(tmp_path), now=NOW)

        assert store.calls_to('submit_task') == []
        assert store.calls_to('commit_planning') == []
        (lookup,) = store.calls_to('find_tasks_by_metadata')
        assert lookup['key'] == heal.HEAL_KEY
        assert lookup['value'] == heal.HEAL_KEY_VALUE
        assert result.outcome is heal.Outcome.ADOPTED
        assert result.task_id == '900'
        assert 'adopted 900' in result.line()

    def test_deferred_owner_is_recommitted_not_refiled(self, tmp_path, store, ports):
        store.tasks['900'] = _owner('900', 'deferred')

        result = heal.route(_simple_drift_report(tmp_path), ports, project_root=_project_root(tmp_path), now=NOW)

        assert store.calls_to('submit_task') == []
        (committed,) = store.calls_to('commit_planning')
        assert committed['task_ids'] == '900'
        assert result.outcome is heal.Outcome.ADOPTED
        assert store.tasks['900']['status'] == 'pending'

    def test_a_human_hold_refusing_the_recommit_is_reported_not_failed(self, tmp_path, store, ports):
        store.tasks['900'] = _owner('900', 'deferred')
        store.held.add('900')

        result = heal.route(_simple_drift_report(tmp_path), ports, project_root=_project_root(tmp_path), now=NOW)

        assert store.calls_to('submit_task') == []
        assert result.outcome is heal.Outcome.ADOPTED
        assert result.task_id == '900'
        assert 'not_planning_hold' in result.line()

    def test_a_done_owner_does_not_suppress_a_fresh_filing(self, tmp_path, store, ports):
        store.tasks['900'] = _owner('900', 'done')

        result = heal.route(_simple_drift_report(tmp_path), ports, project_root=_project_root(tmp_path), now=NOW)

        (lookup,) = store.calls_to('find_tasks_by_metadata')
        assert lookup['statuses'] == sorted(ACTIVE)
        assert len(store.calls_to('submit_task')) == 1
        assert result.outcome is heal.Outcome.FILED
        assert result.task_id != '900'


class TestOkAndUnavailable:
    def test_ok_calls_neither_server(self, tmp_path, servers, ports):
        report = {'status': 'ok', 'exit_code': drift.EXIT_OK, 'parts': {}, 'detail': 'mirrors agree'}

        result = heal.route(report, ports, project_root=_project_root(tmp_path), now=NOW)

        assert result.outcome is heal.Outcome.OK
        for server in servers:
            assert server.requests == ()

    def test_unavailable_files_one_escalate_info_in_the_fixed_slot(self, tmp_path, monkeypatch, store, queue, ports):
        report = _unavailable_report(tmp_path, monkeypatch)

        result = heal.route(report, ports, project_root=_project_root(tmp_path), now=NOW)

        (filed,) = queue.calls_to('escalate_info')
        assert filed['task_id'] == heal.UNAVAILABLE_SLOT
        assert filed['agent_role'] == heal.AGENT_ROLE
        assert filed['category'] == 'infra_issue'
        assert report['detail'] in filed['detail']
        assert '--reify-repo' in filed['suggested_action']
        assert filed['evidence'][0]['measured_at'] == NOW.isoformat()
        assert store.calls == []
        assert result.outcome is heal.Outcome.ESCALATED

    def test_unavailable_with_a_pending_escalation_in_the_slot_files_none(self, tmp_path, monkeypatch, queue, ports):
        queue.pending.append({'id': 'esc-earlier', 'task_id': heal.UNAVAILABLE_SLOT})

        result = heal.route(_unavailable_report(tmp_path, monkeypatch), ports, project_root=_project_root(tmp_path), now=NOW)

        assert queue.calls_to('escalate_info') == []
        assert result.outcome is heal.Outcome.ESCALATION_PENDING
        assert 'escalation already pending' in result.line()


class TestMain:
    def test_a_fused_memory_error_reply_exits_2_naming_the_tool(self, tmp_path, monkeypatch, store, servers, capsys):
        store.fail_submit = True
        monkeypatch.setenv('REIFY_ROOT', str(lock_charter_sibling.plant_divergent_reify(tmp_path / 'reify')))

        rc = heal.main(_main_argv(tmp_path, servers))

        assert rc == 2
        captured = capsys.readouterr()
        assert 'submit_task' in captured.err
        assert 'filed' not in captured.out
        assert 'adopted' not in captured.out

    def test_dry_run_calls_neither_server_and_prints_the_would_be_filing(self, tmp_path, servers, capsys):
        repo = lock_charter_sibling.plant_divergent_reify(tmp_path / 'reify')

        rc = heal.main(_main_argv(tmp_path, servers, '--dry-run', '--reify-repo', str(repo)))

        assert rc == 0
        for server in servers:
            assert server.requests == ()
        out = capsys.readouterr().out
        assert 'Mirror reify lock-charter drift into dark-factory: ' in out
        assert 'newtool' in out
        for site in heal.HEAL_SITES:
            assert site in out

    def test_every_heal_site_is_a_tracked_file_of_this_checkout(self):
        """The filed task declares HEAL_SITES as its lock charter, so each must be real."""
        proc = subprocess.run(
            ['git', '-C', str(REPO_ROOT), 'ls-files', '--error-unmatch', '--', *heal.HEAL_SITES],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr

    def test_done_when_two_runs_over_a_divergent_sibling_file_exactly_one_task(
        self, tmp_path, monkeypatch, store, queue, servers, capsys,
    ):
        """DONE-WHEN, scheduled half: a reify-only drift files ONE heal task, and the next night adopts it."""
        monkeypatch.setenv('REIFY_ROOT', str(lock_charter_sibling.plant_divergent_reify(tmp_path / 'reify')))

        first = heal.main(_main_argv(tmp_path, servers))
        first_out = capsys.readouterr().out
        second = heal.main(_main_argv(tmp_path, servers))
        second_out = capsys.readouterr().out

        assert (first, second) == (0, 0)
        assert len(store.calls_to('submit_task')) == 1
        assert len(store.calls_to('commit_planning')) == 1
        assert 'filed 1000' in first_out
        assert 'adopted 1000' in second_out
        assert queue.calls == []
