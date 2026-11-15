"""`orchestrator verify-cache audit-run`: the merge role's module test legs, uncached, failures module-qualified (task 5655).

Fixtures are kept MODULE-LOCAL (not conftest.py) for the reason stated in
test_config_verify_admission_reload.py. conftest's autouse ``_isolate_orch_config``
pins ``ORCH_PROJECT_ROOT`` to ``tmp_path``, so the fixture tree IS ``tmp_path``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from shared.proc_group import pid_running

from orchestrator.cli import main
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.offline_lane_confirm import ConfirmMode, confirm_mode, reproduce_line
from orchestrator.offline_lane_records import Leg, Phase, RunRecord, RunRecordStore
from orchestrator.offline_lane_runs import OutputFormat, launch
from orchestrator.verify import FailureKind, PytestFailure, parse_pytest_failures
from orchestrator.verify_cache import audit_run

_FAKE_SESSION = '''\
import json, os, pathlib, signal, subprocess, sys, time
module = pathlib.Path(sys.argv[1])
root = pathlib.Path(__file__).parent
with (root / 'order.txt').open('a') as order:
    order.write(str(module.relative_to(root)) + '\\n')
(module / 'env.json').write_text(json.dumps(dict(os.environ)))
if (module / 'grandchild').exists():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    (module / 'grandchild.pid').write_text(str(child.pid))
if (module / 'sleep').exists():
    time.sleep(float((module / 'sleep').read_text()))
if (module / 'signal').exists():
    os.kill(os.getpid(), signal.SIGKILL)
sys.stdout.write((module / 'output.txt').read_text())
sys.exit(int((module / 'rc').read_text()))
'''

_FAKE_PROVISION = '''\
import pathlib, sys
root = pathlib.Path(__file__).parent
with (root / 'order.txt').open('a') as order:
    order.write('provision\\n')
(root / 'provisioned').touch()
sys.exit(int((root / 'provision_rc').read_text()))
'''

_GREEN = '... [100%]\n== 3 passed in 1.0s ==\n'
_ALPHA_RED = (
    'F.. [100%]\n'
    '=========================== short test summary info ============================\n'
    'FAILED tests/test_a.py::test_one - AssertionError: one\n'
    "FAILED tests/test_a.py::test_two[x-y] - RuntimeError: worker crashed while running 'tests/t.py::x'\n"
    '== 2 failed, 1 passed in 1.0s ==\n'
)
_BETA_COLLECTION_ERROR = (
    '=========================== short test summary info ============================\n'
    'ERROR tests/test_b.py - ImportError: no module named nope\n'
    '!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n'
    '== 1 error in 1.0s ==\n'
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DF_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'

_ALPHA_ONE = 'alpha::tests/test_a.py::test_one'
_ALPHA_TWO = 'alpha::tests/test_a.py::test_two[x-y]'
_BETA_FILE = 'beta::tests/test_b.py'


class _Tree:
    """A project at ``tmp_path`` with fake-session modules ``alpha``, ``beta`` and ``tests/scripts``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.script = root / 'fake_session.py'
        self.script.write_text(_FAKE_SESSION)
        self.config_path = root / 'dark-factory-orchestrator.yaml'
        self.config_path.write_text(yaml.safe_dump({
            'project_root': str(root),
            'verify_env': {'AUDIT_SHARED': 'top', 'AUDIT_OVERRIDE': 'top'},
        }))
        for prefix in ('alpha', 'beta', 'tests/scripts'):
            self.module(prefix, output=_GREEN, rc=0)

    def module(self, prefix: str, *, output: str, rc: int, **settings: object) -> Path:
        directory = self.root / prefix
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'orchestrator.yaml').write_text(yaml.safe_dump({
            'test_command': f'{sys.executable} {self.script} {directory}',
            **settings,
        }))
        (directory / 'output.txt').write_text(output)
        (directory / 'rc').write_text(str(rc))
        return directory

    def config(self) -> OrchestratorConfig:
        return load_config(self.config_path)

    def provisioning_config(self, rc: int) -> OrchestratorConfig:
        """The tree's config with a fake cold-preprovision command that exits *rc*.

        conftest's autouse ``_isolate_orch_config`` empties the knob through the env, which wins
        over the yaml, so the command is set on the loaded model instead.
        """
        script = self.root / 'fake_provision.py'
        script.write_text(_FAKE_PROVISION)
        (self.root / 'provision_rc').write_text(str(rc))
        return self.config().model_copy(update={'verify_cold_preprovision_command': f'{sys.executable} {script}'})

    def env_of(self, prefix: str) -> dict[str, str]:
        return json.loads((self.root / prefix / 'env.json').read_text())

    def order(self) -> list[str]:
        return (self.root / 'order.txt').read_text().splitlines()


@pytest.fixture
def tree(tmp_path: Path) -> _Tree:
    return _Tree(tmp_path)


async def _audit(tree: _Tree, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    rc = await audit_run.run_audit(tree.root, tree.config(), tree.root / 'logs')
    return rc, capsys.readouterr().out


async def _lane_record(tmp_path: Path, report: str) -> RunRecord:
    report_path = tmp_path / 'report.txt'
    report_path.write_text(report)
    store = RunRecordStore(tmp_path / 'records', clock=lambda: datetime(2026, 11, 14, tzinfo=UTC))
    lane_pass = store.allocate_pass(audit_run.AUDIT_COMMAND, Leg.GENERIC, 'c0ffee')
    launched = await launch(
        ['sh', '-c', f'cat {report_path}; exit 1'], cwd=tmp_path, env={'PATH': '/usr/bin:/bin'},
        store=store, lane_pass=lane_pass, phase=Phase.DETECT, output_format=OutputFormat.PYTEST,
    )
    return launched.record


def _run(module: str, rc: int, ended: audit_run.ModuleEnd, *failures: PytestFailure) -> audit_run.ModuleRun:
    return audit_run.ModuleRun(module, rc, ended, failures, Path(f'/logs/{module}.log'))


_FAILED_ONE = PytestFailure('tests/test_a.py::test_one', FailureKind.FAILED, 'AssertionError: one')


def _has_summary(lines: list[str]) -> bool:
    return any(line.startswith('== ') and line.endswith(' ==') for line in lines)


class TestAuditNode:
    def test_a_slashed_module_renders_and_parses_back(self):
        node = audit_run.AuditNode('tests/scripts', 'tests/scripts/test_x.py::test_y[a-b]')

        assert str(node) == 'tests/scripts::tests/scripts/test_x.py::test_y[a-b]'
        assert audit_run.AuditNode.parse(str(node)) == node

    def test_a_file_only_node_round_trips(self):
        assert audit_run.AuditNode.parse(_BETA_FILE) == audit_run.AuditNode('beta', 'tests/test_b.py')

    @pytest.mark.parametrize('unqualified', ['tests/test_x.py::t', '::tests/test_x.py::t', 'tests/test_x.py'])
    def test_an_unqualified_id_is_not_an_audit_node(self, unqualified):
        assert audit_run.AuditNode.parse(unqualified) is None


@pytest.mark.asyncio
class TestRunAudit:
    async def test_every_module_runs_in_prefix_order_after_an_earlier_red(self, tree, capsys):
        tree.module('alpha', output=_ALPHA_RED, rc=1)

        rc, report = await _audit(tree, capsys)

        assert rc == 1
        assert tree.order() == ['alpha', 'beta', 'tests/scripts']
        assert f'FAILED {_ALPHA_ONE} - AssertionError: one' in report.splitlines(), report

    async def test_the_report_round_trips_through_the_lanes_parser(self, tree, capsys):
        tree.module('alpha', output=_ALPHA_RED, rc=1)
        tree.module('beta', output=_BETA_COLLECTION_ERROR, rc=2)

        _, report = await _audit(tree, capsys)

        parsed = {(failure.node_id, failure.kind) for failure in parse_pytest_failures(report)}
        assert parsed == {
            (str(audit_run.AuditNode(module, failure.node_id)), failure.kind)
            for module, output in (('alpha', _ALPHA_RED), ('beta', _BETA_COLLECTION_ERROR))
            for failure in parse_pytest_failures(output)
        }, report
        assert {
            (_ALPHA_ONE, FailureKind.FAILED), (_ALPHA_TWO, FailureKind.FAILED), (_BETA_FILE, FailureKind.ERROR),
        } <= parsed, report
        assert all(audit_run.AuditNode.parse(node) for node, _ in parsed), report

    async def test_the_lanes_launch_records_the_qualified_nodes(self, tree, capsys, tmp_path):
        tree.module('alpha', output=_ALPHA_RED, rc=1)
        _, report = await _audit(tree, capsys)

        record = await _lane_record(tmp_path, report)

        assert {_ALPHA_ONE, _ALPHA_TWO} <= {failure.node for failure in record.failures}
        assert all(audit_run.AuditNode.parse(failure.node) for failure in record.failures)
        assert record.session_complete is True

    async def test_a_timed_out_module_is_killed_with_its_group_and_the_next_still_runs(self, tree, capsys):
        alpha = tree.module('alpha', output=_GREEN, rc=0, verify_command_timeout_secs=5)
        (alpha / 'grandchild').touch()
        (alpha / 'sleep').write_text('60')

        rc, report = await _audit(tree, capsys)

        assert rc == 1
        assert tree.order() == ['alpha', 'beta', 'tests/scripts']
        assert '# alpha: rc=' in report and 'ended=timeout' in report, report
        assert not _has_summary(report.splitlines()), report
        grandchild = int((alpha / 'grandchild.pid').read_text())
        deadline = time.monotonic() + 10
        while pid_running(grandchild) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not pid_running(grandchild)

    async def test_a_module_killed_by_a_signal_leaves_the_session_incomplete(self, tree, capsys, tmp_path):
        tree.module('alpha', output=_ALPHA_RED, rc=1)
        (tree.module('beta', output=_GREEN, rc=0) / 'signal').touch()

        rc, report = await _audit(tree, capsys)

        assert rc == 1
        assert 'ended=signal' in report, report
        assert not _has_summary(report.splitlines()), report
        assert (await _lane_record(tmp_path, report)).session_complete is False

    async def test_the_module_child_env_is_the_merge_gates_plus_cache_off(self, tree, capsys, monkeypatch):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '/nonexistent/fake.yaml')
        monkeypatch.setenv('PROJECT_ROOT', str(tree.root))
        monkeypatch.setenv('ORCH_AUDIT_PROBE', 'leak')
        tree.module('beta', output=_GREEN, rc=0, verify_env={'AUDIT_OVERRIDE': 'module'})

        await audit_run.run_audit(tree.root, tree.config(), tree.root / 'logs')

        env = tree.env_of('beta')
        assert env['DF_VERIFY_CACHE'] == 'off'
        assert env['DF_VERIFY_ROLE'] == 'merge'
        assert (env['AUDIT_SHARED'], env['AUDIT_OVERRIDE']) == ('top', 'module')
        assert tree.env_of('alpha')['AUDIT_OVERRIDE'] == 'top'
        assert not [name for name in env if name.startswith('ORCH_')]
        assert 'PROJECT_ROOT' not in env

    async def test_a_verify_env_project_root_still_reaches_the_child(self, tree, capsys, monkeypatch):
        monkeypatch.setenv('PROJECT_ROOT', str(tree.root))
        tree.module('beta', output=_GREEN, rc=0, verify_env={'PROJECT_ROOT': '/from/verify_env'})

        await audit_run.run_audit(tree.root, tree.config(), tree.root / 'logs')

        assert tree.env_of('beta')['PROJECT_ROOT'] == '/from/verify_env'
        assert 'PROJECT_ROOT' not in tree.env_of('alpha')

    async def test_the_tree_is_provisioned_before_the_first_module_runs(self, tree, capsys):
        rc = await audit_run.run_audit(tree.root, tree.provisioning_config(0), tree.root / 'logs')
        report = capsys.readouterr().out

        assert rc == 0
        assert tree.order() == ['provision', 'alpha', 'beta', 'tests/scripts']
        assert (tree.root / 'provisioned').exists()
        assert _has_summary(report.splitlines()), report

    async def test_a_failed_provision_runs_no_module_and_reports_no_test(self, tree, capsys, tmp_path):
        rc = await audit_run.run_audit(tree.root, tree.provisioning_config(1), tree.root / 'logs')
        report = capsys.readouterr().out

        assert rc != 0
        assert tree.order() == ['provision']
        lines = report.splitlines()
        assert parse_pytest_failures(report) == (), report
        assert not [line for line in lines if line.startswith(('FAILED', 'ERROR'))], report
        assert not _has_summary(lines), report
        assert any(line.startswith('#') and 'provision' in line and 'failed' in line for line in lines), report
        record = await _lane_record(tmp_path, report)
        assert record.session_complete is False
        assert record.failures == ()

    async def test_each_modules_raw_output_lands_in_its_named_log(self, tree, capsys):
        tree.module('alpha', output=_ALPHA_RED, rc=1)

        _, report = await _audit(tree, capsys)

        logs = tree.root / 'logs'
        alpha_line = next(line for line in report.splitlines() if line.startswith('# alpha:'))
        alpha_log = Path(alpha_line.partition('log=')[2])
        assert alpha_log.parent == logs
        assert alpha_log.read_text() == _ALPHA_RED
        scripts_line = next(line for line in report.splitlines() if line.startswith('# tests/scripts:'))
        assert Path(scripts_line.partition('log=')[2]).parent == logs

    async def test_all_green_exits_zero_with_a_summary_and_no_failures(self, tree, capsys):
        rc, report = await _audit(tree, capsys)

        assert rc == 0
        assert parse_pytest_failures(report) == ()
        assert _has_summary(report.splitlines()), report


class TestRenderReport:
    def test_red_modules_that_named_their_failures_completed(self):
        lines = audit_run.render_report([
            _run('alpha', 1, audit_run.ModuleEnd.EXIT, _FAILED_ONE), _run('beta', 0, audit_run.ModuleEnd.EXIT),
        ], 2.0)

        assert lines[-1] == '== 1 failed, 0 errors in 2.0s ==', lines

    @pytest.mark.parametrize('run', [
        _run('alpha', 1, audit_run.ModuleEnd.EXIT),
        _run('alpha', 2, audit_run.ModuleEnd.EXIT),
        _run('alpha', 2, audit_run.ModuleEnd.EXIT, _FAILED_ONE),
        _run('alpha', -9, audit_run.ModuleEnd.TIMEOUT),
        _run('alpha', 137, audit_run.ModuleEnd.SIGNAL, _FAILED_ONE),
    ])
    def test_no_summary_unless_every_module_completed(self, run):
        lines = audit_run.render_report([run, _run('beta', 0, audit_run.ModuleEnd.EXIT)], 2.0)

        assert not _has_summary(lines), lines

    def test_a_reason_carries_no_node_id_a_parser_could_lift(self):
        crash = PytestFailure('tests/test_a.py::t', FailureKind.FAILED, "crashed while running 'tests/t.py::x'")

        lines = audit_run.render_report([_run('alpha', 1, audit_run.ModuleEnd.EXIT, crash)], 1.0)

        assert [f.node_id for f in parse_pytest_failures('\n'.join(lines))] == ['alpha::tests/test_a.py::t']


class TestAuditRunCli:
    def _invoke(self, tree: _Tree):
        return CliRunner().invoke(main, ['verify-cache', 'audit-run', '--config', str(tree.config_path)])

    def test_a_red_module_prints_qualified_failures_and_exits_one(self, tree):
        tree.module('alpha', output=_ALPHA_RED, rc=1)

        result = self._invoke(tree)

        assert result.exit_code == 1, result.output
        assert f'FAILED {_ALPHA_ONE} - AssertionError: one' in result.output.splitlines(), result.output

    def test_all_green_exits_zero_and_logs_under_the_default_dir(self, tree):
        result = self._invoke(tree)

        assert result.exit_code == 0, result.output
        assert 'FAILED' not in result.output, result.output
        logs = tree.root / audit_run.DEFAULT_LOG_DIR
        assert sorted(path.name for path in logs.iterdir()) == ['alpha.log', 'beta.log', 'tests--scripts.log']


def _committed_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """The COMMITTED yaml through the real model, as test_warm_lane_bash_bucket_placement.py loads it."""
    assert DF_CONFIG_PATH.exists(), DF_CONFIG_PATH
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(DF_CONFIG_PATH))
    return OrchestratorConfig()


def _detect_naming_ids() -> RunRecord:
    at = datetime(2026, 11, 14, tzinfo=UTC)
    return RunRecord(
        command=audit_run.AUDIT_COMMAND, leg=Leg.GENERIC.value, head='c0ffee', phase=Phase.DETECT.value,
        started_at=at, ended_at=at, returncode=1, ended_by='exit', output_path='/dev/null', junit_path=None,
        failures=(), ids_available=True, outcomes=None, session_complete=True, probe=None,
    )


class TestCommittedLaneEntry:
    def test_the_lane_runs_the_audit_uncached_against_the_heads_own_config(self, monkeypatch):
        entries = [
            entry for entry in _committed_config(monkeypatch).git.offline_lane_commands
            if entry.enabled and entry.name == audit_run.AUDIT_COMMAND
        ]

        assert len(entries) == 1, entries
        entry = entries[0]
        assert entry.command.startswith('DF_VERIFY_CACHE=off '), entry.command
        assert 'PROJECT_ROOT="$PWD"' in entry.command, entry.command
        assert (
            'uv run --project orchestrator orchestrator verify-cache audit-run --config dark-factory-orchestrator.yaml'
            in entry.command
        ), entry.command
        assert (entry.cwd, entry.fix_task_priority, entry.outage_probe) == ('.', 'high', None)

    def test_the_lane_treats_the_audit_as_opaque_and_intersects(self, monkeypatch):
        entry = next(
            entry for entry in _committed_config(monkeypatch).git.offline_lane_commands
            if entry.name == audit_run.AUDIT_COMMAND
        )

        assert reproduce_line(entry, ['m::t.py::x']) is None
        assert confirm_mode(entry, _detect_naming_ids()) is ConfirmMode.INTERSECTED
