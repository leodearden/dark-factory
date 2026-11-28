"""scripts/routing_experiment_arm.sh, driven through its real argv, env, exit
code, stderr and flip record.

Each test copies the script into a temp git repo whose `main` commits the
fixture files, beside a recorded readout (conftest `fake_recorder`). Most runs
have the readout exit 4, its "unproven" code, so the script stops after step 2.
The tests that drive steps 3 and 4 have it exit 0 and also write a stub
`_config_reload_gate` beside the script, which records each port it is asked to
reload and opens no socket. Every run names fixture units on ports 1 and 2,
which no fleet unit listens on, so even a stub mistake cannot reach the live
fleet's escalation ports.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from committed_git_repo import committed_repo
from git_env import scrubbed_git_env
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = Path(__file__).parent.parent / 'routing_experiment_arm.sh'
BASH = shutil.which('bash') or '/bin/bash'
RULES_FILE = 'orchestrator/src/orchestrator/defaults.yaml'
FLIP_LEDGER = 'data/orchestrator/routing_experiments.jsonl'
TIMEOUT = cli_timeout_from_env('ROUTING_EXPERIMENT_ARM_TEST_TIMEOUT')
RULE_A_RULES = 'routing:\n  rules:\n    - id: rule-a\n'
READOUT_UNPROVEN = 4

_STUB_RELOAD_GATE = """\
import enum
import json

CALLS_PATH = {calls_path!r}


class ReloadFailure(str, enum.Enum):
    NO_RELOAD_REPORT = "no_reload_report"
    REPLY_TIMED_OUT = "reload_reply_timed_out"


class ReloadNotConfirmed(Exception):
    pass


def fetch_reload_report(port, *, committed_as, config_path):
    with open(CALLS_PATH, "a") as fh:
        fh.write(json.dumps(port) + "\\n")
    return dict(applied=[])
"""


@pytest.fixture
def arm_repo(tmp_path, fake_recorder):
    """Factory: a temp repo whose `main` commits *files*, with the script copied
    into its scripts/ beside a recorded readout that exits *readout_exit*.

    Returns the repo and the readout's FakeRecorder.
    """
    def _build(files: dict[str, str], *, readout_exit: int = READOUT_UNPROVEN):
        repo = tmp_path / 'repo'
        committed_repo(repo, files)
        scripts = repo / 'scripts'
        scripts.mkdir(exist_ok=True)
        shutil.copy2(SCRIPT, scripts / SCRIPT.name)
        readout = fake_recorder(
            'routing_experiment_readout.py', bin_dir=scripts, exit_code=readout_exit,
        )
        return repo, readout

    return _build


def _install_reload_gate_stub(repo: Path) -> Path:
    """Write the stub `_config_reload_gate` beside the copied script; return
    the file it appends each requested port to."""
    calls = repo.parent / 'reload_calls.jsonl'
    (repo / 'scripts' / '_config_reload_gate.py').write_text(
        _STUB_RELOAD_GATE.format(calls_path=str(calls)),
    )
    return calls


def _run_arm(
    repo: Path, *args: str, units: str = '/srv/untreated-a:1',
) -> subprocess.CompletedProcess[str]:
    env = scrubbed_git_env(os.environ)
    env['ROUTING_EXPERIMENT_UNITS'] = units
    return subprocess.run(
        [BASH, str(repo / 'scripts' / SCRIPT.name), *args],
        env=env, capture_output=True, text=True, timeout=TIMEOUT,
    )


def test_present_rule_id_passes_on_a_rules_file_far_larger_than_a_pipe_buffer(arm_repo):
    rules = (
        'routing:\n  rules:\n    - id: rule-a\n'
        + '# padding that outlasts a pipe buffer\n' * 60000
    )
    repo, readout = arm_repo({RULES_FILE: rules})

    result = _run_arm(repo, 'exp-salt', 'rule-a')

    assert len(readout.calls()) == 1, result.stderr
    assert "is not in main's" not in result.stderr


def test_absent_rule_id_fails_naming_it_before_the_readout(arm_repo):
    rules = 'routing:\n  rules:\n    - id: rule-a\n    - id: rule-b-4\n'
    repo, readout = arm_repo({RULES_FILE: rules})

    result = _run_arm(repo, 'exp-salt', 'rule-a,rule-b')

    assert result.returncode == 1
    assert 'rule-b' in result.stderr
    assert RULES_FILE in result.stderr
    assert readout.calls() == []


def test_rules_file_missing_from_main_fails_with_a_read_error_not_a_missing_id(arm_repo):
    repo, readout = arm_repo({'README.md': 'fixture\n'})

    result = _run_arm(repo, 'exp-salt', 'rule-a')

    assert result.returncode == 1
    assert 'cannot read' in result.stderr
    assert RULES_FILE in result.stderr
    assert "is not in main's" not in result.stderr
    assert readout.calls() == []


@pytest.mark.parametrize('separator', [' ', '\n'])
def test_aa_reads_exactly_the_roots_of_the_experiment_units(arm_repo, separator):
    repo, readout = arm_repo({RULES_FILE: RULE_A_RULES})
    units = separator.join(['/srv/exp-a:1', '/srv/exp-b:2'])

    result = _run_arm(repo, 'exp-salt', 'rule-a', '14', units=units)

    aa_calls = [argv for argv in readout.calls() if '--aa' in argv]
    assert len(aa_calls) == 1, result.stderr
    argv = aa_calls[0]
    assert argv[argv.index('--roots') + 1] == '/srv/exp-a,/srv/exp-b'
    assert argv[argv.index('--salt') + 1] == 'exp-salt'
    assert argv[argv.index('--days') + 1] == '14'
    assert result.returncode == 1
    assert "the readout's A/A check did not pass" in result.stderr


@pytest.mark.parametrize(
    'bad', ['/srv/exp-b', '/srv/exp-b:', ':1', '/srv/exp-b:80x', 'relative:1', '/a,b:1'],
)
def test_malformed_unit_fails_naming_it_before_the_readout(arm_repo, bad):
    repo, readout = arm_repo({RULES_FILE: RULE_A_RULES})

    result = _run_arm(repo, 'exp-salt', 'rule-a', units='/srv/exp-a:1 ' + bad)

    assert result.returncode == 1
    assert bad in result.stderr
    assert '<project_root>:<port>' in result.stderr
    assert readout.calls() == []


def test_units_naming_no_unit_fails_before_the_readout(arm_repo):
    repo, readout = arm_repo({RULES_FILE: RULE_A_RULES})

    result = _run_arm(repo, 'exp-salt', 'rule-a', units='  \n ')

    assert result.returncode == 1
    assert 'ROUTING_EXPERIMENT_UNITS' in result.stderr
    assert readout.calls() == []


@pytest.mark.parametrize('separator', [' ', '\n'])
def test_reload_and_flip_record_cover_exactly_the_validated_units(arm_repo, separator):
    rules = 'routing:\n  rules:\n    - id: rule-a\n    - id: rule-b\n'
    repo, _readout = arm_repo({RULES_FILE: rules}, readout_exit=0)
    reload_calls = _install_reload_gate_stub(repo)
    units = separator.join(['/srv/exp-a:1', '/srv/exp-b:2'])

    result = _run_arm(repo, 'exp-salt', 'rule-a,rule-b', units=units)

    assert result.returncode == 0, result.stderr
    asked = [json.loads(line) for line in reload_calls.read_text().splitlines()]
    assert asked == ['1', '2']
    flips = (repo / FLIP_LEDGER).read_text().splitlines()
    assert len(flips) == 1
    flip = json.loads(flips[0])
    assert flip['salt'] == 'exp-salt'
    assert flip['rule_ids'] == ['rule-a', 'rule-b']
    assert [(u['root'], u['port'], u['status']) for u in flip['units']] == [
        ('/srv/exp-a', '1', 'reloaded'),
        ('/srv/exp-b', '2', 'reloaded'),
    ]
