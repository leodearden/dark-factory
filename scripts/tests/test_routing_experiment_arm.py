"""scripts/routing_experiment_arm.sh, driven through its real argv, env, exit
code, stderr and flip record.

Each test copies the script into a temp git repo whose `main` commits the
fixture files, beside a recorded readout (conftest `fake_recorder`). Most runs
have the readout exit 4, its "unproven" code, so the script stops after step 3.
The tests that drive steps 4 and 5 have it exit 0 and also write a stub
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
from datetime import datetime
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
RULES_FILE_ENV = 'ROUTING_EXPERIMENT_RULES_FILE'
PROJECT_RULES = 'projects/effort.yaml'
EFFORT_RULE_ID = 'effort-impl-high-implementer'
EFFORT_RULES = f'routing:\n  rules:\n    - id: {EFFORT_RULE_ID}\n'

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
    repo: Path, *args: str, units: str = '/srv/untreated-a:1', rules_file: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the copied script; *rules_file* None removes any ambient rules-file override."""
    env = scrubbed_git_env(os.environ)
    env['ROUTING_EXPERIMENT_UNITS'] = units
    if rules_file is None:
        env.pop(RULES_FILE_ENV, None)
    else:
        env[RULES_FILE_ENV] = rules_file
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


def _commit_on_side_branch(repo: Path, path: str, text: str) -> None:
    """Commit *path* on a new branch `side`, left checked out: HEAD and the tree carry it, main does not."""
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    env = scrubbed_git_env(os.environ)
    for args in (
        ('checkout', '-q', '-b', 'side'),
        ('add', path),
        ('-c', 'user.email=t@e', '-c', 'user.name=t', '-c', 'commit.gpgsign=false',
         'commit', '-q', '-m', 'side'),
    ):
        subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True,
                       env=env, timeout=TIMEOUT)


def test_a_named_rules_file_on_main_passes_step_1_without_defaults_yaml(arm_repo):
    repo, readout = arm_repo({PROJECT_RULES: EFFORT_RULES})

    result = _run_arm(repo, 'exp-salt', EFFORT_RULE_ID, rules_file=PROJECT_RULES)

    calls = readout.calls()
    assert len(calls) == 1, result.stderr
    argv = calls[0]
    assert '--aa' in argv
    assert argv[argv.index('--salt') + 1] == 'exp-salt'
    assert result.returncode == 1
    assert "the readout's A/A check did not pass" in result.stderr


def test_without_the_override_the_rules_file_is_still_defaults_yaml(arm_repo):
    repo, readout = arm_repo({PROJECT_RULES: EFFORT_RULES})

    result = _run_arm(repo, 'exp-salt', EFFORT_RULE_ID)

    assert result.returncode == 1
    assert 'cannot read' in result.stderr
    assert RULES_FILE in result.stderr
    assert readout.calls() == []


def test_a_named_rules_file_lacking_the_id_fails_naming_file_and_id(arm_repo):
    repo, readout = arm_repo({PROJECT_RULES: RULE_A_RULES})

    result = _run_arm(repo, 'exp-salt', EFFORT_RULE_ID, rules_file=PROJECT_RULES)

    assert result.returncode == 1
    assert PROJECT_RULES in result.stderr
    assert EFFORT_RULE_ID in result.stderr
    assert "is not in main's" in result.stderr
    assert readout.calls() == []


def test_a_named_rules_file_absent_from_main_is_a_read_error_naming_the_override(arm_repo):
    missing = 'projects/missing.yaml'
    repo, readout = arm_repo({PROJECT_RULES: EFFORT_RULES})

    result = _run_arm(repo, 'exp-salt', EFFORT_RULE_ID, rules_file=missing)

    assert result.returncode == 1
    assert 'cannot read' in result.stderr
    assert missing in result.stderr
    assert RULES_FILE_ENV in result.stderr
    assert "is not in main's" not in result.stderr
    assert readout.calls() == []


def test_a_rules_file_only_on_another_branch_is_not_read(arm_repo):
    repo, readout = arm_repo({'README.md': 'fixture\n'})
    _commit_on_side_branch(repo, PROJECT_RULES, EFFORT_RULES)

    result = _run_arm(repo, 'exp-salt', EFFORT_RULE_ID, rules_file=PROJECT_RULES)

    assert result.returncode == 1
    assert 'cannot read' in result.stderr
    assert PROJECT_RULES in result.stderr
    assert readout.calls() == []


D0 = '2026-11-01T12:00:00+00:00'
D1 = '2026-11-02T12:00:00+00:00'
D2 = '2026-11-03T12:00:00+00:00'
D3 = '2026-11-04T12:00:00+00:00'
RULE_A_B_RULES = 'routing:\n  rules:\n    - id: rule-a\n    - id: rule-b\n'
_IDENTITY = ('-c', 'user.email=t@e', '-c', 'user.name=t', '-c', 'commit.gpgsign=false')


def _seeded_at(arm_repo, monkeypatch, at: str, files: dict[str, str], **kwargs):
    """arm_repo(*files*, ...) with its seed commit authored and committed at *at*."""
    with monkeypatch.context() as dated:
        dated.setenv('GIT_AUTHOR_DATE', at)
        dated.setenv('GIT_COMMITTER_DATE', at)
        return arm_repo(files, **kwargs)


def _commit_dated(
    repo: Path, path: str, text: str, *, at: str,
    branch: str | None = None, merged_at: str | None = None,
) -> str:
    """Commit *text* to *path* at *at*: directly on main, or on a new *branch*
    from main that lands on main with a --no-ff merge at *merged_at*.

    Only *path* is ever added, so the copied scripts/ dir stays untracked
    across the checkouts. Returns `git rev-parse main`.
    """
    def git(*args: str, when: str) -> str:
        env = {**scrubbed_git_env(os.environ), 'GIT_AUTHOR_DATE': when, 'GIT_COMMITTER_DATE': when}
        return subprocess.run(
            ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
            env=env, timeout=TIMEOUT,
        ).stdout.strip()

    if branch is not None:
        git('checkout', '-q', '-b', branch, 'main', when=at)
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8')
    git('add', path, when=at)
    git(*_IDENTITY, 'commit', '-q', '-m', f'edit {path}', when=at)
    if branch is not None:
        assert merged_at is not None, 'a branch commit lands only through a merge'
        git('checkout', '-q', 'main', when=merged_at)
        git(*_IDENTITY, 'merge', '-q', '--no-ff', '-m', 'land', branch, when=merged_at)
    return git('rev-parse', 'main', when=at)


def _aa_until(readout) -> datetime:
    (argv,) = [argv for argv in readout.calls() if '--aa' in argv]
    return datetime.fromisoformat(argv[argv.index('--until') + 1])


def test_aa_window_ends_where_main_first_carried_every_rule_id(arm_repo, monkeypatch):
    repo, readout = _seeded_at(arm_repo, monkeypatch, D0, {RULES_FILE: RULE_A_RULES})
    _commit_dated(repo, RULES_FILE, RULE_A_B_RULES, at=D1, branch='gamma', merged_at=D2)
    _commit_dated(repo, RULES_FILE, RULE_A_B_RULES + '# a later edit\n', at=D3)

    result = _run_arm(repo, 'exp-salt', 'rule-a,rule-b')

    assert result.returncode == 1
    assert "the readout's A/A check did not pass" in result.stderr
    assert _aa_until(readout) == datetime.fromisoformat(D2)


def test_rule_ids_removed_then_restored_anchor_on_the_restoring_commit(arm_repo, monkeypatch):
    repo, readout = _seeded_at(arm_repo, monkeypatch, D0, {RULES_FILE: RULE_A_B_RULES})
    _commit_dated(repo, RULES_FILE, RULE_A_RULES, at=D1)
    _commit_dated(repo, RULES_FILE, RULE_A_B_RULES, at=D2)

    result = _run_arm(repo, 'exp-salt', 'rule-a,rule-b')

    assert result.returncode == 1
    assert _aa_until(readout) == datetime.fromisoformat(D2)


def test_flip_record_carries_treatment_from_and_the_landing_commit_beside_armed_at(
    arm_repo, monkeypatch,
):
    repo, _readout = _seeded_at(
        arm_repo, monkeypatch, D0, {RULES_FILE: RULE_A_RULES}, readout_exit=0,
    )
    _install_reload_gate_stub(repo)
    landing = _commit_dated(repo, RULES_FILE, RULE_A_B_RULES, at=D1, branch='gamma', merged_at=D2)
    _commit_dated(repo, RULES_FILE, RULE_A_B_RULES + '# a later edit\n', at=D3)

    result = _run_arm(repo, 'exp-salt', 'rule-a,rule-b')

    assert result.returncode == 0, result.stderr
    (line,) = (repo / FLIP_LEDGER).read_text().splitlines()
    flip = json.loads(line)
    treatment_from = datetime.fromisoformat(flip['treatment_from'])
    assert treatment_from == datetime.fromisoformat(D2)
    assert flip['rules_landed_in'] == landing
    assert datetime.fromisoformat(flip['armed_at']) > treatment_from
