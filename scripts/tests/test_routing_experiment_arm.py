"""scripts/routing_experiment_arm.sh, driven through its real argv, env, exit
code and stderr.

Each test copies the script into a temp git repo whose `main` commits the
fixture files, beside a stub readout that records its argv and exits 4 (the
readout's "unproven" code). Every run therefore stops after step 2, so step
3's reload never runs, and every run names fixture units on port 1, so even a
stub mistake cannot reach the live fleet's escalation ports.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from cli_subprocess_timeout import cli_timeout_from_env
from git_env import scrubbed_git_env

SCRIPT = Path(__file__).parent.parent / 'routing_experiment_arm.sh'
BASH = shutil.which('bash') or '/bin/bash'
RULES_FILE = 'orchestrator/src/orchestrator/defaults.yaml'
TIMEOUT = cli_timeout_from_env('ROUTING_EXPERIMENT_ARM_TEST_TIMEOUT')
RULE_A_RULES = 'routing:\n  rules:\n    - id: rule-a\n'

_STUB_READOUT = '''\
import json, os, sys
with open(os.environ["STUB_CALLS"], "a") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
sys.exit(4)
'''


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ['git', *args], cwd=repo, env=scrubbed_git_env(os.environ),
        check=True, capture_output=True, timeout=TIMEOUT,
    )


def _arm_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _git(repo, 'add', '-A')
    _git(
        repo, '-c', 'user.name=t', '-c', 'user.email=t@example.invalid',
        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture',
    )
    scripts = repo / 'scripts'
    scripts.mkdir(exist_ok=True)
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)
    (scripts / 'routing_experiment_readout.py').write_text(_STUB_READOUT)
    return repo


def _run_arm(
    repo: Path, *args: str, units: str = '/srv/untreated-a:1',
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    calls = repo.parent / 'stub_calls.jsonl'
    env = scrubbed_git_env(os.environ)
    env['STUB_CALLS'] = str(calls)
    env['ROUTING_EXPERIMENT_UNITS'] = units
    result = subprocess.run(
        [BASH, str(repo / 'scripts' / SCRIPT.name), *args],
        env=env, capture_output=True, text=True, timeout=TIMEOUT,
    )
    if not calls.exists():
        return result, []
    return result, [json.loads(line) for line in calls.read_text().splitlines()]


def test_present_rule_id_passes_on_a_rules_file_far_larger_than_a_pipe_buffer(tmp_path):
    rules = (
        'routing:\n  rules:\n    - id: rule-a\n'
        + '# padding that outlasts a pipe buffer\n' * 60000
    )
    repo = _arm_repo(tmp_path, {RULES_FILE: rules})

    result, calls = _run_arm(repo, 'exp-salt', 'rule-a')

    assert len(calls) == 1, result.stderr
    assert "is not in main's" not in result.stderr


def test_absent_rule_id_fails_naming_it_before_the_readout(tmp_path):
    rules = 'routing:\n  rules:\n    - id: rule-a\n    - id: rule-b-4\n'
    repo = _arm_repo(tmp_path, {RULES_FILE: rules})

    result, calls = _run_arm(repo, 'exp-salt', 'rule-a,rule-b')

    assert result.returncode == 1
    assert 'rule-b' in result.stderr
    assert RULES_FILE in result.stderr
    assert calls == []


def test_rules_file_missing_from_main_fails_with_a_read_error_not_a_missing_id(tmp_path):
    repo = _arm_repo(tmp_path, {'README.md': 'fixture\n'})

    result, calls = _run_arm(repo, 'exp-salt', 'rule-a')

    assert result.returncode == 1
    assert 'cannot read' in result.stderr
    assert RULES_FILE in result.stderr
    assert "is not in main's" not in result.stderr
    assert calls == []


@pytest.mark.parametrize('separator', [' ', '\n'])
def test_aa_reads_exactly_the_roots_of_the_experiment_units(tmp_path, separator):
    repo = _arm_repo(tmp_path, {RULES_FILE: RULE_A_RULES})
    units = separator.join(['/srv/exp-a:1', '/srv/exp-b:2'])

    result, calls = _run_arm(repo, 'exp-salt', 'rule-a', '14', units=units)

    aa_calls = [argv for argv in calls if '--aa' in argv]
    assert len(aa_calls) == 1, result.stderr
    argv = aa_calls[0]
    assert argv[argv.index('--roots') + 1] == '/srv/exp-a,/srv/exp-b'
    assert argv[argv.index('--salt') + 1] == 'exp-salt'
    assert argv[argv.index('--days') + 1] == '14'
    assert result.returncode == 1
    assert "the readout's A/A check did not pass" in result.stderr


@pytest.mark.parametrize('bad', ['/srv/exp-b', '/srv/exp-b:', ':8100', '/srv/exp-b:80x'])
def test_malformed_unit_fails_naming_it_before_the_readout(tmp_path, bad):
    repo = _arm_repo(tmp_path, {RULES_FILE: RULE_A_RULES})

    result, calls = _run_arm(repo, 'exp-salt', 'rule-a', units='/srv/exp-a:1 ' + bad)

    assert result.returncode == 1
    assert bad in result.stderr
    assert '<project_root>:<port>' in result.stderr
    assert calls == []


def test_units_naming_no_unit_fails_before_the_readout(tmp_path):
    repo = _arm_repo(tmp_path, {RULES_FILE: RULE_A_RULES})

    result, calls = _run_arm(repo, 'exp-salt', 'rule-a', units='  \n ')

    assert result.returncode == 1
    assert 'ROUTING_EXPERIMENT_UNITS' in result.stderr
    assert calls == []
