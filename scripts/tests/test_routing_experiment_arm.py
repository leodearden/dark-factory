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

from cli_subprocess_timeout import cli_timeout_from_env
from git_env import scrubbed_git_env

SCRIPT = Path(__file__).parent.parent / 'routing_experiment_arm.sh'
BASH = shutil.which('bash') or '/bin/bash'
RULES_FILE = 'orchestrator/src/orchestrator/defaults.yaml'
TIMEOUT = cli_timeout_from_env('ROUTING_EXPERIMENT_ARM_TEST_TIMEOUT')

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
