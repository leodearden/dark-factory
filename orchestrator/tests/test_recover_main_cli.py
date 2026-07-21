"""Tests for orchestrator.recover_main CLI — step-7 RED.

Mirrors test_b3_gate.py's CLI-invocation + JSON-parse pattern:
monkeypatch recover_red_main to return a known result, capture stdout,
json.loads it, assert required keys.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Minimal YAML that load_config accepts and carries a main_gate_mark_command.
_MINIMAL_CONFIG_YAML = """\
git:
  main_branch: main
  main_gate_mark_command: "echo test-sentinel"
"""

_TARGET_SHA = 'aabbccddeeff00112233445566778899aabbccdd'
_EXPECTED_MAIN = 'ff00112233445566778899aabbccddeeffaabbcc'


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    """Write a minimal orchestrator YAML and return its path."""
    cfg = tmp_path / 'orchestrator.yaml'
    cfg.write_text(_MINIMAL_CONFIG_YAML)
    return cfg


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    """Return a temp directory representing the watched project root."""
    root = tmp_path / 'watched'
    root.mkdir()
    return root


class TestRecoverMainCLI:
    """Tests for the recover_main CLI entry point."""

    def _run_cli(self, argv, capsys):
        """Import and invoke main; return (rc, parsed_json)."""
        from orchestrator.recover_main import main  # noqa: PLC0415
        rc = main(argv)
        out = capsys.readouterr().out.strip()
        return rc, json.loads(out)

    def test_rewound_result_json(self, config_path, project_root, capsys):
        """CLI prints {result: 'rewound', target_sha: ...} and exits 0 on success."""
        with patch(
            'orchestrator.recover_main.GitOps.recover_red_main',
            new_callable=AsyncMock,
            return_value='rewound',
        ):
            rc, data = self._run_cli([
                '--project-root', str(project_root),
                '--config', str(config_path),
                '--target-sha', _TARGET_SHA,
                '--expected-main', _EXPECTED_MAIN,
            ], capsys)

        assert rc == 0, f'Expected exit 0 on rewound; got {rc}'
        assert data['result'] == 'rewound', f'Unexpected result: {data}'
        assert data['target_sha'] == _TARGET_SHA, f'target_sha missing/wrong: {data}'

    def test_cas_failed_result_json(self, config_path, project_root, capsys):
        """CLI prints {result: 'cas_failed', ...} and exits non-zero on CAS failure."""
        with patch(
            'orchestrator.recover_main.GitOps.recover_red_main',
            new_callable=AsyncMock,
            return_value='cas_failed',
        ):
            rc, data = self._run_cli([
                '--project-root', str(project_root),
                '--config', str(config_path),
                '--target-sha', _TARGET_SHA,
                '--expected-main', _EXPECTED_MAIN,
            ], capsys)

        assert rc != 0, f'Expected non-zero exit on cas_failed; got {rc}'
        assert data['result'] == 'cas_failed', f'Unexpected result: {data}'

    def test_cli_passes_correct_args_to_recover_red_main(
        self, config_path, project_root, capsys,
    ):
        """CLI invokes recover_red_main(target_sha, expected_main) with the parsed args."""
        mock = AsyncMock(return_value='rewound')
        with patch('orchestrator.recover_main.GitOps.recover_red_main', mock):
            self._run_cli([
                '--project-root', str(project_root),
                '--config', str(config_path),
                '--target-sha', _TARGET_SHA,
                '--expected-main', _EXPECTED_MAIN,
            ], capsys)

        mock.assert_called_once_with(_TARGET_SHA, _EXPECTED_MAIN)

    def test_output_is_single_parseable_json_line(self, config_path, project_root, capsys):
        """stdout is exactly one parseable JSON object (no stray lines)."""
        with patch(
            'orchestrator.recover_main.GitOps.recover_red_main',
            new_callable=AsyncMock,
            return_value='rewound',
        ):
            self._run_cli([
                '--project-root', str(project_root),
                '--config', str(config_path),
                '--target-sha', _TARGET_SHA,
                '--expected-main', _EXPECTED_MAIN,
            ], capsys)

        raw = capsys.readouterr().out.strip()
        # json.loads already validated in _run_cli; assert single-line here
        assert '\n' not in raw, f'Multiple lines in stdout: {raw!r}'

    def test_error_path_json_contract(self, config_path, project_root, capsys):
        """On load_config / GitOps / recover_red_main exception, JSON contract holds.

        A JSON-parsing consumer (e.g. the escalation-watcher skill script) must
        not receive a raw traceback — it must receive a parseable JSON object so
        it can route the failure to a human rather than failing opaquely.
        """
        from orchestrator.recover_main import main  # noqa: PLC0415

        with patch(
            'orchestrator.recover_main.load_config',
            side_effect=ValueError('config file not found or invalid'),
        ):
            rc = main([
                '--project-root', str(project_root),
                '--config', str(config_path),
                '--target-sha', _TARGET_SHA,
                '--expected-main', _EXPECTED_MAIN,
            ])

        raw = capsys.readouterr().out.strip()
        assert rc != 0, f'Expected non-zero exit on exception; got {rc}'
        data = json.loads(raw)  # must be parseable JSON (not a raw traceback)
        assert data['result'] == 'error', f'Expected result=error; got {data}'
        assert 'detail' in data, f'Missing detail key in error output; got {data}'
        assert data['target_sha'] == _TARGET_SHA, f'target_sha missing/wrong; got {data}'

    def test_cli_engages_bypass_for_exactly_cas_window(self, tmp_path, capsys):
        """End-to-end (real repo, unmocked recover_red_main): bypass engaged for exactly the CAS window.

        Builds a real git repo with a good commit and a simulated bad-merge
        commit, installs a ``reference-transaction`` hook (git>=2.28 feature
        baseline) that records whether the bypass flag-file is present when the
        refs/heads/main transaction fires, and configures the ``git:`` block's
        main_gate_bypass_command (create flag) / main_gate_bypass_clear_command
        (remove flag).  Asserts the flag is PRESENT during the ref txn (engaged
        for exactly the CAS window), the flag is removed afterward (cleared —
        no durable leak), the CLI exits 0 with result 'rewound', and main now
        points at the good SHA.  Proves the config git.* fields flow end-to-end
        through load_config -> GitOps -> recover_red_main.
        """
        repo = tmp_path / 'watched'
        repo.mkdir()

        def _git(*a: str) -> str:
            return subprocess.run(
                ['git', *a], cwd=repo, check=True,
                capture_output=True, text=True,
            ).stdout.strip()

        _git('init', '-b', 'main')
        _git('config', 'user.email', 'test@test.com')
        _git('config', 'user.name', 'Test')
        (repo / 'README.md').write_text('# Test\n')
        _git('add', '-A')
        _git('commit', '-m', 'good commit')
        good_sha = _git('rev-parse', 'HEAD')
        # Simulate a bad merge landing on main (the state to recover FROM).
        (repo / 'bad.txt').write_text('simulated bad merge\n')
        _git('add', '-A')
        _git('commit', '-m', 'bad merge on main')
        bad_sha = _git('rev-parse', 'HEAD')

        flag_file = tmp_path / 'bypass.flag'
        obs_file = tmp_path / 'hook_observations.txt'

        # reference-transaction hook: when the refs/heads/main txn fires, record
        # whether the bypass flag-file is present.  `exit 0` keeps the txn from
        # aborting (the trailing `read` at EOF returns non-zero otherwise).
        hooks_dir = repo / '.git' / 'hooks'
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook = hooks_dir / 'reference-transaction'
        hook.write_text(
            '#!/bin/sh\n'
            'while read -r line; do\n'
            '  case "$line" in\n'
            '    *refs/heads/main)\n'
            f'      if [ -e "{flag_file}" ]; then\n'
            f'        echo "present $1" >> "{obs_file}"\n'
            '      else\n'
            f'        echo "absent $1" >> "{obs_file}"\n'
            '      fi\n'
            '      ;;\n'
            '  esac\n'
            'done\n'
            'exit 0\n',
        )
        hook.chmod(0o755)

        cfg = tmp_path / 'orchestrator.yaml'
        cfg.write_text(
            'git:\n'
            '  main_branch: main\n'
            f'  main_gate_bypass_command: "touch {flag_file}"\n'
            f'  main_gate_bypass_clear_command: "rm -f {flag_file}"\n',
        )

        from orchestrator.recover_main import main  # noqa: PLC0415
        rc = main([
            '--project-root', str(repo),
            '--config', str(cfg),
            '--target-sha', good_sha,
            '--expected-main', bad_sha,
        ])
        out = capsys.readouterr().out.strip()
        data = json.loads(out)

        assert rc == 0, f'Expected exit 0 on rewound; got {rc}; out={out}'
        assert data['result'] == 'rewound', f'Expected rewound; got {data}'

        observations = obs_file.read_text() if obs_file.exists() else ''
        assert 'present' in observations, (
            f'bypass flag NOT observed present during the ref txn — bypass was '
            f'not engaged for the CAS window; observations: {observations!r}'
        )
        assert 'absent' not in observations, (
            f'bypass flag observed ABSENT during the ref txn — engage window is '
            f'wrong; observations: {observations!r}'
        )
        # Cleared afterward — the durable bypass must not leak past the CAS window.
        assert not flag_file.exists(), (
            'bypass flag leaked: main_gate_bypass_clear_command did not run'
        )
        # main now points at the good SHA.
        head = _git('rev-parse', 'refs/heads/main')
        assert head == good_sha, f'main not rewound to good; head={head}, good={good_sha}'
