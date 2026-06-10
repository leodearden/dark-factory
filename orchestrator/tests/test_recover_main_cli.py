"""Tests for orchestrator.recover_main CLI — step-7 RED.

Mirrors test_b3_gate.py's CLI-invocation + JSON-parse pattern:
monkeypatch recover_red_main to return a known result, capture stdout,
json.loads it, assert required keys.
"""

import json
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
