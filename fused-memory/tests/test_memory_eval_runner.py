"""Tests for memory_eval_runner.py — the scheduled memory-eval runner.

The script is loaded via importlib (``_fm_helpers.load_script_module``), as the
producer scripts' own tests do. No test touches a live store: producers are
injected as tiny fake scripts, and escalations are filed into a real
``EscalationQueue`` rooted under ``tmp_path``.
"""
from __future__ import annotations

import functools
import types
from pathlib import Path

import pytest
import yaml
from _fm_helpers import load_script_module

_FM_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = _FM_ROOT / 'scripts' / 'memory_eval_runner.py'
COMMITTED_CONFIG = _FM_ROOT / 'config' / 'memory-eval.yaml'
_REQUIRED_KEYS = ('storm_alarm_limit', 'min_samples', 'baseline_window', 'runs_per_quarter')


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPT_PATH, mod_name='memory_eval_runner')


def _write_config(tmp_path: Path, **changes) -> Path:
    data = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding='utf-8'))
    for key, value in changes.items():
        if value is None:
            data.pop(key)
        else:
            data[key] = value
    path = tmp_path / 'memory-eval.yaml'
    path.write_text(yaml.safe_dump(data), encoding='utf-8')
    return path


class TestRunnerConfig:
    def test_the_committed_config_declares_a_daily_cadence(self):
        config = _mod().load_runner_config(COMMITTED_CONFIG)
        limits = config.limits_config()
        assert limits.runs_per_quarter == 90
        assert limits.min_samples == config.min_samples
        assert limits.baseline_window == config.baseline_window
        assert limits.false_alarm_budget == config.false_alarm_budget

    @pytest.mark.parametrize('key', _REQUIRED_KEYS)
    def test_a_missing_declaration_is_refused(self, tmp_path, key):
        with pytest.raises(ValueError):
            _mod().load_runner_config(_write_config(tmp_path, **{key: None}))

    def test_an_unknown_key_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            _mod().load_runner_config(_write_config(tmp_path, surprise=1))

    def test_a_non_positive_storm_limit_is_refused(self, tmp_path):
        with pytest.raises(ValueError):
            _mod().load_runner_config(_write_config(tmp_path, storm_alarm_limit=0))


class TestQueueDirResolution:
    def test_a_relative_setting_resolves_against_the_repo_root(self, tmp_path):
        resolved = _mod().resolve_recon_queue_dir('./data/reconciliation/escalations', tmp_path)
        assert resolved == tmp_path / 'data' / 'reconciliation' / 'escalations'

    def test_an_absolute_setting_is_left_alone(self, tmp_path):
        absolute = tmp_path / 'elsewhere'
        assert _mod().resolve_recon_queue_dir(str(absolute), Path('/repo')) == absolute


class TestPreflightExits:
    def _run(self, tmp_path: Path, queue_dir: Path, config: Path = COMMITTED_CONFIG) -> int:
        return _mod().main([
            '--config', str(config),
            '--artifact-root', str(tmp_path / 'artifacts'),
            '--queue-dir', str(queue_dir),
        ])

    def _no_artifacts(self, tmp_path: Path) -> bool:
        root = tmp_path / 'artifacts'
        return not root.exists() or not any(root.iterdir())

    def test_a_missing_queue_dir_exits_2_without_artifacts(self, tmp_path, capsys):
        queue_dir = tmp_path / 'no-such-queue'
        assert self._run(tmp_path, queue_dir) == 2
        assert str(queue_dir) in capsys.readouterr().err
        assert self._no_artifacts(tmp_path)

    def test_an_empty_queue_dir_exits_2_without_artifacts(self, tmp_path, capsys):
        queue_dir = tmp_path / 'empty-queue'
        queue_dir.mkdir()
        assert self._run(tmp_path, queue_dir) == 2
        assert str(queue_dir) in capsys.readouterr().err
        assert self._no_artifacts(tmp_path)

    def test_an_invalid_config_exits_2_without_artifacts(self, tmp_path, capsys):
        config = _write_config(tmp_path, storm_alarm_limit=None)
        assert self._run(tmp_path, tmp_path / 'queue', config) == 2
        assert str(config) in capsys.readouterr().err
        assert self._no_artifacts(tmp_path)
