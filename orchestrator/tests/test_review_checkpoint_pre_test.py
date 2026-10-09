"""review.pre_test_commands: the config knobs and their wiring into the review checkpoint.

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from orchestrator.config import OrchestratorConfig, ReviewConfig, apply_reload


class TestReviewPreTestConfig:
    def test_defaults(self):
        cfg = ReviewConfig()
        assert cfg.pre_test_commands == []
        assert cfg.pre_test_timeout_secs == 1800.0

    def test_commands_preserve_order(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig(review={'pre_test_commands': ['a', 'b']})
        assert cfg.review.pre_test_commands == ['a', 'b']

    @pytest.mark.parametrize('bad', [0, -1])
    def test_non_positive_timeout_rejected(self, bad):
        with pytest.raises(ValidationError):
            ReviewConfig(pre_test_timeout_secs=bad)

    def test_apply_reload_hot_applies_pre_test_knobs(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig()
        fresh = OrchestratorConfig(
            review={'pre_test_commands': ['npm run build'], 'pre_test_timeout_secs': 600.0},
        )

        report = apply_reload(live, fresh)

        assert report['reloaded'] is True
        for path in ('review.pre_test_commands', 'review.pre_test_timeout_secs'):
            assert path in report['applied']
            assert path not in report['restart_required']
        assert live.review.pre_test_commands == ['npm run build']
        assert live.review.pre_test_timeout_secs == 600.0
