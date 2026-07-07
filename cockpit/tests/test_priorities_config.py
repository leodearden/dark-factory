"""Tests for cockpit.priority's YAML config loading (load_priorities).

Fail-soft is the hard constraint: a missing user file falls back to the
package-bundled defaults, and a malformed user file falls back to the same
defaults with a logged warning — load_priorities() must never raise.
"""

from __future__ import annotations

import logging

import yaml


class TestLoadPriorities:
    def test_nonexistent_path_returns_bundled_defaults(self, tmp_path):
        from cockpit.priority import Priorities, load_priorities

        missing = tmp_path / 'does-not-exist' / 'priorities.yaml'

        result = load_priorities(missing)

        assert result == Priorities.default()

    def test_explicit_path_returns_custom_weights(self, tmp_path):
        from cockpit.priority import load_priorities

        custom_path = tmp_path / 'priorities.yaml'
        custom_path.write_text(
            yaml.safe_dump(
                {
                    'severity_weights': {'critical': 99.0},
                    'category_weights': {'security': 42.0},
                    'project_weights': {'my-project': 7.0},
                    'defaults': {'severity': 1.0, 'category': 1.0, 'project': 1.0},
                    'age_curve': {'max_bonus': 3.0, 'saturation_seconds': 100.0},
                    'manual_boost': {'weight': 2.0, 'min': -10, 'max': 10},
                }
            )
        )

        result = load_priorities(custom_path)

        assert result.severity_weights == {'critical': 99.0}
        assert result.category_weights == {'security': 42.0}
        assert result.project_weights == {'my-project': 7.0}
        assert result.defaults.severity == 1.0
        assert result.defaults.category == 1.0
        assert result.defaults.project == 1.0
        assert result.age_curve.max_bonus == 3.0
        assert result.age_curve.saturation_seconds == 100.0
        assert result.manual_boost.weight == 2.0
        assert result.manual_boost.min == -10
        assert result.manual_boost.max == 10

    def test_malformed_yaml_falls_back_to_defaults_and_warns(self, tmp_path, caplog):
        """A malformed user file must fail soft: defaults + a WARNING, never an exception."""
        from cockpit.priority import Priorities, load_priorities

        bad_path = tmp_path / 'priorities.yaml'
        bad_path.write_text('severity_weights: {unbalanced')

        with caplog.at_level(logging.WARNING):
            result = load_priorities(bad_path)

        assert result == Priorities.default()
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any('priorities' in msg.lower() for msg in warnings), (
            f'Expected a WARNING mentioning "priorities"; got: {warnings}'
        )
