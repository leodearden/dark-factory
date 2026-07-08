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

    def test_partial_config_merges_missing_sections_from_defaults(self, tmp_path):
        """A YAML file providing only some sections must fall back to defaults for the rest."""
        from cockpit.priority import Priorities, load_priorities

        custom_path = tmp_path / 'priorities.yaml'
        custom_path.write_text(yaml.safe_dump({'severity_weights': {'critical': 42.0}}))

        result = load_priorities(custom_path)
        default = Priorities.default()

        assert result.severity_weights == {'critical': 42.0}
        assert result.category_weights == default.category_weights
        assert result.project_weights == default.project_weights
        assert result.defaults == default.defaults
        assert result.age_curve == default.age_curve
        assert result.manual_boost == default.manual_boost

    def test_partial_section_merges_missing_fields_from_defaults(self, tmp_path):
        """A section provided with only some of its fields must fall back per-field."""
        from cockpit.priority import Priorities, load_priorities

        custom_path = tmp_path / 'priorities.yaml'
        custom_path.write_text(yaml.safe_dump({'age_curve': {'max_bonus': 9.0}}))

        result = load_priorities(custom_path)
        default = Priorities.default()

        assert result.age_curve.max_bonus == 9.0
        assert result.age_curve.saturation_seconds == default.age_curve.saturation_seconds
        assert result.severity_weights == default.severity_weights
        assert result.category_weights == default.category_weights
        assert result.project_weights == default.project_weights
        assert result.defaults == default.defaults
        assert result.manual_boost == default.manual_boost

    def test_saturation_seconds_zero_does_not_raise_in_score(self, tmp_path):
        """A hand-edited priorities.yaml with saturation_seconds<=0 must not break score().

        The cockpit is a view and must never become a hard dependency on a
        well-formed config file — see cockpit.priority.score's non-positive
        saturation_seconds guard.
        """
        from datetime import UTC, datetime

        from cockpit.priority import ScoringItem, load_priorities, score

        custom_path = tmp_path / 'priorities.yaml'
        custom_path.write_text(
            yaml.safe_dump({'age_curve': {'max_bonus': 2.0, 'saturation_seconds': 0}})
        )

        weights = load_priorities(custom_path)
        item = ScoringItem(
            severity='medium',
            category='chore',
            project='some-project',
            filed_at=datetime(2026, 7, 1, tzinfo=UTC),
            manual_boost=0,
            state='open',
        )

        result = score(item, weights, datetime(2026, 7, 7, tzinfo=UTC))

        assert isinstance(result, float)

    def test_null_weights_section_falls_back_and_score_returns_float(self, tmp_path):
        """A hand-edited priorities.yaml with empty (null) weight sections must not crash.

        ``severity_weights:`` with nothing after the colon is the natural way
        an operator empties out a section by hand; yaml.safe_load parses that
        to ``None`` (not ``{}``) because the key is PRESENT but has no value.
        load_priorities must still return usable (non-None) weight tables —
        falling back to Priorities.default() — so score()'s .get() lookups
        never raise AttributeError on None.
        """
        from datetime import UTC, datetime

        from cockpit.priority import Priorities, ScoringItem, load_priorities, score

        custom_path = tmp_path / 'priorities.yaml'
        custom_path.write_text('severity_weights:\ncategory_weights:\nproject_weights:\n')

        result = load_priorities(custom_path)
        default = Priorities.default()

        assert result.severity_weights == default.severity_weights
        assert result.category_weights == default.category_weights
        assert result.project_weights == default.project_weights

        item = ScoringItem(
            severity='medium',
            category='chore',
            project='some-project',
            filed_at=datetime(2026, 7, 1, tzinfo=UTC),
            manual_boost=0,
            state='open',
        )

        assert isinstance(score(item, result, datetime(2026, 7, 7, tzinfo=UTC)), float)


    def test_scalar_or_list_sections_fall_back_without_raising(self, tmp_path):
        """A section set to a scalar/list must fall back to defaults, never raise.

        A hand-edited priorities.yaml where a section is a scalar or list
        (e.g. ``age_curve: 0``, ``defaults: foo``, ``manual_boost: 5``,
        ``severity_weights: [1, 2]``) is parseable YAML but malformed. It must
        not make load_priorities() raise AttributeError — that would violate
        the documented 'never raises' contract and the cockpit's 'a view must
        never be a dependency' guarantee.
        """
        from cockpit.priority import Priorities, load_priorities

        custom_path = tmp_path / 'priorities.yaml'
        custom_path.write_text(
            yaml.safe_dump(
                {
                    'severity_weights': [1, 2],
                    'category_weights': 3,
                    'project_weights': 'nope',
                    'defaults': 'foo',
                    'age_curve': 0,
                    'manual_boost': 5,
                }
            )
        )

        result = load_priorities(custom_path)

        assert result == Priorities.default()

    def test_scalar_section_does_not_raise_in_score(self, tmp_path):
        """The scores from a scalar-section config must still be usable floats."""
        from datetime import UTC, datetime

        from cockpit.priority import ScoringItem, load_priorities, score

        custom_path = tmp_path / 'priorities.yaml'
        custom_path.write_text(yaml.safe_dump({'age_curve': 0, 'severity_weights': 7}))

        weights = load_priorities(custom_path)
        item = ScoringItem(
            severity='medium',
            category='chore',
            project='some-project',
            filed_at=datetime(2026, 7, 1, tzinfo=UTC),
            manual_boost=0,
            state='open',
        )

        assert isinstance(score(item, weights, datetime(2026, 7, 7, tzinfo=UTC)), float)


class TestEnsurePrioritiesFile:
    def test_creates_parent_dirs_and_round_trips_to_defaults(self, tmp_path):
        from cockpit.priority import Priorities, ensure_priorities_file, load_priorities

        target = tmp_path / 'nested' / 'dir' / 'priorities.yaml'
        assert not target.parent.exists()

        ensure_priorities_file(target)

        assert target.exists()
        assert load_priorities(target) == Priorities.default()

    def test_existing_file_is_left_untouched(self, tmp_path):
        from cockpit.priority import ensure_priorities_file

        target = tmp_path / 'priorities.yaml'
        custom_contents = 'severity_weights:\n  critical: 123.0\n'
        target.write_text(custom_contents)

        ensure_priorities_file(target)

        assert target.read_text() == custom_contents

    def test_write_failure_is_fail_soft_and_warns(self, tmp_path, caplog):
        """An unwritable target must fail soft: logged WARNING, never an exception.

        This is the cockpit's 'a view must never be a dependency' guarantee —
        forced here by making the target's parent a *file* rather than a
        directory, so mkdir(parents=True) raises OSError.
        """
        from cockpit.priority import ensure_priorities_file

        blocker = tmp_path / 'blocker'
        blocker.write_text('not a directory')
        target = blocker / 'priorities.yaml'

        with caplog.at_level(logging.WARNING):
            ensure_priorities_file(target)

        assert not target.exists()
        warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any('priorities' in msg.lower() for msg in warnings), (
            f'Expected a WARNING mentioning "priorities"; got: {warnings}'
        )
