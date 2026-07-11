"""Tests for cockpit.panes.weight_editor — pure weight-edit merge + known-projects
helpers (Fleet Cockpit C9b, PRD §9).

Pure, deterministic unit tests only -- no Textual import, no pilot. The
WeightEditorScreen(ModalScreen) widget itself is covered by test_app.py's
pilot tests instead, mirroring test_decision_queue.py/test_spawn_bar.py's own
split between fast pure-helper tests here and slower app-level pilot tests.
"""

from __future__ import annotations

from dataclasses import replace


class TestMergeWeightEdits:
    def test_updates_only_category_and_project_weights(self):
        from cockpit.panes.weight_editor import merge_weight_edits
        from cockpit.priority import Priorities

        base = Priorities.default()

        result = merge_weight_edits(
            base,
            category_edits={'bug': '4.0'},
            project_edits={'df': '9.0', 'newproj': '2.5'},
        )

        assert result.category_weights == {**base.category_weights, 'bug': 4.0}
        assert result.project_weights == {'df': 9.0, 'newproj': 2.5}
        assert result.severity_weights == base.severity_weights
        assert result.defaults == base.defaults
        assert result.age_curve == base.age_curve
        assert result.manual_boost == base.manual_boost

    def test_unparseable_or_empty_edits_are_skipped_and_never_raise(self):
        from cockpit.panes.weight_editor import merge_weight_edits
        from cockpit.priority import Priorities

        base = replace(Priorities.default(), project_weights={'df': 3.0})

        unparseable = merge_weight_edits(base, category_edits={}, project_edits={'df': 'xx'})
        empty = merge_weight_edits(base, category_edits={}, project_edits={'df': ''})

        assert unparseable.project_weights == {'df': 3.0}
        assert empty.project_weights == {'df': 3.0}
