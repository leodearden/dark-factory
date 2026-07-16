"""Tests for dashboard.data.escalation_analytics — escalation lifecycle analytics.

Backend data layer for plans/escalation-lifecycle-dashboard-prd.md Seam 2
(task gamma / 2658): archive aggregator (origin/lifespan/workflow blocks),
regime-markers loader, and the pure-sync `build_escalation_analytics` core.
"""

from __future__ import annotations

from dashboard.data.escalation_analytics import load_regime_markers

# ---------------------------------------------------------------------------
# step-1: load_regime_markers
# ---------------------------------------------------------------------------


class TestLoadRegimeMarkers:
    """load_regime_markers(path) -> (markers, parse_failures_delta). Never raises."""

    def test_default_path_parses_committed_seed_file(self):
        """The committed dashboard/regime-markers.yaml parses to exactly 3 markers."""
        markers, parse_failures_delta = load_regime_markers()

        assert parse_failures_delta == 0
        assert len(markers) == 3
        for m in markers:
            assert set(m) == {'date', 'label', 'tasks'}
            # Must be JSON-serializable: yaml.safe_load parses unquoted
            # YYYY-MM-DD as a datetime.date, which is NOT JSON-serializable —
            # the loader must normalize it to a str.
            assert isinstance(m['date'], str)
            assert isinstance(m['label'], str)
            assert isinstance(m['tasks'], list)

        all_tasks = sorted(t for m in markers for t in m['tasks'])
        assert all_tasks == [2593, 2630, 2631]

    def test_malformed_yaml_returns_empty_and_one_failure(self, tmp_path):
        """Unparseable YAML syntax -> ([], 1), never raises."""
        bad = tmp_path / 'bad.yaml'
        bad.write_text('date: [unclosed')

        markers, delta = load_regime_markers(bad)

        assert markers == []
        assert delta == 1

    def test_non_list_mapping_top_level_returns_empty_and_one_failure(self, tmp_path):
        """A top-level mapping (not a list) -> ([], 1)."""
        mapping_path = tmp_path / 'mapping.yaml'
        mapping_path.write_text('date: 2026-07-15\nlabel: not a list\n')

        markers, delta = load_regime_markers(mapping_path)

        assert markers == []
        assert delta == 1

    def test_non_list_scalar_top_level_returns_empty_and_one_failure(self, tmp_path):
        """A top-level scalar (not a list) -> ([], 1)."""
        scalar_path = tmp_path / 'scalar.yaml'
        scalar_path.write_text('just a string\n')

        markers, delta = load_regime_markers(scalar_path)

        assert markers == []
        assert delta == 1

    def test_missing_path_returns_empty_no_failure(self, tmp_path):
        """A missing file -> ([], 0) — not a parse failure, just absent."""
        missing = tmp_path / 'does-not-exist.yaml'

        markers, delta = load_regime_markers(missing)

        assert markers == []
        assert delta == 0
