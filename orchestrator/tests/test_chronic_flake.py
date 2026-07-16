"""Tests for the chronic-flake auto-file feature (task 2358).

Policy (Leo, 2026-07-08, verify-flakiness survey follow-up): after a verify
completes, detect a chronic pool-infra flake from reify's ``run_all.sh``
FLAKY substrate (reify task 5142) and auto-file a medium-priority De-flake
fix task into the project's task tree — non-blocking (the gate stays
green), with dedup + a 7-day rate limit.

Covers:
- OrchestratorConfig ChronicFlakeConfig submodel defaults + defaults.yaml
  round-trip (step-1/step-2)
- CHRONIC-FLAKY marker line-anchored parsing (step-3/step-4)
- Flaky ledger read + chronic-test computation (step-5/step-6)
- De-flake fix-task argument builder (step-7/step-8)
- FilingLedger rate-limit persistence (step-9/step-10)
- maybe_file_chronic_flake_tasks happy-path/dedup/rate-limit (step-11/step-12)
- Non-blocking guarantee (step-13/step-14)
- SchedulerChronicFlakeTaskClient adapter (step-15/step-16)
"""

from __future__ import annotations

import json
from importlib import resources as pkg_resources

import yaml

from orchestrator.config import OrchestratorConfig


def _load_package_defaults() -> dict:
    """Read the shipped defaults.yaml so tests stay in sync automatically."""
    defaults_file = pkg_resources.files('orchestrator') / 'defaults.yaml'
    return yaml.safe_load(defaults_file.read_text())


# ── Step-1 / Step-2: Config default tests ─────────────────────────────────────


class TestChronicFlakeConfigDefaults:
    """OrchestratorConfig exposes a ``ChronicFlakeConfig`` submodel with the
    reify-sourced defaults, shipped OFF (``enabled: false``) until reify:5142
    lands and is confirmed on the target project's main."""

    def test_pydantic_default_enabled_is_false(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['enabled']
        assert field_info.default is False

    def test_pydantic_default_threshold(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['threshold']
        assert field_info.default == 3

    def test_pydantic_default_window(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['window']
        assert field_info.default == 20

    def test_pydantic_default_rate_limit_days(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['rate_limit_days']
        assert field_info.default == 7

    def test_pydantic_default_ledger_relpath(self):
        from orchestrator.config import ChronicFlakeConfig
        field_info = ChronicFlakeConfig.model_fields['ledger_relpath']
        assert field_info.default == 'data/verify-logs/flaky-ledger.jsonl'

    def test_reachable_as_orchestrator_config_attribute(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.chronic_flake.enabled is False
        assert config.chronic_flake.threshold == 3
        assert config.chronic_flake.window == 20
        assert config.chronic_flake.rate_limit_days == 7
        assert config.chronic_flake.ledger_relpath == 'data/verify-logs/flaky-ledger.jsonl'

    def test_defaults_yaml_block_round_trips(self, monkeypatch, tmp_path):
        """The shipped defaults.yaml declares the same chronic_flake: block
        explicitly (including enabled: false) so the feature is discoverable
        and retunable in orchestrator.yaml without guessing at Pydantic
        defaults, mirroring the git:/psi_admission: precedent."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert 'chronic_flake' in defaults
        block = defaults['chronic_flake']
        assert block['enabled'] is False
        assert config.chronic_flake.enabled == block['enabled']
        assert config.chronic_flake.threshold == block['threshold']
        assert config.chronic_flake.window == block['window']
        assert config.chronic_flake.rate_limit_days == block['rate_limit_days']
        assert config.chronic_flake.ledger_relpath == block['ledger_relpath']


# ── Step-3 / Step-4: CHRONIC-FLAKY marker parsing ─────────────────────────────


class TestMatchChronicFlakyMarker:
    """``_match_chronic_flaky_marker``: line-anchored, mirroring
    ``verify._match_clock_marker`` (reify esc-4791-52 / task 4998 discipline)
    — an embedded/mid-line or log-prefixed occurrence must NOT match."""

    def test_well_formed_marker_matches(self):
        from orchestrator.chronic_flake import ChronicFlakeMarker, _match_chronic_flaky_marker
        line = '=== CHRONIC-FLAKY test=test_pool_x.sh count=3 window=20 ==='
        result = _match_chronic_flaky_marker(line)
        assert result == ChronicFlakeMarker(test='test_pool_x.sh', count=3, window=20)

    def test_leading_whitespace_still_matches(self):
        """A marker with only leading WHITESPACE still matches (lstrip-anchored)."""
        from orchestrator.chronic_flake import ChronicFlakeMarker, _match_chronic_flaky_marker
        line = '   \t=== CHRONIC-FLAKY test=test_pool_x.sh count=3 window=20 ==='
        result = _match_chronic_flaky_marker(line)
        assert result == ChronicFlakeMarker(test='test_pool_x.sh', count=3, window=20)

    def test_embedded_in_prose_does_not_match(self):
        """Regression for reify esc-4791-52 / task 4998: a marker quoted
        MID-LINE (assertion prose) must NOT match — same discipline as the
        clock-stop parser fix."""
        from orchestrator.chronic_flake import _match_chronic_flaky_marker
        line = (
            'PASS: C: stderr contains === CHRONIC-FLAKY test=test_pool_x.sh '
            'count=3 window=20 === (hold)'
        )
        assert _match_chronic_flaky_marker(line) is None

    def test_harness_prefixed_line_does_not_match(self):
        """An arbitrary leading log/harness prefix is NOT tolerated either
        (deliberate tightening, mirroring the clock-stop fix)."""
        from orchestrator.chronic_flake import _match_chronic_flaky_marker
        line = (
            '[harness] 2026-06-26T12:00:00 === CHRONIC-FLAKY test=test_pool_x.sh '
            'count=3 window=20 ==='
        )
        assert _match_chronic_flaky_marker(line) is None

    def test_plain_log_line_returns_none(self):
        from orchestrator.chronic_flake import _match_chronic_flaky_marker
        assert _match_chronic_flaky_marker('running tests/infra/run_all.sh') is None

    def test_empty_line_returns_none(self):
        from orchestrator.chronic_flake import _match_chronic_flaky_marker
        assert _match_chronic_flaky_marker('') is None

    def test_malformed_count_returns_none(self):
        from orchestrator.chronic_flake import _match_chronic_flaky_marker
        line = '=== CHRONIC-FLAKY test=test_pool_x.sh count=abc window=20 ==='
        assert _match_chronic_flaky_marker(line) is None

    def test_malformed_window_returns_none(self):
        from orchestrator.chronic_flake import _match_chronic_flaky_marker
        line = '=== CHRONIC-FLAKY test=test_pool_x.sh count=3 window=abc ==='
        assert _match_chronic_flaky_marker(line) is None


class TestParseChronicFlakyMarkers:
    """``parse_chronic_flaky_markers``: scans a multi-line blob and returns
    every valid (line-anchored) marker, in order, ignoring non-matching
    lines (embedded/prefixed occurrences, plain log lines)."""

    def test_returns_every_valid_marker_in_order(self):
        from orchestrator.chronic_flake import ChronicFlakeMarker, parse_chronic_flaky_markers
        blob = '\n'.join([
            'running tests/infra/run_all.sh',
            '=== CHRONIC-FLAKY test=test_a.sh count=3 window=20 ===',
            'PASS: C: stderr contains === CHRONIC-FLAKY test=test_x.sh count=9 '
            'window=20 === (hold)',
            '=== CHRONIC-FLAKY test=test_b.sh count=5 window=20 ===',
            '',
        ])
        result = parse_chronic_flaky_markers(blob)
        assert result == [
            ChronicFlakeMarker(test='test_a.sh', count=3, window=20),
            ChronicFlakeMarker(test='test_b.sh', count=5, window=20),
        ]

    def test_empty_output_returns_empty_list(self):
        from orchestrator.chronic_flake import parse_chronic_flaky_markers
        assert parse_chronic_flaky_markers('') == []

    def test_no_markers_returns_empty_list(self):
        from orchestrator.chronic_flake import parse_chronic_flaky_markers
        assert parse_chronic_flaky_markers('all good\nnothing to see\n') == []


# ── Step-5 / Step-6: Flaky ledger read + chronic-test computation ─────────────


class TestReadFlakyLedger:
    """``read_flaky_ledger``: best-effort per-line JSON read of reify's
    ``{ts, test, role, flaky_count_window}`` ledger rows — tolerant of blank
    lines, malformed JSON, and non-dict rows; missing file → []."""

    def test_reads_well_formed_rows_in_order_skips_blank_and_malformed(self, tmp_path):
        from orchestrator.chronic_flake import read_flaky_ledger
        row_a = {'ts': '2026-07-01T00:00:00Z', 'test': 'test_a.sh', 'role': 'verifier', 'flaky_count_window': 1}
        row_b = {'ts': '2026-07-02T00:00:00Z', 'test': 'test_b.sh', 'role': 'implementer', 'flaky_count_window': 2}
        ledger_path = tmp_path / 'flaky-ledger.jsonl'
        ledger_path.write_text(
            '\n'.join([
                json.dumps(row_a),
                '',
                'not valid json {{{',
                json.dumps(row_b),
            ])
            + '\n'
        )
        assert read_flaky_ledger(ledger_path) == [row_a, row_b]

    def test_skips_non_dict_rows(self, tmp_path):
        from orchestrator.chronic_flake import read_flaky_ledger
        row_a = {'ts': '2026-07-01T00:00:00Z', 'test': 'test_a.sh', 'role': 'verifier', 'flaky_count_window': 1}
        ledger_path = tmp_path / 'flaky-ledger.jsonl'
        ledger_path.write_text(
            '\n'.join([
                json.dumps(['not', 'a', 'dict']),
                json.dumps('just a string'),
                json.dumps(row_a),
            ])
            + '\n'
        )
        assert read_flaky_ledger(ledger_path) == [row_a]

    def test_missing_file_returns_empty_list(self, tmp_path):
        from orchestrator.chronic_flake import read_flaky_ledger
        assert read_flaky_ledger(tmp_path / 'does-not-exist.jsonl') == []


class TestComputeChronicFlakes:
    """``compute_chronic_flakes``: groups the last ``window`` ledger entries
    by test, flags tests occurring ``>= threshold`` times as chronic."""

    def _entry(self, ts, test, role, count):
        return {'ts': ts, 'test': test, 'role': role, 'flaky_count_window': count}

    def test_flags_test_at_or_above_threshold_excludes_sub_threshold(self):
        from orchestrator.chronic_flake import compute_chronic_flakes
        entries = [
            self._entry('2026-07-01', 'test_a.sh', 'verifier', 1),
            self._entry('2026-07-02', 'test_a.sh', 'verifier', 2),
            self._entry('2026-07-03', 'test_a.sh', 'implementer', 3),
            self._entry('2026-07-04', 'test_b.sh', 'verifier', 1),
            self._entry('2026-07-05', 'test_b.sh', 'verifier', 2),
        ]
        result = compute_chronic_flakes(entries, threshold=3, window=20)
        assert len(result) == 1
        evidence = result[0]
        assert evidence.test == 'test_a.sh'
        assert evidence.count == 3
        assert evidence.window == 20
        assert evidence.dates == ['2026-07-01', '2026-07-02', '2026-07-03']
        assert evidence.roles == ['implementer', 'verifier']

    def test_only_last_window_entries_considered(self):
        """A test that only appears OUTSIDE the last `window` entries must
        not be flagged, even if it would meet threshold unwindowed."""
        from orchestrator.chronic_flake import compute_chronic_flakes
        older = [self._entry(f'2026-06-0{i}', 'test_c.sh', 'verifier', i) for i in range(1, 4)]
        recent = [
            self._entry('2026-07-01', 'test_a.sh', 'verifier', 1),
            self._entry('2026-07-02', 'test_a.sh', 'verifier', 2),
            self._entry('2026-07-03', 'test_a.sh', 'implementer', 3),
        ]
        entries = older + recent
        result = compute_chronic_flakes(entries, threshold=3, window=len(recent))
        assert [e.test for e in result] == ['test_a.sh']

    def test_sub_threshold_test_excluded(self):
        from orchestrator.chronic_flake import compute_chronic_flakes
        entries = [
            self._entry('2026-07-01', 'test_b.sh', 'verifier', 1),
            self._entry('2026-07-02', 'test_b.sh', 'verifier', 2),
        ]
        assert compute_chronic_flakes(entries, threshold=3, window=20) == []

    def test_empty_entries_returns_empty_list(self):
        from orchestrator.chronic_flake import compute_chronic_flakes
        assert compute_chronic_flakes([], threshold=3, window=20) == []
