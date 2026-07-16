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
from datetime import UTC, datetime, timedelta
from importlib import resources as pkg_resources
from unittest.mock import MagicMock

import pytest
import yaml
from _orch_helpers import pydantic_spec

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


# ── Step-7 / Step-8: De-flake fix-task argument builder ───────────────────────


class TestBuildChronicFlakeFixTaskArguments:
    """``build_chronic_flake_fix_task_arguments``: models
    ``workflow.build_offline_lane_fix_task_arguments`` — deterministic
    title/priority/project_root, an evidence-bearing description with an
    explicit root-cause (never blind timeout bumps) instruction, and
    correlation metadata."""

    def _evidence(self):
        from orchestrator.chronic_flake import ChronicFlakeEvidence
        return ChronicFlakeEvidence(
            test='test_pool_x.sh',
            count=3,
            window=20,
            dates=['2026-07-01T00:00:00Z', '2026-07-05T00:00:00Z', '2026-07-10T00:00:00Z'],
            roles=['implementer', 'verifier'],
            entries=[],
        )

    def test_title_priority_and_project_root(self, tmp_path):
        from orchestrator.chronic_flake import build_chronic_flake_fix_task_arguments
        args = build_chronic_flake_fix_task_arguments(self._evidence(), tmp_path)
        assert args['title'] == 'De-flake test_pool_x.sh: chronic pool flake (auto-filed)'
        assert args['priority'] == 'medium'
        assert args['project_root'] == str(tmp_path)

    def test_description_contains_ledger_evidence(self, tmp_path):
        from orchestrator.chronic_flake import build_chronic_flake_fix_task_arguments
        args = build_chronic_flake_fix_task_arguments(self._evidence(), tmp_path)
        description = args['description']
        assert 'test_pool_x.sh' in description
        assert '3' in description
        assert '20' in description
        assert '2026-07-01T00:00:00Z' in description
        assert 'implementer' in description
        assert 'verifier' in description

    def test_description_contains_root_cause_instruction(self, tmp_path):
        from orchestrator.chronic_flake import build_chronic_flake_fix_task_arguments
        args = build_chronic_flake_fix_task_arguments(self._evidence(), tmp_path)
        description = args['description'].lower()
        assert 'condition-poll' in description or 'condition poll' in description
        assert 'structural assert' in description
        assert 'never' in description
        assert 'blind timeout' in description

    def test_metadata_correlation_keys(self, tmp_path):
        from orchestrator.chronic_flake import build_chronic_flake_fix_task_arguments
        args = build_chronic_flake_fix_task_arguments(self._evidence(), tmp_path)
        metadata = args['metadata']
        assert metadata['spawn_context'] == 'chronic_flake_auto_file'
        assert metadata['chronic_flake_test'] == 'test_pool_x.sh'
        assert metadata['chronic_flake_count'] == 3
        assert metadata['chronic_flake_window'] == 20
        assert metadata['roles'] == ['implementer', 'verifier']


# ── Step-9 / Step-10: FilingLedger rate-limit persistence ─────────────────────


class TestFilingLedger:
    """``FilingLedger``: a small persistent JSON store ({test: last_filed_iso})
    backing the per-test rate limit — injectable ``now`` (never
    ``datetime.now()`` internally), tolerant of a missing or corrupt on-disk
    file. Durability verified via re-open (a SECOND instance at the same
    path), mirroring ``LandedOutbox``'s test idiom."""

    def test_fresh_ledger_should_file_is_true(self, tmp_path):
        from orchestrator.chronic_flake import FilingLedger
        ledger = FilingLedger(tmp_path / 'chronic_flake_filings.json')
        now = datetime(2026, 7, 17, tzinfo=UTC)
        assert ledger.should_file('test_a.sh', now, 7) is True

    def test_record_save_reload_rate_limits_within_window(self, tmp_path):
        from orchestrator.chronic_flake import FilingLedger
        path = tmp_path / 'chronic_flake_filings.json'
        now = datetime(2026, 7, 17, tzinfo=UTC)
        ledger = FilingLedger(path)
        ledger.record('test_a.sh', now)
        ledger.save()

        reloaded = FilingLedger(path)
        assert reloaded.should_file('test_a.sh', now, 7) is False

    def test_should_file_true_again_once_rate_limit_window_elapses(self, tmp_path):
        from orchestrator.chronic_flake import FilingLedger
        path = tmp_path / 'chronic_flake_filings.json'
        now = datetime(2026, 7, 17, tzinfo=UTC)
        ledger = FilingLedger(path)
        ledger.record('test_a.sh', now)
        ledger.save()

        reloaded = FilingLedger(path)
        later = now + timedelta(days=8)
        assert reloaded.should_file('test_a.sh', later, 7) is True

    def test_load_tolerates_missing_file(self, tmp_path):
        from orchestrator.chronic_flake import FilingLedger
        ledger = FilingLedger(tmp_path / 'does-not-exist.json')
        now = datetime(2026, 7, 17, tzinfo=UTC)
        assert ledger.should_file('test_a.sh', now, 7) is True

    def test_load_tolerates_corrupt_file(self, tmp_path):
        from orchestrator.chronic_flake import FilingLedger
        path = tmp_path / 'chronic_flake_filings.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('not valid json {{{')
        ledger = FilingLedger(path)
        now = datetime(2026, 7, 17, tzinfo=UTC)
        assert ledger.should_file('test_a.sh', now, 7) is True

    def test_second_test_name_is_independent(self, tmp_path):
        from orchestrator.chronic_flake import FilingLedger
        path = tmp_path / 'chronic_flake_filings.json'
        now = datetime(2026, 7, 17, tzinfo=UTC)
        ledger = FilingLedger(path)
        ledger.record('test_a.sh', now)
        ledger.save()

        reloaded = FilingLedger(path)
        assert reloaded.should_file('test_a.sh', now, 7) is False
        assert reloaded.should_file('test_b.sh', now, 7) is True


# ── Step-11 / Step-12: maybe_file_chronic_flake_tasks happy-path/dedup/rate-limit ──


class _FakeChronicFlakeTaskClient:
    """Fake ``ChronicFlakeTaskClient`` (submit_task/search_tasks) — records
    every ``submit_task`` call and returns canned ``search_tasks`` results
    regardless of the query text, since the dedup contract under test is
    "does an OPEN result's title match the test", not the exact query
    ``maybe_file_chronic_flake_tasks`` happens to build."""

    def __init__(self, search_results: list[dict] | None = None):
        self.search_results = search_results if search_results is not None else []
        self.submit_calls: list[dict] = []

    async def submit_task(self, arguments: dict) -> str:
        self.submit_calls.append(arguments)
        return 'ticket-fake-123'

    async def search_tasks(self, query: str) -> list[dict]:
        return self.search_results


class TestMaybeFileChronicFlakeTasks:
    """``maybe_file_chronic_flake_tasks``: enabled-gate, union(marker, ledger)
    chronic-test detection, and the dedup/rate-limit contract — an OPEN
    (non-terminal, non-deferred) search_tasks title match or an
    unexpired FilingLedger entry both suppress a submit; a CHRONIC-FLAKY
    marker is an authoritative trigger even when the ledger itself is
    sub-threshold for that test."""

    NOW = datetime(2026, 7, 17, tzinfo=UTC)

    def _config(self, tmp_path, **overrides):
        from orchestrator.config import ChronicFlakeConfig
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.project_root = tmp_path
        kwargs = {
            'enabled': True,
            'threshold': 3,
            'window': 20,
            'rate_limit_days': 7,
            'ledger_relpath': 'data/verify-logs/flaky-ledger.jsonl',
        }
        kwargs.update(overrides)
        config.chronic_flake = ChronicFlakeConfig(**kwargs)
        return config

    def _write_ledger(self, path, entries):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(e) for e in entries) + '\n')

    def _chronic_entries(self, test='test_a.sh', n=3):
        return [
            {
                'ts': f'2026-07-0{i}T00:00:00Z',
                'test': test,
                'role': 'verifier',
                'flaky_count_window': i,
            }
            for i in range(1, n + 1)
        ]

    @pytest.mark.asyncio
    async def test_disabled_config_returns_empty_and_zero_submits(self, tmp_path):
        from orchestrator.chronic_flake import maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger.jsonl'
        self._write_ledger(ledger_path, self._chronic_entries())
        client = _FakeChronicFlakeTaskClient()
        result = await maybe_file_chronic_flake_tasks(
            '',
            self._config(tmp_path, enabled=False),
            client,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=tmp_path / 'filings.json',
        )
        assert result == []
        assert client.submit_calls == []

    @pytest.mark.asyncio
    async def test_enabled_chronic_ledger_files_exactly_once_and_records_ledger(self, tmp_path):
        from orchestrator.chronic_flake import FilingLedger, maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger.jsonl'
        self._write_ledger(ledger_path, self._chronic_entries())
        filings_path = tmp_path / 'filings.json'
        client = _FakeChronicFlakeTaskClient()

        result = await maybe_file_chronic_flake_tasks(
            '',
            self._config(tmp_path),
            client,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=filings_path,
        )

        assert result == ['test_a.sh']
        assert len(client.submit_calls) == 1
        args = client.submit_calls[0]
        assert args['title'] == 'De-flake test_a.sh: chronic pool flake (auto-filed)'
        assert args['project_root'] == str(tmp_path)
        assert args['metadata']['chronic_flake_test'] == 'test_a.sh'
        # The rate-limit ledger durably recorded the filing (save() was called).
        reloaded = FilingLedger(filings_path)
        assert reloaded.should_file('test_a.sh', self.NOW, 7) is False

    @pytest.mark.asyncio
    async def test_marker_triggers_even_when_ledger_is_subthreshold(self, tmp_path):
        """The marker is reify's own authoritative "this is chronic" signal
        — a test that hasn't yet reached DF's own threshold in the ledger
        still files when the marker says so."""
        from orchestrator.chronic_flake import maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger.jsonl'
        self._write_ledger(ledger_path, self._chronic_entries(test='test_b.sh', n=2))
        verify_output = '=== CHRONIC-FLAKY test=test_b.sh count=5 window=20 ===\n'
        client = _FakeChronicFlakeTaskClient()

        result = await maybe_file_chronic_flake_tasks(
            verify_output,
            self._config(tmp_path),
            client,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=tmp_path / 'filings.json',
        )

        assert result == ['test_b.sh']
        assert len(client.submit_calls) == 1
        assert client.submit_calls[0]['metadata']['chronic_flake_test'] == 'test_b.sh'

    @pytest.mark.asyncio
    async def test_open_search_match_skips_submit(self, tmp_path):
        from orchestrator.chronic_flake import maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger.jsonl'
        self._write_ledger(ledger_path, self._chronic_entries())
        client = _FakeChronicFlakeTaskClient(
            search_results=[
                {'title': 'De-flake test_a.sh: chronic pool flake (auto-filed)', 'status': 'pending'},
            ]
        )

        result = await maybe_file_chronic_flake_tasks(
            '',
            self._config(tmp_path),
            client,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=tmp_path / 'filings.json',
        )

        assert result == []
        assert client.submit_calls == []

    @pytest.mark.asyncio
    async def test_closed_search_match_does_not_block_submit(self, tmp_path):
        """A DONE/cancelled/deferred search hit is NOT an open dedup match
        (mirrors workflow.py's TERMINAL_STATUSES-or-deferred idiom) — the
        test still files."""
        from orchestrator.chronic_flake import maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger.jsonl'
        self._write_ledger(ledger_path, self._chronic_entries())
        client = _FakeChronicFlakeTaskClient(
            search_results=[
                {'title': 'De-flake test_a.sh: chronic pool flake (auto-filed)', 'status': 'done'},
            ]
        )

        result = await maybe_file_chronic_flake_tasks(
            '',
            self._config(tmp_path),
            client,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=tmp_path / 'filings.json',
        )

        assert result == ['test_a.sh']
        assert len(client.submit_calls) == 1

    @pytest.mark.asyncio
    async def test_recent_filing_in_ledger_skips_submit(self, tmp_path):
        from orchestrator.chronic_flake import FilingLedger, maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger.jsonl'
        self._write_ledger(ledger_path, self._chronic_entries())
        filings_path = tmp_path / 'filings.json'
        pre = FilingLedger(filings_path)
        pre.record('test_a.sh', self.NOW - timedelta(days=1))
        pre.save()
        client = _FakeChronicFlakeTaskClient()

        result = await maybe_file_chronic_flake_tasks(
            '',
            self._config(tmp_path),
            client,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=filings_path,
        )

        assert result == []
        assert client.submit_calls == []


# ── Step-13 / Step-14: non-blocking guarantee ──────────────────────────────


class _SubmitRaisingTaskClient:
    """search_tasks never matches; submit_task raises for one configured
    test and records the rest — pins "one bad test must not take down the
    whole filing pass"."""

    def __init__(self, raise_for_test: str):
        self._raise_for_test = raise_for_test
        self.submit_calls: list[dict] = []

    async def submit_task(self, arguments: dict) -> str:
        test = arguments['metadata']['chronic_flake_test']
        if test == self._raise_for_test:
            raise RuntimeError(f'boom: submit_task failed for {test}')
        self.submit_calls.append(arguments)
        return 'ticket-ok'

    async def search_tasks(self, query: str) -> list[dict]:
        return []


class _SearchRaisingTaskClient:
    """search_tasks always raises; submit_task records calls — pins "a
    search hiccup must degrade to no-match, not block filing"."""

    def __init__(self):
        self.submit_calls: list[dict] = []

    async def submit_task(self, arguments: dict) -> str:
        self.submit_calls.append(arguments)
        return 'ticket-ok'

    async def search_tasks(self, query: str) -> list[dict]:
        raise RuntimeError('boom: search_tasks failed')


class TestMaybeFileChronicFlakeTasksNonBlocking:
    """Pins the task's hardest constraint: "filing failures must never fail
    the verify/merge path." ``maybe_file_chronic_flake_tasks`` must never
    raise, regardless of what the task_client or the ledger file does."""

    NOW = datetime(2026, 7, 17, tzinfo=UTC)

    def _config(self, tmp_path, **overrides):
        from orchestrator.config import ChronicFlakeConfig
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.project_root = tmp_path
        kwargs = {
            'enabled': True,
            'threshold': 3,
            'window': 20,
            'rate_limit_days': 7,
            'ledger_relpath': 'data/verify-logs/flaky-ledger.jsonl',
        }
        kwargs.update(overrides)
        config.chronic_flake = ChronicFlakeConfig(**kwargs)
        return config

    def _write_ledger(self, path, entries):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(json.dumps(e) for e in entries) + '\n')

    def _chronic_entries(self, test='test_a.sh', n=3):
        return [
            {
                'ts': f'2026-07-0{i}T00:00:00Z',
                'test': test,
                'role': 'verifier',
                'flaky_count_window': i,
            }
            for i in range(1, n + 1)
        ]

    @pytest.mark.asyncio
    async def test_submit_task_raise_for_one_test_does_not_propagate_and_continues(self, tmp_path):
        from orchestrator.chronic_flake import maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger.jsonl'
        entries = self._chronic_entries(test='test_a.sh', n=3) + self._chronic_entries(
            test='test_c.sh', n=3
        )
        self._write_ledger(ledger_path, entries)
        client = _SubmitRaisingTaskClient(raise_for_test='test_a.sh')

        result = await maybe_file_chronic_flake_tasks(
            '',
            self._config(tmp_path),
            client,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=tmp_path / 'filings.json',
        )

        assert result == ['test_c.sh']
        assert [a['metadata']['chronic_flake_test'] for a in client.submit_calls] == ['test_c.sh']

    @pytest.mark.asyncio
    async def test_search_tasks_raise_is_treated_as_no_match_and_does_not_propagate(self, tmp_path):
        from orchestrator.chronic_flake import maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger.jsonl'
        self._write_ledger(ledger_path, self._chronic_entries())
        client = _SearchRaisingTaskClient()

        result = await maybe_file_chronic_flake_tasks(
            '',
            self._config(tmp_path),
            client,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=tmp_path / 'filings.json',
        )

        assert result == ['test_a.sh']
        assert len(client.submit_calls) == 1

    @pytest.mark.asyncio
    async def test_unreadable_ledger_file_is_swallowed_and_returns_empty(self, tmp_path):
        """A ledger *path* that raises on read (here: it's actually a
        directory, portably triggering IsADirectoryError) is swallowed by
        the outer catch-all — distinct from a malformed JSON *line*, which
        ``read_flaky_ledger`` itself already tolerates (step-5/step-6)."""
        from orchestrator.chronic_flake import maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger_is_a_directory'
        ledger_path.mkdir()
        client = _FakeChronicFlakeTaskClient()

        result = await maybe_file_chronic_flake_tasks(
            '',
            self._config(tmp_path),
            client,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=tmp_path / 'filings.json',
        )

        assert result == []
        assert client.submit_calls == []

    @pytest.mark.asyncio
    async def test_none_task_client_is_log_only_noop(self, tmp_path):
        from orchestrator.chronic_flake import maybe_file_chronic_flake_tasks
        ledger_path = tmp_path / 'ledger.jsonl'
        self._write_ledger(ledger_path, self._chronic_entries())

        result = await maybe_file_chronic_flake_tasks(
            '',
            self._config(tmp_path),
            None,
            now=self.NOW,
            ledger_path=ledger_path,
            filings_path=tmp_path / 'filings.json',
        )

        assert result == []
