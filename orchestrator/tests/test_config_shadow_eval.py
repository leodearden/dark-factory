"""ShadowEvalConfig: the live-shadow-eval knobs (plans/live-shadow-eval-prd.md C6).

Fixtures are module-local (no conftest edit), following
test_config_reload_integration_gate.py's real-load_config reload rig.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from orchestrator.cli import main
from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    ShadowEvalConfig,
    apply_reload,
    census_config_keys,
    load_config,
)
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.shadow_types import ShadowShape

_FULL_BLOCK = {
    'enabled': True,
    'sample_rate': 0.25,
    'candidate_sample_rates': {'glm-5.3-endpoint': 0.5},
    'shapes': ['implementer', 'architect-consequence'],
    'candidates': ['glm-5.3-endpoint', 'kimi-endpoint@one-step'],
    'end_to_end_candidates': ['kimi-endpoint@one-step'],
    'max_concurrent': 2,
    'daily_budget_usd': 25.0,
    'settle_deadline_hours': 72.0,
    'failure_streak_pause': 3,
    'cost_ratio_ceiling': 2.5,
    'n_min': 20,
    'seed': 7,
}


class TestDefaults:
    def test_code_defaults(self):
        cfg = ShadowEvalConfig()
        assert cfg.enabled is False
        assert cfg.sample_rate == 0.0
        assert cfg.candidate_sample_rates == {}
        assert cfg.shapes == []
        assert cfg.candidates == []
        assert cfg.end_to_end_candidates == []
        assert cfg.max_concurrent == 1
        assert cfg.daily_budget_usd == 50.0
        assert cfg.settle_deadline_hours == 168.0
        assert cfg.failure_streak_pause == 5
        assert cfg.cost_ratio_ceiling == 3.0
        assert cfg.n_min == 12
        assert cfg.seed == 0

    @pytest.mark.usefixtures('code_default_config')
    def test_orchestrator_config_carries_the_code_defaults(self):
        assert OrchestratorConfig().shadow_eval == ShadowEvalConfig()


class TestValidation:
    @pytest.mark.parametrize(
        'overrides,named',
        [
            ({'sample_rate': -0.1}, 'sample_rate'),
            ({'sample_rate': 1.1}, 'sample_rate'),
            ({'candidates': ['c'], 'candidate_sample_rates': {'c': 1.5}}, 'candidate_sample_rates'),
            ({'max_concurrent': 0}, 'max_concurrent'),
            ({'daily_budget_usd': -1}, 'daily_budget_usd'),
            ({'settle_deadline_hours': 0}, 'settle_deadline_hours'),
            ({'failure_streak_pause': 0}, 'failure_streak_pause'),
            ({'cost_ratio_ceiling': 0}, 'cost_ratio_ceiling'),
            ({'n_min': 0}, 'n_min'),
            ({'shapes': ['reviewer']}, 'shapes'),
            ({'candidates': ['']}, 'candidates'),
            ({'candidates': ['c'], 'end_to_end_candidates': ['ghost']}, 'ghost'),
            ({'candidates': ['c'], 'candidate_sample_rates': {'phantom': 0.5}}, 'phantom'),
        ],
    )
    def test_out_of_contract_values_are_rejected(self, overrides, named):
        with pytest.raises(ValidationError) as excinfo:
            ShadowEvalConfig.model_validate(overrides)
        assert named in str(excinfo.value)

    def test_yaml_shape_strings_parse_to_the_enum(self):
        cfg = ShadowEvalConfig.model_validate({'shapes': ['implementer']})
        assert cfg.shapes == [ShadowShape.IMPLEMENTER]
        assert cfg.shapes[0] is ShadowShape.IMPLEMENTER


class TestSampleRateFor:
    def test_the_override_wins_else_the_block_rate(self):
        cfg = ShadowEvalConfig(
            sample_rate=0.1,
            candidates=['a', 'b'],
            candidate_sample_rates={'a': 0.9},
        )
        assert cfg.sample_rate_for('a') == 0.9
        assert cfg.sample_rate_for('b') == 0.1


@pytest.mark.parametrize('leaf', list(ShadowEvalConfig.model_fields))
def test_every_leaf_is_green_tier(leaf):
    assert f'shadow_eval.{leaf}' in RELOADABLE_FIELDS


def _write_yaml(tmp_path: Path, block: dict) -> Path:
    path = tmp_path / 'orchestrator.yaml'
    path.write_text(yaml.safe_dump({'shadow_eval': block}))
    return path


class TestCensusAndCli:
    def test_a_full_block_is_known_and_check_config_passes(self, tmp_path):
        path = _write_yaml(tmp_path, _FULL_BLOCK)
        assert census_config_keys(path).unknown == []
        result = CliRunner().invoke(main, ['check-config', '--config', str(path)])
        assert result.exit_code == 0, result.output

    def test_a_typo_is_named_and_check_config_fails(self, tmp_path):
        block = {k: v for k, v in _FULL_BLOCK.items() if k != 'enabled'}
        block['enabeld'] = True
        path = _write_yaml(tmp_path, block)
        assert [key.path for key in census_config_keys(path).unknown] == ['shadow_eval.enabeld']
        result = CliRunner().invoke(main, ['check-config', '--config', str(path)])
        assert result.exit_code == 1, result.output
        assert 'shadow_eval.enabeld' in result.output


class TestApplyReload:
    @pytest.fixture(autouse=True)
    def _no_config_file(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')

    def test_a_retune_applies_in_place_and_the_report_serialises(self):
        live = OrchestratorConfig()
        held = live.shadow_eval
        fresh = OrchestratorConfig(
            shadow_eval=ShadowEvalConfig(sample_rate=0.5, candidates=['glm-5.3-endpoint']),
        )

        report = apply_reload(live, fresh)

        assert report['reloaded'] is True, report['error']
        for path in ('shadow_eval.sample_rate', 'shadow_eval.candidates'):
            assert path in report['applied']
            assert path not in report['restart_required']
        assert live.shadow_eval is held
        assert held.sample_rate == 0.5
        assert held.candidates == ['glm-5.3-endpoint']
        json.dumps(report)


def _config_reload_rows(event_store: EventStore) -> list[dict]:
    conn = sqlite3.connect(str(event_store.db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM events WHERE event_type = 'config_reload' ORDER BY id",
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


class TestHarnessReload:
    @pytest.mark.asyncio
    async def test_enabling_shadow_eval_is_a_hot_reload_with_an_audit_row(
        self, tmp_path: Path, monkeypatch,
    ):
        config_path = tmp_path / 'orchestrator.yaml'
        config_path.write_text(yaml.safe_dump({'shadow_eval': {'enabled': False}}))
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))
        harness = Harness(load_config())
        harness.event_store = EventStore(tmp_path / 'events.db', 'run-test-0001')

        config_path.write_text(yaml.safe_dump({
            'shadow_eval': {'enabled': True, 'candidates': ['glm-5.3-endpoint']},
        }))
        report = await harness.reload_config()

        assert report['reloaded'] is True, report['error']
        assert 'shadow_eval.enabled' in report['applied']
        assert 'shadow_eval.candidates' in report['applied']
        assert harness.config.shadow_eval.enabled is True
        rows = _config_reload_rows(harness.event_store)
        assert len(rows) == 1
        assert 'shadow_eval.enabled' in json.loads(rows[0]['data'])['applied']
