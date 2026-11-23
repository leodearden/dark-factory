"""The ``merge_pool`` config file: its defaults, bounds, project-YAML path and hot reload (task 9933)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    census_config_keys,
    load_config,
)
from orchestrator.merge_pool_config import MergePoolConfig


def _project_yaml(tmp_path: Path, merge_pool: dict[str, object]) -> Path:
    path = tmp_path / 'orchestrator.yaml'
    path.write_text(yaml.dump({'merge_pool': merge_pool}))
    return path


class TestBounds:

    def test_defaults_keep_arrival_order_with_no_promotion(self) -> None:
        pool = MergePoolConfig()

        assert (pool.order, pool.promote_after_secs, pool.bounce_cap) == ('arrival', None, 20)

    def test_order_accepts_priority(self) -> None:
        assert MergePoolConfig(order='priority').order == 'priority'

    def test_order_rejects_an_unknown_order(self) -> None:
        with pytest.raises(ValidationError):
            MergePoolConfig.model_validate({'order': 'fifo'})

    @pytest.mark.parametrize('period', [0.0, -60.0])
    def test_promotion_period_must_be_positive(self, period: float) -> None:
        with pytest.raises(ValidationError):
            MergePoolConfig(promote_after_secs=period)

    def test_promotion_period_accepts_nine_hours(self) -> None:
        assert MergePoolConfig(promote_after_secs=32400.0).promote_after_secs == 32400.0

    def test_bounce_cap_must_be_at_least_one(self) -> None:
        with pytest.raises(ValidationError):
            MergePoolConfig(bounce_cap=0)


class TestInProjectYaml:

    def test_project_yaml_loads_through(self, tmp_path: Path) -> None:
        path = _project_yaml(tmp_path, {'order': 'priority', 'promote_after_secs': 32400, 'bounce_cap': 5})

        pool = load_config(path).merge_pool

        assert (pool.order, pool.promote_after_secs, pool.bounce_cap) == ('priority', 32400.0, 5)

    def test_every_leaf_is_a_known_census_key(self, tmp_path: Path) -> None:
        path = _project_yaml(tmp_path, {'order': 'priority', 'promote_after_secs': 32400, 'bounce_cap': 5})

        assert census_config_keys(path).unknown == []

    def test_a_typo_under_merge_pool_still_censuses(self, tmp_path: Path) -> None:
        path = _project_yaml(tmp_path, {'bounce_capp': 5})

        assert 'merge_pool.bounce_capp' in [uk.path for uk in census_config_keys(path).unknown]


class TestHotReload:

    def test_every_leaf_is_hot(self) -> None:
        assert {
            'merge_pool.order', 'merge_pool.promote_after_secs', 'merge_pool.bounce_cap',
        } <= RELOADABLE_FIELDS

    @pytest.mark.usefixtures('code_default_config')
    def test_a_reload_retunes_the_held_submodel_in_place(self) -> None:
        live = OrchestratorConfig()
        fresh = OrchestratorConfig(merge_pool=MergePoolConfig(order='priority'))
        held = live.merge_pool

        report = apply_reload(live, fresh)

        assert 'merge_pool.order' in report['applied']
        assert 'merge_pool.order' not in report['restart_required']
        assert live.merge_pool is held
        assert held.order == 'priority'
