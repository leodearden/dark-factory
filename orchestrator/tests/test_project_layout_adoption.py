"""The orchestrator reads its store paths and project id from shared (S14 β).

Contract: plans/review-all-s14-shared-leaves-prd.md (decisions 2, 3, 4; B2, B3).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from shared.project_id import project_id_for_root
from shared.project_layout import (
    ESCALATION_QUEUE_RELPATH,
    park_eviction_requests_db,
    scheduler_overrides_db,
)

from orchestrator.config import OrchestratorConfig, load_config


def _load(monkeypatch, root: Path, body: dict) -> OrchestratorConfig:
    for key in [k for k in os.environ if k.startswith('ORCH_')]:
        monkeypatch.delenv(key)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'dark-factory-orchestrator.yaml'
    config_path.write_text(yaml.safe_dump({'project_root': str(root), **body}))
    return load_config(config_path)


class TestEscalationQueueDirIsTheLayoutValue:
    @pytest.mark.parametrize('queue_dir', ['data/esc2', 'ABSOLUTE'])
    def test_a_non_layout_queue_dir_is_refused_naming_the_layout(
        self, monkeypatch, tmp_path, queue_dir,
    ):
        if queue_dir == 'ABSOLUTE':
            queue_dir = str(tmp_path / 'elsewhere')
        with pytest.raises(ValidationError) as excinfo:
            _load(monkeypatch, tmp_path, {'escalation': {'queue_dir': queue_dir}})
        message = str(excinfo.value)
        assert 'shared.project_layout.escalation_queue_dir' in message
        assert queue_dir in message
        assert ESCALATION_QUEUE_RELPATH in message

    @pytest.mark.parametrize('escalation', [None, {'queue_dir': ESCALATION_QUEUE_RELPATH}])
    def test_the_layout_value_loads_whether_omitted_or_declared(
        self, monkeypatch, tmp_path, escalation,
    ):
        body = {} if escalation is None else {'escalation': escalation}
        config = _load(monkeypatch, tmp_path, body)
        assert config.escalation.queue_dir == ESCALATION_QUEUE_RELPATH


class TestFusedMemoryProjectIdDerivesFromRoot:
    def test_an_omitted_project_id_derives_from_the_root_basename(self, monkeypatch, tmp_path):
        root = tmp_path / 'x' / 'Know-Live'
        config = _load(monkeypatch, root, {})
        assert config.fused_memory.project_id == 'know_live'
        assert config.fused_memory.project_id == project_id_for_root(root)

    def test_a_declared_project_id_wins(self, monkeypatch, tmp_path):
        root = tmp_path / 'reify-fix-3'
        config = _load(monkeypatch, root, {'fused_memory': {'project_id': 'reify'}})
        assert config.fused_memory.project_id == 'reify'

    def test_the_derived_id_survives_the_reload_round_trip(self, monkeypatch, tmp_path):
        config = _load(monkeypatch, tmp_path / 'x' / 'Know-Live', {})
        reloaded = OrchestratorConfig.model_validate(config.model_dump())
        assert reloaded.fused_memory.project_id == 'know_live'


def test_scheduler_store_paths_are_the_layout_paths(tmp_path):
    config = OrchestratorConfig(project_root=tmp_path)
    assert config.overrides_db_path == scheduler_overrides_db(tmp_path)
    assert config.park_eviction_requests_db_path == park_eviction_requests_db(tmp_path)
