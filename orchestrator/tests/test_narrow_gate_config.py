"""The ``narrow_merge_gate`` config section: its defaults, project-YAML path, strictness and hot reload (task 5411)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
    load_config,
)
from orchestrator.narrow_gate_config import NarrowMergeGateConfig


def _project_yaml(tmp_path: Path, narrow_merge_gate: dict[str, object]) -> Path:
    path = tmp_path / 'dark-factory-orchestrator.yaml'
    path.write_text(yaml.safe_dump({'project_root': str(tmp_path), 'narrow_merge_gate': narrow_merge_gate}))
    return path


class TestDefaults:

    def test_the_gate_is_off_with_no_docs_paths(self) -> None:
        gate = NarrowMergeGateConfig()

        assert (gate.enabled, gate.docs_paths) == (False, [])

    @pytest.mark.usefixtures('code_default_config')
    def test_a_bare_orchestrator_config_carries_the_gate_off(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)

        assert config.narrow_merge_gate.enabled is False
        assert config.narrow_merge_gate.docs_paths == []


class TestInProjectYaml:

    def test_project_yaml_loads_through(self, tmp_path: Path) -> None:
        path = _project_yaml(tmp_path, {'enabled': True, 'docs_paths': ['*.md', 'docs/**']})

        gate = load_config(path).narrow_merge_gate

        assert (gate.enabled, gate.docs_paths) == (True, ['*.md', 'docs/**'])

    def test_an_unknown_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match='docs_path'):
            NarrowMergeGateConfig.model_validate({'enabled': True, 'docs_path': ['*.md']})

    def test_an_unknown_key_in_the_project_yaml_is_rejected(self, tmp_path: Path) -> None:
        path = _project_yaml(tmp_path, {'enabeld': True})

        with pytest.raises(ValidationError, match='enabeld'):
            load_config(path)


class TestHotReload:

    def test_every_leaf_is_hot(self) -> None:
        assert {'narrow_merge_gate.enabled', 'narrow_merge_gate.docs_paths'} <= RELOADABLE_FIELDS

    @pytest.mark.usefixtures('code_default_config')
    def test_a_reload_turns_the_held_gate_on_in_place(self) -> None:
        live = OrchestratorConfig()
        fresh = OrchestratorConfig(narrow_merge_gate=NarrowMergeGateConfig(enabled=True))
        held = live.narrow_merge_gate

        report = apply_reload(live, fresh)

        assert 'narrow_merge_gate.enabled' in report['applied']
        assert 'narrow_merge_gate.enabled' not in report['restart_required']
        assert live.narrow_merge_gate is held
        assert held.enabled is True
