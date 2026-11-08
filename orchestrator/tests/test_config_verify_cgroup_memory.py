"""Verify-scope cgroup memory limits as a config submodel + green-tier reload
registration (task 6106).

Fixtures are kept MODULE-LOCAL (not conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to fall back to
running the full owning-package suite instead of a scoped subset (mirrors
test_config_verify_admission_reload.py's stated rationale).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    OrchestratorConfig,
    VerifyCgroupMemoryConfig,
    census_unknown_config_keys,
    diff_config,
    load_config,
)

LEAVES = ['high_gib', 'max_gib', 'swap_max_gib']


def _write_yaml(tmp_path: Path, data: dict[str, Any], name: str) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return p


@pytest.mark.usefixtures('code_default_config')
class TestVerifyCgroupMemoryDefaults:

    def test_bare_config_sets_no_limits(self):
        memory = OrchestratorConfig().verify_cgroup_memory
        assert isinstance(memory, VerifyCgroupMemoryConfig)
        assert memory.high_gib is None
        assert memory.max_gib is None
        assert memory.swap_max_gib is None
        assert memory.systemd_properties() == ()


class TestVerifyCgroupMemoryRender:

    def test_all_leaves_render_in_fixed_order(self):
        memory = VerifyCgroupMemoryConfig(high_gib=24, max_gib=40, swap_max_gib=32)
        assert memory.systemd_properties() == (
            'MemoryHigh=24G', 'MemoryMax=40G', 'MemorySwapMax=32G',
        )

    def test_only_max_renders_only_max(self):
        assert VerifyCgroupMemoryConfig(max_gib=40).systemd_properties() == ('MemoryMax=40G',)

    def test_zero_swap_renders_as_no_swap(self):
        assert VerifyCgroupMemoryConfig(swap_max_gib=0).systemd_properties() == (
            'MemorySwapMax=0G',
        )


class TestVerifyCgroupMemoryBounds:

    @pytest.mark.parametrize('leaf', ['high_gib', 'max_gib'])
    @pytest.mark.parametrize('bad_value', [0, -1])
    def test_non_positive_cap_rejected(self, leaf, bad_value):
        with pytest.raises(ValidationError):
            VerifyCgroupMemoryConfig(**{leaf: bad_value})

    def test_negative_swap_rejected(self):
        with pytest.raises(ValidationError):
            VerifyCgroupMemoryConfig(swap_max_gib=-1)

    def test_zero_swap_accepted(self):
        assert VerifyCgroupMemoryConfig(swap_max_gib=0).swap_max_gib == 0


@pytest.mark.usefixtures('code_default_config')
class TestVerifyCgroupMemoryHighNotAboveMax:

    def test_high_above_max_rejected_naming_both(self):
        with pytest.raises(ValidationError) as excinfo:
            VerifyCgroupMemoryConfig(high_gib=48, max_gib=40)
        message = str(excinfo.value)
        assert 'high_gib=48' in message
        assert 'max_gib=40' in message

    def test_high_above_max_rejected_through_orchestrator_config(self):
        yaml_shaped: dict[str, Any] = {'verify_cgroup_memory': {'high_gib': 48, 'max_gib': 40}}
        with pytest.raises(ValidationError):
            OrchestratorConfig(**yaml_shaped)

    def test_high_equal_to_max_accepted(self):
        memory = VerifyCgroupMemoryConfig(high_gib=40, max_gib=40)
        assert (memory.high_gib, memory.max_gib) == (40, 40)

    def test_high_without_max_accepted(self):
        memory = VerifyCgroupMemoryConfig(high_gib=48)
        assert memory.high_gib == 48
        assert memory.max_gib is None


class TestVerifyCgroupMemoryYaml:

    def test_yaml_values_load_with_clean_census(self, tmp_path, monkeypatch):
        p = _write_yaml(
            tmp_path,
            {
                'project_root': str(tmp_path),
                'verify_cgroup_memory': {'high_gib': 24, 'max_gib': 40, 'swap_max_gib': 32},
            },
            name='config.yaml',
        )
        assert census_unknown_config_keys(p) == []
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(p))
        memory = load_config(p).verify_cgroup_memory
        assert (memory.high_gib, memory.max_gib, memory.swap_max_gib) == (24, 40, 32)

    def test_misspelt_leaf_is_censused(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            {'project_root': str(tmp_path), 'verify_cgroup_memory': {'hgh_gib': 24}},
            name='misspelt.yaml',
        )
        paths = [uk.path for uk in census_unknown_config_keys(p)]
        assert paths == ['verify_cgroup_memory.hgh_gib']


@pytest.mark.usefixtures('code_default_config')
class TestVerifyCgroupMemoryReloadDisposition:
    """Every leaf is green-tier: hot-reloadable without a restart.

    That a reloaded limit reaches the next scope spawn is checked end to end in
    test_verify_scope_memory.py.
    """

    @pytest.mark.parametrize('leaf', LEAVES)
    def test_leaf_edit_lands_in_applied_candidates_not_restart_required(self, leaf):
        before: dict[str, Any] = {'verify_cgroup_memory': {leaf: 8}}
        after: dict[str, Any] = {'verify_cgroup_memory': {leaf: 16}}
        diff = diff_config(OrchestratorConfig(**before), OrchestratorConfig(**after))
        path = f'verify_cgroup_memory.{leaf}'
        assert path in diff.applied_candidates
        assert path not in diff.restart_required
