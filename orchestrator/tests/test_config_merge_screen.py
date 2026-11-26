"""The ``merge_screen`` submodel: defaults, validation and hot-reload.

Task 7919; plans/merge-pre-verify-screen-prd.md decisions 3, 4 and 8. The
submodel is the screen's kill switch (``manifest: null`` runs no screen code)
and its fail-open bounds, so every leaf must be green-tier reloadable.

Configs come from the ``code_default_config`` fixture or an explicit
``merge_screen`` kwarg, so the shipped dark-factory-orchestrator.yaml never
leaks into a default assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    MergeScreenConfig,
    OrchestratorConfig,
    apply_reload,
    load_config,
)

_LEAVES = ('manifest', 'timeout_secs', 'chain_timeout_secs', 'error_streak_alarm')


@pytest.mark.usefixtures('code_default_config')
class TestDefaults:
    def test_model_defaults(self):
        screen = MergeScreenConfig()
        assert screen.manifest is None
        assert screen.timeout_secs == 180.0
        assert screen.chain_timeout_secs == 30.0
        assert screen.error_streak_alarm == 5

    def test_code_default_config_carries_the_kill_switch(self):
        assert OrchestratorConfig().merge_screen == MergeScreenConfig()


class TestValidation:
    @pytest.mark.parametrize(
        'bad_manifest', ['/abs/merge_screen.yaml', '../merge_screen.yaml', 'scripts/../x.yaml', '']
    )
    def test_manifest_must_be_repo_relative(self, bad_manifest):
        with pytest.raises(ValidationError) as excinfo:
            MergeScreenConfig(manifest=bad_manifest)
        assert 'manifest' in str(excinfo.value)

    def test_repo_relative_manifest_accepted(self):
        assert MergeScreenConfig(manifest='scripts/merge_screen.yaml').manifest == (
            'scripts/merge_screen.yaml'
        )

    @pytest.mark.parametrize('leaf', ['timeout_secs', 'chain_timeout_secs'])
    @pytest.mark.parametrize('bad_value', [0, -1, -0.5])
    def test_timeouts_must_be_positive(self, leaf, bad_value):
        with pytest.raises(ValidationError) as excinfo:
            MergeScreenConfig(**{leaf: bad_value})
        assert leaf in str(excinfo.value)

    @pytest.mark.parametrize('bad_value', [0, -3])
    def test_error_streak_alarm_at_least_one(self, bad_value):
        with pytest.raises(ValidationError) as excinfo:
            MergeScreenConfig(error_streak_alarm=bad_value)
        assert 'error_streak_alarm' in str(excinfo.value)

    @pytest.mark.usefixtures('code_default_config')
    def test_rejected_at_load_from_a_project_yaml(self, tmp_path: Path):
        project_cfg = tmp_path / 'orchestrator.yaml'
        project_cfg.write_text(yaml.dump({'merge_screen': {'manifest': '/etc/screen.yaml'}}))
        with pytest.raises(ValidationError) as excinfo:
            load_config(project_cfg)
        assert 'manifest' in str(excinfo.value)


class TestReload:
    @pytest.mark.parametrize('leaf', _LEAVES)
    def test_every_leaf_is_green_tier(self, leaf):
        assert f'merge_screen.{leaf}' in RELOADABLE_FIELDS

    @pytest.mark.usefixtures('code_default_config')
    def test_manifest_toggles_in_place_through_a_yaml_reload(self, tmp_path: Path):
        live = OrchestratorConfig(project_root=tmp_path)
        submodel = live.merge_screen
        project_cfg = tmp_path / 'orchestrator.yaml'

        project_cfg.write_text(yaml.dump({'merge_screen': {'manifest': 'scripts/merge_screen.yaml'}}))
        enable = apply_reload(live, load_config(project_cfg))
        assert enable['reloaded'] is True
        assert enable['applied']['merge_screen.manifest'] == {
            'old': None,
            'new': 'scripts/merge_screen.yaml',
        }
        assert 'merge_screen.manifest' not in enable['restart_required']
        assert live.merge_screen is submodel
        assert submodel.manifest == 'scripts/merge_screen.yaml'

        project_cfg.write_text(yaml.dump({'merge_screen': {'manifest': None}}))
        kill = apply_reload(live, load_config(project_cfg))
        assert kill['reloaded'] is True
        assert kill['applied']['merge_screen.manifest'] == {
            'old': 'scripts/merge_screen.yaml',
            'new': None,
        }
        assert 'merge_screen.manifest' not in kill['restart_required']
        assert live.merge_screen is submodel
        assert submodel.manifest is None
