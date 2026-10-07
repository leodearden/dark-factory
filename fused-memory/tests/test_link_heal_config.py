"""The link-heal config surface (plans/write-triage-link-healing-prd.md H1/H2, tasks 6181, 6184).

The ``link_heal.*`` knobs: the executor's caps and storm escapes (H1) and the
link adjudicator's model, sharding, text cap, share ceilings and failure streak
(H2). All are green-tier because their only consumers are scripts
(``scripts/link_heal.py``, ``scripts/eval_link_adjudicator.py``) that load
config at each run start; and
the ``link-heal-`` prefix on the metadata arm of ``mem0_update``, which the
executor's ``update_memory`` writes need and which must not widen the
content-amend arm.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from fused_memory.config.reload import RELOADABLE_FIELDS, diff_config
from fused_memory.config.schema import (
    FusedMemoryConfig,
    LinkHealConfig,
    Mem0UpdateConfig,
)
from fused_memory.server.mem0_update_authz import resolve_mem0_update_authorization

TRACKED_CONFIG_YAML = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'

LEAVES = tuple(sorted(LinkHealConfig.model_fields))


@pytest.fixture(autouse=True)
def _pin_config_path(tmp_path, monkeypatch):
    monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))


class TestLinkHealConfigDefaults:
    def test_defaults(self):
        cfg = LinkHealConfig()
        assert cfg.max_actions_per_run == 25
        assert cfg.backlog_multiplier == 5
        assert cfg.write_failure_streak == 3
        assert cfg.adjudicator_model == 'opus'
        assert cfg.shard_size == 40
        assert cfg.field_chars == 4000
        assert cfg.misfile_share_ceiling == 0.25
        assert cfg.corrects_share_ceiling == 0.60
        assert cfg.shard_failure_streak == 3

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('max_actions_per_run', -1),
            ('backlog_multiplier', 0),
            ('write_failure_streak', 0),
            ('shard_size', 0),
            ('field_chars', 0),
            ('shard_failure_streak', 0),
            ('misfile_share_ceiling', -0.01),
            ('misfile_share_ceiling', 1.01),
            ('corrects_share_ceiling', -0.01),
            ('corrects_share_ceiling', 1.01),
            ('adjudicator_model', ''),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field, value):
        with pytest.raises(ValidationError):
            LinkHealConfig(**{field: value})

    def test_zero_actions_per_run_is_the_apply_nothing_switch(self):
        assert LinkHealConfig(max_actions_per_run=0).max_actions_per_run == 0


class TestLinkHealSectionOnFusedMemoryConfig:
    def test_is_a_bare_submodel_with_a_default_factory(self):
        field = FusedMemoryConfig.model_fields['link_heal']
        assert field.annotation is LinkHealConfig
        assert field.default_factory is LinkHealConfig

    def test_a_missing_config_file_yields_the_defaults(self):
        assert FusedMemoryConfig().link_heal == LinkHealConfig()


class TestLinkHealLeavesAreGreenTier:
    CHANGED = {
        'max_actions_per_run': 7,
        'backlog_multiplier': 2,
        'write_failure_streak': 9,
        'adjudicator_model': 'sonnet',
        'shard_size': 20,
        'field_chars': 2000,
        'misfile_share_ceiling': 0.5,
        'corrects_share_ceiling': 0.9,
        'shard_failure_streak': 5,
    }

    def test_the_schema_declares_nine_leaves(self):
        assert len(LEAVES) == 9, LEAVES

    def test_every_leaf_is_allowlisted(self):
        missing = {f'link_heal.{name}' for name in LEAVES} - RELOADABLE_FIELDS
        assert not missing, f'unregistered link_heal leaves: {sorted(missing)}'

    def test_the_changed_value_table_covers_every_leaf(self):
        assert set(self.CHANGED) == set(LEAVES)

    @pytest.mark.parametrize('field', LEAVES)
    def test_changed_leaf_lands_in_applied_candidates(self, field):
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        path = f'link_heal.{field}'
        old = getattr(live.link_heal, field)
        new_value = self.CHANGED[field]
        assert old != new_value
        object.__setattr__(fresh.link_heal, field, new_value)

        d = diff_config(live, fresh)

        assert d.applied_candidates[path] == {'old': old, 'new': new_value}
        assert path not in d.restart_required


class TestTrackedConfigAdmitsTheLinkHealPrefixOnTheMetadataArmOnly:
    @pytest.fixture
    def tracked_config(self, monkeypatch) -> FusedMemoryConfig:
        assert TRACKED_CONFIG_YAML.is_file(), TRACKED_CONFIG_YAML
        monkeypatch.setenv('CONFIG_PATH', str(TRACKED_CONFIG_YAML))
        return FusedMemoryConfig()

    def test_metadata_arm_lists_recon_curator_and_link_heal(self, tracked_config):
        prefixes = tracked_config.mem0_update.metadata_patch_allowed_agent_prefixes
        for prefix in ('recon-stage-', 'curator-', 'link-heal-'):
            assert prefix in prefixes

    def test_metadata_arm_keeps_every_schema_default_prefix(self, tracked_config):
        """The YAML list replaces the schema default, so a prefix added there must be mirrored."""
        listed = set(tracked_config.mem0_update.metadata_patch_allowed_agent_prefixes)
        default = set(Mem0UpdateConfig().metadata_patch_allowed_agent_prefixes)
        assert default <= listed, f'missing from config.yaml: {sorted(default - listed)}'

    def test_content_amend_arm_stays_at_the_schema_default(self, tracked_config):
        assert (
            tracked_config.mem0_update.content_amend_allowed_agent_prefixes
            == Mem0UpdateConfig().content_amend_allowed_agent_prefixes
        )

    def test_a_link_heal_agent_may_patch_metadata(self, tracked_config):
        decision = resolve_mem0_update_authorization(
            types.SimpleNamespace(config=tracked_config),
            agent_id='link-heal-0123abcd',
            content_amend=False,
            metadata_patch=True,
        )
        assert decision.allowed, decision

    def test_a_link_heal_agent_may_not_amend_content(self, tracked_config):
        decision = resolve_mem0_update_authorization(
            types.SimpleNamespace(config=tracked_config),
            agent_id='link-heal-0123abcd',
            content_amend=True,
            metadata_patch=False,
        )
        assert not decision.allowed
        assert decision.error_type == 'Mem0UpdateNotAuthorized'
