"""The ``early_start`` config group (``plans/early-start-on-predecessor-branch-prd.md`` section 7, Config).

Seven hot-reloadable leaves whose defaults live in the config model only, not
in ``defaults.yaml``.  Every test runs on the code defaults, so the operational
yaml cannot leak into what is asserted.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from orchestrator.config import (
    RELOADABLE_FIELDS,
    EarlyStartConfig,
    OrchestratorConfig,
    apply_reload,
    diff_config,
    load_config,
)

pytestmark = pytest.mark.usefixtures('code_default_config')

LEAVES = {
    'enabled',
    'pass_interval_secs',
    'picture_max_age_secs',
    'merge_entry_alert_secs',
    'stop_streak_pause',
    'stop_hard_cancel_secs',
    'unknown_streak_alert',
}


def test_the_code_default():
    """One test on purpose: T7 turns ``enabled`` on by changing this one value."""
    assert OrchestratorConfig().early_start == EarlyStartConfig(
        enabled=False,
        pass_interval_secs=60,
        picture_max_age_secs=600,
        merge_entry_alert_secs=43200,
        stop_streak_pause=5,
        stop_hard_cancel_secs=300,
        unknown_streak_alert=3,
    )


def test_every_leaf_is_hot_reloadable():
    assert set(EarlyStartConfig.model_fields) == LEAVES
    assert {f'early_start.{leaf}' for leaf in LEAVES} <= RELOADABLE_FIELDS


def test_a_yaml_stanza_overrides_the_named_leaves_only(tmp_path: Path):
    config_file = tmp_path / 'project.yaml'
    config_file.write_text('early_start:\n  enabled: true\n  stop_streak_pause: 7\n')

    early_start = load_config(config_file).early_start

    assert early_start == EarlyStartConfig(enabled=True, stop_streak_pause=7)
    assert early_start.pass_interval_secs == 60


def test_a_reload_applies_to_the_live_config():
    live = OrchestratorConfig()
    fresh = OrchestratorConfig(early_start=EarlyStartConfig(stop_streak_pause=9))

    assert diff_config(live, fresh).restart_required == {}
    report = apply_reload(live, fresh)

    assert report['reloaded'] is True
    assert report['applied'] == {'early_start.stop_streak_pause': {'old': 5, 'new': 9}}
    assert live.early_start.stop_streak_pause == 9


@pytest.mark.parametrize('leaf', ['stop_streak_pause', 'unknown_streak_alert', 'pass_interval_secs'])
def test_a_zero_is_refused(leaf):
    with pytest.raises(ValidationError, match=leaf):
        EarlyStartConfig(**{leaf: 0})
