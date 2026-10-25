"""Tests for scripts/legibility/plan_deviation_census.py — the nightly
plan-deviation census (plans/plan-deviation-recording-prd.md D8-D11).

Imports use the package form, `from legibility import plan_deviation_census
as mod` (scripts/tests/conftest.py puts scripts/ on sys.path), so every class
this file builds is the very class the module uses.

Every MCP side effect (`call_tool`) is ALWAYS a recording fake in this file:
no test reaches a real escalation server. State files land under the per-test
legibility state root conftest.py's autouse fixture installs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml
from legibility import config as config_mod
from legibility import plan_deviation_census as mod

_LOGGER = 'legibility.plan_deviation_census'

SHIPPED_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / 'docs' / 'legibility' / 'legibility.yaml'
)


def _write_config(tmp_path: Path, *, block: object = None) -> Path:
    doc: dict[str, object] = {
        'project_id': 'testproj',
        'project_root': str(tmp_path),
        'escalation_port': 8199,
        'cwd_prefixes': [str(tmp_path / 'work')],
    }
    if block is not None:
        doc['plan_deviation_census'] = block
    path = tmp_path / 'legibility.yaml'
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    return path


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == _LOGGER and record.levelno == logging.WARNING
    ]


# ---------------------------------------------------------------------------
# settings validation — settings_from_mapping never raises and falls back
# per field, reporting every rejection in one WARNING
# ---------------------------------------------------------------------------

def test_settings_none_returns_defaults_silently(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)

    settings = mod.settings_from_mapping(None)

    assert settings == config_mod.PlanDeviationCensus()
    assert settings.enabled is True
    assert settings.rate_rise_factor == 2.0
    assert settings.min_records == 10
    assert settings.pattern_share == 0.5
    assert settings.prose_only_min == 3
    assert settings.cooldown_days == 7
    assert settings.horizon_days == 28
    assert settings.meta_roots == ('.worktrees/.task-meta',)
    assert _warnings(caplog) == []


def test_settings_valid_overrides_apply_verbatim(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)

    settings = mod.settings_from_mapping({
        'enabled': False,
        'rate_rise_factor': 3,
        'min_records': 5,
        'pattern_share': 0.6,
        'prose_only_min': 2,
        'cooldown_days': 3,
        'horizon_days': 35,
        'meta_roots': ['a', '/abs/b'],
    })

    assert isinstance(settings, config_mod.PlanDeviationCensus)
    assert settings.enabled is False
    assert settings.rate_rise_factor == 3
    assert settings.min_records == 5
    assert settings.pattern_share == 0.6
    assert settings.prose_only_min == 2
    assert settings.cooldown_days == 3
    assert settings.horizon_days == 35
    assert settings.meta_roots == ('a', '/abs/b')
    assert _warnings(caplog) == []


def test_settings_bad_fields_fall_back_per_field_in_one_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    defaults = config_mod.PlanDeviationCensus()

    settings = mod.settings_from_mapping({
        'rate_rise_factor': '2.0',
        'min_records': True,
        'pattern_share': 1.5,
        'prose_only_min': -1,
        'horizon_days': 7,
        'meta_roots': 'x',
        'enabled': 'yes',
        'rate_rise_factr': 4.0,
        'cooldown_days': 4,
    })

    assert settings.rate_rise_factor == defaults.rate_rise_factor
    assert settings.min_records == defaults.min_records
    assert settings.pattern_share == defaults.pattern_share
    assert settings.prose_only_min == defaults.prose_only_min
    assert settings.horizon_days == defaults.horizon_days
    assert settings.meta_roots == defaults.meta_roots
    assert settings.enabled == defaults.enabled
    assert settings.cooldown_days == 4
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for key in (
        'rate_rise_factor', 'min_records', 'pattern_share', 'prose_only_min',
        'horizon_days', 'meta_roots', 'enabled', 'rate_rise_factr',
    ):
        assert key in message
    assert 'cooldown_days' not in message


def test_settings_zero_cooldown_is_rejected(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)

    settings = mod.settings_from_mapping({'cooldown_days': 0})

    assert settings.cooldown_days == config_mod.PlanDeviationCensus().cooldown_days
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert 'cooldown_days' in warnings[0].getMessage()


def test_settings_non_mapping_block_returns_defaults_with_one_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)

    settings = mod.settings_from_mapping(['enabled', False])

    assert settings == config_mod.PlanDeviationCensus()
    assert len(_warnings(caplog)) == 1


def test_garbage_block_never_fails_config_load(tmp_path):
    block = {
        'enabled': 'sometimes',
        'rate_rise_factor': None,
        'min_records': [1, 2],
        'meta_roots': {'not': 'a list'},
    }

    cfg = config_mod.load_config(_write_config(tmp_path, block=block))

    assert cfg.plan_deviation_census == block


def test_shipped_config_block_is_accepted_without_warning(caplog):
    caplog.set_level(logging.DEBUG, logger=_LOGGER)
    cfg = config_mod.load_config(SHIPPED_CONFIG_PATH)

    assert isinstance(cfg.plan_deviation_census, dict)
    settings = mod.settings_from_mapping(cfg.plan_deviation_census)

    assert settings == config_mod.PlanDeviationCensus()
    assert _warnings(caplog) == []
    assert 'plan_deviation_census:' in SHIPPED_CONFIG_PATH.read_text(encoding='utf-8')
