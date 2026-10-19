"""Write triage's attach guard (task 7158): two rules that refuse an attach and store the write standalone.

The FLOOR refuses any attach whose target's cosine to the write is below
``write_triage.min_parent_cosine``; the HOIST GAP refuses, in the judge band
only, a hoist from a matched child to its parent when the parent's cosine is
more than ``write_triage.hoist_max_cosine_gap`` below the child's. Both are
hand-set, hot-reloadable operator knobs; ``None`` switches a rule off. The
evidence for the shipped values lives in the gitignored soak directory of the
main checkout (``data/write-triage-soak-2026-10-19/``).
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from _fm_helpers import load_script_module
from pydantic import ValidationError

from fused_memory.config.reload import RELOADABLE_FIELDS, apply_reload
from fused_memory.config.schema import FusedMemoryConfig, WriteTriageConfig
from fused_memory.server import write_triage
from fused_memory.server.write_triage import (
    resolve_hoist_max_cosine_gap,
    resolve_min_parent_cosine,
)

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SHIPPED_CONFIG = _PACKAGE_ROOT / 'config' / 'config.yaml'
_CALIBRATION_SCRIPT = _PACKAGE_ROOT / 'scripts' / 'calibrate_write_triage.py'

_GUARD_KNOBS = ('min_parent_cosine', 'hoist_max_cosine_gap')


@pytest.fixture(autouse=True)
def _pin_config_path(tmp_path, monkeypatch):
    """Pin CONFIG_PATH at a missing file so ``FusedMemoryConfig()`` is pure schema defaults."""
    monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))


def _service_with(**write_triage_leaves) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        config=types.SimpleNamespace(write_triage=types.SimpleNamespace(**write_triage_leaves)),
    )


class TestTheGuardKnobs:
    def test_the_schema_ships_the_soak_derived_defaults(self):
        config = WriteTriageConfig()

        assert config.min_parent_cosine == 0.60
        assert config.hoist_max_cosine_gap == 0.05

    @pytest.mark.parametrize('knob', _GUARD_KNOBS)
    def test_none_is_the_off_switch(self, knob):
        assert getattr(WriteTriageConfig(**{knob: None}), knob) is None

    @pytest.mark.parametrize('knob', _GUARD_KNOBS)
    @pytest.mark.parametrize('value', [-0.1, 1.5])
    def test_a_value_outside_the_cosine_range_is_rejected(self, knob, value):
        with pytest.raises(ValidationError):
            WriteTriageConfig(**{knob: value})

    @pytest.mark.parametrize('knob', _GUARD_KNOBS)
    def test_each_knob_is_green_tier(self, knob):
        assert f'write_triage.{knob}' in RELOADABLE_FIELDS

    @pytest.mark.parametrize(
        ('knob', 'resolver_name', 'new_value'),
        [
            ('min_parent_cosine', 'resolve_min_parent_cosine', 0.70),
            ('min_parent_cosine', 'resolve_min_parent_cosine', None),
            ('hoist_max_cosine_gap', 'resolve_hoist_max_cosine_gap', 0.10),
            ('hoist_max_cosine_gap', 'resolve_hoist_max_cosine_gap', None),
        ],
    )
    def test_an_applied_reload_is_observed_by_the_live_resolver(
        self, knob, resolver_name, new_value,
    ):
        resolver = getattr(write_triage, resolver_name)
        memory_service = types.SimpleNamespace(config=FusedMemoryConfig())
        assert resolver(memory_service) == getattr(WriteTriageConfig(), knob)

        fresh = FusedMemoryConfig()
        object.__setattr__(fresh.write_triage, knob, new_value)

        report = apply_reload(memory_service.config, fresh)

        assert f'write_triage.{knob}' in report['applied']
        assert resolver(memory_service) == new_value

    @pytest.mark.parametrize('resolver', [resolve_min_parent_cosine, resolve_hoist_max_cosine_gap])
    @pytest.mark.parametrize(
        'memory_service',
        [
            types.SimpleNamespace(config=types.SimpleNamespace()),
            _service_with(),
            _service_with(min_parent_cosine=True, hoist_max_cosine_gap=True),
            _service_with(min_parent_cosine='0.6', hoist_max_cosine_gap='0.05'),
            Mock(),
        ],
        ids=['no-section', 'no-leaf', 'bool', 'str', 'unspecced-mock'],
    )
    def test_an_unreadable_leaf_switches_the_rule_off(self, resolver, memory_service):
        assert resolver(memory_service) is None

    @pytest.mark.parametrize('resolver', [resolve_min_parent_cosine, resolve_hoist_max_cosine_gap])
    def test_an_int_reads_as_a_float(self, resolver):
        value = resolver(_service_with(min_parent_cosine=1, hoist_max_cosine_gap=1))

        assert value == 1.0
        assert isinstance(value, float)

    def test_the_shipped_config_sets_both_knobs_to_cosines(self):
        section = yaml.safe_load(_SHIPPED_CONFIG.read_text())['write_triage']

        for knob in _GUARD_KNOBS:
            value = section[knob]
            assert isinstance(value, int | float) and not isinstance(value, bool), knob
            assert 0.0 <= value <= 1.0, knob

    def test_a_recalibration_never_reverts_a_hand_set_knob(self):
        calibrate = load_script_module(_CALIBRATION_SCRIPT, mod_name='calibrate_write_triage')
        text = _SHIPPED_CONFIG.read_text()

        rewritten = calibrate.write_triage_config_block(
            text, t_high=0.91, t_low=0.51, report_path='calibration/x.json',
        )

        before = yaml.safe_load(text)['write_triage']
        after = yaml.safe_load(rewritten)['write_triage']
        for knob in _GUARD_KNOBS:
            assert after[knob] == before[knob], knob
