"""The ticket janitor's dedup-outage detector is wired only while the curator is enabled.

A deliberately disabled curator resolves every ticket as a fast create with no
combine, which is the outage signature itself; wiring the detector then would
page a blocking escalation every window for a state the operator chose.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from _fm_helpers import pydantic_spec

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig
from fused_memory.server.main import _dedup_outage_detector_config


def _config(curator: CuratorConfig) -> MagicMock:
    config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
    config.curator = curator
    return config


def test_an_enabled_curator_wires_the_configured_detector():
    curator = CuratorConfig(enabled=True)

    assert _dedup_outage_detector_config(_config(curator)) is curator.janitor.dedup_outage


def test_a_disabled_curator_wires_no_detector():
    assert _dedup_outage_detector_config(_config(CuratorConfig(enabled=False))) is None
