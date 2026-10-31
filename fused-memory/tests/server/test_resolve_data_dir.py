"""``server/main.py::resolve_data_dir`` is the one place the server names its data dir.

Task 5462.  Every server-side store lives under the directory it returns, so
it must be absolute and anchored at the checkout, never the process CWD.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _fm_helpers import pydantic_spec

from fused_memory.config.checkout_anchor import CHECKOUT_ROOT
from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.server.main import (
    _resolve_curator_cost_store_path,
    _resolve_curator_escalator_state_path,
    resolve_data_dir,
)


def _config(reconciliation) -> MagicMock:
    config = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
    config.reconciliation = reconciliation
    return config


def test_the_configured_data_dir_is_returned_as_a_path(tmp_path):
    resolved = resolve_data_dir(_config(ReconciliationConfig(data_dir=str(tmp_path))))

    assert isinstance(resolved, Path)
    assert resolved == tmp_path


def test_a_relative_value_that_bypassed_the_validator_is_anchored_at_use(
    monkeypatch, tmp_path
):
    """An in-place mutation or a hand-built double never met the schema validator."""
    monkeypatch.chdir(tmp_path)

    assert resolve_data_dir(_config(MagicMock(data_dir='rel/recon'))) == CHECKOUT_ROOT / 'rel' / 'recon'


def test_without_reconciliation_the_stores_live_under_the_checkout_data_dir():
    assert resolve_data_dir(_config(None)) == CHECKOUT_ROOT / 'data'


@pytest.mark.parametrize(
    'reconciliation',
    [
        pytest.param(ReconciliationConfig(data_dir='/srv/recon'), id='configured'),
        pytest.param(None, id='unconfigured'),
    ],
)
def test_the_curator_state_paths_derive_from_it(reconciliation):
    config = _config(reconciliation)

    assert _resolve_curator_cost_store_path(config) == resolve_data_dir(config) / 'curator_events.db'
    assert (
        _resolve_curator_escalator_state_path(config)
        == resolve_data_dir(config) / 'curator_escalator_state.json'
    )
