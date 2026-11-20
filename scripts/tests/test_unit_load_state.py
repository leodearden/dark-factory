"""scripts/unit_load_state.py: what a systemd LoadState says about the installed unit file."""
from __future__ import annotations

import pytest
from unit_load_state import LoadVerdict, load_verdict


@pytest.mark.parametrize(
    ('load_state', 'verdict'),
    [
        ('loaded', LoadVerdict.LOADED),
        ('not-found', LoadVerdict.NOT_INSTALLED),
        ('', LoadVerdict.NOT_INSTALLED),
        ('masked', LoadVerdict.MASKED),
        ('bad-setting', LoadVerdict.NOT_LOADED),
        ('error', LoadVerdict.NOT_LOADED),
        ('stub', LoadVerdict.NOT_LOADED),
    ],
)
def test_each_load_state_reads_as_its_verdict(load_state: str, verdict: LoadVerdict) -> None:
    assert load_verdict(load_state) is verdict
