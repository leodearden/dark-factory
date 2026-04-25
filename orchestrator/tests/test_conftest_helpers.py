"""Contract tests for the ``mock_orch_config`` fixture.

These verify the *non-trivial* behavior that the fixture promises and which
wouldn't survive accidental refactoring — specifically that ``spec_set``
typo-rejection is wired up on both the top-level mock and every sub-section.

Tests of plain attribute defaults (e.g. ``mock.usage_cap.enabled is False``)
are deliberately omitted — they would just duplicate literals from
``conftest.py`` two lines away. The 7 downstream harness fixtures that
consume the fixture in earnest are the de-facto contract for those.
"""

import pytest


@pytest.mark.parametrize('attr_path', [
    ['projcet_root'],
    ['usage_cap', 'enabld'],
    ['review', 'enabld'],
    ['sandbox', 'bakcend'],
    ['fused_memory', 'projcet_id'],
    ['escalation', 'hsot'],
])
def test_typo_rejected(mock_orch_config, attr_path):
    """Typos on spec_set'd sub-sections raise AttributeError on assignment."""
    obj = mock_orch_config
    for attr in attr_path[:-1]:
        obj = getattr(obj, attr)
    with pytest.raises(AttributeError):
        setattr(obj, attr_path[-1], 'anything')
