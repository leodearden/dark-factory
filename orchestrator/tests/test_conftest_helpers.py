"""Contract tests for the ``mock_orch_config`` fixture.

These verify the *non-trivial* behavior that the fixture promises and which
wouldn't survive accidental refactoring — specifically that ``spec_set``
typo-rejection is wired up on both the top-level mock and every sub-section.

Tests of plain attribute defaults (e.g. ``mock.usage_cap.enabled is False``)
are deliberately omitted — they would just duplicate literals from
``conftest.py`` two lines away. The 7 downstream harness fixtures that
consume the fixture in earnest are the de-facto contract for those.
"""

from pathlib import Path

import pytest


def test_syspath_block_precedes_guarded_imports():
    """sys.path.insert must appear in conftest.py before the guarded imports.

    Ensures that worktree-local source is on sys.path before Python resolves
    ``from _orch_helpers``, ``from shared.config_models``, and
    ``from orchestrator.config`` imports — otherwise the installed-package
    version may be loaded instead of the local worktree copy.
    """
    conftest_text = (Path(__file__).parent / 'conftest.py').read_text()

    syspath_pos = conftest_text.index('sys.path.insert')

    guarded_imports = [
        'from _orch_helpers import',
        'from shared.config_models import',
        'from orchestrator.config import',
    ]

    for stmt in guarded_imports:
        stmt_pos = conftest_text.index(stmt)
        assert syspath_pos < stmt_pos, (
            f'sys.path.insert (at offset {syspath_pos}) must appear before '
            f'{stmt!r} (at offset {stmt_pos}) in conftest.py — '
            f'move the sys.path block above the guarded imports'
        )


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
