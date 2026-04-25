"""Contract tests for ``conftest.py`` correctness.

This module guards two invariants that would silently regress under refactoring
and are NOT already covered by other tests:

1. **sys.path ordering** — ``conftest.py`` must insert worktree-local source
   directories onto ``sys.path`` *before* any ``from _orch_helpers``,
   ``from shared``, or ``from orchestrator`` import, so that worktree-local
   code takes precedence over installed-package versions.

2. **Sub-section ``spec_set`` wiring** — each sub-section of
   ``mock_orch_config`` (usage_cap, review, sandbox, fused_memory, escalation)
   must be ``spec_set``'d so that typos raise ``AttributeError`` immediately
   rather than silently creating phantom attributes.

Tests of plain attribute defaults (e.g. ``mock.usage_cap.enabled is False``)
are deliberately omitted — they would just duplicate literals from
``conftest.py`` two lines away. The 7 downstream harness consumers that set
known fields like ``mock.max_concurrent_tasks`` implicitly validate top-level
``pydantic_spec`` correctness; any regression there breaks them all.
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
    ['usage_cap', 'enabld'],
    ['review', 'enabld'],
    ['sandbox', 'bakcend'],
    ['fused_memory', 'projcet_id'],
    ['escalation', 'hsot'],
])
def test_subsection_typo_rejected(mock_orch_config, attr_path):
    """Typos on spec_set'd sub-sections raise AttributeError on assignment.

    Guards the sub-section ``spec_set`` wiring on ``mock_orch_config``.  If a
    refactor accidentally drops ``spec_set=`` from a sub-section, typos would
    silently create phantom attributes instead of raising.  The 7 downstream
    harness consumers only set *known* top-level fields, so they would not
    catch a sub-section wiring regression — this test does.
    """
    obj = mock_orch_config
    for attr in attr_path[:-1]:
        obj = getattr(obj, attr)
    with pytest.raises(AttributeError):
        setattr(obj, attr_path[-1], 'anything')
