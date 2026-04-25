"""Contract tests for ``conftest.py`` correctness.

This module guards three invariants that would silently regress under refactoring
and are NOT already covered by other tests:

1. **sys.path ordering / module resolution** — ``conftest.py`` must insert
   worktree-local source directories onto ``sys.path`` *before* any
   ``from orchestrator`` or ``from shared`` import, so that worktree-local
   code is the version actually loaded (verified behaviorally via ``__file__``).

2. **Top-level ``spec_set`` wiring** — the top-level ``MagicMock`` in
   ``mock_orch_config`` must use ``spec_set=pydantic_spec(OrchestratorConfig)``
   so that typos raise ``AttributeError`` immediately.  Downstream harness
   consumers only assign *valid* top-level fields, so they would pass silently
   even if ``spec_set`` were dropped — this test does not.

3. **Sub-section ``spec_set`` wiring** — each sub-section of
   ``mock_orch_config`` (usage_cap, review, sandbox, fused_memory, escalation)
   must be ``spec_set``'d so that typos raise ``AttributeError`` immediately
   rather than silently creating phantom attributes.

Tests of plain attribute defaults (e.g. ``mock.usage_cap.enabled is False``)
are deliberately omitted — they would just duplicate literals from
``conftest.py`` two lines away.
"""

from pathlib import Path

import pytest


def test_syspath_block_precedes_guarded_imports():
    """Worktree-local source is the version of ``orchestrator.config`` / ``shared.config_models`` loaded.

    Verifies the *behaviour* produced by conftest.py's sys.path ordering: the
    modules resolved at import time must come from the worktree-local src
    directories, not from an installed-package copy.

    If conftest.py's sys.path.insert block were moved *after* the guarded
    imports, Python would resolve to the installed-package version and this
    test would fail with a path mismatch.
    """
    import shared.config_models

    import orchestrator.config

    _src = (Path(__file__).parent.parent / 'src').resolve()
    _shared_src = (Path(__file__).parent.parent.parent / 'shared' / 'src').resolve()

    orch_file = Path(orchestrator.config.__file__).resolve()
    shared_file = Path(shared.config_models.__file__).resolve()

    assert orch_file.is_relative_to(_src), (
        f'orchestrator.config loaded from {orch_file} — expected a path under '
        f'{_src}. Check that the sys.path.insert block in conftest.py runs '
        f'before the guarded imports.'
    )
    assert shared_file.is_relative_to(_shared_src), (
        f'shared.config_models loaded from {shared_file} — expected a path '
        f'under {_shared_src}. Check that the sys.path.insert block in '
        f'conftest.py runs before the guarded imports.'
    )


def test_toplevel_typo_rejected(mock_orch_config):
    """Top-level ``spec_set`` wiring rejects unknown attribute names.

    Guards against a refactor that drops ``spec_set=pydantic_spec(OrchestratorConfig)``
    from the top-level ``MagicMock`` (e.g. switching to plain ``MagicMock()``
    or ``MagicMock(spec=...)``).  The 7 downstream harness consumers only
    assign *valid* top-level fields, so they would pass silently even if
    ``spec_set`` were removed — this test does not.
    """
    with pytest.raises(AttributeError):
        mock_orch_config.projcet_root = 'anything'


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
