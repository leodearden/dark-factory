"""Contract tests for ``conftest.py`` correctness.

This module guards five invariants that would silently regress under refactoring
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

4. **``@property`` descriptor exposure** — ``pydantic_spec`` must expose
   user-defined ``@property`` descriptors (e.g. ``overrides_db_path``) in
   the proxy class so ``spec_set`` accepts both read and write.
   BaseModel-inherited properties (``model_extra``, ``model_fields_set``, …)
   and BaseModel methods (``model_dump``, …) must remain excluded.

5. **``mock_orch_config.overrides_db_path`` is a real ``Path``** — the
   fixture must set a concrete ``Path`` under ``tmp_path`` for
   ``overrides_db_path`` so ``OverrideStore.from_config(config)`` can call
   ``.parent.mkdir()`` and ``sqlite3.connect(str(...))`` without crashing.

Tests of plain attribute defaults (e.g. ``mock.usage_cap.enabled is False``)
are deliberately omitted — they would just duplicate literals from
``conftest.py`` two lines away.
"""

from pathlib import Path
from unittest.mock import MagicMock

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


def test_pydantic_spec_exposes_user_property_descriptors():
    """pydantic_spec exposes @property descriptors declared on the user's pydantic model.

    Regression pin for the bug introduced by task 1313: ``Harness.__init__:218``
    calls ``OverrideStore.from_config(config)``, which dereferences
    ``config.overrides_db_path`` — a ``@property`` on ``OrchestratorConfig``
    (config.py:741).  Without this fix, ``MagicMock(spec_set=pydantic_spec(
    OrchestratorConfig))`` rejects both read and write of ``overrides_db_path``
    with ``AttributeError``.

    This test FAILS on pre-fix pydantic_spec (model_fields only).
    """
    from _orch_helpers import pydantic_spec

    from orchestrator.config import OrchestratorConfig

    spec = pydantic_spec(OrchestratorConfig)
    m = MagicMock(spec_set=spec)
    # overrides_db_path is a @property on OrchestratorConfig — not in model_fields.
    _ = m.overrides_db_path            # read must not raise AttributeError
    m.overrides_db_path = Path('/x')   # write must not raise AttributeError


def test_pydantic_spec_excludes_basemodel_inherited_members():
    """pydantic_spec still rejects BaseModel methods and inherited properties.

    Preserves the invariant established by task 1064: the proxy class created
    by pydantic_spec must NOT expose BaseModel API surface (model_dump,
    model_validate, model_extra, model_fields_set, …).  Exposing those would
    let tests write ``mock.model_dump = ...`` without error, silently hiding
    bugs in consumers that call real pydantic methods.

    This test PASSES on current (pre-fix) pydantic_spec because the spec proxy
    only contains model_fields names — none of which are BaseModel API.  After
    step-2's fix (broader @property enumeration with BaseModel-inherited
    filtering), it must continue to pass.
    """
    from _orch_helpers import pydantic_spec

    from orchestrator.config import OrchestratorConfig

    m = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    with pytest.raises(AttributeError):
        _ = m.model_dump            # BaseModel method
    with pytest.raises(AttributeError):
        _ = m.model_extra           # BaseModel @property
    with pytest.raises(AttributeError):
        _ = m.model_fields_set      # BaseModel @property


def test_mock_orch_config_overrides_db_path_default(mock_orch_config, tmp_path):
    """mock_orch_config seeds overrides_db_path with a real Path under tmp_path.

    Guards the fixture-level cleanup (step-4): ``Harness.__init__`` calls
    ``OverrideStore.from_config(config)`` unconditionally, which does
    ``self.db_path.parent.mkdir(...)`` and ``sqlite3.connect(str(self.db_path))``.
    A child ``MagicMock`` value for ``overrides_db_path`` would let ``.parent``
    and ``.mkdir()`` silently no-op but then cause ``sqlite3.connect`` to open a
    file named ``'<MagicMock …>'`` under cwd, leaking filesystem state in CI.

    This test FAILS on pre-fix conftest.py (no default set for overrides_db_path).
    """
    db_path = mock_orch_config.overrides_db_path
    assert isinstance(db_path, Path), (
        f'expected overrides_db_path to be a Path, got {type(db_path).__name__!r}'
    )
    assert db_path.is_relative_to(tmp_path), (
        f'expected overrides_db_path under tmp_path={tmp_path}, got {db_path}'
    )
    assert db_path.suffix == '.db', (
        f'expected overrides_db_path to have .db suffix, got {db_path.name!r}'
    )
