"""Shared ``MemoryConsolidator`` construction fixtures for ``tests/reconciliation/``.

Two suites need the same mock-backed ``MemoryConsolidator``, for deliberately
different reasons:

1. ``test_stage1.py`` — behavioural payload/section RENDERING tests, the
   factory's original home and still its heaviest consumer.
2. ``test_stage1_payload_section_parity.py`` — the structural builder-parity
   guard (task 4708), which needs a real instance only to drive
   ``_render_required_sections()``.

WHY A MODULE RATHER THAN A CROSS-SUITE IMPORT. The guard used to obtain the
factory with ``from reconciliation.test_stage1 import _make_consolidator``. That
imported a ~2700-line test module and its entire import graph to get one
callable, and bound the guard to a LEADING-UNDERSCORE private symbol whose
owning module was free to rename it at any time — a rename would have broken the
guard at IMPORT time, taking down its whole suite rather than one test. Both
problems are properties of reaching into a test module for a helper, not of the
helper itself, so the helper moved out.

WHY NOT A CONFTEST FIXTURE. These are plain callables taking arguments
(``make_consolidator(project_root=...)``), invoked ~90 times across the two
suites with several different roots and often more than once in a single test.
A pytest fixture would have to be a factory fixture to serve that, which is
strictly more machinery for the same result, and would additionally be
unavailable at module import time — which the parity guard's collection-time
work relies on. It would also land in a ``conftest.py``, which this subproject
avoids for shared helpers (see ``tests/_ast_guard.py``'s note on the
``sys.modules['conftest']`` collision under root-level multi-subproject
collection).

Follows the ``reconciliation/plural_enum_shapes.py`` precedent exactly: a plain
non-test module inside this package, imported as ``from
reconciliation.consolidator_fixtures import make_consolidator``. ``tests/``
carries no ``__init__.py`` while ``tests/reconciliation/`` does, so pytest puts
``tests/`` on ``sys.path`` for modules in BOTH directories and the package
import resolves identically from each. The filename does not start with
``test_``, so pytest never collects this as a suite.

Names here are PUBLIC (no leading underscore) — that is the point of the move:
this module exists to be imported, so its exports must be part of its contract
rather than something callers reach past.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.models.reconciliation import StageId
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.stages.memory_consolidator import MemoryConsolidator


def make_scope(project_id: str, project_root: str) -> ProjectScope:
    """Build a ProjectScope from raw strings — DRYs the many test call sites."""
    return ProjectScope(ProjectId(project_id), ProjectRoot(project_root))


def make_consolidator(project_root: str = '/tmp/test') -> MemoryConsolidator:
    """Build a MemoryConsolidator with mocked deps — mirrors test_stages.py ~L1418.

    NOTE: callers must pass a non-empty absolute ``project_root``. Passing
    ``project_root=''`` (the pre-task-2146 "unset root" sentinel) raises
    ``InputValidationError`` from ``ProjectScope.__post_init__`` at this call
    site — a required ``scope`` can no longer carry a falsy root. The former
    empty-root tests were removed accordingly (task 2146).

    ``filtered_task_tree`` is left at its class default of ``None``, which
    several callers rely on as the "no section applies" state — set it
    explicitly when a section must render.
    """
    config = ReconciliationConfig()
    memory_mock = AsyncMock()
    memory_mock.get_episodes = AsyncMock(return_value=[])
    memory_mock.mem0 = AsyncMock()
    memory_mock.mem0.get_all = AsyncMock(return_value={'results': []})
    memory_mock.get_status = AsyncMock(return_value={})

    stage = MemoryConsolidator(
        StageId.memory_consolidator,
        memory_mock,
        AsyncMock(),  # taskmaster
        AsyncMock(),  # journal
        config,
        scope=make_scope('test_project', project_root),
    )
    stage.episode_limit = 5
    stage.memory_limit = 10
    return stage
