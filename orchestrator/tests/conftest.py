"""pytest configuration — ensure local src takes precedence over installed package.

Non-fixture helpers (build_usage_gate) live in `_orch_helpers.py` — a
uniquely-named sibling module — so they can be imported from test files
without conflicting with sibling subprojects' conftests under
`sys.modules['conftest']`.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Insert this worktree's src directories at the front of sys.path so that
# `import orchestrator` and `import shared` load the local (possibly modified)
# code rather than whatever editable install the uv workspace has pinned to
# the main tree.
_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SHARED_SRC = Path(__file__).parent.parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))
_ESCALATION_SRC = Path(__file__).parent.parent.parent / "escalation" / "src"
if str(_ESCALATION_SRC) not in sys.path:
    sys.path.insert(0, str(_ESCALATION_SRC))
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from _orch_helpers import pydantic_spec  # noqa: E402
from shared.config_models import UsageCapConfig  # noqa: E402

from orchestrator.config import (  # noqa: E402
    EscalationConfig,
    FusedMemoryConfig,
    GitConfig,
    OrchestratorConfig,
    ReviewConfig,
    SandboxConfig,
)


@pytest.fixture(scope="session")
def repo_root() -> Path | None:
    """Walk up from this conftest to find the repo root anchored by a .git entry.

    Scoped to orchestrator/tests.  If this fixture is ever needed by other test
    packages (reify, fused-memory, shared, …) it should be hoisted to a
    top-level conftest.py or a shared test-helpers module rather than duplicated.

    Returns the repo-root Path if found, or None when not running inside a git
    checkout (e.g. a packaged wheel, partial mirror, or isolated test run).

    Works for both normal checkouts (.git directory) and git worktrees (.git
    file) because ``Path.exists()`` matches either.

    Consumers should call ``pytest.skip(...)`` when None is returned rather than
    failing — the absence of a git sentinel means the repo-root-dependent tests
    are not applicable to this environment (e.g. CI running from a sdist).
    Consumers may also ``pytest.fail(...)`` when the sentinel is found but a
    required file within the repo is absent, so a genuinely missing file cannot
    silently hide behind a skip.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / '.git').exists():
            return parent
    return None


@pytest.fixture(autouse=True)
def _isolate_orch_config(monkeypatch, tmp_path):
    """Isolate OrchestratorConfig from the real project tree for every test.

    ORCH_CONFIG_PATH is removed so tests don't inadvertently load the real
    config via that env var.

    project_root isolation (the load-bearing part): with ORCH_CONFIG_PATH unset,
    the settings source (``settings_customise_sources``) falls back to the
    *relative* ``Path('config.yaml')``.  Running ``cd orchestrator && pytest``
    (exactly how the per-subproject test command invokes us) therefore loads the
    tracked ``orchestrator/config.yaml``, whose ``project_root`` resolves to the
    real repo root.  Any test that builds a bare ``OrchestratorConfig()`` and
    drives ``acquire_next`` then writes
    ``<repo>/data/orchestrator/scheduler_state.json`` (and the overrides SQLite
    DB) into the live tree — which the dashboard reads back as real scheduler
    state.  Pin ``project_root`` (via the ``ORCH_`` env prefix) to this test's
    ``tmp_path`` so those writes land in tmp instead.

    Precedence keeps this safe: ``init_settings`` (explicit ``project_root=...``
    kwargs) still win over the env, and ``env_settings`` only overrides
    ``project_root`` — every other field still loads from config.yaml/defaults,
    so tests that depend on config.yaml values (e.g. lock_depth) are unaffected.
    Config-loading tests already use ``tmp_path`` as their project_root, so the
    env value agrees with the YAML they write.
    """
    monkeypatch.delenv("ORCH_CONFIG_PATH", raising=False)
    monkeypatch.setenv("ORCH_PROJECT_ROOT", str(tmp_path))


@pytest.fixture(autouse=True)
def _no_dry_run_unblock(monkeypatch, request):
    """Replace ``orchestrator.workflow.run_dry_run_unblock`` with an async no-op.

    The real hook fires the Claude CLI fire-and-forget; in the test suite that
    causes pytest-timeout hangs (~60 s × ~8 tests) whenever ``_mark_blocked``
    runs.  Tests that genuinely need the real binding (e.g.
    ``test_workflow_dry_run_hook.py`` stacks its own ``patch(...)`` on top)
    can opt out with the ``exercise_dry_run_unblock`` marker.
    """
    if request.node.get_closest_marker('exercise_dry_run_unblock'):
        return

    async def _noop(**_):
        return None

    monkeypatch.setattr('orchestrator.workflow.run_dry_run_unblock', _noop)


@pytest.fixture(autouse=True)
def _restore_sandbox_backend():
    """Snapshot and restore sandbox_dispatch._preferred around every test.

    Mirrors the worktree-root conftest's fixture so that running this
    subproject in isolation (e.g. ``cd orchestrator && uv run pytest tests/``,
    which is exactly how Fix 2's per-subproject test_command invokes us)
    still restores backend state between tests.
    """
    from orchestrator.agents import sandbox_dispatch
    saved = sandbox_dispatch.get_backend()
    yield
    sandbox_dispatch.set_backend(saved)


@pytest.fixture
def mock_orch_config(tmp_path: Path) -> MagicMock:
    """Return a MagicMock OrchestratorConfig with the standard harness defaults pre-applied.

    Defaults applied:
      - ``git`` = real ``GitConfig`` with main/task/origin/.worktrees
      - ``project_root`` = ``tmp_path``
      - ``usage_cap.enabled`` = False
      - ``review.enabled`` = False
      - ``sandbox.backend`` = 'auto'
      - ``fused_memory`` = pre-created sub-section mock (no default value)
      - ``escalation`` = pre-created sub-section mock (no default value)
      - ``overrides_db_path`` = ``tmp_path / 'overrides.db'``

    The top-level mock and each sub-section (usage_cap, review, sandbox,
    fused_memory, escalation) are spec_set'd against their pydantic model's
    fields so typos raise AttributeError on both get and set.

    Apply test-specific overrides directly on the returned object.

    Gotcha — pydantic_spec hides BaseModel methods
    -----------------------------------------------
    ``pydantic_spec`` (see ``_orch_helpers.py``) intentionally exposes
    ``model.model_fields`` names AND user-defined ``@property`` descriptors
    (e.g. ``overrides_db_path``) to ``MagicMock(spec_set=...)``.  BaseModel
    methods — ``model_dump``, ``model_validate``, ``model_copy``, etc. — are
    NOT in the proxy class, so ``spec_set`` rejects them on both *read* and
    *write*::

        mock_orch_config.model_dump()           # raises AttributeError
        mock_orch_config.model_dump = MagicMock(...)  # also raises AttributeError

    If a test genuinely needs BaseModel API access, use a real
    ``OrchestratorConfig`` (or the relevant sub-section model) instance
    instead of this fixture.  See ``_orch_helpers.pydantic_spec`` for the
    underlying rationale.
    """
    config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
    config.git = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )
    config.project_root = tmp_path
    config.usage_cap = MagicMock(spec_set=pydantic_spec(UsageCapConfig))
    config.usage_cap.enabled = False
    config.review = MagicMock(spec_set=pydantic_spec(ReviewConfig))
    config.review.enabled = False
    config.sandbox = MagicMock(spec_set=pydantic_spec(SandboxConfig))
    config.sandbox.backend = 'auto'
    config.fused_memory = MagicMock(spec_set=pydantic_spec(FusedMemoryConfig))
    config.escalation = MagicMock(spec_set=pydantic_spec(EscalationConfig))
    # Real numeric defaults for fields read by harness lifecycle loops —
    # MagicMock values would crash ``asyncio.sleep(<MagicMock>)``.
    config.orphan_l0_check_interval_secs = 60.0
    config.orphan_l0_reaper_enabled = False
    config.orphan_l0_timeout_secs = 600.0
    config.terminal_status_watcher_enabled = False
    config.terminal_status_poll_interval_secs = 30.0
    # Stranded-in-progress reconcile sweep — disabled by default in tests so
    # the background loop doesn't spin up under fixtures that don't expect
    # an additional asyncio.Task to manage.
    config.stranded_reconcile_enabled = False
    config.stranded_reconcile_interval_secs = 900.0
    # Real Path so OverrideStore.__init__ can call .parent.mkdir() and
    # sqlite3.connect(str(...)) without crashing — config.overrides_db_path
    # is a @property on OrchestratorConfig (see config.py) and Harness wires
    # OverrideStore.from_config(config) at construction (task 1313).
    config.overrides_db_path = tmp_path / 'overrides.db'
    return config
