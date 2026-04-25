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
def _clear_orch_config_path(monkeypatch):
    """Remove ORCH_CONFIG_PATH so tests don't inadvertently load the real config."""
    monkeypatch.delenv("ORCH_CONFIG_PATH", raising=False)


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

    The top-level mock and each sub-section (usage_cap, review, sandbox,
    fused_memory, escalation) are spec_set'd against their pydantic model's
    fields so typos raise AttributeError on both get and set.

    Apply test-specific overrides directly on the returned object.

    Gotcha — pydantic_spec hides BaseModel methods
    -----------------------------------------------
    ``pydantic_spec`` (see ``_orch_helpers.py``) intentionally exposes *only*
    ``model.model_fields`` names to ``MagicMock(spec_set=...)``.  BaseModel
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
    return config
