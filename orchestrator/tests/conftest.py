"""pytest configuration — ensure local src takes precedence over installed package."""
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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

from shared.config_models import AccountConfig, UsageCapConfig  # noqa: E402
from shared.usage_gate import AccountState, UsageGate  # noqa: E402

# Cooperative jobserver: block at session start until a slot is free on the
# system-wide FIFO (pytest-jobserver.service).  No-op when PYTEST_JOBSERVER_FIFO
# is unset or the FIFO is absent.
pytest_plugins = ('shared.pytest_jobserver',)


def build_usage_gate(
    account_configs: list[AccountConfig],
    tokens: Sequence[str | None],
    *,
    wait_for_reset: bool = False,
    session_budget_usd: float | None = None,
    probe_interval_secs: int = 300,
    max_probe_interval_secs: int = 1800,
) -> UsageGate:
    """Create a UsageGate with tokens pre-injected (no os.environ lookup).

    Bypasses UsageGate.__init__ via __new__ and sets all private attrs directly,
    then injects AccountState entries from the parallel ``tokens`` list rather
    than reading from environment variables.  This is the canonical pattern for
    constructing test gates — both _make_gate (test_usage_gate.py) and
    _make_reify_gate (test_reify_multi_account.py) delegate to this helper.

    Parameters
    ----------
    account_configs:
        List of AccountConfig instances (same shape as UsageCapConfig.accounts).
    tokens:
        Parallel list of OAuth token strings (or None for default-credential
        accounts).  Must be the same length as ``account_configs``.
    wait_for_reset:
        Forwarded to UsageCapConfig.
    session_budget_usd:
        Forwarded to UsageCapConfig.
    probe_interval_secs:
        Forwarded to UsageCapConfig.
    max_probe_interval_secs:
        Forwarded to UsageCapConfig.

    Raises
    ------
    TypeError
        If ``tokens`` is a bare ``str`` instead of a list/tuple.
    ValueError
        If ``account_configs`` and ``tokens`` have different lengths.
    """
    if isinstance(tokens, str):
        raise TypeError(
            f'tokens must be a list/tuple of str|None, not a bare str; '
            f'got {tokens!r}. Did you forget to wrap it in a list?'
        )
    if len(account_configs) != len(tokens):
        raise ValueError(
            f'account_configs and tokens must have the same length; '
            f'got {len(account_configs)} account(s) and {len(tokens)} token(s)'
        )

    config = UsageCapConfig(
        wait_for_reset=wait_for_reset,
        session_budget_usd=session_budget_usd,
        probe_interval_secs=probe_interval_secs,
        max_probe_interval_secs=max_probe_interval_secs,
        accounts=account_configs,
    )

    gate = UsageGate.__new__(UsageGate)
    gate._config = config
    gate._open = asyncio.Event()
    gate._open.set()
    gate._lock = asyncio.Lock()
    gate._cumulative_cost = 0.0
    gate._paused_reason = ''
    gate._pause_started_at = None
    gate._total_pause_secs = 0.0
    gate._cost_store = None
    gate._project_id = None
    gate._run_id = None
    gate._last_account_name = None
    gate._background_tasks = set()
    gate._probe_config_dir = MagicMock()
    # Mock _run_probe so tests don't spawn real `claude` processes.
    # Tests that need specific probe behaviour can override this attribute.
    gate._run_probe = AsyncMock(return_value=True)
    gate._accounts = [
        AccountState(name=cfg.name, token=tok)
        for cfg, tok in zip(account_configs, tokens, strict=True)
    ]
    return gate


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
