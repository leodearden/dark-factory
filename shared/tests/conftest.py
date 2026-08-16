"""pytest configuration — ensure local src takes precedence over installed package."""
import importlib
import sys
from pathlib import Path
from types import ModuleType

# Insert this worktree's src directory at the front of sys.path so that
# `import shared` loads the local (possibly modified) code rather than
# whatever editable install the shared .venv has pinned to the main tree.
_SRC = Path(__file__).parent.parent / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

# Suite-wide git isolation (task 3355, incident esc-3072-3).  The verify lane
# runs `cd shared && uv run pytest tests/`, which makes rootdir the SUBPROJECT —
# the repo-root conftest.py is never loaded, so each test-root conftest wires
# the defence itself.  APPEND the repo root, never insert(0, ...): at sys.path[0]
# it would make the subproject directories resolve as namespace packages
# pointing at the project folder instead of src/<pkg>/, beating the insert above.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import pytest  # noqa: E402
from _cross_account_evidence import PAIR_OVERRIDE_VAR  # noqa: E402
from _oauth_accounts import ALL_TOKEN_LETTERS  # noqa: E402
from df_pytest_isolation import (  # noqa: E402
    _df_deploy_clocks_unwritten,  # noqa: F401  — the binding IS the wiring
    _df_git_ceiling_at_basetemp,  # noqa: F401  — the binding IS the wiring
    reject_unsafe_basetemp,
)


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3)."""
    reject_unsafe_basetemp(config)


@pytest.fixture
def reload_module_under_env(monkeypatch):
    """Reload a consumer module under a controlled fake environ.

    Yields ``reload(module_name, **tokens)`` -> the reloaded module.  Every
    ``CLAUDE_OAUTH_TOKEN_*`` letter and the pair-override var are cleared first,
    so the machine's real ``.env`` (which has A..G all set) cannot leak in and
    make these assertions environment-dependent.  Neither the letter set nor the
    override var name is restated here: both are imported from the module that
    owns them, so this fixture cannot become the next drifting copy.

    That clearing happens BEFORE ``import_module``, not merely before ``reload``,
    and the order is load-bearing: even the module's FIRST — cold — import
    already runs under the fake environ, so no import-time side effect (a
    ``warnings.warn``, a module-level pair resolution) can escape into a caller's
    ``catch_warnings`` / ``pytest.warns`` region.  This matters because ``_reload``
    is called from inside the test body, where such a region is already open —
    unlike the pre-consolidation ``reload_integration_module``, which imported
    during fixture SETUP and so dodged the hazard only by accident.

    Shared by every test module in this directory (visible via ``conftest.py``):
    ``test_usage_gate``, ``test_cli_invoke_integration`` and
    ``test_startup_completion_fixtures`` via ``test_oauth_accounts.py``, and
    ``test_cli_invoke_integration`` via ``test_cross_account_evidence.py``.

    Reloading mutates the module object other collected items may hold, so on
    teardown the real environment is restored (``monkeypatch.undo()`` FIRST,
    since this fixture finalises BEFORE monkeypatch's own teardown) and every
    module touched is reloaded once more under it — no reload side effect
    escapes to other collected items.
    """
    reloaded: list[ModuleType] = []

    def _reload(module_name: str, **tokens: str) -> ModuleType:
        for ch in ALL_TOKEN_LETTERS:
            monkeypatch.delenv(f'CLAUDE_OAUTH_TOKEN_{ch}', raising=False)
        monkeypatch.delenv(PAIR_OVERRIDE_VAR, raising=False)
        for name, value in tokens.items():
            monkeypatch.setenv(name, value)
        module = importlib.import_module(module_name)
        if module not in reloaded:
            reloaded.append(module)
        return importlib.reload(module)

    try:
        yield _reload
    finally:
        monkeypatch.undo()
        for module in reloaded:
            importlib.reload(module)
