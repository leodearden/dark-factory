"""Root conftest to add all subproject src dirs to sys.path.

Also pre-imports the subproject packages so pytest's importlib-mode collection
does not register them as namespace packages (which would shadow the real
package and break `from <subproject>.foo import ...`).
"""
import contextlib
import sys
from pathlib import Path

import pytest

# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path, or the
# subproject directories (orchestrator/, shared/, dashboard/, ...) resolve as
# namespace packages pointing at the project folder instead of src/<pkg>/ —
# precisely the failure this module's docstring exists to prevent. The src dirs
# inserted below therefore always win.
if str(_ROOT := Path(__file__).parent) not in sys.path:
    sys.path.append(str(_ROOT))

from df_pytest_isolation import (
    arm_fleet_dir_isolation,
    arm_leaked_drain_process_guard,
    arm_test_root_isolation,
    reject_unsafe_basetemp,
)

for subproject in [
    'cockpit', 'dashboard', 'escalation', 'fused-memory', 'orchestrator', 'sampler', 'shared',
]:
    _src = _ROOT / subproject / 'src'
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

# Pre-import the real package so pytest's rootdir-relative collection does not
# register the subproject directory (e.g. dashboard/) as a namespace package
# pointing at the project folder instead of its src/<name>/ subtree.
for pkg_name in ['cockpit', 'dashboard', 'escalation', 'orchestrator', 'sampler', 'shared']:
    with contextlib.suppress(ImportError):
        __import__(pkg_name)
# fused-memory's package is fused_memory (underscore)
with contextlib.suppress(ImportError):
    __import__('fused_memory')


def pytest_configure(config):
    """Refuse a --basetemp aimed inside a live task worktree (esc-3072-3), then arm the defences.

    df_pytest_isolation.py's module docstring says what each armed defence guards against.
    """
    reject_unsafe_basetemp(config)
    arm_test_root_isolation(config)
    arm_fleet_dir_isolation(config)
    arm_leaked_drain_process_guard(config)


@pytest.fixture(autouse=True)
def _restore_sandbox_backend():
    """Snapshot and restore sandbox_dispatch._preferred around every test.

    Defence-in-depth for cross-subproject pytest collection: tests that
    intentionally call set_backend() without try/finally would otherwise
    leak state into the next test. set_backend's input validator already
    blocks direct attribute corruption (TypeError/ValueError); this fixture
    covers the remaining "forgot to restore" path.
    """
    try:
        from orchestrator.agents import sandbox_dispatch
    except Exception:
        yield
        return
    saved = sandbox_dispatch.get_backend()
    yield
    sandbox_dispatch.set_backend(saved)


@pytest.fixture(autouse=True)
def _isolate_verify_host_roster(monkeypatch):
    """Root-run tests reach the real load_config through scripts/; none may read the operator's roster file.

    The name is orchestrator/src/orchestrator/verify_host_roster_file.py::ROSTER_FILE_ENV,
    spelled out because the orchestrator import above is optional.
    """
    monkeypatch.setenv('ORCH_VERIFY_HOSTS_FILE', '/nonexistent/df-verify-hosts.yaml')
