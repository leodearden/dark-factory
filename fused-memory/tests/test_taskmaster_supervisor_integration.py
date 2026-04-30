"""Integration tests for the Taskmaster supervisor task pattern.

Spawns a *real* Taskmaster Node child via :class:`TaskmasterBackend`, kills
it, and verifies the supervisor reconnects without the parent process dying.
This is the kill-drill that proves the supervisor isolates an inner
transport failure — the unit tests in ``test_taskmaster_supervisor.py``
exercise the loop logic with mocks; this file exercises the real cascade
trigger.

Skipped when ``node`` or the built ``taskmaster-ai`` MCP server are not
available in the expected location.
"""

import asyncio
import json
import os
import shutil
import signal
from pathlib import Path

import psutil
import pytest

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import TaskmasterConfig


# The supervisor talks to the *real* Taskmaster Node binary built into
# ``taskmaster-ai/dist/mcp-server.js``. The build artifact is part of the
# main checkout (worktrees do not duplicate it), so the test honours the
# ``TASKMASTER_DIR`` env var before falling back to common locations.
def _find_taskmaster_js() -> Path | None:
    candidates: list[Path] = []
    env_dir = os.environ.get('TASKMASTER_DIR')
    if env_dir:
        candidates.append(Path(env_dir).expanduser() / 'dist' / 'mcp-server.js')
    repo_parents = Path(__file__).resolve().parents
    for p in repo_parents:
        candidates.append(p / 'taskmaster-ai' / 'dist' / 'mcp-server.js')
    for c in candidates:
        if c.is_file():
            return c
    return None


_TASKMASTER_JS = _find_taskmaster_js()
_NODE_BIN = shutil.which('node')

requires_taskmaster = pytest.mark.skipif(
    _NODE_BIN is None or _TASKMASTER_JS is None,
    reason='node + taskmaster-ai/dist/mcp-server.js required for kill-drill',
)


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Build a minimal Taskmaster project layout under tmp_path.

    Taskmaster requires ``.taskmaster/tasks/tasks.json`` to exist; without it,
    even ``get_tasks`` returns a tool-level error. We seed an empty master tag.
    """
    tasks_dir = tmp_path / '.taskmaster' / 'tasks'
    tasks_dir.mkdir(parents=True)
    (tasks_dir / 'tasks.json').write_text(
        json.dumps({'master': {'tasks': [], 'metadata': {}}}),
    )
    # Taskmaster also looks for ``.taskmaster/config.json`` to disable AI calls
    # in tests; without it the proxy still answers ``get_tasks`` but logs noise.
    (tmp_path / '.taskmaster' / 'config.json').write_text(
        json.dumps({'global': {}}),
    )
    return tmp_path


@pytest.fixture
def config(project_root: Path) -> TaskmasterConfig:
    return TaskmasterConfig(
        transport='stdio',
        command='node',
        args=[str(_TASKMASTER_JS)],
        project_root=str(project_root),
        cwd=str(project_root),
        tool_mode='all',
    )


def _find_node_child(parent_pid: int) -> psutil.Process | None:
    """Return the Node Taskmaster child of ``parent_pid``, or None."""
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return None
    for child in parent.children(recursive=True):
        try:
            cmdline = ' '.join(child.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if 'mcp-server.js' in cmdline:
            return child
    return None


@requires_taskmaster
@pytest.mark.asyncio
async def test_supervisor_reconnects_after_node_kill(config: TaskmasterConfig) -> None:
    """SIGKILL the Taskmaster Node child; supervisor must reopen the
    session and a subsequent ``call_tool`` must succeed.

    This is the kill-drill that demonstrates the supervisor pattern works
    on the real cascade trigger — anyio fires the cancel scope when the
    stdio read pipe goes EOF after the child dies.
    """
    backend = TaskmasterBackend(
        config,
        reconnect_cooldown_seconds=0.5,
        startup_timeout_seconds=20.0,
        session_ready_timeout_seconds=20.0,
    )
    await backend.start()
    assert backend.connected, 'first session did not come up'
    assert backend.restart_count == 1

    # Sanity probe — the proxy responds to get_tasks.
    result = await backend.call_tool('get_tasks', {'projectRoot': config.project_root})
    assert isinstance(result, dict)

    # Locate and kill the Node child.
    child = _find_node_child(os.getpid())
    assert child is not None, 'could not find Taskmaster Node child'
    child_pid = child.pid
    os.kill(child_pid, signal.SIGKILL)

    # Try a call_tool — this should fail with a transport-dead exception
    # once the supervisor / MCP read loop notices the dead pipe. Some
    # transports detect EOF on the next attempt to write, others surface
    # it asynchronously via the read loop. Either way the error must
    # propagate without crashing the test process.
    failed = False
    deadline = asyncio.get_running_loop().time() + 10.0
    while asyncio.get_running_loop().time() < deadline:
        try:
            await backend.call_tool('get_tasks', {'projectRoot': config.project_root})
        except Exception:
            failed = True
            break
        await asyncio.sleep(0.2)
    assert failed, 'call_tool against killed Node child must surface an error'

    # Wait for the supervisor to detect the cascade and respawn the session.
    deadline = asyncio.get_running_loop().time() + 30.0
    while asyncio.get_running_loop().time() < deadline:
        if backend.restart_count >= 2 and backend.connected:
            break
        await asyncio.sleep(0.5)
    assert backend.restart_count >= 2, (
        f'supervisor did not respawn (restart_count={backend.restart_count})'
    )
    assert backend.connected, 'session not ready after respawn'

    # Verify the new child is *different* from the killed one.
    new_child = _find_node_child(os.getpid())
    assert new_child is not None, 'no Node child after respawn'
    assert new_child.pid != child_pid, 'expected a new Node PID after kill'

    # And calls still work.
    result = await backend.call_tool('get_tasks', {'projectRoot': config.project_root})
    assert isinstance(result, dict)

    await backend.close()
    # The respawned Node child should also be gone.
    assert _find_node_child(os.getpid()) is None
