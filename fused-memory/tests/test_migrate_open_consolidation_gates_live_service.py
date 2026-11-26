"""The service scripts/migrate_open_consolidation_gates.py builds for itself.

Every case in scripts/tests/test_migrate_open_consolidation_gates.py injects
its memory, so none reaches the branch where ``run`` opens the live service.
That branch decides how the store is opened: NONE on a dry run, FULL under
``--execute``. A dry run opened FULL would run Graphiti's startup repair and
replay the durable queue while reporting that it wrote nothing.

``FusedMemoryConfig`` and ``MemoryService`` are replaced on their home modules,
where the script's deferred imports find them. The project holds no open gate,
so only the opening and the close are exercised.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.config.schema import ConsolidationAutoConfig

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _ROOT / 'scripts'
_PROJECT_ID = 'dark_factory'


def _load_module() -> types.ModuleType:
    """Load the script by path; its flat siblings resolve only with ``scripts/`` on ``sys.path``."""
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))
    return load_script_module(_SCRIPTS / 'migrate_open_consolidation_gates.py')


mod = _load_module()


@pytest.fixture
def project_root(tmp_path, monkeypatch) -> Path:
    """A project whose tasks.db holds no task, with a writable mem0 history dir."""
    mem0_dir = tmp_path / 'mem0'
    mem0_dir.mkdir()
    monkeypatch.setenv('MEM0_DIR', str(mem0_dir))
    root = tmp_path / 'project'
    db = root / '.taskmaster' / 'tasks' / 'tasks.db'
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    try:
        con.execute('CREATE TABLE tasks (tag TEXT, id INTEGER, status TEXT, metadata TEXT)')
        con.commit()
    finally:
        con.close()
    return root


class _LiveService:
    """What the patched classes saw: every config built, every service built, every lifecycle call."""

    def __init__(self, monkeypatch, *, auto_enabled: bool) -> None:
        self.configs: list[Any] = []
        self.calls: list[str] = []
        config = types.SimpleNamespace(
            consolidation_auto=ConsolidationAutoConfig(
                enabled=auto_enabled, enabled_projects=[_PROJECT_ID],
            ),
        )
        live = self

        class RecordingMemory:
            def __init__(self, given: Any) -> None:
                live.configs.append(given)
                self.config = given

            async def initialize(self) -> None:
                live.calls.append('initialize')

            async def initialize_for_reads(self) -> None:
                live.calls.append('initialize_for_reads')

            async def close(self) -> None:
                live.calls.append('close')

            async def get_memories_by_metadata(
                self, project_id: str, filters: dict[str, Any], limit: int | None = None,
            ) -> list[dict[str, Any]]:
                return []

        self.config = config
        monkeypatch.setattr('fused_memory.config.schema.FusedMemoryConfig', lambda: config)
        monkeypatch.setattr('fused_memory.services.memory_service.MemoryService', RecordingMemory)


def _run(project_root: Path, mode: str) -> int:
    argv = ['--project-root', str(project_root), '--project-id', _PROJECT_ID, '--json', mode]
    return asyncio.run(mod.run(mod.build_parser().parse_args(argv)))


class TestTheSelfBuiltServiceOpensAsTheModeNeeds:
    @pytest.mark.parametrize(
        ('mode', 'lifecycle'),
        [('--dry-run', ['close']), ('--execute', ['initialize', 'close'])],
    )
    def test_the_service_opens_for_its_mode_and_is_closed_once(
        self, project_root, monkeypatch, capsys, mode, lifecycle,
    ):
        live = _LiveService(monkeypatch, auto_enabled=True)

        rc = _run(project_root, mode)

        assert rc == mod.ExitCode.COMPLETED, capsys.readouterr().err
        assert live.configs == [live.config]
        assert live.calls == lifecycle

    def test_the_kill_switch_refuses_before_any_service_exists(
        self, project_root, monkeypatch,
    ):
        live = _LiveService(monkeypatch, auto_enabled=False)

        rc = _run(project_root, '--execute')

        assert rc == mod.ExitCode.COULD_NOT_RUN
        assert live.configs == []
        assert live.calls == []
