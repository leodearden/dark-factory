"""Dashboard configuration — plain dataclass with env var overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Port 8002 is the canonical systemd-managed instance (shared across all projects).
# Ports 8000/8001 were per-project ports retired when the architecture consolidated
# to a single shared server. The singleton lock prevents any legitimate fused-memory
# process from running on those ports, so listing them would only enable silent
# fallback to stale code — a correctness bug, not graceful degradation.
DEFAULT_FUSED_MEMORY_URLS: Final = ('http://localhost:8002',)


@dataclass
class DashboardConfig:
    """Configuration for the Dark Factory dashboard."""

    host: str = '127.0.0.1'
    port: int = 8080
    project_root: Path = field(default_factory=lambda: Path('/home/leo/src/dark-factory'))
    fused_memory_urls: list[str] = field(default_factory=lambda: list(DEFAULT_FUSED_MEMORY_URLS))

    @property
    def reconciliation_db(self) -> Path:
        return self.project_root / 'data' / 'reconciliation' / 'reconciliation.db'

    @property
    def write_queue_db(self) -> Path:
        return self.project_root / 'data' / 'queue' / 'write_queue.db'

    @property
    def write_journal_db(self) -> Path:
        return self.project_root / 'data' / 'reconciliation' / 'write_journal.db'

    @property
    def tasks_json(self) -> Path:
        return self.project_root / '.taskmaster' / 'tasks' / 'tasks.json'

    @property
    def worktrees_dir(self) -> Path:
        return self.project_root / '.worktrees'

    @property
    def runs_db(self) -> Path:
        return self.project_root / 'data' / 'orchestrator' / 'runs.db'

    @property
    def escalations_dir(self) -> Path:
        return self.project_root / 'data' / 'escalations'

    @classmethod
    def from_env(cls) -> DashboardConfig:
        """Create config with DASHBOARD_-prefixed env var overrides."""
        kwargs: dict = {}
        if (host := os.environ.get('DASHBOARD_HOST')) is not None:
            kwargs['host'] = host
        if (port := os.environ.get('DASHBOARD_PORT')) is not None:
            kwargs['port'] = int(port)
        if (root := os.environ.get('DASHBOARD_PROJECT_ROOT')) is not None:
            kwargs['project_root'] = Path(root)
        if (urls := os.environ.get('DASHBOARD_FUSED_MEMORY_URLS')) is not None:
            kwargs['fused_memory_urls'] = [u.strip() for u in urls.split(',') if u.strip()]
        return cls(**kwargs)
