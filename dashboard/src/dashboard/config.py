"""Dashboard configuration — plain dataclass with env var overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_FUSED_MEMORY_URLS = [
    'http://localhost:8000',
    'http://localhost:8001',
    'http://localhost:8002',
]


@dataclass
class DashboardConfig:
    """Configuration for the Dark Factory dashboard."""

    host: str = '127.0.0.1'
    port: int = 8080
    project_root: Path = field(default_factory=lambda: Path('/home/leo/src/dark-factory'))
    fused_memory_urls: list[str] = field(default_factory=lambda: list(_DEFAULT_FUSED_MEMORY_URLS))
    fused_memory_project_id: str = 'dark_factory'

    @property
    def reconciliation_db(self) -> Path:
        return self.project_root / 'fused-memory' / 'data' / 'reconciliation' / 'reconciliation.db'

    @property
    def write_queue_db(self) -> Path:
        return self.project_root / 'fused-memory' / 'data' / 'queue' / 'write_queue.db'

    @property
    def tasks_json(self) -> Path:
        return self.project_root / '.taskmaster' / 'tasks' / 'tasks.json'

    @property
    def worktrees_dir(self) -> Path:
        return self.project_root / '.worktrees'

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
        if (pid := os.environ.get('DASHBOARD_FUSED_MEMORY_PROJECT_ID')) is not None:
            kwargs['fused_memory_project_id'] = pid
        return cls(**kwargs)
