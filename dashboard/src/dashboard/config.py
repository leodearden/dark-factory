"""Dashboard configuration — plain dataclass with env var overrides."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

logger = logging.getLogger(__name__)

# Port 8002 is the canonical systemd-managed instance (shared across all projects).
# Ports 8000/8001 were per-project ports retired when the architecture consolidated
# to a single shared server. The singleton lock prevents any legitimate fused-memory
# process from running on those ports, so listing them would only enable silent
# fallback to stale code — a correctness bug, not graceful degradation.
DEFAULT_FUSED_MEMORY_URLS: Final = ('http://localhost:8002',)


def _discover_escalation_urls(roots: list[Path]) -> dict[str, str]:
    """Read each root's orchestrator.yaml to derive its escalation MCP URL.

    Keyed by project basename — the same key the Merge UI uses. Roots whose
    yaml is missing or malformed are skipped (logged at debug level).
    """
    out: dict[str, str] = {}
    for root in roots:
        yaml_path = root / 'orchestrator.yaml'
        if not yaml_path.is_file():
            logger.debug('No orchestrator.yaml at %s', yaml_path)
            continue
        try:
            with yaml_path.open() as f:
                data: Any = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning('Failed to read %s: %s', yaml_path, exc)
            continue
        esc = (data.get('escalation') or {}) if isinstance(data, dict) else {}
        host = esc.get('host', '127.0.0.1')
        port = esc.get('port')
        if not port:
            logger.debug('No escalation.port in %s', yaml_path)
            continue
        out[root.name] = f'http://{host}:{port}/mcp'
    return out


@dataclass
class DashboardConfig:
    """Configuration for the Dark Factory dashboard."""

    host: str = '127.0.0.1'
    port: int = 8080
    project_root: Path = field(default_factory=lambda: Path('/home/leo/src/dark-factory'))
    fused_memory_urls: list[str] = field(default_factory=lambda: list(DEFAULT_FUSED_MEMORY_URLS))
    known_project_roots: list[Path] = field(default_factory=list)
    escalation_urls: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize all Path fields to their canonical (symlink-resolved) forms.

        This invariant ensures that every construction path — direct kwargs,
        from_env(), dataclass.replace(), test fixtures — produces a config
        whose Path fields are already canonicalized.  Consumers never need to
        call .resolve() on config paths.

        Note: Path.resolve() follows all existing symlink segments, but any trailing
        path components that do not yet exist on disk are appended verbatim (absolute
        but not symlink-canonicalized).  If such components are later created as
        symlinks, the stored path will not reflect their targets.
        """
        self.project_root = self.project_root.resolve()
        self.known_project_roots = [p.resolve() for p in self.known_project_roots]
        if not self.escalation_urls:
            self.escalation_urls = _discover_escalation_urls(
                [self.project_root, *self.known_project_roots],
            )

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

    @property
    def burndown_db(self) -> Path:
        return self.project_root / 'data' / 'burndown' / 'burndown.db'

    @property
    def metrics_db(self) -> Path:
        return self.project_root / 'data' / 'burndown' / 'metrics.db'

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
        if (roots := os.environ.get('DASHBOARD_KNOWN_PROJECT_ROOTS')) is not None:
            kwargs['known_project_roots'] = [Path(p.strip()) for p in roots.split(',') if p.strip()]
        return cls(**kwargs)
