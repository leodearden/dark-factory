"""Which escalation server serves each queue a sitting writes to.

The one hazard this exists to prevent: escalation ids (``esc-<task>-<n>``) are
unique only within one queue, so a call naming an id must reach the server
that serves THAT queue. Sent anywhere else, it resolves a different record that
happens to carry the same id.

A project queue is served on the port its root's orchestrator config names, as
the dashboard discovers it. The reconciliation queue is served by fused-memory's
reconciliation escalation server, never by the project's own port.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dashboard.config import discover_root_escalation_url
from escalation.store_identity import QueueKind
from orchestrator.session_registry import normalize_escalations_dir
from shared.mcp_post import mcp_server_base
from shared.project_layout import ESCALATION_QUEUE_RELPATH
from sitting.inventory import QUEUE_TAGS, RECON_QUEUE_TAG, queue_project_root, queue_tag

# Mirrors fused_memory.config.schema.ReconciliationConfig's escalation_host and
# escalation_port defaults; scripts/tests/test_sitting_ports.py pins the two.
RECON_ESCALATION_URL = 'http://127.0.0.1:8103/mcp'

_PROJECT_TAG = QUEUE_TAGS[Path(ESCALATION_QUEUE_RELPATH)]


@dataclass(frozen=True)
class QueuePort:
    queue_dir: str
    kind: QueueKind
    url: str
    project_root: Path | None

    @property
    def base_url(self) -> str:
        return mcp_server_base(self.url)


@dataclass(frozen=True)
class PortUnknown:
    queue_dir: str
    reason: str


def queue_port(queue_dir: str | Path) -> QueuePort | PortUnknown:
    normalized = normalize_escalations_dir(queue_dir)
    tag = queue_tag(normalized)
    if tag == RECON_QUEUE_TAG:
        return QueuePort(normalized, 'reconciliation', RECON_ESCALATION_URL, None)
    root = queue_project_root(normalized)
    if tag != _PROJECT_TAG or root is None:
        return PortUnknown(normalized, 'not a project or reconciliation escalation queue')
    url = discover_root_escalation_url(root)
    if url is None:
        return PortUnknown(
            normalized,
            f'project root {root} names no escalation port in its orchestrator config',
        )
    return QueuePort(normalized, 'project', url, root)
