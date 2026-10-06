"""Shared fixtures for script-kind ``metadata.delivered_checks`` entries."""

from __future__ import annotations

import os
from pathlib import Path


def install_delivered_check_script(project_root: Path, rel_path: str, body: str) -> None:
    """Write *body* to *rel_path* under *project_root* and mark it executable.

    Until this runs the script is missing, so the real runner's subprocess
    spawn raises ``FileNotFoundError`` (-> ``DeliveredCheckResult.ERRORED``);
    afterwards the next evaluation runs *body*. Deliberately NOT committed to
    git — the script kind is evaluated against the WORKING CHECKOUT, not the
    committed ``main`` tree (unlike the grep kind; see
    ``orchestrator.delivered_checks``'s module docstring).
    """
    target = project_root / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding='utf-8')
    os.chmod(target, 0o755)


def script_check(name: str, script_rel_path: str, *, timeout_secs: float = 5) -> dict:
    """Build a script-kind ``metadata.delivered_checks`` entry — the same
    shape ``commit_planning`` stamps from a capability-manifest sidecar's
    script capability."""
    return {
        'name': name, 'kind': 'script', 'script': script_rel_path,
        'args': [], 'timeout_secs': timeout_secs,
    }
