"""Remote verify-host construction and the task-path host pool.

PRD ``plans/remote-task-leg-verify-prd.md`` §4.4-4.5, §5.4.  This module sits
below both the merge lane and the task workflow: it must never import
``orchestrator.merge_lane`` or ``orchestrator.workflow``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.verify_runner import RemoteRunner, resolve_local_df_checkout

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig


def build_remote_runners(
    config: OrchestratorConfig,
    cwd: str | Path,
    *,
    quarantine: set[str] | None = None,
) -> list[RemoteRunner]:
    """Build the list of RemoteRunner instances from operator config (Lever C).

    Returns REMOTES ONLY — the LocalRunner trust anchor is prepended by callers
    since it needs call-specific arguments (worktree path, module configs, etc.).

    Filters out disabled runners (enabled=False) and any runner whose name is in
    the quarantine set (in-memory worker-level quarantine from DriftDetector).
    quarantine=None is treated as an empty set (no quarantine).

    _build_verify_runners passes main_branch=config.git.main_branch so the
    remote host receives a freshness push before the merge-sha transport.

    Each runner also receives the INV-2 contract-currency paths (task 2884):
    df_remote_checkout=r.df_checkout_path (per-runner, opt-in; None keeps
    auto-sync OFF / byte-identical) and df_local_checkout=the dispatcher's own
    DF code root, resolved ONCE here (call-invariant) and shared across every
    runner so sync_if_stale can HEAD-compare remote-vs-local at dispatch.
    """
    _local_df = resolve_local_df_checkout()
    return [
        RemoteRunner(
            name=r.name,
            ssh_host=r.ssh_host,
            git_remote=r.git_remote,
            cwd=cwd,
            config_path=r.config_path,
            main_branch=config.git.main_branch,
            df_remote_checkout=r.df_checkout_path,
            df_local_checkout=_local_df,
        )
        for r in config.enabled_verify_runners
        if quarantine is None or r.name not in quarantine
    ]
