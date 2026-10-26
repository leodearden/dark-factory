"""The γ2 optional ``project_root`` guard on escalation MCP tools.

One guard per server applies ``plans/escalation-store-ambiguity-prd.md`` §6.3:
an omitted or ``''`` assertion, or a server with no store identity, proceeds
untouched; anything else is put to the server's verdict, and a refusal comes
back as ``{'error': ...}`` after feeding the per-root
:class:`~escalation.mismatch_streak.MismatchStreak` (§6.4).

The verdict — which roots this store refuses, and in what words — is
``escalation.server::_project_root_refusal``, beside the mismatch wording it
shares with the warm-worktree verbs.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from escalation.store_identity import StoreIdentity

if TYPE_CHECKING:
    from escalation.mismatch_streak import MismatchStreak

Verdict = Callable[[StoreIdentity, str], str | None]


class ProjectRootGuard:
    """Refuses a call that asserts a ``project_root`` this store does not serve."""

    def __init__(
        self, identity: StoreIdentity | None, streak: MismatchStreak, verdict: Verdict,
    ) -> None:
        self._identity = identity
        self._streak = streak
        self._verdict = verdict

    def check(self, project_root: str | None) -> dict[str, str] | None:
        """The refusal to return from the tool, or None to proceed."""
        identity = self._identity
        if identity is None or not project_root:
            return None
        refusal = self._verdict(identity, project_root)
        if refusal is None:
            return None
        self._streak.record_mismatch(identity, project_root, refusal)
        return {'error': refusal}

    async def check_off_loop(self, project_root: str | None) -> dict[str, str] | None:
        """:meth:`check` for an async tool, on a worker thread only when a root is asserted.

        Resolving the path and filing a storm record must stay off the
        orchestrator loop this server shares; an omitted assertion, the
        fleet-agent path, costs no thread hop.
        """
        if self._identity is None or not project_root:
            return None
        return await asyncio.to_thread(self.check, project_root)
