"""The HONEST REPLAY FRAME appended to an eval architect's briefing (task 4844).

EVAL-ONLY: an eval replays a historical task in a worktree that shares the live
repository's refs and sits beside live MCP servers, so the architect can see
work that landed after the task's base commit. The frame tells it which
evidence is in frame. Decision, rationale and known gaps: docs/eval-replay-frame.md.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.agents.briefing import BriefingAssembler
from orchestrator.config import OrchestratorConfig

# Bump whenever the block text changes materially, so one id means one briefing.
REPLAY_FRAME_ID = 'honest-frame-v1'


def _require_base_commit(base_commit: object) -> str:
    if not isinstance(base_commit, str) or not base_commit.strip():
        raise ValueError(
            f'replay frame needs a non-empty base commit sha, got {base_commit!r}'
        )
    return base_commit.strip()


def build_replay_frame_block(base_commit: str) -> str:
    sha = _require_base_commit(base_commit)
    return f"""
# Historical Replay — Plan As Of Base Commit `{sha}`

This is an evaluation replay of a historical task, not a live dispatch. HEAD is
detached at `{sha}`, and `{sha}` stands in for `main` at the moment this task
was dispatched. Wherever the role instructions say "main", read "HEAD".

The repository's refs and objects (`main`, other branches, tags, the reflog,
anything `git log --all` shows) and the live MCP servers (task store, memory,
escalation queue) reflect the PRESENT, which is later than `{sha}`. Anything
not reachable from HEAD, and any task status, done-provenance, memory or
escalation describing later work, is post-base information. It is not evidence
about this task's premises.

To stay in frame:
- Wherever the role instructions say `git log --all`, use `git log HEAD -- <path>`.
- Before citing a commit, check it is in frame with
  `git merge-base --is-ancestor <commit> HEAD`.
- Read the files in this worktree; they are the state of the code at `{sha}`.

The decline exits remain the correct behaviour when the IN-FRAME evidence
supports them: the work is already present at HEAD, a premise is false at HEAD,
or a dependency is absent at HEAD and a sibling task is expected to deliver it.
Take the exit exactly as the role contract says. Do NOT ground a decline in
out-of-frame evidence.

Otherwise, plan the task as if `main` were at `{sha}`.
"""


class ReplayFramedBriefingAssembler(BriefingAssembler):
    """An eval-only BriefingAssembler whose architect prompt ends with the replay frame.

    Production never constructs it; see docs/eval-replay-frame.md.
    """

    def __init__(self, config: OrchestratorConfig, *, base_commit: str):
        super().__init__(config)
        self._base_commit = _require_base_commit(base_commit)

    async def build_architect_prompt(
        self,
        task: dict,
        worktree: Path | None = None,
        context: str | None = None,
        *,
        include_prior_proposals: bool = False,
        committed_work: list[dict] | None = None,
    ) -> str:
        return await super().build_architect_prompt(
            task, worktree, context,
            include_prior_proposals=include_prior_proposals,
            committed_work=committed_work,
        ) + build_replay_frame_block(self._base_commit)
