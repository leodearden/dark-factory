"""The steward's landed-sha searches, rendered for one task from the merge lane's own rules.

`orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_steward_initial_prompt`
splices the step built here into its `# Action` list. Each command is a
`skills/_shared/deriving-landed-sha.md` search with this task's branch, merge
subject and citation regex filled in from the same sources production reads:
the marker search mirrors
`orchestrator/src/orchestrator/git_ops.py::GitOps.find_merge_marker`, and the
citation walk mirrors `GitOps.find_task_citation_commit`.
`orchestrator/tests/test_landed_sha_searches.py` runs each command in real git
and holds it in agreement with those methods.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass

from shared.branch_names import canonical_queued_branch_name
from shared.merge_subject import format_merge_subject

from orchestrator.config import GitConfig
from orchestrator.git_ops import task_citation_pattern


@dataclass(frozen=True)
class LandedShaSearches:
    branch: str
    marker_subject: str
    ancestry: str
    marker_search: str
    citation_walk: str | None

    @classmethod
    def for_task(cls, task_id: str, git: GitConfig) -> LandedShaSearches:
        branch = canonical_queued_branch_name(task_id, git.branch_prefix)
        subject = format_merge_subject(branch, git.main_branch)
        main = shlex.quote(git.main_branch)
        quoted_branch = shlex.quote(branch)
        pattern = task_citation_pattern(task_id, git.commit_citation_pattern)
        return cls(
            branch=branch,
            marker_subject=subject,
            ancestry=f'git merge-base --is-ancestor {quoted_branch} {main}; echo "ancestry rc=$?"',
            marker_search=(
                f"git log {main} --fixed-strings --grep={quoted_branch} --format='%H%x09%s'"
                f" | awk -F'\\t' -v s={shlex.quote(subject)} '$2==s && !seen {{print $1; seen=1}}'"
            ),
            citation_walk=(
                None
                if pattern is None
                else f"git log {main} --extended-regexp --format='%H %s' --grep={shlex.quote(pattern)}"
            ),
        )
