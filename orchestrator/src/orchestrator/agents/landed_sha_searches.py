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


_INTRO = """\
2. Check whether this task's branch has already landed on main. Every sha you record in \
`done_provenance` comes from one of the task-scoped searches below, never from main's HEAD or an \
eyeballed `git log main` listing. `skills/_shared/deriving-landed-sha.md` is the full ladder; these \
commands are its searches, rendered for this task from the merge lane's own rules. Run them from \
this task's worktree."""

_ANCESTRY_READING = (
    'rc=0: the branch tip is on main. rc=1: it is not. rc=128: the ref is gone, '
    'the usual state after a landing.'
)

_CITATION_READING = """\
Walk the rows most-recent-first and take the first whose **subject** cites the task: `--grep` \
also matches bodies. Among the rows that qualify, prefer one whose diff touches this task's \
declared `metadata.files`. Stamp that citing sha and record the branch tip in `note` (e.g. \
"landing confirmed by task citation on main; no separate merge commit (branch tip <tip-sha>)"). \
Never stamp the branch's own tip: a branch that never advanced has main's OLD base as its tip and \
passes ancestry trivially. No subject-citing row means a phantom branch: stamp nothing and report \
it as not landed. `escalation/src/escalation/git_authority.py::found_on_main_response` resolves the \
same question the same way."""

_CITATION_OPTED_OUT = """\
   - **Citation gate** (ancestry rc=0 with no qualifying marker): this project sets \
`git.commit_citation_pattern: ""`, so the gate cannot be evaluated. Stamp nothing on this arm and \
report it as un-evaluable, not as not landed."""

_LADDER_ONLY = """\
   - Every other outcome (ancestry rc=1, a `coalesce-*` train, a rebased solo landing) is decided \
by the ladder, not here."""

_DONE_PROVENANCE = """\
   - Once the landing sha is established, set the task status to `done` via fused-memory's \
`set_task_status` tool. **`done_provenance` must always include `kind`**: pass \
`done_provenance={"kind": "found_on_main", "commit": "<landing-sha-on-main>", "note": \
"<one-sentence explanation>"}` (both `commit` and `note` are required for this kind; there is no \
commit-less fallback), e.g. note "covered by sibling task" or "already merged prior to this \
session". If the landing commit came from this session calling `merge_request`, use \
`done_provenance={"kind": "merged", "commit": "<merge-sha>"}` instead. Then call `resolve_issue` \
explaining the task was already merged. Do NOT attempt to fix code or re-merge."""


def _command_bullet(heading: str, command: str, reading: str) -> str:
    return f'   - {heading}\n     ```\n     {command}\n     ```\n     {reading}'


def _marker_reading(searches: LandedShaSearches) -> str:
    containment = (
        f'git merge-base --is-ancestor {shlex.quote(searches.branch)} <marker-sha>; '
        'echo "containment rc=$?"'
    )
    return (
        f'It prints the newest commit on main whose subject is exactly '
        f'`{searches.marker_subject}`, or nothing.\n'
        '     - With ancestry rc=128, a printed sha is the landing commit.\n'
        f'     - With the ref still present, the sha counts only if `{containment}` '
        'prints `containment rc=0`.\n'
        '     - An empty result is not a not-landed verdict.'
    )


def _citation_bullet(citation_walk: str | None) -> str:
    if citation_walk is None:
        return _CITATION_OPTED_OUT
    return _command_bullet(
        '**Citation gate** (ancestry rc=0 with no qualifying marker: a fast-forward, or work '
        'covered by a sibling task):',
        citation_walk,
        _CITATION_READING,
    )


def already_merged_step(searches: LandedShaSearches) -> str:
    """The steward's `# Action` item 2, with this task's searches spliced in."""
    return '\n'.join((
        _INTRO,
        _command_bullet('**Ancestry:**', searches.ancestry, _ANCESTRY_READING),
        _command_bullet('**Merge marker:**', searches.marker_search, _marker_reading(searches)),
        _citation_bullet(searches.citation_walk),
        _LADDER_ONLY,
        _DONE_PROVENANCE,
    ))
