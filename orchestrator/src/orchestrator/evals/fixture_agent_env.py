"""The agent env a corpus cell's agents run with: their fixture project's own.

A corpus cell runs under the campaign's base config, but its worktree is cut
from the fixture's project, whose production agents run with that project's
``role_env_overrides`` (reify's cache redirects keep cargo, npm and XDG writes
inside the shared sandbox write set). So a cell takes the fixture project's
``role_env_overrides`` in place of the base's, whole: the two are not merged,
and every role the project names takes its entry, unsandboxed roles included.

``workflow.py::TaskWorkflow._build_agent_env`` merges ``role_env_overrides``
after ``env_overrides``, so a key both set would let the project's value
outrank the candidate's endpoint env with only a warning. That is refused here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from orchestrator.config import OrchestratorConfig, project_role_env_overrides


def fixture_role_env_overrides(
    base_config: OrchestratorConfig, project_root: Path, candidate_env: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """The role_env_overrides of the project at *project_root*; the base's own when it is that project.

    Raises ValueError when any role's entry sets a key *candidate_env* also sets.
    """
    overrides = (
        base_config.role_env_overrides
        if project_root.resolve() == Path(base_config.project_root).resolve()
        else project_role_env_overrides(project_root)
    )
    outranking = {
        role: sorted(env.keys() & candidate_env.keys())
        for role, env in overrides.items()
        if env.keys() & candidate_env.keys()
    }
    if outranking:
        raise ValueError(
            f'the role_env_overrides of the fixture project at {project_root} set '
            f'keys the candidate env_overrides also sets ({outranking}, role: keys); '
            f'role_env_overrides is merged last, so they would outrank the candidate '
            f'under test',
        )
    return overrides
