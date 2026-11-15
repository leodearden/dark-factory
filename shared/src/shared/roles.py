"""The config-role vocabulary, the reviewer collapse and the per-role settings row.

``RoleName`` names every role that has a row of settings in a service's config.
A dispatched agent's identity is not always a config role: ``config_role`` is
the one mapping from a dispatch identity to the role whose settings it uses.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

__all__ = [
    'DISPATCHABLE_ROLES',
    'FUSED_MEMORY_ROLES',
    'ORCHESTRATOR_ROLES',
    'REVIEWER_VARIANTS',
    'RoleName',
    'UnknownRole',
    'config_role',
]


class RoleName(StrEnum):
    ARCHITECT = 'architect'
    IMPLEMENTER = 'implementer'
    DEBUGGER = 'debugger'
    REVIEWER = 'reviewer'
    MERGER = 'merger'
    STEWARD = 'steward'
    TRIAGE = 'triage'
    MODULE_TAGGER = 'module_tagger'
    DEEP_REVIEWER = 'deep_reviewer'
    JUDGE = 'judge'
    SIMPLE_TASK = 'simple_task'
    UNBLOCK_AUTO = 'unblock_auto'
    WATCHER = 'watcher'
    MEMORY_LLM = 'memory_llm'
    RECON_AGENT = 'recon_agent'
    RECON_JUDGE = 'recon_judge'
    RECON_STAGE_STANDARD = 'recon_stage_standard'
    RECON_STAGE_BACKLOG = 'recon_stage_backlog'
    CURATOR = 'curator'
    PATH_SCOPE_ADJUDICATOR = 'path_scope_adjudicator'
    WRITE_TRIAGE_JUDGE = 'write_triage_judge'
    LINK_ADJUDICATOR = 'link_adjudicator'


ORCHESTRATOR_ROLES: Final[frozenset[RoleName]] = frozenset(
    {
        RoleName.ARCHITECT,
        RoleName.IMPLEMENTER,
        RoleName.DEBUGGER,
        RoleName.REVIEWER,
        RoleName.MERGER,
        RoleName.STEWARD,
        RoleName.TRIAGE,
        RoleName.MODULE_TAGGER,
        RoleName.DEEP_REVIEWER,
        RoleName.JUDGE,
        RoleName.SIMPLE_TASK,
        RoleName.UNBLOCK_AUTO,
        RoleName.WATCHER,
    }
)
DISPATCHABLE_ROLES: Final[frozenset[RoleName]] = ORCHESTRATOR_ROLES - {
    RoleName.UNBLOCK_AUTO,
    RoleName.WATCHER,
}
FUSED_MEMORY_ROLES: Final[frozenset[RoleName]] = frozenset(
    {
        RoleName.MEMORY_LLM,
        RoleName.RECON_AGENT,
        RoleName.RECON_JUDGE,
        RoleName.RECON_STAGE_STANDARD,
        RoleName.RECON_STAGE_BACKLOG,
        RoleName.CURATOR,
        RoleName.PATH_SCOPE_ADJUDICATOR,
        RoleName.WRITE_TRIAGE_JUDGE,
        RoleName.LINK_ADJUDICATOR,
    }
)

REVIEWER_VARIANTS: Final[frozenset[str]] = frozenset({'reviewer_comprehensive'})

_REVIEWER_FAMILY_PREFIX: Final = 'reviewer_'


class UnknownRole(ValueError):
    """A name is neither a config role nor a member of the reviewer family.

    A ``ValueError`` so that a pydantic validator reports it as a
    ``ValidationError`` carrying this message.

    Attributes:
        value: The offending value, verbatim.
        allowed: The names that would have been accepted.
    """

    def __init__(self, value: object, *, allowed: tuple[str, ...]) -> None:
        self.value = value
        self.allowed = allowed
        super().__init__(f'unknown role {value!r}; known roles: {", ".join(allowed)}')


def config_role(dispatch_name: str) -> RoleName:
    """Map a dispatch identity to its config role."""
    if (
        isinstance(dispatch_name, str)
        and dispatch_name.startswith(_REVIEWER_FAMILY_PREFIX)
        and dispatch_name != _REVIEWER_FAMILY_PREFIX
    ):
        return RoleName.REVIEWER
    try:
        return RoleName(dispatch_name)
    except ValueError:
        allowed = (*(r.value for r in RoleName), f'{_REVIEWER_FAMILY_PREFIX}<name>')
        raise UnknownRole(dispatch_name, allowed=allowed) from None
