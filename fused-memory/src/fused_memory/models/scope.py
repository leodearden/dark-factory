"""Scope model — maps a request to backend-specific identifiers."""

from pathlib import PurePosixPath

from pydantic import BaseModel


def resolve_project_id(
    project_root: str, mapping: dict[str, str] | None = None
) -> str:
    """Derive a logical project_id from a filesystem project_root path.

    Priority:
      1. Explicit mapping dict (if provided and contains project_root).
      2. Derive from basename: lowercase, hyphens -> underscores.

    Examples:
        >>> resolve_project_id('/home/leo/src/dark-factory')
        'dark_factory'
        >>> resolve_project_id('/project/')
        'project'
    """
    if mapping and project_root in mapping:
        return mapping[project_root]
    name = PurePosixPath(project_root.rstrip('/')).name
    return name.lower().replace('-', '_')


class Scope(BaseModel):
    """Per-request scope mapping project/agent/session to backend IDs."""

    project_id: str
    agent_id: str | None = None
    session_id: str | None = None

    @property
    def graphiti_group_id(self) -> str:
        """Graphiti group_id = project_id."""
        return self.project_id

    def mem0_collection_name(self, prefix: str) -> str:
        """Qdrant collection name: {prefix}_{project_id}."""
        return f'{prefix}_{self.project_id}'

    @property
    def mem0_user_id(self) -> str:
        """Mem0 requires at least one of user_id/agent_id/run_id.

        We use project_id as user_id to satisfy this requirement.
        """
        return self.project_id
