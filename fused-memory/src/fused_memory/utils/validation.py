"""Shared validation helpers for fused_memory.

Two styles:
- ``validate_*`` functions return an error dict (for MCP tool handlers that
  must return structured responses to the client rather than raising).
- ``require_*`` functions raise ValueError (for internal code where fail-fast
  is appropriate and the caller controls the error boundary).
"""

from __future__ import annotations

import os


def validate_project_root(project_root: str) -> dict[str, str] | None:
    """Return an error dict if project_root is not a non-empty absolute path, else None."""
    if not project_root or not os.path.isabs(project_root):
        return {
            'error': f'project_root must be a non-empty absolute path, got: {project_root!r}',
            'error_type': 'ValidationError',
        }
    return None


def validate_project_id(project_id: str) -> dict[str, str] | None:
    """Return an error dict if project_id is empty, else None."""
    if not project_id or not project_id.strip():
        return {
            'error': 'project_id is required and must be non-empty',
            'error_type': 'ValidationError',
        }
    return None


def require_project_root(project_root: str) -> None:
    """Raise ValueError if project_root is not a non-empty absolute path."""
    if not project_root or not os.path.isabs(project_root):
        raise ValueError(
            f'project_root must be a non-empty absolute path, got: {project_root!r}'
        )
