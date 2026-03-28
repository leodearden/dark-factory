"""Shared validation helpers for fused_memory.

Two styles:
- ``validate_*`` functions return an error dict (for MCP tool handlers that
  must return structured responses to the client rather than raising).
- ``require_*`` functions raise ValueError (for internal code where fail-fast
  is appropriate and the caller controls the error boundary).
"""

from __future__ import annotations

import os
import re

_RUN_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')


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


def validate_run_id(run_id: str) -> dict[str, str] | None:
    """Return an error dict if run_id is empty or contains unsafe characters, else None.

    Accepted characters: ASCII letters, digits, hyphens, underscores.
    This allowlist blocks all prompt-injection vectors (newlines, quotes, backticks,
    braces, semicolons) while remaining forward-compatible with UUID4 and similar
    safe identifier formats.
    """
    if not run_id or not run_id.strip():
        return {
            'error': 'run_id is required and must be non-empty',
            'error_type': 'ValidationError',
        }
    if not _RUN_ID_PATTERN.match(run_id):
        return {
            'error': (
                f'run_id contains invalid characters: {run_id!r}. '
                'Only ASCII letters, digits, hyphens, and underscores are allowed.'
            ),
            'error_type': 'ValidationError',
        }
    return None


def require_project_root(project_root: str) -> None:
    """Raise ValueError if project_root is not a non-empty absolute path."""
    if err := validate_project_root(project_root):
        raise ValueError(err['error'])
