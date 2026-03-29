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

# Single shared pattern enforces symmetric allowlist across all identifier types.
# Accepted: ASCII letters, digits, hyphens, underscores.
# Rejected: all prompt-injection vectors (newlines, quotes, backticks, braces, etc.).
_SAFE_IDENTIFIER_PATTERN = re.compile(r'[a-zA-Z0-9_-]+')


class InputValidationError(ValueError):
    """Raised by require_* functions when an input parameter fails validation.

    Subclasses ValueError for backward compatibility — existing ``except ValueError``
    clauses and ``pytest.raises(ValueError)`` tests continue to work unchanged.
    Using a distinct type lets callers distinguish input validation rejections
    from ValueErrors raised by handler logic (e.g. arithmetic errors, parsing
    failures inside reconciliation handlers).
    """


def _validate_identifier(value: str, field_name: str) -> dict[str, str] | None:
    """Shared private helper: validate a single identifier against the safe allowlist.

    Returns an error dict if value is empty/whitespace or contains unsafe characters,
    else returns None.
    """
    if not value or not value.strip():
        return {
            'error': f'{field_name} is required and must be non-empty',
            'error_type': 'ValidationError',
        }
    if not _SAFE_IDENTIFIER_PATTERN.fullmatch(value):
        return {
            'error': (
                f'{field_name} contains invalid characters: {value!r}. '
                'Only ASCII letters, digits, hyphens, and underscores are allowed.'
            ),
            'error_type': 'ValidationError',
        }
    return None


def validate_project_root(project_root: str) -> dict[str, str] | None:
    """Return an error dict if project_root is not a non-empty absolute path, else None."""
    if not project_root or not os.path.isabs(project_root):
        return {
            'error': f'project_root must be a non-empty absolute path, got: {project_root!r}',
            'error_type': 'ValidationError',
        }
    return None


def validate_project_id(project_id: str) -> dict[str, str] | None:
    """Return an error dict if project_id is empty or contains unsafe characters, else None.

    Accepted characters: ASCII letters, digits, hyphens, underscores.
    This allowlist blocks all prompt-injection vectors (newlines, quotes, backticks,
    braces, semicolons) while remaining forward-compatible with common identifier formats.
    """
    return _validate_identifier(project_id, 'project_id')


def validate_run_id(run_id: str) -> dict[str, str] | None:
    """Return an error dict if run_id is empty or contains unsafe characters, else None.

    Accepted characters: ASCII letters, digits, hyphens, underscores.
    This allowlist blocks all prompt-injection vectors (newlines, quotes, backticks,
    braces, semicolons) while remaining forward-compatible with UUID4 and similar
    safe identifier formats.
    """
    return _validate_identifier(run_id, 'run_id')


def require_project_root(project_root: str) -> None:
    """Raise ValueError if project_root is not a non-empty absolute path."""
    if err := validate_project_root(project_root):
        raise ValueError(err['error'])


def require_project_id(project_id: str) -> None:
    """Raise ValueError if project_id is empty or contains unsafe characters."""
    if err := validate_project_id(project_id):
        raise ValueError(err['error'])


def require_run_id(run_id: str) -> None:
    """Raise ValueError if run_id is empty or contains unsafe characters."""
    if err := validate_run_id(run_id):
        raise ValueError(err['error'])
