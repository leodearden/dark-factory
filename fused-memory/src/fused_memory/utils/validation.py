"""Shared validation helpers for fused_memory.

Two styles:
- ``validate_*`` functions return an error dict (for MCP tool handlers that
  must return structured responses to the client rather than raising).
- ``require_*`` functions raise ValueError (for internal code where fail-fast
  is appropriate and the caller controls the error boundary).
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

# Module-level flag for warn-once behaviour when no registry is configured.
# Task 1143 (read-side) raises UnknownProjectError for unknown project_ids.
# Task 1549 (write-side) uses this flag to warn once that the write-boundary
# registry check is disabled, so operators know writes are ungated.
_empty_registry_warned: bool = False

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


def _safe_repr(value: object, max_len: int = 200) -> str:
    """Return repr(value) truncated to max_len characters.

    If repr(value) exceeds max_len, slices it to max_len characters and appends
    '...(truncated)' so callers can see the value was capped.  Truncation happens
    after calling repr(), so character expansion (e.g. \\n → \\\\n, \\xff → \\\\xff)
    is accounted for before the cap is applied.
    """
    r = repr(value)
    if len(r) > max_len:
        return r[:max_len] + '...(truncated)'
    return r


def _is_plain_int(x: object) -> bool:
    """Return True iff x is a plain int (not a bool subclass).

    Python's bool is a subclass of int, so ``isinstance(True, int)`` is True.
    This predicate makes the intent explicit: only bare int values are accepted;
    bool True/False are rejected so callers can't silently pass boolean flags
    as list[int] element IDs.
    """
    return isinstance(x, int) and not isinstance(x, bool)


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
                f'{field_name} contains invalid characters: {_safe_repr(value)}. '
                'Only ASCII letters, digits, hyphens, and underscores are allowed.'
            ),
            'error_type': 'ValidationError',
        }
    return None


def validate_project_root(project_root: str) -> dict[str, str] | None:
    """Return an error dict if project_root is not a non-empty absolute path, else None."""
    if not project_root or not os.path.isabs(project_root):
        return {
            'error': f'project_root must be a non-empty absolute path, got: {_safe_repr(project_root)}',
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


def validate_int_ids(ids: object, *, name: str = 'ids') -> dict[str, str] | None:
    """Return an error dict if ids is not a list of plain (non-bool) integers, else None.

    Accepted: list or tuple of int values where none are bool subclass instances.
    Rejected: anything that is not a list/tuple, or any element that is not a plain int.

    The ``name`` keyword argument customises the field label in error messages so
    the helper generalises to tools that accept e.g. ``row_ids: list[int]``.
    """
    if not isinstance(ids, (list, tuple)):
        return {
            'error': f'{name} must be a list or tuple of integers, got {type(ids).__name__}',
            'error_type': 'ValidationError',
        }
    bad_idx = next(
        (i for i, x in enumerate(ids) if not _is_plain_int(x)),
        None,
    )
    if bad_idx is not None:
        bad_val = ids[bad_idx]
        return {
            'error': (
                f'{name}[{bad_idx}] must be int, '
                f'got {type(bad_val).__name__}: {_safe_repr(bad_val)}'
            ),
            'error_type': 'ValidationError',
        }
    return None


def require_project_root(project_root: str) -> None:
    """Raise InputValidationError if project_root is not a non-empty absolute path."""
    if err := validate_project_root(project_root):
        raise InputValidationError(err['error'])


def require_project_id(project_id: str) -> None:
    """Raise InputValidationError if project_id is empty or contains unsafe characters."""
    if err := validate_project_id(project_id):
        raise InputValidationError(err['error'])


def require_run_id(run_id: str) -> None:
    """Raise InputValidationError if run_id is empty or contains unsafe characters."""
    if err := validate_run_id(run_id):
        raise InputValidationError(err['error'])


def require_int_ids(ids: object, *, name: str = 'ids') -> None:
    """Raise InputValidationError if ids is not a list of plain (non-bool) integers."""
    if err := validate_int_ids(ids, name=name):
        raise InputValidationError(err['error'])


def validate_known_project_id(
    project_id: str,
    known_projects: dict[str, str] | None,
) -> dict[str, str] | None:
    """Return an error dict if project_id is not in the known_projects registry, else None.

    Write-side complement of read-side strictness (task 1143); see task 1549.

    When ``known_projects`` is falsy (empty or None), the check is skipped
    (permissive mode) — this preserves test/local-dev ergonomics where
    DASHBOARD_KNOWN_PROJECT_ROOTS is unset.  A single WARNING is logged the
    first time this permissive path is taken (warn-once per process, using the
    module-level ``_empty_registry_warned`` flag).

    The literal string 'DASHBOARD_KNOWN_PROJECT_ROOTS' is hardcoded in the
    rejection message (not imported from fused_memory.models.scope) to keep
    utils/validation.py a leaf module, matching the precedent set by
    harness.py:304 which hardcodes the same literal in its misconfiguration
    message.
    """
    global _empty_registry_warned
    if not known_projects:
        if not _empty_registry_warned:
            _empty_registry_warned = True
            logger.warning(
                'validate_known_project_id called with no known_projects registry — '
                'write-boundary registry check is disabled. '
                'Set DASHBOARD_KNOWN_PROJECT_ROOTS to enable it (task 1143 / task 1549).'
            )
        return None
    if project_id not in known_projects:
        return {
            'error': (
                f'project_id {_safe_repr(project_id)} is not a known project '
                f'(known: {sorted(known_projects)}). '
                f'Set DASHBOARD_KNOWN_PROJECT_ROOTS to register it. '
                f'Write-side complement of read-side strictness (task 1143); see task 1549.'
            ),
            'error_type': 'ValidationError',
        }
    return None


def require_known_project_id(
    project_id: str,
    known_projects: dict[str, str] | None,
) -> None:
    """Raise InputValidationError if project_id is not in the known_projects registry.

    Write-side complement of read-side strictness (task 1143); see task 1549.
    When known_projects is falsy, the check is permissive (no exception raised).
    """
    if err := validate_known_project_id(project_id, known_projects):
        raise InputValidationError(err['error'])
