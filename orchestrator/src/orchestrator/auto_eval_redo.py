"""The task text an auto-eval redo sibling is filed with."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shared.elision import elide_at_word

_REDO_TITLE_MAX = 200


def redo_submit_text(
    original: Mapping[str, Any],
    *,
    original_id: str,
    block_phase: str,
    optimistic_path: str,
    new_branch: str,
) -> dict[str, str]:
    """The title, description and (when the original has any) details arguments of the redo's submit_task call."""
    original_title = str(original.get('title') or '')
    description = (
        f'Automated full-architect redo of task {original_id} '
        f'(blocked at phase={block_phase} via optimistic path '
        f'{optimistic_path!r}). Original branch+worktree are renamed to '
        f'{new_branch} for forensic comparison once this redo is released.'
        f'\n\nOriginal title: {original_title}'
    )
    original_description = str(original.get('description') or '')
    if original_description.strip():
        description += f'\n\nOriginal description:\n{original_description}'

    args = {
        'title': elide_at_word(f'[auto-eval redo] {original_title}', _REDO_TITLE_MAX),
        'description': description,
    }
    original_details = str(original.get('details') or '')
    if original_details.strip():
        args['details'] = original_details
    return args
