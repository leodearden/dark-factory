"""The names in the per-project cgroup topology: the slice name rule's builder and
recogniser (one systemd slice level per project id) and the daemon's subgroup
inside its service.

Contract: docs/cgroup-slice-topology.md.
"""

from __future__ import annotations

import re

__all__ = ['DAEMON_SUBGROUP', 'is_project_slice_name', 'project_slice_name']

DAEMON_SUBGROUP = 'daemon'

_SLICE_ALPHABET = 'a-z0-9_'
_SLICE_PREFIX = 'df-'
_SLICE_SUFFIX = '.slice'

_OUTSIDE_SLICE_ALPHABET = re.compile(f'[^{_SLICE_ALPHABET}]')
_PROJECT_SLICE_NAME = re.compile(
    f'{re.escape(_SLICE_PREFIX)}[{_SLICE_ALPHABET}]+{re.escape(_SLICE_SUFFIX)}'
)


def project_slice_name(project_id: str) -> str:
    if not project_id:
        raise ValueError(
            f'project id {project_id!r} is empty; a per-project cgroup slice needs a non-empty project id'
        )
    flattened = _OUTSIDE_SLICE_ALPHABET.sub('_', project_id.lower())
    return f'{_SLICE_PREFIX}{flattened}{_SLICE_SUFFIX}'


def is_project_slice_name(name: str) -> bool:
    return _PROJECT_SLICE_NAME.fullmatch(name) is not None
