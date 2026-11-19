"""The per-project cgroup slice name rule: one systemd slice level per project id.

Contract: docs/cgroup-slice-topology.md.
"""

from __future__ import annotations

import re

__all__ = ['project_slice_name']

_OUTSIDE_SLICE_ALPHABET = re.compile('[^a-z0-9_]')


def project_slice_name(project_id: str) -> str:
    if not project_id:
        raise ValueError(
            f'project id {project_id!r} is empty; a per-project cgroup slice needs a non-empty project id'
        )
    return f'df-{_OUTSIDE_SLICE_ALPHABET.sub("_", project_id.lower())}.slice'
