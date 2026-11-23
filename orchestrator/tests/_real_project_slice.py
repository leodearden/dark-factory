"""A unique per-test project slice for tests that spawn real systemd verify scopes.

A verify scope is spawned under its project's ``df-<project_id>.slice``, and a
slice created implicitly by ``systemd-run --slice=`` stays active after its
scope is collected. A real-systemd test therefore takes this fixture: it gets
a project id no other test or project shares, and teardown stops that one
slice (never ``df.slice``, the shared parent of every production project
slice) and reverts any runtime property set on it.

Imported by bare module name, like ``_fake_systemd_run``. A module using the
fixture re-exports it by redundant alias
(``from _real_project_slice import real_project_slice as real_project_slice``):
the import binding is what registers it with pytest. A module rather than
``conftest.py`` because a conftest.py edit trips verify.py's has_conftest and
widens the merge-time verify to the whole owning-package suite.
"""

from __future__ import annotations

import contextlib
import subprocess
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from shared.cgroup_slice import project_slice_name


@dataclass(frozen=True)
class RealProjectSlice:
    project_id: str
    slice_name: str


@pytest.fixture
def real_project_slice() -> Iterator[RealProjectSlice]:
    project_id = f'dftest_{uuid.uuid4().hex[:10]}'
    slice_name = project_slice_name(project_id)
    yield RealProjectSlice(project_id=project_id, slice_name=slice_name)
    for verb in ('stop', 'revert'):
        with contextlib.suppress(Exception):
            subprocess.run(
                ['systemctl', '--user', verb, slice_name],
                capture_output=True, timeout=15,
            )
