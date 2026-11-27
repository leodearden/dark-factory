"""Capture directories as verify's in-scope script leaves them, for C4 record tests.

The layout is a protocol between ``CommandCapture.wrap``'s script and
``CommandCapture.record``, with one home:
orchestrator/src/orchestrator/verify_command_resources.py. This is the one
test-side writer of it, so a test fills a directory without restating the
file names.

Imported by bare module name, like ``_fake_systemd_run``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from orchestrator.verify_command_resources import _CGROUP_FILE, _COMPLETE_FILE


def write_capture_directory(
    directory: Path,
    files: Mapping[str, str],
    *,
    cgroup: str | None,
    complete: bool,
) -> Path:
    """Write *files*, the unified *cgroup* path the script recorded, and its completion mark.

    ``cgroup=None`` leaves a script that never recorded its cgroup;
    ``complete=False`` one that ended before its copy finished.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (directory / name).write_text(text)
    if cgroup is not None:
        (directory / _CGROUP_FILE).write_text(f'0::{cgroup}\n')
    if complete:
        (directory / _COMPLETE_FILE).write_text('')
    return directory
