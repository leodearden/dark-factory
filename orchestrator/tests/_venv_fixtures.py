"""Venvs on disk for tests: healthy ones, and ones moved after they were built.

A venv's console scripts name their interpreter by absolute path in the
shebang.  :func:`make_venv` with ``built_at=None`` writes that path as the
venv's own (existing) ``bin/python``; with ``built_at`` set to a path that is
never created it writes the path the venv had before it was moved, which is
what a checkout renamed after ``uv sync`` looks like.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _write_script(path: Path, interpreter: str) -> None:
    path.write_text(f'#!{interpreter}\nimport sys; sys.exit(0)\n')
    path.chmod(0o755)


def make_venv(
    venv_dir: Path,
    *,
    built_at: Path | None = None,
    uv_managed: bool = True,
    scripts: tuple[str, ...] = ('pytest', 'mypy'),
) -> Path:
    """Create *venv_dir* as a venv whose scripts name ``{built_at or venv_dir}/bin/python``."""
    bin_dir = venv_dir / 'bin'
    bin_dir.mkdir(parents=True)
    cfg = ['home = /usr/bin', 'version_info = 3.12.3']
    if uv_managed:
        cfg.append('uv = 0.11.6')
    (venv_dir / 'pyvenv.cfg').write_text('\n'.join(cfg) + '\n')
    os.symlink(sys.executable, bin_dir / 'python')
    interpreter = f'{built_at or venv_dir}/bin/python'
    for name in scripts:
        _write_script(bin_dir / name, interpreter)
    return venv_dir


def stale_script(venv_dir: Path, name: str, interpreter: str) -> Path:
    """Add one console script to *venv_dir* whose shebang names *interpreter*."""
    script = venv_dir / 'bin' / name
    _write_script(script, interpreter)
    return script
