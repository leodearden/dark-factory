"""Retire a uv-managed venv that was moved after it was built, whose console scripts can no longer be executed, so ``uv run`` rebuilds it where it now lives."""

from __future__ import annotations

import enum
import logging
import os
import shutil
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SHEBANG_PROBE_BYTES = 512
_SCRIPTS_NAMED_IN_LOG = 3


class Disposition(enum.Enum):
    """What was done with a relocated venv; each value is the action as the log states it."""

    RETIRED = 'removed it; uv run will rebuild it at its current path'
    KEPT_NOT_UV_MANAGED = 'kept it: its pyvenv.cfg has no uv key'
    KEPT_RUNNING_INTERPRETER = 'kept it: the running interpreter lives in it'
    RETIRE_FAILED = 'could not remove it'


@dataclass(frozen=True)
class RelocatedVenv:
    path: Path
    built_at: tuple[Path, ...]
    scripts: tuple[str, ...]
    disposition: Disposition


def retire_relocated_venvs(
    root: Path, run_dirs: Iterable[str] = (), *, running_prefix: Path | None = None,
) -> tuple[RelocatedVenv, ...]:
    """Inspect ``root/.venv`` and ``root/<d>/.venv`` per run dir; retire each relocated one."""
    prefix = (running_prefix if running_prefix is not None else Path(sys.prefix)).resolve()
    found: list[RelocatedVenv] = []
    for venv in _candidates(root, run_dirs):
        relocation = _find(venv)
        if relocation is None:
            continue
        built_at, scripts = relocation
        finding = RelocatedVenv(venv, built_at, scripts, _dispose(venv, prefix))
        _report(finding)
        found.append(finding)
    return tuple(found)


def _candidates(root: Path, run_dirs: Iterable[str]) -> list[Path]:
    seen: set[Path] = set()
    venvs: list[Path] = []
    for base in (root, *(root / d for d in run_dirs)):
        venv = base / '.venv'
        key = venv.resolve()
        if key in seen or venv.is_symlink() or not venv.is_dir():
            continue
        seen.add(key)
        venvs.append(venv)
    return venvs


def _shebang_interpreter(script: Path) -> Path | None:
    try:
        if script.is_symlink() or not script.is_file():
            return None
        with script.open('rb') as fh:
            head = fh.read(_SHEBANG_PROBE_BYTES)
    except OSError:
        return None
    first_line = head.split(b'\n', 1)[0]
    tokens = first_line[2:].split() if first_line.startswith(b'#!') else []
    if not tokens:
        return None
    interpreter = Path(os.fsdecode(tokens[0]))
    return interpreter if interpreter.is_absolute() else None


def _moved_from(venv: Path, interpreter: Path) -> Path | None:
    """Where *venv* was built, if *interpreter* is its own interpreter at that other path."""
    built_at = interpreter.parent.parent
    if (
        interpreter.parent.name == 'bin'
        and built_at.name == venv.name
        and built_at != venv
        and not interpreter.exists()
        and (venv / 'bin' / interpreter.name).exists()
    ):
        return built_at
    return None


def _find(venv: Path) -> tuple[tuple[Path, ...], tuple[str, ...]] | None:
    built_at: set[Path] = set()
    scripts: list[str] = []
    try:
        for script in (venv / 'bin').iterdir():
            interpreter = _shebang_interpreter(script)
            origin = None if interpreter is None else _moved_from(venv, interpreter)
            if origin is not None:
                built_at.add(origin)
                scripts.append(script.name)
    except OSError:
        return None
    if not scripts:
        return None
    return tuple(sorted(built_at)), tuple(sorted(scripts))


def _is_uv_managed(venv: Path) -> bool:
    try:
        lines = (venv / 'pyvenv.cfg').read_text(errors='replace').splitlines()
    except OSError:
        return False
    return any(line.partition('=')[0].strip() == 'uv' for line in lines)


def _dispose(venv: Path, running_prefix: Path) -> Disposition:
    if venv.resolve() == running_prefix:
        return Disposition.KEPT_RUNNING_INTERPRETER
    if not _is_uv_managed(venv):
        return Disposition.KEPT_NOT_UV_MANAGED
    return _retire(venv)


def _retire(venv: Path) -> Disposition:
    try:
        shutil.rmtree(venv)
    except OSError as exc:
        logger.warning('Relocated venv %s: removal failed: %s', venv, exc)
        return Disposition.RETIRE_FAILED
    return Disposition.RETIRED


def _report(finding: RelocatedVenv) -> None:
    named = ', '.join(finding.scripts[:_SCRIPTS_NAMED_IN_LOG])
    more = ', ...' if len(finding.scripts) > _SCRIPTS_NAMED_IN_LOG else ''
    logger.warning(
        'Relocated venv %s: %d console script(s) (%s%s) have a shebang interpreter '
        'under missing build path(s) %s; %s',
        finding.path, len(finding.scripts), named, more,
        ', '.join(str(p) for p in finding.built_at), finding.disposition.value,
    )
