"""Bind this checkout's workspace members ahead of anything the environment resolves.

Importing this module puts every ``<member>/src`` that this checkout's root
pyproject.toml declares at the front of ``sys.path``, in declared order. A script
binds itself with ``import _checkout as _checkout`` placed before its first
first-party import. Contract: plans/review-all-s25-scripts-instruments-prd.md C1/D2.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Final

_MEMBERS_KEY: Final = '[tool.uv.workspace].members'
_GLOB_CHARS: Final = '*?['


class CheckoutBindingError(ImportError, RuntimeError):
    """This checkout's members are unknown, so its script must not run against another's."""


def _refusal(pyproject: Path, reason: str) -> CheckoutBindingError:
    return CheckoutBindingError(
        f'{pyproject}: cannot bind this checkout -- {reason}',
        name='_checkout',
        path=str(pyproject),
    )


def _declared_members(pyproject: Path) -> list[str]:
    try:
        config: object = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _refusal(pyproject, f'{exc.__class__.__name__}: {exc}') from exc
    found: object = config
    for key in ('tool', 'uv', 'workspace', 'members'):
        found = found.get(key) if isinstance(found, dict) else None
    entries: list[object] = found if isinstance(found, list) else []
    members = [
        entry
        for entry in entries
        if isinstance(entry, str) and entry and not any(char in entry for char in _GLOB_CHARS)
    ]
    if not members or len(members) != len(entries):
        raise _refusal(
            pyproject,
            f'{_MEMBERS_KEY} must be a non-empty list of member directory names, '
            f'none of them a glob; found {found!r}',
        )
    return members


def _member_srcs(root: Path) -> tuple[Path, ...]:
    declared = (root / member / 'src' for member in _declared_members(root / 'pyproject.toml'))
    return tuple(src for src in declared if src.is_dir())


def _bind(srcs: tuple[Path, ...]) -> None:
    target = [str(src) for src in srcs]
    if sys.path[: len(target)] != target:
        sys.path[:0] = target


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MEMBER_SRC: Final[tuple[Path, ...]] = _member_srcs(ROOT)
_bind(MEMBER_SRC)
