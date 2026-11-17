"""Bind this checkout's workspace members ahead of anything the environment resolves.

Importing this module puts every ``<member>/src`` that this checkout's root
pyproject.toml declares at the front of ``sys.path``, in declared order. A script
binds itself with ``import _checkout as _checkout`` placed before its first
first-party import. Contract: plans/review-all-s25-scripts-instruments-prd.md
C1/D2.

:func:`workspace_members` extends C1, whose import was the whole API, with a
callable reader of any root's member list. It is the one reader of that list
in scripts/: the binding goes through it, and so does
scripts/source_measures.py when it measures a tree.
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Final

_MEMBERS_PATH: Final = ('tool', 'uv', 'workspace', 'members')
MEMBERS_KEY: Final = f'[{".".join(_MEMBERS_PATH[:-1])}].{_MEMBERS_PATH[-1]}'
_GLOB_CHARS: Final = '*?['


class CheckoutBindingError(ImportError, RuntimeError):
    """A pyproject's [tool.uv.workspace].members cannot be read as a plain member list.

    Raised at import, it means this checkout's members are unknown, so its script
    must not run against another's.
    """


def _refusal(pyproject: Path, reason: str) -> CheckoutBindingError:
    return CheckoutBindingError(f'{pyproject}: {reason}', name='_checkout', path=str(pyproject))


def _read_config(pyproject: Path) -> object:
    try:
        return tomllib.loads(pyproject.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise _refusal(pyproject, f'{exc.__class__.__name__}: {exc}') from exc


def _declared_value(config: object) -> object:
    found: object = config
    for key in _MEMBERS_PATH:
        found = found.get(key) if isinstance(found, dict) else None
    return found


def workspace_members(root: Path) -> tuple[str, ...]:
    """The member directory names *root*'s pyproject.toml declares, in declared order.

    ``[tool.uv.workspace].members`` is accepted only as a non-empty list of
    non-empty str, none of them a glob.
    """
    pyproject = root / 'pyproject.toml'
    found = _declared_value(_read_config(pyproject))
    entries: list[object] = found if isinstance(found, list) else []
    members = tuple(entry for entry in entries if isinstance(entry, str) and entry)
    if not members or len(members) != len(entries):
        raise _refusal(
            pyproject,
            f'{MEMBERS_KEY} must be a non-empty list of non-empty member directory names; '
            f'found {found!r}',
        )
    globbed = [member for member in members if any(char in member for char in _GLOB_CHARS)]
    if globbed:
        raise _refusal(
            pyproject,
            f'{MEMBERS_KEY} holds glob member(s) {globbed!r}, which are not supported '
            '-- list each member directory by name',
        )
    return members


def _member_srcs(root: Path) -> tuple[Path, ...]:
    declared = (root / member / 'src' for member in workspace_members(root))
    return tuple(src for src in declared if src.is_dir())


def _bind(srcs: tuple[Path, ...]) -> None:
    target = [str(src) for src in srcs]
    if sys.path[: len(target)] != target:
        sys.path[:0] = target


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MEMBER_SRC: Final[tuple[Path, ...]] = _member_srcs(ROOT)
_bind(MEMBER_SRC)
