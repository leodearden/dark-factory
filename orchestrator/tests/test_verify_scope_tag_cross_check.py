"""The shared verify scope-tag rule agrees with the copy verify.py still runs.

A transition pin. ``shared/src/shared/verify_scope_tag.py`` is the rule's one
public home (PRD ``plans/per-consumer-resource-recording-prd.md`` leaf 1), but
``orchestrator/src/orchestrator/verify.py::_scope_tag_for`` and
``::_verify_scope_name`` still build the scope names the load recorder parses.
Leaf 4 (task 6355) switches verify.py to the shared rule and DELETES this file.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from shared.verify_scope_tag import parse_verify_scope_unit, scope_tag_for

from orchestrator.verify import _scope_tag_for, _verify_scope_name


def _symlink_to_a_directory(tmp_path: Path) -> Path:
    target = tmp_path / 'linked-target'
    target.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(target)
    return link


_ROOTS: dict[str, Callable[[Path], Path]] = {
    'dark-factory': lambda _: Path('/home/leo/src/dark-factory'),
    'reify': lambda _: Path('/home/leo/src/reify'),
    'spaces-and-underscore': lambda tmp: tmp / 'My_Solar Challenge',
    'forty-char-basename': lambda tmp: tmp / ('long-basename-' * 3)[:40],
    'all-dashes': lambda tmp: tmp / '---',
    'symlink': _symlink_to_a_directory,
    'relative': lambda _: Path('relative/project'),
}


@pytest.mark.parametrize('root_kind', _ROOTS)
def test_shared_rule_matches_verify_py(tmp_path: Path, root_kind: str) -> None:
    root = _ROOTS[root_kind](tmp_path)

    assert scope_tag_for(root) == _scope_tag_for(root), root


@pytest.mark.parametrize('root_kind', _ROOTS)
def test_shared_parser_reads_the_names_verify_py_builds(
    tmp_path: Path, root_kind: str
) -> None:
    tag = _scope_tag_for(_ROOTS[root_kind](tmp_path))

    unit = parse_verify_scope_unit(_verify_scope_name(tag))

    assert unit is not None, tag
    assert unit.tag == tag
