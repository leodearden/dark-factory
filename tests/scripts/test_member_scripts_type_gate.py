"""Gate contract: every workspace member's ``scripts/`` sits inside BOTH of its pyright gates.

Task 5267. A member with a ``[tool.pyright]`` table type-checks through two
entry points that select files independently:

  * a BARE ``pyright`` run from the member directory (``hooks/project-checks``,
    the fleet chain in ``dark-factory-orchestrator.yaml``, an editor), which
    selects by ``[tool.pyright] include``;
  * the member's module verify type leg, ``<member>/orchestrator.yaml``
    ``type_check_command``. Under the root config's
    ``merge_verify_breadth: "full"`` that command runs VERBATIM on every merge,
    and pyright's explicit CLI paths OVERRIDE ``include``, so
    ``pyright src/ tests/`` silently drops ``scripts/`` even when ``include``
    names it.

A file reached by neither is type-checked by nothing, while every gate reports
green. That is how ``fused-memory/scripts`` accumulated 19 pyright errors
(measured at base main ``bde49da794``) with no red anywhere.

SUBJECTS ARE DISCOVERED, never listed: every module config the production walk
registers whose member directory carries a ``[tool.pyright]`` table AND at
least one ``scripts/**/*.py``. A member that grows a ``scripts/`` directory
later is picked up with no edit here. ``KNOWN_SCRIPTS_BEARING_MEMBERS`` is an
anti-vacuity floor, in the style of
``tests/scripts/test_module_type_check_invocation.py::KNOWN_MODULE_CONFIG_PREFIXES``:
a broken discovery fails loudly rather than passing on an empty set.

THE FRAME. Every script is expressed relative to its MEMBER directory
(``scripts/foo.py``), because both gates resolve there: pyright reads
``include`` relative to the pyproject, and the module command runs under
``uv run --directory <member>``. Assertion (C) checks that wrapper rather than
assuming it, since a command running from the repo root would make
member-relative coverage meaningless.

NO CARVE-OUTS. A finding is FIXED, never excluded or ignored — the precedent is
``tests/scripts/test_root_py_type_gate.py::test_root_pyright_include_targets_every_repo_root_py``.

PLACEMENT: ``tests/scripts/`` carries its own registered module config, so this
guard runs on every merge under full breadth and cannot be silenced by an edit
to the member commands it asserts about. It imports the shared command parser
``verify_command_invariants`` and no sibling test file.
"""
from __future__ import annotations

import dataclasses
import fnmatch
import itertools
import pathlib
import tomllib
from collections.abc import Callable
from typing import Any

from orchestrator.config import ModuleConfig
from verify_command_invariants import (
    PYRIGHT,
    anchor_split,
    covers,
    positional_targets,
    required_segment,
)

REPO_ROOT = pathlib.Path(__file__).parents[2]

KNOWN_SCRIPTS_BEARING_MEMBERS = frozenset({'fused-memory', 'orchestrator'})

_NO_CARVE_OUT_PRECEDENT = (
    'tests/scripts/test_root_py_type_gate.py::'
    'test_root_pyright_include_targets_every_repo_root_py'
)

# Pyright flags whose FOLLOWING token is a value, not a file target (from
# `pyright --help` at 1.1.408). `--threads` takes an OPTIONAL count and is
# deliberately absent: dropping its successor would swallow a real target when
# the count is omitted, whereas a phantom numeric target covers no script path
# and can only turn this guard red, never green.
_PYRIGHT_VALUE_FLAGS = frozenset({
    '--createstub', '--level', '-p', '--project', '--pythonplatform',
    '--pythonpath', '--pythonversion', '-t', '--typeshedpath', '-v',
    '--venvpath', '--verifytypes',
})


@dataclasses.dataclass(frozen=True)
class _Subject:
    prefix: str
    pyright: dict[str, Any]
    scripts: tuple[str, ...]
    type_check_command: str | None


def _pyright_table(member_dir: pathlib.Path) -> dict[str, Any] | None:
    pyproject = member_dir / 'pyproject.toml'
    if not pyproject.is_file():
        return None
    table = tomllib.loads(pyproject.read_text(encoding='utf-8')).get('tool', {}).get('pyright')
    return table if isinstance(table, dict) else None


def _member_scripts(member_dir: pathlib.Path) -> tuple[str, ...]:
    return tuple(sorted(
        path.relative_to(member_dir).as_posix()
        for path in (member_dir / 'scripts').rglob('*.py')
        if '__pycache__' not in path.parts
    ))


def _subjects(discover: Callable[[], dict[str, ModuleConfig]]) -> list[_Subject]:
    """Every discovered member with a ``[tool.pyright]`` table and a ``scripts/*.py``."""
    subjects: list[_Subject] = []
    for prefix, mc in sorted(discover().items()):
        member_dir = REPO_ROOT / prefix
        table = _pyright_table(member_dir)
        scripts = _member_scripts(member_dir)
        if table is not None and scripts:
            subjects.append(_Subject(prefix, table, scripts, mc.type_check_command))

    missing = KNOWN_SCRIPTS_BEARING_MEMBERS - {s.prefix for s in subjects}
    assert not missing, (
        f'known scripts-bearing member(s) {sorted(missing)} were not discovered '
        f'as subjects (task 5267) — either the production walk '
        f'(orchestrator.config._discover_module_configs) regressed, or the '
        f'member lost its [tool.pyright] table or its scripts/*.py. The checks '
        f'below would pass vacuously on the shrunken set. Subjects: '
        f'{[s.prefix for s in subjects]}'
    )
    return subjects


def _remedy(prefix: str) -> str:
    return (
        f"FIX any diagnostics this exposes rather than carving files out "
        f"(precedent: {_NO_CARVE_OUT_PRECEDENT}). Probe: "
        f"`cd {prefix} && uv run pyright`."
    )


def _is_carved_out(rel: str, entry: str) -> bool:
    # fnmatch is this caller's policy: pyright accepts glob entries in
    # exclude/ignore, which `covers`' exact-element match cannot see.
    return covers(rel, [entry]) or fnmatch.fnmatch(rel, entry)


def _runs_in_member_dir(wrapper: list[str], prefix: str) -> bool:
    return f'--directory={prefix}' in wrapper or any(
        flag == '--directory' and value == prefix
        for flag, value in itertools.pairwise(wrapper)
    )


def test_pyright_include_covers_every_member_script(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """(A) A bare ``pyright`` from the member directory selects every ``scripts/*.py``."""
    violations: list[str] = []
    for subject in _subjects(discover_module_configs):
        include = subject.pyright.get('include')
        if not (isinstance(include, list) and include):
            violations.append(
                f'{subject.prefix}/pyproject.toml [tool.pyright] declares no '
                f'non-empty `include` (got {include!r}), so coverage cannot be '
                f'established'
            )
            continue
        uncovered = [rel for rel in subject.scripts if not covers(rel, include)]
        if uncovered:
            violations.append(
                f'{subject.prefix}: [tool.pyright] include {include} does not '
                f'cover {uncovered}. Add "scripts" to include in '
                f'{subject.prefix}/pyproject.toml. {_remedy(subject.prefix)}'
            )
    assert not violations, 'scripts/ outside the bare-pyright gate (task 5267):\n  ' + (
        '\n  '.join(violations)
    )


def test_no_pyright_exclude_or_ignore_carves_out_a_member_script(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """(B) No ``exclude``/``ignore`` entry un-gates a ``scripts/*.py`` while pyright reports green."""
    violations: list[str] = []
    for subject in _subjects(discover_module_configs):
        for key in ('exclude', 'ignore'):
            entries = subject.pyright.get(key) or []
            for rel in subject.scripts:
                carving = [entry for entry in entries if _is_carved_out(rel, entry)]
                if carving:
                    violations.append(
                        f'{subject.prefix}: [tool.pyright] {key} carves out '
                        f'{rel!r} via {carving}. Remove the entry. '
                        f'{_remedy(subject.prefix)}'
                    )
    assert not violations, 'scripts/ carved out of pyright (task 5267):\n  ' + (
        '\n  '.join(violations)
    )


def test_module_verify_type_leg_covers_every_member_script(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """(C) The member's module ``type_check_command`` reaches every ``scripts/*.py``.

    Explicit CLI targets override ``include``, so this is checked separately
    from (A). An empty target list means pyright selects by ``include``, which
    (A) already proves.
    """
    violations: list[str] = []
    for subject in _subjects(discover_module_configs):
        prefix = subject.prefix
        label = f'{prefix}/orchestrator.yaml type_check_command'
        cmd = subject.type_check_command
        if not cmd:
            violations.append(
                f'{label} is empty, so the module verify leg type-checks '
                f'nothing, scripts/ included'
            )
            continue
        segment = required_segment(cmd, PYRIGHT, label=label)
        wrapper, _ = anchor_split(segment, PYRIGHT, label=label)
        if not _runs_in_member_dir(wrapper, prefix):
            violations.append(
                f'{label} {cmd!r} does not run pyright under `uv run '
                f'--directory {prefix}`, so its targets are not in the '
                f'member-relative frame this guard checks coverage in'
            )
            continue
        targets = positional_targets(
            segment, PYRIGHT, value_flags=_PYRIGHT_VALUE_FLAGS, label=label
        )
        if not targets:
            continue
        uncovered = [rel for rel in subject.scripts if not covers(rel, targets)]
        if uncovered:
            violations.append(
                f'{label} {cmd!r} names targets {targets}, which override '
                f'[tool.pyright] include and miss {uncovered}. Add scripts/ to '
                f'that command. {_remedy(prefix)}'
            )
    assert not violations, 'scripts/ outside the module verify type leg (task 5267):\n  ' + (
        '\n  '.join(violations)
    )
