"""Only the allowlisted modules may import ``shared.declared_scope``.

plans/follow-up-filing-shape-prd.md §3 decision 8: the declared-scope band is filing-time evidence
for the curator, the packing sweep and the calibration report. Verify admission, verify scope, the
merge-lane disjoint skip, the verify-result cache, simple-path triage and model routing must never
read it, so a module outside ALLOWED_IMPORTERS that imports it fails here.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import NamedTuple

import pytest
from silent_fallthrough_scan import ParsedFile

_TARGET = 'shared.declared_scope'
_TARGET_PACKAGE, _, _TARGET_NAME = _TARGET.rpartition('.')
_DECISION_8 = 'plans/follow-up-filing-shape-prd.md §3 decision 8'


class Allowed(NamedTuple):
    role: str
    ratification: str


ALLOWED_IMPORTERS: Mapping[str, Allowed] = MappingProxyType({
    'shared/src/shared/task_metadata.py': Allowed('registry', _DECISION_8),
    'fused-memory/src/fused_memory/middleware/task_interceptor.py': Allowed('writer', _DECISION_8),
    'fused-memory/src/fused_memory/middleware/task_curator.py': Allowed('reader (γ)', _DECISION_8),
    'scripts/followup_filing_shape_report.py': Allowed('reader (θ)', _DECISION_8),
})

FORBIDDEN_READERS: frozenset[str] = frozenset({
    'orchestrator/src/orchestrator/verify_admission_gate.py',
    'shared/src/shared/verify_admission.py',
    'orchestrator/src/orchestrator/verify_plan.py',
    'orchestrator/src/orchestrator/merge_lane/gates.py',
    'orchestrator/src/orchestrator/verify_evidence.py',
    'orchestrator/src/orchestrator/agents/triage.py',
    'orchestrator/src/orchestrator/routing.py',
})


def _package(relpath: str) -> tuple[str, ...]:
    _, under_src, module_path = relpath.partition('/src/')
    return PurePosixPath(module_path).parent.parts if under_src else ()


def _resolved_module(node: ast.ImportFrom, package: tuple[str, ...]) -> str:
    base = package[: len(package) - node.level + 1] if node.level else ()
    return '.'.join([*base, *([node.module] if node.module else [])])


def _is_target(module: str) -> bool:
    return module == _TARGET or module.startswith(f'{_TARGET}.')


def _imports_target(node: ast.Import | ast.ImportFrom, package: tuple[str, ...]) -> bool:
    if isinstance(node, ast.Import):
        return any(_is_target(alias.name) for alias in node.names)
    module = _resolved_module(node, package)
    if _is_target(module):
        return True
    return module == _TARGET_PACKAGE and any(alias.name == _TARGET_NAME for alias in node.names)


def declared_scope_importers(records: Iterable[ParsedFile]) -> dict[str, list[int]]:
    """``relpath -> import line numbers`` for every record that imports ``shared.declared_scope``."""
    found: dict[str, list[int]] = {}
    for record in records:
        if record.tree is None or _TARGET_NAME not in record.source:
            continue
        package = _package(record.relpath)
        lines = sorted(
            node.lineno for node in ast.walk(record.tree)
            if isinstance(node, ast.Import | ast.ImportFrom) and _imports_target(node, package)
        )
        if lines:
            found[record.relpath] = lines
    return found


def unallowed_importers(importers: Mapping[str, list[int]]) -> list[str]:
    return [
        f'{relpath}:{lineno}'
        for relpath, linenos in sorted(importers.items())
        if relpath not in ALLOWED_IMPORTERS
        for lineno in linenos
    ]


@pytest.fixture(scope='module')
def real_importers(first_party_tree: tuple[ParsedFile, ...]) -> dict[str, list[int]]:
    return declared_scope_importers(first_party_tree)


def test_only_allowlisted_modules_import_declared_scope(real_importers: dict[str, list[int]]) -> None:
    offenders = unallowed_importers(real_importers)

    assert not offenders, (
        f'these modules import {_TARGET} but are not in ALLOWED_IMPORTERS:\n  '
        + '\n  '.join(offenders)
        + f'\nThe declared-scope band must not steer verify, merge, triage or routing ({_DECISION_8}).'
    )


def test_the_sweep_sees_the_known_importers(
    first_party_tree: tuple[ParsedFile, ...], real_importers: dict[str, list[int]],
) -> None:
    assert len(first_party_tree) > 150, f'only {len(first_party_tree)} first-party files parsed'
    assert {
        'shared/src/shared/task_metadata.py',
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
    } <= set(real_importers), f'the sweep missed a known importer: {real_importers}'


def test_forbidden_readers_are_real_and_never_allowed(first_party_tree: tuple[ParsedFile, ...]) -> None:
    present = {record.relpath for record in first_party_tree}

    assert not FORBIDDEN_READERS & set(ALLOWED_IMPORTERS)
    assert present >= FORBIDDEN_READERS, f'renamed or removed: {sorted(FORBIDDEN_READERS - present)}'


def _probe(relpath: str, source: str) -> ParsedFile:
    source = textwrap.dedent(source)
    return ParsedFile(
        path=Path('/probe') / relpath, relpath=relpath, source=source,
        tree=ast.parse(source), syntax_error=None,
    )


_ABSOLUTE_SPELLINGS = (
    'import shared.declared_scope\n',
    'import shared.declared_scope as scope\n',
    'from shared.declared_scope import DeclaredScope\n',
    'from shared import declared_scope\n',
    '''
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from shared.declared_scope import DeclaredScope
    ''',
)
_RELATIVE_SPELLINGS = (
    'from .declared_scope import DeclaredScope\n',
    'from . import declared_scope\n',
)


def _spellings_for(relpath: str) -> tuple[str, ...]:
    inside_shared = _package(relpath)[:1] == ('shared',)
    return _ABSOLUTE_SPELLINGS + (_RELATIVE_SPELLINGS if inside_shared else ())


@pytest.mark.parametrize(
    ('relpath', 'source'),
    [
        (relpath, source)
        for relpath in sorted(FORBIDDEN_READERS)
        for source in _spellings_for(relpath)
    ],
)
def test_a_forbidden_reader_importing_declared_scope_is_reported(relpath: str, source: str) -> None:
    importers = declared_scope_importers([_probe(relpath, source)])

    assert relpath in importers
    assert unallowed_importers(importers) == [f'{relpath}:{line}' for line in importers[relpath]]


@pytest.mark.parametrize(
    ('relpath', 'source'),
    [
        ('orchestrator/src/orchestrator/routing.py', 'from shared.declared_scope_extra import x\n'),
        ('orchestrator/src/orchestrator/routing.py', 'import shared.task_metadata\n'),
        ('orchestrator/src/orchestrator/routing.py', 'from shared import task_metadata\n'),
        ('orchestrator/src/orchestrator/routing.py', 'from .declared_scope import DeclaredScope\n'),
        ('orchestrator/src/orchestrator/routing.py', "SLICE = 'declared_scope'\n"),
    ],
)
def test_a_lookalike_is_not_an_importer(relpath: str, source: str) -> None:
    assert declared_scope_importers([_probe(relpath, source)]) == {}
