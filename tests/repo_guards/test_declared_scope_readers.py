"""Only the allowlisted modules may import ``shared.declared_scope``.

plans/follow-up-filing-shape-prd.md §3 decision 8: the declared-scope band is filing-time evidence
for the curator, the packing sweep and the calibration report. Verify admission, verify scope, the
merge-lane disjoint skip, the verify-result cache, simple-path triage and model routing must never
read it, so a module outside ALLOWED_IMPORTERS that imports it fails here. A FORBIDDEN_READERS module
also fails if it names the metadata key or any name ``shared.declared_scope`` exports, because a raw
``task_metadata.get('declared_scope')`` or the ``DeclaredScope`` that ``shared.task_metadata`` binds
needs no import of this module.
"""

from __future__ import annotations

import ast
import textwrap
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import NamedTuple

import pytest
from shared.declared_scope import DECLARED_SCOPE_METADATA_KEY
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from silent_fallthrough_scan import ParsedFile

from shared import declared_scope

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_TARGET = 'shared.declared_scope'
_TARGET_PACKAGE, _, _TARGET_NAME = _TARGET.rpartition('.')
_DECISION_8 = 'plans/follow-up-filing-shape-prd.md §3 decision 8'
_PACKING_PRD = 'plans/followup-packing-sweep-prd.md §3 decision 7 (task 8451)'


class Allowed(NamedTuple):
    role: str
    ratification: str


ALLOWED_IMPORTERS: Mapping[str, Allowed] = MappingProxyType({
    'shared/src/shared/task_metadata.py': Allowed('registry', _DECISION_8),
    'fused-memory/src/fused_memory/middleware/task_interceptor.py': Allowed('writer', _DECISION_8),
    'fused-memory/src/fused_memory/backends/task_backend_errors.py': Allowed('refusal', _DECISION_8),
    'fused-memory/src/fused_memory/middleware/task_curator.py': Allowed('reader (γ)', _DECISION_8),
    'scripts/followup_filing_shape_report.py': Allowed('reader (θ)', _DECISION_8),
    'fused-memory/src/fused_memory/middleware/followup_packing.py': Allowed('reader (packing P0)', _PACKING_PRD),
    'fused-memory/src/fused_memory/middleware/curator_types.py': Allowed(
        'reader (γ: pool and candidate render)', _DECISION_8,
    ),
})

_PENDING: Mapping[str, str] = MappingProxyType({
    'scripts/followup_filing_shape_report.py': 'θ',
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
        'fused-memory/src/fused_memory/backends/task_backend_errors.py',
    } <= set(real_importers), f'the sweep missed a known importer: {real_importers}'


def test_forbidden_readers_are_real_and_never_allowed(first_party_tree: tuple[ParsedFile, ...]) -> None:
    present = {record.relpath for record in first_party_tree}

    assert not FORBIDDEN_READERS & set(ALLOWED_IMPORTERS)
    assert present >= FORBIDDEN_READERS, f'renamed or removed: {sorted(FORBIDDEN_READERS - present)}'


def test_every_allowlist_entry_is_real_unless_pending(first_party_tree: tuple[ParsedFile, ...]) -> None:
    present = {record.relpath for record in first_party_tree}
    landed = sorted(set(_PENDING) & present)
    missing = sorted(set(ALLOWED_IMPORTERS) - set(_PENDING) - present)

    assert set(_PENDING) <= set(ALLOWED_IMPORTERS)
    assert not landed, f'landed, so delete its _PENDING entry: {landed}'
    assert not missing, f'renamed or removed, so the grant is stale: {missing}'


_EXPORTED_NAMES = frozenset({*declared_scope.__all__, DECLARED_SCOPE_METADATA_KEY})


def _names_declared_scope(node: ast.AST) -> bool:
    match node:
        case ast.Constant(value=str() as value):
            return value == DECLARED_SCOPE_METADATA_KEY
        case ast.Name(id=name) | ast.Attribute(attr=name) | ast.alias(name=name):
            return name in _EXPORTED_NAMES
    return False


def forbidden_reader_mentions(records: Iterable[ParsedFile]) -> dict[str, list[int]]:
    """``relpath -> line numbers`` where a FORBIDDEN_READERS module names the key or an exported name."""
    found: dict[str, list[int]] = {}
    for record in records:
        if record.tree is None or record.relpath not in FORBIDDEN_READERS:
            continue
        lines = sorted({
            node.lineno for node in ast.walk(record.tree)
            if isinstance(node, ast.expr | ast.alias) and _names_declared_scope(node)
        })
        if lines:
            found[record.relpath] = lines
    return found


def test_no_forbidden_reader_names_declared_scope(first_party_tree: tuple[ParsedFile, ...]) -> None:
    mentions = forbidden_reader_mentions(first_party_tree)

    assert not mentions, (
        f'these forbidden readers name {DECLARED_SCOPE_METADATA_KEY!r} or a {_TARGET} export: '
        f'{mentions}\nThe declared-scope band must not steer verify, merge, triage or routing '
        f'({_DECISION_8}).'
    )


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
    ],
)
def test_a_lookalike_is_not_an_importer(relpath: str, source: str) -> None:
    assert declared_scope_importers([_probe(relpath, source)]) == {}


_MENTION_SPELLINGS = (
    "SLICE = 'declared_scope'\n",
    "band = inputs.task_metadata.get('declared_scope', {}).get('band')\n",
    "scope = parse_metadata(blob).model_extra['declared_scope']\n",
    'from shared.task_metadata import DeclaredScope\n',
    'import shared.task_metadata as tm\nrecord_type = tm.DeclaredScope\n',
    'key = task_metadata.DECLARED_SCOPE_METADATA_KEY\n',
    'violation = scope_module.stamp_violation({}, {}, replace=True)\n',
)


@pytest.mark.parametrize(
    ('relpath', 'source'),
    [(relpath, source) for relpath in sorted(FORBIDDEN_READERS) for source in _MENTION_SPELLINGS],
)
def test_a_forbidden_reader_naming_declared_scope_is_reported(relpath: str, source: str) -> None:
    mentions = forbidden_reader_mentions([_probe(relpath, source)])

    assert list(mentions) == [relpath]


@pytest.mark.parametrize(
    ('relpath', 'source'),
    [
        ('orchestrator/src/orchestrator/routing.py', '"""This module never reads declared_scope."""\n'),
        ('orchestrator/src/orchestrator/routing.py', "KEY = 'declared_scope_extra'\n"),
        ('orchestrator/src/orchestrator/routing.py', 'scope = declared_scope_like\n'),
        ('fused-memory/src/fused_memory/middleware/task_curator.py', "SLICE = 'declared_scope'\n"),
    ],
)
def test_a_lookalike_or_an_allowed_reader_is_not_a_mention(relpath: str, source: str) -> None:
    assert forbidden_reader_mentions([_probe(relpath, source)]) == {}
