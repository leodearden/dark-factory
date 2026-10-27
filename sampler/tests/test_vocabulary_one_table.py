"""Guard for the design invariant `sampler-vocabulary-one-table`.

The invariant is stated in docs/legibility/design-invariants.md: the load
sampler's metric vocabulary has one declaring table,
``sampler.metrics.METRIC_GROUPS``, and every other spelling is a projection
of it. The guard has two parts. The behavioural part adds a spec to an
injected table and watches it reach every projection. The structural backstop
scans the vocabulary's consumers for a string constant equal to a table
metric name. Both are exercised on hand-written fixtures before the live scan
runs.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from collections.abc import Iterable
from pathlib import Path
from typing import TypeGuard

from sampler.metrics import (
    METRIC_GROUPS,
    GroupName,
    MetricGroup,
    MetricSpec,
    arm_metrics,
    exact_metric_names,
)
from sampler.sampler import run_tick
from sampler.store import LoadSampleStore

REPO_ROOT = Path(__file__).resolve().parents[2]


def scanned_names(groups: Iterable[MetricGroup]) -> frozenset[str]:
    """Every table metric name, less those that are also one of the table's gate arms.

    The exclusion is derived, never hand-listed: a metric named like an arm
    (``runqueue_ratio``) is legitimately spelled wherever that arm is. The
    table's arms are reconciled with shared.psi's by
    sampler/tests/test_load_metrics.py::TestArmMetricStemParity.
    """
    groups = tuple(groups)
    return frozenset(spec.name for group in groups for spec in group.metrics) - set(
        arm_metrics(groups)
    )


def literal_metric_names(
    source: str, *, names: frozenset[str], declares_specs: bool
) -> list[tuple[int, str]]:
    """``(lineno, value)`` of each string constant in *source* that is one of *names*.

    Docstrings and other bare string statements are skipped. When
    *declares_specs* is set, so is the name argument of each ``MetricSpec(...)``
    call, which is where the table's own declarations spell their names.
    """
    tree = ast.parse(source)
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            skipped.add(id(node.value))
        elif declares_specs and _is_spec_call(node):
            skipped.update(id(arg) for arg in node.args[:1])
            skipped.update(id(kw.value) for kw in node.keywords if kw.arg == 'name')
    return sorted(
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in names
        and id(node) not in skipped
    )


def _is_spec_call(node: ast.AST) -> TypeGuard[ast.Call]:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == MetricSpec.__name__
    )


# ---------------------------------------------------------------------------
# Behavioural: a spec added to the table reaches every projection
# ---------------------------------------------------------------------------

NEW = MetricSpec('fixture_new_metric', arm='fixture_new_arm')


def _table_with(spec: MetricSpec, into: GroupName = GroupName.PROCESS) -> tuple[MetricGroup, ...]:
    """METRIC_GROUPS with *spec* added to one group, whose collector emits only it."""

    def emits_only_the_new_spec():
        return {spec.key(): 1.0}

    return tuple(
        dataclasses.replace(
            group, metrics=(*group.metrics, spec), collect=emits_only_the_new_spec
        )
        if group.name is into
        else dataclasses.replace(group, collect=dict)
        for group in METRIC_GROUPS
    )


class TestASpecAddedToTheTableReachesEveryProjection:
    def test_run_tick_admits_and_stores_it(self, tmp_path: Path):
        import sqlite3

        table = _table_with(NEW)
        db_path = tmp_path / 'db.sqlite'

        written = run_tick(
            LoadSampleStore(db_path), 1_000_000, {group: group.collect() for group in table}
        )

        assert len(written) == len(table), 'a group was dropped at admission'
        conn = sqlite3.connect(str(db_path))
        try:
            stored = conn.execute('SELECT metric, leaf FROM samples').fetchall()
        finally:
            conn.close()
        assert stored == [('fixture_new_metric', None)]

    def test_its_arm_appears_in_arm_metrics(self):
        assert arm_metrics(_table_with(NEW))['fixture_new_arm'] is NEW

    def test_without_an_arm_it_adds_no_arm(self):
        armless = dataclasses.replace(NEW, arm=None)

        assert dict(arm_metrics(_table_with(armless))) == dict(arm_metrics(METRIC_GROUPS))

    def test_it_is_served_exactly_when_its_group_is(self):
        table = _table_with(NEW, into=GroupName.PROCESS)

        assert NEW.name in exact_metric_names(table, {GroupName.PROCESS})
        assert NEW.name not in exact_metric_names(table, {GroupName.PSI})


# ---------------------------------------------------------------------------
# Structural backstop, on hand-written fixture modules
# ---------------------------------------------------------------------------

_FIXTURE_TABLE = (
    MetricGroup(
        GroupName.LOAD,
        collect=dict,
        metrics=(
            MetricSpec('fixture_alpha'),
            MetricSpec('fixture_beta', per_leaf=True),
            MetricSpec('fixture_shared_with_an_arm', arm='fixture_shared_with_an_arm'),
        ),
        windowed=True,
    ),
)
_FIXTURE_NAMES = scanned_names(_FIXTURE_TABLE)

_SEEDED_MODULE = '''\
"""fixture_alpha is described here, which is prose, not a spelling."""

ADMITTED = ('fixture_alpha', 'something_else')
LADDERS = {
    'fixture_beta': (1.0, 2.0),
}
ARM = 'fixture_shared_with_an_arm'
'''

_SPEC_DECLARATIONS = "SPEC = MetricSpec('fixture_alpha')\nOTHER = MetricSpec(name='fixture_beta')\n"


class TestTheScanOnFixtureModules:
    def test_a_seeded_module_is_flagged_at_its_code_lines_only(self):
        found = literal_metric_names(_SEEDED_MODULE, names=_FIXTURE_NAMES, declares_specs=False)

        assert found == [(3, 'fixture_alpha'), (5, 'fixture_beta')]

    def test_a_name_that_is_also_an_arm_is_not_scanned_for(self):
        assert 'fixture_shared_with_an_arm' not in _FIXTURE_NAMES
        assert {'fixture_alpha', 'fixture_beta'} <= _FIXTURE_NAMES

    def test_a_spec_declaration_is_exempt_where_specs_are_declared(self):
        found = literal_metric_names(_SPEC_DECLARATIONS, names=_FIXTURE_NAMES, declares_specs=True)

        assert found == []

    def test_a_spec_declaration_is_flagged_anywhere_else(self):
        found = literal_metric_names(_SPEC_DECLARATIONS, names=_FIXTURE_NAMES, declares_specs=False)

        assert found == [(1, 'fixture_alpha'), (2, 'fixture_beta')]

    def test_a_module_spelling_no_name_flags_nothing(self):
        source = "LEVELS = ('low', 'high')\n"

        assert literal_metric_names(source, names=_FIXTURE_NAMES, declares_specs=False) == []


# ---------------------------------------------------------------------------
# Structural backstop, live
# ---------------------------------------------------------------------------

_LIVE_NAMES = scanned_names(METRIC_GROUPS)
_SPEC_HOME = Path(inspect.getsourcefile(MetricSpec) or '').resolve()


def _scope() -> list[tuple[Path, bool]]:
    """Every consumer of the vocabulary, and whether it is the one declaring specs."""
    sampler_modules = sorted((REPO_ROOT / 'sampler' / 'src' / 'sampler').glob('*.py'))
    assert _SPEC_HOME in sampler_modules, (
        f'MetricSpec is defined in {_SPEC_HOME}, outside the scanned sampler package; '
        'the test is reading another checkout'
    )
    others = [
        REPO_ROOT / 'dashboard' / 'src' / 'dashboard' / 'data' / 'load.py',
        REPO_ROOT / 'scripts' / 'load-threshold-calibration.py',
    ]
    return [(path, path == _SPEC_HOME) for path in sampler_modules] + [
        (path, False) for path in others
    ]


class TestTheLiveTree:
    def test_the_table_module_spells_each_scanned_name_exactly_once(self):
        """Non-vacuity: scanned without its exemption, the scan sees every declaration."""
        found = literal_metric_names(
            _SPEC_HOME.read_text(encoding='utf-8'), names=_LIVE_NAMES, declares_specs=False
        )

        assert sorted(value for _line, value in found) == sorted(_LIVE_NAMES), found

    def test_no_consumer_spells_a_metric_name(self):
        findings = [
            f'{path.relative_to(REPO_ROOT)}:{line}:{value}'
            for path, declares_specs in _scope()
            for line, value in literal_metric_names(
                path.read_text(encoding='utf-8'), names=_LIVE_NAMES, declares_specs=declares_specs
            )
        ]

        assert findings == [], (
            'metric names spelled outside sampler.metrics.METRIC_GROUPS; derive them '
            'from the table instead:\n' + '\n'.join(findings)
        )
