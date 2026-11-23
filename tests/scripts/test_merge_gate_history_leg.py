"""Static guard: every merge-gate test that reads the checkout's git history is marked (task 7248).

``merge_gate_history_pure`` licenses the merge gate to forgo the full re-verify
of a rebased tree byte-identical to the one that passed, running only the tests
``merge_gate_history_test_command`` selects. That is sound only if every gate
test whose verdict can change under a new parent (one that runs ``git log``,
``merge-base``, ``rev-list`` or ``describe`` against the real checkout) carries
the ``reads_git_history`` marker the leg selects.

The detector, the accounting rule and its registry ``READS_ONLY_THROWAWAY_REPOS``
live in ``tests/scripts/git_history_census.py``. This file runs that census over
every gate test root at once, holds the registry to no stale or doubled entry,
and checks the declared leg against the marked tests it must run.

The sweep reads the disk rather than a pytest collection, for the reason
``orchestrator/tests/test_marker_registration_drift.py`` records: a
FILE_SCOPED run of this file would otherwise check nothing else. The gate's
test roots come from the production module-config walk, so a new gate package
is covered without an edit here, and every pytest segment is read through
``verify_cmd.parse_config_command``, the parse the verify runner applies, so
this guard keeps no pytest flag grammar of its own.
"""
from __future__ import annotations

import dataclasses
import functools
import posixpath
import tomllib
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest
import verify_command_invariants as vci
from _pytest.mark.expression import Expression
from git_history_census import (
    MARKER,
    READS_ONLY_THROWAWAY_REPOS,
    GateTestRoot,
    Sweep,
    sweep,
    unaccounted_importer_failure,
    unaccounted_reader_failure,
)
from orchestrator.config import ModuleConfig, OrchestratorConfig, _discover_module_configs
from orchestrator.pytest_markers import locate_pytest_config, resolve_marker_expression
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from orchestrator import verify_cmd

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).parents[2]
_SWEEP_GROUP = pytest.mark.xdist_group('merge_gate_history_leg_sweep')

# ─── The gate's test roots ──────────────────────────────────────────────────


def structured_pytest(segment: str) -> verify_cmd.VerifyCmd:
    """*segment* as the verify runner parses it, refused unless it is one structured pytest run."""
    parsed = verify_cmd.parse_config_command(segment)
    assert parsed.tool is verify_cmd.ToolKind.PYTEST and parsed.raw is None, (
        f'the verify runner cannot structure {segment!r} as one pytest invocation, so this '
        f'guard cannot read which tests it collects'
    )
    return parsed


def pytest_segment_roots(
    segment: str, read: Callable[[str], str | None],
) -> list[GateTestRoot]:
    """The test roots one ``uv run ... pytest <targets>`` segment collects."""
    parsed = structured_pytest(segment)
    directory = parsed.cwd_rel or '.'
    paths = [posixpath.normpath(posixpath.join(directory, target)) for target in parsed.targets]
    config = locate_pytest_config(posixpath.commonpath(paths), read) if paths else None
    return [GateTestRoot(path=path, pytest_config=config) for path in paths]


def _read_repo_file(rel_path: str) -> str | None:
    path = REPO_ROOT / rel_path
    return path.read_text() if path.is_file() else None


def gate_test_roots(module_configs: Mapping[str, ModuleConfig]) -> frozenset[GateTestRoot]:
    roots: set[GateTestRoot] = set()
    for config in module_configs.values():
        segment = vci.optional_token_segment(config.test_command or '', vci.PYTEST)
        if segment is not None:
            roots.update(pytest_segment_roots(segment, _read_repo_file))
    return frozenset(roots)


@functools.cache
def _live_sweep() -> Sweep:
    return sweep(gate_test_roots(_discover_module_configs(REPO_ROOT)))


def _registered_markers(pyproject_text: str) -> frozenset[str]:
    table = tomllib.loads(pyproject_text).get('tool', {}).get('pytest', {}).get('ini_options', {})
    return frozenset(
        line.split(':', 1)[0].split('(', 1)[0].strip() for line in table.get('markers', [])
    )


# ─── The declared leg ───────────────────────────────────────────────────────


def selects(expression: str, item_markers: Iterable[str]) -> bool:
    """Whether pytest's own ``-m`` evaluator selects an item carrying exactly *item_markers*."""
    present = frozenset(item_markers)
    return Expression.compile(expression).evaluate(lambda name, **_: name in present)


@dataclasses.dataclass(frozen=True)
class LegSegment:
    command: str
    roots: tuple[GateTestRoot, ...]

    @property
    def pytest_config(self) -> str | None:
        return self.roots[0].pytest_config if self.roots else None

    def config_text(self) -> str:
        assert self.pytest_config is not None, f'no pytest config governs {self.command!r}'
        return (REPO_ROOT / self.pytest_config).read_text()

    def marker_expression(self) -> str | None:
        return resolve_marker_expression(None, self.command)

    def addopts_deselected_markers(self) -> frozenset[str]:
        text = self.config_text()
        addopts = resolve_marker_expression(text, None)
        if addopts is None or not selects(addopts, ()):
            return frozenset()
        return frozenset(name for name in _registered_markers(text) if not selects(addopts, {name}))

    def timeout_secs(self) -> float | None:
        flags = structured_pytest(self.command).base_flags
        for index, flag in enumerate(flags):
            if flag.startswith('--timeout='):
                return float(flag.split('=', 1)[1])
            if flag == '--timeout' and index + 1 < len(flags):
                return float(flags[index + 1])
        return None


def leg_segments(command: str) -> list[LegSegment]:
    segments = verify_cmd.split_and_chain_segments(command)
    assert segments is not None, (
        f'merge_gate_history_test_command is not an `&&` chain the verify runner can split: '
        f'{command!r}'
    )
    for segment in segments:
        assert segment.cwd_rel == '.', (
            f'{segment.command!r} runs from {segment.cwd_rel!r}; name its member with '
            f'`uv run --directory <member>` instead of a `cd`'
        )
        assert vci.optional_token_segment(segment.command, vci.PYTEST) == segment.command, (
            f'each history leg segment must be one pytest invocation: {segment.command!r}'
        )
    return [
        LegSegment(
            command=segment.command,
            roots=tuple(pytest_segment_roots(segment.command, _read_repo_file)),
        )
        for segment in segments
    ]


# ─── The gate's test roots, pinned on synthetic segments ────────────────────


def test_a_member_segment_is_rooted_at_its_directory_and_reads_its_own_config() -> None:
    files = {'pyproject.toml': '[tool.pytest.ini_options]\n', 'pkg/pyproject.toml': (
        '[tool.pytest.ini_options]\n'
    )}
    roots = pytest_segment_roots(
        "uv run --directory pkg pytest tests/ -m 'not integration' --tb=short -q --timeout=300",
        files.get,
    )
    assert roots == [GateTestRoot(path='pkg/tests', pytest_config='pkg/pyproject.toml')]


def test_a_root_segment_names_each_target_and_reads_the_root_config() -> None:
    files = {'pyproject.toml': '[tool.pytest.ini_options]\n'}
    roots = pytest_segment_roots(
        'uv run --project shared pytest tests/scripts/ scripts/tests/ --timeout=300 '
        '-n auto --dist loadgroup',
        files.get,
    )
    assert roots == [
        GateTestRoot(path='tests/scripts', pytest_config='pyproject.toml'),
        GateTestRoot(path='scripts/tests', pytest_config='pyproject.toml'),
    ]


def test_a_segment_the_verify_runner_cannot_structure_is_refused() -> None:
    with pytest.raises(AssertionError, match='cannot structure'):
        pytest_segment_roots('uv run pytest tests/ | tee log', {}.get)


# ─── The live tree ──────────────────────────────────────────────────────────


@_SWEEP_GROUP
def test_the_sweep_reaches_every_gate_package() -> None:
    roots = {root.path for root in _live_sweep().roots}
    assert {'orchestrator/tests', 'fused-memory/tests', 'dashboard/tests', 'tests/scripts'} <= roots
    assert len(_live_sweep().files) > 1000, len(_live_sweep().files)


@_SWEEP_GROUP
def test_every_history_reader_in_the_gate_is_accounted_for() -> None:
    failure = unaccounted_reader_failure(_live_sweep())
    assert failure is None, failure


@_SWEEP_GROUP
def test_every_test_file_importing_a_history_helper_is_accounted_for() -> None:
    failure = unaccounted_importer_failure(_live_sweep())
    assert failure is None, failure


@_SWEEP_GROUP
def test_every_history_reader_is_accounted_for_exactly_once() -> None:
    live = _live_sweep()
    doubled = {
        swept.rel_path: ways
        for swept in live.files
        if swept.calls and len(ways := live.ways_accounted_for(swept)) > 1
    }
    assert not doubled, doubled


@_SWEEP_GROUP
def test_every_throwaway_repo_entry_still_reaches_a_history_call() -> None:
    live = _live_sweep()
    files = live.by_path()
    stale = [
        rel_path for rel_path in READS_ONLY_THROWAWAY_REPOS
        if rel_path not in files or not live.reaches_a_history_call(files[rel_path])
    ]
    assert not stale, (
        f'READS_ONLY_THROWAWAY_REPOS entries that no longer name a gate test file making a '
        f'history call or importing a history helper: {stale}. Remove them.'
    )


@_SWEEP_GROUP
def test_no_file_is_both_marked_and_allowlisted() -> None:
    files = _live_sweep().by_path()
    both = [
        rel_path for rel_path in READS_ONLY_THROWAWAY_REPOS
        if rel_path in files and files[rel_path].carries_marker
    ]
    assert not both, both


@_SWEEP_GROUP
def test_every_package_owning_a_marked_test_registers_the_marker() -> None:
    configs = {swept.root.pytest_config for swept in _live_sweep().marked}
    unregistered = sorted(
        str(config) for config in configs
        if config is None
        or not config.endswith('.toml')
        or MARKER not in _registered_markers((REPO_ROOT / config).read_text())
    )
    assert not unregistered, (
        f'these pytest configs own a test marked {MARKER} but do not register it in '
        f'[tool.pytest.ini_options] markers: {unregistered}'
    )


def test_pytest_s_evaluator_reads_the_leg_expression_closed_world() -> None:
    expression = f'{MARKER} and not integration'
    assert selects(expression, {MARKER})
    assert not selects(expression, ())
    assert not selects(expression, {MARKER, 'integration'})


def test_a_leg_segment_reads_its_own_marker_expression_and_timeout() -> None:
    segment = LegSegment(
        command=f"uv run --directory pkg pytest tests/ -m '{MARKER}' --timeout=300", roots=(),
    )
    assert segment.marker_expression() == MARKER
    assert segment.timeout_secs() == 300
    assert LegSegment(command='uv run pytest tests/ --timeout 90', roots=()).timeout_secs() == 90


@pytest.fixture
def declared_leg(root_config: OrchestratorConfig) -> list[LegSegment]:
    command = root_config.merge_gate_history_test_command
    assert isinstance(command, str) and command.strip(), command
    return leg_segments(command)


def test_dark_factory_declares_a_history_pure_gate_with_a_history_leg(
    root_config: OrchestratorConfig,
) -> None:
    assert root_config.merge_gate_history_pure is True
    command = root_config.merge_gate_history_test_command
    assert isinstance(command, str) and command.strip(), command


@_SWEEP_GROUP
def test_the_leg_runs_exactly_the_roots_that_own_marked_tests(
    declared_leg: list[LegSegment],
) -> None:
    for segment in declared_leg:
        assert segment.roots, f'{segment.command!r} names no test root'
    leg_roots = {root for segment in declared_leg for root in segment.roots}
    marked_roots = {swept.root for swept in _live_sweep().marked}
    assert leg_roots - marked_roots == set(), (
        f'leg segments over roots with no {MARKER} test exit 5 at the gate and block every '
        f'same-tree merge: {sorted(leg_roots - marked_roots, key=str)}'
    )
    assert marked_roots - leg_roots == set(), (
        f'roots owning a {MARKER} test that no leg segment runs, so a same-tree merge never '
        f'checks them: {sorted(marked_roots - leg_roots, key=str)}'
    )


def test_each_leg_segment_selects_only_history_readers_its_member_would_run(
    declared_leg: list[LegSegment],
) -> None:
    for segment in declared_leg:
        expression = segment.marker_expression()
        assert expression is not None, f'{segment.command!r} carries no -m expression'
        assert selects(expression, {MARKER}), (segment.command, expression)
        assert not selects(expression, ()), (segment.command, expression)
        for deselected in segment.addopts_deselected_markers():
            assert not selects(expression, {MARKER, deselected}), (
                f'{segment.command!r}: its -m replaces addopts\' -m, so it must still '
                f'deselect {deselected!r}'
            )


def test_each_leg_segment_carries_a_gate_pytest_timeout(declared_leg: list[LegSegment]) -> None:
    for segment in declared_leg:
        timeout = segment.timeout_secs()
        assert timeout is not None and timeout > 60, (segment.command, timeout)
