"""Static guard: every merge-gate test that reads the checkout's git history is marked (task 7248).

``merge_gate_history_pure`` licenses the merge gate to forgo the full re-verify
of a rebased tree byte-identical to the one that passed, running only the tests
``merge_gate_history_test_command`` selects. That is sound only if every gate
test whose verdict can change under a new parent (one that runs ``git log``,
``merge-base``, ``rev-list`` or ``describe`` against the real checkout) carries
the ``reads_git_history`` marker the leg selects.

No static signature separates a checkout read from a read of a throwaway
fixture repo, so the sweep fails CLOSED: every gate test file that shells a
history subcommand must be accounted for in exactly one way. It carries the
marker, it is a helper module imported by a marked test file, or it is listed
in :data:`READS_ONLY_THROWAWAY_REPOS` with the reason every call it makes
targets a tmp or fixture repo. A test that reaches the checkout's history only
through production code makes no call this sweep can see; mark it by hand.

The sweep reads the disk rather than a pytest collection, for the reason
``orchestrator/tests/test_marker_registration_drift.py`` records: a
FILE_SCOPED run of this file would otherwise check nothing else. The gate's
test roots come from the production module-config walk, so a new gate package
is covered without an edit here.
"""
from __future__ import annotations

import ast
import dataclasses
import functools
import posixpath
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest
import verify_command_invariants as vci
from _pytest.mark.expression import Expression
from orchestrator.config import ModuleConfig, OrchestratorConfig, _discover_module_configs
from orchestrator.pytest_markers import locate_pytest_config, resolve_marker_expression
from shared.pytest_mark_grammar import mark_elements, marker_name, pytestmark_value

from orchestrator import verify_cmd

REPO_ROOT = Path(__file__).parents[2]
MARKER = 'reads_git_history'
HISTORY_SUBCOMMANDS = frozenset({'log', 'merge-base', 'rev-list', 'describe'})

_SEEDS_ITS_OWN_REPO = 'every history call runs in a tmp_path repo or worktree the test seeds'

READS_ONLY_THROWAWAY_REPOS: Mapping[str, str] = {
    'fused-memory/tests/server/test_topic_cluster_store.py': (
        'the match is memory-text prose passed to pytest.param; no git runs'
    ),
    'fused-memory/tests/test_live_workflow_detector.py': (
        'run_git is patched; each git argv is a mocked CompletedProcess or side-effect input'
    ),
    'orchestrator/tests/test_atomic_train_merge.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_branch_stack.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_branch_work_landed.py': '_Repo wraps the tmp_path git_repo fixture',
    'orchestrator/tests/test_citation_gate_positive_arm.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_convert_to_blocked.py': (
        'a predicate over recorded argv; the probe it intercepts runs in the tmp git_ops repo'
    ),
    'orchestrator/tests/test_git_ops.py': (
        'GitOps over the tmp_path git_repo fixture and its worktrees'
    ),
    'orchestrator/tests/test_git_ops_no_op_landing.py': '_Repo wraps the tmp_path git_repo fixture',
    'orchestrator/tests/test_git_ops_unstack.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_group_merge_candidate_ordering.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_harness_plan_step_rederive.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_harness_warm_lane_wiring.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_landlock.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_chain_unstack.py': 'GitOps over a tmp_path repo',
    'orchestrator/tests/test_merge_gate_already_on_main.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_gates_equivalence_rename.py': 'GitOps over a tmp_path repo',
    'orchestrator/tests/test_merge_gates_plan_files_rename.py': (
        'a predicate over recorded argv; the probe it intercepts runs in the tmp git_repo fixture'
    ),
    'orchestrator/tests/test_merge_lane_ratchet_commit_gate.py': (
        '_Repo.seeded builds its repo under tmp_path'
    ),
    'orchestrator/tests/test_merge_queue.py': 'GitOps over the tmp_path git_repo fixture',
    'orchestrator/tests/test_merge_queue_build_chain.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_queue_deep_dispatch.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_queue_deep_integration_gate.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_queue_deep_landing.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_speculation.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_plan_tools_server.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebase_branch_reset_guard.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebased_solo_landing_ladder.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_reconcile_done_step_commits.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_recut_ruling_gitops.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_roles_ancestry_check.py': 'every call runs in a clone under tmp_path',
    'orchestrator/tests/test_timeout_marker_inversion_guard.py': (
        'create_subprocess_exec is faked; the git argv is a recorded-call literal'
    ),
    'orchestrator/tests/test_train_moved_main.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_verify_phase_rebase.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_warm_lane_abort_teardown.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_warm_lane_integration_gate.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_warm_lane_pool.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_workflow_e2e.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_workflow_helpers.py': _SEEDS_ITS_OWN_REPO,
    'scripts/tests/test_flip_reify_gate_exclude_heavy.py': (
        'every call runs in the fake reify repo under tmp_path'
    ),
    'scripts/tests/test_install_memory_metadata_coverage_census_timer.py': (
        'every call runs in a fake, decoy or outer repo under tmp_path'
    ),
    'scripts/tests/test_legibility_nightly.py': _SEEDS_ITS_OWN_REPO,
    'scripts/tests/test_merge_deep_set_cap.py': _SEEDS_ITS_OWN_REPO,
    'scripts/tests/test_merge_pytest_n_ab_switch.py': _SEEDS_ITS_OWN_REPO,
    'shared/tests/test_merge_subject.py': _SEEDS_ITS_OWN_REPO,
    'tests/scripts/test_merge_gate_history_leg.py': (
        "this guard's own synthetic sources: string literals it parses, never runs"
    ),
    'tests/scripts/test_merge_marker_search_snippets.py': (
        'the snippets run against the shadowed_history repo under tmp_path'
    ),
    'tests/scripts/test_unblock_skill_done_provenance_templates.py': (
        'the matches are skill-prose fixtures it scans; no git runs'
    ),
}

_GIT_OPTIONS_TAKING_A_VALUE = frozenset({'-C', '-c'})
_SHELLED_HISTORY_CALL = re.compile(r'git (-C \S+ )?(log|merge-base|rev-list|describe)\b')
_PYTEST_VALUE_FLAGS = frozenset({
    '-c', '-k', '-m', '-n', '-o', '-p', '-r', '-W',
    '--basetemp', '--color', '--deselect', '--dist', '--durations',
    '--ignore', '--ignore-glob', '--import-mode', '--junitxml',
    '--log-level', '--maxfail', '--rootdir', '--tb', '--timeout',
})
_SWEEP_GROUP = pytest.mark.xdist_group('merge_gate_history_leg_sweep')


@dataclasses.dataclass(frozen=True)
class HistoryCall:
    line: int
    subcommand: str


@dataclasses.dataclass(frozen=True)
class GateTestRoot:
    """A directory one gate pytest invocation collects, and the config file that run reads."""

    path: str
    pytest_config: str | None


# ─── Detection ──────────────────────────────────────────────────────────────


def _string(node: ast.expr) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _rendered(node: ast.expr) -> str | None:
    """A string constant's text, or an f-string's with each placeholder as ``{}``."""
    if isinstance(node, ast.JoinedStr):
        return ''.join(_string(part) or '{}' for part in node.values)
    return _string(node)


def _git_subcommand(args: Iterable[ast.expr]) -> str | None:
    """The subcommand of a git argv that follows the program, past ``-C``/``-c`` options."""
    remaining = list(args)
    while remaining:
        word = _string(remaining.pop(0))
        if word is None:
            return None
        if word in _GIT_OPTIONS_TAKING_A_VALUE:
            remaining = remaining[1:]
        elif not word.startswith('-'):
            return word
    return None


def _is_git_helper(callee: ast.expr) -> bool:
    if isinstance(callee, ast.Name):
        name = callee.id
    elif isinstance(callee, ast.Attribute):
        name = callee.attr
    else:
        return False
    return name in ('git', '_git') or name.endswith('_git')


def _helper_call_subcommand(call: ast.Call) -> str | None:
    positional = [arg for arg in call.args if not isinstance(arg, ast.Starred)]
    first_word = next((i for i, arg in enumerate(positional) if _string(arg) is not None), None)
    return None if first_word is None else _git_subcommand(positional[first_word:])


def _subcommands_of(node: ast.expr) -> Iterable[str | None]:
    if isinstance(node, ast.List | ast.Tuple) and node.elts and _string(node.elts[0]) == 'git':
        yield _git_subcommand(node.elts[1:])
    if isinstance(node, ast.Call):
        if _is_git_helper(node.func):
            yield _helper_call_subcommand(node)
        for arg in node.args:
            text = _rendered(arg)
            if text is not None and (shelled := _SHELLED_HISTORY_CALL.search(text)):
                yield shelled.group(2)


def history_calls(tree: ast.Module) -> list[HistoryCall]:
    """Every place *tree* shells a git history subcommand, in line order."""
    found = {
        HistoryCall(line=node.lineno, subcommand=subcommand)
        for node in ast.walk(tree) if isinstance(node, ast.expr)
        for subcommand in _subcommands_of(node) if subcommand in HISTORY_SUBCOMMANDS
    }
    return sorted(found, key=lambda call: (call.line, call.subcommand))


def applied_marker_names(tree: ast.Module) -> frozenset[str]:
    """Every marker *tree* applies anywhere: decorators at any depth and every ``pytestmark``."""
    elements: list[ast.expr] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            elements.extend(node.decorator_list)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            value = pytestmark_value(node)
            if value is not None:
                elements.extend(mark_elements(value))
    return frozenset(name for name in map(marker_name, elements) if name is not None)


def imported_module_stems(tree: ast.Module) -> frozenset[str]:
    """The last dotted component of every module *tree* imports, at any depth."""
    stems: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            stems.update(alias.name.rsplit('.', 1)[-1] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                stems.add(node.module.rsplit('.', 1)[-1])
            stems.update(alias.name for alias in node.names)
    return frozenset(stems)


# ─── The gate's test roots ──────────────────────────────────────────────────


def _uv_directory(wrapper_tokens: list[str]) -> str:
    for index, token in enumerate(wrapper_tokens):
        if token == '--directory' and index + 1 < len(wrapper_tokens):
            return wrapper_tokens[index + 1]
        if token.startswith('--directory='):
            return token.split('=', 1)[1]
    return '.'


def pytest_segment_roots(
    segment: str, read: Callable[[str], str | None],
) -> list[GateTestRoot]:
    """The test roots one ``uv run ... pytest <targets>`` segment collects."""
    wrapper, _ = vci.anchor_split(segment, vci.PYTEST)
    directory = _uv_directory(wrapper)
    paths = [
        posixpath.normpath(posixpath.join(directory, target))
        for target in vci.positional_targets(segment, vci.PYTEST, value_flags=_PYTEST_VALUE_FLAGS)
    ]
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


# ─── The sweep ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SweptFile:
    rel_path: str
    root: GateTestRoot
    calls: tuple[HistoryCall, ...]
    markers: frozenset[str]
    imports: frozenset[str]

    @property
    def is_test_module(self) -> bool:
        return posixpath.basename(self.rel_path).startswith('test_')

    @property
    def stem(self) -> str:
        return posixpath.splitext(posixpath.basename(self.rel_path))[0]


@dataclasses.dataclass(frozen=True)
class Sweep:
    roots: frozenset[GateTestRoot]
    files: tuple[SweptFile, ...]

    @functools.cached_property
    def marked(self) -> tuple[SweptFile, ...]:
        return tuple(swept for swept in self.files if MARKER in swept.markers)

    def is_helper_of_a_marked_file(self, swept: SweptFile) -> bool:
        return not swept.is_test_module and any(
            marked.root == swept.root and swept.stem in marked.imports
            for marked in self.marked
        )

    def ways_accounted_for(self, swept: SweptFile) -> list[str]:
        ways = []
        if MARKER in swept.markers:
            ways.append(f'marked {MARKER}')
        if self.is_helper_of_a_marked_file(swept):
            ways.append('a helper imported by a marked test file')
        if swept.rel_path in READS_ONLY_THROWAWAY_REPOS:
            ways.append('listed in READS_ONLY_THROWAWAY_REPOS')
        return ways

    def by_path(self) -> dict[str, SweptFile]:
        return {swept.rel_path: swept for swept in self.files}


def _sweep_file(path: Path, root: GateTestRoot) -> SweptFile:
    rel_path = path.relative_to(REPO_ROOT).as_posix()
    try:
        source = path.read_text()
        if 'git' not in source:
            # Every detectable history call and every application of MARKER spells it.
            return SweptFile(
                rel_path=rel_path, root=root, calls=(), markers=frozenset(), imports=frozenset(),
            )
        tree = ast.parse(source)
    except (SyntaxError, OSError) as exc:
        raise AssertionError(
            f'{rel_path} could not be read or parsed by the history-reader sweep: {exc!r}. '
            f'A skipped file would make this guard vacuous for it.'
        ) from exc
    return SweptFile(
        rel_path=rel_path, root=root, calls=tuple(history_calls(tree)),
        markers=applied_marker_names(tree), imports=imported_module_stems(tree),
    )


def sweep(roots: frozenset[GateTestRoot]) -> Sweep:
    files = [
        _sweep_file(path, root)
        for root in sorted(roots, key=lambda root: root.path)
        for path in sorted((REPO_ROOT / root.path).rglob('*.py'))
    ]
    return Sweep(roots=roots, files=tuple(files))


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
        _, own_args = vci.anchor_split(self.command, vci.PYTEST)
        for index, token in enumerate(own_args):
            if token.startswith('--timeout='):
                return float(token.split('=', 1)[1])
            if token == '--timeout' and index + 1 < len(own_args):
                return float(own_args[index + 1])
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


# ─── The detector, pinned on synthetic sources ──────────────────────────────


@pytest.mark.parametrize(
    ('source', 'subcommand'),
    [
        pytest.param(
            "subprocess.run(['git', '-C', str(tmp_path), 'log', '-1'])", 'log',
            id='fixture-repo-argv',
        ),
        pytest.param(
            "subprocess.run(('git', '-C', str(REPO_ROOT), 'merge-base', 'HEAD', 'main'))",
            'merge-base', id='checkout-argv',
        ),
        pytest.param("_git(repo, 'rev-list', '--count', 'HEAD')", 'rev-list', id='git-helper'),
        pytest.param("repo.git('describe', '--tags')", 'describe', id='git-method'),
        pytest.param("_run_git(root, 'log', '--format=%H')", 'log', id='suffixed-helper'),
        pytest.param("subprocess.run('git log -1 --format=%H', shell=True)", 'log', id='shelled'),
        pytest.param("run(f'git -C {root} merge-base HEAD main')", 'merge-base', id='f-string'),
    ],
)
def test_a_history_call_is_detected(source: str, subcommand: str) -> None:
    assert [call.subcommand for call in history_calls(ast.parse(source))] == [subcommand]


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('"""Reads git log against the checkout."""', id='docstring'),
        pytest.param('x = 1  # git log --oneline', id='comment'),
        pytest.param("subprocess.run(['git', 'commit', '-m', 'log'])", id='message-argv'),
        pytest.param("_git(repo, 'commit', '-m', 'merge-base')", id='message-helper'),
        pytest.param("subprocess.run(['git', 'status'])", id='other-subcommand'),
        pytest.param("subprocess.run(['git', *args])", id='unknowable-argv'),
        pytest.param("MESSAGE = 'git log is a history read'", id='string-not-passed'),
    ],
)
def test_a_non_history_call_is_not_detected(source: str) -> None:
    assert history_calls(ast.parse(source)) == []


def test_a_detected_call_carries_its_line() -> None:
    source = "import subprocess\n\n\nsubprocess.run(['git', 'rev-list', 'HEAD'])\n"
    assert history_calls(ast.parse(source)) == [HistoryCall(line=4, subcommand='rev-list')]


def test_the_marker_is_seen_on_one_class_among_several() -> None:
    source = (
        'import pytest\n'
        'class TestTree:\n'
        '    def test_a(self): ...\n'
        f'@pytest.mark.{MARKER}\n'
        'class TestHistory:\n'
        '    def test_b(self): ...\n'
    )
    assert MARKER in applied_marker_names(ast.parse(source))


def test_the_marker_is_seen_in_a_pytestmark_list() -> None:
    source = f'import pytest\npytestmark = [pytest.mark.timeout(60), pytest.mark.{MARKER}]\n'
    assert MARKER in applied_marker_names(ast.parse(source))


def test_a_marker_named_only_in_a_string_is_not_applied() -> None:
    assert MARKER not in applied_marker_names(ast.parse(f"NOTE = 'pytest.mark.{MARKER}'\n"))


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('from _lme_artifacts import load', id='from-module'),
        pytest.param('import _lme_artifacts', id='import'),
        pytest.param('from . import _lme_artifacts', id='relative-submodule'),
        pytest.param('from tests.arm_harness._lme_artifacts import load', id='dotted'),
        pytest.param('def f():\n    import _lme_artifacts\n', id='nested'),
    ],
)
def test_a_helper_import_is_resolved_to_its_stem(source: str) -> None:
    assert '_lme_artifacts' in imported_module_stems(ast.parse(source))


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


# ─── The live tree ──────────────────────────────────────────────────────────


@_SWEEP_GROUP
def test_the_sweep_reaches_every_gate_package() -> None:
    roots = {root.path for root in _live_sweep().roots}
    assert {'orchestrator/tests', 'fused-memory/tests', 'dashboard/tests', 'tests/scripts'} <= roots
    assert len(_live_sweep().files) > 1000, len(_live_sweep().files)


@_SWEEP_GROUP
def test_every_history_reader_in_the_gate_is_accounted_for() -> None:
    live = _live_sweep()
    unaccounted = [
        f'  {swept.rel_path}:{swept.calls[0].line}: `git {swept.calls[0].subcommand}`'
        for swept in live.files
        if swept.calls and not live.ways_accounted_for(swept)
    ]
    assert not unaccounted, (
        f'{len(unaccounted)} merge-gate test file(s) shell a git history subcommand but are '
        f'not accounted for:\n' + '\n'.join(unaccounted) + '\n'
        f'Remedies, one per file: (1) mark every test that reads the REAL checkout\'s '
        f'history with @pytest.mark.{MARKER}, at the narrowest scope covering them; '
        f'(2) for a helper module, import it from a marked test file; (3) if every history '
        f'call targets a tmp or fixture repo, add the file to READS_ONLY_THROWAWAY_REPOS '
        f'with that reason.'
    )


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
def test_every_throwaway_repo_entry_still_reads_history() -> None:
    files = _live_sweep().by_path()
    stale = [
        rel_path for rel_path in READS_ONLY_THROWAWAY_REPOS
        if rel_path not in files or not files[rel_path].calls
    ]
    assert not stale, (
        f'READS_ONLY_THROWAWAY_REPOS entries that no longer name a gate test file making a '
        f'history call: {stale}. Remove them.'
    )


@_SWEEP_GROUP
def test_no_file_is_both_marked_and_allowlisted() -> None:
    files = _live_sweep().by_path()
    both = [
        rel_path for rel_path in READS_ONLY_THROWAWAY_REPOS
        if rel_path in files and MARKER in files[rel_path].markers
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
