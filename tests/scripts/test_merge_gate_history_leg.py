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
targets a tmp or fixture repo. A helper's history call is invisible from the
test files that call into it. Listing the helper vouches for every importer.
A marked importer vouches only for itself, so once a helper is accounted for
that way each other test file importing it is held to the same choice:
marked, or listed with the reason none of its calls into the helper reads the
checkout. A test that reaches the checkout's history
only through production code, or through a helper's own helper, makes no call
this sweep can see; mark it by hand.

The sweep reads the disk rather than a pytest collection, for the reason
``orchestrator/tests/test_marker_registration_drift.py`` records: a
FILE_SCOPED run of this file would otherwise check nothing else. The gate's
test roots come from the production module-config walk, so a new gate package
is covered without an edit here, and every pytest segment is read through
``verify_cmd.parse_config_command``, the parse the verify runner applies, so
this guard keeps no pytest flag grammar of its own.
"""
from __future__ import annotations

import ast
import dataclasses
import functools
import posixpath
import re
import tomllib
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import pytest
import verify_command_invariants as vci
from _pytest.mark.expression import Expression
from orchestrator.config import ModuleConfig, OrchestratorConfig, _discover_module_configs
from orchestrator.pytest_markers import locate_pytest_config, resolve_marker_expression
from shared.pytest_mark_grammar import mark_elements, marker_name, pytestmark_value
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from orchestrator import verify_cmd

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).parents[2]
MARKER = 'reads_git_history'
HISTORY_SUBCOMMANDS = frozenset({'log', 'merge-base', 'rev-list', 'describe'})

_SEEDS_ITS_OWN_REPO = 'every history call runs in a tmp_path repo or worktree the test seeds'
_PARSES_CACHE_BUSTER_TEXT_ONLY = (
    'imports only redux_cache_buster_versions from _cache_buster_helpers, a text parse; '
    'no git runs'
)

READS_ONLY_THROWAWAY_REPOS: Mapping[str, str] = {
    'dashboard/tests/test_cache_buster_freshness.py': (
        'resolve_redux_base_state runs only on the repos _init_repo builds under tmp_path'
    ),
    'dashboard/tests/test_tab_tasks_deferral.py': _PARSES_CACHE_BUSTER_TEXT_ONLY,
    'dashboard/tests/test_tab_tasks_invariant_chip.py': _PARSES_CACHE_BUSTER_TEXT_ONLY,
    'escalation/tests/_rebased_landing_git.py': (
        'every builder takes a caller-built repo root and caps discovery there'
    ),
    'escalation/tests/test_merge_status_git_authority.py': _SEEDS_ITS_OWN_REPO,
    'fused-memory/tests/server/test_topic_cluster_store.py': (
        'the match is memory-text prose passed to pytest.param; no git runs'
    ),
    'fused-memory/tests/test_live_workflow_detector.py': (
        'run_git is patched; each git argv is a mocked CompletedProcess or side-effect input'
    ),
    'orchestrator/tests/_dispatch_base_scene.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_atomic_train_merge.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_branch_base.py': _SEEDS_ITS_OWN_REPO,
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
    'orchestrator/tests/test_info_l0_router_boundary.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_landlock.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_chain_unstack.py': 'GitOps over a tmp_path repo',
    'orchestrator/tests/test_merge_gate_already_on_main.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_gates_equivalence_rename.py': 'GitOps over a tmp_path repo',
    'orchestrator/tests/test_merge_gates_plan_files_rename.py': (
        'a predicate over recorded argv; the probe it intercepts runs in the tmp git_repo fixture'
    ),
    'orchestrator/tests/test_merge_gates_rename_family.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_lane_lease_at_verdict.py': (
        'GitOps over the tmp_path git_repo fixture'
    ),
    'orchestrator/tests/test_merge_lane_ratchet_commit_gate.py': (
        '_Repo.seeded builds its repo under tmp_path'
    ),
    'orchestrator/tests/test_merge_queue.py': 'GitOps over the tmp_path git_repo fixture',
    'orchestrator/tests/test_merge_queue_build_chain.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_queue_chain_intact.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_queue_deep_dispatch.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_queue_deep_integration_gate.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_queue_deep_landing.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_queue_deep_reservation.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_speculation.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_plan_tools_server.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebase_branch_reset_guard.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebase_drops.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebase_drops_wiring.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebased_solo_landing_ladder.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_reconcile_done_step_commits.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_recut_ruling_gitops.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_review_inputs.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_roles_already_done_commit_search.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_roles_ancestry_check.py': 'every call runs in a clone under tmp_path',
    'orchestrator/tests/test_spawn_budget_guard.py': (
        'create_subprocess_exec is faked; the git argv is a recorded-call literal'
    ),
    'orchestrator/tests/test_train_moved_main.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_verify_events_units.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_verify_phase_rebase.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_warm_lane_abort_teardown.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_warm_lane_integration_gate.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_warm_lane_pool.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_workflow_e2e.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_workflow_helpers.py': _SEEDS_ITS_OWN_REPO,
    'scripts/tests/test_flip_reify_gate_exclude_heavy.py': (
        'every call runs in the fake reify repo under tmp_path'
    ),
    'scripts/tests/test_legibility_nightly.py': _SEEDS_ITS_OWN_REPO,
    'scripts/tests/test_memory_metadata_coverage_census_wrapper.py': (
        'every call runs in a fake, decoy or outer repo under tmp_path'
    ),
    'scripts/tests/test_merge_deep_set_cap.py': _SEEDS_ITS_OWN_REPO,
    'scripts/tests/test_merge_pytest_n_ab_switch.py': _SEEDS_ITS_OWN_REPO,
    'scripts/tests/test_merge_queue_landing.py': _SEEDS_ITS_OWN_REPO,
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


# ─── The text prefilter ─────────────────────────────────────────────────────

_GIT_OR_MARKER_NAMES = ('git', MARKER)
_ESCAPE_SPELLING_A_LETTER_OF_GIT = re.compile(
    r'\\(?:x6[79]|x74|u00(?:6[79]|74)|U000000(?:6[79]|74)|1(?:47|51)|164|N|\r|\n)'
)
_BETWEEN_ADJACENT_LITERALS = r"""['"](?:[\s'"rRuUfF]|\\\n|#[^\n]*)*['"]"""
_GIT_SPLIT_ACROSS_ADJACENT_LITERALS = re.compile(
    f'g{_BETWEEN_ADJACENT_LITERALS}it'
    f'|gi{_BETWEEN_ADJACENT_LITERALS}t'
    f'|g{_BETWEEN_ADJACENT_LITERALS}i{_BETWEEN_ADJACENT_LITERALS}t'
)


def _mentions_any(source: str, words: Iterable[str]) -> bool:
    """Whether a word is in *source* as written or as the parser compares identifiers (NFKC)."""
    words = tuple(words)
    if any(word in source for word in words):
        return True
    return not source.isascii() and any(
        word in unicodedata.normalize('NFKC', source) for word in words
    )


def may_hold_a_history_call_or_marker(source: str) -> bool:
    """False proves :func:`history_calls` and :func:`applied_marker_names` find nothing in *source*.

    Every detected call needs the word ``git`` (an argv's first element, a helper callee
    ``git``/``*_git``, a shelled ``git log ...`` string) and every marker the name
    :data:`MARKER`, which contains it. A callee or marker is an identifier, so the word is
    in the NFKC-normalised text. A string constant is not normalised but can be written
    without the contiguous word: through an escape that spells one of its letters
    (``'\\x67it'``), a backslash-newline continuation, or adjacent literals (``'g' 'it'``).
    Each such form admits the file, so the filter only ever over-admits.
    """
    return (
        _mentions_any(source, _GIT_OR_MARKER_NAMES)
        or _ESCAPE_SPELLING_A_LETTER_OF_GIT.search(source) is not None
        or _GIT_SPLIT_ACROSS_ADJACENT_LITERALS.search(source) is not None
    )


def may_import_any_of(source: str, stems: Iterable[str]) -> bool:
    """False proves *source* imports none of *stems*: an import names each stem as an identifier."""
    return _mentions_any(source, stems)


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


# ─── The sweep ──────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SweptFile:
    """What the sweep learned of one gate file.

    ``imported_history_modules`` holds only the stems of the sweep's history-calling
    non-test modules that this file imports, and is empty for a non-test file: the one
    question asked of imports is which test files import a history helper.
    """

    rel_path: str
    root: GateTestRoot
    calls: tuple[HistoryCall, ...]
    carries_marker: bool
    imported_history_modules: frozenset[str]

    @property
    def is_test_module(self) -> bool:
        return posixpath.basename(self.rel_path).startswith('test_')

    @property
    def stem(self) -> str:
        return posixpath.splitext(posixpath.basename(self.rel_path))[0]

    @property
    def is_history_module(self) -> bool:
        return bool(self.calls) and not self.is_test_module


@dataclasses.dataclass(frozen=True)
class Sweep:
    roots: frozenset[GateTestRoot]
    files: tuple[SweptFile, ...]

    @functools.cached_property
    def marked(self) -> tuple[SweptFile, ...]:
        return tuple(swept for swept in self.files if swept.carries_marker)

    @functools.cached_property
    def history_helpers(self) -> tuple[SweptFile, ...]:
        """Non-test modules whose history calls run on whatever repo an importer hands them."""
        return tuple(
            swept for swept in self.files
            if swept.is_history_module and swept.rel_path not in READS_ONLY_THROWAWAY_REPOS
        )

    def importing_test_files(self, helper: SweptFile) -> tuple[SweptFile, ...]:
        assert helper.is_history_module, (
            f'{helper.rel_path} makes no history call, so the sweep never resolved who imports it'
        )
        return tuple(
            swept for swept in self.files
            if swept.is_test_module and swept.root == helper.root
            and helper.stem in swept.imported_history_modules
        )

    def is_helper_of_a_marked_file(self, swept: SweptFile) -> bool:
        return swept.is_history_module and any(
            importer.carries_marker for importer in self.importing_test_files(swept)
        )

    def unaccounted_helper_importers(self) -> list[tuple[SweptFile, SweptFile]]:
        """Each importer, neither marked nor listed, of a helper a marked importer accounts for.

        A helper nothing accounts for is the history-reader check's to report; its importers
        wait on that choice, since listing the helper accounts for all of them.
        """
        return [
            (helper, importer)
            for helper in self.history_helpers
            if self.is_helper_of_a_marked_file(helper)
            for importer in self.importing_test_files(helper)
            if not importer.carries_marker
            and importer.rel_path not in READS_ONLY_THROWAWAY_REPOS
        ]

    def reaches_a_history_call(self, swept: SweptFile) -> bool:
        return bool(swept.calls) or any(
            swept in self.importing_test_files(helper) for helper in self.history_helpers
        )

    def ways_accounted_for(self, swept: SweptFile) -> list[str]:
        ways = []
        if swept.carries_marker:
            ways.append(f'marked {MARKER}')
        if self.is_helper_of_a_marked_file(swept):
            ways.append('a helper imported by a marked test file')
        if swept.rel_path in READS_ONLY_THROWAWAY_REPOS:
            ways.append('listed in READS_ONLY_THROWAWAY_REPOS')
        return ways

    def by_path(self) -> dict[str, SweptFile]:
        return {swept.rel_path: swept for swept in self.files}


@dataclasses.dataclass(frozen=True)
class SweepSource:
    """A gate file to sweep and how to read its text, which the sweep reads once per phase."""

    rel_path: str
    root: GateTestRoot
    read: Callable[[], str]


def text_source(rel_path: str, root: GateTestRoot, text: str) -> SweepSource:
    return SweepSource(rel_path=rel_path, root=root, read=lambda: text)


def _unreadable_file_error(source: SweepSource, exc: Exception) -> AssertionError:
    return AssertionError(
        f'{source.rel_path} could not be read or parsed by the history-reader sweep: {exc!r}. '
        f'A skipped file would make this guard vacuous for it.'
    )


def _text_of(source: SweepSource) -> str:
    try:
        return source.read()
    except OSError as exc:
        raise _unreadable_file_error(source, exc) from exc


def _tree_of(source: SweepSource, text: str) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        raise _unreadable_file_error(source, exc) from exc


def _scan_for_history(source: SweepSource) -> SweptFile:
    """Phase 1: the file's history calls and marker, parsed only if its text can hold one."""
    text = _text_of(source)
    calls: tuple[HistoryCall, ...] = ()
    carries_marker = False
    if may_hold_a_history_call_or_marker(text):
        tree = _tree_of(source, text)
        calls = tuple(history_calls(tree))
        carries_marker = _mentions_any(text, (MARKER,)) and MARKER in applied_marker_names(tree)
    return SweptFile(
        rel_path=source.rel_path, root=source.root, calls=calls, carries_marker=carries_marker,
        imported_history_modules=frozenset(),
    )


def _with_imported_history_modules(
    source: SweepSource, scanned: SweptFile, history_modules: frozenset[str],
) -> SweptFile:
    """Phase 2: a test file's imports of *history_modules*, parsed only if its text names one."""
    if not scanned.is_test_module:
        return scanned
    text = _text_of(source)
    if not may_import_any_of(text, history_modules):
        return scanned
    imported = imported_module_stems(_tree_of(source, text)) & history_modules
    return dataclasses.replace(scanned, imported_history_modules=imported)


def sweep_sources(roots: frozenset[GateTestRoot], sources: Sequence[SweepSource]) -> Sweep:
    scanned = [_scan_for_history(source) for source in sources]
    history_modules = frozenset(swept.stem for swept in scanned if swept.is_history_module)
    files = tuple(
        _with_imported_history_modules(source, swept, history_modules)
        for source, swept in zip(sources, scanned, strict=True)
    )
    return Sweep(roots=roots, files=files)


def sweep(roots: frozenset[GateTestRoot], repo_root: Path = REPO_ROOT) -> Sweep:
    return sweep_sources(roots, [
        SweepSource(rel_path=path.relative_to(repo_root).as_posix(), root=root, read=path.read_text)
        for root in sorted(roots, key=lambda root: root.path)
        for path in sorted((repo_root / root.path).rglob('*.py'))
    ])


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



def test_a_segment_the_verify_runner_cannot_structure_is_refused() -> None:
    with pytest.raises(AssertionError, match='cannot structure'):
        pytest_segment_roots('uv run pytest tests/ | tee log', {}.get)


_SYNTHETIC_ROOT = GateTestRoot(path='pkg/tests', pytest_config='pkg/pyproject.toml')
_HISTORY_HELPER_SOURCE = "def base(root):\n    return _git(root, 'merge-base', 'main', 'HEAD')\n"


def _swept(*sources: SweepSource) -> Sweep:
    return sweep_sources(frozenset(source.root for source in sources), sources)


def test_a_file_that_never_spells_git_is_still_found_as_an_importer() -> None:
    helper = text_source('pkg/tests/_history.py', _SYNTHETIC_ROOT, _HISTORY_HELPER_SOURCE)
    importer = text_source(
        'pkg/tests/test_versions.py', _SYNTHETIC_ROOT, 'from _history import versions\n',
    )
    synthetic = _swept(helper, importer)
    swept_helper, swept_importer = synthetic.files

    assert swept_importer.calls == ()
    assert synthetic.importing_test_files(swept_helper) == (swept_importer,)


def test_a_marked_importer_of_a_history_helper_does_not_vouch_for_an_unmarked_one() -> None:
    other_root = GateTestRoot(path='other/tests', pytest_config=None)
    synthetic = _swept(
        text_source('pkg/tests/_history.py', _SYNTHETIC_ROOT, _HISTORY_HELPER_SOURCE),
        text_source(
            'pkg/tests/test_reads.py', _SYNTHETIC_ROOT,
            f'import pytest\nfrom _history import base\npytestmark = pytest.mark.{MARKER}\n',
        ),
        text_source(
            'pkg/tests/test_also_reads.py', _SYNTHETIC_ROOT, 'from _history import base\n',
        ),
        text_source('other/tests/test_other.py', other_root, 'from _history import base\n'),
    )
    files = synthetic.by_path()
    helper, unmarked = files['pkg/tests/_history.py'], files['pkg/tests/test_also_reads.py']
    elsewhere = files['other/tests/test_other.py']

    assert synthetic.ways_accounted_for(helper) == ['a helper imported by a marked test file']
    assert synthetic.unaccounted_helper_importers() == [(helper, unmarked)]
    assert synthetic.reaches_a_history_call(unmarked)
    assert not synthetic.reaches_a_history_call(elsewhere)


def test_the_importers_of_a_helper_nothing_accounts_for_wait_on_the_helper() -> None:
    synthetic = _swept(
        text_source('pkg/tests/_history.py', _SYNTHETIC_ROOT, _HISTORY_HELPER_SOURCE),
        text_source('pkg/tests/test_reads.py', _SYNTHETIC_ROOT, 'from _history import base\n'),
    )
    helper = synthetic.by_path()['pkg/tests/_history.py']

    assert synthetic.ways_accounted_for(helper) == []
    assert synthetic.unaccounted_helper_importers() == []


def test_importers_are_resolved_only_for_a_module_that_makes_a_history_call() -> None:
    synthetic = _swept(
        text_source('pkg/tests/_plain.py', _SYNTHETIC_ROOT, 'VALUE = 1\n'),
        text_source('pkg/tests/test_uses.py', _SYNTHETIC_ROOT, 'from _plain import VALUE\n'),
    )
    with pytest.raises(AssertionError, match='makes no history call'):
        synthetic.importing_test_files(synthetic.by_path()['pkg/tests/_plain.py'])


# ─── The prefilter must not blind the guard ─────────────────────────────────

_SPELLINGS_OF_A_HISTORY_CALL = [
    pytest.param("subprocess.run(['git', 'log', '-1'])", 'log', id='argv-list'),
    pytest.param("subprocess.run(('git', '-C', str(root), 'merge-base', 'a', 'b'))", 'merge-base', id='argv-tuple'),
    pytest.param("_git(root, 'rev-list', 'HEAD')", 'rev-list', id='underscore-helper'),
    pytest.param("def f(self):\n    return self._git('describe', '--tags')\n", 'describe', id='method'),
    pytest.param("run_git(root, 'log')", 'log', id='suffixed-helper'),
    pytest.param("git(root, 'log')", 'log', id='bare-helper'),
    pytest.param("run('git log -1 --format=%H')", 'log', id='shelled'),
    pytest.param("run(f'git -C {root} merge-base HEAD main')", 'merge-base', id='shelled-in-f-string'),
    pytest.param("\uff47\uff49\uff54(root, 'log')", 'log', id='full-width-helper-identifier'),
    pytest.param("run_\uff47\uff49\uff54(root, 'log')", 'log', id='full-width-helper-suffix'),
    pytest.param(r"subprocess.run(['\x67it', 'log'])", 'log', id='escaped-argv-program'),
    pytest.param(r"subprocess.run(['\u0067it', 'log'])", 'log', id='unicode-escaped-argv-program'),
    pytest.param(r"subprocess.run(['\147it', 'log'])", 'log', id='octal-escaped-argv-program'),
    pytest.param(r"subprocess.run(['\N{LATIN SMALL LETTER G}it', 'log'])", 'log', id='named-escape'),
    pytest.param("subprocess.run(['g' 'it', 'log'])", 'log', id='adjacent-literals-argv'),
    pytest.param("subprocess.run(['gi'\n    't', 'log'])", 'log', id='adjacent-literals-across-lines'),
    pytest.param(
        "subprocess.run(['g'  # the program\n    'i'\n    't', 'log'])", 'log',
        id='adjacent-literals-around-a-comment',
    ),
    pytest.param("run('g' 'it log -1')", 'log', id='shelled-across-adjacent-literals'),
    pytest.param("run(f'g' f'it log {x}')", 'log', id='shelled-across-adjacent-f-strings'),
    pytest.param("run('g\\\nit log -1')", 'log', id='shelled-across-a-line-continuation'),
    pytest.param("run(r'g' 'it log')", 'log', id='raw-prefixed-adjacent-literal'),
]


@pytest.mark.parametrize(('source', 'subcommand'), _SPELLINGS_OF_A_HISTORY_CALL)
def test_the_real_sweep_reports_a_history_call_however_it_is_spelled(
    source: str, subcommand: str, tmp_path: Path,
) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/_reads.py').write_text(source)

    (swept,) = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path).files

    assert [call.subcommand for call in swept.calls] == [subcommand]
    assert swept.calls == tuple(history_calls(ast.parse(source)))


def _escapes_spelling(letter: str) -> list[str]:
    code = ord(letter)
    return [
        f'\\x{code:02x}', f'\\u{code:04x}', f'\\U{code:08x}', f'\\{code:o}',
        f'\\N{{LATIN SMALL LETTER {letter.upper()}}}',
    ]


@pytest.mark.parametrize('position', range(len('git')))
def test_the_real_sweep_reports_a_call_whose_program_is_escaped_in_any_one_letter(
    position: int, tmp_path: Path,
) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    for index, escape in enumerate(_escapes_spelling('git'[position])):
        program = 'git'[:position] + escape + 'git'[position + 1:]
        source = f"subprocess.run(['{program}', 'log'])\n"
        (tmp_path / f'pkg/tests/_reads_{index}.py').write_text(source)

    swept = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path)

    assert [[call.subcommand for call in file.calls] for file in swept.files] == [['log']] * 5


@pytest.mark.parametrize(
    'marker_spelling',
    [
        pytest.param(f'pytest.mark.{MARKER}', id='attribute'),
        pytest.param(f'pytest.mark.{MARKER}()', id='call'),
        pytest.param(f'pytest.mark.{MARKER.replace("r", chr(0xFF52), 1)}', id='full-width-identifier'),
    ],
)
def test_the_real_sweep_sees_the_marker_in_a_file_that_makes_no_call(
    marker_spelling: str, tmp_path: Path,
) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/test_marked.py').write_text(
        f'import pytest\npytestmark = {marker_spelling}\n'
    )

    (swept,) = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path).files

    assert swept.calls == ()
    assert swept.carries_marker


def test_the_real_sweep_finds_every_importer_of_a_history_helper(tmp_path: Path) -> None:
    sources = {
        '_history.py': _HISTORY_HELPER_SOURCE,
        'test_from_import.py': 'from _history import base\n',
        'test_plain_import.py': 'import _history\n',
        'test_dotted_import.py': 'import pkg.tests._history as history\n',
        'test_nested_import.py': 'def test_it():\n    from . import _history\n',
        'test_full_width_import.py': 'from _\uff48istory import base\n',
        'test_other_helper.py': 'from _other import base\n',
        'test_mentions_only.py': '# _history is not imported here\nNAME = "_history"\n',
    }
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    for name, source in sources.items():
        (tmp_path / 'pkg/tests' / name).write_text(source)

    swept = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path)
    helper = swept.by_path()['pkg/tests/_history.py']

    assert sorted(importer.stem for importer in swept.importing_test_files(helper)) == [
        'test_dotted_import', 'test_from_import', 'test_full_width_import',
        'test_nested_import', 'test_plain_import',
    ]


def test_a_file_with_no_git_in_it_yields_no_call_and_is_not_parsed(tmp_path: Path) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/test_never_parsed.py').write_text('def broken(:\n')

    (swept,) = sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path).files

    assert swept.calls == ()
    assert not swept.carries_marker


def test_an_unparseable_file_that_may_hold_a_history_call_fails_the_sweep(tmp_path: Path) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/_reads.py').write_text("def broken(:\n    _git(root, 'log')\n")

    with pytest.raises(AssertionError, match='_reads.py could not be read or parsed'):
        sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path)


def test_an_unparseable_test_file_naming_a_history_helper_fails_the_sweep(tmp_path: Path) -> None:
    (tmp_path / 'pkg/tests').mkdir(parents=True)
    (tmp_path / 'pkg/tests/_history.py').write_text(_HISTORY_HELPER_SOURCE)
    (tmp_path / 'pkg/tests/test_uses.py').write_text('from _history import (\n')

    with pytest.raises(AssertionError, match='test_uses.py could not be read or parsed'):
        sweep(frozenset({_SYNTHETIC_ROOT}), repo_root=tmp_path)


@pytest.mark.parametrize(
    'source',
    [
        pytest.param('x = 1\n', id='nothing'),
        pytest.param('import os\nprint("a\\nb", os.sep)\n', id='harmless-escape'),
        pytest.param('names = ["msg", "item"]\n', id='strings-separated-by-a-comma'),
        pytest.param(r"print('\x1b[0m \u2014 \n')" + '\n', id='escapes-spelling-no-letter-of-git'),
        pytest.param('a = "msg"\nb = "item"\n', id='strings-on-separate-statements'),
        pytest.param('import pytest\npytestmark = pytest.mark.slow\n', id='another-marker'),
    ],
)
def test_a_source_that_cannot_spell_git_is_not_admitted(source: str) -> None:
    assert not may_hold_a_history_call_or_marker(source)


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
        f'(2) for a helper module, import it from a marked test file, then account for every '
        f'other test file importing it; (3) if every history '
        f'call targets a tmp or fixture repo, add the file to READS_ONLY_THROWAWAY_REPOS '
        f'with that reason. Listing a helper accounts for every test file importing it; do '
        f'not list its importers too, or they read as stale entries.'
    )


@_SWEEP_GROUP
def test_every_test_file_importing_a_history_helper_is_accounted_for() -> None:
    unaccounted = [
        f'  {importer.rel_path} imports {helper.rel_path}, which runs '
        f'`git {helper.calls[0].subcommand}` at line {helper.calls[0].line}'
        for helper, importer in _live_sweep().unaccounted_helper_importers()
    ]
    assert not unaccounted, (
        f'{len(unaccounted)} merge-gate test file(s) import a helper that shells a git history '
        f'subcommand but are not accounted for:\n' + '\n'.join(unaccounted) + '\n'
        f'The history call lives in the helper, and a marked test file importing it vouches '
        f'only for itself. If every caller hands the helper a tmp or fixture repo, unmark '
        f'its marked importers and list the helper alone in READS_ONLY_THROWAWAY_REPOS. '
        f'Otherwise, remedies, one per file: (1) mark every test that hands the helper '
        f'the REAL checkout with @pytest.mark.{MARKER}; (2) if every call into the helper '
        f'targets a tmp or fixture repo, or none reaches its history call, add the file to '
        f'READS_ONLY_THROWAWAY_REPOS with that reason.'
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
