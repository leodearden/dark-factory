"""Census: every gate test file under the named roots that shells a git history subcommand is accounted for.

A gate test that runs ``git log``, ``merge-base``, ``rev-list`` or ``describe``
against the real checkout must carry the :data:`MARKER` the merge gate's history
leg selects. No static signature separates a checkout read from a read of a
throwaway fixture repo, so the census fails CLOSED: every gate file that shells
a history subcommand must be accounted for in exactly one way. It carries the
marker, it is a helper module imported by a marked test file, or it is listed in
:data:`READS_ONLY_THROWAWAY_REPOS` with the reason every call it makes targets a
tmp or fixture repo. A helper's history call is invisible from the test files
that call into it. Listing the helper vouches for every importer. A marked
importer vouches only for itself, so once a helper is accounted for that way
each other test file importing it is held to the same choice: marked, or listed
with the reason none of its calls into the helper reads the checkout. A test
that reaches the checkout's history only through production code, or through a
helper's own helper, makes no call this census can see; mark it by hand.

The registry is held to the same rule from the other side: an entry under the
swept roots must still name a file that reaches a history call, and no file may
be accounted for in more than one way. An entry outside every swept root is left
to the sweep whose roots hold it.

RUN AS A SCRIPT, ``python tests/scripts/git_history_census.py ROOT...`` sweeps
the ``.py`` files under each root of the checkout holding this file: exit 0
silently when every verdict holds, 1 with a report when one does not, and 2
when a root is not a directory under the checkout or a file cannot be read or
an admitted one parsed, since then the census is incomplete. Each module's
``lint_command`` runs it over that module's gate test roots through
``uv run --project shared python``, because it reads markers through
``shared.pytest_mark_grammar``. ``tests/scripts/test_merge_gate_history_leg.py``
runs the same verdicts over every gate root at once, and holds every registry
entry to lie under one.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import functools
import posixpath
import re
import sys
import types
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath

from shared.pytest_mark_grammar import mark_elements, marker_name, pytestmark_value

CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
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
    'orchestrator/tests/_landing_git_scenarios.py': 'LandingRepo wraps the tmp_path git_repo fixture',
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
    'orchestrator/tests/test_git_ops_stacked_worktree.py': _SEEDS_ITS_OWN_REPO,
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
    'orchestrator/tests/test_merge_queue_undead_speculative_base.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_merge_speculation.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_plan_tools_server.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebase_branch_reset_guard.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebase_drops.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebase_drops_wiring.py': _SEEDS_ITS_OWN_REPO,
    'orchestrator/tests/test_rebase_recovery.py': _SEEDS_ITS_OWN_REPO,
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
    'orchestrator/tests/test_warm_lane_resume_foreign_head.py': _SEEDS_ITS_OWN_REPO,
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
    'tests/scripts/test_git_history_census.py': (
        "this census's own synthetic sources: string literals it parses, never runs"
    ),
    'tests/scripts/test_merge_marker_search_snippets.py': (
        'the snippets run against the shadowed_history repo under tmp_path'
    ),
    'tests/scripts/test_unblock_skill_done_provenance_templates.py': (
        'the matches are skill-prose fixtures it scans; no git runs'
    ),
}

CENSUS_FILE = Path(__file__).resolve().relative_to(CHECKOUT_ROOT).as_posix()
_REGISTRY_HOME = f'READS_ONLY_THROWAWAY_REPOS in {CENSUS_FILE}'
_GIT_OPTIONS_TAKING_A_VALUE = frozenset({'-C', '-c'})
_SHELLED_HISTORY_CALL = re.compile(r'git (-C \S+ )?(log|merge-base|rev-list|describe)\b')


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
    """The swept files, and a read-only copy of the throwaway-repo listing they are accounted against."""

    roots: frozenset[GateTestRoot]
    files: tuple[SweptFile, ...]
    throwaway: Mapping[str, str] = dataclasses.field(hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'throwaway', types.MappingProxyType(dict(self.throwaway)))

    def lies_under_a_root(self, rel_path: str) -> bool:
        return any(PurePosixPath(rel_path).is_relative_to(root.path) for root in self.roots)

    @functools.cached_property
    def marked(self) -> tuple[SweptFile, ...]:
        return tuple(swept for swept in self.files if swept.carries_marker)

    @functools.cached_property
    def history_helpers(self) -> tuple[SweptFile, ...]:
        """Non-test modules whose history calls run on whatever repo an importer hands them."""
        return tuple(
            swept for swept in self.files
            if swept.is_history_module and swept.rel_path not in self.throwaway
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

    def unaccounted_readers(self) -> list[SweptFile]:
        """Each file making a history call that nothing accounts for."""
        return [swept for swept in self.files if swept.calls and not self.ways_accounted_for(swept)]

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
            and importer.rel_path not in self.throwaway
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
        if swept.rel_path in self.throwaway:
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


class UnreadableGateFile(Exception):
    """A gate file the sweep had to read or parse and could not, so the census is incomplete."""

    def __init__(self, source: SweepSource, cause: Exception) -> None:
        super().__init__(
            f'{source.rel_path} could not be read or parsed by the history-reader sweep: '
            f'{cause!r}. A skipped file would make this census vacuous for it.'
        )


def _text_of(source: SweepSource) -> str:
    try:
        return source.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise UnreadableGateFile(source, exc) from exc


def _tree_of(source: SweepSource, text: str) -> ast.Module:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        raise UnreadableGateFile(source, exc) from exc


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


def sweep_sources(
    roots: frozenset[GateTestRoot],
    sources: Sequence[SweepSource],
    throwaway: Mapping[str, str] = READS_ONLY_THROWAWAY_REPOS,
) -> Sweep:
    scanned = [_scan_for_history(source) for source in sources]
    history_modules = frozenset(swept.stem for swept in scanned if swept.is_history_module)
    files = tuple(
        _with_imported_history_modules(source, swept, history_modules)
        for source, swept in zip(sources, scanned, strict=True)
    )
    return Sweep(roots=roots, files=files, throwaway=throwaway)


def sweep(
    roots: frozenset[GateTestRoot],
    repo_root: Path = CHECKOUT_ROOT,
    throwaway: Mapping[str, str] = READS_ONLY_THROWAWAY_REPOS,
) -> Sweep:
    return sweep_sources(roots, [
        SweepSource(
            rel_path=path.relative_to(repo_root).as_posix(), root=root,
            read=functools.partial(path.read_text, encoding='utf-8'),
        )
        for root in sorted(roots, key=lambda root: root.path)
        for path in sorted((repo_root / root.path).rglob('*.py'))
    ], throwaway)


# ─── The verdicts ───────────────────────────────────────────────────────────

_REQUEST_THE_REGISTRY = (
    f'If {CENSUS_FILE} is outside your task\'s declared files, request it with '
    f"escalate_blocker(category='scope_violation')"
)
_REGISTRY_SCOPE_ROUTE = f'{_REQUEST_THE_REGISTRY}.'
_SCOPE_ROUTE = (
    f'{_REQUEST_THE_REGISTRY}; do not mark a throwaway-repo reader to stay in scope, '
    f'since the mark puts the file on the merge gate\'s history leg.'
)


def unaccounted_reader_failure(census: Sweep) -> str | None:
    """The report on each file making a history call that nothing accounts for, else None."""
    unaccounted = [
        f'  {swept.rel_path}:{swept.calls[0].line}: `git {swept.calls[0].subcommand}`'
        for swept in census.unaccounted_readers()
    ]
    if not unaccounted:
        return None
    return (
        f'{len(unaccounted)} merge-gate test file(s) shell a git history subcommand but are '
        f'not accounted for:\n' + '\n'.join(unaccounted) + '\n'
        f'Remedies, one per file: (1) mark every test that reads the REAL checkout\'s '
        f'history with @pytest.mark.{MARKER}, at the narrowest scope covering them; '
        f'(2) for a helper module, import it from a marked test file, then account for every '
        f'other test file importing it; (3) if every history call targets a tmp or fixture '
        f'repo, add the file to {_REGISTRY_HOME} with that reason. {_SCOPE_ROUTE} '
        f'Listing a helper accounts for every test file importing it; do not list its '
        f'importers too, or they read as stale entries.'
    )


def unaccounted_importer_failure(census: Sweep) -> str | None:
    """The report on each unaccounted importer of a helper a marked importer accounts for, else None."""
    unaccounted = [
        f'  {importer.rel_path} imports {helper.rel_path}, which runs '
        f'`git {helper.calls[0].subcommand}` at line {helper.calls[0].line}'
        for helper, importer in census.unaccounted_helper_importers()
    ]
    if not unaccounted:
        return None
    return (
        f'{len(unaccounted)} merge-gate test file(s) import a helper that shells a git history '
        f'subcommand but are not accounted for:\n' + '\n'.join(unaccounted) + '\n'
        f'The history call lives in the helper, and a marked test file importing it vouches '
        f'only for itself. If every caller hands the helper a tmp or fixture repo, unmark '
        f'its marked importers and list the helper alone in {_REGISTRY_HOME}. '
        f'Otherwise, remedies, one per file: (1) mark every test that hands the helper '
        f'the REAL checkout with @pytest.mark.{MARKER}; (2) if every call into the helper '
        f'targets a tmp or fixture repo, or none reaches its history call, add the file to '
        f'{_REGISTRY_HOME} with that reason. {_SCOPE_ROUTE}'
    )


def stale_throwaway_entry_failure(census: Sweep) -> str | None:
    """The report on each listed path under the swept roots that reaches no history call, else None."""
    files = census.by_path()
    stale = [
        f'  {rel_path}'
        for rel_path in sorted(census.throwaway)
        if census.lies_under_a_root(rel_path)
        and (rel_path not in files or not census.reaches_a_history_call(files[rel_path]))
    ]
    if not stale:
        return None
    return (
        f'{len(stale)} {_REGISTRY_HOME} entry(ies) no longer name a gate test file that makes '
        f'a history call or imports a history helper:\n' + '\n'.join(stale) + '\n'
        f'Remove each. A deleted, renamed or split file takes its entry with it; any file '
        f'that still shells a history call is accounted for on its own. {_REGISTRY_SCOPE_ROUTE}'
    )


def doubly_accounted_failure(census: Sweep) -> str | None:
    """The report on each swept file accounted for in more than one way, else None."""
    doubled = [
        f'  {swept.rel_path}: {"; ".join(ways)}'
        for swept in census.files
        if len(ways := census.ways_accounted_for(swept)) > 1
    ]
    if not doubled:
        return None
    return (
        f'{len(doubled)} merge-gate file(s) are accounted for in more than one way:\n'
        + '\n'.join(doubled) + '\n'
        f'Keep exactly one. If every history call the file reaches targets a tmp or fixture '
        f'repo, keep its entry in {_REGISTRY_HOME} and drop the mark (for a helper, the mark on '
        f'an importer that carries it only for the helper); otherwise drop the entry, since '
        f'the mark already accounts for the file. {_REGISTRY_SCOPE_ROUTE}'
    )


_VERDICTS: tuple[Callable[[Sweep], str | None], ...] = (
    unaccounted_reader_failure,
    unaccounted_importer_failure,
    stale_throwaway_entry_failure,
    doubly_accounted_failure,
)


# ─── The command line ───────────────────────────────────────────────────────


def _is_directory_under(checkout: Path, directory: str) -> bool:
    candidate = (checkout / directory).resolve()
    return candidate.is_dir() and candidate.is_relative_to(checkout.resolve())


def main(argv: Sequence[str]) -> int:
    """Census the gate test roots *argv* names; return the exit code."""
    parser = argparse.ArgumentParser(
        description=(
            'Report gate test files that shell a git history subcommand unaccounted for, '
            'and stale or doubled accounting.'
        ),
    )
    parser.add_argument('roots', nargs='+', metavar='ROOT', help='relative to the checkout')
    parser.add_argument(
        '--checkout', type=Path, default=CHECKOUT_ROOT,
        help='the checkout the roots are under (default: the one holding this file)',
    )
    args = parser.parse_args(argv)
    checkout: Path = args.checkout
    not_directories = [repr(root) for root in args.roots if not _is_directory_under(checkout, root)]
    if not_directories:
        print(f'not a directory under {checkout}: {", ".join(not_directories)}', file=sys.stderr)
        return 2
    roots = frozenset(
        GateTestRoot(path=Path(root).as_posix(), pytest_config=None) for root in args.roots
    )
    try:
        census = sweep(roots, repo_root=checkout)
    except UnreadableGateFile as exc:
        print(f'the census is incomplete: {exc}', file=sys.stderr)
        return 2
    failures = [failure for verdict in _VERDICTS if (failure := verdict(census)) is not None]
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
