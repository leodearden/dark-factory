"""Tests for audit_wrong_binding_edges.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors ``tests/test_audit_unverified_completion_claims.py``
/ ``test_audit_found_on_main_provenance.py`` / ``test_audit_duplicate_memories.py``.

WHAT THIS SUITE IS PROTECTING, beyond the detector's arithmetic: the script is
the RETROSPECTIVE counterpart of the write-time referent guard, and it must
IMPORT its detection vocabulary from
``fused_memory.utils.canonical_labels`` rather than re-derive it. INV-5 (task
3667) makes that a structural invariant — ``tests/test_canonical_labels.py``
and ``tests/test_task_naming.py`` already pin that no second compiled copy of
"what a task label is" may exist — so this module pins it for the script too,
mechanically (:class:`TestNoSecondVocabulary`), not merely by convention.
"""
from __future__ import annotations

import ast
import importlib.util
import types
from pathlib import Path

import pytest

from fused_memory.utils.canonical_labels import Referent

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'audit_wrong_binding_edges.py'


def _load_module() -> types.ModuleType:
    """Load audit_wrong_binding_edges.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'audit_wrong_binding_edges'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
fact_referents = _mod.fact_referents

GRAPH = 'reify'


def _task(number: str) -> Referent:
    """The own-project task referent for *number*, spelled once."""
    return Referent(kind='task', number=number)


class TestFactReferents:
    """The fact-side extractor: what task ids does this fact NAME?

    Every case below is drawn from the LIVE corpus, not invented — the
    specimens are the ones esc-4639-1 and task 4717 name, plus the four
    false positives a hand-rolled ``Task (\\d+)`` prototype produced during
    planning and the shared scanner does not.
    """

    def test_names_the_single_task_it_mentions(self) -> None:
        """Specimen 63fa5c78's fact, verbatim (edge bound to node 'Task 6165')."""
        assert fact_referents(
            'Task 6164 described landing the same artefact.', GRAPH
        ) == frozenset({_task('6164')})

    def test_a_bare_number_is_not_a_referent(self) -> None:
        """'Ruling 6164' is a bare number, invisible to the shared scanner.

        Only the 'task 6164' spelling contributes, and the two collapse onto
        ONE referent — the scan de-duplicates on (kind, project_id, number).
        """
        assert fact_referents(
            "Ruling 6164's HALF 2 was described in task 6164.", GRAPH
        ) == frozenset({_task('6164')})

    def test_the_hyphen_spelling_is_a_documented_blind_spot(self) -> None:
        """'task-1836' is NOT seen, and this script may not patch around it.

        ``canonical_labels._LOCAL_MENTION_PATTERN`` requires whitespace, '#'
        or ':' as the separator, so the hyphen spelling is invisible by
        design. Widening it is an edit to canonical_labels at its single
        site — which would also change the LIVE write-time guard's behaviour
        — and is out of scope here. The detector absorbs the recall loss
        through ``bare_id_present`` instead (see TestBareIdPresent), which
        mints no referent and can only SUPPRESS a flag.
        """
        assert fact_referents(
            'Task 1841 found the real SIGHUP bug that '
            "task-1836's timeout widening had masked.",
            GRAPH,
        ) == frozenset({_task('1841')})

    def test_a_fact_naming_no_task_yields_nothing(self) -> None:
        assert (
            fact_referents(
                'The merge worker holds .git/index.lock for the whole hook run.',
                GRAPH,
            )
            == frozenset()
        )

    def test_a_commit_sha_and_a_line_pin_contribute_nothing(self) -> None:
        """Neither a bare sha nor 'foo.rs:1794' may mint a referent.

        The line pin is the interesting half: ``_QUALIFIED_REF_PATTERN``'s
        lookbehind refuses a qualifier glued to path punctuation, and its
        qualifier must be >=3 characters — so neither 'rs' nor 'py' can
        become a project id.
        """
        assert (
            fact_referents(
                'commit e6a7e971ed touched foo.rs:1794 and bar.py:22.', GRAPH
            )
            == frozenset()
        )

    def test_multiple_referents_are_all_returned(self) -> None:
        """Specimen 8a51e13b's fact — it names BOTH 6126 and 6080."""
        assert fact_referents(
            'Task 6126 is landing to remove the last admission of '
            "dimensionless in the transform family, which task 6080's "
            'decision addresses.',
            GRAPH,
        ) == frozenset({_task('6126'), _task('6080')})

    @pytest.mark.parametrize('fact', ['', None])
    def test_empty_content_is_tolerated(self, fact: str | None) -> None:
        """A NULL ``r.fact`` column must not abort a whole-corpus sweep."""
        assert fact_referents(fact, GRAPH) == frozenset()

    def test_a_foreign_ref_keeps_its_qualifier(self) -> None:
        """'dark_factory:2500' read inside the reify graph is FOREIGN.

        Never flattened onto a bare 'Task 2500' — that collapse is the bug
        ``utils/cross_project_refs.py`` exists to detect, and the detector
        compares FULL Referents so the two can never be confused.
        """
        assert fact_referents(
            'Ported from dark_factory:2500 into this tree.', GRAPH
        ) == frozenset({Referent(kind='task', project_id='dark_factory', number='2500')})


class TestNoSecondVocabulary:
    """INV-5, as a test: the script compiles NO task-label pattern of its own.

    ``fused_memory/utils/canonical_labels.py`` is the single normative site
    for "what a task label is" (task 3667). Before it existed the vocabulary
    lived as separate compiled copies in ``utils/task_naming.py`` and
    ``utils/cross_project_refs.py``, and they had ALREADY drifted — one was
    structurally unable to see 'task #1153', the other had grown a
    colon-spelled mention pattern the first never got.

    A retrospective sweep is exactly the place a second copy would reappear,
    because the shared scanner's precision-over-recall narrowings are
    inconvenient for a survey. The narrow escape hatch this script DOES take
    (``bare_id_present``) is a containment check over an ALREADY-PARSED id
    and compiles no vocabulary — which is why the assertion below is about
    regex literals rather than about the absence of ``re`` entirely.
    """

    def test_no_compiled_regex_mentions_the_word_task(self) -> None:
        """Structural: no ``re.compile(...)`` literal contains 'task'.

        Asserted over the AST rather than as a substring scan, so the module
        docstring may name the hazard in order to warn against it — the same
        reason the precedent's mutation-call test went AST-based.
        """
        tree = ast.parse(SCRIPT_PATH.read_text())
        offenders: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, 'attr', None) or getattr(node.func, 'id', None)
            if name != 'compile':
                continue
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and 'task' in arg.value.lower()
                ):
                    offenders.append(f'{arg.value!r} (line {node.lineno})')
        assert offenders == [], (
            f'the script compiles a task-label pattern of its own: {offenders}. '
            'The vocabulary lives at exactly one site — '
            'fused_memory.utils.canonical_labels — and must be IMPORTED.'
        )

    def test_the_shared_scanner_is_imported(self) -> None:
        """Behavioural twin of the structural test above.

        Absence of a second regex proves nothing on its own — a script that
        parsed ids with ``str.split()`` would pass it. This pins that the
        script actually routes through the shared module.
        """
        imported = {
            alias.name
            for node in ast.walk(ast.parse(SCRIPT_PATH.read_text()))
            if isinstance(node, ast.ImportFrom)
            and node.module == 'fused_memory.utils.canonical_labels'
            for alias in node.names
        }
        assert 'scan_content' in imported
