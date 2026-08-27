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
endpoint_referent = _mod.endpoint_referent
bare_id_present = _mod.bare_id_present

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

    def test_the_anchored_parser_is_imported(self) -> None:
        """The endpoint side routes through the shared module too.

        Split from the scan_content assertion above because the two enter the
        script at different steps; keeping them apart makes a regression name
        which HALF of the vocabulary was re-derived.
        """
        imported = {
            alias.name
            for node in ast.walk(ast.parse(SCRIPT_PATH.read_text()))
            if isinstance(node, ast.ImportFrom)
            and node.module == 'fused_memory.utils.canonical_labels'
            for alias in node.names
        }
        assert 'parse_node_name' in imported


class TestEndpointReferent:
    """The endpoint-side parser: is this entity NAME a task label?

    A thin adapter over the IMPORTED ``canonical_labels.parse_node_name``,
    which is ANCHORED by design — it answers "is this NAME a task label",
    not "does this text mention a task" (the fact side answers that with the
    unanchored scanner). The anchoring is what keeps a name that merely
    CONTAINS a task reference out of the population entirely.
    """

    @pytest.mark.parametrize(
        ('name', 'number'),
        [
            ('Task 6165', '6165'),  # the canonical specimen's node
            ('task 4755', '4755'),
            ('task #1153', '1153'),  # one of the PRD's 53 measured variants
            ('Task: 132', '132'),
        ],
    )
    def test_task_shaped_names_parse(self, name: str, number: str) -> None:
        assert endpoint_referent(name) == Referent(kind='task', number=number)

    @pytest.mark.parametrize(
        'name',
        [
            'ElasticResult.rotation',  # live object-end of specimen 63fa5c78
            'IMPLEMENTATION COORDINATION',
            '6185 GUI-channel-bridge',  # bare digits: a shared blind spot
            'commit e6a7e971ed',
            'Dependencies 1720',
            'Task 42 orchestrator',  # a MENTION, not a label — anchoring matters
            '',
            None,
        ],
    )
    def test_non_labels_are_refused(self, name: str | None) -> None:
        """These must never enter the population.

        A None/empty ``a.name`` column is included deliberately: the sweep
        reads every live RELATES_TO row in the graph, and one odd historical
        node must not raise into a whole-corpus read.
        """
        assert endpoint_referent(name) is None

    def test_a_foreign_qualified_name_keeps_its_project(self) -> None:
        """'reify:132' is a DIFFERENT referent from a local 'Task 132'.

        Flattening the qualifier away is exactly the cross-project collapse
        ``utils/cross_project_refs.py`` exists to detect, so the detector
        must compare FULL Referents rather than bare numbers.
        """
        assert endpoint_referent('reify:132') == Referent(
            kind='task', project_id='reify', number='132'
        )

    def test_a_foreign_referent_never_equals_the_local_one(self) -> None:
        """The equality the detector's set-membership test relies on.

        Asserted on the Referent type itself rather than only through the
        detector, because this is the property that makes comparing FULL
        referents safe: a fact naming local 'Task 132' must NOT satisfy an
        endpoint named 'reify:132', and vice versa.
        """
        local = endpoint_referent('Task 132')
        foreign = endpoint_referent('reify:132')
        assert local != foreign
        assert local not in {foreign}
        assert foreign not in {local}
        assert local is not None and foreign is not None
        assert local.number == foreign.number  # the bare numbers DO collide


class TestBareIdPresent:
    """The containment backstop — the ONLY id check this script does itself.

    It is not a second vocabulary and cannot become one: it compiles no
    task-label pattern, it can MINT no referent, and it takes an id the
    shared parser has ALREADY produced. Its only power is to SUPPRESS a
    flag, which is the conservative direction for a report a human
    adjudicates by hand.

    It exists because two of the shared scanner's documented blind spots —
    '#4262' and 'task-1836' — would otherwise make an endpoint that IS named
    in the fact look unnamed, which is a false positive.
    """

    def test_the_bare_hash_spelling_is_seen(self) -> None:
        """'#4262' is invisible to the shared scanner; the digits are not."""
        assert bare_id_present(
            _task('4262'),
            "#4262's cache is separated from the engine-level tables in Task 4351.",
        )

    def test_the_hyphen_spelling_is_seen(self) -> None:
        assert bare_id_present(
            _task('1836'),
            'Task 1841 found the real SIGHUP bug that '
            "task-1836's timeout widening had masked.",
        )

    def test_an_absent_id_is_absent(self) -> None:
        assert not bare_id_present(
            _task('6165'), 'Task 6164 described landing the same artefact.'
        )

    def test_a_digit_run_must_stand_alone(self) -> None:
        """Word-boundary matched, so a SUBSTRING of a longer id never counts.

        Without this, endpoint 'Task 616' would read as named by a fact about
        task 6165 and the flag would be suppressed wrongly — silently
        under-reporting exactly the near-miss population this sweep measures.
        """
        assert not bare_id_present(_task('616'), 'Task 6165 landed.')
        assert not bare_id_present(_task('165'), 'Task 6165 landed.')

    def test_empty_content_is_tolerated(self) -> None:
        assert not bare_id_present(_task('6165'), '')
        assert not bare_id_present(_task('6165'), None)

    def test_a_foreign_referent_is_matched_on_its_number(self) -> None:
        """Deliberate and documented: containment sees DIGITS, not projects.

        A foreign endpoint 'reify:132' whose fact says 'task 132' is
        suppressed, because the containment check cannot tell the two apart.
        That is a suppression — a lost flag, never an invented one — and so
        it lands on the safe side of a report a human adjudicates. The
        PRECISE comparison still happens in the set-membership test, which
        compares full Referents; this is only the backstop.
        """
        assert bare_id_present(
            Referent(kind='task', project_id='reify', number='132'),
            'Ported from task 132 into this tree.',
        )
