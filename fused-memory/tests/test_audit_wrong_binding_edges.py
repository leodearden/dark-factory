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
classify_edge = _mod.classify_edge
Finding = _mod.Finding
# vars() does not work on a slots dataclass — the script's own accessor does.
vars_of = _mod.vars_of

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


#: The live specimens, as (edge_uuid, subject_name, object_name, fact) tuples.
#: Named constants rather than inline literals so the OUT-OF-REACH contract
#: below can pin the SAME rows the in-reach cases use, and so a reader can
#: cross-check any of them against the graph by uuid.
SPEC_63FA5C78 = (
    '63fa5c78',
    'Task 6165',
    'ElasticResult.rotation',
    'Task 6164 described landing the same artefact.',
)
SPEC_6AEFAC16 = (
    '6aefac16',
    'Task 3421',
    'commit e6a7e971ed',
    'Task 3429 made _check_scope_invariant directional as per commit e6a7e971ed.',
)
SPEC_8A51E13B = (
    '8a51e13b',
    'Task 6080',
    'Task 6128',
    'Task 6126 is landing to remove the last admission of dimensionless in '
    "the transform family, which task 6080's decision addresses.",
)
SPEC_1CF19488 = (
    '1cf19488',
    'Task 6346',
    'Task 6347',
    'The recurring-attention task #6347 depends on task #6346.',
)
SPEC_01E3D75E = (
    '01e3e75e',
    'Task 5997',
    'Task 6014',
    'Task 6014 carries task 5997 as a hard dependency.',
)
SPEC_993A9A7B = (
    '993a9a7b',
    'Task 6004',
    'Task 5997',
    "Task 6004's rulings were ported verbatim into task 5997.",
)


def _classify(spec: tuple[str, str, str, str], graph: str = GRAPH) -> list:
    """Run the detector over a specimen tuple."""
    edge_uuid, subject, obj, fact = spec
    return classify_edge(subject, obj, fact, edge_uuid, graph)


class TestClassifyEdge:
    """The Class-A detector: an endpoint the fact does not name.

    Mirrors the write-time guard's ``set-membership`` check
    (memory_service.py:3524-3529) deliberately, so the retrospective and live
    views of one defect cannot drift into two different verdicts.
    """

    def test_the_canonical_specimen_flags_its_subject(self) -> None:
        """Node 'Task 6165' carrying a fact about task 6164."""
        findings = _classify(SPEC_63FA5C78)
        assert len(findings) == 1
        (finding,) = findings
        assert finding.end == 'subject'
        assert finding.edge_uuid == '63fa5c78'
        assert finding.graph == GRAPH
        assert finding.node_name == 'Task 6165'
        assert finding.node_referent == _task('6165')
        assert set(finding.fact_referents) == {_task('6164')}

    def test_a_non_task_object_end_is_never_examined(self) -> None:
        """'ElasticResult.rotation' is not a task label, so it cannot be wrong.

        The detector only ever indicts an endpoint whose NAME claims to be a
        task; every other node is outside the question this sweep asks.
        """
        assert [f.end for f in _classify(SPEC_63FA5C78)] == ['subject']

    def test_the_dark_factory_specimen_flags_its_subject(self) -> None:
        """6aefac16: node 'Task 3421' carrying a fact about task 3429."""
        findings = _classify(SPEC_6AEFAC16, graph='dark_factory')
        assert len(findings) == 1
        assert findings[0].end == 'subject'
        assert findings[0].node_referent == _task('3421')
        assert findings[0].graph == 'dark_factory'

    def test_the_object_end_specimen_a_subject_only_detector_misses(self) -> None:
        """8a51e13b: (Task 6080)->(Task 6128), fact names 6126 and 6080.

        The SUBJECT is correctly named, so the detector the task description
        proposes — "fact names an id differing from the id in its SUBJECT
        node's name" — would report this edge clean. The mis-bound end is the
        OBJECT. That is why both endpoints are checked.
        """
        findings = _classify(SPEC_8A51E13B)
        assert len(findings) == 1
        (finding,) = findings
        assert finding.end == 'object'
        assert finding.node_name == 'Task 6128'
        assert finding.node_referent == _task('6128')
        assert set(finding.fact_referents) == {_task('6126'), _task('6080')}

    def test_a_fact_naming_no_referent_is_unverifiable_not_clean(self) -> None:
        """Never a finding — and, per build_report, never in the denominator."""
        assert (
            classify_edge(
                'Task 6165',
                'Task 6164',
                'The merge worker holds .git/index.lock for the whole hook run.',
                'edge-1',
                GRAPH,
            )
            == []
        )

    def test_no_task_shaped_endpoint_yields_nothing(self) -> None:
        assert (
            classify_edge(
                'ElasticResult.rotation',
                'IMPLEMENTATION COORDINATION',
                'Task 6164 described landing the same artefact.',
                'edge-2',
                GRAPH,
            )
            == []
        )

    def test_a_scanner_blind_spot_does_not_become_a_false_positive(self) -> None:
        """'#4262' is invisible to scan_content; bare_id_present rescues it."""
        assert (
            classify_edge(
                'Task 4262',
                'Task 4351',
                "#4262's cache is separated from the engine-level tables in "
                'Task 4351.',
                'edge-3',
                'dark_factory',
            )
            == []
        )

    def test_a_null_fact_is_never_a_finding(self) -> None:
        assert classify_edge('Task 6165', 'Task 6164', None, 'edge-4', GRAPH) == []

    def test_episodes_are_carried_onto_the_finding(self) -> None:
        """``r.episodes`` is the re-derivation path a reader must use.

        Any prior investigation that read "the Task 6165 instance" out of the
        graph was reading task 6164's ruling; re-deriving the truth means
        going back to the SOURCE episode, so its uuid has to travel with the
        finding rather than being looked up again by hand.
        """
        findings = classify_edge(
            'Task 6165',
            'ElasticResult.rotation',
            'Task 6164 described landing the same artefact.',
            '63fa5c78',
            GRAPH,
            episodes=['779b7b7d'],
        )
        assert findings[0].episodes == ('779b7b7d',)

    def test_findings_are_frozen(self) -> None:
        """A finding is evidence for a human's adjudication, not an accumulator."""
        (finding,) = _classify(SPEC_63FA5C78)
        with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError
            finding.end = 'object'  # type: ignore[misc]

    def test_vars_of_exposes_every_field(self) -> None:
        """``vars()`` does not work on a slots dataclass; the accessor does."""
        (finding,) = _classify(SPEC_63FA5C78)
        fields = vars_of(finding)
        assert fields['edge_uuid'] == '63fa5c78'
        assert set(fields) == set(Finding.__dataclass_fields__)


class TestOutOfReachByConstruction:
    """The two sub-classes this detector provably does NOT cover.

    Asserted rather than merely absent, so the gap is a pinned CONTRACT and
    the next reader cannot mistake "the sweep found none of these" for "the
    corpus holds none of these". Both are recorded in the report's
    ``known_gaps`` key for the same reason.
    """

    @pytest.mark.parametrize(
        'spec', [SPEC_1CF19488, SPEC_01E3D75E], ids=['1cf19488', '01e3e75e']
    )
    def test_direction_reversal_is_out_of_reach(self, spec: tuple) -> None:
        """Both endpoints ARE named in the fact; only the direction is wrong.

        The cheap heuristic (leftmost id named == object id != subject id)
        WAS measured during planning: 85/7131 flagged, overwhelmingly benign
        grammatical voice ('Task 2660 depends on Task 2659 landing' on edge
        (2659)->(2660)). Adjudicating direction needs the authoritative task
        dependency graph, not the fact text, so it is deliberately not
        shipped.
        """
        assert _classify(spec) == []

    def test_fact_contradicts_its_source_episode_is_out_of_reach(self) -> None:
        """993a9a7b: reachable by no text or topology rule at all.

        Both endpoints are named, so set-membership is satisfied; the defect
        is that the fact disagrees with the episode it was extracted from,
        which only a re-read of the episode body could show.
        """
        assert _classify(SPEC_993A9A7B) == []
