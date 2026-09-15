"""Tests for eval_write_triage_judge.py — the judge measured against leaf alpha's labels.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf γ, decision D10.

Structure mirrors ``test_calibrate_write_triage.py``: the script is loaded by
path through importlib (``scripts/`` is not an importable package), and the
loader is invoked LAZILY via ``_mod()`` so the label-vocabulary tests below
stay runnable independently of the script's existence.

Every test in this file is free of ``OPENAI_API_KEY``, network and Qdrant —
the judge is injected as a plain callable, exactly as ``run_calibration``
injects ``embed_fn``/``search_fn``.

WHAT THIS SUITE DELIBERATELY DOES NOT ASSERT: any accuracy FLOOR. D10 makes
the committed report the arbiter and the human at the task-3169 flip gate the
decision-maker. A floor asserted here would silently become that gate,
pre-empting a decision this task is explicitly told not to make. The
assertions are about report SHAPE, per-class presence with an explicit ``n``,
and traceability — never about whether a number is large enough.
"""
from __future__ import annotations

import functools
import json
import logging
import types
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
)

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'eval_write_triage_judge.py'
CALIBRATE_PATH = Path(__file__).parent.parent / 'scripts' / 'calibrate_write_triage.py'
FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'write_triage_calibration.jsonl'


def _load_module(path: Path, mod_name: str) -> types.ModuleType:
    """Load a ``scripts/`` file by path, registered in ``sys.modules``.

    Registration is required for reflection-based decorators, which resolve
    ``sys.modules.get(cls.__module__)``. Same loader as
    ``test_calibrate_write_triage.py``.
    """
    return load_script_module(path, mod_name=mod_name)


@functools.cache
def _mod() -> types.ModuleType:
    return _load_module(SCRIPT_PATH, 'eval_write_triage_judge')


@functools.cache
def _calib() -> types.ModuleType:
    """Leaf alpha's script, loaded INDEPENDENTLY of the eval's own import.

    Loaded separately on purpose: the whole point of the label-vocabulary
    tests is that the eval does not RE-SPELL alpha's labels, and reading them
    back through the eval's own re-export could not tell a faithful re-export
    from a hand-typed copy that happens to agree today.
    """
    return _load_module(CALIBRATE_PATH, 'calibrate_write_triage')


@pytest.fixture(scope='module')
def records() -> list[dict]:
    """The committed curator corpus, parsed with the stdlib.

    Parsed here rather than through either script's loader, so a loader bug
    cannot mask a data defect (and vice versa) — the discipline the sibling
    suite's own ``records`` fixture states.
    """
    assert FIXTURE_PATH.exists(), f'fixture missing: {FIXTURE_PATH}'
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text().splitlines()
        if line.strip()
    ]


def _rec(memory_id: str, cluster_id: str, label: str, *, category: str = 'procedural_knowledge') -> dict:
    """A minimal fixture record — only the keys the eval's pure core reads."""
    return {
        'memory_id': memory_id,
        'content': f'content of {memory_id}',
        'category': category,
        'cluster_id': cluster_id,
        'label': label,
    }


def _corpus() -> list[dict]:
    """Three clusters, every label represented, deliberately unsorted on input.

    Unsorted so a determinism assertion cannot pass by accident on an input
    that was already in canonical order.

    ``cluster_id`` IS the canonical record's own ``memory_id`` — the fixture's
    referential-integrity invariant, pinned against the committed corpus by
    `test_the_synthetic_corpus_uses_the_fixtures_cluster_key` below. A
    synthetic corpus that named clusters some other way would let the eval
    build slates out of ids that exist nowhere, and every test here would
    still pass.
    """
    return [
        _rec('c2-dup-1', 'c2-canon', 'duplicate'),
        _rec('c1-canon', 'c1-canon', 'canonical'),
        _rec('c3-pseudo', 'c3-canon', 'pseudo_contradiction'),
        _rec('c1-dup-2', 'c1-canon', 'duplicate'),
        _rec('c2-canon', 'c2-canon', 'canonical'),
        _rec('c1-dup-1', 'c1-canon', 'duplicate'),
        _rec('c3-canon', 'c3-canon', 'canonical'),
        _rec('c2-distinct', 'c2-canon', 'distinct'),
        _rec('c3-dup-1', 'c3-canon', 'duplicate'),
    ]


def _by_class(cases: list[dict], expected_class: str) -> list[dict]:
    return [c for c in cases if c['expected_class'] == expected_class]


# ---------------------------------------------------------------------------
# _rotated
# ---------------------------------------------------------------------------

class TestRotated:
    """The draw itself, tested directly — which is how `--limit` got away.

    `_rotated` had no test of its own; it was exercised only through
    `build_judge_cases`, where per-cluster pool exclusion masks it. That is
    how a `--limit` run narrowing every slate to a single-cluster pool went
    unnoticed: the narrowing happens HERE, in the `len(pool) < count` arm,
    and nothing looked at it.
    """

    POOL = ['a', 'b', 'c', 'd']

    def test_it_takes_count_entries_from_offset(self) -> None:
        assert _mod()._rotated(self.POOL, 1, 2) == ['b', 'c']

    def test_it_wraps_around_the_end(self) -> None:
        """Wrapping is what keeps a late-index case from getting a short slate."""
        assert _mod()._rotated(self.POOL, 3, 3) == ['d', 'a', 'b']

    @pytest.mark.parametrize('offset', [4, 5, 9, 400])
    def test_an_offset_past_the_end_is_the_same_window_as_its_modulus(
        self, offset: int,
    ) -> None:
        """`build_judge_cases` passes the corpus index, which exceeds any pool."""
        rotated = _mod()._rotated
        assert rotated(self.POOL, offset, 2) == rotated(
            self.POOL, offset % len(self.POOL), 2,
        )

    def test_a_short_pool_truncates_rather_than_repeating(self, caplog) -> None:
        """Fewer entries than asked for, each ONCE, and said out loud.

        Repeating an entry to reach the requested width would show the judge
        the same record twice and quietly change what the accuracy figure
        means; returning fewer is honest, but only if the report's reader can
        tell, which is what the warning is for.
        """
        with caplog.at_level(logging.WARNING):
            drawn = _mod()._rotated(['a', 'b'], 0, 5)
        assert drawn == ['a', 'b']
        assert len(drawn) == len(set(drawn))
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            'a narrowed slate must be announced, not inferred from the report'
        )

    def test_an_empty_pool_returns_empty_without_raising(self, caplog) -> None:
        """`start = offset % len(pool)` is a ZeroDivisionError on an empty pool.

        Reachable for real: a single-cluster corpus has no cross-cluster
        records at all, so the pool is genuinely empty rather than merely
        short.
        """
        with caplog.at_level(logging.WARNING):
            assert _mod()._rotated([], 3, 2) == []

    @pytest.mark.parametrize('count', [0, -1, -10])
    def test_a_non_positive_count_returns_empty(self, count: int) -> None:
        """`--distractors 0` is a legitimate run: canonical-only slates."""
        assert _mod()._rotated(self.POOL, 1, count) == []

    def test_a_zero_count_on_an_empty_pool_is_not_a_warning(self, caplog) -> None:
        """Asking for nothing and getting nothing is not a narrowed slate."""
        with caplog.at_level(logging.WARNING):
            assert _mod()._rotated([], 0, 0) == []
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# ---------------------------------------------------------------------------
# build_judge_cases
# ---------------------------------------------------------------------------

class TestBuildJudgeCases:
    """Ground truth derived from alpha's labels — never invented here.

    Each case is one judge call: a submitted entry plus the candidate slate it
    is shown. The label the curator assigned to the submitted entry is what
    names the acceptable answers, so every expectation in this class traces
    back to a human adjudication rather than to a guess about what the judge
    ought to say.
    """

    def test_every_non_canonical_record_yields_a_labelled_case(self) -> None:
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        labelled = [
            c for c in cases
            if c['expected_class'] != _mod().CLASS_DISTRACTOR
        ]
        assert {c['memory_id'] for c in labelled} == {
            'c1-dup-1', 'c1-dup-2', 'c2-dup-1', 'c2-distinct', 'c3-dup-1', 'c3-pseudo',
        }

    def test_the_synthetic_corpus_uses_the_fixtures_cluster_key(
        self, records,
    ) -> None:
        """`cluster_id` is the canonical's own `memory_id`, in BOTH corpora.

        Load-bearing, because `build_judge_cases` puts `cluster_id` straight
        onto the slate as the attach target. If the synthetic corpus named
        clusters any other way, every slate here would carry an id that
        resolves to no record — and nothing else in this file would notice,
        since none of it dereferences the slate.
        """
        by_id = {r['memory_id']: r for r in records}
        for record in records:
            canonical = by_id.get(record['cluster_id'])
            assert canonical is not None, f'dangling cluster_id: {record!r}'
            assert canonical['label'] == _mod().LABEL_CANONICAL, f'{record!r}'
        for record in _corpus():
            assert record['cluster_id'] in {r['memory_id'] for r in _corpus()}

    def test_a_canonical_record_is_never_a_submitted_entry(self) -> None:
        """A canonical IS the attach target; asking the judge to compare it to
        itself would score the fixture's construction, not the judge.
        """
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        canonicals = {'c1-canon', 'c2-canon', 'c3-canon'}
        assert not (canonicals & {c['memory_id'] for c in cases}), (
            f'a canonical was submitted as a case: {cases!r}'
        )

    def test_a_labelled_case_always_shows_the_judge_its_cluster_canonical(self) -> None:
        """The attach target must be on the slate or the answer is about
        a different memory than the one an attach would touch.
        """
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        for case in cases:
            if case['expected_class'] == _mod().CLASS_DISTRACTOR:
                continue
            cluster = case['memory_id'].split('-')[0]
            assert f'{cluster}-canon' in case['candidates'], f'{case!r}'

    def test_the_case_carries_the_submitted_content_and_category(self) -> None:
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        case = next(c for c in cases if c['memory_id'] == 'c1-dup-1')
        assert case['content'] == 'content of c1-dup-1'
        assert case['category'] == 'procedural_knowledge'

    # -- the label -> acceptable-outcomes table ----------------------------

    def test_a_duplicate_accepts_either_attach_and_nothing_else(self) -> None:
        """BOTH attach outcomes are correct, and that is a measurement
        decision, not laxity.

        Alpha's labels do not separate a verbatim restatement from a
        rediscovery that carries a novel fragment — the curator recorded
        "same claim as the canonical" and stopped there. Scoring one of
        `restated`/`amended` as WRONG would invent a label the curator never
        assigned and report a made-up error rate as a measured one. The split
        between them is reported as a DISTRIBUTION instead (see
        `TestScoreCases`), which says what the judge did without claiming to
        know which was right.
        """
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        case = next(c for c in cases if c['memory_id'] == 'c1-dup-1')
        assert set(case['acceptable_outcomes']) == {OUTCOME_RESTATED, OUTCOME_AMENDED}
        assert OUTCOME_STORED not in case['acceptable_outcomes'], (
            'a curator-confirmed rediscovery answered `stored` is the exact '
            'miss triage exists to catch'
        )
        assert OUTCOME_CONTESTED not in case['acceptable_outcomes']

    def test_a_distinct_record_accepts_only_stored(self) -> None:
        """The hard negative: same cluster, same topic, curator-ruled NOT the
        same claim. Any attach here destroys a distinction a human drew.
        """
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        case = next(c for c in cases if c['memory_id'] == 'c2-distinct')
        assert set(case['acceptable_outcomes']) == {OUTCOME_STORED}

    def test_a_pseudo_contradiction_accepts_everything_but_contested(self) -> None:
        """These are curator-adjudicated BOTH-CORRECT pairs (esc-5557/esc-5626)
        — "the contradiction was an omission, not a disagreement".

        So the measurable property is narrow and negative: the judge must not
        MANUFACTURE a contradiction. Whether it stores or attaches is not
        something alpha's labels adjudicate, so neither is scored.
        """
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        case = next(c for c in cases if c['memory_id'] == 'c3-pseudo')
        assert set(case['acceptable_outcomes']) == set(TRIAGE_OUTCOMES) - {OUTCOME_CONTESTED}
        assert OUTCOME_CONTESTED not in case['acceptable_outcomes']

    def test_the_table_is_alphas_vocabulary_imported_not_re_spelled(self) -> None:
        """A fifth label added to alpha must fail HERE, loudly.

        The failure mode this forbids: a new curator label lands in the
        fixture, this eval has no entry for it, and it is quietly bucketed as
        something — producing an accuracy figure computed against an
        expectation nobody ever set.
        """
        calib = _calib()
        assert set(_mod().ACCEPTABLE_OUTCOMES) == {
            calib.LABEL_DUPLICATE,
            calib.LABEL_DISTINCT,
            calib.LABEL_PSEUDO_CONTRADICTION,
        }
        assert _mod().LABEL_CANONICAL == calib.LABEL_CANONICAL

    def test_an_unknown_label_raises_rather_than_being_bucketed(self) -> None:
        """The TYPE is contractual, not just the message.

        `pytest.raises(Exception, match=...)` is satisfied by any exception
        carrying that substring — including the `KeyError`/`TypeError` a
        careless refactor would raise while LOOKING like the deliberate
        refusal. `score_cases` raises `UnknownLabelError` for the same
        condition at the scoring boundary, so pinning the type here is what
        keeps the two boundaries agreeing.
        """
        corpus = [*_corpus(), _rec('c1-mystery', 'c1', 'newly_invented_label')]
        with pytest.raises(_mod().UnknownLabelError, match='newly_invented_label'):
            _mod().build_judge_cases(corpus, distractors=2)

    def test_every_label_in_the_committed_fixture_is_covered(self, records) -> None:
        """The live guard against the case above, run on the real corpus."""
        known = set(_mod().ACCEPTABLE_OUTCOMES) | {_mod().LABEL_CANONICAL}
        assert {r['label'] for r in records} <= known

    # -- distractors -------------------------------------------------------

    def test_a_distractor_never_comes_from_the_cases_own_cluster(self) -> None:
        """A same-cluster record on the "unrelated" slate would make an attach
        to it CORRECT while being scored as a distraction.
        """
        corpus = _corpus()
        by_id = {r['memory_id']: r['cluster_id'] for r in corpus}
        for case in _mod().build_judge_cases(corpus, distractors=2):
            own = by_id.get(case['memory_id'])
            cluster = case['memory_id'].split('-')[0]
            extras = [
                cid for cid in case['candidates']
                if cid != f'{cluster}-canon'
            ]
            assert own is not None
            for cid in extras:
                assert by_id[cid] != own, f'{case!r} carries a same-cluster distractor'

    def test_the_slate_is_the_canonical_plus_exactly_n_distractors(self) -> None:
        """PRD C1's "top 3-5" is a WIDTH; a slate that silently narrows makes
        the measurement easier than the production call it stands in for.
        """
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        for case in cases:
            if case['expected_class'] == _mod().CLASS_DISTRACTOR:
                continue
            assert len(case['candidates']) == 3, f'{case!r}'
            assert len(set(case['candidates'])) == 3, f'duplicated slot: {case!r}'

    def test_selection_is_deterministic_and_seedless(self) -> None:
        """Committed artifacts must be reproducible from the fixture alone.

        `random` is not merely discouraged here: a seeded shuffle would make
        the committed report un-reproducible by anyone who did not also know
        the seed, and an unseeded one un-reproducible by anyone at all.
        """
        first = _mod().build_judge_cases(_corpus(), distractors=2)
        second = _mod().build_judge_cases(list(reversed(_corpus())), distractors=2)
        assert first == second, 'case construction depends on input ORDER'

    def test_distractors_are_spread_rather_than_the_same_slate_every_time(
        self,
    ) -> None:
        """Two cases from the SAME cluster must not receive the same slate.

        The property `_rotated` actually provides, asserted against the only
        population that can show it. A corpus-wide `len(slates) > 1` is
        guaranteed by per-cluster pool EXCLUSION alone — two cases in
        different clusters draw from different pools and carry different
        canonicals, so their slates differ however the draw is made. That is
        why the previous form still passed with `_rotated` monkeypatched to
        the `pool[:count]` its own docstring forbids: it was measuring
        exclusion, not rotation.

        Same-cluster cases share a canonical AND a pool, so a non-rotating
        draw hands them a byte-identical slate — one arbitrary handful of
        clusters measured over and over instead of the corpus.
        """
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        by_memory = {c['memory_id']: c for c in _by_class(cases, 'duplicate')}
        first, second = by_memory['c1-dup-1'], by_memory['c1-dup-2']
        assert first['candidates'][0] == second['candidates'][0] == 'c1-canon', (
            'both cases must share the cluster canonical as the attach target'
        )
        assert first['candidates'] != second['candidates'], (
            'two cases in one cluster were shown an identical slate'
        )

    def test_the_spread_assertion_fails_without_rotation(
        self, monkeypatch,
    ) -> None:
        """The guard on the guard: pin that the assertion above can FAIL.

        `_rotated`'s docstring names `pool[:count]` as the thing it exists not
        to be, so that substitution is the exact defect to reproduce. Without
        this, a future edit that weakens the spread assertion back into
        something exclusion alone satisfies would go unnoticed.
        """
        module = _mod()
        monkeypatch.setattr(
            module, '_rotated', lambda pool, offset, count: pool[:count],
        )
        cases = module.build_judge_cases(_corpus(), distractors=2)
        by_memory = {c['memory_id']: c for c in _by_class(cases, 'duplicate')}
        assert (
            by_memory['c1-dup-1']['candidates']
            == by_memory['c1-dup-2']['candidates']
        ), 'a non-rotating draw was expected to collapse the two slates'

    # -- the distractor control class --------------------------------------

    def test_a_distractor_control_case_shows_no_same_cluster_record_at_all(
        self,
    ) -> None:
        """The negative control: nothing on the slate is the right answer.

        Without it the eval cannot tell a judge that classifies from a judge
        that attaches to whatever it is shown — the labelled cases all carry
        the correct target, so "always attach" scores well on every one of
        them.
        """
        corpus = _corpus()
        by_id = {r['memory_id']: r['cluster_id'] for r in corpus}
        controls = _by_class(_mod().build_judge_cases(corpus, distractors=2), 'distractor')
        assert controls, 'no distractor-control cases were built'
        for case in controls:
            own = by_id[case['memory_id']]
            for cid in case['candidates']:
                assert by_id[cid] != own, f'{case!r} shows its own cluster'

    def test_a_distractor_control_accepts_only_stored(self) -> None:
        controls = _by_class(_mod().build_judge_cases(_corpus(), distractors=2), 'distractor')
        for case in controls:
            assert set(case['acceptable_outcomes']) == {OUTCOME_STORED}, f'{case!r}'

    def test_the_control_slate_is_as_wide_as_a_labelled_one(self) -> None:
        """Same width, so a difference in the answer is about the CONTENT of
        the slate rather than about how much of it there was.
        """
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        controls = _by_class(cases, 'distractor')
        for case in controls:
            assert len(case['candidates']) == 3, f'{case!r}'

    def test_the_control_class_is_capped_at_one_case_per_cluster(self) -> None:
        """Cost control, stated rather than silent: every case is a paid LLM
        call, and one control per cluster already answers the question the
        control exists to ask.
        """
        controls = _by_class(_mod().build_judge_cases(_corpus(), distractors=2), 'distractor')
        clusters = [c['memory_id'].split('-')[0] for c in controls]
        assert sorted(clusters) == ['c1', 'c2', 'c3']

    def test_the_class_name_is_a_module_constant_not_a_literal(self) -> None:
        """The control cases are LABELLED from the constant, not from a literal.

        The name says "not a literal" and the assertion used to say
        `CLASS_DISTRACTOR == 'distractor'` — which is the literal, asserted.
        The property worth holding is that renaming the constant MOVES the
        cases with it: a rename that missed `build_judge_cases` would leave
        the controls carrying the old spelling while every class-vocabulary
        test above still passed, and the mismatch would surface only as an
        `UnknownLabelError` at scoring time.

        Selected without naming the control class, so this cannot pass by
        agreeing with itself.
        """
        alpha_labels = {
            _calib().LABEL_DUPLICATE,
            _calib().LABEL_DISTINCT,
            _calib().LABEL_PSEUDO_CONTRADICTION,
        }
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        controls = [c for c in cases if c['expected_class'] not in alpha_labels]
        assert controls, 'no control cases were built'
        for case in controls:
            assert case['expected_class'] == _mod().CLASS_DISTRACTOR, case

    def test_the_class_vocabulary_is_alphas_labels_plus_the_control(self) -> None:
        """Composition AND wire spelling, which is what the test above claimed.

        The tuple pins that the vocabulary is COMPOSED from alpha's constants
        rather than re-typed here; the spelling is pinned separately because
        `distractor` is a wire value — it appears in the committed artifact
        and in the confusion table an operator reads at the task-3169 flip.
        """
        assert _mod().CLASS_DISTRACTOR == 'distractor'
        assert tuple(_mod().EVAL_CLASSES) == (
            _calib().LABEL_DUPLICATE,
            _calib().LABEL_DISTINCT,
            _calib().LABEL_PSEUDO_CONTRADICTION,
            _mod().CLASS_DISTRACTOR,
        )


# ---------------------------------------------------------------------------
# score_cases
# ---------------------------------------------------------------------------

def _cases(*specs: tuple[str, str]) -> list[dict]:
    """Hand-built cases: ``(memory_id, expected_class)`` pairs.

    Built directly rather than through ``build_judge_cases`` so a scoring bug
    and a case-construction bug cannot cancel each other out.
    """
    out = []
    for memory_id, expected_class in specs:
        out.append({
            'memory_id': memory_id,
            'content': f'content of {memory_id}',
            'category': 'procedural_knowledge',
            'candidates': ['x-canon'],
            'expected_class': expected_class,
            'acceptable_outcomes': _mod().ACCEPTABLE_OUTCOMES.get(
                expected_class, frozenset({OUTCOME_STORED}),
            ),
        })
    return out


class TestEvalOutcomes:
    """The confusion table's OTHER axis, and why it is a tuple too.

    `EVAL_CLASSES` already exists for the class axis, with a comment saying
    one list feeding two consumers is what keeps "not measured" and "measured
    perfect" distinguishable. `TRIAGE_OUTCOMES` is a frozenset, whose
    iteration order varies with PYTHONHASHSEED — and this script's output is a
    COMMITTED artifact read by an operator at the task-3169 flip gate. Item 2
    of `scripts/check_write_triage_flip_preconditions.sh` is exactly this.
    """

    def test_the_outcome_order_is_derived_and_sorted(self) -> None:
        assert tuple(sorted(TRIAGE_OUTCOMES)) == _mod().EVAL_OUTCOMES

    def test_no_outcome_is_added_or_dropped_on_the_way(self) -> None:
        """Derived, not hand-written: a fifth outcome joins the report itself.

        A hand-spelled tuple would be a second list to keep in sync with
        `write_triage.TRIAGE_OUTCOMES` — the very drift `EVAL_CLASSES`' own
        comment says the shared list prevents.
        """
        assert set(_mod().EVAL_OUTCOMES) == set(TRIAGE_OUTCOMES)


class TestScoreCases:
    """Counting only — every judgment call was made in the table above."""

    def test_per_class_reports_n_correct_and_accuracy(self) -> None:
        cases = _cases(('a', 'duplicate'), ('b', 'duplicate'), ('c', 'distinct'))
        got = _mod().score_cases(cases, [OUTCOME_RESTATED, OUTCOME_STORED, OUTCOME_STORED])
        assert got['per_class']['duplicate'] == {'n': 2, 'correct': 1, 'accuracy': 0.5}
        assert got['per_class']['distinct'] == {'n': 1, 'correct': 1, 'accuracy': 1.0}

    def test_all_four_classes_are_always_present_with_an_explicit_n(self) -> None:
        """An omitted class reads identically to a perfect one.

        `distinct` (n=3) and `pseudo_contradiction` (n=6) are small enough
        that a construction bug could empty either without anything looking
        wrong, so the report must state the population it measured even when
        that population is zero.
        """
        got = _mod().score_cases([], [])
        assert set(got['per_class']) == set(_mod().EVAL_CLASSES)
        for name, entry in got['per_class'].items():
            assert entry['n'] == 0, name
            assert entry['correct'] == 0, name

    def test_an_empty_class_scores_none_never_zero(self) -> None:
        """`0.0` reads as "measured, and the judge failed everything"."""
        got = _mod().score_cases([], [])
        for name, entry in got['per_class'].items():
            assert entry['accuracy'] is None, f'{name}: {entry!r}'

    def test_the_confusion_map_is_every_class_by_every_outcome(self) -> None:
        """Full 4xN, so a systematic failure is visible as a SHAPE.

        "Every duplicate answered `stored`" is a wiring bug and "duplicates
        split across the two attaches" is a working judge; a per-class
        accuracy alone cannot tell them apart.
        """
        cases = _cases(('a', 'duplicate'), ('b', 'pseudo_contradiction'))
        got = _mod().score_cases(cases, [OUTCOME_STORED, OUTCOME_CONTESTED])
        assert set(got['confusion']) == set(_mod().EVAL_CLASSES)
        for name, row in got['confusion'].items():
            assert set(row) == set(TRIAGE_OUTCOMES), name
        assert got['confusion']['duplicate'][OUTCOME_STORED] == 1
        assert got['confusion']['duplicate'][OUTCOME_RESTATED] == 0
        assert got['confusion']['pseudo_contradiction'][OUTCOME_CONTESTED] == 1

    def test_every_confusion_row_is_keyed_in_eval_outcome_order(self) -> None:
        """Order, not membership: `list(row)`, never `set(row)`.

        The rows are serialized to the committed JSON verbatim, so a row whose
        key order follows a frozenset's iteration order makes two identical
        runs produce two different artifacts — and the markdown the operator
        reads stops being provably the render of the JSON beside it.
        """
        cases = _cases(('a', 'duplicate'), ('b', 'distinct'))
        got = _mod().score_cases(cases, [OUTCOME_RESTATED, OUTCOME_STORED])
        for name, row in got['confusion'].items():
            assert list(row) == list(_mod().EVAL_OUTCOMES), name

    def test_the_duplicate_split_is_a_distribution_not_an_error_term(self) -> None:
        """Reported, and deliberately not scored — see the table's rationale."""
        cases = _cases(('a', 'duplicate'), ('b', 'duplicate'), ('c', 'duplicate'))
        got = _mod().score_cases(
            cases, [OUTCOME_RESTATED, OUTCOME_AMENDED, OUTCOME_AMENDED],
        )
        assert got['duplicate_outcome_split'] == {'restated': 1, 'amended': 2}
        assert got['per_class']['duplicate']['correct'] == 3, (
            'the split must not be charged as error'
        )

    def test_false_contested_counts_every_contested_verdict(self) -> None:
        """Every one of them is a false positive, and that is a property of
        the CORPUS, not a simplification.

        Alpha carries no positively-labelled contradiction anywhere: all six
        `pseudo_contradiction` records were adjudicated NOT contradictions.
        So the fixture can measure the judge's contested false-positive rate
        and nothing else — there is no contested recall to compute, and
        reporting one would be a number with no measurement behind it.
        """
        cases = _cases(('a', 'duplicate'), ('b', 'pseudo_contradiction'), ('c', 'distinct'))
        got = _mod().score_cases(
            cases, [OUTCOME_CONTESTED, OUTCOME_CONTESTED, OUTCOME_STORED],
        )
        assert got['false_contested'] == 2

    def test_no_contested_verdict_scores_zero_not_none(self) -> None:
        """Unlike an empty class, this one WAS measured — 0 is the finding."""
        cases = _cases(('a', 'duplicate'))
        got = _mod().score_cases(cases, [OUTCOME_RESTATED])
        assert got['false_contested'] == 0

    def test_the_result_round_trips_through_json(self) -> None:
        """It is written to disk verbatim; a set or a tuple in here is a
        TypeError at the end of a paid run.
        """
        cases = _cases(('a', 'duplicate'), ('b', 'distractor'))
        got = _mod().score_cases(cases, [OUTCOME_RESTATED, OUTCOME_STORED])
        assert json.loads(json.dumps(got)) == got

    def test_a_verdict_count_mismatch_raises(self) -> None:
        """Silently zipping to the shorter list would drop cases from the
        denominator and report an accuracy over a population nobody chose.
        """
        with pytest.raises(Exception, match='(?i)verdict'):
            _mod().score_cases(_cases(('a', 'duplicate'), ('b', 'distinct')), [OUTCOME_STORED])

    def test_the_distractor_class_is_scored_like_any_other(self) -> None:
        cases = _cases(('a', 'distractor'), ('b', 'distractor'))
        got = _mod().score_cases(cases, [OUTCOME_STORED, OUTCOME_RESTATED])
        assert got['per_class']['distractor'] == {'n': 2, 'correct': 1, 'accuracy': 0.5}

    def test_an_unknown_expected_class_raises_rather_than_being_absorbed(
        self,
    ) -> None:
        """A fifth class must not be silently bucketed at SCORING time either.

        `_acceptable_for` already raises for the same condition at
        case-construction time; the two boundaries have to agree. An absorbed
        row inflates `case_count` in `build_report` while `render_markdown`
        iterates only `EVAL_CLASSES`, so the row vanishes from the artifact
        entirely — the denominator moves and nothing in the report says so.
        """
        cases = _cases(('a', 'newly_invented_class'))
        with pytest.raises(_mod().UnknownLabelError) as excinfo:
            _mod().score_cases(cases, [OUTCOME_STORED])
        message = str(excinfo.value)
        assert 'newly_invented_class' in message, message
        for name in _mod().EVAL_CLASSES:
            assert name in message, f'{name} not named in: {message}'

    def test_a_verdict_outside_the_closed_vocabulary_raises(self) -> None:
        """`row.get(verdict, 0) + 1` would grow the row a fifth column.

        `render_markdown` iterates `EVAL_OUTCOMES`, so that column is dropped
        from the markdown the operator reads while it still sits in the JSON —
        two artifacts disagreeing about what was measured, which is the same
        traceability failure gate item 2 exists for.
        """
        cases = _cases(('a', 'duplicate'))
        with pytest.raises(ValueError) as excinfo:
            _mod().score_cases(cases, ['not_a_triage_outcome'])
        message = str(excinfo.value)
        assert 'not_a_triage_outcome' in message, message
        assert not isinstance(excinfo.value, _mod().UnknownLabelError), (
            'a bad VERDICT is not an unknown LABEL — they name different holes'
        )


# ---------------------------------------------------------------------------
# Report assembly / rendering / the runner
# ---------------------------------------------------------------------------

def _fake_judge(answer: str = OUTCOME_RESTATED):
    """A judge_fn that always answers *answer*, recording every call.

    Deliberately not a Mock: the recorded `.calls` list is asserted on
    positionally, and a Mock's call objects would let a signature change pass
    unnoticed.
    """
    def judge_fn(case, candidates):
        judge_fn.calls.append((
            case['memory_id'],
            case['expected_class'],
            [c['memory_id'] for c in candidates],
        ))
        return answer

    judge_fn.calls = []
    return judge_fn


_PROVENANCE = {
    'fixture_path': 'tests/fixtures/write_triage_calibration.jsonl',
    'judge_provider': 'openai',
    'judge_model': 'gpt-4o-mini',
}


def _run(
    tmp_path: Path,
    *,
    judge=None,
    corpus=None,
    distractors: int = 2,
    provenance=None,
):
    return _mod().run_judge_eval(
        records=corpus if corpus is not None else _corpus(),
        judge_fn=judge if judge is not None else _fake_judge(),
        report_path=tmp_path / 'report.json',
        provenance=dict(_PROVENANCE if provenance is None else provenance),
        distractors=distractors,
    )


class TestBuildReport:
    """The report is the deliverable — D10 makes it the operator's input."""

    def _report(self, **kwargs):
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        verdicts = [OUTCOME_RESTATED] * len(cases)
        return _mod().build_report(
            scored=_mod().score_cases(cases, verdicts),
            provenance=dict(_PROVENANCE),
            **kwargs,
        )

    def test_carries_every_top_level_key_the_operator_reads(self) -> None:
        assert set(self._report()) >= {
            'per_class', 'confusion', 'duplicate_outcome_split', 'false_contested',
            'contested_ground_truth', 'caveats', 'provenance',
        }

    def test_states_in_machine_readable_form_that_contested_is_unmeasurable(
        self,
    ) -> None:
        """The single most misleading reading this report could invite.

        `false_contested: 0` looks like "the contradiction detector is
        precise". It is not: alpha's corpus carries no positively-labelled
        contradiction, so a judge that can NEVER say `contested` scores
        identically to a perfect one. That has to be stated in the artifact
        rather than left to the reader, and machine-readably rather than only
        in prose, so a future consumer cannot join on the number without
        tripping over the caveat.
        """
        ground_truth = self._report()['contested_ground_truth']
        assert ground_truth['available'] is False
        assert ':' in ground_truth['reason'], (
            "alpha's '<code>: <measured detail>' reason format"
        )
        assert ground_truth['reason'].split(':')[0].strip(), 'a bare detail is not a code'

    def test_all_four_classes_are_present_with_an_explicit_n(self) -> None:
        per_class = self._report()['per_class']
        assert set(per_class) == set(_mod().EVAL_CLASSES)
        for name, entry in per_class.items():
            assert isinstance(entry['n'], int), name

    def test_provenance_names_the_model_that_produced_the_numbers(self) -> None:
        """An accuracy figure with no model attached cannot be acted on: the
        operator at the 3169 gate is deciding about a SHIPPED judge, and the
        judge follows `llm.model` unless pinned.
        """
        provenance = self._report()['provenance']
        for key in (
            'fixture_path', 'judge_provider', 'judge_model',
            'record_count', 'case_count', 'candidate_count', 'distractor_count',
        ):
            assert key in provenance, key

    def test_the_provenance_vocabulary_is_a_module_constant(self) -> None:
        """One place names the fields, so two places cannot disagree.

        `EVAL_CLASSES` and `EVAL_OUTCOMES` already work this way. Provenance
        did not: `build_report` backfilled a hand-typed triple and `_run`
        supplied a different hand-typed set, so a field added to one was
        simply absent from reports assembled through the other.
        """
        keys = _mod().PROVENANCE_KEYS
        assert isinstance(keys, tuple)
        assert set(keys) >= {
            'fixture_path', 'judge_provider', 'judge_model', 'limit',
            'record_count', 'case_count',
            'candidate_count', 'candidate_count_min',
            'distractor_count', 'distractor_count_requested',
            'judge_candidate_count', 'judge_enabled',
        }
        assert len(set(keys)) == len(keys), 'a duplicated key is a typo'

    def test_provenance_carries_every_key_even_when_nothing_was_measured(
        self,
    ) -> None:
        """The backfill covers the WHOLE vocabulary, not three of its members.

        `build_report`'s own docstring states the rule it then breaks: "An
        ABSENT key cannot be told apart from an artifact predating the field."
        It `setdefault`s `record_count`, `candidate_count` and
        `distractor_count` and omits `candidate_count_min` and
        `distractor_count_requested` — precisely the pair that discloses a
        NARROWED slate. A report assembled by any caller that did not supply
        them therefore reads exactly like a full-width run.
        """
        report = _mod().build_report(
            scored=_mod().score_cases([], []),
            provenance={},
        )
        provenance = report['provenance']
        assert set(provenance) == set(_mod().PROVENANCE_KEYS)
        for key in _mod().PROVENANCE_KEYS:
            if key == 'case_count':
                continue
            assert provenance[key] is None, (
                f'{key!r} was never measured, so it must read None rather '
                f'than be absent — absent is indistinguishable from an '
                f'artifact predating the field'
            )
        assert provenance['case_count'] == 0

    def test_a_supplied_provenance_value_is_never_overwritten(self) -> None:
        """The backfill fills GAPS. A measurement always wins over the default."""
        report = _mod().build_report(
            scored=_mod().score_cases([], []),
            provenance={'candidate_count_min': 1, 'judge_enabled': False},
        )
        provenance = report['provenance']
        assert provenance['candidate_count_min'] == 1
        assert provenance['judge_enabled'] is False

    def test_the_report_round_trips_through_json(self) -> None:
        report = self._report()
        assert json.loads(json.dumps(report)) == report

    def test_caveats_is_a_list_of_prose_strings(self) -> None:
        caveats = self._report()['caveats']
        assert isinstance(caveats, list) and caveats
        assert all(isinstance(c, str) and c for c in caveats)


class TestRenderMarkdown:
    """Positional column binding — the `_row_cells` idiom from the sibling suite."""

    COLUMNS = ('class', 'n', 'correct', 'accuracy')

    @staticmethod
    def _cells(row: str) -> list[str]:
        return [c.strip() for c in row.strip().strip('|').split('|')]

    @classmethod
    def _row_cells(cls, md: str, name: str) -> dict[str, str]:
        """The per-class row for *name*, read from THAT table only.

        Scoped to the section rather than scanned over the whole document.
        Both tables key their rows `| <class> |`, so an unscoped `next(...)`
        binds whichever table happens to come first and would silently answer
        with confusion-table cells if the two were ever reordered — reading a
        verdict count as an accuracy. Measured against a reordered render: the
        unscoped lookup returns the confusion row and reports `accuracy = 2`
        where the scorer holds `1.0`.

        The cell-count assertion below is not a substitute. Today it would
        catch that particular mis-bind by WIDTH (a confusion row is
        `1 + len(EVAL_OUTCOMES)` = 5 cells against these 4) and report `row
        has 5 cells` — the wrong diagnosis for the right failure. It stops
        catching it entirely the moment the two widths coincide, e.g. a fifth
        per-class column, or a fifth outcome paired with a dropped column.
        """
        section = cls._section(md, 'Per-class accuracy')
        row = next(ln for ln in section.splitlines() if ln.startswith(f'| {name} |'))
        cells = cls._cells(row)
        assert len(cells) == len(cls.COLUMNS), (
            f'row has {len(cells)} cells, header declares {len(cls.COLUMNS)}: {row}'
        )
        return dict(zip(cls.COLUMNS, cells, strict=True))

    def _md(self, verdict: str = OUTCOME_RESTATED) -> str:
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        scored = _mod().score_cases(cases, [verdict] * len(cases))
        return _mod().render_markdown(
            _mod().build_report(scored=scored, provenance=dict(_PROVENANCE)),
        )

    def test_the_header_declares_the_columns_this_class_binds(self) -> None:
        """Pins the binding itself: a reordered header fails here, once."""
        section = self._section(self._md(), 'Per-class accuracy')
        header = next(ln for ln in section.splitlines() if ln.startswith('| class |'))
        assert self._cells(header) == list(self.COLUMNS)

    def test_emits_one_row_per_class_including_the_empty_ones(self) -> None:
        """Every class appears, with ITS numbers — not merely a row.

        `assert self._row_cells(md, name)` was an assertion on a non-empty
        dict, which `_row_cells` returns for any row it finds at all; it could
        not fail except by the row being absent, and said nothing about the
        contents. The empty classes are the point: a class rendered with `0`
        where the scorer holds `None` reads as a measured failure rather than
        as not-measured, which is the distinction the whole pre-seeded
        `per_class` dict exists to preserve.
        """
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        scored = _mod().score_cases(cases, [OUTCOME_RESTATED] * len(cases))
        md = _mod().render_markdown(
            _mod().build_report(scored=scored, provenance=dict(_PROVENANCE)),
        )
        assert all(scored['per_class'][n]['n'] for n in _mod().EVAL_CLASSES), (
            'the fixture must measure every class for the first half to bite'
        )
        for name in _mod().EVAL_CLASSES:
            entry = scored['per_class'][name]
            assert self._row_cells(md, name) == {
                'class': name,
                'n': str(entry['n']),
                'correct': str(entry['correct']),
                'accuracy': str(entry['accuracy']),
            }, name

        # And the empty ones the name promises: nothing measured at all, yet
        # every class still renders a row, carrying `None` rather than `0.0`.
        empty_md = _mod().render_markdown(
            _mod().build_report(
                scored=_mod().score_cases([], []), provenance=dict(_PROVENANCE),
            ),
        )
        for name in _mod().EVAL_CLASSES:
            assert self._row_cells(empty_md, name) == {
                'class': name, 'n': '0', 'correct': '0', 'accuracy': 'None',
            }, name

    def test_the_row_carries_that_classes_own_numbers(self) -> None:
        cases = _mod().build_judge_cases(_corpus(), distractors=2)
        scored = _mod().score_cases(cases, [OUTCOME_RESTATED] * len(cases))
        md = _mod().render_markdown(
            _mod().build_report(scored=scored, provenance=dict(_PROVENANCE)),
        )
        cells = self._row_cells(md, 'duplicate')
        entry = scored['per_class']['duplicate']
        assert cells['n'] == str(entry['n'])
        assert cells['correct'] == str(entry['correct'])
        assert cells['accuracy'] == str(entry['accuracy'])

    def test_an_unmeasured_class_renders_its_none_rather_than_a_number(self) -> None:
        """`0.0` in this cell would read as a measured failure."""
        scored = _mod().score_cases([], [])
        md = _mod().render_markdown(
            _mod().build_report(scored=scored, provenance=dict(_PROVENANCE)),
        )
        assert self._row_cells(md, 'distinct')['accuracy'] == 'None'

    @staticmethod
    def _section(md: str, heading: str) -> str:
        """The lines under ``## heading``, up to the next ``## `` heading.

        The document carries two tables whose rows both start ``| <class> |``,
        so anything reading a row has to say WHICH table it means.
        """
        lines = md.splitlines()
        start = next(
            i for i, ln in enumerate(lines) if ln.startswith(f'## {heading}')
        )
        rest = lines[start + 1:]
        end = next(
            (i for i, ln in enumerate(rest) if ln.startswith('## ')), len(rest),
        )
        return '\n'.join(rest[:end])

    def test_the_confusion_columns_are_the_eval_outcome_order(self) -> None:
        """The committed markdown's column order must be reproducible.

        Measured 2026-08-27: the committed `.md` header and the committed
        `.json` confusion keys DISAGREE, which is only possible because both
        were rendered from a frozenset whose order moved between processes.
        """
        section = self._section(self._md(), 'Confusion')
        header = next(ln for ln in section.splitlines() if ln.startswith('| class |'))
        assert self._cells(header) == ['class', *_mod().EVAL_OUTCOMES]

    def test_each_confusion_row_lines_up_with_that_header(self) -> None:
        """A cell read by position is only a measurement if the columns bind."""
        md = self._md()
        section = self._section(md, 'Confusion')
        header = next(ln for ln in section.splitlines() if ln.startswith('| class |'))
        columns = self._cells(header)
        scored = _mod().score_cases(
            _mod().build_judge_cases(_corpus(), distractors=2),
            [OUTCOME_RESTATED] * len(_mod().build_judge_cases(_corpus(), distractors=2)),
        )
        for name in _mod().EVAL_CLASSES:
            row = next(
                ln for ln in section.splitlines() if ln.startswith(f'| {name} |')
            )
            cells = dict(zip(columns, self._cells(row), strict=True))
            for outcome in _mod().EVAL_OUTCOMES:
                assert cells[outcome] == str(scored['confusion'][name][outcome]), (
                    f'{name}/{outcome}'
                )

    def test_every_caveat_reaches_the_markdown_as_its_own_bullet(self) -> None:
        """Every ``CAVEATS`` entry renders verbatim, so none is silently dropped.

        Derived from the module's own constant BY IDENTITY, so it pins no
        prose: rewording a caveat rewords the expectation with it. What it
        does catch is the drift a keyword check could not — the renderer
        ceasing to iterate ``report['caveats']``, or an entry going missing
        on the way to the operator who reads the markdown, not the JSON.
        """
        md = self._md()
        for caveat in _mod().CAVEATS:
            assert f'- {caveat}' in md, caveat

    def test_renders_a_provenance_bullet_list(self) -> None:
        md = self._md()
        assert '## Provenance' in md
        assert '- `judge_model`: `gpt-4o-mini`' in md


class TestRunJudgeEval:
    """The runner: build cases, call the judge once each, score, write, return."""

    def test_writes_the_json_report(self, tmp_path: Path) -> None:
        report = _run(tmp_path)
        assert json.loads((tmp_path / 'report.json').read_text()) == report

    def test_writes_a_markdown_sibling(self, tmp_path: Path) -> None:
        _run(tmp_path)
        assert (tmp_path / 'report.md').exists()

    # --- the markdown sibling's composition (gate item 4) --------------------
    #
    # `run_judge_eval` writes the JSON and then a markdown sibling. Deriving
    # that sibling with `Path.with_suffix('.md')` REPLACES the last suffix, so
    # `--report-path foo.md` composes back to `foo.md` and the markdown
    # silently overwrites the JSON that was just written — with no error, on a
    # script whose output is a committed artifact. `guard_committed_report`
    # does not cover it: that guard addresses dry-run/`--limit` publishing and
    # returns early for any non-committed path.

    @staticmethod
    def _sibling_run(tmp_path: Path, name: str):
        return _mod().run_judge_eval(
            records=_corpus(),
            judge_fn=_fake_judge(),
            report_path=tmp_path / name,
            provenance=dict(_PROVENANCE),
            distractors=2,
        )

    def test_a_json_report_path_gets_a_dot_md_sibling(self, tmp_path: Path) -> None:
        """The anchor case: `.json` in, `.md` beside it, JSON still parseable."""
        report = self._sibling_run(tmp_path, 'r.json')
        assert json.loads((tmp_path / 'r.json').read_text()) == report
        assert (tmp_path / 'r.md').exists()
        assert (tmp_path / 'r.md').read_text().startswith('# ')

    def test_a_multi_suffix_path_keeps_its_json_and_gains_a_sibling(
        self, tmp_path: Path,
    ) -> None:
        """`foo.tar.gz` — the sibling is composed from the STEM, not a suffix swap."""
        report = self._sibling_run(tmp_path, 'foo.tar.gz')
        assert json.loads((tmp_path / 'foo.tar.gz').read_text()) == report
        assert (tmp_path / 'foo.tar.md').exists()

    def test_a_suffixless_path_gains_a_dot_md_sibling(self, tmp_path: Path) -> None:
        """`--report-path /tmp/smoke` is a reasonable ad-hoc spelling."""
        report = self._sibling_run(tmp_path, 'smoke')
        assert json.loads((tmp_path / 'smoke').read_text()) == report
        assert (tmp_path / 'smoke.md').exists()

    def test_a_dot_md_report_path_raises_instead_of_eating_the_json(
        self, tmp_path: Path,
    ) -> None:
        """The destructive case, and the reason this is gate item 4.

        `with_suffix('.md')` composes `foo.md` back to `foo.md`: the report is
        written and then overwritten by its own markdown, losing every number
        the run paid for, silently. `markdown_sibling` raises before the run
        does ANY work — not merely before the two writes — so the mistake
        costs nothing at all rather than costing a whole corpus of LLM calls
        for no artifact. That stronger claim is what
        :meth:`test_a_dot_md_report_path_costs_nothing_because_it_raises_first`
        pins; this one stays on the files. A warning instead of a raise, on a
        script whose output is a committed artifact, would be read after the
        loss.
        """
        target = tmp_path / 'foo.md'
        with pytest.raises(ValueError) as excinfo:
            self._sibling_run(tmp_path, 'foo.md')
        message = str(excinfo.value)
        assert str(target) in message, message
        assert not target.exists() or json.loads(target.read_text()), (
            f'{target} was left holding markdown where the JSON should be'
        )

    def test_a_dot_md_report_path_costs_nothing_because_it_raises_first(
        self, tmp_path: Path,
    ) -> None:
        """The guard's whole claim is that the mistake is free. Pin the COST.

        Raising before the two WRITES only protects the two files. The
        precondition is a function of *report_path* alone — knowable before
        `build_judge_cases`, before the per-case `judge_fn` loop, before
        `score_cases`. Evaluated after them, `--report-path foo.md` on a LIVE
        run spends every LLM call for the whole 102-case corpus and then
        raises with no artifact written at all: the operator pays for the run
        and gets nothing. The unit tests above cannot see that, because
        `_sibling_run` injects a free stub judge — so this asserts on the stub
        judge's CALL LOG instead of on the files.
        """
        judge = _fake_judge()
        with pytest.raises(ValueError):
            _mod().run_judge_eval(
                records=_corpus(),
                judge_fn=judge,
                report_path=tmp_path / 'foo.md',
                provenance=dict(_PROVENANCE),
                distractors=2,
            )
        assert judge.calls == [], (
            f'the guard raised only after {len(judge.calls)} judge call(s); on a '
            f'live run that is the whole corpus paid for and nothing written'
        )

    def test_creates_the_report_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / 'calibration' / 'nested'
        _mod().run_judge_eval(
            records=_corpus(),
            judge_fn=_fake_judge(),
            report_path=nested / 'report.json',
            provenance=dict(_PROVENANCE),
            distractors=2,
        )
        assert (nested / 'report.json').exists()

    def test_calls_the_judge_exactly_once_per_case(self, tmp_path: Path) -> None:
        """Every call is paid for; a re-ask would double the bill silently."""
        judge = _fake_judge()
        _run(tmp_path, judge=judge)
        expected = _mod().build_judge_cases(_corpus(), distractors=2)
        assert len(judge.calls) == len(expected)

    def test_the_judge_is_handed_resolved_candidate_records_not_bare_ids(
        self, tmp_path: Path,
    ) -> None:
        """A judge shown ids has no text to compare and is answering noise —
        the same defect `TestTheRealJudgeIsWiredAtTheToolSeam` pins at the
        production seam.
        """
        judge = _fake_judge()
        _run(tmp_path, judge=judge)
        for memory_id, _class, candidate_ids in judge.calls:
            assert candidate_ids, f'{memory_id} was handed an empty slate'

        # Keyed on (memory_id, class) because one record produces BOTH a
        # labelled case and, if it is its cluster's first, a control — and the
        # whole difference between them is the slate.
        by_case = {(mid, cls): ids for mid, cls, ids in judge.calls}
        assert by_case[('c1-dup-1', 'duplicate')][0] == 'c1-canon', (
            'the attach target must lead the resolved slate'
        )
        control = by_case[('c1-dup-1', _mod().CLASS_DISTRACTOR)]
        assert 'c1-canon' not in control, (
            'the control slate must carry no correct attach target at all'
        )

    def test_a_judge_that_is_wrong_on_every_case_still_produces_a_report(
        self, tmp_path: Path,
    ) -> None:
        """No pass/fail verdict, at any accuracy. D10 makes the report the
        arbiter and the human the gate; a runner that raised or flagged on a
        low score would be that gate, made of code.
        """
        report = _run(tmp_path, judge=_fake_judge(OUTCOME_CONTESTED))
        assert report['per_class']['duplicate']['accuracy'] == 0.0
        assert 'verdict' not in report
        assert 'passed' not in report and 'failed' not in report

    def test_a_judge_failure_propagates_rather_than_scoring_a_wrong_answer(
        self, tmp_path: Path,
    ) -> None:
        """The same discipline `run_calibration` applies to embed_fn/search_fn.

        A swallowed judge error is indistinguishable from a genuine
        misclassification, so it would silently shrink the measured population
        AND depress the accuracy it reports — a doubly wrong number.
        """
        def boom(case, candidates):
            raise RuntimeError('the judge exploded')

        with pytest.raises(RuntimeError, match='exploded'):
            _run(tmp_path, judge=boom)

    def test_provenance_records_the_measured_population(self, tmp_path: Path) -> None:
        report = _run(tmp_path)
        provenance = report['provenance']
        assert provenance['record_count'] == len(_corpus())
        assert provenance['case_count'] == len(
            _mod().build_judge_cases(_corpus(), distractors=2),
        )
        assert provenance['distractor_count'] == 2
        assert provenance['judge_model'] == 'gpt-4o-mini'

    def test_slate_width_provenance_is_measured_not_asserted(
        self, tmp_path: Path,
    ) -> None:
        """The report must publish the width it BUILT, not the one asked for.

        `_rotated` truncates when the distractor pool is short, so
        `distractors + 1` is a REQUEST that can silently exceed what the
        slates actually carried. Recording the request as `candidate_count`
        is the "silently narrows a slate the report claims was 5 wide"
        failure `run_judge_eval`'s docstring says its KeyError prevents,
        arriving by the other door — and the artifact is the operator's
        stated input at the task-3169 flip gate.

        A single-cluster corpus is the concrete trigger: `_distractor_pool`
        draws only from OTHER clusters, so it is empty and every labelled
        slate collapses to the canonical alone.
        """
        single = [r for r in _corpus() if r['cluster_id'] == _corpus()[0]['cluster_id']]
        report = _run(tmp_path, corpus=single, distractors=4)
        provenance = report['provenance']

        widths = {
            len(c['candidates'])
            for c in _mod().build_judge_cases(single, distractors=4)
        }
        assert max(widths) < 5, 'precondition: this corpus cannot fill a 5-wide slate'
        assert provenance['candidate_count'] == max(widths)
        assert provenance['candidate_count'] != 5, (
            'the report must not claim a width it never built'
        )
        assert provenance['distractor_count_requested'] == 4
        assert provenance['distractor_count'] == max(widths) - 1

    def test_a_short_pool_is_logged_loudly(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Silence here reads as "covered everything" when it did not."""
        single = [r for r in _corpus() if r['cluster_id'] == _corpus()[0]['cluster_id']]
        with caplog.at_level(logging.WARNING):
            _run(tmp_path, corpus=single, distractors=4)
        assert any(
            record.levelno >= logging.WARNING for record in caplog.records
        ), 'a narrowed slate must warn, not pass silently'

    def test_a_full_width_run_records_the_requested_width(
        self, tmp_path: Path,
    ) -> None:
        """Measuring must not CHANGE the numbers when the pool is adequate.

        The committed artifact reports candidate_count 5 / distractor_count 4;
        measurement has to agree with that on a corpus that can supply it,
        or this fix would silently restate the operator's own input.
        """
        report = _run(tmp_path, distractors=2)
        provenance = report['provenance']
        assert provenance['candidate_count'] == 3
        assert provenance['candidate_count_min'] == 3
        assert provenance['distractor_count'] == 2
        assert provenance['distractor_count_requested'] == 2

    def test_the_effective_cap_is_recorded_beside_the_built_width(
        self, tmp_path: Path,
    ) -> None:
        """`candidate_count` is the slate BUILT. The model sees it TRIMMED.

        `judge_write` re-trims via
        `select_judge_candidates(..., resolve_judge_candidate_count(...))`, so
        a slate built 3 wide against a cap of 2 reaches the model 2 wide and
        `candidate_count: 3` overstates what was measured. With the shipped
        `judge_candidate_count: 5` and the default `--distractors 4` the two
        agree, which is why this is latent rather than visible.

        Recorded ALONGSIDE the measurement, never in place of it: the cap is
        a config value and `candidate_count` stays the width that was built.
        """
        report = _run(tmp_path, distractors=2, provenance={
            **_PROVENANCE, 'judge_candidate_count': 2,
        })
        provenance = report['provenance']
        assert provenance['candidate_count'] == 3, 'precondition: built 3 wide'
        assert provenance['judge_candidate_count'] == 2

    def test_a_slate_wider_than_the_effective_cap_is_logged_loudly(
        self, tmp_path: Path, caplog,
    ) -> None:
        """The same silence the short-pool warning exists to break.

        A run that builds wider than the model can see is measuring a
        different slate from the one it publishes, and nothing in the
        artifact says so.
        """
        with caplog.at_level(logging.WARNING):
            _run(tmp_path, distractors=2, provenance={
                **_PROVENANCE, 'judge_candidate_count': 2,
            })
        warnings = [
            record.getMessage()
            for record in caplog.records if record.levelno >= logging.WARNING
        ]
        assert any('measured' in message for message in warnings), warnings
        assert any('3' in message and '2' in message for message in warnings), (
            'the warning must name the width built and the cap it exceeds'
        )

    def test_a_slate_within_the_effective_cap_does_not_warn(
        self, tmp_path: Path, caplog,
    ) -> None:
        """A warning that always fires is a warning nobody reads."""
        with caplog.at_level(logging.WARNING):
            _run(tmp_path, distractors=2, provenance={
                **_PROVENANCE, 'judge_candidate_count': 5,
            })
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestRunResolvesTheJudgeConfigIntoProvenance:
    """`_run` records the two config values that decide what was measured.

    `run_judge_eval` measures the width it BUILT; what the model saw is that
    width trimmed by `judge_candidate_count`. And whether the model saw
    anything at all is `judge_enabled`: a run against a config with the kill
    switch off returns `stored` on `judge_write`'s FIRST line, spends nothing,
    and still writes a committed-looking report — in which `distractor` scores
    1.0 (its only acceptable outcome IS `stored`) and `duplicate` scores 0.0.
    Neither fact is recoverable from the artifact today.

    Driven through `_run` with `run_judge_eval` captured, so this pins the
    resolution wiring rather than the runner it hands off to.
    """

    def _captured(self, tmp_path: Path, monkeypatch) -> dict:
        captured: dict = {}

        def fake_run_judge_eval(**kwargs):
            captured.update(kwargs['provenance'])
            return _mod().build_report(
                scored=_mod().score_cases([], []),
                provenance=dict(kwargs['provenance']),
            )

        monkeypatch.setattr(_mod(), 'run_judge_eval', fake_run_judge_eval)
        args = types.SimpleNamespace(
            config=None,
            report_path=str(tmp_path / 'report.json'),
            fixture=str(FIXTURE_PATH),
            distractors=2,
            limit=None,
            dry_run=True,
        )
        assert _mod()._run(args) == 0
        return captured

    @staticmethod
    def _service():
        from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

        return types.SimpleNamespace(config=FusedMemoryConfig())

    def test_provenance_records_the_effective_candidate_cap(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        captured = self._captured(tmp_path, monkeypatch)
        expected = write_triage_judge.resolve_judge_candidate_count(self._service())
        assert captured['judge_candidate_count'] == expected
        assert isinstance(captured['judge_candidate_count'], int)

    def test_provenance_records_whether_the_judge_arm_was_live(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        captured = self._captured(tmp_path, monkeypatch)
        expected = write_triage_judge.resolve_judge_enabled(self._service())
        assert captured['judge_enabled'] is expected
        assert isinstance(captured['judge_enabled'], bool)

    def test_a_disabled_judge_is_machine_detectable_in_the_artifact(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The kill switch is the one that costs the operator a decision.

        `judge_write` returns `stored` on its first line when the switch is
        off, so every case is answered without a provider call and the report
        still looks like a measurement.
        """
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        monkeypatch.setattr(
            write_triage_judge, 'resolve_judge_enabled', lambda service: False,
        )
        captured = self._captured(tmp_path, monkeypatch)
        assert captured['judge_enabled'] is False


class TestADotMdReportPathIsRejectedAtArgumentTime:
    """`--report-path foo.md` is a bad ARGUMENT, so `_run` refuses it up front.

    Nothing about the collision depends on what the run measures, so making
    the operator wait for a paid corpus-wide run to be told is pure waste.
    Driven with a fixture that does NOT exist: if the check were ordered
    after the fixture load — never mind after the judge loop — the failure
    would be a `FileNotFoundError` naming the fixture instead of the
    `ValueError` naming the report path.
    """

    @staticmethod
    def _args(tmp_path: Path, name: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            config=None,
            report_path=str(tmp_path / name),
            fixture=str(tmp_path / 'no-such-fixture.jsonl'),
            distractors=2,
            limit=None,
            dry_run=True,
        )

    def test_run_refuses_it_before_it_even_reads_the_fixture(
        self, tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            _mod()._run(self._args(tmp_path, 'foo.md'))
        assert str(tmp_path / 'foo.md') in str(excinfo.value), str(excinfo.value)

    def test_a_json_report_path_is_not_refused_by_that_check(
        self, tmp_path: Path,
    ) -> None:
        """The control: a well-formed path gets past it and fails LATER, on
        the missing fixture, which is the next real precondition."""
        with pytest.raises(FileNotFoundError):
            _mod()._run(self._args(tmp_path, 'report.json'))


# ---------------------------------------------------------------------------
# The committed artifact
# ---------------------------------------------------------------------------

class TestGuardCommittedReport:
    """A run that did not measure must not publish itself as the measurement.

    The committed report is what the operator reads at the task-3169 flip
    gate. The regression this class pins: the overwrite guard used to live
    INSIDE ``_run``'s ``if args.limit is not None:`` block, so a bare
    ``--dry-run`` — the FIRST invocation in the module docstring, and the one
    that measures nothing at all — defaulted to the committed path and
    rewrote both artifacts with fixed-answer numbers under a `dry-run`
    provider nobody was obliged to notice.
    """

    def test_an_unrelated_path_is_returned_untouched(self, tmp_path: Path) -> None:
        target = str(tmp_path / 'somewhere-else.json')
        assert _mod().guard_committed_report(
            target, dry_run=True, limit=None) == target

    def test_a_dry_run_is_redirected_off_the_committed_artifact(self) -> None:
        committed = _mod()._DEFAULT_REPORT_PATH
        got = _mod().guard_committed_report(committed, dry_run=True, limit=None)
        assert Path(got).resolve() != Path(committed).resolve(), (
            'a --dry-run scores a fixed-answer stub; publishing it puts '
            'fabricated numbers in front of the flip-gate operator'
        )
        assert Path(got).name == _mod()._DRY_RUN_REPORT_NAME

    def test_a_relative_spelling_of_the_committed_path_is_also_caught(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A string ``==`` misses this; ``resolve()`` on both sides catches it."""
        committed = Path(_mod()._DEFAULT_REPORT_PATH)
        monkeypatch.chdir(committed.parent.parent)
        relative = str(Path('calibration') / committed.name)
        got = _mod().guard_committed_report(relative, dry_run=True, limit=None)
        assert Path(got).resolve() != committed.resolve()

    def test_the_dry_run_redirect_is_logged_loudly(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            _mod().guard_committed_report(
                _mod()._DEFAULT_REPORT_PATH, dry_run=True, limit=None)
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_a_limit_run_still_writes_the_committed_path_but_warns(
        self, caplog,
    ) -> None:
        """`--limit` measures REALLY, just partially.

        Its numbers are the judge's own and ``provenance.limit`` records the
        truncation in the artifact, so this one warns and proceeds rather
        than redirecting — the artifact-level tell is what
        :meth:`TestCommittedJudgeAccuracyReportIsTraceable.test_the_committed_report_is_a_full_measurement`
        reads.
        """
        committed = _mod()._DEFAULT_REPORT_PATH
        with caplog.at_level(logging.WARNING):
            got = _mod().guard_committed_report(committed, dry_run=False, limit=5)
        assert got == committed
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_a_full_live_run_is_neither_redirected_nor_warned(
        self, caplog,
    ) -> None:
        committed = _mod()._DEFAULT_REPORT_PATH
        with caplog.at_level(logging.WARNING):
            got = _mod().guard_committed_report(committed, dry_run=False, limit=None)
        assert got == committed
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestABareDryRunCannotReachTheCommittedArtifact:
    """The guard must be consulted on EVERY path, driven rather than read.

    Replaces a test that asserted `'guard_committed_report(' in
    inspect.getsource(_run).split('if args.limit is not None:')[0]`. That
    passes UNCONDITIONALLY the moment the split literal is reworded or moved:
    `str.split` on an absent separator returns the WHOLE function body, so the
    guard could sink back inside the `--limit` block — the exact structural
    mistake being pinned — and the assertion would still find it. Per the repo
    norm it is replaced with behaviour, not hardened into a better grep.

    Driven through `main()` rather than a hand-built args namespace, so the
    argparse DEFAULT for `--report-path` is what gets guarded — that default
    is the whole hazard: a bare `--dry-run`, the first invocation in the
    module docstring, aims at the committed artifact without the operator
    naming it.
    """

    def test_a_bare_dry_run_leaves_both_committed_artifacts_byte_identical(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import sys  # noqa: PLC0415
        import tempfile  # noqa: PLC0415

        fixture = tmp_path / 'synthetic_corpus.jsonl'
        fixture.write_text(
            ''.join(json.dumps(record) + '\n' for record in _corpus()),
        )

        committed = Path(_mod()._DEFAULT_REPORT_PATH)
        sibling = committed.parent / (committed.stem + '.md')
        assert committed.exists() and sibling.exists(), (
            'precondition: both committed artifacts are on disk to be clobbered'
        )
        before = {path: path.read_bytes() for path in (committed, sibling)}
        before_mtimes = {path: path.stat().st_mtime_ns for path in before}

        # The redirect target, pointed somewhere hermetic. `guard_committed_report`
        # imports `tempfile` inside itself, so the module attribute is the seam.
        redirect_dir = tmp_path / 'redirected'
        redirect_dir.mkdir()
        monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(redirect_dir))
        monkeypatch.setattr(sys, 'argv', [
            'eval_write_triage_judge.py', '--dry-run', '--fixture', str(fixture),
        ])

        assert _mod().main() == 0

        for path, content in before.items():
            assert path.read_bytes() == content, (
                f'a --dry-run rewrote {path.name} with fixed-answer numbers — '
                f'the guard is not reached on the bare --dry-run path'
            )
            assert path.stat().st_mtime_ns == before_mtimes[path], (
                f'{path.name} was rewritten (identically, this time) — the '
                f'guard must not let a dry run touch it at all'
            )

        redirected = redirect_dir / _mod()._DRY_RUN_REPORT_NAME
        assert redirected.exists(), (
            'the dry run must still publish its throwaway where the redirect '
            'warning says it did, or "prove the pipeline" proves nothing'
        )
        assert json.loads(redirected.read_text())['provenance'][
            'judge_provider'
        ] == 'dry-run'


class TestCommittedJudgeAccuracyReportIsTraceable:
    """A number gating a decision must be a number some run measured.

    Modelled on `test_calibrate_write_triage.py::TestCommittedCalibrationIsTraceable`,
    and for the same reason: `judge_accuracy_report_path` exists so a reader
    can get from "the judge was evaluated" back to the run that evaluated it,
    without taking anyone's word that the figures were not typed in. Committed
    artifacts only — no network, no script execution, no Qdrant.

    EXPLICITLY NO ASSERTION ON ANY ACCURACY VALUE, and that is the point of
    this docstring. D10 makes this report the operator's input at the
    task-3169 flip gate. A floor asserted here would silently BECOME that
    gate — every future run would have to clear a bar this task invented,
    pre-empting a judgment it is explicitly told to leave to a human. What is
    asserted is only that each accuracy is WELL-FORMED: `None`, or a float in
    [0.0, 1.0]. Never that it is large enough.
    """

    CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'

    # Decorator order is load-bearing: `staticmethod` OUTERMOST, so
    # `self._committed()` resolves through the descriptor and calls the cached
    # function with zero arguments. Stacked the other way, `functools.cache`'s
    # plain-function wrapper is the class attribute, binds `self` as its first
    # argument, and raises TypeError on every access. Same note as the
    # sibling suite's.
    @staticmethod
    @functools.cache
    def _committed():
        import yaml  # noqa: PLC0415

        cls = TestCommittedJudgeAccuracyReportIsTraceable
        block = yaml.safe_load(cls.CONFIG_PATH.read_text()).get('write_triage') or {}
        report_path = block.get('judge_accuracy_report_path')
        report = None
        resolved = None
        if report_path is not None:
            resolved = Path(__file__).parent.parent / report_path
            if resolved.exists():
                report = json.loads(resolved.read_text())
        return block, report, resolved

    def test_the_report_path_is_set_and_relative(self) -> None:
        block, _report, _resolved = self._committed()
        assert block.get('judge_accuracy_report_path'), (
            'a judge shipped without a measurement is an untraceable judge'
        )
        assert not Path(block['judge_accuracy_report_path']).is_absolute(), (
            'the path must not bake in the checkout it was produced in — this '
            'script runs in per-task worktrees that get reset'
        )

    def test_the_path_resolves_to_a_committed_report(self) -> None:
        block, report, resolved = self._committed()
        assert report is not None, (
            f'judge_accuracy_report_path '
            f'{block.get("judge_accuracy_report_path")!r} does not resolve to a '
            f'committed report (looked at {resolved})'
        )

    def test_the_report_carries_every_key_the_operator_reads(self) -> None:
        _block, report, _resolved = self._committed()
        assert report is not None
        assert set(report) >= {
            'per_class', 'confusion', 'duplicate_outcome_split', 'false_contested',
            'contested_ground_truth', 'caveats', 'provenance',
        }

    def test_all_four_classes_are_present_with_an_integer_n(self) -> None:
        _block, report, _resolved = self._committed()
        assert report is not None
        per_class = report['per_class']
        assert set(per_class) == set(_mod().EVAL_CLASSES)
        for name, entry in per_class.items():
            assert isinstance(entry['n'], int), f'{name}: {entry!r}'

    def test_the_small_classes_match_the_fixtures_own_label_census(
        self, records,
    ) -> None:
        """`distinct` and `pseudo_contradiction` are the two the task names by
        hand — n=3 and n=6 today.

        Recounted from the JSONL rather than hardcoded, so editing the fixture
        surfaces as a mismatch demanding a re-measurement instead of leaving a
        stale constant agreeing with a stale report.
        """
        _block, report, _resolved = self._committed()
        assert report is not None
        for label in (_mod().LABEL_DISTINCT, _mod().LABEL_PSEUDO_CONTRADICTION):
            expected = sum(1 for r in records if r['label'] == label)
            assert expected > 0, f'{label} vanished from the fixture'
            assert report['per_class'][label]['n'] == expected, (
                f'{label}: report says {report["per_class"][label]["n"]}, the '
                f'fixture holds {expected} — re-run the eval'
            )

    def test_every_accuracy_is_well_formed_and_none_is_asserted_to_be_good(
        self,
    ) -> None:
        """The only shape assertion this class makes about a number."""
        _block, report, _resolved = self._committed()
        assert report is not None
        for name, entry in report['per_class'].items():
            accuracy = entry['accuracy']
            if accuracy is None:
                assert entry['n'] == 0, f'{name}: measured but unscored'
                continue
            assert isinstance(accuracy, float), f'{name}: {accuracy!r}'
            assert 0.0 <= accuracy <= 1.0, f'{name}: {accuracy!r}'

    def test_the_contested_caveat_survived_into_the_artifact(self) -> None:
        _block, report, _resolved = self._committed()
        assert report is not None
        ground_truth = report['contested_ground_truth']
        assert ground_truth['available'] is False
        assert ground_truth['reason'].split(':')[0].strip(), 'a reason code is required'

    def test_a_markdown_sibling_was_committed_beside_the_json(self) -> None:
        """The operator reads the markdown, so the artifact must exist.

        Only existence is checked. Deliberately NOT a ``CAVEATS``-identity
        check like the renderer's: ``CAVEATS`` is source, whereas this ``.md``
        is a measured artifact regenerable only by a live LLM run, so coupling
        the two would make a one-word caveat edit require a paid
        re-measurement to get back to green. The caveat's substance is pinned
        in the JSON instead, by
        :meth:`test_the_contested_caveat_survived_into_the_artifact`.
        """
        _block, report, resolved = self._committed()
        assert report is not None and resolved is not None
        sibling = resolved.with_suffix('.md')
        assert sibling.exists(), f'no markdown sibling at {sibling}'

    def test_the_committed_markdown_is_the_render_of_the_committed_json(
        self,
    ) -> None:
        """The operator reads the `.md`; the `.json` is what a reviewer diffs.

        `run_judge_eval` writes both from ONE report dict in ONE process, so
        a committed pair that disagrees provably did not come from a single
        run — and the markdown the task-3169 flip operator acts on is then
        not the render of the evidence beside it. Measured 2026-08-27: the
        committed `.json` keyed its confusion rows
        `[stored, amended, contested, restated]` while the committed `.md`
        header read `| class | stored | restated | amended | contested |`,
        which is a frozenset's iteration order moving between two processes.
        """
        _block, report, resolved = self._committed()
        assert report is not None and resolved is not None
        sibling = resolved.with_suffix('.md')
        assert sibling.exists(), f'no markdown sibling at {sibling}'
        assert sibling.read_text() == _mod().render_markdown(report), (
            f'{sibling.name} is not the render of {resolved.name} — the '
            f'markdown the task-3169 operator reads must provably be the '
            f'render of the committed JSON, not a second artifact that drifted'
        )

    def test_the_committed_report_is_a_full_measurement(self) -> None:
        """Not a `--dry-run` stub, and not a truncated `--limit` smoke.

        The guard in the script stops the first case at write time; this
        stops BOTH at review time, so a clobbered artifact is a red test
        rather than a silent substitution of fabricated (or partial) numbers
        for the measurement the task-3169 flip gate reads.

        `judge_provider` is checked against the shipped `_KNOWN_PROVIDERS`
        tuple, which the `dry-run` sentinel is deliberately not a member of.
        Still no assertion on any accuracy VALUE — this is about whether a
        run happened, never about whether it scored well.
        """
        from fused_memory.server import write_triage_judge  # noqa: PLC0415

        _block, report, _resolved = self._committed()
        assert report is not None
        provenance = report['provenance']
        assert provenance['limit'] is None, (
            f'the committed report is a --limit {provenance["limit"]!r} smoke, '
            f'not the corpus-wide measurement — re-run the eval without '
            f'--limit. `guard_committed_report` only WARNS on a --limit run '
            f'(deliberately: its numbers are the judge\'s own, just partial), '
            f'so nothing but this assertion stands between a partial artifact '
            f'and the task-3169 flip gate'
        )
        assert provenance['case_count'] == sum(
            entry['n'] for entry in report['per_class'].values()
        ), (
            'the population the provenance CLAIMS was measured and the one the '
            'per-class table actually scores disagree — the artifact is not '
            'internally consistent'
        )
        assert provenance['judge_provider'] in write_triage_judge._KNOWN_PROVIDERS, (
            f'judge_provider {provenance["judge_provider"]!r} is not a real '
            f'provider {list(write_triage_judge._KNOWN_PROVIDERS)} — this '
            f'artifact was written by a stub, not measured'
        )

    def test_the_committed_provenance_carries_the_whole_vocabulary(self) -> None:
        """A field added to `PROVENANCE_KEYS` must reach the artifact too.

        `build_report` backfills every key in the vocabulary, so a FRESH
        report cannot omit one — `test_provenance_carries_every_key_even_when_nothing_was_measured`
        pins that. The committed artifact is the gap: it is regenerated only
        by a paid live run, so a field added afterwards leaves the file the
        task-3169 operator actually reads silent on it, with the per-key
        assertions below unable to notice because they each name one key by
        hand. Measured 2026-08-27: `judge_candidate_count` and `judge_enabled`
        were added and the committed report still carried the ten keys that
        predated them.

        A superset, not equality: an artifact from a run that recorded MORE
        than the current vocabulary is stale provenance, not a lie, and must
        not be a red test.
        """
        _block, report, _resolved = self._committed()
        assert report is not None
        missing = set(_mod().PROVENANCE_KEYS) - set(report['provenance'])
        assert not missing, (
            f'the committed report does not disclose {sorted(missing)} — either '
            f're-run the eval, or record the values the original run used and '
            f'say so in the caveats'
        )

    def test_the_committed_report_says_the_judge_arm_was_live(self) -> None:
        """The field exists because a disabled judge reads like a measurement.

        `judge_write` returns `stored` on its first line when the kill switch
        is off: every case answered with no provider call, `distractor`
        scoring 1.0 and `duplicate` 0.0, and an artifact otherwise
        indistinguishable from the corpus-wide run the flip gate reads. An
        artifact silent on the switch reproduces exactly the hazard the field
        was added to close.

        `judge_candidate_count` is asserted only for WELL-FORMEDNESS, never
        against the shipped config value: the knob is hot-reloadable, and
        binding the two would make an operator's config edit demand a paid
        re-measurement to get back to green.
        """
        _block, report, _resolved = self._committed()
        assert report is not None
        provenance = report['provenance']
        assert provenance['judge_enabled'] is True, (
            f'judge_enabled={provenance["judge_enabled"]!r}: this artifact was '
            f'produced with the judge arm off, so its per-class figures are '
            f'the kill switch being measured, not the judge'
        )
        cap = provenance['judge_candidate_count']
        assert isinstance(cap, int) and not isinstance(cap, bool) and cap > 0, (
            f'judge_candidate_count={cap!r} — the width the model actually saw '
            f'must be a positive integer'
        )

    def test_provenance_names_the_model_and_the_fixture(self) -> None:
        _block, report, _resolved = self._committed()
        assert report is not None
        provenance = report['provenance']
        for key in ('judge_provider', 'judge_model'):
            assert isinstance(provenance[key], str) and provenance[key], key
        fixture = Path(__file__).parent.parent / provenance['fixture_path']
        assert fixture.resolve() == FIXTURE_PATH.resolve(), (
            f'the report measured {provenance["fixture_path"]!r}, not the '
            f'committed fixture'
        )
