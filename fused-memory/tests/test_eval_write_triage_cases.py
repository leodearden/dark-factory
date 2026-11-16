"""Tests for eval_write_triage_cases.py — the cases a run asks, their slates, and the plan.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf γ, decision D10.

Ground truth comes from leaf alpha's labels, so leaf alpha's script is loaded
here independently of the eval's own import. Every test is free of
``OPENAI_API_KEY``, network and Qdrant.
"""
from __future__ import annotations

import functools
import logging
import types
from pathlib import Path

import pytest
from _eval_write_triage_fixtures import cases_mod as _cases_mod
from _eval_write_triage_fixtures import committed_records
from _eval_write_triage_fixtures import corpus as _corpus
from _eval_write_triage_fixtures import crossed_retrieved_plan as _crossed_retrieved_plan
from _eval_write_triage_fixtures import live as _live
from _eval_write_triage_fixtures import one_cluster_corpus as _one_cluster_corpus
from _eval_write_triage_fixtures import rec as _rec
from _eval_write_triage_fixtures import scored as _scored
from _eval_write_triage_fixtures import scoring_mod as _scoring
from _eval_write_triage_fixtures import slate as _slate
from _fm_helpers import load_script_module

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
)

SCRIPTS = Path(__file__).parent.parent / 'scripts'
CALIBRATE_PATH = SCRIPTS / 'calibrate_write_triage.py'


@functools.cache
def _calib() -> types.ModuleType:
    """Leaf alpha's script, loaded INDEPENDENTLY of the eval's own import.

    Loaded separately on purpose: the whole point of the label-vocabulary
    tests is that the eval does not RE-SPELL alpha's labels, and reading them
    back through the eval's own re-export could not tell a faithful re-export
    from a hand-typed copy that happens to agree today.
    """
    return load_script_module(CALIBRATE_PATH, 'calibrate_write_triage')


@pytest.fixture(scope='module')
def records() -> list[dict]:
    return committed_records()


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
        assert _cases_mod()._rotated(self.POOL, 1, 2) == ['b', 'c']

    def test_it_wraps_around_the_end(self) -> None:
        """Wrapping is what keeps a late-index case from getting a short slate."""
        assert _cases_mod()._rotated(self.POOL, 3, 3) == ['d', 'a', 'b']

    @pytest.mark.parametrize('offset', [4, 5, 9, 400])
    def test_an_offset_past_the_end_is_the_same_window_as_its_modulus(
        self, offset: int,
    ) -> None:
        """`build_judge_cases` passes the corpus index, which exceeds any pool."""
        rotated = _cases_mod()._rotated
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
            drawn = _cases_mod()._rotated(['a', 'b'], 0, 5)
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
            assert _cases_mod()._rotated([], 3, 2) == []

    @pytest.mark.parametrize('count', [0, -1, -10])
    def test_a_non_positive_count_returns_empty(self, count: int) -> None:
        """`--distractors 0` is a legitimate run: canonical-only slates."""
        assert _cases_mod()._rotated(self.POOL, 1, count) == []

    def test_a_zero_count_on_an_empty_pool_is_not_a_warning(self, caplog) -> None:
        """Asking for nothing and getting nothing is not a narrowed slate."""
        with caplog.at_level(logging.WARNING):
            assert _cases_mod()._rotated([], 0, 0) == []
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
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        labelled = [
            c for c in cases
            if c['expected_class'] != _scoring().CLASS_DISTRACTOR
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
            assert canonical['label'] == _scoring().LABEL_CANONICAL, f'{record!r}'
        for record in _corpus():
            assert record['cluster_id'] in {r['memory_id'] for r in _corpus()}

    def test_a_canonical_record_is_never_a_submitted_entry(self) -> None:
        """A canonical IS the attach target; asking the judge to compare it to
        itself would score the fixture's construction, not the judge.
        """
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        canonicals = {'c1-canon', 'c2-canon', 'c3-canon'}
        assert not (canonicals & {c['memory_id'] for c in cases}), (
            f'a canonical was submitted as a case: {cases!r}'
        )

    def test_a_labelled_case_always_shows_the_judge_its_cluster_canonical(self) -> None:
        """The attach target must be on the slate or the answer is about
        a different memory than the one an attach would touch.
        """
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        for case in cases:
            if case['expected_class'] == _scoring().CLASS_DISTRACTOR:
                continue
            cluster = case['memory_id'].split('-')[0]
            assert f'{cluster}-canon' in case['candidates'], f'{case!r}'

    def test_the_case_carries_the_submitted_content_and_category(self) -> None:
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
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
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
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
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        case = next(c for c in cases if c['memory_id'] == 'c2-distinct')
        assert set(case['acceptable_outcomes']) == {OUTCOME_STORED}

    def test_a_pseudo_contradiction_accepts_everything_but_contested(self) -> None:
        """These are curator-adjudicated BOTH-CORRECT pairs (esc-5557/esc-5626)
        — "the contradiction was an omission, not a disagreement".

        So the measurable property is narrow and negative: the judge must not
        MANUFACTURE a contradiction. Whether it stores or attaches is not
        something alpha's labels adjudicate, so neither is scored.
        """
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
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
        assert set(_scoring().ACCEPTABLE_OUTCOMES) == {
            calib.LABEL_DUPLICATE,
            calib.LABEL_DISTINCT,
            calib.LABEL_PSEUDO_CONTRADICTION,
        }
        assert _scoring().LABEL_CANONICAL == calib.LABEL_CANONICAL

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
        with pytest.raises(_scoring().UnknownLabelError, match='newly_invented_label'):
            _cases_mod().build_judge_cases(corpus, distractors=2)

    def test_every_label_in_the_committed_fixture_is_covered(self, records) -> None:
        """The live guard against the case above, run on the real corpus."""
        known = set(_scoring().ACCEPTABLE_OUTCOMES) | {_scoring().LABEL_CANONICAL}
        assert {r['label'] for r in records} <= known

    # -- distractors -------------------------------------------------------

    def test_a_distractor_never_comes_from_the_cases_own_cluster(self) -> None:
        """A same-cluster record on the "unrelated" slate would make an attach
        to it CORRECT while being scored as a distraction.
        """
        corpus = _corpus()
        by_id = {r['memory_id']: r['cluster_id'] for r in corpus}
        for case in _cases_mod().build_judge_cases(corpus, distractors=2):
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
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        for case in cases:
            if case['expected_class'] == _scoring().CLASS_DISTRACTOR:
                continue
            assert len(case['candidates']) == 3, f'{case!r}'
            assert len(set(case['candidates'])) == 3, f'duplicated slot: {case!r}'

    def test_selection_is_deterministic_and_seedless(self) -> None:
        """Committed artifacts must be reproducible from the fixture alone.

        `random` is not merely discouraged here: a seeded shuffle would make
        the committed report un-reproducible by anyone who did not also know
        the seed, and an unseeded one un-reproducible by anyone at all.
        """
        first = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        second = _cases_mod().build_judge_cases(list(reversed(_corpus())), distractors=2)
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
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
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
        module = _cases_mod()
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
        controls = _by_class(_cases_mod().build_judge_cases(corpus, distractors=2), 'distractor')
        assert controls, 'no distractor-control cases were built'
        for case in controls:
            own = by_id[case['memory_id']]
            for cid in case['candidates']:
                assert by_id[cid] != own, f'{case!r} shows its own cluster'

    def test_a_distractor_control_accepts_only_stored(self) -> None:
        controls = _by_class(_cases_mod().build_judge_cases(_corpus(), distractors=2), 'distractor')
        for case in controls:
            assert set(case['acceptable_outcomes']) == {OUTCOME_STORED}, f'{case!r}'

    def test_the_control_slate_is_as_wide_as_a_labelled_one(self) -> None:
        """Same width, so a difference in the answer is about the CONTENT of
        the slate rather than about how much of it there was.
        """
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        controls = _by_class(cases, 'distractor')
        for case in controls:
            assert len(case['candidates']) == 3, f'{case!r}'

    def test_the_control_class_is_capped_at_one_case_per_cluster(self) -> None:
        """Cost control, stated rather than silent: every case is a paid LLM
        call, and one control per cluster already answers the question the
        control exists to ask.
        """
        controls = _by_class(_cases_mod().build_judge_cases(_corpus(), distractors=2), 'distractor')
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
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        controls = [c for c in cases if c['expected_class'] not in alpha_labels]
        assert controls, 'no control cases were built'
        for case in controls:
            assert case['expected_class'] == _scoring().CLASS_DISTRACTOR, case

    def test_the_class_vocabulary_is_alphas_labels_plus_the_control(self) -> None:
        """Composition AND wire spelling, which is what the test above claimed.

        The tuple pins that the vocabulary is COMPOSED from alpha's constants
        rather than re-typed here; the spelling is pinned separately because
        `distractor` is a wire value — it appears in the committed artifact
        and in the confusion table an operator reads at the task-3169 flip.
        """
        assert _scoring().CLASS_DISTRACTOR == 'distractor'
        assert tuple(_scoring().EVAL_CLASSES) == (
            _calib().LABEL_DUPLICATE,
            _calib().LABEL_DISTINCT,
            _calib().LABEL_PSEUDO_CONTRADICTION,
            _scoring().CLASS_DISTRACTOR,
        )


# ---------------------------------------------------------------------------
# plan_from_slates, seeded_plan and EvalPlan
# ---------------------------------------------------------------------------

class TestPlanFromSlates:
    """The retrieved mode's case construction, with the live edge stubbed."""

    @staticmethod
    def _records() -> list[dict]:
        return [
            _rec('c1-canon', 'c1-canon', 'canonical'),
            _rec('c1-dup-1', 'c1-canon', 'duplicate'),
            _rec('c2-distinct', 'c2-canon', 'distinct'),
        ]

    @staticmethod
    def _slates() -> list:
        return [
            _slate('c1-dup-1', candidates=[_live('c1-canon'), _live('zz')],
                   band_winner_id='c1-canon', band=OUTCOME_JUDGE),
            _slate('c2-distinct', candidates=[_live('zz')],
                   band_winner_id='zz', band=OUTCOME_RESTATED,
                   canonical_present=False),
        ]

    def _plan(self, **overrides):
        return _cases_mod().plan_from_slates(
            self._records(), self._slates(), provenance=overrides or {'t_low': 0.5},
        )

    def test_one_case_per_non_canonical_record_and_no_control(self) -> None:
        plan = self._plan()
        assert [c['memory_id'] for c in plan.cases] == ['c1-dup-1', 'c2-distinct']
        assert not [
            c for c in plan.cases
            if c['expected_class'] == _scoring().CLASS_DISTRACTOR
        ]

    def test_the_expected_class_stays_the_curator_label(self) -> None:
        plan = self._plan()
        assert [c['expected_class'] for c in plan.cases] == ['duplicate', 'distinct']

    def test_the_band_and_its_winner_come_from_the_retrieval(self) -> None:
        plan = self._plan()
        judged, deterministic = plan.cases
        assert judged['band'] == OUTCOME_JUDGE
        assert judged['band_winner_id'] == 'c1-canon'
        assert deterministic['band'] == OUTCOME_RESTATED
        assert deterministic['band_winner_id'] == 'zz'

    def test_a_record_whose_canonical_is_gone_stays_in_the_population(self) -> None:
        """They are production reality, so dropping them would flatter the run."""
        plan = self._plan()
        assert plan.cases[1]['canonical_present'] is False
        assert len(plan.cases) == 2

    def test_the_plan_discloses_its_own_mode(self) -> None:
        assert self._plan().provenance['slate_mode'] == _scoring().SLATE_RETRIEVED
        assert _cases_mod().seeded_plan(
            _corpus(), distractors=2,
        ).provenance['slate_mode'] == _scoring().SLATE_SEEDED

    def test_a_slate_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match='positional'):
            _cases_mod().plan_from_slates(self._records(), [], provenance={})

    def test_a_hoisted_child_counts_as_the_canonical_on_the_slate(self) -> None:
        """`canonical_id_of` hoists a sighting to its parent, so a slate
        holding only the CHILD still shows the model the match."""
        slates = [
            _slate('c1-dup-1', candidates=[_live('child', canonical_id='c1-canon')],
                   band_winner_id='c1-canon', band=OUTCOME_JUDGE),
            _slate('c2-distinct', candidates=[_live('zz')],
                   band_winner_id='zz', band=OUTCOME_JUDGE),
        ]
        plan = _cases_mod().plan_from_slates(self._records(), slates, provenance={})
        row = _scoring().case_row(
            1, plan.cases[0], _scoring().JudgeAnswer(outcome=OUTCOME_AMENDED),
            plan.candidate_records[0], plan.fixture_by_id,
        )
        assert row['canonical_in_slate'] is True

    def test_every_case_carries_its_clusters_alias(self) -> None:
        plan = _cases_mod().plan_from_slates(
            self._records(), self._slates(), provenance={},
            aliases={'c1-canon': 'c1-new'},
        )
        assert [c['canonical_alias_id'] for c in plan.cases] == ['c1-new', None]

    def test_without_aliases_no_case_carries_one(self) -> None:
        assert [c['canonical_alias_id'] for c in self._plan().cases] == [None, None]

    def test_a_seeded_case_carries_its_clusters_alias_too(self) -> None:
        plan = _cases_mod().seeded_plan(_corpus(), distractors=2, aliases={'c1-canon': 'c1-new'})
        for case in plan.cases:
            expected = 'c1-new' if case['cluster_id'] == 'c1-canon' else None
            assert case['canonical_alias_id'] == expected, case['memory_id']
        assert _scoring().CLASS_DISTRACTOR in {
            c['expected_class'] for c in plan.cases if c['canonical_alias_id']
        }, 'the control case of an aliased cluster carries the alias too'


class TestEachCaseIsShownItsOwnRetrieval:
    """A retrieved row carries ITS query's cosine, so rows are never pooled.

    `judge_write` re-sorts every slate by that cosine (`select_judge_candidates`),
    so a row borrowed from another case's retrieval hands this case's prompt an
    order that another query produced. Two duplicates retrieving the same two
    records at opposite cosines make the borrowing observable.
    """

    def test_the_plan_resolves_each_case_to_its_own_slate(self) -> None:
        plan, slates = _crossed_retrieved_plan()
        for case, rows, slate in zip(plan.cases, plan.candidate_records, slates, strict=True):
            assert [row['memory_id'] for row in rows] == case['candidates']
            assert _scored(rows) == _scored(slate.candidates)

        corpus = _corpus()
        fixture_record = {record['memory_id']: record for record in corpus}
        seeded = _cases_mod().seeded_plan(corpus, distractors=2)
        for case, rows in zip(seeded.cases, seeded.candidate_records, strict=True):
            assert [row['memory_id'] for row in rows] == case['candidates']
            assert list(rows) == [fixture_record[cid] for cid in case['candidates']]

    def test_records_that_do_not_resolve_the_slates_are_refused(self) -> None:
        plan, _ = _crossed_retrieved_plan()

        def rebuilt(candidate_records):
            return _cases_mod().EvalPlan(
                cases=plan.cases, candidate_records=candidate_records,
                fixture_by_id=plan.fixture_by_id,
                record_count=plan.record_count, provenance=plan.provenance,
            )

        with pytest.raises(ValueError, match='dup-b') as excinfo:
            rebuilt((plan.candidate_records[0], plan.candidate_records[0]))
        assert 'dup-a' not in str(excinfo.value), 'only the unresolved case is named'
        with pytest.raises(ValueError):
            rebuilt(plan.candidate_records[:1])


class TestAJudgeBandCaseIsNeverShownAnEmptySlate:
    """Task 5277 A3: a verdict nobody gave must not be scored.

    `judge_write` answers an empty slate `stored` with no provider call, and
    `stored` is exactly what the distractor control accepts, so an eval that
    routed such a case to the judge would count a correct answer the model
    never produced. `EvalPlan` refuses the case where the plan is built:
    before any spend, and before the cases file is opened.
    """

    @staticmethod
    def _retrieved(*slates):
        records = [
            _rec('canon', 'canon', 'canonical'),
            *(_rec(slate.memory_id, 'canon', 'duplicate') for slate in slates),
        ]
        return _cases_mod().plan_from_slates(records, list(slates), provenance={})

    def test_a_single_cluster_seeded_plan_is_refused_naming_its_control(self) -> None:
        with pytest.raises(ValueError, match='c2-distinct') as excinfo:
            _cases_mod().seeded_plan(_one_cluster_corpus(), distractors=4)
        assert 'c2-dup-1' not in str(excinfo.value), 'only the empty slate is named'

    def test_every_offending_case_is_named(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            self._retrieved(
                _slate('empty-one', candidates=[], band_winner_id=None, band=OUTCOME_JUDGE),
                _slate('empty-two', candidates=[], band_winner_id=None, band=OUTCOME_JUDGE),
            )
        message = str(excinfo.value)
        assert 'empty-one' in message and 'empty-two' in message, message

    def test_a_judge_band_with_one_candidate_is_accepted(self) -> None:
        plan = self._retrieved(_slate(
            'd1', candidates=[_live('canon')], band_winner_id='canon', band=OUTCOME_JUDGE,
        ))
        assert [c['candidates'] for c in plan.cases] == [['canon']]

    def test_a_stored_band_with_an_empty_slate_is_accepted(self) -> None:
        """Production's own novel or degraded path, answered with no judge at all."""
        plan = self._retrieved(_slate(
            'd1', candidates=[], band_winner_id=None, band=OUTCOME_STORED, similarity=None,
        ))
        assert [c['candidates'] for c in plan.cases] == [[]]
