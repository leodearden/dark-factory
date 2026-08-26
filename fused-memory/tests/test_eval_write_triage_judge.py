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
import importlib.util
import json
import types
from pathlib import Path

import pytest

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
    import sys  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


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
        corpus = [*_corpus(), _rec('c1-mystery', 'c1', 'newly_invented_label')]
        with pytest.raises(Exception, match='newly_invented_label'):
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
        assert 'import random' not in SCRIPT_PATH.read_text()

    def test_distractors_are_spread_rather_than_the_same_slate_every_time(
        self, records,
    ) -> None:
        """A single globally-smallest slate reused 84 times would measure one
        arbitrary pair of clusters, not the corpus.
        """
        cases = _mod().build_judge_cases(records, distractors=4)
        slates = {tuple(sorted(c['candidates'])) for c in cases}
        assert len(slates) > 1, 'every case was shown an identical slate'

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


def _run(tmp_path: Path, *, judge=None, corpus=None, distractors: int = 2):
    return _mod().run_judge_eval(
        records=corpus if corpus is not None else _corpus(),
        judge_fn=judge if judge is not None else _fake_judge(),
        report_path=tmp_path / 'report.json',
        provenance=dict(_PROVENANCE),
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
        row = next(ln for ln in md.splitlines() if ln.startswith(f'| {name} |'))
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
        header = next(ln for ln in self._md().splitlines() if ln.startswith('| class |'))
        assert self._cells(header) == list(self.COLUMNS)

    def test_emits_one_row_per_class_including_the_empty_ones(self) -> None:
        md = self._md()
        for name in _mod().EVAL_CLASSES:
            assert self._row_cells(md, name), name

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

    def test_no_accuracy_floor_is_compared_anywhere_in_the_module(self) -> None:
        """Asserted against the SOURCE, because this is a prohibition rather
        than a behaviour.

        A floor could be added in a branch no test happens to reach, and it
        would then be the 3169 gate without anyone having decided that. The
        scan is deliberately crude — it is a tripwire, and the right response
        to a false positive is to say in a comment why the comparison is not
        a floor.
        """
        import re  # noqa: PLC0415

        source = SCRIPT_PATH.read_text()
        offenders = re.findall(
            r'accuracy[^\n]{0,40}?[<>]=?\s*[0-9]', source,
        ) + re.findall(r'[0-9.]+\s*[<>]=?[^\n]{0,40}?accuracy', source)
        assert not offenders, f'an accuracy floor is being compared: {offenders}'


# ---------------------------------------------------------------------------
# The committed artifact
# ---------------------------------------------------------------------------

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
