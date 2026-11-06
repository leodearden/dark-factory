"""Tests for eval_write_triage_scoring.py — what counts as correct, and how a run reads.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf γ, decision D10.

The module is pure, so every test here is free of ``OPENAI_API_KEY``, network
and Qdrant. Input is hand-built, or built by eval_write_triage_cases.py, which
is loaded under its own file stem. No test asserts an accuracy floor (D10).
"""
from __future__ import annotations

import json
import os
import re

import pytest
from _eval_write_triage_fixtures import PROVENANCE as _PROVENANCE
from _eval_write_triage_fixtures import aliased_row as _aliased_row
from _eval_write_triage_fixtures import cases_mod as _cases_mod
from _eval_write_triage_fixtures import corpus as _corpus
from _eval_write_triage_fixtures import live as _live
from _eval_write_triage_fixtures import scoring_mod as _scoring

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
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
            'acceptable_outcomes': _scoring().ACCEPTABLE_OUTCOMES.get(
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
        assert tuple(sorted(TRIAGE_OUTCOMES)) == _scoring().EVAL_OUTCOMES

    def test_no_outcome_is_added_or_dropped_on_the_way(self) -> None:
        """Derived, not hand-written: a fifth outcome joins the report itself.

        A hand-spelled tuple would be a second list to keep in sync with
        `write_triage.TRIAGE_OUTCOMES` — the very drift `EVAL_CLASSES`' own
        comment says the shared list prevents.
        """
        assert set(_scoring().EVAL_OUTCOMES) == set(TRIAGE_OUTCOMES)


class TestScoreCases:
    """Counting only — every judgment call was made in the table above."""

    def test_per_class_reports_n_correct_and_accuracy(self) -> None:
        cases = _cases(('a', 'duplicate'), ('b', 'duplicate'), ('c', 'distinct'))
        got = _scoring().score_cases(cases, [OUTCOME_RESTATED, OUTCOME_STORED, OUTCOME_STORED])
        assert got['per_class']['duplicate'] == {'n': 2, 'correct': 1, 'accuracy': 0.5}
        assert got['per_class']['distinct'] == {'n': 1, 'correct': 1, 'accuracy': 1.0}

    def test_all_four_classes_are_always_present_with_an_explicit_n(self) -> None:
        """An omitted class reads identically to a perfect one.

        `distinct` (n=3) and `pseudo_contradiction` (n=6) are small enough
        that a construction bug could empty either without anything looking
        wrong, so the report must state the population it measured even when
        that population is zero.
        """
        got = _scoring().score_cases([], [])
        assert set(got['per_class']) == set(_scoring().EVAL_CLASSES)
        for name, entry in got['per_class'].items():
            assert entry['n'] == 0, name
            assert entry['correct'] == 0, name

    def test_an_empty_class_scores_none_never_zero(self) -> None:
        """`0.0` reads as "measured, and the judge failed everything"."""
        got = _scoring().score_cases([], [])
        for name, entry in got['per_class'].items():
            assert entry['accuracy'] is None, f'{name}: {entry!r}'

    def test_the_confusion_map_is_every_class_by_every_outcome(self) -> None:
        """Full 4xN, so a systematic failure is visible as a SHAPE.

        "Every duplicate answered `stored`" is a wiring bug and "duplicates
        split across the two attaches" is a working judge; a per-class
        accuracy alone cannot tell them apart.
        """
        cases = _cases(('a', 'duplicate'), ('b', 'pseudo_contradiction'))
        got = _scoring().score_cases(cases, [OUTCOME_STORED, OUTCOME_CONTESTED])
        assert set(got['confusion']) == set(_scoring().EVAL_CLASSES)
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
        got = _scoring().score_cases(cases, [OUTCOME_RESTATED, OUTCOME_STORED])
        for name, row in got['confusion'].items():
            assert list(row) == list(_scoring().EVAL_OUTCOMES), name

    def test_the_duplicate_split_is_a_distribution_not_an_error_term(self) -> None:
        """Reported, and deliberately not scored — see the table's rationale."""
        cases = _cases(('a', 'duplicate'), ('b', 'duplicate'), ('c', 'duplicate'))
        got = _scoring().score_cases(
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
        got = _scoring().score_cases(
            cases, [OUTCOME_CONTESTED, OUTCOME_CONTESTED, OUTCOME_STORED],
        )
        assert got['false_contested'] == 2

    def test_no_contested_verdict_scores_zero_not_none(self) -> None:
        """Unlike an empty class, this one WAS measured — 0 is the finding."""
        cases = _cases(('a', 'duplicate'))
        got = _scoring().score_cases(cases, [OUTCOME_RESTATED])
        assert got['false_contested'] == 0

    def test_the_result_round_trips_through_json(self) -> None:
        """It is written to disk verbatim; a set or a tuple in here is a
        TypeError at the end of a paid run.
        """
        cases = _cases(('a', 'duplicate'), ('b', 'distractor'))
        got = _scoring().score_cases(cases, [OUTCOME_RESTATED, OUTCOME_STORED])
        assert json.loads(json.dumps(got)) == got

    def test_a_verdict_count_mismatch_raises(self) -> None:
        """Silently zipping to the shorter list would drop cases from the
        denominator and report an accuracy over a population nobody chose.
        """
        with pytest.raises(Exception, match='(?i)verdict'):
            _scoring().score_cases(_cases(('a', 'duplicate'), ('b', 'distinct')), [OUTCOME_STORED])

    def test_the_distractor_class_is_scored_like_any_other(self) -> None:
        cases = _cases(('a', 'distractor'), ('b', 'distractor'))
        got = _scoring().score_cases(cases, [OUTCOME_STORED, OUTCOME_RESTATED])
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
        with pytest.raises(_scoring().UnknownLabelError) as excinfo:
            _scoring().score_cases(cases, [OUTCOME_STORED])
        message = str(excinfo.value)
        assert 'newly_invented_class' in message, message
        for name in _scoring().EVAL_CLASSES:
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
            _scoring().score_cases(cases, ['not_a_triage_outcome'])
        message = str(excinfo.value)
        assert 'not_a_triage_outcome' in message, message
        assert not isinstance(excinfo.value, _scoring().UnknownLabelError), (
            'a bad VERDICT is not an unknown LABEL — they name different holes'
        )


# ---------------------------------------------------------------------------
# Report assembly and rendering
# ---------------------------------------------------------------------------

def _inserted_span(longer: str, shorter: str) -> str:
    """The one contiguous span *longer* inserts into *shorter*.

    Fails unless inserting that span is the ONLY difference, so a caller can
    check what a clause says without the rest of the prose answering for it.
    """
    head = len(os.path.commonprefix([longer, shorter]))
    tail = len(os.path.commonprefix([longer[head:][::-1], shorter[head:][::-1]]))
    assert longer[:head] + longer[len(longer) - tail:] == shorter, (longer, shorter)
    return longer[head:len(longer) - tail]


class TestBuildReport:
    """The report is the deliverable — D10 makes it the operator's input."""

    def _report(self, **kwargs):
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        verdicts = [OUTCOME_RESTATED] * len(cases)
        return _scoring().build_report(
            scored=_scoring().score_cases(cases, verdicts),
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
        assert set(per_class) == set(_scoring().EVAL_CLASSES)
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
        keys = _scoring().PROVENANCE_KEYS
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
        report = _scoring().build_report(
            scored=_scoring().score_cases([], []),
            provenance={},
        )
        provenance = report['provenance']
        assert set(provenance) == set(_scoring().PROVENANCE_KEYS)
        for key in _scoring().PROVENANCE_KEYS:
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
        report = _scoring().build_report(
            scored=_scoring().score_cases([], []),
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

    def test_the_prompt_hash_is_provenance_vocabulary(self) -> None:
        assert 'judge_system_prompt_sha256' in _scoring().PROVENANCE_KEYS

    @staticmethod
    def _answering(
        cases: list[dict], label: str, answers, *,
        production_shape=None, judge_model=None,
    ) -> dict:
        """*cases* scored with the *label* cases answering in turn from
        *answers* and every other case answering `restated`."""
        answer = iter(answers)
        verdicts = [
            next(answer) if case['expected_class'] == label else OUTCOME_RESTATED
            for case in cases
        ]
        provenance = dict(_PROVENANCE)
        if judge_model is not None:
            provenance['judge_model'] = judge_model
        return _scoring().build_report(
            scored=_scoring().score_cases(cases, verdicts),
            provenance=provenance,
            production_shape=production_shape,
        )

    @staticmethod
    def _caveat_counting(report: dict, label: str) -> str:
        """The one caveat naming how many *label* cases answered `contested`."""
        k = report['confusion'][label][OUTCOME_CONTESTED]
        n = report['per_class'][label]['n']
        count = f'{k} of {n} {label}'
        naming = [caveat for caveat in report['caveats'] if count in caveat]
        assert len(naming) == 1, (count, report['caveats'])
        return naming[0]

    _COUNTED_LABELS = ('LABEL_PSEUDO_CONTRADICTION', 'LABEL_DUPLICATE')

    @pytest.mark.parametrize('label_name', _COUNTED_LABELS)
    def test_a_caveat_counts_the_contested_answers_in_the_class(
        self, label_name: str,
    ) -> None:
        label = getattr(_scoring(), label_name)
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        report = self._answering(cases, label, [OUTCOME_CONTESTED] * len(cases))
        assert report['per_class'][label]['n'] > 0, f'the corpus lost its {label} cases'
        self._caveat_counting(report, label)

    @pytest.mark.parametrize('label_name', _COUNTED_LABELS)
    def test_the_count_follows_the_verdicts(self, label_name: str) -> None:
        label = getattr(_scoring(), label_name)
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        contested = self._answering(cases, label, [OUTCOME_CONTESTED] * len(cases))
        amended = self._answering(cases, label, [OUTCOME_AMENDED] * len(cases))
        assert (
            self._caveat_counting(contested, label)
            != self._caveat_counting(amended, label)
        )

    @pytest.mark.parametrize('label_name', _COUNTED_LABELS)
    def test_the_judged_share_is_named_when_the_band_split_is_known(
        self, label_name: str,
    ) -> None:
        label = getattr(_scoring(), label_name)
        cases = _cases(*((f'c{i}', label) for i in range(6)))
        answers = [
            OUTCOME_RESTATED, OUTCOME_CONTESTED, OUTCOME_CONTESTED,
            OUTCOME_AMENDED, OUTCOME_AMENDED, OUTCOME_AMENDED,
        ]
        judged = 5
        shape = {'band_split': {label: {
            OUTCOME_RESTATED: 1, OUTCOME_JUDGE: judged, OUTCOME_STORED: 0,
        }}}

        with_split = self._caveat_counting(
            self._answering(cases, label, answers, production_shape=shape), label,
        )
        without_split = self._caveat_counting(
            self._answering(cases, label, answers), label,
        )
        added = _inserted_span(with_split, without_split)
        assert str(judged) in re.findall(r'\d+', added), added

    def test_the_duplicate_caveat_names_the_judge_it_measured(self) -> None:
        label = _scoring().LABEL_DUPLICATE
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        report = self._answering(
            cases, label, [OUTCOME_CONTESTED] * len(cases),
            judge_model='judge-under-test',
        )
        assert 'judge-under-test' in self._caveat_counting(report, label)


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
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        scored = _scoring().score_cases(cases, [verdict] * len(cases))
        return _scoring().render_markdown(
            _scoring().build_report(scored=scored, provenance=dict(_PROVENANCE)),
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
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        scored = _scoring().score_cases(cases, [OUTCOME_RESTATED] * len(cases))
        md = _scoring().render_markdown(
            _scoring().build_report(scored=scored, provenance=dict(_PROVENANCE)),
        )
        assert all(scored['per_class'][n]['n'] for n in _scoring().EVAL_CLASSES), (
            'the fixture must measure every class for the first half to bite'
        )
        for name in _scoring().EVAL_CLASSES:
            entry = scored['per_class'][name]
            assert self._row_cells(md, name) == {
                'class': name,
                'n': str(entry['n']),
                'correct': str(entry['correct']),
                'accuracy': str(entry['accuracy']),
            }, name

        # And the empty ones the name promises: nothing measured at all, yet
        # every class still renders a row, carrying `None` rather than `0.0`.
        empty_md = _scoring().render_markdown(
            _scoring().build_report(
                scored=_scoring().score_cases([], []), provenance=dict(_PROVENANCE),
            ),
        )
        for name in _scoring().EVAL_CLASSES:
            assert self._row_cells(empty_md, name) == {
                'class': name, 'n': '0', 'correct': '0', 'accuracy': 'None',
            }, name

    def test_the_row_carries_that_classes_own_numbers(self) -> None:
        cases = _cases_mod().build_judge_cases(_corpus(), distractors=2)
        scored = _scoring().score_cases(cases, [OUTCOME_RESTATED] * len(cases))
        md = _scoring().render_markdown(
            _scoring().build_report(scored=scored, provenance=dict(_PROVENANCE)),
        )
        cells = self._row_cells(md, 'duplicate')
        entry = scored['per_class']['duplicate']
        assert cells['n'] == str(entry['n'])
        assert cells['correct'] == str(entry['correct'])
        assert cells['accuracy'] == str(entry['accuracy'])

    def test_an_unmeasured_class_renders_its_none_rather_than_a_number(self) -> None:
        """`0.0` in this cell would read as a measured failure."""
        scored = _scoring().score_cases([], [])
        md = _scoring().render_markdown(
            _scoring().build_report(scored=scored, provenance=dict(_PROVENANCE)),
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
        assert self._cells(header) == ['class', *_scoring().EVAL_OUTCOMES]

    def test_each_confusion_row_lines_up_with_that_header(self) -> None:
        """A cell read by position is only a measurement if the columns bind."""
        md = self._md()
        section = self._section(md, 'Confusion')
        header = next(ln for ln in section.splitlines() if ln.startswith('| class |'))
        columns = self._cells(header)
        scored = _scoring().score_cases(
            _cases_mod().build_judge_cases(_corpus(), distractors=2),
            [OUTCOME_RESTATED] * len(_cases_mod().build_judge_cases(_corpus(), distractors=2)),
        )
        for name in _scoring().EVAL_CLASSES:
            row = next(
                ln for ln in section.splitlines() if ln.startswith(f'| {name} |')
            )
            cells = dict(zip(columns, self._cells(row), strict=True))
            for outcome in _scoring().EVAL_OUTCOMES:
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
        for caveat in _scoring().caveats_for(_scoring().SLATE_SEEDED):
            assert f'- {caveat}' in md, caveat

    def test_the_two_slate_modes_state_different_populations(self) -> None:
        """The seeded population caveat is FALSE of a retrieved run.

        It says `decide_band`, `t_high` and `t_low` never enter the picture
        and the band handed to `judge_write` is synthesized — which is exactly
        what the retrieved mode stops being true. A report carrying the wrong
        one tells the task-3169 operator its numbers cover a population they
        do not.
        """
        seeded = _scoring().caveats_for(_scoring().SLATE_SEEDED)
        retrieved = _scoring().caveats_for(_scoring().SLATE_RETRIEVED)
        assert seeded[:len(_scoring().CAVEATS)] == retrieved[:len(_scoring().CAVEATS)]
        assert seeded[len(_scoring().CAVEATS):] != retrieved[len(_scoring().CAVEATS):]
        assert set(_scoring().MODE_CAVEATS[_scoring().SLATE_SEEDED]).isdisjoint(retrieved)

    def test_a_report_naming_no_mode_reads_as_seeded(self) -> None:
        """Which is the only thing this script could do before retrieval."""
        assert _scoring().caveats_for(None) == _scoring().caveats_for(_scoring().SLATE_SEEDED)

    def test_renders_a_provenance_bullet_list(self) -> None:
        md = self._md()
        assert '## Provenance' in md
        assert '- `judge_model`: `gpt-4o-mini`' in md


# ---------------------------------------------------------------------------
# One case's answer, and where it attached
# ---------------------------------------------------------------------------

class TestAJudgeAnswerNamesOnlyAnAttach:
    """`JudgeAnswer.candidate_id` obeys the rule production's judge contract enforces."""

    def test_an_attach_verdict_carries_the_candidate_it_named(self) -> None:
        answer = _scoring().JudgeAnswer(
            outcome=OUTCOME_AMENDED, verdict=OUTCOME_AMENDED, candidate_id='x',
        )
        assert answer.candidate_id == 'x'

    def test_by_default_nothing_is_named(self) -> None:
        answer = _scoring().JudgeAnswer(outcome=OUTCOME_AMENDED, verdict=OUTCOME_AMENDED)
        assert answer.candidate_id is None

    def test_a_stored_verdict_naming_a_candidate_is_refused(self) -> None:
        """Production refuses it too: `stored` attaches to nothing."""
        with pytest.raises(ValueError):
            _scoring().JudgeAnswer(
                outcome=OUTCOME_STORED, verdict=OUTCOME_STORED, candidate_id='x',
            )

    def test_a_band_that_decided_itself_names_nothing(self) -> None:
        """No judge was asked, so no candidate was named."""
        with pytest.raises(ValueError):
            _scoring().JudgeAnswer(outcome=OUTCOME_RESTATED, verdict=None, candidate_id='x')


class TestScoreAttachments:
    """WHERE a case attached, which `score_cases` cannot see."""

    @staticmethod
    def _row(**kwargs) -> dict:
        row = {
            'memory_id': 'm', 'cluster_id': 'c', 'expected_class': 'duplicate',
            'candidates': ['c'], 'attach_target_id': 'c',
            'attach_target_cluster_id': None, 'attach_target_category': None,
            'attach_target_label': None, 'attach_target_is_canonical': True,
            'canonical_in_slate': True, 'canonical_present': True,
            'band': OUTCOME_JUDGE, 'similarity': 0.7, 'verdict': OUTCOME_AMENDED,
            'outcome': OUTCOME_AMENDED, 'correct': True, 'entry_elided': False,
            'candidates_elided': {}, 'usage': None,
        }
        row.update(kwargs)
        return row

    def test_an_attach_to_another_record_is_not_a_strict_attach(self) -> None:
        scored = _scoring().score_attachments([
            self._row(),
            self._row(memory_id='m2', attach_target_id='other',
                      attach_target_is_canonical=False),
        ])
        assert scored['duplicate_attach'] == {
            'n': 2, 'strict': 1, 'strict_rate': 0.5, 'any': 2, 'any_rate': 1.0,
        }

    def test_a_wrong_record_attach_names_where_it_landed(self) -> None:
        scored = _scoring().score_attachments([
            self._row(attach_target_id='other', attach_target_is_canonical=False,
                      attach_target_cluster_id='c9',
                      attach_target_category='preferences_and_norms'),
        ])
        wrong = scored['wrong_record_attach']
        assert wrong['n'] == 1 and wrong['rate_of_attaches'] == 1.0
        assert wrong['cases'][0]['attach_target_cluster_id'] == 'c9'
        assert wrong['cases'][0]['attach_target_category'] == 'preferences_and_norms'

    def test_stored_attaches_to_nothing(self) -> None:
        """`triage_write` nulls the canonical on `stored`, so it is no attach."""
        scored = _scoring().score_attachments([
            self._row(outcome=OUTCOME_STORED, verdict=OUTCOME_STORED, correct=False),
        ])
        assert scored['duplicate_attach']['any'] == 0
        assert scored['wrong_record_attach']['n'] == 0

    def test_a_deterministic_band_is_not_a_judge_call(self) -> None:
        scored = _scoring().score_attachments([
            self._row(band=OUTCOME_RESTATED, verdict=None, outcome=OUTCOME_RESTATED),
            self._row(memory_id='m2'),
        ])
        assert scored['judge_calls'] == 1
        assert scored['middle_band']['n'] == 1
        assert scored['band_split_all'] == {
            OUTCOME_RESTATED: 1, OUTCOME_JUDGE: 1, OUTCOME_STORED: 0,
        }

    def test_the_middle_band_accuracy_reads_the_same_correctness_rule(self) -> None:
        scored = _scoring().score_attachments([
            self._row(), self._row(memory_id='m2', correct=False),
        ])
        assert scored['middle_band'] == {'n': 2, 'correct': 1, 'accuracy': 0.5}

    def test_an_unmeasured_rate_is_none_never_zero(self) -> None:
        scored = _scoring().score_attachments([
            self._row(outcome=OUTCOME_STORED, verdict=OUTCOME_STORED),
        ])
        assert scored['wrong_record_attach']['rate_of_attaches'] is None

    def test_a_band_outside_the_vocabulary_raises(self) -> None:
        with pytest.raises(ValueError, match='outside'):
            _scoring().score_attachments([self._row(band='sideways')])

    def test_the_result_round_trips_through_json(self) -> None:
        scored = _scoring().score_attachments([self._row()])
        assert json.loads(json.dumps(scored)) == scored

    def test_an_attach_to_the_alias_is_strict_and_not_a_wrong_record(self) -> None:
        scored = _scoring().score_attachments([_aliased_row([_live('new-canon')], 'new-canon')])
        assert scored['duplicate_attach']['strict'] == 1
        assert scored['wrong_record_attach']['n'] == 0
