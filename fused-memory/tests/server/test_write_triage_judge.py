"""Unit tests for the middle-band write-triage judge (task 3128, PRD leaf gamma).

``server/write_triage_judge.py`` replaces leaf beta's ``_stub_judge`` at the
``tools.py`` wiring point. It DETECTS the relationship between a submitted
write and the candidates retrieval found — ``distinct``/``restates``/
``amends``/``contests`` — and never adjudicates which text is true.

Three properties this file exists to pin, none of which are incidental:

* the output vocabulary is CLOSED and derived from ``write_triage``'s own
  ``TRIAGE_OUTCOMES``, so a fifth outcome added to beta fails here rather than
  drifting into an unwired verdict;
* every failure RAISES, because ``triage_write`` owns the fail-open counting
  (INV-4) and a parser that silently defaulted to ``stored`` would make a
  broken judge indistinguishable from a healthy one;
* nothing is captured at import — the config resolvers read live, which is
  what makes the green-tier reload registration real rather than
  restart-only in disguise.

No test in this file needs an API key, a network, or Qdrant.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import write_triage_judge as judge_module
from fused_memory.server.grouped_read import AMENDMENT_KIND, PARENT_ID_KEY
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
    BandDecision,
    TriageFailOpenCounter,
    triage_write,
)
from fused_memory.server.write_triage_judge import (
    _DEFAULT_JUDGE_CANDIDATE_COUNT,
    _DEFAULT_JUDGE_ENABLED,
    _DEFAULT_JUDGE_MODEL,
    _DEFAULT_JUDGE_PROVIDER,
    _DEFAULT_JUDGE_TIMEOUT_SECONDS,
    _DEFAULT_MODEL_BY_PROVIDER,
    _ELIDED_MARKER,
    _FIELD_CHARS,
    _JUDGE_MAX_TOKENS,
    _KNOWN_PROVIDERS,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_VERDICTS,
    VERDICT_KEY,
    JudgeOutputError,
    _call_llm,
    _provider_credentials,
    build_judge_prompt,
    judge_write,
    parse_judge_verdict,
    resolve_judge_candidate_count,
    resolve_judge_enabled,
    resolve_judge_model,
    resolve_judge_provider,
    resolve_judge_timeout,
    select_judge_candidates,
)
from fused_memory.services.memory_service import RRF_K, SearchResults


def _payload(word: object) -> str:
    """A well-formed verdict payload keyed off the module's own VERDICT_KEY.

    Built rather than spelled, so a rename of the key is exercised by these
    tests instead of silently bypassing them.
    """
    return json.dumps({VERDICT_KEY: word})


# ---------------------------------------------------------------------------
# the closed 4-way vocabulary
# ---------------------------------------------------------------------------


class TestJudgeVerdictVocabulary:
    """The judge speaks four words and they map onto beta's ack vocabulary.

    The mapping's VALUE set is asserted against ``TRIAGE_OUTCOMES`` rather
    than restated as a literal set, so a fifth outcome added to leaf beta
    fails HERE — at the one place that would have to grow a fifth judge word
    and a fifth attach kind — instead of silently arriving as an
    out-of-vocabulary verdict counted as a fail-open forever.
    """

    def test_the_four_judge_words_are_exactly_these(self) -> None:
        """The judge-facing side of the mapping is closed at four words."""
        assert set(JUDGE_VERDICTS) == {'distinct', 'restates', 'amends', 'contests'}

    def test_the_mapped_values_are_exactly_the_triage_outcomes(self) -> None:
        """Derived from beta, not restated — a fifth outcome fails here."""
        assert set(JUDGE_VERDICTS.values()) == set(TRIAGE_OUTCOMES)
        assert len(JUDGE_VERDICTS) == len(TRIAGE_OUTCOMES)

    @pytest.mark.parametrize(
        ('word', 'outcome'),
        [
            ('distinct', OUTCOME_STORED),
            ('restates', OUTCOME_RESTATED),
            ('amends', OUTCOME_AMENDED),
            ('contests', OUTCOME_CONTESTED),
        ],
        ids=['distinct', 'restates', 'amends', 'contests'],
    )
    def test_each_word_maps_to_its_outcome(self, word: str, outcome: str) -> None:
        """Each judge word lands on the outcome the ack contract publishes."""
        assert JUDGE_VERDICTS[word] == outcome


# ---------------------------------------------------------------------------
# the worked examples that teach the vocabulary
# ---------------------------------------------------------------------------

#: The committed curator corpus the judge is MEASURED against. Named here so
#: the leakage guard below can ask whether an exemplar was drawn from it.
CALIBRATION_FIXTURE_PATH = (
    Path(__file__).parent.parent / 'fixtures' / 'write_triage_calibration.jsonl'
)


@pytest.fixture(scope='module')
def records() -> list[dict]:
    """The committed curator corpus, parsed with the stdlib.

    Parsed here rather than through the eval script's loader, so a loader bug
    cannot mask a data defect (and vice versa) — the discipline the sibling
    suite's own ``records`` fixture states.
    """
    assert CALIBRATION_FIXTURE_PATH.exists(), (
        f'fixture missing: {CALIBRATION_FIXTURE_PATH}'
    )
    return [
        json.loads(line)
        for line in CALIBRATION_FIXTURE_PATH.read_text().splitlines()
        if line.strip()
    ]


class TestJudgeExemplars:
    """The vocabulary's worked examples, held as DATA rather than as prose.

    Four words with no worked example is what the 2026-08-27 measurement
    indicts: 31 of 75 duplicates were answered ``stored`` with the correct
    canonical sitting in the slate, i.e. ``restates``/``amends`` were
    under-produced. A vocabulary word the model has never seen USED is the one
    it under-produces, so full verdict coverage is the invariant that targets
    the defect rather than a tidiness rule.

    Structured records, not a pre-formatted blob (heuristic 12): declaring
    ``entry``/``candidate``/``verdict`` as separate fields is what lets
    vocabulary closure, verdict coverage and corpus disjointness be CHECKED
    here instead of grepped for in a string. The renderer owns the formatting.
    """

    def test_the_exemplars_are_structured_records(self) -> None:
        """Three separate fields per record — the data carries no formatting.

        A pre-formatted blob would reduce every assertion below to substring
        grepping, and would move the prompt's layout out of the renderer and
        into the data, where two exemplars can disagree about it.
        """
        exemplars = judge_module.JUDGE_EXEMPLARS
        assert isinstance(exemplars, tuple), 'exemplars are an ordered, frozen tuple'
        assert exemplars, 'an empty exemplar tuple teaches nothing'
        for exemplar in exemplars:
            fields = (exemplar.entry, exemplar.candidate, exemplar.verdict)
            for field in fields:
                assert isinstance(field, str) and field.strip(), (
                    f'every field is a non-empty string: {exemplar!r}'
                )
                assert '\n' not in field, (
                    f'line breaks are the renderer\'s business, not the data\'s: '
                    f'{exemplar!r}'
                )
            assert len(set(fields)) == len(fields), (
                f'the three fields are distinct values, not one blob repeated: '
                f'{exemplar!r}'
            )

    def test_every_exemplar_verdict_is_in_the_closed_vocabulary(self) -> None:
        """An out-of-vocabulary exemplar teaches a word the parser REJECTS.

        ``parse_judge_verdict`` raises on anything outside ``JUDGE_VERDICTS``
        and ``write_triage`` counts that raise as a fail-open — so the damage
        surfaces as a storm escalation describing an outage, not as a bad
        verdict anyone would trace back to a typo in a prompt example.
        """
        for exemplar in judge_module.JUDGE_EXEMPLARS:
            assert exemplar.verdict in JUDGE_VERDICTS, (
                f'{exemplar.verdict!r} is not one of {sorted(JUDGE_VERDICTS)}'
            )

    def test_every_verdict_has_at_least_one_worked_example(self) -> None:
        """Coverage is the invariant aimed at the measured defect.

        Derived from ``JUDGE_VERDICTS`` rather than spelled as four literals,
        so a fifth word added to the vocabulary arrives here already demanding
        its example instead of shipping unexemplified.
        """
        covered = {exemplar.verdict for exemplar in judge_module.JUDGE_EXEMPLARS}
        assert covered == set(JUDGE_VERDICTS), (
            f'verdicts with no worked example: {sorted(set(JUDGE_VERDICTS) - covered)}'
        )

    def test_no_exemplar_text_is_drawn_from_the_eval_corpus(
        self, records: list[dict],
    ) -> None:
        """Exemplars are prompt content; fixture records are a MEASUREMENT.

        Hand-writing an exemplar is ordinary prompt engineering — nothing is
        scored against it. Drawing one from the corpus the judge is scored on
        is training on the test set, and would make the accuracy report
        unreadable as evidence. This is the one way an exemplar can corrupt a
        measurement, so it is asserted rather than remembered.
        """
        corpus = '\n'.join(str(record.get('content', '')) for record in records)
        assert corpus.strip(), 'the corpus parsed empty — the guard would be vacuous'
        for exemplar in judge_module.JUDGE_EXEMPLARS:
            for field_name in ('entry', 'candidate'):
                text = getattr(exemplar, field_name)
                assert text not in corpus, (
                    f'exemplar {field_name} is drawn from the eval corpus — '
                    f'that is training on the test set: {text!r}'
                )

    def test_each_exemplar_renders_as_an_answered_pair(self) -> None:
        """The PAIRING is the property. The three fields separately are not.

        A pair rendered without its verdict is a riddle, and a verdict
        rendered without its pair is an assertion — so what has to hold is
        that each exemplar's three fields reach the model AS ONE BLOCK.
        Checking the fields individually cannot see that: all four verdict
        words already appear in the vocabulary bullets above the examples, so
        ``exemplar.verdict in prompt`` is true whatever the renderer emits.
        Measured by simulation — a ``_render_exemplars`` that dropped its
        ``answer:`` line entirely, shipping four unanswered riddles, left the
        per-field version of this test green.

        Asserted as the contiguous triple, which is executable structure and
        not a wording pin. It restates ``_render_exemplars``' block layout on
        purpose — that layout IS the contract between the tuple and the model
        — while leaving the prompt's prose around the examples free to be
        reworded. It subsumes the per-field presence check, so there is no
        longer a separate one.
        """
        for exemplar in judge_module.JUDGE_EXEMPLARS:
            block = (
                f'new entry: {exemplar.entry}\n'
                f'candidate: {exemplar.candidate}\n'
                f'answer: {exemplar.verdict}'
            )
            assert block in JUDGE_SYSTEM_PROMPT, (
                f'exemplar does not reach the model as an ANSWERED pair — its '
                f'fields may all be present but not together: {block!r}'
            )

    def test_the_exemplars_render_once_each_in_declaration_order(self) -> None:
        """A REPRODUCIBLE measurement needs a prompt that does not move.

        This suite has been bitten once already by an iteration order moving
        between two processes — the committed `.json`/`.md` confusion-row
        disagreement that
        ``test_the_committed_markdown_is_the_render_of_the_committed_json``
        now pins. A tuple cannot reorder itself, so what is left to check is
        that the RENDERER walks it in order and does not double-render.

        Asserted on ``entry``, which uniquely identifies an exemplar and
        occurs nowhere else in the prompt. ``candidate`` and ``verdict``
        deliberately recur — one candidate is shared by all four exemplars, so
        the only variable is the relationship, and each verdict word already
        appears three to five times in the vocabulary section above the
        examples. Counting occurrences of either would measure the prompt's
        prose, not the renderer's determinism.
        """
        prompt = JUDGE_SYSTEM_PROMPT
        positions = []
        for exemplar in judge_module.JUDGE_EXEMPLARS:
            assert prompt.count(exemplar.entry) == 1, (
                f'exemplar rendered {prompt.count(exemplar.entry)} times, not '
                f'once: {exemplar.entry!r}'
            )
            positions.append(prompt.index(exemplar.entry))
        assert positions == sorted(positions), (
            f'the render walks JUDGE_EXEMPLARS out of declaration order: '
            f'{positions}'
        )

    def test_the_user_prompt_carries_no_exemplar_text(self) -> None:
        """Exemplars are constant, so they are paid for ONCE, system-side.

        Two reasons beyond the token bill. ``scripts/check_write_triage_attach_target.py``
        is the behavioural gate probe for flip-predicate item 1; its
        ``_echoes_argument`` / ``_swap_verdict`` controls attribute a
        rendering difference to a SPECIFIC candidate in the user turn, and
        constant example lines there would be extra material those controls
        would have to reason around. And the system half is the half a
        provider can cache — rendered per call, the exemplars would be paid
        for on all 102 cases of an eval run instead of once.

        Verdict WORDS are excluded: ``build_judge_prompt`` names the closed
        vocabulary by design, which is a different thing from carrying an
        example.
        """
        candidates = [
            _result('mem-aaa', 0.9, content='first candidate body'),
            _result('mem-bbb', 0.8, content='second candidate body'),
        ]
        prompt = build_judge_prompt('the new entry text', candidates)
        for exemplar in judge_module.JUDGE_EXEMPLARS:
            for field_name in ('entry', 'candidate'):
                text = getattr(exemplar, field_name)
                assert text not in prompt, (
                    f'exemplar {field_name} leaked into the per-call user turn: '
                    f'{text!r}'
                )

    def test_the_worst_case_prompt_stays_within_the_char_budget(self) -> None:
        """PRD C1 bounds the whole call, and the exemplars spend against it.

        The worst case is not hypothetical: the calibration fixture holds a
        ~9k-char canonical, so a full slate of over-long candidates plus an
        over-long entry is what a real call looks like when the corpus is at
        its largest. Built rather than arithmetic, so the scaffolding between
        the fields is counted too.

        BUILT THE WAY ``judge_write`` CALLS IT, which is the whole point —
        a construction the production path never makes bounds nothing. Two
        details were missing when this test used 5-char stand-in ids and no
        attach target, and together they cost 209 chars, enough to put the
        real call over a budget this test reported as met. Both are now
        asserted rather than assumed, because either could be quietly undone
        by an edit that still left the test green: candidate ids are the
        36-char uuids every record actually carries (all 104 in
        ``tests/fixtures/write_triage_calibration.jsonl`` are, and
        ``build_judge_prompt`` renders ``- id:`` UN-elided), and
        ``attach_target_id`` is passed, because ``judge_write`` forwards it on
        EVERY call — so its line is part of the worst case, not an extra.

        THE FIELDS ARE OVER ``_FIELD_CHARS``, NOT AT IT. ``_elide`` returns a
        field of exactly ``_FIELD_CHARS`` untouched and cuts a longer one to
        ``_FIELD_CHARS`` PLUS ``_ELIDED_MARKER`` — so the input that elides
        renders 9 chars wider per field, 54 across a full slate, than the
        input that merely fills. A worst case built at the cap is therefore
        not the worst case; it is the widest input that never trips the
        behaviour this budget exists to bound.

        The ceiling is a module constant, not a literal here, so the budget
        has one home — raising it is an edit to the thing being budgeted,
        made next to the C1 rationale, rather than a number quietly relaxed in
        a test.
        """
        maximal = 'x' * (_FIELD_CHARS + 1)
        candidates = [
            _result(str(uuid.uuid4()), 0.9, content=maximal)
            for _ in range(_DEFAULT_JUDGE_CANDIDATE_COUNT)
        ]
        assert {len(c.id) for c in candidates} == {36}, (
            'the slate must carry the 36-char uuids production carries — a '
            'shorter stand-in id under-measures every candidate line'
        )
        rendered = build_judge_prompt(
            maximal, candidates, attach_target_id=candidates[0].id,
        )
        assert _ELIDED_MARKER in rendered, (
            'the worst case must be an ELIDED render — otherwise it misses '
            'the marker _elide appends, and under-measures the real ceiling'
        )
        assert f'  attach_target: {candidates[0].id}' in rendered, (
            'the attach_target line is rendered on every production call, so '
            'a worst case measured without it is not the worst case'
        )
        worst_case = len(JUDGE_SYSTEM_PROMPT) + len(rendered)
        assert worst_case <= judge_module._PROMPT_CHAR_BUDGET, (
            f'worst-case prompt is {worst_case} chars against a budget of '
            f'{judge_module._PROMPT_CHAR_BUDGET}'
        )


class TestParseJudgeVerdict:
    """Parsing is where the closed output is ENFORCED (D3).

    A schema-forcing transport can still return a refusal, so the parse
    boundary is the only enforcement that always runs. Every rejection path
    raises ``JudgeOutputError``; none of them default to ``stored``, because
    ``triage_write`` counts a fail-open only for something that raises or
    returns out-of-vocabulary, and a silent default would make a broken judge
    read exactly like a healthy one answering "nothing matched".
    """

    @pytest.mark.parametrize(
        ('word', 'outcome'),
        [
            ('distinct', OUTCOME_STORED),
            ('restates', OUTCOME_RESTATED),
            ('amends', OUTCOME_AMENDED),
            ('contests', OUTCOME_CONTESTED),
        ],
        ids=['distinct', 'restates', 'amends', 'contests'],
    )
    def test_a_bare_json_object_round_trips(self, word: str, outcome: str) -> None:
        """The happy path: exactly what `response_format=json_object` returns."""
        assert parse_judge_verdict(_payload(word)) == outcome

    @pytest.mark.parametrize(
        ('word', 'outcome'),
        [
            ('Restates', OUTCOME_RESTATED),
            (' restates ', OUTCOME_RESTATED),
            ('RESTATES', OUTCOME_RESTATED),
            ('\nAmends\n', OUTCOME_AMENDED),
            ('  Contests', OUTCOME_CONTESTED),
            ('Distinct\t', OUTCOME_STORED),
        ],
        ids=['title-case', 'padded', 'upper', 'newline-wrapped',
             'leading-space', 'trailing-tab'],
    )
    def test_case_and_whitespace_are_normalised(
        self, word: str, outcome: str,
    ) -> None:
        """`.strip().lower()` is load-bearing, and nothing else pinned it.

        Every other case in this class is already bare lowercase, so deleting
        the normalisation left the whole suite green while turning a model
        that answered `"Restates"` — an entirely reasonable thing for an LLM
        to emit through a JSON schema that does not enumerate the casing —
        into a `JudgeOutputError` on EVERY middle-band write. That is a
        counted fail-open per write, which surfaces as a storm escalation
        describing an outage that is not happening.
        """
        assert parse_judge_verdict(_payload(word)) == outcome

    def test_a_fenced_json_block_parses(self) -> None:
        """A model that ignores the JSON mode and fences its answer still parses."""
        raw = f'```json\n{_payload("amends")}\n```'
        assert parse_judge_verdict(raw) == OUTCOME_AMENDED

    def test_json_surrounded_by_prose_parses(self) -> None:
        """`extract_json` brace-scans, so leading/trailing prose is tolerated."""
        raw = (
            'Looking at candidate mem-1, the new entry adds a detail.\n'
            f'{_payload("amends")}\n'
            'That is my answer.'
        )
        assert parse_judge_verdict(raw) == OUTCOME_AMENDED

    def test_a_verdict_with_extra_keys_still_parses(self) -> None:
        """Extra keys are ignored; only the verdict word is contractual."""
        raw = json.dumps({VERDICT_KEY: 'restates', 'reasoning': 'same fact'})
        assert parse_judge_verdict(raw) == OUTCOME_RESTATED

    def test_an_array_wrapped_object_reads_as_that_object(self) -> None:
        """Pinned deliberately, because it is a behaviour and not an accident.

        ``extract_json`` brace-scans for the first BALANCED OBJECT, so an
        answer wrapped in a list is read as the object inside it. That
        leniency is inherited from the repo's one JSON extractor rather than
        chosen here, and it is safe at this seam for a specific reason: the
        BAND names the attach target, not the judge, so a verdict is a single
        word however it was wrapped. A model that answered with several
        elements has its first taken — a bounded wrong-word risk against a
        target that is fixed either way, with the canonical never mutated and
        the attach always re-parentable (C1).

        Asserted so that a future change to ``extract_json`` surfaces here
        rather than silently changing what the judge is understood to have
        said.
        """
        raw = f'[{_payload("restates")}]'
        assert parse_judge_verdict(raw) == OUTCOME_RESTATED

    @pytest.mark.parametrize(
        ('raw', 'label'),
        [
            (_payload('supersedes'), 'unrecognised-verdict-word'),
            (_payload(OUTCOME_RESTATED), 'ack-word-not-judge-word'),
            (json.dumps({'answer': 'restates'}), 'missing-verdict-key'),
            (_payload(None), 'null-verdict'),
            (_payload(['restates']), 'non-string-verdict'),
            ('["restates"]', 'non-object-payload'),
            ('"restates"', 'bare-string-payload'),
            ('', 'empty-text'),
            ('   \n\t  ', 'whitespace-only-text'),
            ('I cannot answer that request.', 'prose-with-no-json'),
            (_payload('restates').rstrip('}'), 'unbalanced-json'),
            (f'{{{VERDICT_KEY}: restates}}', 'unquoted-json'),
        ],
        ids=[
            'unrecognised-verdict-word',
            'ack-word-not-judge-word',
            'missing-verdict-key',
            'null-verdict',
            'non-string-verdict',
            'non-object-payload',
            'bare-string-payload',
            'empty-text',
            'whitespace-only-text',
            'prose-with-no-json',
            'unbalanced-json',
            'unquoted-json',
        ],
    )
    def test_every_rejection_raises_rather_than_defaulting(
        self, raw: str, label: str,
    ) -> None:
        """A raise is what `triage_write`'s fail-open counter can SEE."""
        with pytest.raises(JudgeOutputError):
            parse_judge_verdict(raw)

    def test_the_rejection_message_quotes_the_offending_payload(self) -> None:
        """An operator reading the fail-open log needs the actual output."""
        with pytest.raises(JudgeOutputError) as excinfo:
            parse_judge_verdict(_payload('supersedes'))
        assert 'supersedes' in str(excinfo.value)

    def test_a_pathological_payload_is_truncated_in_the_message(self) -> None:
        """The judge output reaches a log line; an unbounded one is a hazard."""
        raw = 'x' * 20_000
        with pytest.raises(JudgeOutputError) as excinfo:
            parse_judge_verdict(raw)
        assert len(str(excinfo.value)) < 2_000

    def test_a_non_string_input_raises_rather_than_crashing(self) -> None:
        """An SDK that returned None for the body is a judge failure, not a TypeError."""
        with pytest.raises(JudgeOutputError):
            parse_judge_verdict(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# candidate selection + prompt rendering
# ---------------------------------------------------------------------------

# The real post-RRF relevance_score for a rank-1 mem0 hit — an ORDINAL, not a
# cosine. Spelled the same way `test_write_triage.py` spells it, so a test
# that passes only because something read `relevance_score` instead of the
# per-store cosine fails here too.
_RRF_RANK1 = 1.0 / (RRF_K + 1)


def _result(
    id_: str,
    score: float | None,
    *,
    content: str = 'some procedural content',
    relevance_score: float = _RRF_RANK1,
    omit_store_score: bool = False,
    extra_metadata: dict | None = None,
) -> MemoryResult:
    """A POST-RRF ``MemoryResult``: *score* is the COSINE, in metadata.

    Same shape as ``test_write_triage.py::_result`` — kept local rather than
    imported across suites, matching how the two triage suites already stand
    alone.
    """
    metadata: dict = {'store_rank': 1}
    if not omit_store_score:
        metadata['store_score'] = score
    if extra_metadata:
        metadata.update(extra_metadata)
    return MemoryResult(
        id=id_,
        content=content,
        category=MemoryCategory.procedural_knowledge,
        source_store=SourceStore.mem0,
        relevance_score=relevance_score,
        metadata=metadata,
    )


def _decision(canonical_id: str | None, similarity: float | None = 0.80) -> BandDecision:
    """A middle-band decision naming *canonical_id* as the attach target."""
    return BandDecision(OUTCOME_JUDGE, canonical_id, similarity, 0.95, 0.70)


#: The repo root, reached from `<repo>/fused-memory/tests/server/`.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The flip gate's own attach-target checker, and the tree it reads.
_PROBE_PATH = _REPO_ROOT / 'scripts' / 'check_write_triage_attach_target.py'
_JUDGE_SRC_ROOT = _REPO_ROOT / 'fused-memory' / 'src'


def _marked_ids(candidates: list[MemoryResult], attach_target_id: str) -> set[str]:
    """Which candidate ids the attach-target mark names, read from the DIFF.

    Diffed against the same slate rendered with no target rather than grepped
    for a mark's spelling: what the invariant asserts is that naming a target
    changes the rendering in a way attributable to a specific candidate, which
    is exactly what `check_write_triage_attach_target.py::_swap_verdict`
    measures. A test keyed on the literal mark text would instead pin the
    mechanism and pass for a marker that names the wrong record.
    """
    unmarked = build_judge_prompt('new', candidates, attach_target_id=None)
    marked = build_judge_prompt(
        'new', candidates, attach_target_id=attach_target_id,
    )
    added = set(marked.splitlines()) - set(unmarked.splitlines())
    return {
        candidate.id
        for candidate in candidates
        for line in added
        if candidate.id in line
    }


def _added_lines(candidates: list[MemoryResult], attach_target_id: str) -> list[str]:
    """The lines naming a target ADDS to the rendering, in order.

    A LIST, not a set. `_marked_ids` answers "which candidates does the mark
    name", which is the right question for a marker that names the wrong
    record — but it cannot see a marker that names TWO records whose ids
    happen to collapse, and a set-valued assertion reads the same either way.
    The count is its own invariant: AT MOST ONE candidate is ever marked, so
    a second mark has to show up as a visible extra element rather than be
    absorbed. Spelling-independent — it diffs against the same slate rendered
    with no target instead of grepping for the mark's text.
    """
    unmarked = build_judge_prompt(
        'new', candidates, attach_target_id=None,
    ).splitlines()
    marked = build_judge_prompt(
        'new', candidates, attach_target_id=attach_target_id,
    ).splitlines()
    return [line for line in marked if line not in unmarked]


class TestSelectJudgeCandidates:
    """Which of the retrieved results the judge actually gets to see.

    ``triage_write`` hands the judge the WHOLE ``SearchResults`` object
    (un-transformed, so `degraded`/`failed_stores` survive). Trimming to the
    PRD's top 3-5 is the judge's own job, and this is where it happens.
    """

    def test_at_most_n_candidates_are_returned(self) -> None:
        """The width is a cap, and the prompt budget depends on it holding."""
        results = [_result(f'm{i}', 0.90 - i / 100) for i in range(12)]
        selected = select_judge_candidates(results, 5, canonical_id='m0')
        assert len(selected) == 5

    def test_candidates_are_ordered_by_descending_cosine(self) -> None:
        """Highest per-store cosine first: the judge reads the strongest evidence first."""
        results = [_result('lo', 0.62), _result('hi', 0.95), _result('mid', 0.70)]
        selected = select_judge_candidates(results, 3, canonical_id='hi')
        assert [r.id for r in selected] == ['hi', 'mid', 'lo']

    def test_the_ordering_follows_store_score_not_relevance_score(self) -> None:
        """The RRF ordinal is NOT a similarity — the same trap `decide_band` avoids.

        Mirrors ``test_write_triage.py::test_the_winner_is_not_the_max_relevance_score``:
        `relevance_score` is deliberately inverted against the cosine here, so
        a selector that sorted on it produces the opposite order and fails.
        """
        results = [
            _result('rank1', 0.61, relevance_score=1.0 / (RRF_K + 1)),
            _result('rank2', 0.96, relevance_score=1.0 / (RRF_K + 2)),
        ]
        selected = select_judge_candidates(results, 2, canonical_id='rank2')
        assert [r.id for r in selected] == ['rank2', 'rank1']

    @pytest.mark.parametrize(
        ('label', 'uncomparable'),
        [
            ('store_score key absent', _result('pin', None, omit_store_score=True)),
            ('store_score is None', _result('pin', None)),
            ('store_score is a bool', _result('pin', True)),  # type: ignore[arg-type]
            ('store_score is a string', _result('pin', '0.99')),  # type: ignore[arg-type]
        ],
        ids=['absent', 'none', 'bool', 'string'],
    )
    def test_an_uncomparable_candidate_is_dropped(
        self, label: str, uncomparable: MemoryResult,
    ) -> None:
        """A record with no numeric cosine spends a judge slot for nothing.

        A topic-anchored pin (services/topic_anchor.py) deliberately carries
        no ``store_score``. ``decide_band`` already drops it because it can
        never clear a threshold; the judge drops it for the adjacent reason —
        there is no measured similarity to put in front of the model, and the
        slot is better spent on a record that can actually be compared.
        """
        results = [uncomparable, _result('m1', 0.80)]
        selected = select_judge_candidates(results, 5, canonical_id='m1')
        assert [r.id for r in selected] == ['m1'], label

    def test_the_bands_winner_is_always_present(self) -> None:
        """A judge asked about a set excluding the attach target cannot answer.

        The verdict routes the write to `decision.canonical_id` and nowhere
        else (D3: the band names the target, the judge only names the
        relationship). If that record is not in the prompt, every verdict is
        about a different memory than the one the attach will touch.
        """
        results = [_result(f'm{i}', 0.95 - i / 100) for i in range(8)]
        # 'm7' is the weakest, so a plain top-3 would drop it.
        selected = select_judge_candidates(results, 3, canonical_id='m7')
        assert 'm7' in [r.id for r in selected]
        assert len(selected) <= 3

    def test_the_winner_is_present_even_when_hoisted(self) -> None:
        """`_canonical_id_of` HOISTS a child winner to its parent id.

        So `decision.canonical_id` can name a record that is not itself in the
        result set at all. The parent is what the attach targets, so when the
        hoisted id is absent the CHILD that carried the evidence must stay —
        dropping both would leave the judge with no view of the match at all.

        The child is scored BELOW every peer on purpose. Scored above them it
        is plain top-1, an unconditional `sorted(...)[:n]` returns the same
        slate, and the branch this test names is never the reason it passes —
        which is why deleting the `PARENT_ID_KEY` fallback outright used to
        leave the whole suite green. At 0.55 against six records at
        0.90..0.85 the rescue arm is the ONLY thing that can put it in.

        The eviction victim is asserted too. The sibling
        `test_the_bands_winner_is_always_present` checks membership and
        `len <= n`, so a rescue that dropped the STRONGEST candidate instead
        of the weakest would satisfy both it and a bare `in` check here.
        """
        child = _result(
            'child-1', 0.55,
            extra_metadata={'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1'},
        )
        peers = [_result(f'm{i}', 0.90 - i / 100) for i in range(6)]
        selected = select_judge_candidates(
            [child, *peers], 3, canonical_id='parent-1',
        )
        ids = [r.id for r in selected]
        assert 'child-1' in ids
        # The rescue EVICTS rather than widens: still exactly n.
        assert len(selected) == 3
        # And it evicts the WEAKEST of the window (m2 at 0.88), not an
        # arbitrary one — m0 and m1 are the two strongest and both survive.
        assert ids == ['m0', 'm1', 'child-1']

    def test_an_empty_input_returns_empty_without_raising(self) -> None:
        """Nothing to compare is a decision, not a failure."""
        assert select_judge_candidates([], 5, canonical_id=None) == []

    def test_a_wholly_uncomparable_input_returns_empty(self) -> None:
        """A slate of pins yields no candidates and still does not raise."""
        results = [_result(f'pin{i}', None, omit_store_score=True) for i in range(4)]
        assert select_judge_candidates(results, 5, canonical_id='pin0') == []

    def test_a_search_results_object_is_accepted(self) -> None:
        """`triage_write` passes the un-transformed SearchResults, not a list.

        That object is a sequence of `MemoryResult` carrying `degraded` and
        `failed_stores` alongside; the selector must iterate it as given
        rather than requiring the caller to slice it (which would drop those
        fields — the failure mode `retrieve_candidates` warns about).
        """
        results = SearchResults([_result('m1', 0.80), _result('m2', 0.60)])
        selected = select_judge_candidates(results, 5, canonical_id='m1')
        assert [r.id for r in selected] == ['m1', 'm2']


class TestBuildJudgePrompt:
    """What actually reaches the model — and, as loudly, what must not."""

    def test_the_submitted_content_is_rendered(self) -> None:
        prompt = build_judge_prompt('the new entry text', [_result('m1', 0.9)])
        assert 'the new entry text' in prompt

    def test_every_candidate_id_and_text_is_rendered(self) -> None:
        candidates = [
            _result('mem-aaa', 0.9, content='first candidate body'),
            _result('mem-bbb', 0.8, content='second candidate body'),
        ]
        prompt = build_judge_prompt('new', candidates)
        for candidate in candidates:
            assert candidate.id in prompt
            assert candidate.content in prompt

    def test_the_four_closed_verdicts_are_named(self) -> None:
        """A model cannot answer in a vocabulary it was never shown."""
        prompt = build_judge_prompt('new', [_result('m1', 0.9)])
        for word in JUDGE_VERDICTS:
            assert word in prompt

    def test_the_prompt_asks_for_a_bare_json_object(self) -> None:
        """The parser's happy path is what the instructions must request."""
        prompt = build_judge_prompt('new', [_result('m1', 0.9)])
        assert VERDICT_KEY in prompt
        assert 'JSON' in prompt or 'json' in prompt

    def test_a_long_candidate_is_truncated_and_marked(self) -> None:
        """The fixture contains a ~9k-char canonical; the budget is ~2.5k tokens.

        Truncating silently would be worse than not truncating: a model told
        nothing would treat a severed sentence as the whole record. The elided
        marker is what keeps the cut honest.
        """
        long_body = 'x' * 9_000
        prompt = build_judge_prompt('new', [_result('m1', 0.9, content=long_body)])
        assert long_body not in prompt
        assert len(prompt) < 9_000
        assert _ELIDED_MARKER in prompt

    def test_a_long_submitted_content_is_truncated_and_marked(self) -> None:
        """The submitted side is bounded for the same reason as the candidates."""
        long_body = 'y' * 9_000
        prompt = build_judge_prompt(long_body, [_result('m1', 0.9)])
        assert long_body not in prompt
        assert _ELIDED_MARKER in prompt

    def test_short_content_is_not_marked_as_elided(self) -> None:
        """The marker means something only if it is absent when nothing was cut."""
        prompt = build_judge_prompt('short', [_result('m1', 0.9, content='also short')])
        assert _ELIDED_MARKER not in prompt

    def test_no_repo_or_task_context_reaches_the_model(self) -> None:
        """PRD C1: the judge sees CONTENT, and nothing about who wrote it.

        Structural rather than incidental — ``build_judge_prompt`` interpolates
        no metadata at all — so this holds for a candidate carrying every
        context key a real record can. Rendered with all of them present, so a
        future edit that started quoting `metadata` fails here.
        """
        leaky = _result(
            'mem-1', 0.9,
            content='candidate body',
            extra_metadata={
                'task_id': '3128',
                'agent_id': 'claude-task-3128-implementer',
                'project_id': 'dark_factory',
                'source': 'fused-memory/src/fused_memory/server/tools.py',
                'topic': 'write-triage',
            },
        )
        prompt = build_judge_prompt('new entry', [leaky])
        for forbidden in (
            'task_id', '3128',
            'agent_id', 'claude-task-3128-implementer',
            'project_id', 'dark_factory',
            'fused-memory/src/fused_memory/server/tools.py',
        ):
            assert forbidden not in prompt, forbidden

    def test_an_empty_candidate_list_still_renders(self) -> None:
        """Pure and total: rendering never raises, whatever it is handed."""
        assert isinstance(build_judge_prompt('new', []), str)

    # --- the attach target (gate item 1, option (b)) -------------------------
    #
    # `select_judge_candidates` guarantees the band's winner is in the slate
    # but NOT where it sits: the hoisted-parent rescue APPENDS the evidence
    # child, so the attach target is LAST there and first on a flat slate.
    # Position is therefore not a sound encoding of "the candidate this
    # verdict will be filed against" — the prompt has to name it.
    # `plans/write-triage-attach-target-contradiction.md` §2 carries the
    # measurement; `scripts/check_write_triage_flip_preconditions.sh` item 1
    # is the gate that reads it.

    def test_the_named_candidate_is_the_only_one_marked(self) -> None:
        """A flat slate: the mark lands on the id it was asked for, alone."""
        candidates = [_result(f'm{i}', 0.9 - i / 100) for i in range(3)]
        marked = _marked_ids(candidates, 'm1')
        assert marked == {'m1'}

    def test_a_hoisted_parent_marks_the_child_that_carries_the_evidence(
        self,
    ) -> None:
        """The canonical id can be absent from the slate ENTIRELY.

        `_canonical_id_of` hoists a child winner to its parent id, so
        `decision.canonical_id` names a record retrieval never returned. The
        child carrying `PARENT_ID_KEY` is the one the judge is really looking
        at, and a naive `r.id == canonical_id` marker marks NOTHING here —
        which is the silent version of the defect, not a fix for it.
        """
        child = _result(
            'child-1', 0.60,
            extra_metadata={'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1'},
        )
        candidates = [_result('m0', 0.90), _result('m1', 0.89), child]
        assert 'parent-1' not in [c.id for c in candidates]
        assert _marked_ids(candidates, 'parent-1') == {'child-1'}

    def test_the_target_is_marked_wherever_it_sits_in_the_slate(self) -> None:
        """Built through `select_judge_candidates`, so the rescue produces it.

        The rescue appends (`[*selected[: max(n - 1, 0)], winner]`), so the
        attach target lands LAST. A marker keyed on position — `candidates[0]`
        — marks the wrong record on exactly this slate, and the gate's own
        report says so.
        """
        child = _result(
            'child-1', 0.60,
            extra_metadata={'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1'},
        )
        results = [*[_result(f'm{i}', 0.90 - i / 100) for i in range(6)], child]
        selected = select_judge_candidates(results, 3, canonical_id='parent-1')
        assert [r.id for r in selected] == ['m0', 'm1', 'child-1']
        assert _marked_ids(selected, 'parent-1') == {'child-1'}

    def test_an_unrecognised_target_marks_nothing_and_does_not_perturb(
        self,
    ) -> None:
        """The mark MATCHES against the slate; it does not echo its argument.

        This is the control `scripts/check_write_triage_attach_target.py`
        applies (`_echoes_argument`): an implementation that merely
        interpolates the value satisfies a swap test while binding no verdict
        to any candidate. A matcher recognises neither nonce and renders the
        same prompt for both — and the same prompt as for no target at all.
        """
        candidates = [_result(f'm{i}', 0.9 - i / 100) for i in range(3)]
        unmarked = build_judge_prompt('new', candidates, attach_target_id=None)
        first = build_judge_prompt(
            'new', candidates, attach_target_id='not-on-this-slate-1',
        )
        second = build_judge_prompt(
            'new', candidates, attach_target_id='not-on-this-slate-2',
        )
        assert first == unmarked
        assert second == unmarked
        assert first == second
        assert 'not-on-this-slate-1' not in first
        assert 'not-on-this-slate-2' not in second

    def test_two_targets_on_one_slate_render_differently(self) -> None:
        """The swap test the gate applies: the rendering DEPENDS on the target.

        Necessary and not sufficient on its own — hence the echo control
        above — but a prompt that renders identically for two different attach
        targets has told the model nothing about which candidate the verdict
        will be filed against.
        """
        candidates = [_result(f'm{i}', 0.9 - i / 100) for i in range(3)]
        first = build_judge_prompt('new', candidates, attach_target_id='m0')
        second = build_judge_prompt('new', candidates, attach_target_id='m2')
        assert first != second
        first_only = set(first.splitlines()) - set(second.splitlines())
        second_only = set(second.splitlines()) - set(first.splitlines())
        assert any('m0' in line for line in first_only)
        assert any('m2' in line for line in second_only)

    # --- AT MOST ONE candidate is ever marked -------------------------------
    #
    # The two clauses above ("is this the id?" / "does this carry that
    # `parent_id`?") are both true SOMEWHERE on a slate holding a canonical
    # parent AND one of its children, and that is the ordinary consolidated-
    # topic case, not an exotic one: `_canonical_id_of` hoists a child winner
    # to its parent id and `retrieve_candidates` returns children un-filtered,
    # so parent+child co-occurrence in the top-n is expected. Deciding the
    # question per candidate marks EVERY one of them, and the constant
    # instruction sentence — "The candidate marked `attach_target` is the one
    # this verdict will be filed against" — is then simply false. The target
    # has to be resolved ONCE for the whole slate, with the same ORDERED
    # precedence `select_judge_candidates`' rescue arm uses.

    def test_a_parent_and_its_child_on_one_slate_are_marked_once(self) -> None:
        """The regression: two clauses, both true, must still yield ONE mark."""
        child = _result(
            'child-1', 0.95,
            extra_metadata={'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1'},
        )
        selected = select_judge_candidates(
            [_result('parent-1', 0.90), child], 5, canonical_id='parent-1',
        )
        assert {r.id for r in selected} == {'parent-1', 'child-1'}
        assert _marked_ids(selected, 'parent-1') == {'parent-1'}
        assert _added_lines(selected, 'parent-1') == [
            '  attach_target: parent-1',
        ]

    def test_several_children_do_not_multiply_the_mark(self) -> None:
        """More children of the same parent must not mean more marks.

        A consolidated topic accretes amendments, so three children of one
        parent is the steady state rather than the edge. Per-candidate
        evaluation scales the defect with the topic's age.
        """
        children = [
            _result(
                f'child-{i}', 0.95 - i / 100,
                extra_metadata={
                    'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1',
                },
            )
            for i in range(3)
        ]
        candidates = [*children, _result('parent-1', 0.80)]
        assert _marked_ids(candidates, 'parent-1') == {'parent-1'}
        assert _added_lines(candidates, 'parent-1') == [
            '  attach_target: parent-1',
        ]

    def test_an_exact_id_wins_over_a_child_that_points_at_it(self) -> None:
        """Precedence is EXACT-ID-FIRST, and does not depend on slate order.

        The same ordered `next(...) or next(...)` the rescue arm uses: the
        `PARENT_ID_KEY` clause is the FALLBACK for a hoisted parent that is
        absent from the slate, not a co-equal alternative. A resolver that
        merely took the first candidate satisfying EITHER clause would mark
        the child whenever the child outranks its parent — which is the
        common case, since the child is why the parent was hoisted.
        """
        def _slate(child_first: bool) -> list[MemoryResult]:
            child = _result(
                'child-1', 0.95,
                extra_metadata={
                    'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1',
                },
            )
            parent = _result('parent-1', 0.90)
            return [child, parent] if child_first else [parent, child]

        for child_first in (True, False):
            candidates = _slate(child_first)
            assert _marked_ids(candidates, 'parent-1') == {'parent-1'}, (
                f'child_first={child_first}'
            )
            assert _added_lines(candidates, 'parent-1') == [
                '  attach_target: parent-1',
            ], f'child_first={child_first}'

    def test_the_single_mark_holds_on_the_shapes_that_already_worked(
        self,
    ) -> None:
        """Regression guard: neither existing shape may lose or gain a mark.

        The flat exact-id slate and the HOISTED slate (the parent id absent
        entirely, only the child carrying `PARENT_ID_KEY`) are what the two
        clauses exist for. Resolving one target for the whole slate must leave
        both marking exactly what they marked before — the fallback clause is
        narrowed in precedence, not removed.
        """
        flat = [_result(f'm{i}', 0.9 - i / 100) for i in range(3)]
        assert _marked_ids(flat, 'm1') == {'m1'}
        assert _added_lines(flat, 'm1') == ['  attach_target: m1']

        child = _result(
            'child-1', 0.60,
            extra_metadata={'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1'},
        )
        hoisted = [_result('m0', 0.90), _result('m1', 0.89), child]
        assert 'parent-1' not in [c.id for c in hoisted]
        assert _marked_ids(hoisted, 'parent-1') == {'child-1'}
        assert _added_lines(hoisted, 'parent-1') == [
            '  attach_target: child-1',
        ]

    def test_the_echo_control_still_holds_on_a_parent_and_child_slate(
        self,
    ) -> None:
        """Resolving one target must not turn the marker into an echo.

        Same control as `test_an_unrecognised_target_marks_nothing_and_does_
        not_perturb`, re-applied to the slate the fix is about: an id naming
        no candidate — and no `PARENT_ID_KEY` pointing at it — still renders
        exactly as no target at all, and two such nonces render identically.
        That is what `check_write_triage_attach_target.py::_echoes_argument`
        separates a real marker from a free-text parameter by.
        """
        child = _result(
            'child-1', 0.95,
            extra_metadata={'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1'},
        )
        candidates = [child, _result('parent-1', 0.90)]
        unmarked = build_judge_prompt('new', candidates, attach_target_id=None)
        first = build_judge_prompt(
            'new', candidates, attach_target_id='not-on-this-slate-1',
        )
        second = build_judge_prompt(
            'new', candidates, attach_target_id='not-on-this-slate-2',
        )
        assert first == unmarked
        assert second == unmarked
        assert first == second
        assert 'not-on-this-slate-1' not in first
        assert 'not-on-this-slate-2' not in second


class TestAttachTargetGateProbe:
    """The flip gate's own checker, run against this worktree's source.

    `scripts/check_write_triage_flip_preconditions.sh` item 1 delegates to
    `scripts/check_write_triage_attach_target.py`, which asserts the INVARIANT
    (a verdict binds to a determinate candidate) rather than any mechanism.
    Running the gate's own probe as the oracle is what keeps this suite and
    the gate from ever disagreeing about whether item 1 is closed — nothing
    about the invariant is re-implemented here.
    """

    def test_the_probe_reports_a_clean_pass(self) -> None:
        """Exit 0, and NOT the `PASS-NEEDS-CONFIRMATION` downgrade.

        The downgrade means the marker echoed its argument and was forgiven
        only because the parameter is target-NAMED; it demands an operator
        eyeball before the flip. A marker that matches against the slate earns
        the clean pass instead.
        """
        if not _PROBE_PATH.exists():
            pytest.skip(f'attach-target probe not present at {_PROBE_PATH}')
        completed = subprocess.run(
            [sys.executable, str(_PROBE_PATH), '--src-root', str(_JUDGE_SRC_ROOT)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (
            'PASS  the judge path binds a verdict to a determinate candidate'
            in completed.stdout
        ), completed.stdout
        assert 'PASS-NEEDS-CONFIRMATION' not in completed.stdout, completed.stdout


# ---------------------------------------------------------------------------
# defensive config resolvers
# ---------------------------------------------------------------------------


def _svc(llm: object = None, **write_triage) -> types.SimpleNamespace:
    """A memory_service double whose config leaves are REAL namespaces.

    Same shape as ``test_write_triage.py::_svc``, extended with the ``llm``
    section the judge resolvers INHERIT from. A plain ``Mock()`` is used
    deliberately in the negative cases: an unspecced Mock auto-generates every
    attribute, so ``config.write_triage.judge_enabled`` yields a truthy Mock
    rather than a bool — which is precisely the shape these resolvers refuse.
    """
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            write_triage=types.SimpleNamespace(**write_triage),
            llm=llm,
        ),
    )


_MISSING_HOPS = [
    ('no config', types.SimpleNamespace()),
    ('config is None', types.SimpleNamespace(config=None)),
    (
        'write_triage is None',
        types.SimpleNamespace(config=types.SimpleNamespace(write_triage=None, llm=None)),
    ),
    ('no leaf', _svc()),
    ('unspecced mock', Mock()),
]


class TestResolveJudgeEnabled:
    """The judge's own kill switch — a finer lever than `write_triage.enabled`.

    Defaults TRUE, unlike its sibling, and that asymmetry is deliberate. The
    judge is structurally INERT while `write_triage.enabled` is false (no
    triage code runs at all), so default-True costs nothing on today's shipped
    config. Default-False would be the footgun: at the task-3169 flip the
    operator would turn `enabled` on, silently get stub behaviour, and read
    the resulting all-`stored` ack stream as evidence the corpus is novel.
    """

    @pytest.mark.parametrize('value', [True, False])
    def test_a_configured_bool_is_used(self, value: bool) -> None:
        assert resolve_judge_enabled(_svc(judge_enabled=value)) is value

    @pytest.mark.parametrize(
        ('label', 'service'), _MISSING_HOPS,
        ids=[label for label, _ in _MISSING_HOPS],
    )
    def test_a_missing_hop_falls_back_to_the_default(
        self, label: str, service: object,
    ) -> None:
        assert resolve_judge_enabled(service) is _DEFAULT_JUDGE_ENABLED, label

    @pytest.mark.parametrize('value', [1, 0, 'true', 'false', [], None, object()])
    def test_a_non_bool_falls_back_to_the_default(self, value: object) -> None:
        """`isinstance(bool)` only — a truthy 1 must not enable by accident.

        Nor may a falsy 0 DISABLE by accident: an operator who wrote `0` into
        the wrong leaf would otherwise silently turn the judge off and see the
        exact symptom the default-True choice exists to prevent.
        """
        assert resolve_judge_enabled(_svc(judge_enabled=value)) is _DEFAULT_JUDGE_ENABLED


class TestResolveJudgeProvider:
    """`judge_provider` — pinned, else INHERITED from `llm.provider`."""

    @pytest.mark.parametrize('value', ['openai', 'anthropic'])
    def test_a_known_provider_is_used(self, value: str) -> None:
        assert resolve_judge_provider(_svc(judge_provider=value)) == value

    def test_none_inherits_the_llm_provider(self) -> None:
        """The judge follows the model the deployment already trusts.

        Asserted explicitly against `anthropic`, not the module default, so
        an implementation that merely returned the default would fail here
        rather than passing by coincidence.
        """
        service = _svc(
            judge_provider=None,
            llm=types.SimpleNamespace(provider='anthropic', model=None),
        )
        assert resolve_judge_provider(service) == 'anthropic'

    @pytest.mark.parametrize(
        ('label', 'service'), _MISSING_HOPS,
        ids=[label for label, _ in _MISSING_HOPS],
    )
    def test_a_missing_hop_falls_back_to_the_default(
        self, label: str, service: object,
    ) -> None:
        assert resolve_judge_provider(service) == _DEFAULT_JUDGE_PROVIDER, label

    @pytest.mark.parametrize('value', ['gemini', '', 42, True, []])
    def test_an_unknown_provider_falls_back_rather_than_being_honoured(
        self, value: object,
    ) -> None:
        """A provider string no arm implements would fail every single write.

        Falling back is right here and RAISING is wrong: this resolver runs on
        the write path, where C1 forbids raising. The unresolvable-provider
        raise lives in the fan-out (`judge_write`), which IS inside
        `triage_write`'s fail-open arm.
        """
        service = _svc(
            judge_provider=value,
            llm=types.SimpleNamespace(provider=None, model=None),
        )
        assert resolve_judge_provider(service) == _DEFAULT_JUDGE_PROVIDER

    def test_an_unknown_llm_provider_falls_back_to_the_module_default(self) -> None:
        service = _svc(
            judge_provider=None,
            llm=types.SimpleNamespace(provider='gemini', model=None),
        )
        assert resolve_judge_provider(service) == _DEFAULT_JUDGE_PROVIDER


class TestResolveJudgeModel:
    """`judge_model` — pinned, else INHERITED from `llm.model`."""

    def test_a_configured_model_is_used(self) -> None:
        assert resolve_judge_model(_svc(judge_model='gpt-4.1-nano')) == 'gpt-4.1-nano'

    def test_none_inherits_the_llm_model(self) -> None:
        service = _svc(
            judge_model=None,
            llm=types.SimpleNamespace(provider=None, model='claude-3-5-haiku-latest'),
        )
        assert resolve_judge_model(service) == 'claude-3-5-haiku-latest'

    @pytest.mark.parametrize(
        ('label', 'service'), _MISSING_HOPS,
        ids=[label for label, _ in _MISSING_HOPS],
    )
    def test_a_missing_hop_falls_back_to_the_default(
        self, label: str, service: object,
    ) -> None:
        assert resolve_judge_model(service) == _DEFAULT_JUDGE_MODEL, label

    @pytest.mark.parametrize('value', ['', '   ', 42, True, [], object()])
    def test_a_non_string_or_empty_model_falls_back(self, value: object) -> None:
        """An empty model name reaches the SDK as a 404 on every write."""
        service = _svc(
            judge_model=value,
            llm=types.SimpleNamespace(provider=None, model=None),
        )
        assert resolve_judge_model(service) == _DEFAULT_JUDGE_MODEL

    def test_a_pinned_foreign_provider_does_not_borrow_the_llm_model(self) -> None:
        """The documented cross-provider pin must not post an openai id to anthropic.

        `judge_provider`'s schema description explicitly invites pinning the
        judge away from the rest of the server, and `judge_model` ships as
        `null`. Inheriting `llm.model` unconditionally would send
        `gpt-4o-mini` to `anthropic.messages.create` on THIS deployment
        (llm.provider=openai) — a 404 on every middle-band write, i.e. a total
        judge outage counted as a fail-open on every write and escalated as a
        storm whose stated cause is not what is wrong. That is the same
        outage the blank-name fallback above exists to prevent, arriving by a
        schema-VALID door.
        """
        service = _svc(
            judge_provider='anthropic',
            judge_model=None,
            llm=types.SimpleNamespace(provider='openai', model='gpt-4o-mini'),
        )
        assert resolve_judge_provider(service) == 'anthropic'
        assert resolve_judge_model(service) != 'gpt-4o-mini'
        assert resolve_judge_model(service) == _DEFAULT_MODEL_BY_PROVIDER['anthropic']

    def test_the_mirror_pin_does_not_borrow_either(self) -> None:
        """Symmetric: an anthropic-configured server with the judge on openai."""
        service = _svc(
            judge_provider='openai',
            judge_model=None,
            llm=types.SimpleNamespace(
                provider='anthropic', model='claude-3-5-haiku-latest',
            ),
        )
        assert resolve_judge_model(service) == _DEFAULT_MODEL_BY_PROVIDER['openai']

    def test_every_known_provider_has_a_default_model(self) -> None:
        """A provider with no entry would fall back to the OTHER vendor's id.

        `_DEFAULT_MODEL_BY_PROVIDER.get(provider, _DEFAULT_JUDGE_MODEL)` is a
        safe shape only while every arm `_call_llm` implements is a key here;
        a third arm added without an entry silently reintroduces the 404.
        """
        for provider in _KNOWN_PROVIDERS:
            assert _DEFAULT_MODEL_BY_PROVIDER.get(provider), provider

    def test_an_agreeing_provider_still_inherits(self) -> None:
        """The pin only SKIPS the inheritance when the two vendors differ."""
        service = _svc(
            judge_provider='openai',
            judge_model=None,
            llm=types.SimpleNamespace(provider='openai', model='gpt-4.1-nano'),
        )
        assert resolve_judge_model(service) == 'gpt-4.1-nano'

    def test_an_explicit_pin_wins_over_the_per_provider_default(self) -> None:
        """Skipping the inheritance must not also override an operator's pin."""
        service = _svc(
            judge_provider='anthropic',
            judge_model='claude-sonnet-4-5',
            llm=types.SimpleNamespace(provider='openai', model='gpt-4o-mini'),
        )
        assert resolve_judge_model(service) == 'claude-sonnet-4-5'


class TestResolveJudgeTimeout:
    """The timeout is new to this codebase and it is not optional.

    No LLM call anywhere in fused-memory sets one today, and the openai SDK
    default is 600 seconds. On the SYNCHRONOUS `add_memory` write path that is
    a wedge, not a degradation — the caller waits ten minutes for a write that
    contract C1 promises never to block.
    """

    def test_the_default_is_bounded_well_under_the_sdk_default(self) -> None:
        assert 0 < _DEFAULT_JUDGE_TIMEOUT_SECONDS <= 60

    @pytest.mark.parametrize('value', [1, 2.5, 30, 0.01])
    def test_a_configured_positive_number_is_used(self, value: float) -> None:
        assert resolve_judge_timeout(_svc(judge_timeout_seconds=value)) == value

    @pytest.mark.parametrize(
        ('label', 'service'), _MISSING_HOPS,
        ids=[label for label, _ in _MISSING_HOPS],
    )
    def test_a_missing_hop_falls_back_to_the_default(
        self, label: str, service: object,
    ) -> None:
        assert resolve_judge_timeout(service) == _DEFAULT_JUDGE_TIMEOUT_SECONDS, label

    @pytest.mark.parametrize('value', [0, 0.0, -1, -2.5, True, False, '10', [], None])
    def test_a_non_positive_or_non_numeric_timeout_falls_back(
        self, value: object,
    ) -> None:
        """A zero timeout fails EVERY call and reads as a total judge outage.

        Honouring it would turn a config typo into a permanent storm
        escalation whose stated cause (a broken judge) is not what is wrong.
        `bool` is excluded for the usual reason: `True` would resolve to a
        one-second budget with nothing to explain it.
        """
        service = _svc(judge_timeout_seconds=value)
        assert resolve_judge_timeout(service) == _DEFAULT_JUDGE_TIMEOUT_SECONDS


class TestResolveJudgeCandidateCount:
    """How many candidates reach the prompt — PRD C1's "top 3-5"."""

    def test_the_default_is_the_top_of_the_prd_range(self) -> None:
        assert _DEFAULT_JUDGE_CANDIDATE_COUNT == 5

    def test_a_configured_int_is_used(self) -> None:
        assert resolve_judge_candidate_count(_svc(judge_candidate_count=3)) == 3

    @pytest.mark.parametrize(
        ('label', 'service'), _MISSING_HOPS,
        ids=[label for label, _ in _MISSING_HOPS],
    )
    def test_a_missing_hop_falls_back_to_the_default(
        self, label: str, service: object,
    ) -> None:
        assert (
            resolve_judge_candidate_count(service) == _DEFAULT_JUDGE_CANDIDATE_COUNT
        ), label

    @pytest.mark.parametrize('value', [0, -1, 2.5, '5', True, False, [], None])
    def test_a_non_positive_or_non_int_count_falls_back(self, value: object) -> None:
        """A zero count would empty the slate and answer `stored` every time."""
        service = _svc(judge_candidate_count=value)
        assert (
            resolve_judge_candidate_count(service) == _DEFAULT_JUDGE_CANDIDATE_COUNT
        )


class TestEveryResolverReadsLive:
    """Nothing is captured at import or construction.

    This is the precondition that makes the green-tier RELOADABLE_FIELDS
    registration REAL rather than restart-only in disguise: `apply_reload`
    mutates the shared config object IN PLACE, and a captured value cannot
    observe an in-place mutation. A kill switch sitting in the allowlist while
    silently requiring a restart is worse than no kill switch, because the
    operator believes they turned it off.

    Same property `test_write_triage.py::test_the_flag_is_read_live_not_captured`
    pins for leaf beta.
    """

    @pytest.mark.parametrize(
        ('resolver', 'attr', 'first', 'second'),
        [
            (resolve_judge_enabled, 'judge_enabled', True, False),
            (resolve_judge_provider, 'judge_provider', 'openai', 'anthropic'),
            (resolve_judge_model, 'judge_model', 'model-a', 'model-b'),
            (resolve_judge_timeout, 'judge_timeout_seconds', 5.0, 12.0),
            (resolve_judge_candidate_count, 'judge_candidate_count', 3, 4),
        ],
        ids=['enabled', 'provider', 'model', 'timeout', 'candidate_count'],
    )
    def test_a_mutation_is_observed_on_the_very_next_call(
        self, resolver, attr: str, first: object, second: object,
    ) -> None:
        service = _svc(**{attr: first})
        assert resolver(service) == first
        setattr(service.config.write_triage, attr, second)
        assert resolver(service) == second


# ---------------------------------------------------------------------------
# the async entry point
# ---------------------------------------------------------------------------


def _client_double() -> MagicMock:
    """A client double that is its own async context manager, like the SDKs.

    `AsyncOpenAI.__aenter__` and `AsyncAnthropic.__aenter__` both `return
    self` (read off openai 2.31.0 / anthropic 0.92.0), so `async with
    client as c` binds the SAME object. A bare `MagicMock` instead returns a
    FRESH child from `__aenter__`, which would make every `create` assertion
    in this file inspect a different mock than the one `_call_llm` called —
    passing or failing for reasons that have nothing to do with the code.

    `__aexit__` returns False, because the real ones do: releasing a client
    must not suppress an in-flight exception (see `TestTheClientIsReleased`).
    """
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _openai_client(content: str | None) -> MagicMock:
    """A fake ``AsyncOpenAI`` yielding *content* as the message body.

    Same construction shape as ``test_classifier.py::_make_mock_client`` — the
    established openai double in this repo, plus the async-CM protocol the
    client is used through.
    """
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = _client_double()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


@dataclass
class FakeAnthropicTextBlock:
    """Fake Anthropic TextBlock.

    A hand-rolled dataclass rather than a ``MagicMock``, copying
    ``test_judge.py``: ``MagicMock`` treats ``.name`` specially, which makes
    content blocks the one place a mock silently misbehaves.
    """

    type: str = 'text'
    text: str = ''


def _anthropic_client(blocks: list[FakeAnthropicTextBlock]) -> MagicMock:
    """A fake ``AsyncAnthropic`` whose ``messages.create`` yields *blocks*."""
    response = MagicMock()
    response.content = blocks
    client = _client_double()
    client.messages.create = AsyncMock(return_value=response)
    return client


def _judge_svc(provider: str = 'openai', **write_triage) -> types.SimpleNamespace:
    """A service double configured for the judge, with an llm section."""
    write_triage.setdefault('judge_provider', provider)
    write_triage.setdefault('judge_model', 'test-model')
    return _svc(
        llm=types.SimpleNamespace(
            provider=provider, model='inherited-model', providers=None,
        ),
        **write_triage,
    )


def _creds_svc(**providers: object) -> types.SimpleNamespace:
    """A service double carrying an `llm.providers.<name>` section per kwarg."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            write_triage=types.SimpleNamespace(),
            llm=types.SimpleNamespace(
                provider='openai',
                model='m',
                providers=types.SimpleNamespace(**providers),
            ),
        ),
    )


class TestProviderCredentials:
    """The config -> SDK-kwargs hop, which had no coverage at all.

    Replacing the whole body of `_provider_credentials` with `return {}` left
    the suite green, so neither the `api_url` -> `base_url` RENAME nor the
    `api_key` passthrough was pinned by anything. The rename is the part that
    matters: both SDKs take `base_url`, so forwarding `api_url` verbatim is a
    `TypeError` inside `_call_llm` — landing in `triage_write`'s fail-open arm
    as a counted failure on every middle-band write, for a deployment that
    pins a local OpenAI-compatible endpoint.
    """

    def test_a_configured_section_maps_onto_the_sdk_kwarg_names(self) -> None:
        """`api_url` becomes `base_url`; `api_key` keeps its name."""
        service = _creds_svc(
            openai=types.SimpleNamespace(
                api_key='sk-pinned', api_url='http://localhost:8000/v1',
            ),
        )
        assert _provider_credentials(service, 'openai') == {
            'api_key': 'sk-pinned',
            'base_url': 'http://localhost:8000/v1',
        }

    def test_either_leaf_alone_is_forwarded(self) -> None:
        """Pinning a key without an endpoint (and vice versa) is a real config."""
        key_only = _creds_svc(openai=types.SimpleNamespace(api_key='sk-only'))
        assert _provider_credentials(key_only, 'openai') == {'api_key': 'sk-only'}
        url_only = _creds_svc(anthropic=types.SimpleNamespace(api_url='http://h/v1'))
        assert _provider_credentials(url_only, 'anthropic') == {'base_url': 'http://h/v1'}

    @pytest.mark.parametrize(
        ('label', 'service'),
        [
            ('no providers section', _svc(llm=types.SimpleNamespace(
                provider='openai', model='m', providers=None,
            ))),
            ('no entry for this provider', _creds_svc(
                anthropic=types.SimpleNamespace(api_key='sk-other'),
            )),
            ('entry with neither leaf', _creds_svc(
                openai=types.SimpleNamespace(),
            )),
            ('no llm section', _svc()),
            ('unspecced mock', Mock()),
        ],
        ids=['no-providers', 'other-provider-only', 'empty-entry',
             'no-llm', 'unspecced-mock'],
    )
    def test_an_unconfigured_provider_yields_an_empty_dict(
        self, label: str, service: object,
    ) -> None:
        """Empty is a FIRST-CLASS result: both SDKs fall back to the env.

        That is how this deployment is actually configured (`OPENAI_API_KEY`
        in the shell, nothing in config.yaml), so raising here would break the
        shipped path rather than a misconfigured one. `other-provider-only`
        also pins that a sibling provider's key is not handed to the arm that
        was actually selected.
        """
        assert _provider_credentials(service, 'openai') == {}, label

    @pytest.mark.parametrize('value', ['', None, 0, b'sk-bytes', object()])
    def test_a_blank_or_non_string_leaf_is_not_forwarded(self, value: object) -> None:
        """An empty string must not be sent as a credential.

        `AsyncOpenAI(api_key='')` does NOT fall back to the environment — it
        authenticates with an empty key and 401s, which reads as an outage
        rather than as the empty-leaf config that caused it. Dropping it
        restores the env fallback, which is the behaviour an operator who
        cleared the leaf is asking for.
        """
        service = _creds_svc(
            openai=types.SimpleNamespace(api_key=value, api_url=value),
        )
        assert _provider_credentials(service, 'openai') == {}

    @pytest.mark.asyncio
    async def test_the_credentials_reach_the_sdk_constructor(self) -> None:
        """The resolved kwargs are what the client is actually built with."""
        service = _creds_svc(
            openai=types.SimpleNamespace(api_key='sk-wire', api_url='http://h/v1'),
        )
        service.config.write_triage = types.SimpleNamespace(
            judge_provider='openai', judge_model='test-model',
        )
        client = _openai_client(_payload('distinct'))
        with patch('openai.AsyncOpenAI', return_value=client) as ctor:
            await judge_write(
                memory_service=service,
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )
        assert ctor.call_args.kwargs == {
            'api_key': 'sk-wire', 'base_url': 'http://h/v1',
        }


class TestJudgeWriteDecisionsThatAreNotFailures:
    """Two paths answer `stored` WITHOUT raising, and they must not be counted.

    Everything else in this module raises so `triage_write` can count exactly
    one fail-open. These two are DECISIONS, not outages, and the boundary is
    the same one `_stub_judge`'s docstring draws for leaf beta: counting them
    would guarantee a storm escalation describing a failure that is not
    happening, which trains an operator to ignore the alarm.
    """

    @pytest.mark.asyncio
    async def test_the_disabled_branch_says_so_in_the_log(self, caplog) -> None:
        """An unlogged kill switch is indistinguishable from a novel corpus.

        This branch returns `stored` with no log line, no counter and nothing
        on the ack to tell it apart — reproducing exactly the state
        `_DEFAULT_JUDGE_ENABLED = True` is justified against in its own
        comment: "the operator would flip `enabled`, get stub behaviour, and
        read the all-`stored` ack stream as evidence the corpus is novel".
        Defaulting the knob to True does not help the operator who sets it to
        False and then reads the logs.

        INFO, not a counter: the reviewer is right that counting this would be
        wrong. It is a decision, not a failure, and routing it through the
        fail-open counter would fire a storm escalation describing an outage
        that is not happening.
        """
        with caplog.at_level(logging.INFO):
            verdict = await judge_write(
                memory_service=_judge_svc(judge_enabled=False),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )
        assert verdict == OUTCOME_STORED
        disabled = [
            r for r in caplog.records
            if r.levelno == logging.INFO and 'judge_enabled' in r.getMessage()
        ]
        assert disabled, [r.getMessage() for r in caplog.records]
        message = disabled[0].getMessage()
        assert OUTCOME_STORED in message, message

    @pytest.mark.asyncio
    async def test_the_enabled_path_emits_no_such_record(self, caplog) -> None:
        """One line per write is affordable only while the switch is ENGAGED."""
        client = _openai_client(_payload('restates'))
        with caplog.at_level(logging.INFO), \
                patch('openai.AsyncOpenAI', return_value=client):
            await judge_write(
                memory_service=_judge_svc(),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )
        assert not [
            r for r in caplog.records if 'judge_enabled' in r.getMessage()
        ], [r.getMessage() for r in caplog.records]

    @pytest.mark.asyncio
    async def test_the_empty_slate_branch_stays_quiet(self, caplog) -> None:
        """Deliberately NOT logged: it is per-write and would be noise.

        The kill switch is an operator ACTION and is worth a line per write
        while it is engaged; "this write matched nothing comparable" is the
        ordinary case and would drown it.
        """
        with caplog.at_level(logging.INFO):
            verdict = await judge_write(
                memory_service=_judge_svc(),
                content='c', project_id='p',
                decision=_decision(None), candidates=[],
            )
        assert verdict == OUTCOME_STORED
        assert not caplog.records, [r.getMessage() for r in caplog.records]

    @pytest.mark.asyncio
    async def test_a_disabled_judge_answers_stored_and_makes_no_call(self) -> None:
        """The kill switch must not cost a round-trip, and must not raise."""
        client = _openai_client(_payload('restates'))
        with patch('openai.AsyncOpenAI', return_value=client):
            verdict = await judge_write(
                memory_service=_judge_svc(judge_enabled=False),
                content='c',
                project_id='p',
                decision=_decision('m1'),
                candidates=[_result('m1', 0.80)],
            )
        assert verdict == OUTCOME_STORED
        client.chat.completions.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_empty_candidate_set_answers_stored_and_makes_no_call(self) -> None:
        """Nothing to compare against — a call that cannot produce an attach is waste."""
        client = _openai_client(_payload('restates'))
        with patch('openai.AsyncOpenAI', return_value=client):
            verdict = await judge_write(
                memory_service=_judge_svc(),
                content='c',
                project_id='p',
                decision=_decision(None),
                candidates=[],
            )
        assert verdict == OUTCOME_STORED
        client.chat.completions.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_wholly_uncomparable_slate_answers_stored(self) -> None:
        """A slate of topic pins selects to empty, and takes the same path."""
        client = _openai_client(_payload('restates'))
        with patch('openai.AsyncOpenAI', return_value=client):
            verdict = await judge_write(
                memory_service=_judge_svc(),
                content='c',
                project_id='p',
                decision=_decision('pin0'),
                candidates=[_result('pin0', None, omit_store_score=True)],
            )
        assert verdict == OUTCOME_STORED
        client.chat.completions.create.assert_not_awaited()


class TestJudgeWriteNamesTheAttachTarget:
    """The band names the target; the prompt has to say which candidate it is.

    `select_judge_candidates` guarantees the winner is IN the slate but not
    WHERE — the hoisted-parent rescue appends it — so a judge shown an
    unmarked slate is answering about a set, while the attach touches exactly
    one record in it. Gate item 1
    (`scripts/check_write_triage_flip_preconditions.sh`) is that gap.
    """

    @staticmethod
    def _sent_prompt(client: MagicMock) -> str:
        """The user turn that actually reached the provider."""
        messages = client.chat.completions.create.await_args.kwargs['messages']
        return next(m['content'] for m in messages if m['role'] == 'user')

    @pytest.mark.asyncio
    async def test_the_bands_hoisted_winner_is_marked_in_the_sent_prompt(
        self,
    ) -> None:
        """`decision.canonical_id` names a parent absent from the slate.

        Read off the prompt the fake provider was actually handed, not off a
        patched renderer — what matters is what the model sees.
        """
        child = _result(
            'child-1', 0.60,
            extra_metadata={'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1'},
        )
        client = _openai_client(_payload('restates'))
        with patch('openai.AsyncOpenAI', return_value=client):
            await judge_write(
                memory_service=_judge_svc(),
                content='c',
                project_id='p',
                decision=_decision('parent-1'),
                candidates=[_result('m0', 0.90), _result('m1', 0.89), child],
            )
        prompt = self._sent_prompt(client)
        marked = [line for line in prompt.splitlines() if 'attach_target:' in line]
        assert marked == ['  attach_target: child-1'], prompt

    @pytest.mark.asyncio
    async def test_a_decision_naming_nothing_marks_nothing(self) -> None:
        """A band decision with no canonical id must not mark an arbitrary row.

        Marking `candidates[0]` "because something has to be the target" is
        the exact defect: it would tell the model a record is the attach
        target when nothing said so.
        """
        client = _openai_client(_payload('restates'))
        with patch('openai.AsyncOpenAI', return_value=client):
            await judge_write(
                memory_service=_judge_svc(),
                content='c',
                project_id='p',
                decision=_decision(None),
                candidates=[_result('m0', 0.90), _result('m1', 0.89)],
            )
        prompt = self._sent_prompt(client)
        assert [line for line in prompt.splitlines() if 'attach_target:' in line] == []

    @pytest.mark.asyncio
    async def test_the_selector_and_the_renderer_are_given_the_same_id(self) -> None:
        """ONE expression for "the band's winner" on this path.

        The selector guarantees the winner is present and the renderer marks
        it; feeding them different ids would let the prompt mark a record the
        selector never promised to keep, and neither call site would look
        wrong on its own.
        """
        client = _openai_client(_payload('restates'))
        with (
            patch('openai.AsyncOpenAI', return_value=client),
            patch.object(
                judge_module, 'select_judge_candidates',
                wraps=judge_module.select_judge_candidates,
            ) as selector,
            patch.object(
                judge_module, 'build_judge_prompt',
                wraps=judge_module.build_judge_prompt,
            ) as renderer,
        ):
            await judge_write(
                memory_service=_judge_svc(),
                content='c',
                project_id='p',
                decision=_decision('m1'),
                candidates=[_result('m0', 0.90), _result('m1', 0.89)],
            )
        assert selector.call_args.kwargs['canonical_id'] == 'm1'
        assert renderer.call_args.kwargs['attach_target_id'] == 'm1'
        assert (
            renderer.call_args.kwargs['attach_target_id']
            == selector.call_args.kwargs['canonical_id']
        )


class TestJudgeWriteOpenAIArm:
    """The shipped arm: `llm.provider` is openai on this deployment."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('word', 'outcome'),
        [
            ('distinct', OUTCOME_STORED),
            ('restates', OUTCOME_RESTATED),
            ('amends', OUTCOME_AMENDED),
            ('contests', OUTCOME_CONTESTED),
        ],
        ids=['distinct', 'restates', 'amends', 'contests'],
    )
    async def test_each_verdict_round_trips(self, word: str, outcome: str) -> None:
        client = _openai_client(_payload(word))
        with patch('openai.AsyncOpenAI', return_value=client):
            verdict = await judge_write(
                memory_service=_judge_svc(),
                content='c',
                project_id='p',
                decision=_decision('m1'),
                candidates=[_result('m1', 0.80)],
            )
        assert verdict == outcome

    @pytest.mark.asyncio
    async def test_the_call_is_made_once_with_the_resolved_shape(self) -> None:
        """Deterministic, bounded, and JSON-forced — one call, no retry loop."""
        client = _openai_client(_payload('amends'))
        with patch('openai.AsyncOpenAI', return_value=client):
            await judge_write(
                memory_service=_judge_svc(judge_model='pinned-model'),
                content='c',
                project_id='p',
                decision=_decision('m1'),
                candidates=[_result('m1', 0.80)],
            )
        client.chat.completions.create.assert_awaited_once()
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs['model'] == 'pinned-model'
        assert kwargs['temperature'] == 0.0
        assert kwargs['max_tokens'] == _JUDGE_MAX_TOKENS
        assert kwargs['response_format'] == {'type': 'json_object'}
        assert kwargs['messages'][0] == {
            'role': 'system', 'content': JUDGE_SYSTEM_PROMPT,
        }
        assert kwargs['messages'][1]['role'] == 'user'

    @pytest.mark.asyncio
    async def test_at_most_the_configured_candidate_count_reaches_the_prompt(self) -> None:
        """The width knob is only worth configuring if it reaches the wire."""
        client = _openai_client(_payload('distinct'))
        candidates = [_result(f'm{i}', 0.90 - i / 100) for i in range(9)]
        with patch('openai.AsyncOpenAI', return_value=client):
            await judge_write(
                memory_service=_judge_svc(judge_candidate_count=3),
                content='c',
                project_id='p',
                decision=_decision('m0'),
                candidates=candidates,
            )
        rendered = client.chat.completions.create.call_args.kwargs['messages'][1]['content']
        assert sum(1 for c in candidates if f'id: {c.id}\n' in rendered) == 3


class TestJudgeWriteAnthropicArm:
    """Implemented and selectable by config, for a deployment that has the key.

    Unused here — ANTHROPIC_API_KEY is unset on this deployment (CLAUDE.md:
    agents use OAuth) — so it is covered by construction rather than by the
    eval, and "haiku-class" in the PRD is a cost/size class, not a vendor pin.
    """

    @pytest.mark.asyncio
    async def test_a_verdict_round_trips_through_the_anthropic_arm(self) -> None:
        client = _anthropic_client([FakeAnthropicTextBlock(text=_payload('contests'))])
        with patch('anthropic.AsyncAnthropic', return_value=client):
            verdict = await judge_write(
                memory_service=_judge_svc('anthropic'),
                content='c',
                project_id='p',
                decision=_decision('m1'),
                candidates=[_result('m1', 0.80)],
            )
        assert verdict == OUTCOME_CONTESTED

    @pytest.mark.asyncio
    async def test_the_system_prompt_goes_via_the_system_parameter(self) -> None:
        """Anthropic has no system ROLE — a system message would be a user turn."""
        client = _anthropic_client([FakeAnthropicTextBlock(text=_payload('amends'))])
        with patch('anthropic.AsyncAnthropic', return_value=client):
            await judge_write(
                memory_service=_judge_svc('anthropic', judge_model='pinned-model'),
                content='c',
                project_id='p',
                decision=_decision('m1'),
                candidates=[_result('m1', 0.80)],
            )
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs['system'] == JUDGE_SYSTEM_PROMPT
        assert kwargs['model'] == 'pinned-model'
        assert kwargs['max_tokens'] == _JUDGE_MAX_TOKENS
        assert [m['role'] for m in kwargs['messages']] == ['user']

    @pytest.mark.asyncio
    async def test_this_arm_is_pinned_deterministic_like_the_other(self) -> None:
        """`temperature=0.0` on BOTH arms — the openai one already had it.

        Omitting it here does not mean "unset": Anthropic's default is 1.0,
        so this arm was sampling. `_call_llm`'s own docstring scopes only
        `response_format` to one provider, so the asymmetry contradicts the
        module's stated contract as well as the openai arm.

        It is not cosmetic. This is a classifier answering ONE word from a
        closed vocabulary under a 64-token cap with no JSON mode on this arm,
        so sampling buys nothing and raises the odds of a preamble or a
        truncated payload — each of which is a `JudgeOutputError`, and
        therefore a COUNTED fail-open on the write path rather than a bad
        answer.
        """
        client = _anthropic_client([FakeAnthropicTextBlock(text=_payload('amends'))])
        with patch('anthropic.AsyncAnthropic', return_value=client):
            await judge_write(
                memory_service=_judge_svc('anthropic'),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )
        assert client.messages.create.call_args.kwargs['temperature'] == 0.0

    @pytest.mark.asyncio
    async def test_a_non_text_first_block_does_not_crash_the_read(self) -> None:
        """A leading non-text block must not be read as the answer."""
        client = _anthropic_client([
            FakeAnthropicTextBlock(type='thinking', text=''),
            FakeAnthropicTextBlock(text=_payload('restates')),
        ])
        with patch('anthropic.AsyncAnthropic', return_value=client):
            verdict = await judge_write(
                memory_service=_judge_svc('anthropic'),
                content='c',
                project_id='p',
                decision=_decision('m1'),
                candidates=[_result('m1', 0.80)],
            )
        assert verdict == OUTCOME_RESTATED


class TestTheClientIsReleased:
    """The SDK client is built per middle-band write and must not be leaked.

    `_call_llm` constructs `AsyncOpenAI`/`AsyncAnthropic` on every triaged
    write. Neither defines `__del__` (verified against openai 2.31.0), so an
    unclosed client abandons an `httpx` connection pool — sockets and TLS
    state — to the garbage collector, on a path that runs once per write in a
    single long-lived server process.

    The per-call CONSTRUCTION is deliberate and stays: a cached client keyed
    to a config that hot-reloads would pin a stale `model`/`api_url` past a
    reload the operator was told had applied, silently turning a green-tier
    knob into a restart-only one. That rationale is fully preserved by
    closing the client at the end of the call, which is what these tests pin.
    """

    @pytest.mark.asyncio
    async def test_the_openai_client_is_released_on_the_happy_path(self) -> None:
        client = (_openai_client(_payload('restates')))
        with patch('openai.AsyncOpenAI', return_value=client):
            await judge_write(
                memory_service=_judge_svc(),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )
        client.__aexit__.assert_awaited()

    @pytest.mark.asyncio
    async def test_the_anthropic_client_is_released_on_the_happy_path(self) -> None:
        client = _anthropic_client(
            [FakeAnthropicTextBlock(text=_payload('amends'))],
        )
        with patch('anthropic.AsyncAnthropic', return_value=client):
            await judge_write(
                memory_service=_judge_svc('anthropic'),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )
        client.__aexit__.assert_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('provider', 'ctor', 'attr'),
        [
            ('openai', 'openai.AsyncOpenAI', 'chat'),
            ('anthropic', 'anthropic.AsyncAnthropic', 'messages'),
        ],
        ids=['openai', 'anthropic'],
    )
    async def test_the_client_is_released_on_the_timeout_path(
        self, provider: str, ctor: str, attr: str,
    ) -> None:
        """The WORST case, and the one an `await ...; close()` shape misses.

        `asyncio.wait_for` CANCELS the in-flight request, so a close written
        after the awaited call never runs — the pool is abandoned precisely
        when the provider is slow, i.e. exactly when writes are piling up.
        """
        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(5)

        client = _client_double()
        if provider == 'openai':
            client.chat.completions.create = AsyncMock(side_effect=_hang)
        else:
            client.messages.create = AsyncMock(side_effect=_hang)
        with patch(ctor, return_value=client), pytest.raises(TimeoutError):
            await judge_write(
                memory_service=_judge_svc(provider, judge_timeout_seconds=0.01),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )
        client.__aexit__.assert_awaited()

    @pytest.mark.asyncio
    async def test_the_client_is_released_when_the_transport_raises(self) -> None:
        """A release must not depend on the call having succeeded."""
        client = _client_double()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError('boom'))
        with patch('openai.AsyncOpenAI', return_value=client), \
                pytest.raises(RuntimeError):
            await judge_write(
                memory_service=_judge_svc(),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )
        client.__aexit__.assert_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('module_name', 'attr'),
        [('openai', 'AsyncOpenAI'), ('anthropic', 'AsyncAnthropic')],
        ids=['openai', 'anthropic'],
    )
    async def test_the_real_clients_exit_does_not_swallow_an_exception(
        self, module_name: str, attr: str,
    ) -> None:
        """A NEW dependency the `async with` introduces, pinned against the SDKs.

        `async with` delegates the suppression decision to the object: an
        `__aexit__` returning True swallows the in-flight exception. That
        would hand `triage_write` a silent success and destroy `_call_llm`'s
        "NO try/except ANYWHERE" property — no `exc_info` log, no counted
        fail-open, and a broken judge reading exactly like a healthy one
        answering "nothing matched" (INV-4).

        The doubles above cannot pin this, because a double asserts only what
        it was told to return. So this asks the REAL client classes, which is
        where the contract actually lives, and it is what would fail if an SDK
        upgrade ever started suppressing. No network: `__aexit__` closes the
        transport and returns.
        """
        module = pytest.importorskip(module_name)
        client = getattr(module, attr)(api_key='sk-not-used')
        suppressed = await client.__aexit__(
            RuntimeError, RuntimeError('boom'), None,
        )
        assert not suppressed, (
            f'{attr}.__aexit__ returned {suppressed!r}; a truthy value would '
            f'make `async with` swallow a judge failure into a silent `stored`'
        )


class TestJudgeWriteFailuresRaise:
    """Every failure PROPAGATES. `triage_write` owns the fail-open counting.

    A judge that caught its own errors and returned `stored` would produce
    the exact silent degradation INV-4 exists to prevent: every write during
    an outage would look identical to "nothing matched", the counter would
    never increment, and no storm escalation would ever fire.
    """

    @pytest.mark.asyncio
    async def test_a_transport_error_propagates(self) -> None:
        client = _client_double()
        client.chat.completions.create = AsyncMock(side_effect=RuntimeError('boom'))
        with patch('openai.AsyncOpenAI', return_value=client), \
                pytest.raises(RuntimeError):
            await judge_write(
                memory_service=_judge_svc(),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )

    @pytest.mark.asyncio
    async def test_a_hang_past_the_configured_timeout_raises(self) -> None:
        """The timeout is NEW here and non-negotiable: the SDK default is 600s.

        On the synchronous `add_memory` write path that is a wedge, not a
        degradation. A TimeoutError propagates into `triage_write`'s fail-open
        arm, which is exactly C1's "judge error/timeout => stored + storm
        counter".
        """
        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(5)

        client = _client_double()
        client.chat.completions.create = AsyncMock(side_effect=_hang)
        with patch('openai.AsyncOpenAI', return_value=client), \
                pytest.raises(TimeoutError):
            await judge_write(
                memory_service=_judge_svc(judge_timeout_seconds=0.01),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )

    @pytest.mark.asyncio
    async def test_a_hang_past_the_timeout_raises_on_the_anthropic_arm_too(
        self,
    ) -> None:
        """The timeout is per-ARM, and only the openai arm was pinned.

        `_call_llm` wraps each arm in its own `asyncio.wait_for`, so a bound
        that was dropped from one of them would leave that deployment on the
        SDK's 600s default while this suite stayed green — the wedge described
        in the openai case above, reachable by flipping one config key.
        """
        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(5)

        client = _client_double()
        client.messages.create = AsyncMock(side_effect=_hang)
        with patch('anthropic.AsyncAnthropic', return_value=client), \
                pytest.raises(TimeoutError):
            await judge_write(
                memory_service=_judge_svc('anthropic', judge_timeout_seconds=0.01),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('body', 'label'),
        [
            ('I cannot answer that.', 'prose'),
            (_payload('supersedes'), 'out-of-vocabulary'),
            ('', 'empty-body'),
            (None, 'null-body'),
        ],
        ids=['prose', 'out-of-vocabulary', 'empty-body', 'null-body'],
    )
    async def test_an_unusable_response_raises_judge_output_error(
        self, body: str | None, label: str,
    ) -> None:
        client = _openai_client(body)
        with patch('openai.AsyncOpenAI', return_value=client), \
                pytest.raises(JudgeOutputError):
            await judge_write(
                memory_service=_judge_svc(),
                content='c', project_id='p',
                decision=_decision('m1'), candidates=[_result('m1', 0.80)],
            )

    @pytest.mark.asyncio
    async def test_an_unresolvable_provider_raises_rather_than_guessing(self) -> None:
        """Silently picking an arm would bill an account the operator did not choose.

        Unreachable through `resolve_judge_provider`, which falls back — this
        guards the fan-out itself, so a future caller passing a provider
        directly cannot land on a silent default.
        """
        with pytest.raises(ValueError, match='provider'):
            await _call_llm(
                provider='gemini', model='m', prompt='p',
                memory_service=_judge_svc(), timeout=1.0,
            )


class TestJudgeWriteInheritsBetasFailOpenApparatus:
    """The INTEGRATION property, asserted directly rather than argued.

    Gamma adds no second fail-open apparatus; it inherits beta's. So the thing
    worth pinning is not "the judge raises" but what `triage_write` DOES with
    a raising judge: exactly one counted fail-open, and an ack of `stored`.
    """

    @staticmethod
    def _mid_band_service(**write_triage) -> types.SimpleNamespace:
        """A service whose retrieval lands squarely in the middle band."""
        service = _judge_svc(t_high=0.95, t_low=0.50, **write_triage)
        service.search = AsyncMock(
            return_value=SearchResults([_result('m1', 0.80)]),
        )
        return service

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('body', 'side_effect', 'label'),
        [
            (None, RuntimeError('transport down'), 'transport-error'),
            ('I cannot answer that.', None, 'unparseable'),
            ('', None, 'empty-body'),
        ],
        ids=['transport-error', 'unparseable', 'empty-body'],
    )
    async def test_a_broken_judge_stores_and_counts_exactly_once(
        self, body: str | None, side_effect: Exception | None, label: str,
    ) -> None:
        counter = TriageFailOpenCounter()
        service = self._mid_band_service()
        client = _openai_client(body)
        if side_effect is not None:
            client.chat.completions.create = AsyncMock(side_effect=side_effect)

        with patch('openai.AsyncOpenAI', return_value=client):
            decision = await triage_write(
                service, content='c', project_id='p',
                counter=counter, judge=judge_write,
            )

        assert decision.outcome == OUTCOME_STORED, label
        assert decision.canonical_id is None, label
        assert counter.live_count() == 1, label

    @pytest.mark.asyncio
    async def test_a_timeout_stores_and_counts_exactly_once(self) -> None:
        async def _hang(*_args, **_kwargs):
            await asyncio.sleep(5)

        counter = TriageFailOpenCounter()
        service = self._mid_band_service(judge_timeout_seconds=0.01)
        client = _client_double()
        client.chat.completions.create = AsyncMock(side_effect=_hang)

        with patch('openai.AsyncOpenAI', return_value=client):
            decision = await triage_write(
                service, content='c', project_id='p',
                counter=counter, judge=judge_write,
            )

        assert decision.outcome == OUTCOME_STORED
        assert counter.live_count() == 1

    @pytest.mark.asyncio
    async def test_a_deliberately_disabled_judge_counts_zero(self) -> None:
        """The boundary that keeps the alarm meaningful.

        A disabled judge is not an outage. Counting it would make the first
        ten middle-band writes after the 3169 flip fire a storm escalation
        describing a failure that is not happening.
        """
        counter = TriageFailOpenCounter()
        service = self._mid_band_service(judge_enabled=False)

        decision = await triage_write(
            service, content='c', project_id='p',
            counter=counter, judge=judge_write,
        )

        assert decision.outcome == OUTCOME_STORED
        assert counter.live_count() == 0

    @pytest.mark.asyncio
    async def test_a_healthy_judge_routes_and_counts_zero(self) -> None:
        """The happy path end-to-end: a verdict becomes an ack, nothing counted."""
        counter = TriageFailOpenCounter()
        service = self._mid_band_service()
        client = _openai_client(_payload('amends'))

        with patch('openai.AsyncOpenAI', return_value=client):
            decision = await triage_write(
                service, content='c', project_id='p',
                counter=counter, judge=judge_write,
            )

        assert decision.outcome == OUTCOME_AMENDED
        assert decision.canonical_id == 'm1'
        assert counter.live_count() == 0
