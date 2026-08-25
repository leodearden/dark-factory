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

import inspect
import json

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import write_triage, write_triage_judge
from fused_memory.server.grouped_read import AMENDMENT_KIND, PARENT_ID_KEY
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
    BandDecision,
)
from fused_memory.server.write_triage_judge import (
    _ELIDED_MARKER,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_VERDICTS,
    VERDICT_KEY,
    JudgeOutputError,
    build_judge_prompt,
    parse_judge_verdict,
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

    def test_the_module_spells_no_outcome_string_by_hand(self) -> None:
        """INV-5: the outcome constants are IMPORTED, never re-spelled.

        ``is``-identity is not a usable check here — CPython interns short
        string literals, so a hand-written ``'restated'`` would pass an
        identity assertion against the imported constant and the drift would
        be invisible. A source-level check is the only one that actually
        catches the copy.
        """
        source = inspect.getsource(write_triage_judge)
        # Strip the import block: the names appear there legitimately.
        body = source.split('JUDGE_VERDICTS', 1)[-1]
        for outcome in sorted(TRIAGE_OUTCOMES):
            assert f"'{outcome}'" not in body, outcome
            assert f'"{outcome}"' not in body, outcome

    def test_the_outcome_constants_come_from_write_triage(self) -> None:
        """The values are the ones beta publishes, whatever they are spelled."""
        assert JUDGE_VERDICTS['restates'] == write_triage.OUTCOME_RESTATED
        assert JUDGE_VERDICTS['amends'] == write_triage.OUTCOME_AMENDED
        assert JUDGE_VERDICTS['contests'] == write_triage.OUTCOME_CONTESTED
        assert JUDGE_VERDICTS['distinct'] == write_triage.OUTCOME_STORED


class TestParseJudgeVerdict:
    """Parsing is where the closed output is ENFORCED (D3).

    A schema-forcing transport can still return a refusal, so the parse
    boundary is the only enforcement that always runs. Every rejection path
    raises ``JudgeOutputError``; none of them default to ``stored``, because
    ``triage_write`` counts a fail-open only for something that raises or
    returns out-of-vocabulary, and a silent default would make a broken judge
    read exactly like a healthy one answering "nothing matched".
    """

    def test_judge_output_error_is_an_exception_subclass(self) -> None:
        """Module-local, so a caller can distinguish it from a transport error."""
        assert issubclass(JudgeOutputError, Exception)

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
        """
        child = _result(
            'child-1', 0.97,
            extra_metadata={'kind': AMENDMENT_KIND, PARENT_ID_KEY: 'parent-1'},
        )
        results = [child, *[_result(f'm{i}', 0.90 - i / 100) for i in range(6)]]
        selected = select_judge_candidates(results, 3, canonical_id='parent-1')
        assert 'child-1' in [r.id for r in selected]

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

    def test_the_system_prompt_states_detect_not_adjudicate(self) -> None:
        """D3 lives in the model's own instructions, not only in a docstring.

        A judge told merely to "classify" would happily decide which of two
        contradictory memories is TRUE. Reify esc-5557/esc-5626 showed that
        adjudication needs code-reading the synchronous write path cannot do,
        so the instruction has to say the judge is detecting a contradiction
        and routing it, not resolving it.
        """
        lowered = JUDGE_SYSTEM_PROMPT.lower()
        assert 'relationship' in lowered
        assert 'not' in lowered
        for word in JUDGE_VERDICTS:
            assert word in lowered
