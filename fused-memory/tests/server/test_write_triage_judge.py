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
import types
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from fused_memory.models.enums import MemoryCategory, SourceStore
from fused_memory.models.memory import MemoryResult
from fused_memory.server import write_triage
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
    _KNOWN_PROVIDERS,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_VERDICTS,
    VERDICT_KEY,
    JudgeOutputError,
    _call_llm,
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

    def test_the_default_is_on(self) -> None:
        assert _DEFAULT_JUDGE_ENABLED is True

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


def _openai_client(content: str | None) -> MagicMock:
    """A fake ``AsyncOpenAI`` yielding *content* as the message body.

    Same construction shape as ``test_classifier.py::_make_mock_client`` — the
    established openai double in this repo.
    """
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
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
    client = MagicMock()
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


class TestJudgeWriteDecisionsThatAreNotFailures:
    """Two paths answer `stored` WITHOUT raising, and they must not be counted.

    Everything else in this module raises so `triage_write` can count exactly
    one fail-open. These two are DECISIONS, not outages, and the boundary is
    the same one `_stub_judge`'s docstring draws for leaf beta: counting them
    would guarantee a storm escalation describing a failure that is not
    happening, which trains an operator to ignore the alarm.
    """

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
        assert 0 < kwargs['max_tokens'] <= 256
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
        assert 0 < kwargs['max_tokens'] <= 256
        assert [m['role'] for m in kwargs['messages']] == ['user']

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


class TestJudgeWriteFailuresRaise:
    """Every failure PROPAGATES. `triage_write` owns the fail-open counting.

    A judge that caught its own errors and returned `stored` would produce
    the exact silent degradation INV-4 exists to prevent: every write during
    an outage would look identical to "nothing matched", the counter would
    never increment, and no storm escalation would ever fire.
    """

    @pytest.mark.asyncio
    async def test_a_transport_error_propagates(self) -> None:
        client = MagicMock()
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

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_hang)
        with patch('openai.AsyncOpenAI', return_value=client), \
                pytest.raises(TimeoutError):
            await judge_write(
                memory_service=_judge_svc(judge_timeout_seconds=0.01),
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
        client = MagicMock()
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
