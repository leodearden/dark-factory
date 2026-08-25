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

import pytest

from fused_memory.server import write_triage, write_triage_judge
from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
)
from fused_memory.server.write_triage_judge import (
    JUDGE_VERDICTS,
    VERDICT_KEY,
    JudgeOutputError,
    parse_judge_verdict,
)

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
        assert parse_judge_verdict('{"%s": "%s"}' % (VERDICT_KEY, word)) == outcome

    def test_a_fenced_json_block_parses(self) -> None:
        """A model that ignores the JSON mode and fences its answer still parses."""
        raw = '```json\n{"%s": "amends"}\n```' % VERDICT_KEY
        assert parse_judge_verdict(raw) == OUTCOME_AMENDED

    def test_json_surrounded_by_prose_parses(self) -> None:
        """`extract_json` brace-scans, so leading/trailing prose is tolerated."""
        raw = (
            'Looking at candidate mem-1, the new entry adds a detail.\n'
            '{"%s": "amends"}\n'
            'That is my answer.' % VERDICT_KEY
        )
        assert parse_judge_verdict(raw) == OUTCOME_AMENDED

    def test_a_verdict_with_extra_keys_still_parses(self) -> None:
        """Extra keys are ignored; only the verdict word is contractual."""
        raw = '{"%s": "restates", "reasoning": "same fact"}' % VERDICT_KEY
        assert parse_judge_verdict(raw) == OUTCOME_RESTATED

    @pytest.mark.parametrize(
        ('raw', 'label'),
        [
            ('{"%s": "supersedes"}' % VERDICT_KEY, 'unrecognised-verdict-word'),
            ('{"%s": "stored"}' % VERDICT_KEY, 'ack-word-not-judge-word'),
            ('{"answer": "restates"}', 'missing-verdict-key'),
            ('{"%s": null}' % VERDICT_KEY, 'null-verdict'),
            ('{"%s": ["restates"]}' % VERDICT_KEY, 'non-string-verdict'),
            ('[{"%s": "restates"}]' % VERDICT_KEY, 'non-object-payload'),
            ('', 'empty-text'),
            ('   \n\t  ', 'whitespace-only-text'),
            ('I cannot answer that request.', 'prose-with-no-json'),
            ('{"%s": "restates"' % VERDICT_KEY, 'unbalanced-json'),
            ('{%s: restates}' % VERDICT_KEY, 'unquoted-json'),
        ],
        ids=[
            'unrecognised-verdict-word',
            'ack-word-not-judge-word',
            'missing-verdict-key',
            'null-verdict',
            'non-string-verdict',
            'non-object-payload',
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
            parse_judge_verdict('{"%s": "supersedes"}' % VERDICT_KEY)
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
