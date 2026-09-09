"""C1 detection contract — the hand-authored half.

The rows here are the ones a human decided, transcribed from
``plans/uuid-prefix-resolution-prd.md`` §4-C1 (normative). They pin
CORRECTNESS. The committed fixture corpus replayed by
``test_uuid_prefix_corpus.py`` pins regression-stability and determinism over
real contexts and deliberately claims nothing about correctness — the two
files divide the work exactly as the markup guard's do.

No test in this file asserts on docstring or comment prose.
"""

from __future__ import annotations

import pytest

from shared.uuid_prefix import PrefixToken, find_prefix_tokens

# A 31-hex run: the INCLUSIVE upper bound of the grammar.
HEX31 = 'bff81530aacc4f1e9d2b7c6a5e4d3f0'
# One character longer: above the ceiling, so not a token.
HEX32 = HEX31 + 'a'


def tokens_for(value: str) -> tuple[PrefixToken, ...]:
    """Run the detector over a flat single-key argument map."""
    return find_prefix_tokens({'content': value})


def spans_for(value: str) -> list[str]:
    return [t.token for t in tokens_for(value)]


# --- positive rows: what IS a token -------------------------------------


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        pytest.param('see bff81530 for the record', ['bff81530'], id='plain-8-hex'),
        pytest.param(f'see {HEX31} here', [HEX31], id='31-hex-upper-bound-inclusive'),
        # 131 all-decimal tokens are genuine class-A references, so an
        # all-decimal run is NOT excluded by the grammar. A date that happens
        # to look like one falls to the `none` outcome at the resolver, not to
        # a detector rule.
        pytest.param('on 20260904 the sweep ran', ['20260904'], id='all-decimal-kept'),
        pytest.param('bff81530', ['bff81530'], id='whole-string-is-the-token'),
        pytest.param('(bff81530)', ['bff81530'], id='punctuation-delimited'),
        pytest.param('`bff81530`', ['bff81530'], id='backtick-delimited'),
        pytest.param('id=bff81530.', ['bff81530'], id='trailing-period'),
        # PRD boundary row B7: the glue rule must keep '/'-separated lists,
        # which is where nearly all the true references it costs would live.
        pytest.param(
            'f1c4a651/b7b0f63b',
            ['f1c4a651', 'b7b0f63b'],
            id='B7-slash-list-keeps-both',
        ),
        pytest.param(
            'bff81530 and 8bec9cd6',
            ['bff81530', '8bec9cd6'],
            id='two-tokens-one-string',
        ),
    ],
)
def test_grammar_accepts(value: str, expected: list[str]) -> None:
    assert spans_for(value) == expected


# --- negative rows: what is NOT a token ---------------------------------


@pytest.mark.parametrize(
    ('value', 'why'),
    [
        pytest.param('see bff8153 here', 'below the 8-char floor', id='7-hex-floor'),
        pytest.param(f'see {HEX32} here', 'above the 31-char ceiling', id='32-hex-ceiling'),
        # The not-followed-by-hyphen rule is what makes group 1 of a full uuid
        # never match, so no separate uuid predicate is needed (INV-5).
        pytest.param('bff81530-aacc', "followed by '-'", id='followed-by-hyphen'),
        # THE GLUE RULE. Composite identifiers are 29% of the noise class.
        pytest.param('recon-3f2a9c1e', "preceded by '-'", id='glue-hyphen-recon'),
        pytest.param('episode_74b902f8', "preceded by '_'", id='glue-underscore-episode'),
        pytest.param('STAGE2_0b179fd4', "preceded by '_'", id='glue-underscore-stage2'),
        pytest.param('DEADBEEF', 'uppercase hex is not matched', id='uppercase'),
        pytest.param('BFF81530', 'uppercase hex is not matched', id='uppercase-real-prefix'),
        pytest.param('', 'the empty string', id='empty-string'),
        pytest.param('no identifiers at all here', 'ordinary prose', id='prose'),
    ],
)
def test_grammar_rejects(value: str, why: str) -> None:
    assert tokens_for(value) == (), why


def test_no_sub_run_of_a_longer_hex_run_is_reported() -> None:
    """"Not preceded/followed by a hex char" is an ANTI-SUBSTRING rule.

    It does not reject a run that happens to sit next to a letter — it stops
    the detector reporting a sub-run of a longer hex run. A 9-hex run is one
    9-char token, never the 8-char prefix or the 8-char suffix inside it.
    """
    run = 'abff81530'
    assert len(run) == 9
    found = spans_for(f'see {run} here')
    assert found == [run]
    assert 'abff8153' not in found
    assert 'bff81530' not in found


def test_over_ceiling_run_reports_no_sub_run_either() -> None:
    """The strongest form of the same rule: a 32-hex run yields NOTHING.

    Every one of its many 8-31 length sub-runs is suppressed, which is why a
    bare undashed uuid cannot be mistaken for a prefix.
    """
    assert tokens_for(f'see {HEX32} here') == ()
    assert HEX31 in HEX32


def test_full_uuid_in_prose_yields_no_tokens() -> None:
    """A dead full uuid is provenance by corpus convention, never a token.

    Every group is excluded by the grammar alone: group 1 is 8 hex followed by
    '-', groups 2-4 are 4 hex (below the floor), group 5 is 12 hex preceded by
    '-'. That is how INV-5 is honoured without a second full-uuid predicate.
    """
    prose = 'superseded by 48433882-ee71-480d-aff7-c91aa4640ff5 on the same day'
    assert tokens_for(prose) == ()


def test_second_full_uuid_in_prose_yields_no_tokens() -> None:
    prose = 'and also ffa913a1-ffdc-4f30-b435-bb4f06771fd5 in the same record'
    assert tokens_for(prose) == ()


def test_bare_32_char_undashed_uuid_yields_no_tokens() -> None:
    """32 hex undashed exceeds the 31-char ceiling, so it is not a token."""
    undashed = '48433882ee71480daff7c91aa4640ff5'
    assert len(undashed) == 32
    assert tokens_for(undashed) == ()


# --- the reported shape -------------------------------------------------


def test_token_carries_flat_path_and_exact_span() -> None:
    found = tokens_for('see bff81530 for it')
    assert found == (PrefixToken(path=('content',), token='bff81530', start=4, end=12),)


@pytest.mark.parametrize(
    'value',
    [
        'see bff81530 for the record',
        f'see {HEX31} here',
        'on 20260904 the sweep ran',
        'bff81530',
        'f1c4a651/b7b0f63b',
        'bff81530 and 8bec9cd6',
    ],
)
def test_span_is_exact_on_every_positive_row(value: str) -> None:
    """value[start:end] is the token itself — the invariant substitute relies on."""
    found = tokens_for(value)
    assert found
    for token in found:
        assert value[token.start : token.end] == token.token
