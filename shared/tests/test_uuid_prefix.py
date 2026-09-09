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

import copy
from typing import Any

import pytest

from shared.uuid_prefix import PrefixToken, find_prefix_tokens, substitute

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


# --- the nested walk (D11) ----------------------------------------------


def test_reaches_the_4643_shape_and_reports_the_structured_path() -> None:
    """The literal DF 4643 instance: metadata.cluster_memory_ids, prefixes one level down.

    The markup guard scans top-level arguments only. D11 widens this detector
    precisely because the incident's prefixes lived inside a list inside a
    dict, and the measured field spread is 20+ paths.
    """
    arguments = {
        'project_root': '/home/leo/src/dark-factory',
        'metadata': {
            'cluster_memory_ids': [
                '48433882-ee71-480d-aff7-c91aa4640ff5',
                'ffa913a1-ffdc-4f30-b435-bb4f06771fd5',
                '8bec9cd6',
                'bff81530',
                '2b0a4f1c-0c1e-4f7a-9c3d-8a1b2c3d4e5f',
                'f1c4a651',
                '5d6e7f80-1a2b-3c4d-5e6f-708192a3b4c5',
                'b7b0f63b',
                '9e8d7c6b-5a49-4382-9170-6f5e4d3c2b1a',
            ],
        },
    }
    found = find_prefix_tokens(arguments)
    assert [t.path for t in found] == [
        ('metadata', 'cluster_memory_ids', 2),
        ('metadata', 'cluster_memory_ids', 3),
        ('metadata', 'cluster_memory_ids', 5),
        ('metadata', 'cluster_memory_ids', 7),
    ]
    assert [t.token for t in found] == ['8bec9cd6', 'bff81530', 'f1c4a651', 'b7b0f63b']
    # Four of nine entries are prefixes; the five full uuids are not tokens.
    assert len(arguments['metadata']['cluster_memory_ids']) == 9


def test_walks_dicts_inside_lists_inside_dicts() -> None:
    arguments = {'outer': [{'inner': ['see bff81530 here']}]}
    found = find_prefix_tokens(arguments)
    assert [t.path for t in found] == [('outer', 0, 'inner', 0)]
    assert found[0].token == 'bff81530'


def test_a_prefix_shaped_dict_key_is_not_reported() -> None:
    """Only VALUES are scanned. A key that looks like a prefix is not a citation."""
    arguments = {'bff81530': 'no identifiers in this value'}
    assert find_prefix_tokens(arguments) == ()


def test_a_nested_prefix_shaped_dict_key_is_not_reported() -> None:
    arguments = {'metadata': {'8bec9cd6': {'f1c4a651': 'plain prose'}}}
    assert find_prefix_tokens(arguments) == ()


def test_document_order_across_arguments_containers_and_spans() -> None:
    """Argument insertion order, then container order, then span within a string."""
    arguments = {
        'first': 'aaaaaaa1 then aaaaaaa2',
        'second': {'nested': ['bbbbbbb1', 'bbbbbbb2']},
        'third': 'ccccccc1',
    }
    found = find_prefix_tokens(arguments)
    assert [(t.path, t.token) for t in found] == [
        (('first',), 'aaaaaaa1'),
        (('first',), 'aaaaaaa2'),
        (('second', 'nested', 0), 'bbbbbbb1'),
        (('second', 'nested', 1), 'bbbbbbb2'),
        (('third',), 'ccccccc1'),
    ]


# --- purity and total-ness ----------------------------------------------


def test_input_is_not_mutated() -> None:
    arguments = {
        'content': 'see bff81530',
        'metadata': {'cluster_memory_ids': ['8bec9cd6', 'f1c4a651']},
    }
    before = copy.deepcopy(arguments)
    find_prefix_tokens(arguments)
    assert arguments == before


@pytest.mark.parametrize(
    'arguments',
    [
        pytest.param({}, id='empty-map'),
        pytest.param({'a': None}, id='none'),
        pytest.param({'a': 12345678}, id='int'),
        pytest.param({'a': 1.5}, id='float'),
        pytest.param({'a': True}, id='bool'),
        pytest.param({'a': {}}, id='empty-dict'),
        pytest.param({'a': []}, id='empty-list'),
        pytest.param({'a': ''}, id='empty-string'),
        pytest.param({'a': [None, 1, True, {}, []]}, id='mixed-non-strings'),
        pytest.param({'a': {'b': [{'c': None}]}}, id='nested-nones'),
    ],
)
def test_never_raises_on_hostile_json_shaped_input(arguments: dict) -> None:
    assert find_prefix_tokens(arguments) == ()


def test_non_string_leaves_are_skipped_not_coerced() -> None:
    """12345678 as an int is not a token — only str values are scanned."""
    assert find_prefix_tokens({'a': 12345678}) == ()
    assert find_prefix_tokens({'a': '12345678'}) != ()


def test_deep_nesting_does_not_raise_recursion_error() -> None:
    """5,000 alternating containers: the walk must be iterative, not recursive.

    A RecursionError here would be raised on the event-loop thread inside a
    middleware hook, turning a malformed argument map into an outage.
    """
    innermost: Any = ['see bff81530 here']
    node: Any = innermost
    for i in range(5_000):
        node = {'k': node} if i % 2 else [node]
    found = find_prefix_tokens({'root': node})
    assert len(found) == 1
    assert found[0].token == 'bff81530'


# --- substitute: span-exact, non-mutating, monotone ---------------------

FULL_BFF = 'bff81530-1c2d-4e3f-a5b6-7c8d9e0f1a2b'
FULL_8BE = '8bec9cd6-2d3e-4f50-b6c7-8d9e0f1a2b3c'
FULL_F1C = 'f1c4a651-3e4f-4051-c7d8-9e0f1a2b3c4d'
FULL_B7B = 'b7b0f63b-4f50-4162-d8e9-0f1a2b3c4d5e'


def test_substitute_replaces_only_the_span() -> None:
    arguments = {'content': 'before bff81530 after'}
    token = find_prefix_tokens(arguments)[0]
    result = substitute(arguments, token, FULL_BFF)
    assert result['content'] == f'before {FULL_BFF} after'


def test_substitute_leaves_an_adjacent_second_token_byte_identical() -> None:
    """Only the named span moves; a sibling token in the same string does not."""
    arguments = {'content': 'f1c4a651/b7b0f63b'}
    first, second = find_prefix_tokens(arguments)
    assert (first.token, second.token) == ('f1c4a651', 'b7b0f63b')
    result = substitute(arguments, first, FULL_F1C)
    assert result['content'] == f'{FULL_F1C}/b7b0f63b'


def test_substitute_does_not_mutate_its_input() -> None:
    arguments = {
        'content': 'see bff81530',
        'metadata': {'cluster_memory_ids': ['8bec9cd6', FULL_BFF]},
    }
    before = copy.deepcopy(arguments)
    substitute(arguments, find_prefix_tokens(arguments)[0], FULL_BFF)
    assert arguments == before


def test_substitute_returns_a_new_object_at_the_tokens_path() -> None:
    arguments = {'metadata': {'cluster_memory_ids': ['8bec9cd6', FULL_BFF]}}
    token = find_prefix_tokens(arguments)[0]
    result = substitute(arguments, token, FULL_8BE)
    assert result is not arguments
    assert result['metadata'] is not arguments['metadata']
    assert result['metadata']['cluster_memory_ids'] is not arguments['metadata']['cluster_memory_ids']


def test_substitute_shares_untouched_substructures_by_identity() -> None:
    """A PATH COPY, not a deep copy: only the containers on the route are rebuilt."""
    untouched = {'deep': ['no identifiers here']}
    arguments = {'content': 'see bff81530', 'other': untouched}
    result = substitute(arguments, find_prefix_tokens(arguments)[0], FULL_BFF)
    assert result['other'] is untouched


def test_substitute_at_a_nested_path_leaves_siblings_untouched() -> None:
    """PRD boundary row B8: index 0 expands, the full uuid at index 1 does not."""
    arguments = {'metadata': {'cluster_memory_ids': ['8bec9cd6', FULL_BFF]}}
    token = find_prefix_tokens(arguments)[0]
    assert token.path == ('metadata', 'cluster_memory_ids', 0)
    result = substitute(arguments, token, FULL_8BE)
    assert result['metadata']['cluster_memory_ids'] == [FULL_8BE, FULL_BFF]


# --- monotonicity: the safety property ----------------------------------


def test_result_has_the_original_token_as_a_prefix_of_the_replacement() -> None:
    """The MONOTONE invariant, stated positively.

    A wrong expansion stays visible in the stored text and reversible by
    truncation; a non-prefix replacement would be silent information loss.
    """
    arguments = {'content': 'see bff81530 here'}
    token = find_prefix_tokens(arguments)[0]
    result = substitute(arguments, token, FULL_BFF)
    assert FULL_BFF.startswith(token.token)
    assert result['content'][token.start :].startswith(token.token)


@pytest.mark.parametrize(
    'full_id',
    [
        pytest.param('c0ffee00-1c2d-4e3f-a5b6-7c8d9e0f1a2b', id='different-uuid'),
        pytest.param('bff8153', id='truncation-not-expansion'),
        pytest.param('', id='empty'),
        pytest.param('BFF81530-1c2d-4e3f-a5b6-7c8d9e0f1a2b', id='case-changed'),
    ],
)
def test_substitute_refuses_a_non_prefix_replacement(full_id: str) -> None:
    """The one door that could destroy information is closed at the door."""
    arguments = {'content': 'see bff81530 here'}
    token = find_prefix_tokens(arguments)[0]
    with pytest.raises(ValueError) as excinfo:
        substitute(arguments, token, full_id)
    message = str(excinfo.value)
    assert token.token in message
    assert repr(full_id) in message or full_id in message


# --- ordering: reverse document order is what keeps spans valid ---------


def test_two_substitutions_in_reverse_document_order_are_both_span_exact() -> None:
    arguments = {'content': 'f1c4a651/b7b0f63b'}
    first, second = find_prefix_tokens(arguments)
    result = substitute(arguments, second, FULL_B7B)
    result = substitute(result, first, FULL_F1C)
    assert result['content'] == f'{FULL_F1C}/{FULL_B7B}'


def test_forward_order_is_what_would_corrupt_the_second_span() -> None:
    """Why step-14 folds in REVERSE order — pinned here as a property of the offsets.

    Applying the earlier token first shifts every later offset by the length
    the string grew, so the second token's recorded span no longer covers it.
    """
    arguments = {'content': 'f1c4a651/b7b0f63b'}
    first, second = find_prefix_tokens(arguments)
    after_first = substitute(arguments, first, FULL_F1C)
    assert after_first['content'][second.start : second.end] != second.token
