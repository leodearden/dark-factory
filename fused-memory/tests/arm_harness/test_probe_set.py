"""The embedding axis's fixed probe set (arm_harness/probe_set.py)."""

import hashlib
from pathlib import Path

import pytest
from graphiti_core.search import search_utils
from pydantic import ValidationError

from fused_memory.arm_harness.probe_set import (
    FULLTEXT_TERM_CUTOFF,
    SCHEMA_VERSION,
    TRANSCRIPT_QUERY_CAP,
    FrozenReference,
    KnownItem,
    Mem0KnownItem,
    Mem0SnapshotPin,
    ProbeSet,
    ProbeSetError,
    TranscriptPin,
    build_probe_set,
    derive_query_words,
    known_item_query,
    load_probe_set,
    probe_set_sha,
    serialize_probe_set,
)
from fused_memory.backends.falkor_fulltext import build_query

CORPUS_SHA = 'b' * 64
REFERENCE = FrozenReference(
    graph='evalmem_lme_ref_incumbent_a',
    node_count=1834,
    edge_count=3101,
    topology_hash='56de651d10aff70b81e78911cc125eb8be0e81a254a856ef8e3c923c01a9085f',
)
UUID_A = '1a1a1a1a-0000-4000-8000-00000000000a'
UUID_B = '2b2b2b2b-0000-4000-8000-00000000000b'
UUID_C = '3c3c3c3c-0000-4000-8000-00000000000c'
EPISODES = (
    (UUID_B, 'merge lane halted on a\nred verify after the rebase'),
    (UUID_A, 'alpha beta  gamma delta epsilon'),
    (UUID_C, 'an episode no edge cites'),
)
CITED = frozenset({UUID_A, UUID_B})
CONTROL_OK = frozenset({UUID_A, UUID_B, UUID_C})
TRANSCRIPT = TranscriptPin(
    path='2026/10/corpus-20261009T000000Z.jsonl',
    sha256='d' * 64,
    queries=('one two three', 'one two three four', 'one two three four five'),
)
SNAPSHOT = Mem0SnapshotPin(
    source='fused_dark_factory', sha256='e' * 64, point_count=30941, excluded_empty=3
)
MEM0_ITEMS = (
    Mem0KnownItem(
        topic='merge-lane',
        phrasing='how does the merge lane verify',
        held_out=False,
        canonical_content_hash='0123456789abcdef',
        canonical_last_known_id='9e8d7c6b-5a49-4382-9170-6f5e4d3c2b03',
    ),
    Mem0KnownItem(
        topic='code-quality',
        phrasing='where is code quality defined',
        held_out=True,
        canonical_content_hash='fedcba9876543210',
        canonical_last_known_id=None,
    ),
)


def _build(**overrides) -> ProbeSet:
    arguments = {
        'corpus_sha': CORPUS_SHA,
        'reference': REFERENCE,
        'episodes': EPISODES,
        'cited_episode_uuids': CITED,
        'control_ok_episode_uuids': CONTROL_OK,
        'transcript': TRANSCRIPT,
        'mem0_snapshot': SNAPSHOT,
        'mem0_known_items': MEM0_ITEMS,
    }
    return build_probe_set(**(arguments | overrides))


def _words(count: int) -> str:
    return ' '.join(f'term{index}' for index in range(count))


# --- known_item_query --------------------------------------------------------------------


def test_a_known_item_query_is_the_first_words_joined_by_single_spaces():
    assert known_item_query('alpha  beta\ngamma\tdelta epsilon', 3) == 'alpha beta gamma'


def test_a_known_item_query_is_the_whole_content_when_it_is_shorter():
    assert known_item_query('one\n two', 5) == 'one two'


# --- derive_query_words ------------------------------------------------------------------


def test_query_words_is_the_median_word_count():
    assert derive_query_words(['a', 'a b c', 'a b c d e f g']) == 3


@pytest.mark.parametrize(
    ('queries', 'expected'),
    [(['a b', 'a b c'], 3), (['a', 'a b', 'a b c', 'a b c d'], 3), (['a b', 'a b c d'], 3)],
)
def test_an_even_median_rounds_half_up(queries, expected):
    assert derive_query_words(queries) == expected


def test_no_transcript_query_is_refused():
    with pytest.raises(ProbeSetError, match='transcript'):
        derive_query_words([])


def test_a_median_of_zero_words_is_refused():
    with pytest.raises(ProbeSetError, match='0'):
        derive_query_words(['', '   ', 'a'])


def test_the_cutoff_is_graphitis_own_constant():
    assert FULLTEXT_TERM_CUTOFF == search_utils.MAX_QUERY_LENGTH


def test_the_bm25_leg_dies_at_64_searchable_terms_not_128():
    alive = [
        count
        for count in range(1, FULLTEXT_TERM_CUTOFF)
        if build_query(_words(count), ['evalmem_probe'], FULLTEXT_TERM_CUTOFF)
    ]

    assert max(alive) == 63
    assert derive_query_words([_words(63)]) == 63
    with pytest.raises(ProbeSetError, match='64'):
        derive_query_words([_words(64)])


# --- build_probe_set ---------------------------------------------------------------------


def test_known_items_are_the_cited_episodes_sorted_by_uuid_with_their_prefix_queries():
    probe_set = _build()

    assert probe_set.query_words == 4
    assert probe_set.known_items == (
        KnownItem(episode_uuid=UUID_A, query='alpha beta gamma delta'),
        KnownItem(episode_uuid=UUID_B, query='merge lane halted on'),
    )


def test_uncited_episodes_are_disclosed_as_a_count():
    assert _build().uncited_episodes == 1


def test_the_probe_set_carries_its_provenance_and_pins_verbatim():
    probe_set = _build()

    assert probe_set.schema_version == SCHEMA_VERSION == 1
    assert probe_set.corpus_sha == CORPUS_SHA
    assert probe_set.reference == REFERENCE
    assert probe_set.transcript == TRANSCRIPT
    assert probe_set.mem0_snapshot == SNAPSHOT
    assert set(probe_set.mem0_known_items) == set(MEM0_ITEMS)


def test_episodes_that_differ_from_control_as_ok_replays_are_refused_naming_both_sides():
    extra = '4d4d4d4d-0000-4000-8000-00000000000d'

    with pytest.raises(ProbeSetError) as caught:
        _build(control_ok_episode_uuids=frozenset({UUID_A, UUID_B, extra}))

    assert extra in str(caught.value)
    assert UUID_C in str(caught.value)


def test_a_cited_uuid_that_is_no_reference_episode_is_refused():
    stray = '5e5e5e5e-0000-4000-8000-00000000000e'

    with pytest.raises(ProbeSetError, match=stray):
        _build(cited_episode_uuids=CITED | {stray})


def test_no_cited_episode_is_refused():
    with pytest.raises(ProbeSetError, match='cited'):
        _build(cited_episode_uuids=frozenset())


def test_a_known_item_query_whose_tokens_split_past_the_cutoff_is_refused():
    dotted = '.'.join('bcdefghijklmnopq')
    episodes = ((UUID_A, ' '.join([dotted] * 4)), *EPISODES[:1], *EPISODES[2:])

    with pytest.raises(ProbeSetError, match=UUID_A):
        _build(episodes=episodes)


def test_a_transcript_pin_holds_at_most_the_query_cap():
    with pytest.raises(ValidationError):
        TranscriptPin(
            path='corpus.jsonl', sha256='d' * 64, queries=('q',) * (TRANSCRIPT_QUERY_CAP + 1)
        )


# --- serialize / load / sha --------------------------------------------------------------


def test_a_serialized_probe_set_loads_back_identical(tmp_path: Path):
    path = tmp_path / 'probe-set.json'
    path.write_text(serialize_probe_set(_build()), encoding='utf-8')

    assert load_probe_set(path) == _build()


def test_the_serialization_is_canonical_whatever_the_input_order():
    shuffled = _build(episodes=tuple(reversed(EPISODES)), mem0_known_items=MEM0_ITEMS[::-1])

    assert serialize_probe_set(shuffled) == serialize_probe_set(_build())


def test_the_probe_set_sha_is_the_sha256_of_its_bytes():
    data = serialize_probe_set(_build()).encode('utf-8')

    assert probe_set_sha(data) == hashlib.sha256(data).hexdigest()
