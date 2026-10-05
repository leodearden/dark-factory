"""Replay and probe inputs: δ's manifest into ReplayItems, transcript corpus into queries."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

from fused_memory.arm_harness.corpus import (
    CorpusIntegrityError,
    corpus_sha,
    select_replay_items,
)
from fused_memory.arm_harness.replay import ReplayItem
from fused_memory.arm_harness.transcript_queries import iter_transcript_queries

FUSED_MEMORY_ROOT = Path(__file__).parents[2]
LME_DIR = FUSED_MEMORY_ROOT / 'scripts' / 'local_memory_models_eval'
COMMITTED_MANIFEST = LME_DIR / 'corpus_manifest.json'

build_corpus = load_script_module(LME_DIR / 'build_corpus.py', mod_name='lme_build_corpus')
transcript_corpus = load_script_module(
    FUSED_MEMORY_ROOT / 'scripts' / 'memory_eval_transcript_corpus.py',
    mod_name='memory_eval_transcript_corpus',
)

STAMP = '20261005T120000Z'


def _episode(uuid: str, content: str, created_at: str = '2026-05-16T10:00:00+00:00'):
    return build_corpus.EpisodeRecord(
        uuid=uuid,
        name=f'name-{uuid}',
        group_id='dark_factory',
        source_description='add_memory:temporal_facts',
        created_at=created_at,
        content=content,
    )


def _manifest_for(*episodes) -> dict:
    return {
        'episodes': [
            {'uuid': episode.uuid, 'content_hash': build_corpus.content_hash(episode.content)}
            for episode in episodes
        ]
    }


EP_A = _episode('ep-a', 'alpha body', created_at='2026-05-16T10:00:00+02:00')
EP_B = _episode('ep-b', 'beta body', created_at='2026-06-01T08:30:00Z')
EP_C = _episode('ep-c', 'gamma body', created_at='2026-07-04 12:00:00')


class TestCorpusSha:
    def test_is_sha256_of_the_exact_bytes(self):
        assert corpus_sha(b'{}\n') == corpus_sha(b'{}\n')
        assert corpus_sha(b'{}\n') != corpus_sha(b'{} \n')

    def test_committed_manifest_parses_with_200_episodes_and_a_64_hex_sha(self):
        manifest_bytes = COMMITTED_MANIFEST.read_bytes()
        assert len(json.loads(manifest_bytes)['episodes']) == 200
        sha = corpus_sha(manifest_bytes)
        assert len(sha) == 64
        assert set(sha) <= set('0123456789abcdef')


class TestSelectReplayItems:
    def test_items_come_back_in_manifest_order(self):
        manifest = _manifest_for(EP_C, EP_A, EP_B)
        items = select_replay_items(
            manifest, [EP_A, EP_B, EP_C], content_hash=build_corpus.content_hash
        )
        assert [item.episode_id for item in items] == ['ep-c', 'ep-a', 'ep-b']

    def test_item_carries_the_episode_body_and_metadata(self):
        (item,) = select_replay_items(
            _manifest_for(EP_B), [EP_B], content_hash=build_corpus.content_hash
        )
        assert item == ReplayItem(
            episode_id='ep-b',
            name='name-ep-b',
            content='beta body',
            source_description='add_memory:temporal_facts',
            reference_time=datetime(2026, 6, 1, 8, 30, tzinfo=UTC),
        )

    @pytest.mark.parametrize(
        ('episode', 'expected'),
        [
            (EP_A, datetime(2026, 5, 16, 8, 0, tzinfo=UTC)),
            (EP_B, datetime(2026, 6, 1, 8, 30, tzinfo=UTC)),
            (EP_C, datetime(2026, 7, 4, 12, 0, tzinfo=UTC)),
        ],
        ids=['offset', 'z-suffix', 'naive-space-separator'],
    )
    def test_reference_time_is_aware_utc(self, episode, expected):
        (item,) = select_replay_items(
            _manifest_for(episode), [episode], content_hash=build_corpus.content_hash
        )
        assert item.reference_time == expected
        assert item.reference_time.tzinfo is UTC

    def test_population_records_outside_the_manifest_are_ignored(self):
        items = select_replay_items(
            _manifest_for(EP_A), [EP_A, EP_B, EP_C], content_hash=build_corpus.content_hash
        )
        assert [item.episode_id for item in items] == ['ep-a']

    def test_missing_episode_raises_listing_it(self):
        with pytest.raises(CorpusIntegrityError) as caught:
            select_replay_items(
                _manifest_for(EP_A, EP_B, EP_C),
                [EP_A, EP_C],
                content_hash=build_corpus.content_hash,
            )
        assert caught.value.missing_ids == ('ep-b',)
        assert caught.value.drifted_ids == ()
        assert 'ep-b' in str(caught.value)

    def test_drifted_body_raises_listing_it(self):
        drifted_c = _episode('ep-c', 'gamma body, edited after the manifest was cut')
        with pytest.raises(CorpusIntegrityError) as caught:
            select_replay_items(
                _manifest_for(EP_A, EP_B, EP_C),
                [EP_A, EP_B, drifted_c],
                content_hash=build_corpus.content_hash,
            )
        assert caught.value.drifted_ids == ('ep-c',)
        assert caught.value.missing_ids == ()
        assert 'ep-c' in str(caught.value)

    def test_missing_and_drifted_are_reported_together(self):
        drifted_a = _episode('ep-a', 'alpha body ')
        with pytest.raises(CorpusIntegrityError) as caught:
            select_replay_items(
                _manifest_for(EP_A, EP_B, EP_C),
                [drifted_a, EP_C],
                content_hash=build_corpus.content_hash,
            )
        assert caught.value.missing_ids == ('ep-b',)
        assert caught.value.drifted_ids == ('ep-a',)


def _search_turn(tool_use_id: str, query: str) -> dict:
    return {
        'message': {
            'content': [
                {
                    'type': 'tool_use',
                    'id': tool_use_id,
                    'name': 'mcp__fused-memory__search',
                    'input': {'query': query, 'project_id': 'dark_factory'},
                }
            ]
        }
    }


def _result_turn(tool_use_id: str, *, is_error: bool = False) -> dict:
    return {
        'message': {
            'content': [
                {
                    'type': 'tool_result',
                    'tool_use_id': tool_use_id,
                    'is_error': is_error,
                    'content': json.dumps({'results': [{'id': 'm1', 'content': 'x'}]}),
                }
            ]
        }
    }


def _write_transcript_corpus(tmp_path: Path) -> Path:
    transcript = [
        _search_turn('t1', 'how does the merge lane park?'),
        _result_turn('t1'),
        _search_turn('t2', 'a search whose call failed'),
        _result_turn('t2', is_error=True),
        _search_turn('t3', 'a search whose answer never arrived'),
        _search_turn('t4', 'what is INV-4?'),
        _result_turn('t4'),
    ]
    records = transcript_corpus.extract_searches(transcript, source='3718/x/session.jsonl')
    coverage = transcript_corpus.stamp_status({'transcripts_found': 1, 'transcripts_read': 1})
    corpus_path, _, _ = transcript_corpus.write_corpus(records, coverage, tmp_path, stamp=STAMP)
    return corpus_path


def _append_line(path: Path, line: str) -> None:
    with path.open('a', encoding='utf-8') as handle:
        handle.write(line + '\n')


class TestIterTranscriptQueries:
    def test_yields_only_ok_queries_in_file_order(self, tmp_path):
        corpus_path = _write_transcript_corpus(tmp_path)
        assert list(iter_transcript_queries(corpus_path)) == [
            'how does the merge lane park?',
            'what is INV-4?',
        ]

    def test_blank_lines_are_skipped(self, tmp_path):
        corpus_path = _write_transcript_corpus(tmp_path)
        _append_line(corpus_path, '')
        _append_line(corpus_path, '   ')
        assert len(list(iter_transcript_queries(corpus_path))) == 2

    def test_unknown_schema_version_raises_naming_the_line(self, tmp_path):
        corpus_path = _write_transcript_corpus(tmp_path)
        line_count = len(corpus_path.read_text(encoding='utf-8').splitlines())
        _append_line(
            corpus_path,
            json.dumps({'schema_version': 2, 'query': 'future', 'result_status': 'ok'}),
        )
        with pytest.raises(ValueError, match=rf'line {line_count + 1}\b'):
            list(iter_transcript_queries(corpus_path))

    def test_ok_record_without_a_string_query_raises_naming_the_line(self, tmp_path):
        corpus_path = _write_transcript_corpus(tmp_path)
        line_count = len(corpus_path.read_text(encoding='utf-8').splitlines())
        _append_line(
            corpus_path,
            json.dumps({'schema_version': 1, 'query': None, 'result_status': 'ok'}),
        )
        with pytest.raises(ValueError, match=rf'line {line_count + 1}\b'):
            list(iter_transcript_queries(corpus_path))
