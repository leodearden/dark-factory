"""Tests for the runtime topic-cluster store and its auto-seed (task 3135).

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf ζ: every
``consolidate_memories`` run teaches the write-time topic guard its topic, by
persisting a derived ``ProceduralTopicCluster`` that the guard merges with the
config seeds.

Covers:
- ``TopicClusterStore`` durable lifecycle, restart round-trip and project
  scoping (TestStoreLifecycle)
- Mistyped/duplicate rows fail loud at ``open()`` like the config path does at
  load (TestStoreFailsLoudLikeTheConfigPath)
- ``derive_topic_cluster``'s conservative, abstaining, deterministic phrase
  derivation (TestDeriveTopicCluster)
- ``seed_topic_cluster``, the one non-raising seed and its outcome vocabulary
  (TestSeedTopicCluster)
- ``server.main._build_topic_cluster_store`` hands the server an OPEN store
  beside the other server-owned SQLite files (TestServerFactoryWiring)
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pytest

from fused_memory.config.schema import ProceduralTopicCluster
from fused_memory.server.near_duplicate_guard import find_matching_topic_cluster
from fused_memory.server.topic_cluster_store import (
    TopicClusterStore,
    TopicClusterStoreError,
    derive_topic_cluster,
    seed_topic_cluster,
)

_PROJECT = 'dark_factory'


def _cluster(
    topic_id: str = 'pytest-xdist-serial-override',
    phrases: list[str] | None = None,
    hint: str = 'Consolidated topic; update the canonical instead.',
) -> ProceduralTopicCluster:
    return ProceduralTopicCluster(
        topic_id=topic_id,
        phrases=phrases if phrases is not None else ['--dist loadgroup', 'max-worker-restart'],
        min_phrase_hits=2,
        sufficient_phrases=[],
        hint=hint,
    )


def _upsert(store: TopicClusterStore, cluster: ProceduralTopicCluster, *, project_id: str = _PROJECT) -> None:
    store.upsert(
        cluster,
        source='consolidate_memories',
        project_id=project_id,
        canonical_id=str(uuid.uuid4()),
        category='procedural_knowledge',
        run_id='r1',
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / 'sub' / 'topic_clusters.db'


class TestStoreLifecycle:
    """The store's durable contract: a derived cluster survives a restart."""

    def test_open_creates_missing_parent_dir_and_starts_empty(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        store.open()
        try:
            assert db_path.parent.is_dir()
            assert store.db_path == db_path
            assert store.list_clusters(_PROJECT) == []
        finally:
            store.close()

    def test_upsert_round_trips_the_real_model(self, db_path: Path) -> None:
        cluster = _cluster(phrases=['zeta-phrase', '--dist loadgroup', 'max-worker-restart'])
        store = TopicClusterStore(db_path)
        store.open()
        try:
            _upsert(store, cluster)
            listed = store.list_clusters(_PROJECT)
        finally:
            store.close()
        assert listed == [cluster]
        assert listed[0].phrases == ['zeta-phrase', '--dist loadgroup', 'max-worker-restart']

    def test_restart_preserves_the_cluster(self, db_path: Path) -> None:
        cluster = _cluster()
        first = TopicClusterStore(db_path)
        first.open()
        _upsert(first, cluster)
        first.close()

        second = TopicClusterStore(db_path)
        second.open()
        try:
            assert second is not first
            assert second.list_clusters(_PROJECT) == [cluster]
            assert second.list_clusters('reify') == []
        finally:
            second.close()

    def test_double_open_raises(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        store.open()
        try:
            with pytest.raises(RuntimeError):
                store.open()
        finally:
            store.close()

    def test_close_is_idempotent(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        store.close()
        store.open()
        store.close()
        store.close()

    def test_upsert_before_open_raises(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        with pytest.raises(RuntimeError):
            _upsert(store, _cluster())


def _create_schema(db_path: Path) -> None:
    store = TopicClusterStore(db_path)
    store.open()
    store.close()


def _insert_raw_row(db_path: Path, *, project_id: str, topic_id: str, cluster_json: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'INSERT INTO topic_clusters (project_id, topic_id, cluster_json, source, updated_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (project_id, topic_id, cluster_json, 'consolidate_memories', 0.0),
        )
        conn.commit()
    finally:
        conn.close()


def _row_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute('SELECT COUNT(*) FROM topic_clusters').fetchone()[0]
    finally:
        conn.close()


def _cluster_json(**overrides: object) -> str:
    payload = _cluster().model_dump(mode='json')
    payload.update(overrides)
    return json.dumps(payload)


_SNAKE_TOPIC = 'pytest_xdist_serial_override'

_CORRUPT_ROWS = {
    'snake_case_topic_id': (_SNAKE_TOPIC, _cluster_json(topic_id=_SNAKE_TOPIC)),
    'unknown_key': ('pytest-xdist-serial-override', _cluster_json(severity='high')),
    'non_json': ('pytest-xdist-serial-override', '{not json'),
}


class TestStoreFailsLoudLikeTheConfigPath:
    """A row that fails re-validation is a startup condition, never a silent skip."""

    @pytest.mark.parametrize('shape', sorted(_CORRUPT_ROWS))
    def test_a_corrupt_row_fails_open_naming_the_row_path_and_recovery(
        self, db_path: Path, shape: str
    ) -> None:
        topic_id, cluster_json = _CORRUPT_ROWS[shape]
        _create_schema(db_path)
        _insert_raw_row(db_path, project_id=_PROJECT, topic_id=topic_id, cluster_json=cluster_json)

        store = TopicClusterStore(db_path)
        with pytest.raises(TopicClusterStoreError) as excinfo:
            store.open()

        message = str(excinfo.value)
        assert _PROJECT in message
        assert topic_id in message
        assert str(db_path) in message
        assert 'machine-derived' in message
        assert 're-seed' in message

    def test_a_snake_case_topic_id_names_the_slug_rule(self, db_path: Path) -> None:
        _create_schema(db_path)
        _insert_raw_row(
            db_path, project_id=_PROJECT, topic_id=_SNAKE_TOPIC,
            cluster_json=_cluster_json(topic_id=_SNAKE_TOPIC),
        )
        with pytest.raises(TopicClusterStoreError, match=r'fused_memory\.topic_slug'):
            TopicClusterStore(db_path).open()

    def test_every_bad_row_is_named_in_one_error(self, db_path: Path) -> None:
        _create_schema(db_path)
        _insert_raw_row(
            db_path, project_id=_PROJECT, topic_id=_SNAKE_TOPIC,
            cluster_json=_cluster_json(topic_id=_SNAKE_TOPIC),
        )
        _insert_raw_row(db_path, project_id='reify', topic_id='broken-row', cluster_json='{not json')

        with pytest.raises(TopicClusterStoreError) as excinfo:
            TopicClusterStore(db_path).open()

        message = str(excinfo.value)
        assert _SNAKE_TOPIC in message
        assert 'broken-row' in message
        assert 'reify' in message

    def test_a_failed_open_leaks_no_half_open_state(self, db_path: Path) -> None:
        _create_schema(db_path)
        _insert_raw_row(db_path, project_id=_PROJECT, topic_id='broken-row', cluster_json='{not json')
        store = TopicClusterStore(db_path)
        with pytest.raises(TopicClusterStoreError):
            store.open()

        conn = sqlite3.connect(str(db_path))
        conn.execute('DELETE FROM topic_clusters')
        conn.commit()
        conn.close()

        store.open()
        try:
            assert store.list_clusters(_PROJECT) == []
        finally:
            store.close()

    def test_a_duplicate_upsert_replaces_the_row(self, db_path: Path) -> None:
        earlier = _cluster(phrases=['--dist loadgroup', 'max-worker-restart'])
        later = _cluster(phrases=['pytest-xdist', 'xdist_group'], hint='New canonical.')
        store = TopicClusterStore(db_path)
        store.open()
        try:
            _upsert(store, earlier)
            _upsert(store, later)
            listed = store.list_clusters(_PROJECT)
        finally:
            store.close()
        assert _row_count(db_path) == 1
        assert listed == [later]

    def test_the_same_slug_in_two_projects_is_two_rows(self, db_path: Path) -> None:
        ours = _cluster(phrases=['--dist loadgroup', 'max-worker-restart'])
        theirs = _cluster(phrases=['reify-only-phrase', 'reify_other_phrase'])
        store = TopicClusterStore(db_path)
        store.open()
        try:
            _upsert(store, ours, project_id=_PROJECT)
            _upsert(store, theirs, project_id='reify')
            assert store.list_clusters(_PROJECT) == [ours]
            assert store.list_clusters('reify') == [theirs]
        finally:
            store.close()
        assert _row_count(db_path) == 2

    def test_an_unvalidated_dict_is_rejected_before_any_sql(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        store.open()
        try:
            with pytest.raises(TypeError):
                store.upsert(
                    _cluster().model_dump(mode='json'),  # type: ignore[arg-type]
                    source='consolidate_memories',
                    project_id=_PROJECT,
                )
            assert store.list_clusters(_PROJECT) == []
        finally:
            store.close()
        assert _row_count(db_path) == 0


_XDIST_TEXTS = [
    'When running the fused-memory suite under pytest-xdist, pass --dist loadgroup so '
    'tests marked with the same group stay on one worker, and set --max-worker-restart 0 '
    'to fail fast.',
    'Gotcha: pytest-xdist workers crash silently unless you pin --max-worker-restart 0. '
    'Use --dist loadgroup for the serial SQLite tests.',
    'To serialise the WAL tests, run pytest-xdist with --dist loadgroup and keep '
    '--max-worker-restart at 0 so a crashed worker is visible.',
]

_UNRELATED_NOTE = (
    'When running the dashboard build, pass the flag so the page stays on one worker '
    'and fail fast when a crashed build is visible to the tests.'
)

_TOPIC = 'pytest-xdist-serial-override'
_HINT = 'Consolidated topic; update canonical 8bb3eb15 instead.'


def _derive(texts: list[str]) -> ProceduralTopicCluster | None:
    return derive_topic_cluster(texts, topic_id=_TOPIC, hint=_HINT)


def _phrases_in(text: str, phrases: list[str]) -> list[str]:
    return [phrase for phrase in phrases if phrase.lower() in text.lower()]


class TestDeriveTopicCluster:
    """Derivation abstains rather than emit a weak cluster (the measured over-block hazard)."""

    def test_near_duplicates_yield_a_conservative_cluster(self) -> None:
        cluster = _derive(_XDIST_TEXTS)

        assert isinstance(cluster, ProceduralTopicCluster)
        assert cluster.topic_id == _TOPIC
        assert cluster.hint == _HINT
        assert cluster.min_phrase_hits == 2
        assert cluster.sufficient_phrases == []
        assert 2 <= len(cluster.phrases) <= 6

    def test_every_phrase_is_supported_by_two_texts(self) -> None:
        cluster = _derive(_XDIST_TEXTS)
        assert cluster is not None
        for phrase in cluster.phrases:
            support = [text for text in _XDIST_TEXTS if phrase.lower() in text.lower()]
            assert len(support) >= 2, phrase

    def test_no_phrase_nests_inside_another(self) -> None:
        cluster = _derive(_XDIST_TEXTS)
        assert cluster is not None
        lowered = [phrase.lower() for phrase in cluster.phrases]
        for i, outer in enumerate(lowered):
            for j, inner in enumerate(lowered):
                if i != j:
                    assert inner not in outer, (inner, outer)

    def test_the_cluster_matches_its_own_sources(self) -> None:
        cluster = _derive(_XDIST_TEXTS)
        assert cluster is not None
        carriers = [text for text in _XDIST_TEXTS if len(_phrases_in(text, cluster.phrases)) >= 2]
        assert carriers
        for text in carriers:
            assert find_matching_topic_cluster(text, [cluster]) is not None

    def test_the_cluster_ignores_a_note_sharing_only_generic_english(self) -> None:
        cluster = _derive(_XDIST_TEXTS)
        assert cluster is not None
        assert find_matching_topic_cluster(_UNRELATED_NOTE, [cluster]) is None

    @pytest.mark.parametrize(
        'texts',
        [
            pytest.param([], id='empty'),
            pytest.param(_XDIST_TEXTS[:1], id='single_text'),
            pytest.param(
                [
                    'Always run the tests before you commit the change so the reviewer sees green.',
                    'Run the tests before you push the change and tell the reviewer about it.',
                ],
                id='generic_english_only',
            ),
            pytest.param(
                [
                    'Install pytest-xdist in the dev group before running anything in parallel.',
                    'The pytest-xdist plugin parallelises collection across all available cores.',
                ],
                id='one_qualifying_phrase',
            ),
            pytest.param(
                [_XDIST_TEXTS[0], _XDIST_TEXTS[0], 'Rotate the API key when the vault lease expires.'],
                id='identical_texts',
            ),
            pytest.param(
                [
                    _XDIST_TEXTS[0],
                    '  ' + _XDIST_TEXTS[0].upper().replace(' ', '\n  '),
                    'Rotate the API key when the vault lease expires.',
                ],
                id='case_and_whitespace_variant',
            ),
        ],
    )
    def test_abstains_when_it_has_too_little_to_go_on(self, texts: list[str]) -> None:
        assert _derive(texts) is None

    def test_is_deterministic_across_calls_and_input_order(self) -> None:
        first = _derive(_XDIST_TEXTS)
        second = _derive(_XDIST_TEXTS)
        shuffled_texts = list(_XDIST_TEXTS)
        random.Random(3135).shuffle(shuffled_texts)
        assert shuffled_texts != _XDIST_TEXTS
        shuffled = _derive(shuffled_texts)

        assert first is not None and second is not None and shuffled is not None
        assert first.phrases == second.phrases == shuffled.phrases


_CANONICAL = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
_GENERIC_TEXTS = [
    'Always run the tests before you commit the change so the reviewer sees green.',
    'Run the tests before you push the change and tell the reviewer about it.',
]


class _UpsertRaisesStore:
    def upsert(self, *args: Any, **kwargs: Any) -> None:
        raise sqlite3.OperationalError('disk I/O error')


def _seed(store: Any, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        'enabled': True,
        'texts': _XDIST_TEXTS,
        'topic': _TOPIC,
        'canonical_id': _CANONICAL,
        'project_id': _PROJECT,
        'category': 'procedural_knowledge',
        'run_id': 'run-1',
        'source': 'consolidate_memories',
    }
    kwargs.update(overrides)
    return seed_topic_cluster(store, **kwargs)


@pytest.fixture
def store(db_path: Path):
    opened = TopicClusterStore(db_path)
    opened.open()
    try:
        yield opened
    finally:
        opened.close()


class TestSeedTopicCluster:
    """The ONE non-raising seed; it owns the seeded/skipped/failed/disabled vocabulary."""

    def test_seeded_persists_the_derived_cluster(self, store: TopicClusterStore) -> None:
        outcome = _seed(store)

        (cluster,) = store.list_clusters(_PROJECT)
        assert outcome == {'outcome': 'seeded', 'topic_id': _TOPIC, 'phrases': cluster.phrases}
        assert cluster.phrases
        assert cluster.topic_id == _TOPIC

    def test_the_hint_names_the_canonical_and_the_escape_hatch(self, store: TopicClusterStore) -> None:
        _seed(store)

        (cluster,) = store.list_clusters(_PROJECT)
        assert _CANONICAL in cluster.hint
        assert 'allow_near_duplicate' in cluster.hint

    def test_skipped_when_the_texts_abstain(self, store: TopicClusterStore) -> None:
        outcome = _seed(store, texts=_GENERIC_TEXTS)

        assert outcome['outcome'] == 'skipped'
        assert isinstance(outcome['reason'], str) and outcome['reason']
        assert store.list_clusters(_PROJECT) == []

    def test_disabled_persists_nothing(self, store: TopicClusterStore) -> None:
        outcome = _seed(store, enabled=False)

        assert outcome == {'outcome': 'disabled'}
        assert store.list_clusters(_PROJECT) == []

    def test_a_failing_store_is_disclosed_not_raised(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger='fused_memory.server.topic_cluster_store'):
            outcome = _seed(_UpsertRaisesStore())

        assert outcome['outcome'] == 'failed'
        assert outcome['topic_id'] == _TOPIC
        assert outcome['error'] == 'disk I/O error'
        assert outcome['error_type'] == 'OperationalError'
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(_TOPIC in r.getMessage() for r in warnings)

    def test_reseeding_a_topic_refreshes_its_canonical(self, store: TopicClusterStore) -> None:
        second_canonical = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb'
        _seed(store)
        _seed(store, canonical_id=second_canonical)

        (cluster,) = store.list_clusters(_PROJECT)
        assert second_canonical in cluster.hint
        assert _CANONICAL not in cluster.hint
        assert _row_count(store.db_path) == 1


class TestServerFactoryWiring:
    """main.py builds the store once, opened, and a corrupt file fails startup loudly."""

    def test_the_factory_returns_an_open_store_in_the_data_dir(self, tmp_path: Path) -> None:
        from fused_memory.server.main import _build_topic_cluster_store  # noqa: PLC0415

        store = _build_topic_cluster_store(tmp_path)
        try:
            assert isinstance(store, TopicClusterStore)
            assert store.db_path == tmp_path / 'topic_clusters.db'
            assert store.list_clusters('p') == []
            _upsert(store, _cluster(), project_id='p')
            assert store.list_clusters('p') == [_cluster()]
        finally:
            store.close()

    def test_a_corrupt_row_fails_the_factory(self, tmp_path: Path) -> None:
        from fused_memory.server.main import _build_topic_cluster_store  # noqa: PLC0415

        db_path = tmp_path / 'topic_clusters.db'
        _create_schema(db_path)
        _insert_raw_row(db_path, project_id=_PROJECT, topic_id='broken-row', cluster_json='{not json')

        with pytest.raises(TopicClusterStoreError):
            _build_topic_cluster_store(tmp_path)
