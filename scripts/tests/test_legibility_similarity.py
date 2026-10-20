"""Tests for scripts/legibility/similarity.py — the census similarity stage:
the index over a codebook's open entries and pending candidates, the View one
run builds, its codebook transforms, its report section and its queue CLI
(plans/census-incremental-prd.md L7, D3, D4)."""
from __future__ import annotations

import math

import pytest
from legibility import config, embeddings, similarity

_MODEL = 'text-embedding-3-small'


def _sighting(session: str, *, date: str = '2026-08-01') -> dict:
    return {
        'date': date,
        'project': 'dark_factory',
        'session': session,
        'origin_phase': 'implement',
        'manifested_phase': 'verify',
    }


def _entry(entry_id: str, title: str, *, cause: str | None = None, status: str = 'open',
           sessions: tuple[str, ...] = ('s-entry',)) -> dict:
    entry = {
        'id': entry_id,
        'title': title,
        'severity': 'medium',
        'status': status,
        'origin_phase': 'implement',
        'manifested_phase': 'verify',
        'sightings': [_sighting(session) for session in sessions],
    }
    if cause is not None:
        entry['cause'] = cause
    return entry


def _candidate(cand_id: str, title: str, *, cause: str | None = 'a cause',
               disposition: str = 'pending', first_seen: str = '2026-08-01',
               sessions: tuple[str, ...] = ('s-1',), **extra: object) -> dict:
    return {
        'id': cand_id,
        'title': title,
        'cause': cause,
        'area': 'scripts',
        'first_seen': first_seen,
        'disposition': disposition,
        'sightings': [_sighting(session, date=first_seen) for session in sessions],
        **extra,
    }


def _codebook(entries=(), candidates=()) -> dict:
    return {'version': 2, 'entries': list(entries), 'candidates': list(candidates)}


def _novel(title: str, *, cause: str | None = 'novel cause', session: str = 's-novel') -> dict:
    return {
        'title': title,
        'summary': cause or title,
        'cause': cause,
        'area': 'scripts',
        'origin_phase': 'implement',
        'manifested_phase': 'verify',
        'evidence': [],
        'sightings': [{'session': session, 'origin_phase': 'implement', 'manifested_phase': 'verify'}],
    }


def _index(items, vectors_by_text):
    return similarity.Index.of(items, {item.key: vectors_by_text[item.text] for item in items})


class TestRecordText:
    def test_title_and_cause_collapsed_to_one_line(self):
        assert similarity.record_text('A title', 'first line\n  second\tline ') == (
            'A title — first line second line'
        )

    @pytest.mark.parametrize('cause', [None, '', '   '])
    def test_bare_title_without_a_cause(self, cause):
        assert similarity.record_text('A title', cause) == 'A title'


class TestItemsOf:
    def test_open_entries_pending_candidates_and_novel_clusters(self):
        cb = _codebook(
            entries=[
                _entry('entry-open', 'Open entry', cause='open cause', sessions=('s-a', 's-b', 's-a')),
                _entry('entry-partially', 'Partial entry', cause='p', status='partially'),
                _entry('entry-fixed', 'Fixed entry', cause='f', status='fixed'),
                _entry('entry-retired', 'Retired entry', cause='r', status='retired'),
            ],
            candidates=[
                _candidate('cand-20260801-1', 'Pending one', first_seen='2026-08-01'),
                _candidate('cand-20260801-2', 'Promoted', disposition='promoted', promoted_to='entry-open'),
                _candidate('cand-20260801-3', 'Rejected', disposition='rejected'),
            ],
        )
        items = similarity.items_of(cb, [_novel('A novel cluster')])

        by_kind = {kind: [item for item in items if item.kind is kind] for kind in similarity.Kind}
        assert [item.record_id for item in by_kind[similarity.Kind.ENTRY]] == ['entry-open', 'entry-partially']
        assert [item.record_id for item in by_kind[similarity.Kind.PENDING]] == ['cand-20260801-1']
        [novel] = by_kind[similarity.Kind.NOVEL]
        assert novel.record_id is None
        assert novel.title == 'A novel cluster'
        assert novel.text == 'A novel cluster — novel cause'
        assert novel.sessions == ('s-novel',)

        [open_entry] = [item for item in items if item.record_id == 'entry-open']
        assert open_entry.title == 'Open entry'
        assert open_entry.text == 'Open entry — open cause'
        assert open_entry.sessions == ('s-a', 's-b')
        [pending] = by_kind[similarity.Kind.PENDING]
        assert pending.first_seen == '2026-08-01'
        assert pending.text == 'Pending one — a cause'

    def test_novel_clusters_default_to_none(self):
        items = similarity.items_of(_codebook(candidates=[_candidate('cand-20260801-1', 'P')]))
        assert [item.kind for item in items] == [similarity.Kind.PENDING]

    @pytest.mark.parametrize('joined_by', [
        {'attached_by': {'method': 'similarity', 'score': 0.93, 'run_id': 'run-1'}},
        {'adjudicated_with': 'cand-20260802-1'},
    ])
    def test_entry_without_a_cause_takes_its_own_promoted_candidates(self, joined_by):
        cb = _codebook(
            entries=[_entry('entry-cand-20260802-1', 'The entry')],
            candidates=[
                _candidate('cand-20260801-9', 'An older joiner', cause='the joiner cause',
                           disposition='promoted', promoted_to='entry-cand-20260802-1', **joined_by),
                _candidate('cand-20260802-1', 'The entry', cause='the own cause',
                           disposition='promoted', promoted_to='entry-cand-20260802-1'),
            ],
        )
        [entry] = similarity.items_of(cb)

        assert entry.text == 'The entry — the own cause'


class TestIndex:
    def _items(self):
        cb = _codebook(
            entries=[_entry('entry-a', 'A'), _entry('entry-b', 'B')],
            candidates=[_candidate('cand-20260801-1', 'P', cause=None)],
        )
        return similarity.items_of(cb)

    def test_nearest_entry_is_the_highest_cosine_independent_of_length(self):
        items = self._items()
        index = _index(items, {'A': [10.0, 0.0, 0.0], 'B': [0.0, 0.1, 0.0], 'P': [1.0, 3.0, 0.0]})
        [pending] = [item for item in items if item.kind is similarity.Kind.PENDING]

        nearest = index.nearest_entry(pending)

        assert nearest is not None
        assert nearest.entry.record_id == 'entry-b'
        assert nearest.score == pytest.approx(3.0 / math.sqrt(10.0))

    def test_no_entries_means_no_nearest(self):
        items = similarity.items_of(_codebook(candidates=[_candidate('cand-20260801-1', 'P', cause=None)]))
        index = _index(items, {'P': [1.0, 0.0]})

        assert index.nearest_entry(items[0]) is None

    def test_clusters_are_single_link(self):
        cb = _codebook(candidates=[
            _candidate(f'cand-20260801-{n}', title, cause=None)
            for n, title in enumerate(['A', 'B', 'C', 'D'], start=1)
        ])
        items = similarity.items_of(cb)
        a_b = math.cos(math.radians(30))
        vectors = {
            'A': [1.0, 0.0, 0.0],
            'B': [math.cos(math.radians(30)), math.sin(math.radians(30)), 0.0],
            'C': [math.cos(math.radians(60)), math.sin(math.radians(60)), 0.0],
            'D': [0.0, 0.0, 1.0],
        }
        index = _index(items, vectors)

        clusters = index.clusters(items, a_b - 1e-9)

        assert sorted(sorted(item.title for item in cluster) for cluster in clusters) == [
            ['A', 'B', 'C'], ['D'],
        ]

    def test_a_threshold_above_every_pair_leaves_singletons(self):
        cb = _codebook(candidates=[
            _candidate('cand-20260801-1', 'A', cause=None), _candidate('cand-20260801-2', 'B', cause=None),
        ])
        items = similarity.items_of(cb)
        index = _index(items, {'A': [1.0, 0.0], 'B': [1.0, 1.0]})

        assert len(index.clusters(items, 0.99)) == 2


class TestRank:
    def _cluster(self, *candidates):
        return tuple(similarity.items_of(_codebook(candidates=candidates)))

    def test_order_sessions_then_newest_then_representative_id(self):
        two_sessions = self._cluster(
            _candidate('cand-20260801-1', 'Old', first_seen='2026-08-01', sessions=('s-1',)),
            _candidate('cand-20260801-2', 'Old too', first_seen='2026-08-01', sessions=('s-2',)),
        )
        newer = self._cluster(_candidate('cand-20260805-1', 'Newer', first_seen='2026-08-05'))
        older = self._cluster(_candidate('cand-20260803-1', 'Older', first_seen='2026-08-03'))
        same_day_low = self._cluster(_candidate('cand-20260805-2', 'Same day 2', first_seen='2026-08-05'))
        same_day_high = self._cluster(_candidate('cand-20260805-10', 'Same day 10', first_seen='2026-08-05'))

        ranked = similarity.rank([older, newer, same_day_low, two_sessions, same_day_high])

        assert [queued.representative.record_id for queued in ranked] == [
            'cand-20260801-2', 'cand-20260805-10', 'cand-20260805-2', 'cand-20260805-1', 'cand-20260803-1',
        ]
        assert ranked[0].sessions == 2
        assert [item.record_id for item in ranked[0].members] == ['cand-20260801-1', 'cand-20260801-2']

    def test_sessions_are_distinct_across_members(self):
        cluster = self._cluster(
            _candidate('cand-20260801-1', 'A', sessions=('s-1',)),
            _candidate('cand-20260801-2', 'B', sessions=('s-1',)),
        )
        [queued] = similarity.rank([cluster])

        assert queued.sessions == 1

    def test_representative_is_the_newest_member(self):
        cluster = self._cluster(
            _candidate('cand-20260809-3', 'Newest', first_seen='2026-08-09'),
            _candidate('cand-20260801-7', 'Oldest', first_seen='2026-08-01'),
            _candidate('cand-20260809-2', 'Same day, lower', first_seen='2026-08-09'),
        )
        [queued] = similarity.rank([cluster])

        assert queued.representative.record_id == 'cand-20260809-3'
        assert queued.newest == '2026-08-09'


class _KeyedEmbed:
    """A fake embed_fn: the vector the test chose for each text."""

    def __init__(self, vectors: dict[str, list[float]], *, fail_on: int | None = None) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []
        self._fail_on = fail_on

    def __call__(self, texts, *, model):
        self.calls.append(list(texts))
        if self._fail_on == len(self.calls):
            raise embeddings.EmbeddingFailed('embeddings endpoint answered HTTP 503')
        return [self.vectors[text] for text in texts]


def _angle(degrees: float) -> list[float]:
    """A unit vector in the x-y plane; cosine between two is cos(difference)."""
    return [math.cos(math.radians(degrees)), math.sin(math.radians(degrees)), 0.0]


_SETTINGS = config.Adjudication(attach_min_similarity=0.90, cluster_min_similarity=0.85)


def _view(tmp_path, cb, novel=(), *, vectors, settings=_SETTINGS, fake=None):
    fake = fake or _KeyedEmbed(vectors)
    view = similarity.build_view(
        cb, list(novel), embedder=embeddings.Embedder(fake, tmp_path / 'e.sqlite'), settings=settings,
    )
    return view, fake


class TestBuildView:
    def test_a_novel_cluster_near_an_entry_is_screened(self, tmp_path):
        cb = _codebook(entries=[_entry('entry-a', 'Entry A')])
        near, far = _novel('Near', cause=None), _novel('Far', cause=None)
        view, _ = _view(tmp_path, cb, [near, far], vectors={
            'Entry A': _angle(0), 'Near': _angle(20), 'Far': _angle(40),
        })

        [screened] = view.screened
        assert screened.item.title == 'Near'
        assert screened.entry_id == 'entry-a'
        assert screened.score == pytest.approx(math.cos(math.radians(20)), abs=1e-4)
        assert view.screens(near) is True
        assert view.screens(far) is False

    def test_a_pending_candidate_near_an_entry_is_attached_not_queued(self, tmp_path):
        cb = _codebook(
            entries=[_entry('entry-a', 'Entry A')],
            candidates=[
                _candidate('cand-20260801-1', 'Near', cause=None),
                _candidate('cand-20260801-2', 'Far', cause=None),
            ],
        )
        view, _ = _view(tmp_path, cb, vectors={'Entry A': _angle(0), 'Near': _angle(10), 'Far': _angle(90)})

        [attach] = view.attaches
        assert (attach.item.record_id, attach.entry_id) == ('cand-20260801-1', 'entry-a')
        assert attach.score == pytest.approx(math.cos(math.radians(10)), abs=1e-4)
        queued = [member.record_id for cluster in view.queue for member in cluster.members]
        assert queued == ['cand-20260801-2']

    @pytest.mark.parametrize('status', ['fixed', 'retired'])
    def test_a_candidate_near_only_a_fixed_entry_stays_queued(self, tmp_path, status):
        cb = _codebook(
            entries=[_entry('entry-a', 'Entry A', status=status), _entry('entry-b', 'Entry B')],
            candidates=[_candidate('cand-20260801-1', 'Near A', cause=None)],
        )
        view, _ = _view(tmp_path, cb, vectors={
            'Entry A': _angle(0), 'Entry B': _angle(90), 'Near A': _angle(1),
        })

        assert view.attaches == ()
        assert [cluster.representative.record_id for cluster in view.queue] == ['cand-20260801-1']

    def test_the_rest_is_clustered_and_ranked_with_its_depth(self, tmp_path):
        cb = _codebook(candidates=[
            _candidate('cand-20260801-1', 'P1', cause=None, sessions=('s-1',)),
            _candidate('cand-20260801-2', 'P2', cause=None, sessions=('s-2',)),
            _candidate('cand-20260805-1', 'P3', cause=None, first_seen='2026-08-05'),
            _candidate('cand-20260803-1', 'P4', cause=None, first_seen='2026-08-03'),
        ])
        view, _ = _view(tmp_path, cb, vectors={
            'P1': _angle(0), 'P2': _angle(20), 'P3': _angle(60), 'P4': _angle(120),
        })

        assert [
            [member.record_id for member in cluster.members] for cluster in view.queue
        ] == [['cand-20260801-1', 'cand-20260801-2'], ['cand-20260805-1'], ['cand-20260803-1']]
        assert view.depth == (4, 3)

    def test_select_takes_the_first_ranked_clusters(self, tmp_path):
        cb = _codebook(candidates=[
            _candidate(f'cand-20260801-{n}', f'P{n}', cause=None) for n in range(1, 4)
        ])
        view, _ = _view(tmp_path, cb, vectors={'P1': _angle(0), 'P2': _angle(60), 'P3': _angle(120)})

        assert view.select(2) == view.queue[:2]
        assert view.select(0) == ()

    def test_model_thresholds_and_embedding_stats_are_exposed(self, tmp_path):
        cb = _codebook(
            entries=[_entry('entry-a', 'Entry A')],
            candidates=[_candidate('cand-20260801-1', 'P1', cause=None)],
        )
        vectors = {'Entry A': _angle(0), 'P1': _angle(90), 'Novel': _angle(45)}
        _view(tmp_path, cb, vectors=vectors)

        view, fake = _view(tmp_path, cb, [_novel('Novel', cause=None)], vectors=vectors)

        assert view.skipped is None
        assert view.model == 'text-embedding-3-small'
        assert view.settings.attach_min_similarity == 0.90
        assert view.settings.cluster_min_similarity == 0.85
        assert fake.calls == [['Novel']]
        assert (view.requests, view.embedded, view.cached, view.pruned) == (1, 1, 2, 0)

    def test_an_embedding_failure_skips_the_whole_stage(self, tmp_path):
        cb = _codebook(
            entries=[_entry('entry-a', 'Entry A')],
            candidates=[_candidate('cand-20260801-1', 'Near', cause=None)],
        )
        fake = _KeyedEmbed({'Entry A': _angle(0), 'Near': _angle(1), 'Novel': _angle(2)}, fail_on=1)

        view, _ = _view(tmp_path, cb, [_novel('Novel', cause=None)], vectors={}, fake=fake)

        assert view.skipped is not None and '503' in view.skipped
        assert (view.screened, view.attaches, view.queue, view.depth) == ((), (), (), None)
        assert view.requests == 1
        assert view.screens(_novel('Novel', cause=None)) is False
        assert len(fake.calls) == 1

    def test_an_unreadable_cache_skips_the_whole_stage(self, tmp_path):
        (tmp_path / 'e.sqlite').write_bytes(b'not a database at all' * 50)
        cb = _codebook(candidates=[_candidate('cand-20260801-1', 'P1', cause=None)])

        view, fake = _view(tmp_path, cb, vectors={'P1': _angle(0)})

        assert view.skipped is not None and str(tmp_path / 'e.sqlite') in view.skipped
        assert (view.queue, view.depth, view.requests) == ((), None, 0)
        assert fake.calls == []

    def test_a_failure_is_never_retried_per_candidate(self, tmp_path):
        cb = _codebook(candidates=[
            _candidate(f'cand-20260801-{n}', f'P{n}', cause=None) for n in range(1, 6)
        ])
        fake = _KeyedEmbed({f'P{n}': _angle(n * 30) for n in range(1, 6)}, fail_on=1)

        view, _ = _view(tmp_path, cb, vectors={}, fake=fake)

        assert view.skipped is not None
        assert len(fake.calls) == 1
