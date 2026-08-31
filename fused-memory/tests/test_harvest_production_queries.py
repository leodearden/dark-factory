"""Tests for harvest_production_queries.py — the production query set (task 4004).

The harvester reads the live reconciliation write journal READ-ONLY and
samples the query shapes that actually reach `search` in production, so the
read transforms can be scored on real traffic rather than only on the
blind-authored E2 query set.

Every test here builds a SYNTHETIC SQLite DB in ``tmp_path`` with the real
``write_ops`` column shape.  No test in this file may open the live journal:
it is a ~10 GB file the running fused-memory server is writing to, and a test
that opened it would be measuring a moving target under xdist.

The script is loaded via importlib so it can be tested without sys.path
pollution — the loader is copied verbatim from
``test_bake_off_storage_shape.py:48-73`` and is invoked lazily.

LANE DISCIPLINE — READ BEFORE ADDING A TEST
-------------------------------------------
Every test in this file must be free of network, Qdrant and OPENAI_API_KEY.
If a live test is ever added it carries its markers PER-TEST
(``@pytest.mark.integration`` + ``@pytest.mark.timeout(N)`` +
``qdrant_skipif()`` + an OPENAI_API_KEY skipif), never via a module-level
``pytestmark``: ``fused-memory/pyproject.toml`` sets
``addopts = "-n auto --dist loadgroup -m 'not integration'"``, so a
module-level integration marker would deselect every pure test in this file
from the merge lane too — see the same warning at
``test_bake_off_storage_shape.py:9-24``.
"""
from __future__ import annotations

import functools
import importlib.util
import types
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'harvest_production_queries.py'
)

FIXTURES_DIR = Path(__file__).parent / 'fixtures'


def _load_module() -> types.ModuleType:
    """Load harvest_production_queries.py from its file path.

    The module is registered in sys.modules under its bare name so that
    @dataclass and other reflection-based decorators work correctly (they
    call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'harvest_production_queries'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


@functools.cache
def _mod() -> types.ModuleType:
    return _load_module()


# ---------------------------------------------------------------------------
# Synthetic journal builder
# ---------------------------------------------------------------------------
# The real `write_ops` DDL, copied from the live journal (read-only inspection
# at plan time). Only the columns the harvester reads are load-bearing, but the
# full shape is kept so a schema drift in the real journal surfaces here.
WRITE_OPS_DDL = """
CREATE TABLE write_ops (
    id TEXT PRIMARY KEY,
    causation_id TEXT,
    source TEXT,
    provenance TEXT DEFAULT 'original',
    operation TEXT,
    project_id TEXT,
    agent_id TEXT,
    session_id TEXT,
    kind TEXT NOT NULL DEFAULT 'write',
    params TEXT DEFAULT '{}',
    result_summary TEXT,
    success INTEGER DEFAULT 1,
    error TEXT,
    created_at TEXT NOT NULL,
    terminal_status TEXT,
    terminal_at TEXT,
    terminal_error TEXT
)
"""

OVERVIEW = 'project overview architecture goals'
CONVENTIONS = 'coding conventions and project norms'
DECISIONS = 'recent decisions and rationale'
TASK_TEMPLATE = 'task {task_id} context and related decisions'


def _build_journal(
    path: Path,
    rows: list[tuple[str, str, str]],
) -> Path:
    """Write a synthetic journal at `path`. Rows are (operation, kind, params)."""
    import sqlite3  # noqa: PLC0415

    con = sqlite3.connect(str(path))
    try:
        con.execute(WRITE_OPS_DDL)
        for i, (operation, kind, params) in enumerate(rows):
            con.execute(
                'INSERT INTO write_ops (id, operation, kind, params, created_at)'
                ' VALUES (?, ?, ?, ?, ?)',
                (f'op-{i:06d}', operation, kind, params, '2026-08-12T00:00:00Z'),
            )
        con.commit()
    finally:
        con.close()
    return path


def _search_rows(text: str, n: int, *, limit: int = 5) -> list[tuple[str, str, str]]:
    import json  # noqa: PLC0415

    params = json.dumps({'query': text, 'limit': limit})
    return [('search', 'read', params)] * n


def _standard_journal(tmp_path: Path) -> Path:
    """A journal whose shares are hand-computable.

    200 search ops total:
      overview      60  -> 30%
      conventions   40  -> 20%
      decisions     20  -> 10%
      task {id}     40  -> 20%   (across 4 distinct task ids, 10 each)
      long tail     40  -> 20%   (40 distinct one-off queries)
    Plus 25 non-search ops that must be ignored entirely.
    """
    rows: list[tuple[str, str, str]] = []
    rows += _search_rows(OVERVIEW, 60)
    rows += _search_rows(CONVENTIONS, 40)
    rows += _search_rows(DECISIONS, 20)
    for task_id in ('4004', '3560', '3111', '3.1'):
        rows += _search_rows(TASK_TEMPLATE.format(task_id=task_id), 10)
    for i in range(40):
        rows += _search_rows(f'one off question number {i:02d}', 1)
    # Noise that must never be counted.
    rows += [('add_memory', 'write', '{"content": "not a query"}')] * 20
    rows += [('get_task', 'read', '{"task_id": "4004"}')] * 5
    return _build_journal(tmp_path / 'journal.db', rows)


class TestHarvestSelectsOnlySearchOps:
    """Only `operation='search'` rows carrying query text are counted."""

    def test_non_search_operations_are_ignored(self, tmp_path):
        mod = _mod()
        result = mod.harvest(_standard_journal(tmp_path))
        # 200 search ops, not 225.
        assert result.total_search_ops == 200

    def test_search_ops_without_query_text_are_excluded_from_the_denominator(
        self, tmp_path
    ):
        mod = _mod()
        rows = _search_rows(OVERVIEW, 10)
        rows += [('search', 'read', '{"limit": 5}')] * 7 # no `query` key
        rows += [('search', 'read', 'not json at all')] * 3
        db = _build_journal(tmp_path / 'j.db', rows)
        result = mod.harvest(db)
        assert result.total_search_ops == 10
        assert result.unparsed_search_ops == 10 # 7 keyless + 3 malformed

    def test_query_text_is_parsed_out_of_the_params_json(self, tmp_path):
        mod = _mod()
        db = _build_journal(tmp_path / 'j.db', _search_rows(OVERVIEW, 3))
        result = mod.harvest(db)
        assert [t.text for t in result.templates if t.observed_count] == [OVERVIEW]


class TestTemplateClassification:
    """The four briefing-assembler templates, three literal and one parameterized."""

    def test_the_three_literals_are_classified(self, tmp_path):
        mod = _mod()
        result = mod.harvest(_standard_journal(tmp_path))
        by_text = {t.text: t for t in result.templates}
        assert by_text[OVERVIEW].observed_count == 60
        assert by_text[CONVENTIONS].observed_count == 40
        assert by_text[DECISIONS].observed_count == 20
        for text in (OVERVIEW, CONVENTIONS, DECISIONS):
            assert by_text[text].match == 'literal'

    def test_the_task_family_is_matched_as_a_template_not_a_literal(self, tmp_path):
        mod = _mod()
        result = mod.harvest(_standard_journal(tmp_path))
        family = [t for t in result.templates if t.match == 'parameterized']
        assert len(family) == 1, 'exactly one parameterized family'
        fam = family[0]
        # All four distinct task ids collapse into ONE class.
        assert fam.observed_count == 40
        assert fam.distinct_instances == 4
        assert fam.template == TASK_TEMPLATE

    def test_a_parameterized_instance_is_not_counted_in_the_long_tail(self, tmp_path):
        mod = _mod()
        result = mod.harvest(_standard_journal(tmp_path))
        tail_texts = {r['text'] for r in result.rows if r['source'] == 'production_tail'}
        assert not any(t.startswith('task ') for t in tail_texts)

    def test_a_near_miss_does_not_join_the_family(self, tmp_path):
        mod = _mod()
        rows = _search_rows(TASK_TEMPLATE.format(task_id='4004'), 5)
        rows += _search_rows('task context and related decisions', 5) # no id
        rows += _search_rows('task 4004 context and related choices', 5) # wrong tail
        db = _build_journal(tmp_path / 'j.db', rows)
        result = mod.harvest(db)
        fam = next(t for t in result.templates if t.match == 'parameterized')
        assert fam.observed_count == 5
        assert result.tail_count == 10


class TestTrafficShares:
    """Each class's share, and the residual long tail, are reported."""

    def test_shares_are_reported_per_class(self, tmp_path):
        mod = _mod()
        result = mod.harvest(_standard_journal(tmp_path))
        by_text = {t.text: t.traffic_share for t in result.templates}
        assert by_text[OVERVIEW] == 0.30
        assert by_text[CONVENTIONS] == 0.20
        assert by_text[DECISIONS] == 0.10

    def test_the_three_literals_and_the_family_are_reported_separately(self, tmp_path):
        mod = _mod()
        result = mod.harvest(_standard_journal(tmp_path))
        assert result.literal_share == 0.60 # 120/200
        assert result.family_share == 0.80 # 160/200

    def test_the_residual_long_tail_share_is_reported(self, tmp_path):
        mod = _mod()
        result = mod.harvest(_standard_journal(tmp_path))
        assert result.tail_share == 0.20
        assert result.tail_distinct == 40
        assert result.tail_count == 40

    def test_the_shares_partition_the_traffic(self, tmp_path):
        mod = _mod()
        result = mod.harvest(_standard_journal(tmp_path))
        total = sum(t.traffic_share for t in result.templates) + result.tail_share
        assert abs(total - 1.0) < 1e-9

    def test_an_empty_journal_reports_no_share_rather_than_a_zero_share(self, tmp_path):
        """No traffic is no measurement — never a measured 0.0 share."""
        mod = _mod()
        db = _build_journal(tmp_path / 'j.db', [])
        result = mod.harvest(db)
        assert result.total_search_ops == 0
        assert result.tail_share is None
        assert all(t.traffic_share is None for t in result.templates)


class TestDeterministicTailSample:
    """The tail sample is regenerable: same DB + same args => same bytes."""

    def test_harvesting_twice_yields_identical_rows(self, tmp_path):
        mod = _mod()
        db = _standard_journal(tmp_path)
        first = mod.harvest(db, tail_sample=10, seed=4004)
        second = mod.harvest(db, tail_sample=10, seed=4004)
        assert first.rows == second.rows

    def test_the_frequency_led_portion_is_seed_independent(self, tmp_path):
        mod = _mod()
        rows = _search_rows(OVERVIEW, 10)
        # A tail with an unambiguous frequency order.
        for i in range(20):
            rows += _search_rows(f'tail query {i:02d}', 20 - i)
        db = _build_journal(tmp_path / 'j.db', rows)
        a = mod.harvest(db, tail_sample=8, tail_top=4, seed=1)
        b = mod.harvest(db, tail_sample=8, tail_top=4, seed=2)
        top_a = [r['text'] for r in a.rows if r.get('tail_rank') is not None][:4]
        top_b = [r['text'] for r in b.rows if r.get('tail_rank') is not None][:4]
        assert top_a == top_b == [f'tail query {i:02d}' for i in range(4)]

    def test_the_tail_sample_is_bounded_by_tail_sample(self, tmp_path):
        mod = _mod()
        db = _standard_journal(tmp_path)
        result = mod.harvest(db, tail_sample=7)
        tail_rows = [r for r in result.rows if r['source'] == 'production_tail']
        assert len(tail_rows) == 7

    def test_the_emitted_rows_are_sorted_deterministically(self, tmp_path):
        mod = _mod()
        db = _standard_journal(tmp_path)
        rows = mod.harvest(db, tail_sample=10).rows
        tail = [r['text'] for r in rows if r['source'] == 'production_tail']
        assert tail == sorted(tail)


class TestFixtureRowShape:
    """Production queries are UNLABELED by construction."""

    REQUIRED = ('query_id', 'text', 'source', 'observed_count', 'traffic_share')

    def test_every_row_carries_the_required_fields(self, tmp_path):
        mod = _mod()
        rows = mod.harvest(_standard_journal(tmp_path), tail_sample=5).rows
        assert rows
        for row in rows:
            for field in self.REQUIRED:
                assert field in row, f'{field} missing from {row}'

    def test_no_row_carries_expects_claim_ids(self, tmp_path):
        """A labeled column here would be fabricated ground truth."""
        mod = _mod()
        rows = mod.harvest(_standard_journal(tmp_path), tail_sample=5).rows
        for row in rows:
            assert 'expects_claim_ids' not in row
            assert 'expects_topic' not in row

    def test_query_ids_are_unique_and_stable(self, tmp_path):
        mod = _mod()
        db = _standard_journal(tmp_path)
        first = mod.harvest(db, tail_sample=5)
        second = mod.harvest(db, tail_sample=5)
        ids = [r['query_id'] for r in first.rows]
        assert len(ids) == len(set(ids))
        assert ids == [r['query_id'] for r in second.rows]

    def test_the_four_briefing_rows_are_sourced_as_templates(self, tmp_path):
        mod = _mod()
        rows = mod.harvest(_standard_journal(tmp_path), tail_sample=3).rows
        briefing = [r for r in rows if r['source'] == 'briefing_template']
        assert len(briefing) == 4

    def test_rows_record_the_limit_the_journal_actually_recorded(self, tmp_path):
        """`observed_limit` is a READING, not the briefing constant.

        In the standard journal every op happens to run at 5, so every row
        reports 5 — but it reports it because that is what was measured.
        """
        mod = _mod()
        rows = mod.harvest(_standard_journal(tmp_path), tail_sample=3).rows
        assert all(r['observed_limit'] == 5 for r in rows)
        assert all(r['observed_limits'] == {'5': r['observed_count']}
                   for r in rows if r['source'] == 'production_tail')

    def test_a_tail_row_is_not_stamped_with_the_briefing_limit(self, tmp_path):
        """The regression: briefing.py:1376 governs the briefing family ONLY.

        A tail query fired by some other caller at limit=20 must report 20.
        Stamping BRIEFING_SEARCH_LIMIT on it published a number nothing
        observed, under a field named `observed_limit`, into the artifact a
        selection gate reads.
        """
        mod = _mod()
        rows = _search_rows(OVERVIEW, 10)
        rows += _search_rows('a tail query some other caller fires', 7, limit=20)
        db = _build_journal(tmp_path / 'j.db', rows)
        harvested = mod.harvest(db, tail_sample=3).rows
        tail = [r for r in harvested if r['source'] == 'production_tail']
        assert len(tail) == 1
        assert tail[0]['observed_limit'] == 20
        assert tail[0]['observed_limit'] != mod.BRIEFING_SEARCH_LIMIT
        assert tail[0]['observed_limits'] == {'20': 7}

    def test_a_query_whose_instances_disagree_reports_no_single_limit(self, tmp_path):
        """Disagreement is None — never a modal pick, never a default.

        The full histogram rides alongside, so a reader who wants a modal
        value takes it from the measurement and owns that choice explicitly.
        """
        mod = _mod()
        rows = _search_rows(OVERVIEW, 10)
        rows += _search_rows('mixed limit tail query', 6, limit=10)
        rows += _search_rows('mixed limit tail query', 2, limit=50)
        db = _build_journal(tmp_path / 'j.db', rows)
        tail = [r for r in mod.harvest(db, tail_sample=3).rows
                if r['source'] == 'production_tail']
        assert len(tail) == 1
        assert tail[0]['observed_limit'] is None
        assert tail[0]['observed_limits'] == {'10': 6, '50': 2}

    def test_the_sidecar_reports_the_scored_limit_as_a_choice(self, tmp_path):
        """The scoring window is named a choice and sits beside the readings."""
        mod = _mod()
        rows = _search_rows(OVERVIEW, 10)
        rows += _search_rows('a tail query', 4, limit=30)
        db = _build_journal(tmp_path / 'j.db', rows)
        prov = mod.harvest(db, tail_sample=3).provenance()
        assert prov['scored_limit'] == mod.BRIEFING_SEARCH_LIMIT
        assert prov['scored_limit_is_a_choice'] is True
        assert 'CHOICE' in prov['scored_limit_basis']
        assert prov['briefing_observed_limits'] == {'5': 10}
        assert prov['tail_observed_limits'] == {'30': 4}

    def test_an_op_with_no_usable_limit_is_bucketed_not_defaulted(self, tmp_path):
        """A missing limit is `unspecified`, not silently the scoring window."""
        import json  # noqa: PLC0415

        mod = _mod()
        rows = _search_rows(OVERVIEW, 10)
        rows += [('search', 'read', json.dumps({'query': 'no limit recorded'}))] * 3
        db = _build_journal(tmp_path / 'j.db', rows)
        tail = [r for r in mod.harvest(db, tail_sample=3).rows
                if r['source'] == 'production_tail']
        assert tail[0]['observed_limits'] == {mod.UNSPECIFIED_LIMIT: 3}
        assert tail[0]['observed_limit'] is None


class TestPinnedTail:
    """A pinned harvest re-measures without re-drawing the query set.

    The journal is appended to by a running server, so an unpinned
    re-harvest draws different tail queries — every one a miss in the
    committed fetch cache, i.e. correcting one field would demand a paid
    re-seed. Pinning holds WHICH queries are emitted fixed while every
    count, share and limit is measured fresh.
    """

    def test_a_pin_holds_the_tail_query_set_fixed(self, tmp_path):
        mod = _mod()
        rows = _search_rows(OVERVIEW, 10)
        rows += _search_rows('pinned tail query', 4, limit=10)
        rows += _search_rows('newly arrived tail query', 9, limit=8)
        db = _build_journal(tmp_path / 'j.db', rows)
        pinned = mod.harvest(db, tail_sample=5, pin_tail_texts=['pinned tail query'])
        tail = [r for r in pinned.rows if r['source'] == 'production_tail']
        assert [r['text'] for r in tail] == ['pinned tail query']
        # ...and the retained row is still MEASURED, not copied forward.
        assert tail[0]['observed_limit'] == 10
        assert tail[0]['observed_count'] == 4

    def test_pinning_to_a_query_the_journal_lacks_raises(self, tmp_path):
        """Emitting a pinned row with no observations would fabricate it."""
        mod = _mod()
        db = _build_journal(tmp_path / 'j.db', _search_rows(OVERVIEW, 10))
        with pytest.raises(mod.EmptyHarvestError, match='pinned tail'):
            mod.harvest(db, pin_tail_texts=['a query nobody ever ran'])


class TestReadOnlyAccess:
    """The live journal is a 10 GB file a running server is writing to."""

    def test_the_connection_is_opened_read_only(self, tmp_path):
        mod = _mod()
        db = _standard_journal(tmp_path)
        con = mod._connect_readonly(db)
        try:
            assert con.execute('PRAGMA query_only').fetchone()[0] == 1
        finally:
            con.close()

    def test_a_write_attempt_raises(self, tmp_path):
        import sqlite3  # noqa: PLC0415

        import pytest  # noqa: PLC0415

        mod = _mod()
        db = _standard_journal(tmp_path)
        con = mod._connect_readonly(db)
        try:
            with pytest.raises(sqlite3.OperationalError):
                con.execute("INSERT INTO write_ops (id, created_at) VALUES ('x', 'y')")
        finally:
            con.close()

    def test_harvesting_does_not_modify_the_journal(self, tmp_path):
        import hashlib  # noqa: PLC0415

        mod = _mod()
        db = _standard_journal(tmp_path)
        before = hashlib.sha256(db.read_bytes()).hexdigest()
        mod.harvest(db)
        assert hashlib.sha256(db.read_bytes()).hexdigest() == before


class TestLoudDegradation:
    """An unreadable journal is a named error, never an empty sample."""

    def test_an_absent_journal_raises_a_named_error(self, tmp_path):
        import pytest  # noqa: PLC0415

        mod = _mod()
        missing = tmp_path / 'nope.db'
        with pytest.raises(mod.JournalUnavailableError) as exc:
            mod.harvest(missing)
        assert 'nope.db' in str(exc.value)

    def test_a_journal_without_write_ops_raises_rather_than_returning_empty(
        self, tmp_path
    ):
        import sqlite3  # noqa: PLC0415

        import pytest  # noqa: PLC0415

        mod = _mod()
        db = tmp_path / 'wrong.db'
        con = sqlite3.connect(str(db))
        con.execute('CREATE TABLE other (id TEXT)')
        con.commit()
        con.close()
        with pytest.raises(mod.JournalUnavailableError):
            mod.harvest(db)

    def test_no_fixture_is_written_when_the_journal_is_unavailable(self, tmp_path):
        import pytest  # noqa: PLC0415

        mod = _mod()
        out = tmp_path / 'production_query_sample.jsonl'
        with pytest.raises(mod.JournalUnavailableError):
            mod.main(['--journal', str(tmp_path / 'nope.db'), '--out', str(out)])
        assert not out.exists()

    def test_an_empty_journal_writes_no_fixture_either(self, tmp_path):
        """Zero traffic must not silently become an empty-but-valid fixture."""
        import pytest  # noqa: PLC0415

        mod = _mod()
        db = _build_journal(tmp_path / 'j.db', [])
        out = tmp_path / 'sample.jsonl'
        with pytest.raises(mod.EmptyHarvestError):
            mod.main(['--journal', str(db), '--out', str(out)])
        assert not out.exists()


class TestFixtureWrite:
    """The committed fixture is JSONL plus a provenance sidecar."""

    def test_main_writes_jsonl_rows_and_a_sidecar(self, tmp_path):
        import json  # noqa: PLC0415

        mod = _mod()
        db = _standard_journal(tmp_path)
        out = tmp_path / 'production_query_sample.jsonl'
        mod.main(['--journal', str(db), '--out', str(out), '--tail-sample', '5'])
        lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
        assert len(lines) == 9 # 4 briefing templates + 5 tail
        sidecar = out.with_suffix('.provenance.json')
        assert sidecar.exists()
        prov = json.loads(sidecar.read_text())
        assert prov['total_search_ops'] == 200
        assert prov['literal_share'] == 0.60
        assert prov['family_share'] == 0.80
        assert prov['tail_share'] == 0.20
        assert prov['tail_distinct'] == 40
        assert 'harvested_at' in prov
        # The scoring window is published as a CHOICE, beside the readings —
        # there is no bare `search_limit` a reader could mistake for one.
        assert 'search_limit' not in prov
        assert prov['scored_limit'] == 5
        assert prov['scored_limit_is_a_choice'] is True
        assert prov['briefing_observed_limits'] == {'5': 160}
        assert prov['tail_observed_limits'] == {'5': 40}

    def test_the_written_fixture_round_trips_through_the_reader(self, tmp_path):
        mod = _mod()
        db = _standard_journal(tmp_path)
        out = tmp_path / 'sample.jsonl'
        mod.main(['--journal', str(db), '--out', str(out), '--tail-sample', '5'])
        rows = mod.read_fixture(out)
        assert len(rows) == 9
        assert all('expects_claim_ids' not in r for r in rows)
