"""Tests for scripts/measure_plural_enum_guard_recall.py. (task 3949)

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in
test_cleanup_count_snapshots.py.

Every test here drives SYNTHETIC facts or a fake edge source. Nothing in
this file touches a live backend: the probe's whole point is that its
pure band (scan / triage / simulate / paginate / render) is checkable
without FalkorDB, so a broken probe fails in CI rather than silently
reporting a zero.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'measure_plural_enum_guard_recall.py'
)
SWEEP_TEST_PATH = (
    Path(__file__).parent / 'reconciliation' / 'test_stale_status_snapshot_edge_sweep.py'
)


def _load_module(mod_name: str, path: Path) -> types.ModuleType:
    """Load a module from its file path, bypassing sys.path entirely.

    The module is registered in sys.modules under *mod_name* so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).

    Used for BOTH the script under test (not on PYTHONPATH) and the sweep
    TEST module whose parametrize markers this file re-validates against:
    ``tests/`` carries no __init__.py, so the sweep suite's importable name
    depends on pytest's rootdir insertion and differs between a direct run
    and a collected one. Loading by path sidesteps that entirely — the
    re-validation gate must not be able to break on an import-path detail.
    """
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module('measure_plural_enum_guard_recall', SCRIPT_PATH)
enumerate_valid_edge_facts = _mod.enumerate_valid_edge_facts
run = _mod.run
exit_code = _mod.exit_code
build_parser = _mod._build_parser
render_json = _mod.render_json
render_markdown = _mod.render_markdown
scan_corpus = _mod.scan_corpus
triage_rejection = _mod.triage_rejection
simulate_candidate = _mod.simulate_candidate
extract_plural_ids = _mod.extract_plural_ids


def _pinned_shapes(test_name: str) -> list:
    """Pull a pinned shape list off the sweep suite's parametrize marker.

    Read from the MARKER rather than copied into this file on purpose. Task
    3949 requires each candidate tightening be re-validated 'against the
    full precision-guard parametrization'. A hardcoded copy satisfies that
    on the day it is written and silently stops covering the full set the
    moment someone adds a shape — the same silent-loss failure mode the
    sweep suite's own test_every_guarded_preposition_is_exercised_against_a
    _plural_head exists to prevent for _ENUM_PREP_WORDS. Reading the marker
    makes the gate mechanical: a shape added upstream automatically
    re-validates both candidates here.
    """
    import inspect

    sweep_tests = _load_module(
        '_sweep_tests_for_revalidation', SWEEP_TEST_PATH,
    )

    for _cls_name, cls in inspect.getmembers(sweep_tests, inspect.isclass):
        fn = getattr(cls, test_name, None)
        if fn is None:
            continue
        for mark in getattr(fn, 'pytestmark', []):
            if mark.name != 'parametrize':
                continue
            argnames, argvalues = mark.args[0], mark.args[1]
            # 'fact'-only markers yield bare strings; ('fact', 'expected')
            # markers yield tuples.
            if isinstance(argnames, str):
                return [(v, None) for v in argvalues]
            return list(argvalues)
    raise AssertionError(
        f'{test_name} not found in the sweep test module — the re-validation '
        f'gate has lost its corpus and would silently pass on nothing.'
    )


# Facts the shipped guard must keep suppressing. Any candidate that admits
# one of these has re-opened over-selection, which is the unrecoverable
# direction and disqualifies it outright.
_PRECISION_SHAPES = [f for f, _ in _pinned_shapes(
    'test_plural_enumeration_precision_guards')]
_SUPPRESSION_SHAPES = [f for f, _ in _pinned_shapes(
    'test_guard_rejected_enumeration_suppresses_ids_on_every_path')]
# Facts the shipped guard extracts, and that no candidate may disturb.
_POSITIVE_SHAPES = _pinned_shapes(
    'test_plural_enumeration_subject_positives_survive_precision_guards')
# The documented recall loss — what a tightening would be FOR.
_PREAMBLE_SHAPES = [f for f, _ in _pinned_shapes(
    'test_adverbial_preamble_is_a_documented_under_selection')]

_ALL_GUARDED_SHAPES = _PRECISION_SHAPES + _SUPPRESSION_SHAPES + _PREAMBLE_SHAPES

_DATE_STAMP_PREAMBLE = 'As of 2026-08-09, tasks 1020 and 1030 are pending.'
_INTRA_CLAUSE_COMMA = (
    'Blockers for down-stream, still-unmerged tasks 1020 and 1030 are pending.'
)


# The five synthetic facts below are the probe's own positive control: the
# headline live result is a ZERO, and a zero produced by broken wiring is
# indistinguishable from a zero produced by a clean corpus. Pinning one fact
# per outcome class means a probe that has silently stopped matching
# anything fails HERE rather than publishing a meaningless report.
_SUBJECT_POSITIVE = 'Tasks 1020 and 1030 are pending.'
_COMPLEMENT_REJECTION = 'Reviews of tasks 1020 and 1030 are pending.'
_PREAMBLE_REJECTION = 'As of 2026-08-09, tasks 1020 and 1030 are pending.'
# Carries the lexical precondition (`tasks <digits>`) but does NOT match
# PLURAL_ENUM_SNAPSHOT_RE — 'related to' is no status marker. This class is
# what separates 'the corpus has no plural-enum shapes at all' from 'the
# corpus has them and the guard eats them', so it gets its own counter.
_LEXICAL_NEAR_MISS = 'Tasks 1752 and 1753 are related to the uptime feed.'
_UNRELATED = 'The merge worker restarted after the fleet redeploy.'

_SYNTHETIC_CORPUS = [
    _SUBJECT_POSITIVE,
    _COMPLEMENT_REJECTION,
    _PREAMBLE_REJECTION,
    _LEXICAL_NEAR_MISS,
    _UNRELATED,
]


def test_scan_corpus_counts_matches_and_guard_rejections():
    """Every count field is asserted independently, not just the headline.

    A single aggregate assertion would pass while two counters were wrong
    in opposite directions. The fields are also deliberately not required
    to sum: a fact with two matches, one rejected and one surviving,
    counts toward BOTH guard_rejected and selected (see the probe's
    docstring for the counting rule).
    """
    result = scan_corpus(_SYNTHETIC_CORPUS)

    assert result.facts_scanned == 5
    assert result.lexical_precondition == 4  # every fact but _UNRELATED
    assert result.regex_matched == 3  # near-miss fails the regex itself
    assert result.guard_rejected == 2  # complement + preamble
    assert result.selected == 1  # the subject-position positive only


def test_scan_corpus_rejections_are_triageable_records():
    """A nonzero future run has to be diagnosable without a re-run.

    'Zero matches' is a point-in-time fact about a corpus that grows every
    cycle. When it stops being zero, whoever reads the report needs the
    offending fact text and the match offset in the artifact itself —
    otherwise the report says only that recall was lost, not where.
    """
    result = scan_corpus(_SYNTHETIC_CORPUS)

    assert len(result.rejections) == 2
    facts = [r.fact for r in result.rejections]
    assert facts == [_COMPLEMENT_REJECTION, _PREAMBLE_REJECTION]

    # The offsets are the matched span's start, so the fact can be split at
    # the exact prefix the guard was handed.
    by_fact = {r.fact: r.match_start for r in result.rejections}
    assert by_fact[_COMPLEMENT_REJECTION] == _COMPLEMENT_REJECTION.index('tasks 1020')
    assert by_fact[_PREAMBLE_REJECTION] == _PREAMBLE_REJECTION.index('tasks 1020')


@pytest.mark.parametrize(
    'fact',
    [
        # The three shapes pinned by the sweep suite's
        # test_adverbial_preamble_is_a_documented_under_selection. These are
        # the RECALL LOSS: the enumeration really is its copula's subject and
        # the preposition belongs to a scene-setting preamble in front of it.
        'As of 2026-08-09, tasks 1020 and 1030 are pending.',
        'In the merge queue, tasks 1020 and 1030 are pending.',
        'For this cycle, tasks 1020 and 1030 are pending.',
    ],
)
def test_triage_rejection_labels_adverbial_preamble(fact):
    """Rejections that cost recall must be separable from correct ones.

    Task 3949 asks for a hand-classification of guard rejections into
    genuine complements (correct) and adverbial-preamble subjects (recall
    loss). Mechanizing it here means a future nonzero run is triaged by the
    committed probe rather than by whoever happens to read the artifact.
    """
    match_start = fact.index('tasks 1020')
    assert triage_rejection(fact, match_start) == 'adverbial_preamble'


@pytest.mark.parametrize(
    'fact',
    [
        # A representative spread of the pinned precision shapes: in each,
        # the copula's real subject is an OUTER HEAD NOUN and the marker
        # describes that, not the tasks. Rejecting these is correct.
        'Reviews of tasks 1020, 1030, and 1031 are pending.',  # plural outer head
        'Statuses of the tasks 1020 and 1030 are pending.',  # determiner
        'Notes on remaining tasks 1020 and 1030 are pending.',  # open-class gap
        # LOAD-BEARING. This one contains a comma but is NOT a preamble —
        # the comma is intra-clause, inside a compound modifier. It is the
        # exact shape candidate (b) was measured to get wrong, so a triage
        # that keyed naively on 'is there a comma' would mislabel it as
        # recall loss and thereby manufacture the very evidence that would
        # wrongly justify tightening.
        'Blockers for down-stream, still-unmerged tasks 1020 and 1030 are pending.',
    ],
)
def test_triage_rejection_labels_prepositional_complement(fact):
    """Correct rejections must not be reported as recall loss.

    Over-reporting recall loss is the failure that matters here: it is what
    would wrongly justify a tightening, and a tightening trades toward the
    unrecoverable over-selection direction.
    """
    match_start = fact.index('tasks 1020')
    assert triage_rejection(fact, match_start) == 'prepositional_complement'


def test_pinned_shape_corpus_is_non_trivial():
    """The re-validation gate must fail loudly if its corpus evaporates.

    Marker introspection is the whole mechanism here. If the upstream test
    names or marker shapes change, every candidate assertion below would
    pass vacuously against an empty list — a gate that silently stops
    gating. These floors are the tripwire; they are minima, not exact
    counts, precisely so ADDING a shape upstream never breaks this file.
    """
    assert len(_PRECISION_SHAPES) >= 43
    assert len(_SUPPRESSION_SHAPES) >= 4
    assert len(_POSITIVE_SHAPES) >= 11
    assert len(_PREAMBLE_SHAPES) >= 3
    assert _DATE_STAMP_PREAMBLE in _PREAMBLE_SHAPES
    assert _INTRA_CLAUSE_COMMA in _PRECISION_SHAPES


def test_shipped_guard_baseline_over_selects_nothing():
    """The baseline that makes the candidate deltas meaningful.

    Simulating the shipped guard against itself must be a no-op. If this
    ever reports a difference, the simulator disagrees with the shipped
    extraction path and every candidate number below is measuring the
    simulator's bug rather than the candidate.
    """
    result = simulate_candidate('shipped', _ALL_GUARDED_SHAPES)

    assert result.over_selected == []
    assert result.recovered == []
    assert result.unchanged == _ALL_GUARDED_SHAPES


def test_candidate_a_re_opens_nothing_but_misses_the_motivating_shape():
    """Candidate (a): plural/capitalized head required before the preposition.

    It clears the precision bar — no pinned over-selection re-opens — but
    it does NOT recover 'As of <date>, ...', which is the finding's OWN
    motivating shape, because 'of' there is preceded by the capitalized
    sentence-initial 'As'. A tightening that cannot fix the case that
    motivated it buys very little.
    """
    result = simulate_candidate('a', _ALL_GUARDED_SHAPES)

    assert result.over_selected == []
    assert _DATE_STAMP_PREAMBLE not in result.recovered
    assert set(result.recovered) == set(_PREAMBLE_SHAPES) - {_DATE_STAMP_PREAMBLE}


def test_candidate_b_recovers_every_preamble_but_re_opens_an_over_selection():
    """Candidate (b): restart the backward scan after a preposition-free comma.

    It recovers all three preamble shapes — and re-opens a pinned
    over-selection, 'Blockers for down-stream, still-unmerged tasks ...',
    where the comma is intra-clause rather than a preamble boundary. That
    failure IS the disqualification: task 3949 requires each candidate be
    re-validated against the full precision parametrization before
    shipping, and this one does not pass it.
    """
    result = simulate_candidate('b', _ALL_GUARDED_SHAPES)

    assert result.recovered == _PREAMBLE_SHAPES
    assert result.over_selected == [_INTRA_CLAUSE_COMMA]


@pytest.mark.parametrize('candidate', ['shipped', 'a', 'b'])
def test_subject_position_positives_extract_unchanged_under_both_candidates(
    candidate,
):
    """Neither candidate may cost a currently-extracted snapshot.

    The candidates are evaluated on what they RE-OPEN and what they
    RECOVER; this pins the third direction, that neither silently drops an
    id the shipped guard already yields. Asserting the exact expected id
    sets (not merely 'unchanged') means a candidate that yields a
    DIFFERENT id set for the same fact is caught too.
    """
    for fact, expected in _POSITIVE_SHAPES:
        assert extract_plural_ids(fact, candidate=candidate) == expected, fact


# ---------------------------------------------------------------------------
# Edge enumeration must survive FalkorDB's server-wide result-set cap
# ---------------------------------------------------------------------------

_PAGE_SIZE = 4


class _FakeCappedEdgeQuery:
    """A query_fn that truncates every response exactly as the server does.

    FalkorDB's server-wide RESULTSET_SIZE is 10000 and nothing in this repo
    overrides it, so an unpaginated query is SILENTLY cut at that many rows
    — no error, no warning. Measured on the live dark_factory graph at
    planning time: get_all_valid_edges' exact query has 24902 rows and
    returns 10000, exposing 6376 of 12488 distinct valid edges (51%).

    This fake reproduces that hazard at test scale by setting its cap equal
    to the page size, which is the worst case: a full page is
    indistinguishable from a truncated one by row count alone, so the
    enumerator cannot cheat by noticing a suspiciously round number.
    """

    def __init__(self, rows: list[list], cap: int) -> None:
        self.rows = rows
        self.cap = cap
        self.cyphers: list[str] = []

    async def __call__(self, cypher: str) -> list[list]:
        import re as _re

        self.cyphers.append(cypher)
        window = _re.search(r'SKIP\s+(\d+)\s+LIMIT\s+(\d+)', cypher)
        if window:
            skip, limit = int(window.group(1)), int(window.group(2))
            page = self.rows[skip: skip + limit]
        else:
            page = list(self.rows)
        return page[: self.cap]  # server-side RESULTSET_SIZE truncation


def _fake_corpus() -> tuple[list[list], int]:
    """10 distinct edges spread over 13 rows — N > 2 * page_size, with repeats.

    The repeated uuids model the undirected MATCH's double-attribution, which
    get_all_valid_edges' own docstring documents: a directed A->B edge matches
    from both endpoints, so the same edge uuid legitimately arrives twice.
    """
    rows: list[list] = [[f'edge-{i:02d}', f'Fact number {i}.'] for i in range(10)]
    rows[3][1] = None  # a NULL fact must survive as '', not crash or vanish
    # double-attribution repeats, interleaved so they land on different pages
    rows = rows[:2] + [rows[0]] + rows[2:6] + [rows[4]] + rows[6:] + [rows[9]]
    return rows, 10


@pytest.mark.asyncio
async def test_edge_enumeration_paginates_past_the_resultset_cap():
    """All N distinct edges must be enumerated, not just the first page.

    A probe that reuses the unpaginated production query would measure a
    51% sample and report it as the whole corpus. Since this task's headline
    result is a ZERO, a truncated zero would be worthless — it would say
    'no plural-enum facts found' when half the corpus was never looked at.
    """
    rows, distinct = _fake_corpus()
    query_fn = _FakeCappedEdgeQuery(rows, cap=_PAGE_SIZE)

    facts, complete = await enumerate_valid_edge_facts(
        query_fn, page_size=_PAGE_SIZE,
    )

    assert complete is True
    assert len(facts) == distinct
    assert set(facts) == {f'edge-{i:02d}' for i in range(10)}
    # fact text is preserved verbatim...
    assert facts['edge-07'] == 'Fact number 7.'
    # ...and a NULL fact becomes '' rather than None
    assert facts['edge-03'] == ''
    # pagination actually happened: more than one page was requested
    assert len(query_fn.cyphers) > 1
    assert all('SKIP' in c and 'LIMIT' in c for c in query_fn.cyphers)


@pytest.mark.asyncio
async def test_unpaginated_query_against_the_same_fake_is_truncated():
    """The anti-regression, asserted directly rather than assumed.

    This is what the paginated enumerator exists to defeat. If someone
    'simplifies' enumerate_valid_edge_facts back to a single query, the
    test above goes red — and this one documents exactly what that
    simplification would silently cost.
    """
    rows, distinct = _fake_corpus()
    query_fn = _FakeCappedEdgeQuery(rows, cap=_PAGE_SIZE)

    single_page = await query_fn(
        'MATCH (n:Entity)-[e:RELATES_TO]-() WHERE e.invalid_at IS NULL '
        'RETURN DISTINCT e.uuid, e.fact'
    )

    assert len(single_page) == _PAGE_SIZE
    assert len({row[0] for row in single_page}) < distinct


@pytest.mark.asyncio
async def test_edge_enumeration_of_empty_graph_is_complete_and_empty():
    """Zero edges is a valid, COMPLETE result — knowlive held exactly that."""
    query_fn = _FakeCappedEdgeQuery([], cap=_PAGE_SIZE)

    facts, complete = await enumerate_valid_edge_facts(
        query_fn, page_size=_PAGE_SIZE,
    )

    assert facts == {}
    assert complete is True


# ---------------------------------------------------------------------------
# `complete` must be EARNED, not asserted
# ---------------------------------------------------------------------------
#
# The defect these tests pin, reproduced first-hand against the shipped code:
# ``if len(rows) < page_size: break`` mistakes the SERVER's ceiling for
# end-of-data. With a cap of 10 and page_size=20 over a 50-edge corpus the
# first page returns 10 rows, the break fires, and the function returns
# 10 of 50 edges with ``complete=True`` — after which run() publishes a
# fifth-of-the-corpus zero as a COMPLETE measurement.
#
# The band was untested because every pre-existing enumeration test uses
# cap == page_size, which is precisely the BENIGN case: there a short page
# really is end-of-data, so the bug cannot show. And the fail-closed rule at
# run() level was only ever asserted against _FakeEdgeSource's hand-supplied
# flag, never against a flag real code had to compute.


def _fake_rows(count: int) -> list[list]:
    """*count* distinct edge rows, one row per uuid.

    Deliberately repeat-free (unlike ``_fake_corpus``, which models the
    undirected MATCH's double-attribution): here a shortfall can only ever
    come from truncation, never from dedup, so an assertion on ``len(facts)``
    is unambiguous.
    """
    return [[f'edge-{i:03d}', f'Fact number {i}.'] for i in range(count)]


@pytest.mark.asyncio
async def test_enumeration_fails_closed_when_page_size_exceeds_the_server_cap():
    """A page_size above the server cap makes the short-page break a LIE.

    This is the exact repro: cap=10, page_size=20, 50 edges. The shipped code
    returns 10 of 50 with complete=True. Fail-closed is the only acceptable
    answer — an under-enumerated corpus is a FAILURE, not a smaller report,
    because this task's headline is a zero and a truncated zero is worthless.
    """
    query_fn = _FakeCappedEdgeQuery(_fake_rows(50), cap=10)

    facts, complete = await enumerate_valid_edge_facts(
        query_fn, page_size=20, resultset_size=10,
    )

    assert complete is False, (
        'a page_size at or above the server cap cannot yield a provably '
        'complete enumeration'
    )
    assert facts == {}, (
        'a PARTIAL dict invites the caller to use it anyway; an unsound '
        'paging configuration must yield nothing at all'
    )
    assert query_fn.cyphers == [], (
        'the configuration is unsound before any row is read — refuse up '
        'front rather than after doing the work'
    )


@pytest.mark.asyncio
async def test_enumeration_fails_closed_at_the_module_default_cap():
    """The boundary is `>=`, not `>`, deliberately.

    page_size == RESULTSET_SIZE is arithmetically safe on a server configured
    at exactly the constant we assume: a full page would genuinely be a full
    page. But RESULTSET_SIZE is an ASSUMPTION about server configuration —
    nothing in this repo sets it — so equality leaves ZERO margin, and a
    server configured one row lower silently re-opens the truncation.
    Refusing at the boundary costs one page of throughput and buys the margin.
    """
    query_fn = _FakeCappedEdgeQuery(_fake_rows(50), cap=10)

    facts, complete = await enumerate_valid_edge_facts(
        query_fn, page_size=_mod.RESULTSET_SIZE,
    )

    assert _mod.RESULTSET_SIZE == 10000, "FalkorDB's server-wide default"
    assert complete is False
    assert facts == {}
    assert query_fn.cyphers == []


@pytest.mark.asyncio
async def test_enumeration_fails_closed_when_the_page_cap_is_exhausted():
    """A bounded loop must report the bound it hit, not pretend it finished.

    Mirrors census_foreign_nodes' MAX_CENSUS_PAGES precedent
    (scripts/migrate_cross_graph_leak.py:161, :302): bound the loop so a
    pathological corpus cannot spin the probe forever, and WARN + report
    incomplete when the cap is hit while the last page was still full.
    No silent caps.
    """
    query_fn = _FakeCappedEdgeQuery(_fake_rows(50), cap=_PAGE_SIZE)

    facts, complete = await enumerate_valid_edge_facts(
        query_fn, page_size=_PAGE_SIZE, max_pages=3,
    )

    page_cyphers = [c for c in query_fn.cyphers if 'SKIP' in c]
    assert len(page_cyphers) == 3, 'the loop must TERMINATE at the cap'
    assert complete is False, (
        'hitting the page cap on a still-full page is a suspected shortfall, '
        'not a successful enumeration'
    )
    assert len(facts) < 50


@pytest.mark.asyncio
async def test_benign_pagination_is_still_complete():
    """The regression guard: the fix must not degenerate to 'always False'.

    cap == page_size, both well under RESULTSET_SIZE, over a corpus needing
    many pages — the ordinary case, and the one the live run depends on. All
    50 edges, complete True.
    """
    query_fn = _FakeCappedEdgeQuery(_fake_rows(50), cap=_PAGE_SIZE)

    facts, complete = await enumerate_valid_edge_facts(
        query_fn, page_size=_PAGE_SIZE,
    )

    assert complete is True
    assert len(facts) == 50


@pytest.mark.asyncio
async def test_paged_cypher_requests_a_stable_order():
    """DISTINCT + SKIP/LIMIT with no ORDER BY has no cross-query boundary.

    Every page is a SEPARATE query. Absent an explicit total order the store
    is free to return rows in a different order per query, so `SKIP n` on
    page 2 can skip rows page 1 never returned (silently dropped) or
    re-return rows it did (duplicated — masked here by the uuid dedup, but
    the drop is silent and permanent). Ordering on e.uuid makes the page
    boundary mean the same thing in every query.
    """
    query_fn = _FakeCappedEdgeQuery(_fake_rows(20), cap=_PAGE_SIZE)

    await enumerate_valid_edge_facts(query_fn, page_size=_PAGE_SIZE)

    page_cyphers = [c for c in query_fn.cyphers if 'SKIP' in c]
    assert page_cyphers, 'no paged query was issued at all'
    for cypher in page_cyphers:
        assert 'ORDER BY' in cypher, cypher
        assert cypher.index('ORDER BY') < cypher.index('SKIP'), cypher


def test_build_parser_rejects_a_page_size_at_or_above_the_server_cap():
    """The operator gets a clear CLI error, not a fail-closed exit later.

    The in-function check stays regardless — it is the one that is testable
    and that protects direct callers — but an argparse error names the
    problem at the moment the operator makes it.
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--page-size', str(_mod.RESULTSET_SIZE)])
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--page-size', str(_mod.RESULTSET_SIZE + 1)])

    ok = build_parser().parse_args(['--page-size', '5000'])
    assert ok.page_size == 5000


# ---------------------------------------------------------------------------
# run(): aggregation and the fail-closed rule
# ---------------------------------------------------------------------------


class _FakeEdgeSource:
    """The ONLY corpus source these tests use — no live backend is reachable.

    Records the project ids it was asked for, so a probe that silently
    reached past the injected seam to a real graph would be caught by the
    call log rather than by a mysteriously large number.
    """

    def __init__(self, corpora: dict[str, tuple[list[str], bool]]) -> None:
        self.corpora = corpora
        self.asked: list[str] = []

    async def __call__(self, project_id: str, *, page_size: int):
        self.asked.append(project_id)
        facts, complete = self.corpora[project_id]
        return {f'{project_id}-edge-{i}': f for i, f in enumerate(facts)}, complete


_ALPHA_FACTS = [
    'Tasks 1020 and 1030 are pending.',  # selected
    'Reviews of tasks 1020 and 1030 are pending.',  # guard-rejected
    'The merge worker restarted.',  # nothing
]
_BETA_FACTS = [
    'As of 2026-08-09, tasks 1020 and 1030 are pending.',  # guard-rejected
    'Tasks 1752 and 1753 are related to the uptime feed.',  # lexical near-miss
]


def _args(**overrides):
    parsed = build_parser().parse_args([])
    for key, value in overrides.items():
        setattr(parsed, key, value)
    return parsed


@pytest.mark.asyncio
async def test_run_aggregates_per_project_counts_into_the_totals():
    """Totals must be the sum of the parts, not an independently-derived number.

    The report is committed and cited; a totals row that can drift from its
    own per-project rows is a report that can lie without anything failing.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'beta': (_BETA_FACTS, True),
    })

    report = await run(
        _args(project_id=['alpha', 'beta']), edge_source=source,
    )

    assert source.asked == ['alpha', 'beta']
    assert [p.project_id for p in report.projects] == ['alpha', 'beta']

    by_id = {p.project_id: p for p in report.projects}
    assert by_id['alpha'].valid_edges == 3
    assert by_id['alpha'].scan.selected == 1
    assert by_id['alpha'].scan.guard_rejected == 1
    assert by_id['beta'].valid_edges == 2
    assert by_id['beta'].scan.guard_rejected == 1
    assert by_id['beta'].scan.lexical_precondition == 2

    for attr in (
        'facts_scanned', 'lexical_precondition',
        'regex_matched', 'guard_rejected', 'selected',
    ):
        assert getattr(report.totals, attr) == sum(
            getattr(p.scan, attr) for p in report.projects
        ), attr
    assert report.total_valid_edges == 5


@pytest.mark.asyncio
async def test_run_carries_candidate_simulations_and_a_verdict():
    """The report must stand alone: simulations and verdict live IN it."""
    source = _FakeEdgeSource({'alpha': (_ALPHA_FACTS, True)})

    report = await run(_args(project_id=['alpha']), edge_source=source)

    assert [c.name for c in report.candidates] == list(_mod.CANDIDATE_NAMES)
    assert report.verdict
    assert report.complete is True
    assert exit_code(report) == 0


@pytest.mark.asyncio
async def test_run_fails_closed_when_any_project_enumeration_is_incomplete():
    """An under-enumerated corpus is a FAILURE, not a smaller report.

    This task's headline result is a zero. A zero measured over part of the
    corpus is worthless and, worse, indistinguishable from a real one — so
    the incompleteness has to reach the exit code, not just a log line.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'beta': (_BETA_FACTS, False),  # truncated enumeration
    })

    report = await run(
        _args(project_id=['alpha', 'beta']), edge_source=source,
    )

    assert report.complete is False
    by_id = {p.project_id: p for p in report.projects}
    assert by_id['alpha'].complete is True
    assert by_id['beta'].complete is False
    assert exit_code(report) != 0


@pytest.mark.asyncio
async def test_run_triages_every_rejection_it_reports():
    """Rejection counts and their triage labels must agree.

    A report whose triage breakdown does not add up to its rejection count
    would understate recall loss silently — the one direction that matters,
    since it is the number a future reader would use to decide whether to
    tighten.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'beta': (_BETA_FACTS, True),
    })

    report = await run(
        _args(project_id=['alpha', 'beta']), edge_source=source,
    )

    by_id = {p.project_id: p for p in report.projects}
    # 'Reviews of tasks ...' is a correct rejection
    assert by_id['alpha'].triage == {'prepositional_complement': 1}
    # 'As of <date>, tasks ...' is the documented recall loss
    assert by_id['beta'].triage == {'adverbial_preamble': 1}
    assert sum(report.triage_totals.values()) == report.totals.guard_rejected


# ---------------------------------------------------------------------------
# Artifact rendering
# ---------------------------------------------------------------------------

_FIXED_TIMESTAMP = '2026-08-17T00:00:00+00:00'


async def _fixed_report():
    """A report built from a fixed corpus and a fixed, INJECTED timestamp.

    Project ids are deliberately supplied out of alphabetical order so the
    renderers' explicit sorting is actually exercised rather than
    accidentally satisfied by insertion order.
    """
    source = _FakeEdgeSource({
        'zeta': (_BETA_FACTS, True),
        'alpha': (_ALPHA_FACTS, True),
    })
    return await run(
        _args(project_id=['zeta', 'alpha'], measured_at=_FIXED_TIMESTAMP),
        edge_source=source,
    )


@pytest.mark.asyncio
async def test_report_rendering_is_deterministic():
    """Identical reports must render byte-identically.

    These artifacts are COMMITTED, so any nondeterminism — dict iteration
    order, set ordering, a clock read inside the renderer — shows up
    forever as meaningless diff churn that hides the one line that
    actually changed.
    """
    first = await _fixed_report()
    second = await _fixed_report()

    assert render_json(first) == render_json(second)
    assert render_markdown(first) == render_markdown(second)


@pytest.mark.asyncio
async def test_rendered_sections_are_explicitly_sorted():
    """Ordering must come from an explicit sort, not from insertion order."""
    report = await _fixed_report()

    payload = json.loads(render_json(report))
    assert [p['project_id'] for p in payload['projects']] == ['alpha', 'zeta']

    markdown = render_markdown(report)
    assert markdown.index('`alpha`') < markdown.index('`zeta`')


@pytest.mark.asyncio
async def test_rendered_json_carries_the_measurement_and_the_verdict():
    """The JSON must stand alone as the citable record."""
    report = await _fixed_report()
    payload = json.loads(render_json(report))

    assert payload['schema_version'] == _mod.SCHEMA_VERSION
    assert payload['measured_at'] == _FIXED_TIMESTAMP
    assert payload['verdict'] == report.verdict
    assert payload['complete'] is True
    assert payload['totals']['guard_rejected'] == report.totals.guard_rejected
    assert payload['total_valid_edges'] == report.total_valid_edges

    by_id = {p['project_id']: p for p in payload['projects']}
    assert by_id['alpha']['valid_edges'] == 3
    assert by_id['alpha']['scan']['selected'] == 1
    assert by_id['zeta']['triage'] == {'adverbial_preamble': 1}

    assert [c['name'] for c in payload['candidates']] == list(_mod.CANDIDATE_NAMES)
    assert payload['revalidation_test'] == _mod.REVALIDATION_TEST


@pytest.mark.asyncio
async def test_rendered_markdown_opens_with_provenance_and_regenerate_command():
    """A report nobody can regenerate is a transcript with better formatting.

    'Zero matches' is a point-in-time fact about a growing corpus, so the
    exact command that reproduces it has to travel WITH the numbers.
    """
    report = await _fixed_report()
    markdown = render_markdown(report)

    head = markdown[:1200]
    assert 'measure_plural_enum_guard_recall.py' in head
    assert 'uv run python scripts/measure_plural_enum_guard_recall.py' in head
    assert _FIXED_TIMESTAMP in head

    # the sections the verdict rests on
    assert 'VERDICT' in markdown
    assert report.verdict in markdown
    assert 'RESULTSET_SIZE' in markdown  # the enumeration-coverage note
    assert _mod.REVALIDATION_TEST in markdown


@pytest.mark.asyncio
async def test_renderers_never_read_the_clock():
    """Determinism has to be structural, not merely observed.

    The two renders in test_report_rendering_is_deterministic run
    milliseconds apart, so a renderer reading a coarse clock could pass
    that test by luck. Rendering a report whose timestamp was injected as a
    sentinel proves the value came from the DATA.
    """
    report = await _fixed_report()
    sentinel = 'SENTINEL-NOT-A-TIMESTAMP'
    stamped = dataclasses.replace(report, measured_at=sentinel)

    assert sentinel in render_markdown(stamped)
    assert json.loads(render_json(stamped))['measured_at'] == sentinel


def test_scan_corpus_of_empty_corpus_is_all_zeroes():
    """An empty corpus must be a clean zero, not a crash or a None.

    The live run enumerates several project graphs and at least one of them
    (knowlive) held zero valid edges at planning time, so this is a real
    input shape rather than a defensive one.
    """
    result = scan_corpus([])

    assert result.facts_scanned == 0
    assert result.lexical_precondition == 0
    assert result.regex_matched == 0
    assert result.guard_rejected == 0
    assert result.selected == 0
    assert result.rejections == []
