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

# Shared with tests/reconciliation/test_stale_status_snapshot_edge_sweep.py,
# which parametrizes its guard tests off these same lists. ``tests/`` carries
# no __init__.py while ``tests/reconciliation/`` does, so pytest puts
# ``tests/`` on sys.path for modules in both directories and this package
# import resolves identically from either.
from reconciliation.plural_enum_shapes import (
    ADVERBIAL_PREAMBLE_SHAPES,
    GUARD_REJECTED_SUPPRESSION_SHAPES,
    PRECISION_GUARD_SHAPES,
    SUBJECT_POSITIVE_SHAPES,
)

# The PRODUCTION guard object, imported the same way the sweep suite imports
# it. Held here only so the baseline test can assert the probe's 'shipped'
# candidate IS this object rather than a drifted copy — an identity check that
# has to span the module boundary to mean anything.
from fused_memory.reconciliation.stale_status_snapshot_edge_sweep import (
    _enumeration_is_prepositional_complement,
)

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'measure_plural_enum_guard_recall.py'
)


def _load_module(mod_name: str, path: Path) -> types.ModuleType:
    """Load a module from its file path, bypassing sys.path entirely.

    The module is registered in sys.modules under *mod_name* so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).

    Used ONLY for the script under test, which lives in ``scripts/`` and is
    not on PYTHONPATH. The pinned shape corpora are an ordinary package
    import (see below) and need no such machinery.
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
_CANDIDATE_GUARDS = _mod._CANDIDATE_GUARDS


# The SAME lists the sweep suite parametrizes its guard tests off — imported,
# never copied. Task 3949 requires each candidate tightening be re-validated
# 'against the full precision-guard parametrization', and a second hardcoded
# copy would satisfy that on the day it was written and silently stop covering
# the full set the moment someone added a shape. Sharing the data module keeps
# the gate mechanical: a shape appended in reconciliation/plural_enum_shapes.py
# re-validates both candidates here automatically, with no dependency on the
# other suite's test-function names or marker internals.
#
# Facts the shipped guard must keep suppressing. Any candidate that admits one
# of these has re-opened over-selection, which is the unrecoverable direction
# and disqualifies it outright.
_PRECISION_SHAPES = PRECISION_GUARD_SHAPES
_SUPPRESSION_SHAPES = GUARD_REJECTED_SUPPRESSION_SHAPES
# Facts the shipped guard extracts, and that no candidate may disturb.
_POSITIVE_SHAPES = SUBJECT_POSITIVE_SHAPES
# The documented recall loss — what a tightening would be FOR.
_PREAMBLE_SHAPES = ADVERBIAL_PREAMBLE_SHAPES

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


def test_one_fact_can_count_as_both_rejected_and_selected():
    """The counting rule the report's whole column semantics rest on.

    The module header and the test above both state it — 'a fact with two
    matches, one rejected and one surviving, counts toward BOTH
    guard_rejected and selected, so the columns are not required to sum' —
    and the rendered markdown repeats it to a reader deciding whether the
    numbers add up. Nothing exercised it: every fact in _SYNTHETIC_CORPUS
    carries exactly one match, and _TWO_REJECTIONS_IN_ONE_FACT has both of
    its matches rejected, so `if len(fact_rejections) < len(matches)` was
    unpinned in the one direction that makes the rule visible.

    The fact below is the minimal witness: 'Reviews of ...' is a correct
    complement rejection and the second sentence is a subject-position
    selection, in ONE fact. regex_matched is 1 because that counter is
    per-fact; guard_rejected and selected are both 1 for the same fact,
    which is exactly the non-summing the rule describes.
    """
    both = (
        'Reviews of tasks 1020 and 1030 are pending. '
        'Tasks 2040 and 2050 are pending.'
    )

    result = scan_corpus([both])

    assert result.facts_scanned == 1
    assert result.regex_matched == 1, 'per-FACT, however many matches it holds'
    assert result.guard_rejected == 1
    assert result.selected == 1
    # ...and the rejection is still recorded per MATCH, with its offset
    assert [r.match_start for r in result.rejections] == [both.index('tasks 1020')]


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


def _first_enumeration_start(fact: str) -> int:
    """Where the shipped regex's first match begins in *fact*.

    Derived from PLURAL_ENUM_SNAPSHOT_RE rather than from ``fact.index('tasks
    1020')`` so a shape appended to the shared corpus with different ids — or
    a differently-spelled plural head — is still located. ``triage_rejection``
    is defined against the offset the extractor hands it, so taking that
    offset from the extractor's own pattern is also the faithful input.
    """
    match = _mod.PLURAL_ENUM_SNAPSHOT_RE.search(fact)
    assert match is not None, f'shape does not reach the guard at all: {fact}'
    return match.start()


# Parametrized over the SHARED corpus, not over a copy of it. A fourth preamble
# shape appended to reconciliation/plural_enum_shapes.py is picked up by the
# sweep suite and by the candidate simulation automatically; hardcoding the
# three here would leave triage — the thing that decides whether a future
# nonzero run reads as recall loss or as a correct rejection — the one consumer
# that silently stopped covering the full set. A mislabel there UNDERSTATES
# recall loss with nothing failing.
@pytest.mark.parametrize('fact', _PREAMBLE_SHAPES)
def test_triage_rejection_labels_adverbial_preamble(fact):
    """Rejections that cost recall must be separable from correct ones.

    Task 3949 asks for a hand-classification of guard rejections into
    genuine complements (correct) and adverbial-preamble subjects (recall
    loss). Mechanizing it here means a future nonzero run is triaged by the
    committed probe rather than by whoever happens to read the artifact.
    """
    assert triage_rejection(fact, _first_enumeration_start(fact)) == 'adverbial_preamble'


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
        # The two shapes below reach the two comma-tail branches, which no
        # shape above did — both are ways a comma can appear WITHOUT opening
        # a preamble, and both must stay on the fail-safe label.
        # A listed preposition AFTER the last comma governs the enumeration
        # directly, whatever the preamble in front of it was doing.
        'As noted, reviews of tasks 1020 and 1030 are pending.',
        # A plural head noun between the comma and the enumeration could be
        # what the copula agrees with, so the enumeration is not
        # unambiguously the subject — the case that branch's own comment
        # describes, and it survives a genuine 'As of <date>,' preamble.
        'As of 2026-08-09, blockers tasks 1020 and 1030 are pending.',
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


def test_triage_falls_back_when_it_cannot_see_what_the_guard_saw():
    """The scope-disagreement branch — the heuristic's last fail-safe.

    ``triage_rejection`` reuses ``_last_clause_break`` so it cannot normally
    disagree with the guard about where the clause starts: the guard fired,
    therefore a listed preposition is somewhere in that span, therefore
    ``first_prep`` is found. The branch exists for the case where that
    reasoning stops holding — a caller triaging a match the guard did NOT
    reject, or a future edit moving one of the two scopes out from under the
    other. Both are silent failures, and the branch's job is to answer
    conservatively rather than to claim recall loss it cannot substantiate.

    Driven directly, since it is by construction unreachable through a real
    rejection: the fact below carries a comma and no listed preposition at
    all, so the guard SELECTS it (asserted, or the test would be triaging a
    rejection after all) while triage still returns the fail-safe label.
    """
    fact = 'Reviewed today, tasks 1020 and 1030 are pending.'
    start = _first_enumeration_start(fact)

    assert extract_plural_ids(fact) == {1020, 1030}, 'the guard does NOT reject it'
    assert triage_rejection(fact, start) == 'prepositional_complement'


def test_the_shared_corpora_and_the_two_deciding_shapes_are_present():
    """Everything the candidate verdicts below silently depend on.

    Not a size floor — the old size floors were a meta-test guarding the
    marker-introspection mechanism, and that mechanism is gone. But an
    earlier docstring here over-claimed what replacing it bought: that a
    missing corpus would be 'an ImportError or an obviously-failing
    assertion rather than a silent vacuous pass'. The ImportError half
    holds; the EMPTY half does not. An emptied list makes
    ``@pytest.mark.parametrize`` report a skip and makes a bare ``for``
    loop pass green, so emptying any of the four upstream lists would
    quietly stop several tests here from testing anything. Hence the
    explicit non-empty pins: they are the cheapest thing that turns that
    silent vacuity into a failure, in ONE place covering all four lists.

    The two named shapes are the narrower, behavioural half. Candidate (a)
    is rejected specifically because it cannot recover
    _DATE_STAMP_PREAMBLE, and candidate (b) specifically because it
    re-opens _INTRA_CLAUSE_COMMA. Both verdicts are stated as set
    differences against the shared corpus, so if either shape were dropped
    upstream those assertions would keep passing while silently no longer
    testing the thing they are named for.
    """
    assert _PRECISION_SHAPES
    assert _SUPPRESSION_SHAPES
    assert _POSITIVE_SHAPES
    assert _PREAMBLE_SHAPES

    assert _DATE_STAMP_PREAMBLE in _PREAMBLE_SHAPES
    assert _INTRA_CLAUSE_COMMA in _PRECISION_SHAPES


def test_the_shipped_baseline_is_a_precondition_not_a_measurement():
    """What actually makes the candidate deltas meaningful — falsifiably.

    An earlier spelling of this test asserted only that simulating the
    shipped guard against itself is a no-op, with a docstring claiming that
    a difference would prove 'the simulator disagrees with the shipped
    extraction path'. It could not: ``simulate_candidate`` reaches its
    ``guard(prefix)`` check only after ``_enumeration_is_prepositional_
    complement(prefix)`` returned True on the SAME prefix, and for
    ``'shipped'`` those two calls are the same function object, so the
    no-op holds for any input and any behaviour of the guard. Verified by
    execution: substituting a maximally broken all-reject guard left the
    result byte-identical. A test that cannot fail underwrites nothing —
    the same silent-vacuity class that produced two blocking findings on
    this task.

    The three assertions below are ordered precondition-first, and the two
    that can fail come first:

    1. IDENTITY (falsifiable, and spans the module boundary): the
       simulator's 'shipped' guard is the PRODUCTION guard object, not a
       copy that has drifted. Re-spelling the guard inside the probe is the
       exact failure the script's import block exists to prevent, and it
       would fail here rather than surface as a quietly wrong number.
    2. CORPUS (falsifiable): every shape the candidates are scored against
       really is one the shipped guard rejects. This is what makes
       ``recovered``/``over_selected`` interpretable at all — a shape the
       shipped guard already selects can never appear in either list, so a
       corpus that silently drifted into selected shapes would report a
       reassuring pair of zeroes.
    3. The no-op baseline itself, stated as what it is: a CONSEQUENCE of
       (1) and (2), not an independent measurement.
    """
    assert _CANDIDATE_GUARDS['shipped'] is _enumeration_is_prepositional_complement

    assert _ALL_GUARDED_SHAPES, 'the guarded corpus emptied upstream'
    for fact in _ALL_GUARDED_SHAPES:
        assert extract_plural_ids(fact) == set(), fact

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

    The guard on the loop is load-bearing: this body's only assertion is
    INSIDE the loop, so an emptied _POSITIVE_SHAPES upstream would leave
    this test green while checking nothing at all.
    """
    assert _POSITIVE_SHAPES, 'the subject-position corpus emptied upstream'
    for fact, expected in _POSITIVE_SHAPES:
        assert extract_plural_ids(fact, candidate=candidate) == expected, fact


# ---------------------------------------------------------------------------
# Edge enumeration must survive FalkorDB's server-wide result-set cap
# ---------------------------------------------------------------------------

_PAGE_SIZE = 4

# Sentinel: derive the census count from the fake's own rows. A plain None
# default would be indistinguishable from 'the probe answered None', which
# is one of the fail-closed cases under test.
_DERIVE_COUNT = object()


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

    It also answers the census ``count(DISTINCT e.uuid)`` probe, because the
    enumerator cross-checks what it enumerated against it. That answer is
    DERIVED from the same rows the paged query serves — never supplied
    independently — so the fake cannot accidentally agree with a broken
    enumerator. It is returned UNCAPPED, modelling the real property that
    makes the probe a proof: a single-row result cannot be truncated by a
    row cap. ``count_rows`` overrides it to model a probe that answers
    nothing.
    """

    def __init__(
        self, rows: list[list], cap: int, *, count_rows: object = _DERIVE_COUNT,
    ) -> None:
        self.rows = rows
        self.cap = cap
        self.count_rows = count_rows
        self.cyphers: list[str] = []

    async def __call__(self, cypher: str) -> list[list]:
        import re as _re

        self.cyphers.append(cypher)
        if 'count(' in cypher:
            if self.count_rows is not _DERIVE_COUNT:
                return self.count_rows  # type: ignore[return-value]
            return [[len({row[0] for row in self.rows})]]
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
    page_cyphers = [c for c in query_fn.cyphers if 'count(' not in c]
    assert len(page_cyphers) > 1
    assert all('SKIP' in c and 'LIMIT' in c for c in page_cyphers)


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
# The EMPIRICAL completeness proof
# ---------------------------------------------------------------------------
#
# The structural checks above still only reason about a constant we guessed.
# RESULTSET_SIZE is an ASSUMPTION about server configuration, while the
# defect class is 'a short page was mistaken for end-of-data' — so any check
# that reasons only from that constant inherits the guess.
#
# A `RETURN count(DISTINCT e.uuid)` probe returns exactly ONE row, so it can
# never itself be truncated by the cap it is measuring. That is what makes
# `len(facts) == expected` a PROOF rather than one more heuristic, and it
# catches causes nobody enumerated in advance: a server configured below the
# assumed constant, unstable DISTINCT+SKIP/LIMIT page boundaries, a dropped
# page.


@pytest.mark.asyncio
async def test_enumeration_fails_closed_when_the_server_caps_below_the_assumed_constant():
    """The residual hazard the constant check structurally CANNOT reach.

    Here the fake caps at 10 rows, page_size is 20, and ``resultset_size`` is
    left at its DEFAULT 10000 — so `20 < 10000` and the structural check
    PROVABLY passes. The first page comes back with 10 rows, the short-page
    break fires, and without a cross-check the function reports 10 of 50
    edges as a complete enumeration: the identical silent truncation,
    undetected.

    This test is what proves the census cross-check is load-bearing rather
    than redundant with the constant check.
    """
    query_fn = _FakeCappedEdgeQuery(_fake_rows(50), cap=10)

    facts, complete = await enumerate_valid_edge_facts(query_fn, page_size=20)

    assert complete is False, (
        'enumerated 10 against a census-reported 50 — a count mismatch is a '
        'shortfall however it was caused'
    )
    assert len(facts) < 50


@pytest.mark.asyncio
async def test_enumeration_is_complete_when_the_enumerated_count_matches_the_census():
    """The proof must be able to say YES.

    The other half of the regression guard: a cross-check that always fails
    closed is just a broken enumerator with better manners, and would make
    the live run — and therefore this task's whole deliverable —
    unpublishable.
    """
    query_fn = _FakeCappedEdgeQuery(_fake_rows(50), cap=_PAGE_SIZE)

    facts, complete = await enumerate_valid_edge_facts(
        query_fn, page_size=_PAGE_SIZE,
    )

    assert complete is True
    assert len(facts) == 50


@pytest.mark.asyncio
async def test_count_probe_brackets_the_enumeration_and_is_never_paged():
    """Unpaged is what makes the probe a proof; twice is what dates it.

    Adding SKIP/LIMIT to it would subject the proof to the very truncation it
    exists to detect. Issuing it PER PAGE would scale its cost with the corpus
    for no extra information — and, worse, invite a mid-enumeration answer to
    be treated as the total.

    Exactly two, bracketing the paging, is the minimum that can date the
    proof. These graphs are written live, so one probe cannot tell a
    truncated enumeration from one that raced a concurrent write; a second
    probe after the last page can (see the mid-enumeration test below).

    Both must count the SAME population the paged query enumerates: the same
    MATCH and WHERE, and DISTINCT on e.uuid so the undirected match's
    double-attribution collapses exactly as the paged dict-dedup collapses
    it. Two numbers derived from different populations are not comparable,
    and comparing them would manufacture a mismatch on every run.
    """
    query_fn = _FakeCappedEdgeQuery(_fake_rows(20), cap=_PAGE_SIZE)

    await enumerate_valid_edge_facts(query_fn, page_size=_PAGE_SIZE)

    count_cyphers = [c for c in query_fn.cyphers if 'count(' in c]
    assert len(count_cyphers) == 2, query_fn.cyphers
    # ...and they BRACKET the paging rather than both landing at one end: a
    # post-probe issued before the last page would date nothing.
    assert 'count(' in query_fn.cyphers[0], query_fn.cyphers
    assert 'count(' in query_fn.cyphers[-1], query_fn.cyphers
    assert all('SKIP' in c for c in query_fn.cyphers[1:-1]), query_fn.cyphers
    for probe in count_cyphers:
        assert 'SKIP' not in probe, probe
        assert 'LIMIT' not in probe, probe
        assert 'DISTINCT e.uuid' in probe, probe
        assert '(n:Entity)-[e:RELATES_TO]-()' in probe, probe
        assert 'e.invalid_at IS NULL' in probe, probe


class _FakeMovingCorpusQuery(_FakeCappedEdgeQuery):
    """A LIVE corpus: a write lands between the two census probes.

    Not a defensive hypothetical. The graphs this probe measures are the
    orchestrator's and the reconciler's working memory, being added to and
    invalidated continuously, and a full run pages ~30k edges across several
    queries — so a count that moves under the enumeration is the ordinary
    case, not the exotic one.
    """

    def __init__(self, rows: list[list], cap: int, *, counts: list[int]) -> None:
        super().__init__(rows, cap)
        self.counts = list(counts)

    async def __call__(self, cypher: str) -> list[list]:
        if 'count(' in cypher:
            self.cyphers.append(cypher)
            return [[self.counts.pop(0)]]
        return await super().__call__(cypher)


@pytest.mark.asyncio
async def test_a_corpus_that_moves_mid_enumeration_is_reported_as_a_race(caplog):
    """A concurrent write must not be diagnosed as a server misconfiguration.

    With one census probe, ANY write landing during the run makes
    ``len(facts) != expected`` and the operator is told the most likely cause
    is 'a server result-set cap below the assumed 10000'. On a busy graph that
    is the LEAST likely cause, and it sends them to configuration they do not
    need to change — while the actual remedy is simply to re-run.

    Fail-closed either way (an enumeration that raced a write is not a proven
    one); what this pins is that the two are reported as DIFFERENT things.
    """
    query_fn = _FakeMovingCorpusQuery(
        _fake_rows(50), cap=_PAGE_SIZE, counts=[50, 51],
    )

    with caplog.at_level('WARNING'):
        facts, complete = await enumerate_valid_edge_facts(
            query_fn, page_size=_PAGE_SIZE,
        )

    # The enumeration itself was fine — 50 fetched against a pre-count of 50.
    assert len(facts) == 50
    assert complete is False, 'a raced enumeration is not a proven one'

    warnings = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'CHANGED MID-ENUMERATION' in warnings
    assert 're-run' in warnings.lower()
    assert 'result-set cap' not in warnings, (
        'a concurrent write must not be reported as a suspected truncation'
    )


@pytest.mark.asyncio
async def test_a_stable_count_that_disagrees_is_still_a_suspected_shortfall(caplog):
    """The other side of the same fork — the race check must not swallow it.

    Only a count that held STILL across the whole run makes a shortfall
    evidence of truncation. This is the regression guard that stops the
    mid-enumeration branch above from becoming the answer to every mismatch.
    """
    query_fn = _FakeCappedEdgeQuery(_fake_rows(50), cap=10)

    with caplog.at_level('WARNING'):
        facts, complete = await enumerate_valid_edge_facts(
            query_fn, page_size=20,
        )

    assert complete is False
    assert len(facts) < 50

    warnings = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'result-set cap' in warnings
    assert 'CHANGED MID-ENUMERATION' not in warnings


@pytest.mark.parametrize(
    'count_rows',
    [
        pytest.param([], id='empty-result-set'),
        pytest.param(None, id='null-result-set'),
        pytest.param([[None]], id='null-count'),
        pytest.param([[]], id='row-with-no-columns'),
        # _census_count's docstring promises 'a non-integer' collapses to
        # None like every other flavour of missing evidence. It is the one
        # shape reached through the try/except rather than an explicit
        # check, so it is also the one that would start RAISING — turning a
        # fail-closed None into a crashed run — if the except clause were
        # ever narrowed.
        pytest.param([['abc']], id='non-integer-count'),
    ],
)
@pytest.mark.asyncio
async def test_enumeration_fails_closed_when_the_count_probe_returns_nothing(
    count_rows,
):
    """A missing proof is not a passing proof.

    When the census probe answers nothing there is no evidence either way,
    and failing OPEN would publish an unverified `complete: true` under the
    very mechanism whose purpose is to make that claim earned. Note the
    enumeration itself SUCCEEDS in every case here — all 50 edges are
    fetched — so nothing but the missing proof is driving the False.
    """
    query_fn = _FakeCappedEdgeQuery(
        _fake_rows(50), cap=_PAGE_SIZE, count_rows=count_rows,
    )

    facts, complete = await enumerate_valid_edge_facts(
        query_fn, page_size=_PAGE_SIZE,
    )

    assert complete is False
    assert len(facts) == 50, 'the paging worked; only the proof was unavailable'


@pytest.mark.asyncio
async def test_a_census_that_answers_then_stops_answering_fails_closed(caplog):
    """The pre/post asymmetry — half a proof is not a proof.

    The census is probed twice, and the SECOND probe is what tells a
    truncation apart from a raced write. Every case above loses both
    probes together (``_FakeCappedEdgeQuery`` applies one ``count_rows``
    to each), so the branch for 'the pre-count answered and the post-count
    did not' was unreachable and unpinned — and it is the asymmetric case
    that a real store produces, since the two probes are separated by the
    whole paging run.

    Without the post-count there is no evidence either way: a shortfall
    cannot be distinguished from a write landing mid-run, and the operator
    must not be told a cause. Fail-closed, and say only what is known.
    """
    query_fn = _FakeMovingCorpusQuery(
        _fake_rows(50), cap=_PAGE_SIZE, counts=[50, None],
    )

    with caplog.at_level('WARNING'):
        facts, complete = await enumerate_valid_edge_facts(
            query_fn, page_size=_PAGE_SIZE,
        )

    assert len(facts) == 50, 'the paging itself was fine'
    assert complete is False, 'an unavailable proof is not a passing one'

    warnings = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'no usable count' in warnings
    # ...and NEITHER diagnosis is offered, because neither is known
    assert 'CHANGED MID-ENUMERATION' not in warnings
    assert 'result-set cap' not in warnings


@pytest.mark.asyncio
async def test_a_row_with_a_null_uuid_is_skipped_not_counted():
    """A row that cannot be keyed is not an edge this run enumerated.

    The page loop skips a NULL uuid, and that skip is load-bearing for the
    census cross-check rather than cosmetic: keying facts by uuid is how
    the undirected MATCH's double-attribution is deduped, so a null-keyed
    row has no identity to dedupe on. Counting it would inflate
    ``len(facts)`` toward the census total and could make a genuinely
    short enumeration read as complete — the fail-OPEN direction. Skipping
    it instead leaves the count short, which the census then catches.
    """
    rows = _fake_rows(4)
    rows[2] = [None, 'Tasks 1020 and 1030 are pending.']
    query_fn = _FakeCappedEdgeQuery(rows, cap=_PAGE_SIZE, count_rows=[[3]])

    facts, complete = await enumerate_valid_edge_facts(
        query_fn, page_size=_PAGE_SIZE,
    )

    assert len(facts) == 3, 'the null-uuid row is not an enumerated edge'
    assert None not in facts
    assert complete is True, 'and the census agrees 3 is the whole corpus'


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
        # RECORDED, not just accepted. An earlier version discarded
        # page_size, so run() passing the wrong one — or none at all —
        # failed no test, while a page_size at or above the server's
        # result-set cap is the exact condition that makes the enumerator's
        # short-page proof a lie. The seam has to carry the argument, not
        # merely tolerate it.
        self.page_sizes: list[int] = []

    async def __call__(self, project_id: str, *, page_size: int):
        self.asked.append(project_id)
        self.page_sizes.append(page_size)
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
async def test_run_passes_the_requested_page_size_through_to_every_graph():
    """--page-size has to REACH the enumerator, for every graph.

    It is the flag the whole completeness argument turns on: the
    enumerator's short-page break is only sound while page_size stays below
    the server's result-set cap, and _page_size_arg validates that at the
    CLI. All of which is worthless if run() then drops the value or passes
    it to only the first graph — a silent fail-OPEN that no assertion on the
    report could catch, since the report records the page size run() was
    ASKED for, not the one the enumerator was HANDED.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'beta': (_BETA_FACTS, True),
    })

    report = await run(
        # deliberately not DEFAULT_PAGE_SIZE, so a hardcoded default fails
        _args(project_id=['alpha', 'beta'], page_size=37), edge_source=source,
    )

    assert source.page_sizes == [37, 37], 'every graph, not just the first'
    assert report.page_size == 37, 'and the artifact records what was used'


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

    The reconciliation is against ``len(totals.rejections)``, NOT
    ``totals.guard_rejected``: triage labels one entry per rejected MATCH,
    while guard_rejected counts one per rejected FACT (the probe's documented
    counting rule — a fact with two rejected enumerations adds 2 and 1
    respectively). Comparing the triage sum against guard_rejected mixes those
    units and holds only for corpora with at most one rejected match per fact.
    The third project below is exactly such a fact, so this test cannot go back
    to passing vacuously on that coincidence.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'beta': (_BETA_FACTS, True),
        # Two rejected matches in ONE fact — the shape that separates the two
        # units and falsifies the per-fact spelling of the invariant below.
        'gamma': ([_TWO_REJECTIONS_IN_ONE_FACT], True),
    })

    report = await run(
        _args(project_id=['alpha', 'beta', 'gamma']), edge_source=source,
    )

    by_id = {p.project_id: p for p in report.projects}
    # 'Reviews of tasks ...' is a correct rejection
    assert by_id['alpha'].triage == {'prepositional_complement': 1}
    # 'As of <date>, tasks ...' is the documented recall loss
    assert by_id['beta'].triage == {'adverbial_preamble': 1}
    # One FACT, but TWO rejected matches, so two triage labels.
    assert by_id['gamma'].triage == {'adverbial_preamble': 2}
    assert by_id['gamma'].scan.guard_rejected == 1

    assert sum(report.triage_totals.values()) == len(report.totals.rejections)
    # ...and the two units really are distinct here, so the assertion above is
    # doing work rather than comparing a number to itself.
    assert len(report.totals.rejections) != report.totals.guard_rejected


# A single fact carrying TWO guard-rejected enumerations, so a fact that is fed
# to the simulation once per rejected MATCH (rather than once per fact) scores
# every one of its matches twice.
_TWO_REJECTIONS_IN_ONE_FACT = (
    'As of 2026-08-09, tasks 1020 and 1030 are pending. '
    'In the queue, tasks 2040 and 2050 are pending.'
)


@pytest.mark.asyncio
async def test_candidate_simulation_scores_each_match_once_per_fact():
    """A multi-rejection fact must not be simulated once per rejection.

    ``totals.rejections`` holds one entry per rejected MATCH while
    ``simulate_candidate`` itself re-scans every match of every fact handed to
    it, so feeding it the raw rejection list enters an N-rejection fact N times
    and scores each of its matches N times — an N-fold inflation of exactly the
    ``recovered`` / ``over_selected`` pair the committed report tells a future
    reader to re-decide the tightening against.

    The live corpus currently has zero matches, which masks this entirely; it
    would first surface on precisely the re-run this script exists to enable.
    """
    source = _FakeEdgeSource({'alpha': ([_TWO_REJECTIONS_IN_ONE_FACT], True)})

    report = await run(_args(project_id=['alpha']), edge_source=source)

    # Both enumerations are rejected, and both are the documented recall loss.
    assert len(report.totals.rejections) == 2
    assert report.triage_totals == {'adverbial_preamble': 2}

    by_name = {c.name: c for c in report.candidates}
    # Candidate 'b' admits both matches: 2 recovered, one per MATCH — not 4.
    assert len(by_name['b'].recovered) == 2
    # Candidate 'a' admits only the 'As of ...' shape: 1, not 2.
    assert len(by_name['a'].recovered) == 1
    # Neither re-opens an over-selection, and the shipped baseline changes
    # nothing — one unchanged FACT, not one per rejection.
    for name, candidate in by_name.items():
        assert candidate.over_selected == [], name
        assert len(candidate.unchanged) <= 1, name
    assert len(by_name['shipped'].unchanged) == 1


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


# ---------------------------------------------------------------------------
# Graph-set completeness
# ---------------------------------------------------------------------------
#
# The enumerator can prove it read every ROW of a graph. Nothing proved the
# SET of graphs was the whole store — a project graph that existed in FalkorDB
# but was absent from the hardcoded DEFAULT_PROJECT_IDS was silently excluded
# while the artifact still said `complete: true` and the verdict still read
# 'No fact in any enumerated project graph'.
#
# Measured 2026-08-17, that was 2829 valid edges (~9% of the corpus) across
# autopilot_video, know_live, my_solar_challenge, pump_web_ui, solar_challenge
# and a test leftover. Of every shortfall class this probe guards against, the
# un-enumerated GRAPH was the one with no detection at all.


def _lister(*names: str):
    async def graph_lister() -> list[str]:
        return list(names)
    return graph_lister


@pytest.mark.asyncio
async def test_the_measured_graph_set_is_discovered_not_hardcoded():
    """With a lister and no --project-id, every populated graph is measured."""
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'beta': (_BETA_FACTS, True),
        'gamma': ([], True),
    })

    report = await run(
        _args(project_id=None),
        edge_source=source,
        # deliberately unsorted, and deliberately NOT DEFAULT_PROJECT_IDS
        graph_lister=_lister('gamma', 'alpha', 'beta'),
    )

    assert sorted(source.asked) == ['alpha', 'beta', 'gamma']
    assert report.project_ids_source == 'discovered'
    assert report.unmeasured_graphs == []
    assert report.complete is True
    assert exit_code(report) == 0


@pytest.mark.asyncio
async def test_a_populated_graph_left_unmeasured_fails_closed(caplog):
    """The shortfall class that previously had NO detection at all.

    A populated graph the run did not read is a coverage shortfall, not a
    scoping preference: the verdict quantifies over 'any project graph'. It
    gets the same fail-closed treatment as a row-count shortfall — incomplete,
    non-zero exit, and named in the artifact rather than merely absent from it.
    """
    source = _FakeEdgeSource({'alpha': (_ALPHA_FACTS, True)})

    with caplog.at_level('WARNING'):
        report = await run(
            _args(project_id=['alpha']),
            edge_source=source,
            graph_lister=_lister('alpha', 'omega', 'sigma'),
        )

    assert source.asked == ['alpha'], 'the narrowing must still be honoured'
    assert report.project_ids_source == 'cli'
    assert report.unmeasured_graphs == ['omega', 'sigma']
    assert report.complete is False, 'un-enumerated graphs are a shortfall'
    assert exit_code(report) != 0
    # ...every project it DID measure was complete, so nothing else explains it
    assert all(p.complete for p in report.projects)

    warnings = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'COVERAGE SHORTFALL' in warnings
    assert 'omega' in warnings and 'sigma' in warnings

    # and the shortfall is IN the artifacts, not just in a log nobody kept
    payload = json.loads(render_json(report))
    assert payload['unmeasured_graphs'] == ['omega', 'sigma']
    assert payload['project_ids_source'] == 'cli'
    markdown = render_markdown(report)
    assert 'COVERAGE SHORTFALL' in markdown
    assert '`omega`' in markdown and '`sigma`' in markdown


@pytest.mark.asyncio
async def test_a_narrowing_that_leaves_nothing_out_is_still_complete():
    """The other side of the shortfall rule, and the doc claim it corrects.

    ``run``'s docstring and ``--project-id``'s help both used to say a
    narrowing marks the report incomplete FULL STOP. That is wrong in both
    directions and the rendered coverage note already said so correctly:
    incompleteness here is EVIDENCE-based, derived by comparing the lister's
    inventory against what was read.

    So a ``--project-id`` naming every populated graph left nothing out and
    is genuinely complete — pinned here — while one that skips a populated
    graph is not (test_a_populated_graph_left_unmeasured_fails_closed).
    Without a lister there is no inventory at all, so the shortfall is
    UNKNOWABLE rather than zero, which is what ``project_ids_source`` and
    the coverage note record instead (pinned by
    test_without_a_lister_the_report_says_its_graph_set_was_unchecked).

    Together those three pin the conditional the docs now state, so the
    claim is checkable rather than prose.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'beta': (_BETA_FACTS, True),
    })

    report = await run(
        # a narrowing by FORM — but it names the whole populated store
        _args(project_id=['alpha', 'beta']),
        edge_source=source,
        graph_lister=_lister('alpha', 'beta'),
    )

    assert report.project_ids_source == 'cli'
    assert report.unmeasured_graphs == [], 'nothing was left out'
    assert report.complete is True
    assert exit_code(report) == 0


@pytest.mark.asyncio
async def test_an_empty_store_fails_closed_instead_of_verdicting_over_nothing(
    caplog,
):
    """The same shortfall class as above, one level up: an EMPTY measured set.

    `unmeasured_graphs` catches a graph the store reports and the run skipped.
    A store that reports NO graphs has nothing to skip, so that guard reads
    clean: `unmeasured_graphs == []`, `projects == []`, `all([]) is True`. The
    report would then be `complete`, exit 0, and carry the strongest verdict in
    the file — 'no fact in ANY project graph matches ... provably zero measured
    benefit' — derived from zero facts, and `_write_artifacts` would overwrite
    the committed report with it. An empty or wrong store (a fresh
    `docker-compose up -d`, a `--config` aimed elsewhere) must be a shortfall,
    not a complete measurement of nothing.
    """
    source = _FakeEdgeSource({})

    with caplog.at_level('WARNING'):
        report = await run(
            _args(project_id=None),
            edge_source=source,
            graph_lister=_lister(),
        )

    assert source.asked == [], 'there was nothing to enumerate'
    assert report.projects == []
    assert report.project_ids_source == 'discovered'
    # the older guard genuinely has nothing to say here — that is the point
    assert report.unmeasured_graphs == []
    assert report.complete is False, 'measuring zero graphs is a shortfall'
    assert exit_code(report) != 0

    warnings = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'COVERAGE SHORTFALL' in warnings
    assert 'ZERO graphs' in warnings

    # and it is legible in the artifacts, which are what a reader actually sees
    payload = json.loads(render_json(report))
    assert payload['complete'] is False
    markdown = render_markdown(report)
    assert 'COVERAGE SHORTFALL' in markdown
    assert 'ZERO graphs' in markdown
    assert '- Enumeration complete: `False`' in markdown


@pytest.mark.asyncio
async def test_one_graphs_failure_is_a_shortfall_not_a_lost_run(caplog):
    """A graph that fails to enumerate must not discard the graphs that did.

    Reachable without any bug in this file, and cheaply: ``list_graphs()``
    returns every non-``*_db`` FalkorDB key, and the committed artifact's
    discovered set includes dozens of EPHEMERAL pytest graphs
    (``probe_e1_gw*``, ``sweep_e4_gw*``, ``_test_*_``). A concurrent pytest
    run dropping one of those keys between the listing and the enumeration
    query raises mid-loop. Letting that propagate killed a 40-graph run at
    graph #39 and wrote NO artifact — contradicting the rule ``exit_code``'s
    own docstring states, that 'the artifact still lands — the evidence of
    the shortfall is IN it — but the exit code refuses to call an
    under-enumerated measurement a successful one'.

    So a failure is recorded as exactly what it is: one INCOMPLETE project,
    named in the artifact, forcing the whole report incomplete and a
    non-zero exit — the identical fail-closed treatment a row-count
    shortfall and an unmeasured graph already get. The measurement of every
    other graph survives, which is the difference between a shortfall and a
    lost run.

    ``_FakeEdgeSource`` raises KeyError for a graph it has no corpus for,
    which IS the vanished-key case rather than a stand-in for it.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'gamma': (_BETA_FACTS, True),
    })

    with caplog.at_level('WARNING'):
        report = await run(
            _args(project_id=None),
            edge_source=source,
            # 'beta' vanished between the listing and the query
            graph_lister=_lister('alpha', 'beta', 'gamma'),
        )

    assert source.asked == ['alpha', 'beta', 'gamma'], 'the loop kept going'
    by_id = {p.project_id: p for p in report.projects}
    assert set(by_id) == {'alpha', 'beta', 'gamma'}, 'the failure is RECORDED'
    assert by_id['beta'].complete is False
    assert by_id['beta'].valid_edges == 0
    assert by_id['beta'].scan.facts_scanned == 0

    # the graphs that DID enumerate are still measured — not discarded
    assert by_id['alpha'].complete is True
    assert by_id['alpha'].valid_edges == len(_ALPHA_FACTS)
    assert by_id['gamma'].valid_edges == len(_BETA_FACTS)
    assert report.totals.selected == (
        by_id['alpha'].scan.selected + by_id['gamma'].scan.selected
    )

    # ...and the run is still fail-closed about what it could not read
    assert report.unmeasured_graphs == [], 'nothing was SKIPPED; one FAILED'
    assert report.complete is False
    assert exit_code(report) != 0

    warnings = '\n'.join(r.getMessage() for r in caplog.records)
    assert 'COVERAGE SHORTFALL' in warnings
    assert 'beta' in warnings

    # the artifact LANDS, carrying the evidence — the whole point
    payload = json.loads(render_json(report))
    assert payload['complete'] is False
    assert [p['project_id'] for p in payload['projects'] if not p['complete']] == [
        'beta',
    ]
    assert '| `beta` | 0 |' in render_markdown(report)


@pytest.mark.asyncio
async def test_without_a_lister_the_report_says_its_graph_set_was_unchecked():
    """No inventory to compare against is not the same as a verified set.

    ``graph_lister=None`` is every test and any future caller with no store
    behind it. Falling back to the built-in list is fine; implying the set was
    cross-checked would not be, so the artifact records `fallback`.
    """
    source = _FakeEdgeSource(
        {p: ([], True) for p in _mod.DEFAULT_PROJECT_IDS},
    )

    report = await run(_args(project_id=None), edge_source=source)

    assert source.asked == list(_mod.DEFAULT_PROJECT_IDS)
    assert report.project_ids_source == 'fallback'
    assert report.unmeasured_graphs == []
    assert 'NOT cross-checked' in render_markdown(report)


@pytest.mark.asyncio
async def test_an_empty_but_complete_graph_is_collapsed_out_of_the_table():
    """A legibility cut that says so — the store holds dozens of probe graphs.

    Zero valid edges and a proven-complete enumeration means there is nothing
    the verdict could have missed there, so a row of zeroes is noise. The
    collapse must be STATED and the JSON must still carry the graph, or it is
    exactly the silent truncation this report is written not to do.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'probe_e1_gw0': ([], True),
    })

    report = await run(
        _args(project_id=None, measured_at=_FIXED_TIMESTAMP),
        edge_source=source,
        graph_lister=_lister('alpha', 'probe_e1_gw0'),
    )
    markdown = render_markdown(report)

    table = markdown[markdown.index('| project |'): markdown.index('## Rejection')]
    assert '`alpha`' in table
    assert '1 further graph(s)' in table, 'the cut must be stated'
    assert '`probe_e1_gw0`' in table, 'and must name what it cut'
    row_prefixes = [ln for ln in table.splitlines() if ln.startswith('| `')]
    assert row_prefixes == [ln for ln in row_prefixes if 'probe_e1_gw0' not in ln]

    payload = json.loads(render_json(report))
    assert [p['project_id'] for p in payload['projects']] == [
        'alpha', 'probe_e1_gw0',
    ], 'the JSON is the untruncated record'


# ---------------------------------------------------------------------------
# The regenerate command
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_regenerate_command_is_bare_for_an_all_defaults_run():
    """No flags to reconstruct means no flags emitted."""
    source = _FakeEdgeSource({'alpha': (_ALPHA_FACTS, True)})
    report = await run(
        _args(project_id=None, measured_at=_FIXED_TIMESTAMP),
        edge_source=source,
        graph_lister=_lister('alpha'),
    )

    assert _mod.regenerate_command(report) == _mod.REGENERATE_COMMAND
    assert json.loads(render_json(report))['regenerate'] == _mod.REGENERATE_COMMAND


@pytest.mark.asyncio
async def test_the_regenerate_command_reconstructs_the_flags_that_were_used():
    """An artifact must not tell the reader to run a DIFFERENT measurement.

    census_memory_metadata.py grew _regen_command for exactly this (task 3507:
    the committed artifact came from a non-default --top-n while the header
    printed the bare command, so following it silently produced a ~4,800-row
    -shorter markdown and still exited 0). Here the trap is worse than short:
    `--project-id reify` measures a different POPULATION, so a bare command
    would quietly re-measure another one and disagree with the numbers it is
    printed above.
    """
    source = _FakeEdgeSource({'alpha': (_ALPHA_FACTS, True)})
    report = await run(
        _args(
            project_id=['alpha'],
            page_size=1000,
            max_samples=5,
            measured_at=_FIXED_TIMESTAMP,
        ),
        edge_source=source,
    )

    command = _mod.regenerate_command(report)

    assert command.startswith(_mod.REGENERATE_COMMAND)
    assert '--project-id alpha' in command
    assert '--page-size 1000' in command
    assert '--max-samples 5' in command
    # the reconstruction has to survive the artifact round-trip
    payload = json.loads(render_json(report))
    assert payload['regenerate'] == command
    assert payload['max_samples'] == 5
    assert payload['page_size'] == 1000
    assert command in render_markdown(report)


@pytest.mark.asyncio
async def test_a_discovered_graph_set_is_not_frozen_into_the_command():
    """Re-running must re-discover, not replay today's inventory.

    Emitting --project-id per graph for a DISCOVERED set would turn the
    regenerate command into the very hardcoded list the discovery replaced:
    a graph created after this artifact was written would then be excluded by
    the artifact's own instructions.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True), 'beta': (_BETA_FACTS, True),
    })
    report = await run(
        _args(project_id=None, measured_at=_FIXED_TIMESTAMP),
        edge_source=source,
        graph_lister=_lister('alpha', 'beta'),
    )

    assert '--project-id' not in _mod.regenerate_command(report)


@pytest.mark.asyncio
async def test_the_verdict_keys_on_triaged_recall_loss_not_on_raw_matches():
    """'The guard cost nothing' and 'the guard was never reached' differ.

    Keying the verdict on ``regex_matched`` conflates them, and calls the
    whole measurement provisional the moment ONE correctly-rejected fact
    appears. That is not hypothetical: the first full-graph-set live run found
    exactly two, both prepositional complements, and the report's own text
    already told the reader adverbial_preamble was the deciding number.
    """
    # (1) nothing reaches the guard — the strongest form, no heuristic used
    empty = await run(
        _args(project_id=['alpha']),
        edge_source=_FakeEdgeSource({'alpha': (['The worker restarted.'], True)}),
    )
    assert empty.totals.regex_matched == 0
    assert empty.verdict == _mod._VERDICT_ZERO_MATCHES

    # (2) facts reach the guard, every rejection CORRECT — cost still zero,
    #     but now resting on the triage heuristic, which the verdict must say
    correct_only = await run(
        _args(project_id=['alpha']),
        edge_source=_FakeEdgeSource({
            'alpha': (['Reviews of tasks 1020 and 1030 are pending.'], True),
        }),
    )
    assert correct_only.totals.regex_matched == 1
    assert correct_only.triage_totals == {'prepositional_complement': 1}
    assert correct_only.verdict == _mod._VERDICT_NO_TRIAGED_RECALL_LOSS
    assert 'HEURISTIC' in correct_only.verdict, 'the caveat is the point'

    # (3) a single adverbial_preamble rejection IS recall loss — assert nothing
    recall_loss = await run(
        _args(project_id=['alpha']),
        edge_source=_FakeEdgeSource({'alpha': (_BETA_FACTS, True)}),
    )
    assert recall_loss.triage_totals == {'adverbial_preamble': 1}
    assert recall_loss.verdict == _mod._VERDICT_PROVISIONAL


def test_build_parser_rejects_a_max_samples_below_one():
    """A cap of zero is not a cap, it is a hide — and it had no validator.

    ``--page-size`` got ``_page_size_arg``; ``--max-samples`` was a bare
    ``type=int``, so 0 (or -1) parsed cleanly and suppressed every rejection
    sample in the markdown.
    """
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--max-samples', '0'])
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--max-samples', '-1'])

    ok = build_parser().parse_args(['--max-samples', '1'])
    assert ok.max_samples == 1


@pytest.mark.asyncio
async def test_a_non_positive_max_samples_still_reports_what_it_dropped():
    """The structural half: render_markdown itself must never cap silently.

    The CLI validator above stops an operator getting here, but
    ``render_markdown`` is called directly — by these tests, and by anything
    that builds a Report in code. It used to skip a project whose sample list
    came back empty, which skipped it BEFORE the 'N further rejection(s)
    omitted' note could fire: the triage table said `adverbial_preamble 1`
    while the samples section said `_None._` and claimed nothing was dropped.
    That is precisely the silent cap the section's own comment forbids.
    """
    source = _FakeEdgeSource({
        'alpha': (_ALPHA_FACTS, True),
        'beta': (_BETA_FACTS, True),
    })
    report = await run(
        _args(
            project_id=['alpha', 'beta'],
            measured_at=_FIXED_TIMESTAMP,
            max_samples=0,
        ),
        edge_source=source,
    )

    markdown = render_markdown(report)

    # Both projects have exactly one rejection, and both must be accounted for.
    assert sum(report.triage_totals.values()) == 2
    assert '_None._' not in markdown
    assert markdown.count('further rejection(s) omitted here') == 2
    assert '(showing 0 of 1)' in markdown
    # ...and the JSON is still the untruncated record it points readers at.
    payload = json.loads(render_json(report))
    assert sum(len(p['rejections']) for p in payload['projects']) == 2


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
