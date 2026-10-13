"""Tests for ``scripts/sweep_uuid_prefixes.py`` (task 5325, PRD C6).

Fixture ids are HEX-LETTERED (``bff81530``, ``cafe0123``) so a case or prefix
assertion cannot pass vacuously on a digits-only id.

No test touches a live store: escalation files are built under ``tmp_path``,
the tasks store is a seeded sqlite file, and the resolver, the mem0 corpus and
the MCP task client are injected fakes.
"""
import asyncio
import copy

import pytest

# `import sweep_uuid_prefixes` resolves because scripts/tests/conftest.py puts
# scripts/ on sys.path; the script's own bootstrap then binds `shared` to this
# checkout.
import sweep_uuid_prefixes as sweep
from shared.uuid_prefix import PrefixToken
from shared.uuid_prefix_guard import Candidate, Resolution, ResolverUnavailable

UNIQUE = 'bff81530'
FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
UNIQUE_CAFE = 'cafe0123'
FULL_CAFE = 'cafe0123-1111-4222-8333-444455556666'
FULL_OTHER = 'abcdef00-0000-4000-8000-00000000004a'
AMBIG = '208f1bdf'
AMBIG_A = '208f1bdf-0000-4000-8000-00000000000a'
AMBIG_B = '208f1bdf-0000-4000-8000-00000000000b'
NONE_TOKENS = ('deadbeef', '20260904')


def _unique(full_id: str) -> Resolution:
    return Resolution('unique', (Candidate('mem0', full_id, 'a preview'),))


RESOLUTIONS = {
    UNIQUE: _unique(FULL),
    UNIQUE_CAFE: _unique(FULL_CAFE),
    AMBIG: Resolution(
        'ambiguous',
        (
            Candidate('mem0', AMBIG_A, 'first'),
            Candidate('graphiti_node', AMBIG_B, 'second'),
        ),
    ),
    **{token: Resolution('none', ()) for token in NONE_TOKENS},
}

NO_UNWRITABLE: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# plan_record — the pure planner.
# ---------------------------------------------------------------------------


class TestPlanRecord:
    def test_every_unique_occurrence_is_expanded_at_its_span(self):
        detail = f'see {UNIQUE} and again {UNIQUE}; pair {UNIQUE}/{UNIQUE_CAFE} done'
        fields = {'detail': detail, 'other': 'untouched text'}
        snapshot = copy.deepcopy(fields)

        plan = sweep.plan_record(fields, RESOLUTIONS, NO_UNWRITABLE)

        assert plan.after == {
            'detail': detail.replace(UNIQUE, FULL).replace(UNIQUE_CAFE, FULL_CAFE),
            'other': 'untouched text',
        }
        assert [(token.token, full) for token, full in plan.expansions] == [
            (UNIQUE, FULL),
            (UNIQUE, FULL),
            (UNIQUE, FULL),
            (UNIQUE_CAFE, FULL_CAFE),
        ]
        assert fields == snapshot

    def test_an_ambiguous_token_is_reported_with_its_candidates_and_not_expanded(self):
        fields = {'detail': f'ambiguous {AMBIG} here'}

        plan = sweep.plan_record(fields, RESOLUTIONS, NO_UNWRITABLE)

        assert plan.expansions == ()
        assert plan.after is None
        assert plan.ambiguous == (
            sweep.AmbiguousSite(path=('detail',), token=AMBIG, candidates=(AMBIG_A, AMBIG_B)),
        )

    def test_none_tokens_are_only_counted(self):
        plan = sweep.plan_record(
            {'detail': f'{NONE_TOKENS[0]} on {NONE_TOKENS[1]}'}, RESOLUTIONS, NO_UNWRITABLE
        )

        assert plan.untouched == 2
        assert plan.expansions == ()
        assert plan.ambiguous == ()
        assert plan.after is None

    def test_glued_hex_runs_are_not_tokens_at_all(self):
        plan = sweep.plan_record(
            {'detail': 'recon-3f2a9c1e and episode_74b902f8'}, {}, NO_UNWRITABLE
        )

        assert plan == sweep.RecordPlan(
            expansions=(), ambiguous=(), unwritable=(), untouched=0, after=None
        )

    def test_a_nested_list_element_is_expanded(self):
        fields = {'metadata': {'cluster_memory_ids': [UNIQUE, FULL_OTHER], 'kind': 'x'}}

        plan = sweep.plan_record(fields, RESOLUTIONS, NO_UNWRITABLE)

        assert plan.after == {'metadata': {'cluster_memory_ids': [FULL, FULL_OTHER], 'kind': 'x'}}
        assert [token.path for token, _ in plan.expansions] == [
            ('metadata', 'cluster_memory_ids', 0)
        ]

    def test_a_token_in_an_unwritable_field_is_listed_and_left_alone(self):
        fields = {'test_strategy': f'check {UNIQUE}', 'details': f'{UNIQUE} cited'}

        plan = sweep.plan_record(fields, RESOLUTIONS, frozenset({'test_strategy'}))

        assert plan.unwritable == (
            PrefixToken(path=('test_strategy',), token=UNIQUE, start=6, end=14),
        )
        assert plan.after == {'test_strategy': f'check {UNIQUE}', 'details': f'{FULL} cited'}

    @pytest.mark.parametrize(
        ('fields', 'unwritable'),
        [
            ({'test_strategy': UNIQUE}, frozenset({'test_strategy'})),
            ({'detail': f'{AMBIG} {NONE_TOKENS[0]}'}, NO_UNWRITABLE),
            ({'detail': 'no hex at all'}, NO_UNWRITABLE),
        ],
        ids=['only-unwritable', 'only-ambiguous-and-none', 'no-tokens'],
    )
    def test_no_unique_writable_token_means_no_after_document(self, fields, unwritable):
        assert sweep.plan_record(fields, RESOLUTIONS, unwritable).after is None

    def test_the_after_document_is_a_fixed_point(self):
        fields = {
            'detail': f'{UNIQUE}/{UNIQUE_CAFE} and {AMBIG}',
            'metadata': {'ids': [UNIQUE]},
        }

        first = sweep.plan_record(fields, RESOLUTIONS, NO_UNWRITABLE)
        assert first.after is not None
        second = sweep.plan_record(first.after, RESOLUTIONS, NO_UNWRITABLE)

        assert second.expansions == ()
        assert second.after is None
        assert len(second.ambiguous) == len(first.ambiguous) == 1

    def test_counts_for_a_mixed_record(self):
        fields = {
            'title': f'{UNIQUE} x {AMBIG}',
            'details': f'{NONE_TOKENS[0]} {UNIQUE} {NONE_TOKENS[1]} {UNIQUE_CAFE}',
            'test_strategy': UNIQUE_CAFE,
            'metadata': {'ids': [AMBIG, NONE_TOKENS[0]]},
        }

        plan = sweep.plan_record(fields, RESOLUTIONS, frozenset({'test_strategy'}))

        assert len(plan.expansions) == 3
        assert len(plan.ambiguous) == 2
        assert plan.untouched == 3
        assert len(plan.unwritable) == 1

    def test_a_token_with_no_resolution_is_a_programming_error(self):
        with pytest.raises(KeyError):
            sweep.plan_record({'detail': UNIQUE}, {}, NO_UNWRITABLE)


# ---------------------------------------------------------------------------
# resolve_distinct — one resolution per distinct token, bounded concurrency.
# ---------------------------------------------------------------------------


class FakeResolver:
    """An async ``PrefixResolver`` that records how it was driven.

    *gate*, when given, holds every call until it is set, so a test can look at
    how many resolutions are in flight at once. A token in *unavailable* raises
    ``ResolverUnavailable`` for that store instead of answering.
    """

    def __init__(self, resolutions=None, *, gate=None, unavailable=None):
        self.resolutions = RESOLUTIONS if resolutions is None else resolutions
        self.gate = gate
        self.unavailable = unavailable or {}
        self.calls: list[tuple[str, str]] = []
        self.completed: list[str] = []
        self.cancelled: list[str] = []
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, project_id: str, prefix: str) -> Resolution:
        self.calls.append((project_id, prefix))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if prefix in self.unavailable:
                raise ResolverUnavailable(self.unavailable[prefix])
            if self.gate is not None:
                await self.gate.wait()
            else:
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled.append(prefix)
            raise
        finally:
            self.in_flight -= 1
        self.completed.append(prefix)
        return self.resolutions.get(prefix, Resolution('none', ()))


async def _settle():
    for _ in range(50):
        await asyncio.sleep(0)


class TestResolveDistinct:
    def test_each_distinct_token_is_resolved_once_under_the_given_project(self):
        resolver = FakeResolver()
        tokens = [UNIQUE, AMBIG, UNIQUE, NONE_TOKENS[0], UNIQUE, AMBIG]

        result = asyncio.run(sweep.resolve_distinct('proj_a', tokens, resolver, {}))

        assert sorted(resolver.calls) == sorted(
            [('proj_a', UNIQUE), ('proj_a', AMBIG), ('proj_a', NONE_TOKENS[0])]
        )
        assert result == {
            UNIQUE: RESOLUTIONS[UNIQUE],
            AMBIG: RESOLUTIONS[AMBIG],
            NONE_TOKENS[0]: RESOLUTIONS[NONE_TOKENS[0]],
        }

    def test_memoised_tokens_are_carried_and_not_re_resolved(self):
        resolver = FakeResolver()
        memo = {UNIQUE: RESOLUTIONS[UNIQUE]}
        snapshot = dict(memo)

        result = asyncio.run(sweep.resolve_distinct('proj_a', [UNIQUE, UNIQUE_CAFE], resolver, memo))

        assert resolver.calls == [('proj_a', UNIQUE_CAFE)]
        assert result == {UNIQUE: RESOLUTIONS[UNIQUE], UNIQUE_CAFE: RESOLUTIONS[UNIQUE_CAFE]}
        assert memo == snapshot

    def test_in_flight_resolutions_never_exceed_the_bound(self):
        tokens = [f'{n:x}abcdef0' for n in range(10, 16)]

        async def scenario():
            gate = asyncio.Event()
            resolver = FakeResolver(gate=gate)
            pending = asyncio.create_task(
                sweep.resolve_distinct('proj_a', tokens, resolver, {}, concurrency=2)
            )
            await _settle()
            held = resolver.in_flight
            gate.set()
            return resolver, held, await pending

        resolver, held, result = asyncio.run(scenario())

        assert held == 2
        assert resolver.peak == 2
        assert sorted(result) == sorted(tokens)
        assert sorted(resolver.completed) == sorted(tokens)

    def test_an_unavailable_store_surfaces_as_itself_and_leaves_nothing_pending(self):
        tokens = [UNIQUE, AMBIG, UNIQUE_CAFE, NONE_TOKENS[0]]

        async def scenario():
            resolver = FakeResolver(gate=asyncio.Event(), unavailable={AMBIG: 'mem0'})
            with pytest.raises(ResolverUnavailable) as raised:
                await sweep.resolve_distinct('proj_a', tokens, resolver, {}, concurrency=4)
            return resolver, raised.value

        resolver, error = asyncio.run(scenario())

        assert error.store == 'mem0'
        started = {prefix for _, prefix in resolver.calls}
        assert started == set(resolver.completed) | set(resolver.cancelled) | {AMBIG}
        assert resolver.in_flight == 0

    def test_no_tokens_means_no_calls(self):
        resolver = FakeResolver()
        memo = {UNIQUE: RESOLUTIONS[UNIQUE]}

        result = asyncio.run(sweep.resolve_distinct('proj_a', [], resolver, memo))

        assert resolver.calls == []
        assert result == memo
        assert result is not memo
