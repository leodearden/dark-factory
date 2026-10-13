"""Tests for ``scripts/sweep_uuid_prefixes.py`` (task 5325, PRD C6).

Fixture ids are HEX-LETTERED (``bff81530``, ``cafe0123``) so a case or prefix
assertion cannot pass vacuously on a digits-only id.

No test touches a live store: escalation files are built under ``tmp_path``,
the tasks store is a seeded sqlite file, and the resolver, the mem0 corpus and
the MCP task client are injected fakes.
"""
import copy

import pytest

# `import sweep_uuid_prefixes` resolves because scripts/tests/conftest.py puts
# scripts/ on sys.path; the script's own bootstrap then binds `shared` to this
# checkout.
import sweep_uuid_prefixes as sweep
from shared.uuid_prefix import PrefixToken
from shared.uuid_prefix_guard import Candidate, Resolution

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
