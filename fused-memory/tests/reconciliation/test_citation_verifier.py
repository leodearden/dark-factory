"""Tests for ``reconciliation.citation_verifier.verify_cited_memories``.

``verify_cited_memories`` is the Stage-1 post-assembly citation-verification
pass (task 2978): for each finding's ``cited_memories`` it resolves the
``store == 'mem0'`` ids via ``memory_service.get_memory_by_id`` and drops
phantom (genuinely not-found) ids while keeping resolved ones, so a finding's
claim is never silently backed by an id that does not exist. See the task
plan for the full three-way contract (keep / drop+mark / keep+mark).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, call

import pytest

from fused_memory.reconciliation.citation_verifier import (
    build_citation_tombstone,
    find_citation_occurrences,
    is_concrete_memory_id,
    repoint_metadata,
    verify_cited_memories,
)


@pytest.mark.asyncio
async def test_mem0_keeps_resolved_drops_phantom():
    """A resolving mem0 id is kept; a genuinely not-found one is dropped and
    recorded on the finding, with matching stats."""
    finding = {
        'description': 'a finding',
        'cited_memories': [
            {'memory_id': 'A', 'store': 'mem0'},
            {'memory_id': 'B', 'store': 'mem0'},
        ],
    }

    async def _get(project_id, memory_id):
        # 'A' exists (raw point read returns a payload dict); 'B' is a phantom.
        if memory_id == 'A':
            return {'id': 'A', 'content': 'x', 'metadata': {}}
        return None

    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(side_effect=_get)

    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    # 'A' resolves and stays; 'B' is a phantom and is stripped.
    assert [c['memory_id'] for c in finding['cited_memories']] == ['A']

    # The dropped phantom is recorded verbatim (exactly one entry).
    assert finding['citation_failures'] == [
        {'memory_id': 'B', 'store': 'mem0', 'reason': 'memory_not_found'},
    ]

    # Both ids were resolved against the raw point-id read, scoped to project_id.
    memory_service.get_memory_by_id.assert_has_awaits(
        [call('test_project', 'A'), call('test_project', 'B')],
        any_order=True,
    )

    # Stats reflect one verified + one dropped, no errors.
    assert stats['stage1_citations_verified'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 1
    assert stats['stage1_citation_verification_errors'] == 0


@pytest.mark.asyncio
async def test_backend_error_keeps_citation_and_marks_it():
    """A backend error (e.g. Qdrant timeout) is 'unknown', not 'absent': the
    citation is KEPT and marked verification_error — never dropped, never
    propagated (dropping-on-unknown would itself be a silent-fail)."""
    finding = {
        'description': 'a finding',
        'cited_memories': [{'memory_id': 'A', 'store': 'mem0'}],
    }

    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(
        side_effect=TimeoutError('qdrant timeout'),
    )

    # The exception must NOT propagate out of the verifier.
    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    # The citation is kept (unknown != absent).
    assert [c['memory_id'] for c in finding['cited_memories']] == ['A']

    # The uncertainty is surfaced via a verification_error marker.
    assert finding['citation_failures'] == [
        {
            'memory_id': 'A',
            'store': 'mem0',
            'reason': 'verification_error',
            'error_type': 'TimeoutError',
        },
    ]

    # Stats: one error, nothing dropped, nothing verified.
    assert stats['stage1_citation_verification_errors'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 0
    assert stats['stage1_citations_verified'] == 0


@pytest.mark.asyncio
async def test_graphiti_citation_left_untouched_and_never_looked_up():
    """A store=='graphiti' citation is preserved verbatim and NEVER resolved
    via get_memory_by_id (a Mem0/Qdrant-only read that would false-flag every
    graphiti edge uuid as a phantom)."""
    finding = {
        'description': 'a finding',
        'cited_memories': [
            {'memory_id': 'm1', 'store': 'mem0'},
            {'memory_id': 'g1', 'store': 'graphiti'},
        ],
    }
    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(
        return_value={'id': 'm1', 'content': 'x', 'metadata': {}},
    )

    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    # Both citations remain; the graphiti one is untouched.
    assert finding['cited_memories'] == [
        {'memory_id': 'm1', 'store': 'mem0'},
        {'memory_id': 'g1', 'store': 'graphiti'},
    ]
    assert 'citation_failures' not in finding

    # Only the mem0 id was resolved — the graphiti uuid was never looked up.
    memory_service.get_memory_by_id.assert_awaited_once_with('test_project', 'm1')

    assert stats['stage1_citations_verified'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 0
    assert stats['stage1_citation_verification_errors'] == 0


@pytest.mark.asyncio
async def test_malformed_entries_skipped_without_error():
    """Non-dict entries and dicts with a missing/empty memory_id are left in
    place without a lookup and without erroring."""
    finding = {
        'description': 'a finding',
        'cited_memories': [
            'not-a-dict',
            {'store': 'mem0'},  # missing memory_id
            {'memory_id': '', 'store': 'mem0'},  # empty memory_id
            {'memory_id': 'm1', 'store': 'mem0'},
        ],
    }
    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(
        return_value={'id': 'm1', 'content': 'x', 'metadata': {}},
    )

    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    # Every malformed entry is preserved verbatim; only the real mem0 id resolves.
    assert finding['cited_memories'] == [
        'not-a-dict',
        {'store': 'mem0'},
        {'memory_id': '', 'store': 'mem0'},
        {'memory_id': 'm1', 'store': 'mem0'},
    ]
    assert 'citation_failures' not in finding

    memory_service.get_memory_by_id.assert_awaited_once_with('test_project', 'm1')

    assert stats['stage1_citations_verified'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 0
    assert stats['stage1_citation_verification_errors'] == 0


@pytest.mark.asyncio
async def test_finding_without_cited_memories_is_noop():
    """A finding with no 'cited_memories' key is left ENTIRELY untouched — no
    lookup, no citation_failures, and (critically) no empty cited_memories key
    is added (else the pass would mutate every citation-less finding it walks)."""
    finding = {'description': 'a finding with no citations'}
    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock()

    stats = await verify_cited_memories([finding], memory_service, 'test_project')

    # The finding dict is unchanged — no cited_memories, no citation_failures.
    assert finding == {'description': 'a finding with no citations'}
    assert 'cited_memories' not in finding
    assert 'citation_failures' not in finding
    memory_service.get_memory_by_id.assert_not_awaited()
    assert stats == {
        'stage1_phantom_citations_dropped': 0,
        'stage1_citations_verified': 0,
        'stage1_citation_verification_errors': 0,
    }


@pytest.mark.asyncio
async def test_multiple_findings_processed_independently():
    """Each finding in one call is verified independently — a phantom in one
    finding never leaks its marker onto another."""
    finding_a = {
        'description': 'finding A',
        'cited_memories': [{'memory_id': 'good', 'store': 'mem0'}],
    }
    finding_b = {
        'description': 'finding B',
        'cited_memories': [{'memory_id': 'phantom', 'store': 'mem0'}],
    }

    async def _get(project_id, memory_id):
        return {'id': memory_id} if memory_id == 'good' else None

    memory_service = AsyncMock()
    memory_service.get_memory_by_id = AsyncMock(side_effect=_get)

    stats = await verify_cited_memories(
        [finding_a, finding_b], memory_service, 'test_project',
    )

    # A's good citation stays and is unmarked.
    assert [c['memory_id'] for c in finding_a['cited_memories']] == ['good']
    assert 'citation_failures' not in finding_a

    # B's phantom is dropped and marked on B alone.
    assert finding_b['cited_memories'] == []
    assert finding_b['citation_failures'] == [
        {'memory_id': 'phantom', 'store': 'mem0', 'reason': 'memory_not_found'},
    ]

    assert stats['stage1_citations_verified'] == 1
    assert stats['stage1_phantom_citations_dropped'] == 1
    assert stats['stage1_citation_verification_errors'] == 0


# --------------------------------------------------------------------------- #
# find_citation_occurrences — the mechanical all-keys metadata scan (task 3108)
# --------------------------------------------------------------------------- #

# Two distinct canonical UUIDs that deliberately SHARE an 8-char prefix, so the
# prefix-collision guard (case g) exercises the truncated-UUID hazard that
# ``prompts/stage1.py`` warns about at :99-113.
_DOOMED = '2531b4d8-1111-4aaa-8bbb-000000000001'
_PREFIX_TWIN = '2531b4d8-2222-4ccc-8ddd-000000000002'
_SURVIVOR = '9f3ac071-3333-4eee-8fff-000000000003'


class TestFindCitationOccurrences:
    """``find_citation_occurrences`` is a pure recursive scan over ALL keys.

    It exists because incident failure mode (1) was a hand-written enumeration
    of citation-bearing tasks that found 3 of 8 — the 5 it missed included the
    pending/dispatchable ones. A key allowlist cannot be trusted, so the scan
    descends every dict key, every list index and every free-text string.
    """

    def test_top_level_scalar_value_is_the_uuid(self):
        """(a) A top-level scalar whose value IS the UUID yields its bare key."""
        metadata = {'mem0_canonical_entry': _DOOMED}

        assert find_citation_occurrences(metadata, _DOOMED) == ['mem0_canonical_entry']

    def test_uuid_nested_in_a_list_of_dicts_yields_indexed_path(self):
        """(b) A UUID nested in a list of dicts yields a dotted+indexed path."""
        metadata = {
            'x_memory_write_caution': [
                {'entry': _SURVIVOR, 'note': 'survivor'},
                {'entry': _DOOMED, 'note': 'doomed'},
            ],
        }

        assert find_citation_occurrences(metadata, _DOOMED) == [
            'x_memory_write_caution[1].entry',
        ]

    def test_uuid_embedded_as_substring_of_free_text_query(self):
        """(c) A UUID embedded INSIDE free text is found — the real
        ``memory_hints.queries[0]`` shape (shared/task_metadata.py:185,193-194)
        that the incident actually hit."""
        metadata = {
            'memory_hints': {
                'entities': ['MemoryConsolidator'],
                'queries': [f'see canonical entry {_DOOMED} for the consolidated advice'],
            },
        }

        assert find_citation_occurrences(metadata, _DOOMED) == ['memory_hints.queries[0]']

    def test_every_occurrence_in_one_blob_is_returned(self):
        """(d) Multiple occurrences across unrelated keys are ALL returned."""
        metadata = {
            'mem0_canonical_entry': _DOOMED,
            'memory_hints': {'entities': [], 'queries': [f'prose about {_DOOMED} here']},
            'mem0_cluster_entries': [_SURVIVOR, _DOOMED],
            'unrelated': {'deep': {'deeper': [{'k': _DOOMED}]}},
        }

        paths = find_citation_occurrences(metadata, _DOOMED)

        assert sorted(paths) == sorted([
            'mem0_canonical_entry',
            'memory_hints.queries[0]',
            'mem0_cluster_entries[1]',
            'unrelated.deep.deeper[0].k',
        ])

    def test_absent_uuid_yields_empty_list(self):
        """(e) A blob that does not cite the id yields no paths."""
        metadata = {'mem0_canonical_entry': _SURVIVOR, 'memory_hints': {'queries': ['x']}}

        assert find_citation_occurrences(metadata, _DOOMED) == []

    @pytest.mark.parametrize('metadata', [{}, None, 'a string', 42, ['a', 'list']])
    def test_malformed_or_empty_input_yields_empty_list_without_raising(self, metadata):
        """(f) ``{}`` / ``None`` / non-dict input returns ``[]`` and never raises."""
        assert find_citation_occurrences(metadata, _DOOMED) == []

    def test_prefix_twin_uuid_is_not_matched(self):
        """(g) A DIFFERENT uuid sharing an 8-char prefix is NOT a citation.

        Guards the truncated-prefix hazard: matching on `'2531b4d8'` would
        falsely repoint an unrelated entry.
        """
        metadata = {
            'mem0_canonical_entry': _PREFIX_TWIN,
            'memory_hints': {'queries': [f'mentions {_PREFIX_TWIN} only']},
        }

        assert find_citation_occurrences(metadata, _DOOMED) == []

        # And when BOTH ids are present, only the doomed one is a citation —
        # the twin is never dragged along by its shared prefix.
        mixed = {
            'twin': _PREFIX_TWIN,
            'doomed': _DOOMED,
            'prose': f'{_PREFIX_TWIN} and {_DOOMED} are different entries',
        }
        assert find_citation_occurrences(mixed, _DOOMED) == ['doomed', 'prose']


class TestRepointMetadata:
    """``repoint_metadata`` rewrites every occurrence to the survivor.

    Pure and deep-copying: the caller's blob is never mutated, so a failed
    write can never leave a half-rewritten object behind.
    """

    def test_exact_scalar_occurrence_is_replaced(self):
        """(a) A scalar whose whole value is the doomed id becomes the survivor."""
        metadata = {'mem0_canonical_entry': _DOOMED}

        repointed, count = repoint_metadata(metadata, _DOOMED, _SURVIVOR)

        assert repointed == {'mem0_canonical_entry': _SURVIVOR}
        assert count == 1

    def test_free_text_substring_rewritten_with_prose_preserved(self):
        """(b) An embedded id is rewritten in place; the surrounding prose is
        preserved verbatim."""
        metadata = {
            'memory_hints': {
                'entities': ['MemoryConsolidator'],
                'queries': [f'see canonical entry {_DOOMED} for the consolidated advice'],
            },
        }

        repointed, count = repoint_metadata(metadata, _DOOMED, _SURVIVOR)

        assert repointed['memory_hints']['queries'] == [
            f'see canonical entry {_SURVIVOR} for the consolidated advice',
        ]
        assert repointed['memory_hints']['entities'] == ['MemoryConsolidator']
        assert count == 1

    def test_nested_list_and_dict_occurrences_all_rewritten(self):
        """(c) Occurrences at every depth are rewritten, not just top-level ones."""
        metadata = {
            'x_memory_write_caution': [
                {'entry': _DOOMED},
                {'entry': _SURVIVOR},
            ],
            'deep': {'deeper': {'deepest': [_DOOMED, 'unrelated']}},
        }

        repointed, count = repoint_metadata(metadata, _DOOMED, _SURVIVOR)

        assert repointed['x_memory_write_caution'] == [
            {'entry': _SURVIVOR},
            {'entry': _SURVIVOR},
        ]
        assert repointed['deep']['deeper']['deepest'] == [_SURVIVOR, 'unrelated']
        assert count == 2

    def test_count_agrees_with_find_citation_occurrences(self):
        """(d) The two functions cannot drift: the rewrite count equals the
        number of paths the scanner reports for the same blob."""
        metadata = {
            'mem0_canonical_entry': _DOOMED,
            'memory_hints': {'entities': [], 'queries': [f'prose {_DOOMED} prose']},
            'mem0_cluster_entries': [_SURVIVOR, _DOOMED],
            'unrelated': {'deep': [{'k': _DOOMED}]},
        }

        paths = find_citation_occurrences(metadata, _DOOMED)
        _, count = repoint_metadata(metadata, _DOOMED, _SURVIVOR)

        assert count == len(paths) == 4

    def test_input_metadata_is_not_mutated(self):
        """(e) Deep-copy semantics — the caller's object still cites the doomed id."""
        metadata = {
            'mem0_canonical_entry': _DOOMED,
            'memory_hints': {'queries': [f'prose {_DOOMED}']},
        }

        repointed, _ = repoint_metadata(metadata, _DOOMED, _SURVIVOR)

        assert metadata['mem0_canonical_entry'] == _DOOMED
        assert metadata['memory_hints']['queries'] == [f'prose {_DOOMED}']
        # ...and the returned blob is a genuinely separate object graph.
        assert repointed['memory_hints'] is not metadata['memory_hints']

    def test_absent_uuid_leaves_blob_unchanged(self):
        """(f) Nothing to repoint -> an equal blob and a zero count."""
        metadata = {'mem0_canonical_entry': _SURVIVOR, 'memory_hints': {'queries': ['x']}}

        repointed, count = repoint_metadata(metadata, _DOOMED, _SURVIVOR)

        assert repointed == metadata
        assert count == 0

    def test_unrelated_keys_and_values_are_identical_afterwards(self):
        """(g) Everything that is not a citation round-trips byte-identically,
        including non-string scalars the walk must not coerce."""
        metadata = {
            'cited': _DOOMED,
            'priority': 'high',
            'attempts': 3,
            'flag': True,
            'nothing': None,
            'ratio': 0.5,
            'nested': {'list': ['a', 'b'], 'twin': _PREFIX_TWIN},
        }

        repointed, count = repoint_metadata(metadata, _DOOMED, _SURVIVOR)

        assert count == 1
        assert repointed['cited'] == _SURVIVOR
        untouched = {k: v for k, v in repointed.items() if k != 'cited'}
        assert untouched == {k: v for k, v in metadata.items() if k != 'cited'}
        # The prefix twin is emphatically not collateral damage.
        assert repointed['nested']['twin'] == _PREFIX_TWIN


# The literal string Stage 2 wrote as a "correction" during the incident.
# Running that query live returned only superseded cluster members and routed
# dispatch straight back into the contradictory advice consolidation existed
# to collapse. It must never be accepted as a forwarding pointer.
_INCIDENT_SEARCH_INSTRUCTION = 're-derive the current canonical entry via search(query=...)'


class TestConcreteReplacementPointer:
    """A forwarding pointer is only valid if it is a concrete id.

    This is the mechanical guard against incident failure mode (2): UUID-shape
    is a hard precondition of the tombstone builder, not advice in a prompt.
    """

    def test_canonical_uuid_is_concrete(self):
        """(a) A canonical 36-char UUID is a concrete pointer."""
        assert is_concrete_memory_id(_SURVIVOR) is True
        assert is_concrete_memory_id(_DOOMED) is True

    def test_incident_search_instruction_is_not_concrete(self):
        """(b) The incident's re-derive-via-search prose is NOT a pointer."""
        assert is_concrete_memory_id(_INCIDENT_SEARCH_INSTRUCTION) is False

    @pytest.mark.parametrize(
        'value',
        [
            _SURVIVOR[:8],          # truncated 8-char prefix
            '',                     # empty
            '   ',                  # whitespace
            None,                   # missing
            12345,                  # non-str
            ['a-uuid'],             # non-str container
            _SURVIVOR + 'x',        # 37 chars
            _SURVIVOR[:-1],         # 35 chars
            'ZZZZZZZZ-3333-4eee-8fff-000000000003',  # non-hex
        ],
    )
    def test_non_uuid_values_are_not_concrete(self, value):
        """(c) Prefixes, empties, None, non-strs and malformed UUIDs are rejected."""
        assert is_concrete_memory_id(value) is False

    def test_tombstone_names_both_ids_paths_and_run(self):
        """(d) The tombstone record preserves the old->new mapping explicitly."""
        record = build_citation_tombstone(
            superseded_id=_DOOMED,
            replacement_id=_SURVIVOR,
            paths=['mem0_canonical_entry', 'memory_hints.queries[0]'],
            run_id='run-abc',
        )

        assert record['superseded_memory_id'] == _DOOMED
        assert record['replacement_memory_id'] == _SURVIVOR
        assert record['paths'] == ['mem0_canonical_entry', 'memory_hints.queries[0]']
        assert record['run_id'] == 'run-abc'

    def test_tombstone_refuses_a_search_instruction_replacement(self):
        """(e) A search instruction can never become a forwarding pointer."""
        with pytest.raises(ValueError):
            build_citation_tombstone(
                superseded_id=_DOOMED,
                replacement_id=_INCIDENT_SEARCH_INSTRUCTION,
                paths=['mem0_canonical_entry'],
                run_id='run-abc',
            )

    @pytest.mark.parametrize('bad', [_SURVIVOR[:8], '', None, 12345])
    def test_tombstone_refuses_any_non_concrete_replacement(self, bad):
        """(e) ...and the same refusal covers every non-concrete shape."""
        with pytest.raises(ValueError):
            build_citation_tombstone(
                superseded_id=_DOOMED,
                replacement_id=bad,
                paths=[],
                run_id='run-abc',
            )
