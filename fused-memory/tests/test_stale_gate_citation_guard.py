"""Tests for the stale-gate-citation guard (task 4919).

Closes the incident where task 3708's evidence-relay prose kept citing
``3660`` as a pending external gate for three relay cycles after 3660 was
coalesced into 4856; the incident and the corpus measurement are in the task
4919 record.

The relay fixture constants below are VERBATIM excerpts from task 3708's live
``details`` field (tag=master).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fm_helpers import _init_git_repo

from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation import stale_gate_citation_guard
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.stale_gate_citation_guard import find_gate_citation_ids
from fused_memory.server.tools import create_mcp_server

# --------------------------------------------------------------------------- #
# Live relay excerpts (task 3708, tag=master, re-confirmed 2026-09-11)
# --------------------------------------------------------------------------- #

# The two STALE spellings: the defect itself. Live `dependencies` is
# [3658, 3659, 3707, 4856, 4987], so the cited 3660 is stale.
STALE_A = (
    'the only real remediation lever remains external deps 3658/3659/3660 '
    'landing so this task (γ) can be dispatched'
)
STALE_B = (
    'The only real remediation lever remains external deps 3658/3659/3660/4856 '
    'landing so this task (γ) can be dispatched'
)

# The 2026-08-31 correction relay. The capture stops at ' only', so the
# HISTORICAL 3660 named after it is never read as a citation — no separate
# negation heuristic is needed.
CORRECTION = (
    'All future evidence-log relays into this field must cite pending external '
    'gates as 3658/3659/4856 only — 3660 must be dropped permanently, it no '
    'longer exists as a separate gate'
)

# Correct relays naming TRANSITIVE gates (gates of this task's gates),
# legitimately absent from this task's own `dependencies`.
TRANSITIVE = (
    'The only real remediation lever remains external deps 3659 and 4856 '
    'landing (via their own upstream gates 3212 and 4006 respectively)'
)
PARENTHETICAL = (
    'The only real remediation levers remain: 3659 (blocked on 3212→3207), '
    '4856 (blocked on 3659+4006), and now also 4987 (blocked on 4932+4986) '
    'landing'
)
ARROW_LEVERS = (
    'The only real remediation levers remain 3659 (→3212), 4856 (→3659+4006), '
    'and 4987 (→4932+4986)'
)
LEVER_CHAIN = (
    'The only real remediation levers remain 3212 (in-progress) -> 3659, and '
    '4985 -> 4986 -> 4987, landing so this task (γ) can be dispatched.'
)

CAPS_GATING = (
    'GATING DEPENDENCY 4987 OBSERVED (append-only; not evidence of a status '
    'change)'
)


class TestFindGateCitationIds:
    """The pure marker-anchored contiguous id-list scanner."""

    def test_stale_a_captures_the_three_cited_gates(self):
        assert find_gate_citation_ids(STALE_A) == {3658, 3659, 3660}

    def test_stale_b_captures_all_four_cited_gates(self):
        assert find_gate_citation_ids(STALE_B) == {3658, 3659, 3660, 4856}

    def test_correction_relay_stops_before_the_historical_id(self):
        # Capture stops at ' only'; the trailing historical 3660 is NOT a
        # citation, so no negation heuristic is needed to exonerate it.
        assert find_gate_citation_ids(CORRECTION) == {3658, 3659, 4856}

    def test_transitive_gates_are_excluded(self):
        # Stops at ' landing' — upstream gates 3212/4006 are not this task's
        # dependencies and must not be read as citations of them.
        assert find_gate_citation_ids(TRANSITIVE) == {3659, 4856}

    def test_capture_stops_at_a_parenthetical(self):
        # Deliberate under-fire: only the first id of the list is captured.
        assert find_gate_citation_ids(
            'external deps 3659 (blocked on 3212→3207), 4856'
        ) == {3659}

    def test_marker_with_a_colon_connector_anchors(self):
        assert find_gate_citation_ids('external gates: 3659/4856 landing') == {
            3659, 4856,
        }

    def test_all_caps_marker_matches_case_insensitively(self):
        assert find_gate_citation_ids(CAPS_GATING) == {4987}

    def test_remediation_levers_is_not_a_marker(self):
        # A remediation lever may be a transitive gate: LEVER_CHAIN leads with
        # 3212, which gates 3659, not this task.
        for text in (PARENTHETICAL, ARROW_LEVERS, LEVER_CHAIN):
            assert find_gate_citation_ids(text) == set()

    def test_a_date_after_the_marker_is_not_a_task_id(self):
        assert find_gate_citation_ids(
            'pending external gates: 2026-09-11 check found none open'
        ) == set()

    def test_a_date_after_the_id_list_does_not_extend_it(self):
        assert find_gate_citation_ids(
            'external deps 3658/3659, 2026-09-11 check'
        ) == {3658, 3659}

    def test_marker_with_no_adjacent_id_list_is_a_no_op(self):
        assert find_gate_citation_ids(
            'FRESH STATUS CHECK ON ALL FIVE GATING DEPENDENCIES:'
        ) == set()

    def test_text_with_no_marker_is_a_no_op(self):
        assert find_gate_citation_ids(
            'actual_total=2 / expected_total=38 (36 missing)'
        ) == set()

    def test_blocked_on_is_not_a_marker(self):
        # Measured to false-positive on real relay prose: '3659: pending —
        # blocked on 3212' cites a correct TRANSITIVE gate.
        assert find_gate_citation_ids('blocked on 3212') == set()

    def test_upstream_gates_is_not_a_marker(self):
        # Same reason: these are gates of this task's gates.
        assert find_gate_citation_ids(
            'via their own upstream gates 3212 and 4006'
        ) == set()

    def test_empty_text_is_a_no_op(self):
        assert find_gate_citation_ids('') == set()

    def test_text_with_no_digits_is_a_no_op(self):
        assert find_gate_citation_ids(
            'the only real remediation lever remains external deps landing'
        ) == set()


class TestTerminalOutcomeEscape:
    """A RETROSPECTIVE statement about gates that already landed is a
    legitimate relay, even when written against an already-emptied
    `dependencies` array. Only an outcome reported directly of the id list
    makes it one."""

    def test_have_landed_is_not_a_pending_gate_assertion(self):
        assert find_gate_citation_ids(
            'external deps 3658/3659 have landed and this task is unblocked'
        ) == set()

    def test_are_all_done_is_not_a_pending_gate_assertion(self):
        assert find_gate_citation_ids(
            'The external deps 3658/3659/4856 are all done'
        ) == set()

    def test_were_merged_is_not_a_pending_gate_assertion(self):
        assert find_gate_citation_ids(
            'pending external gates 3658/3659 were merged last week'
        ) == set()

    def test_have_all_been_merged_is_not_a_pending_gate_assertion(self):
        assert find_gate_citation_ids(
            'external deps 3658/3659 have all been merged'
        ) == set()

    def test_escape_does_not_suppress_the_live_stale_spelling(self):
        # 'landing' is not a terminal outcome.
        assert find_gate_citation_ids(STALE_A) == {3658, 3659, 3660}

    def test_escape_does_not_suppress_the_transitive_relay(self):
        assert find_gate_citation_ids(TRANSITIVE) == {3659, 4856}

    def test_escape_does_not_suppress_the_caps_relay(self):
        assert find_gate_citation_ids(CAPS_GATING) == {4987}

    def test_terminal_cue_about_this_task_does_not_suppress(self):
        # 'done' describes this task, not the gates.
        assert find_gate_citation_ids(
            'external deps 3658/3659/3660 landing so this task can be done'
        ) == {3658, 3659, 3660}

    def test_terminal_cue_about_one_listed_id_does_not_suppress_the_list(self):
        assert find_gate_citation_ids(
            'external deps 3658/3659/3660 (3658 landed)'
        ) == {3658, 3659, 3660}

    def test_negated_terminal_cue_does_not_suppress(self):
        assert find_gate_citation_ids(
            'external deps 3658/3659/3660 are not done'
        ) == {3658, 3659, 3660}


# Recon-stage agent_id — the only caller class this guard polices. Matches
# test_recon_write_policy.py's AGENT_ID.
AGENT_ID = 'recon-stage-task_knowledge_sync'

# Task 3708's real dependencies array, re-confirmed against the live store
# (tag=master) on 2026-09-11. `dependencies` is the separate relational table
# dependencies(tag, task_id, depends_on), not a task column.
LIVE_DEPS = [3658, 3659, 3707, 4856, 4987]


class TestStaleGateCitationError:
    """The predicate: does a gate assertion cite an id absent from the live
    `dependencies` array? Flat ``dict | None``, mirroring
    ``premise_lint_guard.premise_lint_error``."""

    def test_stale_a_is_rejected(self):
        err = stale_gate_citation_guard.stale_gate_citation_error(
            STALE_A, AGENT_ID, live_dependencies=LIVE_DEPS,
        )
        assert err is not None
        assert err['error_type'] == 'ReconStaleGateCitationRejected'
        assert '3660' in err['error']

    def test_stale_b_is_rejected(self):
        err = stale_gate_citation_guard.stale_gate_citation_error(
            STALE_B, AGENT_ID, live_dependencies=LIVE_DEPS,
        )
        assert err is not None
        assert err['error_type'] == 'ReconStaleGateCitationRejected'
        assert '3660' in err['error']

    def test_rejection_names_the_corrective_path(self):
        err = stale_gate_citation_guard.stale_gate_citation_error(
            STALE_A, AGENT_ID, live_dependencies=LIVE_DEPS,
        )
        assert err is not None
        assert 'dependencies' in err['hint']

    def test_rejection_carries_the_live_array_so_no_second_read_is_needed(self):
        # The whole reason blocking does not lose the evidence: Stage 2 can
        # rewrite the one sentence and retry in the same turn.
        err = stale_gate_citation_guard.stale_gate_citation_error(
            STALE_A, AGENT_ID, live_dependencies=LIVE_DEPS,
        )
        assert err is not None
        message = err['error'] + err['hint']
        for dep in LIVE_DEPS:
            assert str(dep) in message

    # --- fail-open: no violation ------------------------------------------- #

    def test_correct_relays_are_not_rejected(self):
        for text in (
            CORRECTION, TRANSITIVE, PARENTHETICAL, CAPS_GATING, ARROW_LEVERS,
            LEVER_CHAIN,
        ):
            assert stale_gate_citation_guard.stale_gate_citation_error(
                text, AGENT_ID, live_dependencies=LIVE_DEPS,
            ) is None

    def test_a_dated_relay_is_not_rejected(self):
        assert stale_gate_citation_guard.stale_gate_citation_error(
            'pending external gates: 2026-09-11 check found none open',
            AGENT_ID,
            live_dependencies=[],
        ) is None

    def test_non_recon_callers_are_not_policed(self):
        for agent_id in ('claude-task-4919-implementer', None, ''):
            assert stale_gate_citation_guard.stale_gate_citation_error(
                STALE_A, agent_id, live_dependencies=LIVE_DEPS,
            ) is None

    def test_absent_or_non_str_details_is_a_no_op(self):
        for details in (None, '', 123, ['external deps 3660']):
            assert stale_gate_citation_guard.stale_gate_citation_error(
                details, AGENT_ID, live_dependencies=LIVE_DEPS,
            ) is None

    def test_uninterpretable_dependencies_fails_open(self):
        for deps in (None, 'reify:6508', {3658: True}, ['reify:6508', 'nope']):
            assert stale_gate_citation_guard.stale_gate_citation_error(
                STALE_A, AGENT_ID, live_dependencies=deps,
            ) is None

    def test_text_without_a_marker_is_a_no_op(self):
        assert stale_gate_citation_guard.stale_gate_citation_error(
            'actual_total=2 / expected_total=38 (36 missing)',
            AGENT_ID,
            live_dependencies=[],
        ) is None

    # --- normalisation: load-bearing, not defensive padding ----------------- #
    #
    # sqlite_task_backend._row_to_task types `dependencies` as list[int] on
    # READ while TaskBackend.update_task(..., dependencies: list[str] | None)
    # takes list[str] on WRITE, so this predicate genuinely receives both
    # shapes depending on which source the interceptor picks.

    def test_string_dependencies_behave_identically_to_ints(self):
        as_strings = ['3658', '3659', '3707', '4856', '4987']
        assert stale_gate_citation_guard.stale_gate_citation_error(
            CORRECTION, AGENT_ID, live_dependencies=as_strings,
        ) is None
        assert stale_gate_citation_guard.stale_gate_citation_error(
            CORRECTION, AGENT_ID, live_dependencies=LIVE_DEPS,
        ) is None
        assert stale_gate_citation_guard.stale_gate_citation_error(
            STALE_A, AGENT_ID, live_dependencies=as_strings,
        ) is not None
        assert stale_gate_citation_guard.stale_gate_citation_error(
            STALE_A, AGENT_ID, live_dependencies=LIVE_DEPS,
        ) is not None

    def test_mixed_int_and_str_dependencies_are_accepted(self):
        err = stale_gate_citation_guard.stale_gate_citation_error(
            STALE_A, AGENT_ID, live_dependencies=[3658, '3659', 3707],
        )
        assert err is not None
        assert '3660' in err['error']

    def test_cross_project_gate_in_external_deps_is_live(self):
        # Cross-project gates live in metadata.external_deps, not dependencies.
        assert stale_gate_citation_guard.stale_gate_citation_error(
            'pending external gates: 6508 (reify) landing',
            AGENT_ID,
            live_dependencies=[],
            metadata_payloads=({'external_deps': ['reify:6508']},),
        ) is None

    def test_external_deps_in_a_json_metadata_payload_are_read(self):
        assert stale_gate_citation_guard.stale_gate_citation_error(
            'external deps 6508 landing',
            AGENT_ID,
            live_dependencies=[],
            metadata_payloads=(None, '{"external_deps": ["reify:6508"]}'),
        ) is None

    def test_external_deps_do_not_excuse_an_unrelated_stale_id(self):
        err = stale_gate_citation_guard.stale_gate_citation_error(
            'external deps 6508/3660 landing',
            AGENT_ID,
            live_dependencies=[],
            metadata_payloads=({'external_deps': ['reify:6508', 'garbage']},),
        )
        assert err is not None
        assert '3660' in err['error']

    def test_empty_dependencies_still_enforces(self):
        # An empty array is interpretable, not missing: citing pending gates
        # when there are none is exactly the invariant violation.
        err = stale_gate_citation_guard.stale_gate_citation_error(
            STALE_A, AGENT_ID, live_dependencies=[],
        )
        assert err is not None
        assert err['error_type'] == 'ReconStaleGateCitationRejected'

    def test_retrospective_against_an_empty_array_is_allowed(self):
        assert stale_gate_citation_guard.stale_gate_citation_error(
            'external deps 3658/3659 have landed and this task is unblocked',
            AGENT_ID,
            live_dependencies=[],
        ) is None


# --------------------------------------------------------------------------- #
# Interceptor boundary fixtures (copied from test_recon_write_policy.py — this
# repo defines the quartet per test file rather than in a conftest)
# --------------------------------------------------------------------------- #


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '2', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.remove_tasks = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': [{'type': 'knowledge_captured'}]})
    return r


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'interceptor_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer)


@pytest.fixture
def live_row(taskmaster):
    """Task 3708's live row. `taskmaster.get_task`'s return value IS the fake
    live row — no real sqlite backend is needed for these boundary tests."""
    row = {
        'id': '3708',
        'status': 'pending',
        'title': 'γ',
        'dependencies': [3658, 3659, 3707, 4856, 4987],
    }
    taskmaster.get_task = AsyncMock(return_value=row)
    return row


class TestUpdateTaskBoundary:
    """The guard must actually be wired into TaskInterceptor.update_task —
    a green predicate that nothing calls closes nothing."""

    @pytest.mark.asyncio
    async def test_stale_citation_is_rejected_before_the_write(
        self, interceptor, taskmaster, live_row,
    ):
        result = await interceptor.update_task(
            '3708', '/project', details=STALE_A, append=True, agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconStaleGateCitationRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_corrected_citation_is_written(self, interceptor, taskmaster, live_row):
        result = await interceptor.update_task(
            '3708', '/project', details=TRANSITIVE, append=True, agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()
        assert result.get('error_type') is None

    @pytest.mark.asyncio
    async def test_non_recon_writer_is_not_policed(self, interceptor, taskmaster, live_row):
        await interceptor.update_task(
            '3708', '/project', details=STALE_A, append=True,
            agent_id='claude-task-4919-implementer',
        )

        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_without_details_is_a_no_op(self, interceptor, taskmaster, live_row):
        await interceptor.update_task(
            '3708', '/project', metadata={'x_relay_note': 'ok'}, agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_guard_keys_on_details_not_on_append_mode(
        self, interceptor, taskmaster, live_row,
    ):
        await interceptor.update_task(
            '3708', '/project', details=TRANSITIVE, agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_replace_mode_is_also_rejected(self, interceptor, taskmaster, live_row):
        result = await interceptor.update_task(
            '3708', '/project', details=STALE_A, append=False, agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconStaleGateCitationRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dependency_rewrite_in_the_same_call_is_judged_against_the_kwarg(
        self, interceptor, taskmaster, live_row,
    ):
        # One call may legitimately rewire the array AND relay that change.
        # The citation must be checked against the array the write LEAVES
        # BEHIND, so 3660 becoming a real dependency makes STALE_A correct.
        # The string form is deliberate: that is the kwarg's declared type on
        # sqlite_task_backend.TaskBackend.update_task, so this also pins that
        # the int/str normalisation is reached on the real path.
        await interceptor.update_task(
            '3708', '/project', details=STALE_A,
            dependencies=['3658', '3659', '3660', '3707'], agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dependency_rewrite_that_drops_a_cited_gate_is_rejected(
        self, interceptor, taskmaster, live_row,
    ):
        result = await interceptor.update_task(
            '3708', '/project', details=CAPS_GATING,
            dependencies=['3658'], agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconStaleGateCitationRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cross_project_gate_on_the_live_row_is_written(
        self, interceptor, taskmaster,
    ):
        taskmaster.get_task = AsyncMock(
            return_value={
                'id': '3708', 'status': 'pending', 'title': 'γ',
                'dependencies': [],
                'metadata': {'external_deps': ['reify:6508']},
            },
        )

        result = await interceptor.update_task(
            '3708', '/project', details='pending external gates: 6508 (reify) landing',
            append=True, agent_id=AGENT_ID,
        )

        assert result.get('error_type') is None
        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_row_without_dependencies_fails_open(self, interceptor, taskmaster):
        taskmaster.get_task = AsyncMock(
            return_value={'id': '3708', 'status': 'pending', 'title': 'γ'},
        )

        await interceptor.update_task(
            '3708', '/project', details=STALE_A, append=True, agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rejection_issues_no_second_read(self, interceptor, taskmaster, live_row):
        # The guard reuses the `before` read the recon-write-policy gate already
        # took; a recon-stage write still issues exactly one get_task.
        await interceptor.update_task(
            '3708', '/project', details=STALE_A, append=True, agent_id=AGENT_ID,
        )

        assert taskmaster.get_task.await_count == 1

    @pytest.mark.asyncio
    async def test_terminal_write_policy_keeps_priority(self, interceptor, taskmaster):
        # Ordering: the pre-existing gate fires first, so a stale citation on a
        # done task reports the terminal rejection, not this guard's.
        taskmaster.get_task = AsyncMock(
            return_value={
                'id': '3708', 'status': 'done', 'title': 'γ',
                'dependencies': [3658, 3659, 3707, 4856, 4987],
            },
        )

        result = await interceptor.update_task(
            '3708', '/project', details=STALE_A, append=True, agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconTerminalWriteRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_none_dependencies_is_judged_against_the_live_row(
        self, interceptor, taskmaster, live_row,
    ):
        # The production call shape: server/tools.py::update_task always
        # forwards the key, `None` when the write leaves the array untouched.
        result = await interceptor.update_task(
            '3708', '/project', details=STALE_A, append=True,
            dependencies=None, agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconStaleGateCitationRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_explicit_none_dependencies_allows_a_correct_relay(
        self, interceptor, taskmaster, live_row,
    ):
        result = await interceptor.update_task(
            '3708', '/project', details=TRANSITIVE, append=True,
            dependencies=None, agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()
        assert result.get('error_type') is None

    @pytest.mark.asyncio
    async def test_explicit_empty_dependencies_is_a_rewrite_not_an_absence(
        self, interceptor, taskmaster, live_row,
    ):
        # Regression guard, not a RED signal: `[]` clears the array, so it must
        # not fall back to the live row the way `None` does.
        result = await interceptor.update_task(
            '3708', '/project', details=STALE_A, append=True,
            dependencies=[], agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconStaleGateCitationRejected'
        taskmaster.update_task.assert_not_awaited()


@pytest.fixture
def project_root(tmp_path):
    """A real git repo, so the MCP tool's project-root normalisation resolves
    it as a main checkout without monkeypatching."""
    _init_git_repo(tmp_path)
    return str(tmp_path)


@pytest.fixture
def update_task_tool(interceptor):
    """The `update_task` MCP tool's own closure over the real interceptor.

    Called directly rather than through FastMCP's `call_tool`, whose injected
    Context raises outside a live request; the caller-identity resolution then
    sees no ctx, and an omitted `dependencies` still takes the tool's `None`
    default exactly as a live call does.
    """
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    tool = server._tool_manager.get_tool('update_task')
    assert tool is not None
    return tool.fn


class TestUpdateTaskMcpTool:
    """Drives the real `update_task` MCP tool, so the interceptor receives
    exactly the kwargs a live Stage 2 call sends — including the
    `dependencies=None` the tool always forwards when the array is untouched."""

    @pytest.mark.asyncio
    async def test_stale_relay_via_the_mcp_tool_is_rejected(
        self, update_task_tool, taskmaster, live_row, project_root,
    ):
        result = await update_task_tool(
            id='3708', project_root=project_root,
            details=STALE_A, append=True, agent_id=AGENT_ID,
        )

        assert result['error_type'] == 'ReconStaleGateCitationRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_correct_relay_via_the_mcp_tool_is_written(
        self, update_task_tool, taskmaster, live_row, project_root,
    ):
        await update_task_tool(
            id='3708', project_root=project_root,
            details=TRANSITIVE, append=True, agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()
        assert taskmaster.update_task.await_args.kwargs.get('dependencies') is None


class TestGateCitationPromptSection:
    """The generative half. The guard alone makes a bad relay unlandable only
    after Stage 2 has spent the turn composing it; the prompt alone is
    advisory and demonstrably insufficient — this incident IS an agent
    re-reading live task state and still copy-forwarding the prose. Both
    ship, sharing ERROR_TYPE and the marker tuple so they cannot drift.

    Assertions stay at the level of "the mandate is present and
    machine-linkable" — never sentence wording, which would make this a prose
    pin rather than a behaviour test.
    """

    def test_section_is_rendered(self):
        section = stale_gate_citation_guard.render_gate_citation_section()
        assert isinstance(section, str)
        assert section

    def test_section_names_the_error_type_the_guard_returns(self):
        # Single source of truth: renaming ERROR_TYPE cannot silently orphan
        # the prompt's description of the rejection.
        assert (
            stale_gate_citation_guard.ERROR_TYPE
            in stale_gate_citation_guard.render_gate_citation_section()
        )

    def test_section_lists_every_policed_marker_phrase(self):
        # An agent reading the prompt must be able to tell which phrasings are
        # policed, so the prompt cannot describe a narrower rule than the regex
        # enforces.
        section = stale_gate_citation_guard.render_gate_citation_section()
        for phrase in stale_gate_citation_guard.GATE_CITATION_MARKER_PHRASES:
            assert phrase in section

    def test_section_is_embedded_in_the_stage2_prompt_exactly_once(self):
        from fused_memory.reconciliation.prompts.stage2 import (
            STAGE2_SYSTEM_PROMPT,
            build_stage2_system_prompt,
        )

        section = stale_gate_citation_guard.render_gate_citation_section()
        assert STAGE2_SYSTEM_PROMPT.count(section) == 1
        assert build_stage2_system_prompt('dark_factory').count(section) == 1
