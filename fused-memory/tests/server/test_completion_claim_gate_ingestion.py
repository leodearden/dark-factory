"""Integration tests for the completion-claim verification gate on episode
ingestion (task 3142, PRD leaf pi / contract C4).

The gate is the code-level enforcement of the "Terminal-State Pre-Check
Discipline" that until now existed only as Stage-1 prompt prose. Reify
escalation ``esc-5603-1`` is the motivating incident: an episode asserting a
fix "has been applied" for a still-in-progress task was fanned out by
Graphiti's extraction pipeline into FIVE false edges. ``esc-3085-1`` extended
the scope to filing/dispatch claims naming a ticket that does not exist, and
across projects.

CONTRACT, and the two ways it differs from its closest sibling
(``_premature_completion_block``, task 2824):

* It TAGS, never rejects. The episode is always ingested; a non-verified claim
  only adds ``unverified_claim=True`` to the service call (which rides through
  to the Graphiti ``source_description`` prefix and every derived Mem0 fact's
  metadata) plus a structured flag on the tool response.
* It applies to ALL writers, not only ``recon-stage-`` agents.

The content used throughout says "has been applied" deliberately: `applied` is
NOT in ``task_filter.PRESENT_TENSE_COMPLETION_RE``'s vocabulary, so the 2824
gate cannot fire on it and these tests observe THIS gate in isolation even for
a recon-stage agent id.
"""

from __future__ import annotations

import logging
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.server.tools import create_mcp_server

# An applied-work completion claim naming task 5422 — the esc-5603-1 shape.
_APPLIED_CONTENT = "task 5422's de-flake fix has been applied"
_PROJECT_ID = 'dark_factory'
_KNOWN_PROJECTS = {'dark_factory': '/root'}


def _episode_service():
    """An AsyncMock memory_service whose add_episode returns a dict-dumpable
    result (so the tool's `return result.model_dump()` yields a real dict).
    """
    mock_service = AsyncMock()
    _ep_result = MagicMock()
    _ep_result.model_dump.return_value = {'id': 'ep'}
    mock_service.add_episode.return_value = _ep_result
    return mock_service


def _server(
    mock_service,
    *,
    statuses: dict | None = None,
    known_projects: dict | None = None,
):
    """Build a hermetic server whose task_interceptor answers `statuses`.

    `statuses` is the string-keyed {id: status} map real get_statuses returns.
    get_ticket_row is stubbed to the "no such ticket" answer; tests that care
    about tickets override it.
    """
    task_interceptor = MagicMock()
    task_interceptor.get_statuses = AsyncMock(return_value=statuses or {})
    task_interceptor.get_ticket_row = AsyncMock(return_value=None)
    return create_mcp_server(
        mock_service,
        task_interceptor=task_interceptor,
        known_projects=_KNOWN_PROJECTS if known_projects is None else known_projects,
    )


def _service_kwargs(mock_service) -> dict:
    """The kwargs memory_service.add_episode was called with."""
    return mock_service.add_episode.call_args.kwargs


class TestAddEpisodeUnverifiedClaimTagging:
    """PRIMARY SIGNAL (first half): a completion claim naming a task whose LIVE
    status is non-terminal is INGESTED and TAGGED — never rejected.
    """

    @pytest.mark.asyncio
    async def test_tags_applied_work_claim_for_in_progress_task(self):
        """The acceptance case. Episode claims task 5422's fix "has been
        applied"; task 5422 is live in-progress. The episode must still be
        ingested (add_episode awaited exactly once, no error dict), the service
        call must carry unverified_claim=True, and the tool response must carry
        a structured flag naming the claim text, the ref, and the OBSERVED live
        status.
        """
        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'in-progress'})

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        assert isinstance(result, dict), f'Expected dict, got {type(result)}: {result!r}'
        assert 'error' not in result, (
            f'The gate must TAG, never reject — got an error dict: {result!r}'
        )
        mock_service.add_episode.assert_awaited_once()
        assert _service_kwargs(mock_service).get('unverified_claim') is True, (
            'memory_service.add_episode must be called with unverified_claim=True '
            f'so the tag reaches the Graphiti/Mem0 artefacts; got kwargs: '
            f'{_service_kwargs(mock_service)!r}'
        )

        flag = result.get('unverified_claim')
        assert isinstance(flag, dict), (
            f"Expected a structured 'unverified_claim' flag on the response, got: {result!r}"
        )
        assert flag.get('tag') == 'unverified_claim', f'Unexpected flag shape: {flag!r}'
        claims = flag.get('claims')
        assert isinstance(claims, list) and len(claims) == 1, (
            f'Expected exactly one flagged claim, got: {flag!r}'
        )
        entry = claims[0]
        assert entry.get('ref') == '5422', f'Flag must name the ref; got: {entry!r}'
        assert entry.get('subject') == 'task', f'Expected subject=task; got: {entry!r}'
        assert entry.get('kind') == 'applied_work', f'Expected kind=applied_work; got: {entry!r}'
        assert entry.get('project_id') == _PROJECT_ID, (
            f"Flag must name the claim's resolved project; got: {entry!r}"
        )
        assert entry.get('status') == 'mismatch', (
            f'A live non-terminal status CONTRADICTS the claim; got: {entry!r}'
        )
        assert entry.get('observed') == 'in-progress', (
            f'Flag must record the OBSERVED live status verbatim (INV-2); got: {entry!r}'
        )
        assert 'has been applied' in entry.get('text', ''), (
            f'Flag must quote the claiming clause; got: {entry!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'agent_id',
        [
            'recon-stage-task_knowledge_sync',
            'claude-task-5422-implementer',
            'claude-interactive',
            None,
        ],
    )
    async def test_gate_is_not_recon_stage_scoped(self, agent_id):
        """Unlike the 2824 premature-completion gate, this one is NOT wrapped in
        an `agent_id.startswith('recon-stage-')` guard: a false completion claim
        does the same corpus damage whoever writes it. Every writer — recon
        stage, task implementer, interactive, or an unset agent_id — gets the
        same tag.
        """
        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'in-progress'})

        args = {'content': _APPLIED_CONTENT, 'project_id': _PROJECT_ID}
        if agent_id is not None:
            args['agent_id'] = agent_id
        result = await server._tool_manager.call_tool('add_episode', args)

        assert 'error' not in result, (
            f'The gate must never reject (agent_id={agent_id!r}); got: {result!r}'
        )
        mock_service.add_episode.assert_awaited_once()
        assert _service_kwargs(mock_service).get('unverified_claim') is True, (
            f'Expected the tag for agent_id={agent_id!r}; got kwargs: '
            f'{_service_kwargs(mock_service)!r}'
        )
        assert result.get('unverified_claim', {}).get('claims'), (
            f'Expected a flag on the response for agent_id={agent_id!r}; got: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_status_read_is_scoped_to_the_claimed_project_root(self):
        """The status read must go to the claimed project's root from the
        known_projects registry, batched over the claimed ids — the same read
        shape _premature_completion_block uses.
        """
        mock_service = _episode_service()
        task_interceptor = MagicMock()
        task_interceptor.get_statuses = AsyncMock(return_value={'5422': 'in-progress'})
        task_interceptor.get_ticket_row = AsyncMock(return_value=None)
        server = create_mcp_server(
            mock_service,
            task_interceptor=task_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )

        await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        task_interceptor.get_statuses.assert_awaited_once()
        kwargs = task_interceptor.get_statuses.call_args.kwargs
        assert kwargs.get('project_root') == _KNOWN_PROJECTS[_PROJECT_ID], (
            f'Status read must target the claimed project root; got: {kwargs!r}'
        )
        assert list(kwargs.get('ids') or []) == ['5422'], (
            f'Status read must be batched over the claimed ids; got: {kwargs!r}'
        )


# The `tkt_` id from esc-3085-1 instance (2), verbatim.
_TICKET_ID = 'tkt_0RRRC5AASJ9Z630VP4PCN9H376'
_TICKET_CONTENT = f'the follow-up was filed as ticket {_TICKET_ID}'
# Ref but no completion phrasing, and completion phrasing is nowhere in it —
# the shape the gate must be entirely inert for.
_NO_CLAIM_CONTENT = 'task 5422 is pending review and the manifest gate is still open'

# Exactly the kwargs add_episode passed to the service BEFORE this gate existed.
_BASELINE_SERVICE_KWARGS = frozenset(
    {
        'content',
        'source',
        'project_id',
        'agent_id',
        'session_id',
        'source_description',
        'causation_id',
        'temporal_context',
        'reference_time',
        '_source',
    }
)


@pytest.fixture
def git_repo(tmp_path):
    """A throwaway repo with one commit; yields (root, sha)."""
    root = tmp_path / 'repo'
    root.mkdir()
    subprocess.run(['git', 'init', '-q', '.'], cwd=root, check=True)
    subprocess.run(
        ['git', '-c', 'user.email=a@b', '-c', 'user.name=a',
         'commit', '-q', '--allow-empty', '-m', 'x'],
        cwd=root, check=True,
    )
    sha = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return root, sha


def _gate_warnings(caplog) -> list[str]:
    """Every completion_claim_gate log line captured, at any level."""
    return [
        r.getMessage()
        for r in caplog.records
        if 'completion_claim_gate' in r.getMessage()
    ]


class TestVerifiedClaimsAreInert:
    """PRIMARY SIGNAL (second half): a claim the authority CONFIRMS must leave
    the write exactly as it was. A gate that tagged verified claims too would
    make the tag meaningless — the whole point is that its presence is a signal.
    """

    @pytest.mark.asyncio
    async def test_done_task_claim_is_not_tagged(self, caplog):
        """Task 5422 is live 'done' (terminal) — the claim is TRUE. No tag on
        the service call, no flag on the response, no warning logged.
        """
        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'done'})

        with caplog.at_level(logging.DEBUG):
            result = await server._tool_manager.call_tool(
                'add_episode',
                {
                    'content': _APPLIED_CONTENT,
                    'agent_id': 'claude-task-5422-implementer',
                    'project_id': _PROJECT_ID,
                },
            )

        mock_service.add_episode.assert_awaited_once()
        kwargs = _service_kwargs(mock_service)
        assert kwargs.get('unverified_claim', False) is False, (
            f'A verified claim must not tag the episode; got kwargs: {kwargs!r}'
        )
        assert 'unverified_claim' not in result, (
            f'A verified claim must leave no flag on the response; got: {result!r}'
        )
        assert _gate_warnings(caplog) == [], (
            f'A verified claim must log nothing; got: {_gate_warnings(caplog)!r}'
        )

    @pytest.mark.asyncio
    async def test_existing_commit_claim_is_not_tagged(self, caplog, git_repo):
        """A claim naming a commit that really exists in the claimed project's
        repository verifies against git and is not tagged.
        """
        root, sha = git_repo
        mock_service = _episode_service()
        server = _server(
            mock_service,
            statuses={},
            known_projects={_PROJECT_ID: str(root)},
        )

        with caplog.at_level(logging.DEBUG):
            result = await server._tool_manager.call_tool(
                'add_episode',
                {
                    'content': f'the de-flake fix landed in commit {sha}',
                    'agent_id': 'claude-task-5422-implementer',
                    'project_id': _PROJECT_ID,
                },
            )

        mock_service.add_episode.assert_awaited_once()
        assert _service_kwargs(mock_service).get('unverified_claim', False) is False, (
            f'An existing commit must not tag; got: {_service_kwargs(mock_service)!r}'
        )
        assert 'unverified_claim' not in result, f'Unexpected flag: {result!r}'
        assert _gate_warnings(caplog) == [], f'Unexpected logs: {_gate_warnings(caplog)!r}'

    @pytest.mark.asyncio
    async def test_resolving_ticket_claim_is_not_tagged(self, caplog):
        """A `tkt_` claim whose id resolves in the registry verifies by primary
        key — the row's own project is reported, and nothing is tagged.
        """
        mock_service = _episode_service()
        task_interceptor = MagicMock()
        task_interceptor.get_statuses = AsyncMock(return_value={})
        task_interceptor.get_ticket_row = AsyncMock(
            return_value={'ticket_id': _TICKET_ID, 'project_id': 'dark_factory',
                          'status': 'pending'},
        )
        server = create_mcp_server(
            mock_service,
            task_interceptor=task_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )

        with caplog.at_level(logging.DEBUG):
            result = await server._tool_manager.call_tool(
                'add_episode',
                {
                    'content': _TICKET_CONTENT,
                    'agent_id': 'claude-task-5422-implementer',
                    'project_id': _PROJECT_ID,
                },
            )

        task_interceptor.get_ticket_row.assert_awaited_once_with(_TICKET_ID)
        mock_service.add_episode.assert_awaited_once()
        assert _service_kwargs(mock_service).get('unverified_claim', False) is False, (
            f'A resolving ticket must not tag; got: {_service_kwargs(mock_service)!r}'
        )
        assert 'unverified_claim' not in result, f'Unexpected flag: {result!r}'
        assert _gate_warnings(caplog) == [], f'Unexpected logs: {_gate_warnings(caplog)!r}'


class TestNoClaimPathIsUntouched:
    """The control: content carrying no completion claim must not pay for this
    gate at all — no authority read, and a service call identical to the one the
    pre-gate code made.
    """

    @pytest.mark.asyncio
    async def test_no_claim_content_reads_nothing_and_passes_baseline_kwargs(
        self, caplog, monkeypatch
    ):
        import fused_memory.server.tools as tools_mod

        commit_probe_factory = MagicMock(
            side_effect=AssertionError('git must not be touched without a commit claim')
        )
        monkeypatch.setattr(tools_mod, 'make_commit_probe', commit_probe_factory)

        mock_service = _episode_service()
        task_interceptor = MagicMock()
        task_interceptor.get_statuses = AsyncMock(return_value={})
        task_interceptor.get_ticket_row = AsyncMock(return_value=None)
        server = create_mcp_server(
            mock_service,
            task_interceptor=task_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )

        with caplog.at_level(logging.DEBUG):
            result = await server._tool_manager.call_tool(
                'add_episode',
                {
                    'content': _NO_CLAIM_CONTENT,
                    'agent_id': 'claude-task-5422-implementer',
                    'project_id': _PROJECT_ID,
                },
            )

        task_interceptor.get_statuses.assert_not_awaited()
        task_interceptor.get_ticket_row.assert_not_awaited()
        commit_probe_factory.assert_not_called()
        mock_service.add_episode.assert_awaited_once()
        assert set(_service_kwargs(mock_service)) == set(_BASELINE_SERVICE_KWARGS), (
            'A no-claim write must reach the service with exactly the pre-gate '
            f'kwargs; got: {sorted(_service_kwargs(mock_service))!r}'
        )
        assert 'unverified_claim' not in result, f'Unexpected flag: {result!r}'
        assert _gate_warnings(caplog) == [], f'Unexpected logs: {_gate_warnings(caplog)!r}'


# A well-formed sha that no throwaway repo will contain.
_ABSENT_SHA = 'deadbee' + 'f' * 33


def _assert_tagged(result, mock_service, *, ref):
    """The whole fail-safe contract in one place: ingested, tagged, and the flag
    says WHAT WAS OBSERVED rather than merely that something went wrong.
    """
    assert 'error' not in result, f'The gate must never reject; got: {result!r}'
    mock_service.add_episode.assert_awaited_once()
    kwargs = _service_kwargs(mock_service)
    assert kwargs.get('unverified_claim') is True, (
        f'An unresolvable authority must TAG, not pass; got kwargs: {kwargs!r}'
    )
    claims = (result.get('unverified_claim') or {}).get('claims') or []
    assert [c for c in claims if c.get('ref') == ref], (
        f'Expected a flagged claim naming {ref!r}; got: {result!r}'
    )
    entry = next(c for c in claims if c.get('ref') == ref)
    assert entry.get('status') in {'mismatch', 'unverifiable'}, (
        f'A tagged claim is either contradicted or unchecked; got: {entry!r}'
    )
    assert entry.get('observed'), (
        f'The flag must name what was observed, never just that it failed; got: {entry!r}'
    )
    return entry


class TestUnresolvableAuthoritiesTag:
    """FAIL-SAFE CONTRACT, and the deliberate INVERSION of
    _premature_completion_block's fail-open.

    That gate rejects a write, so an infra hiccup must not bounce a legitimate
    one. This gate only labels, so its costs are asymmetric the other way: a
    spurious tag is one extra source_description prefix, while a missed tag is
    another batch of false Graphiti edges like esc-5603-1's five. Every
    unresolvable authority therefore lands on 'unverifiable' and is TAGGED.
    """

    @pytest.mark.asyncio
    async def test_task_interceptor_unconfigured_tags(self):
        """No task_interceptor at all → the live status cannot be read. The 2824
        gate fails OPEN here; this one tags.
        """
        mock_service = _episode_service()
        server = create_mcp_server(mock_service, known_projects=_KNOWN_PROJECTS)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        entry = _assert_tagged(result, mock_service, ref='5422')
        assert entry.get('status') == 'unverifiable', (
            f'An unreadable authority is unchecked, not contradicted; got: {entry!r}'
        )

    @pytest.mark.asyncio
    async def test_project_absent_from_known_projects_tags(self):
        """An empty registry puts the tool in permissive mode (the write is
        accepted) but leaves the claimed project's root unresolvable — so the
        status read cannot be scoped and the claim is tagged, never passed.
        """
        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'done'}, known_projects={})

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        entry = _assert_tagged(result, mock_service, ref='5422')
        assert entry.get('status') == 'unverifiable', f'Got: {entry!r}'

    @pytest.mark.asyncio
    async def test_get_statuses_raising_tags(self):
        mock_service = _episode_service()
        task_interceptor = MagicMock()
        task_interceptor.get_statuses = AsyncMock(side_effect=RuntimeError('taskmaster down'))
        task_interceptor.get_ticket_row = AsyncMock(return_value=None)
        server = create_mcp_server(
            mock_service,
            task_interceptor=task_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        entry = _assert_tagged(result, mock_service, ref='5422')
        assert entry.get('status') == 'unverifiable', f'Got: {entry!r}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'statuses',
        [
            pytest.param({'5422': 'unknown'}, id='unknown-sentinel'),
            pytest.param({}, id='absent-from-map'),
        ],
    )
    async def test_unknown_or_absent_status_tags(self, statuses):
        """`unknown` is get_statuses' documented sentinel for a NULL/absent DB
        status. It is not a live status, so it can neither confirm nor
        contradict — which under this gate's inverted fail direction means
        tagged, exactly where the 2824 gate deliberately passes.
        """
        mock_service = _episode_service()
        server = _server(mock_service, statuses=statuses)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        entry = _assert_tagged(result, mock_service, ref='5422')
        assert entry.get('status') == 'unverifiable', f'Got: {entry!r}'

    @pytest.mark.asyncio
    async def test_get_ticket_row_raising_tags(self):
        mock_service = _episode_service()
        task_interceptor = MagicMock()
        task_interceptor.get_statuses = AsyncMock(return_value={})
        task_interceptor.get_ticket_row = AsyncMock(side_effect=RuntimeError('tickets.db locked'))
        server = create_mcp_server(
            mock_service,
            task_interceptor=task_interceptor,
            known_projects=_KNOWN_PROJECTS,
        )

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _TICKET_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        entry = _assert_tagged(result, mock_service, ref=_TICKET_ID)
        assert entry.get('status') == 'unverifiable', (
            f'A registry that could not be consulted is UNVERIFIABLE, never a '
            f'false accusation that the ticket does not exist; got: {entry!r}'
        )

    @pytest.mark.asyncio
    async def test_ticket_store_unconfigured_tags(self):
        """The real get_ticket_row returns None (never raises) when no ticket
        store is configured. The tools layer cannot tell that from "no such
        ticket" — both are non-verified, so both tag.
        """
        mock_service = _episode_service()
        server = _server(mock_service, statuses={})

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _TICKET_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        _assert_tagged(result, mock_service, ref=_TICKET_ID)

    @pytest.mark.asyncio
    async def test_unresolvable_commit_probe_tags(self, tmp_path):
        """The claimed project's root is not a git repository, so commit
        existence cannot be answered at all → tagged as unverifiable, NOT
        reported as "the writer named a commit that does not exist".
        """
        mock_service = _episode_service()
        server = _server(
            mock_service,
            statuses={},
            known_projects={_PROJECT_ID: str(tmp_path)},
        )

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': f'the de-flake fix landed in commit {_ABSENT_SHA}',
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        entry = _assert_tagged(result, mock_service, ref=_ABSENT_SHA)
        assert entry.get('status') == 'unverifiable', f'Got: {entry!r}'

    @pytest.mark.asyncio
    async def test_gate_defect_never_breaks_the_write(self, caplog, monkeypatch):
        """The gate is advisory machinery bolted onto the write path. A defect
        INSIDE it must degrade to "no tag", never to a failed ingestion — the
        write is the thing the caller actually asked for.
        """
        import fused_memory.server.tools as tools_mod

        monkeypatch.setattr(
            tools_mod,
            'extract_completion_claims',
            MagicMock(side_effect=RuntimeError('gate exploded')),
        )
        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'in-progress'})

        with caplog.at_level(logging.DEBUG):
            result = await server._tool_manager.call_tool(
                'add_episode',
                {
                    'content': _APPLIED_CONTENT,
                    'agent_id': 'claude-task-5422-implementer',
                    'project_id': _PROJECT_ID,
                },
            )

        assert 'error' not in result, (
            f'A gate defect must not fail the write; got: {result!r}'
        )
        mock_service.add_episode.assert_awaited_once()
        assert 'unverified_claim' not in _service_kwargs(mock_service), (
            f'A gate that could not run tags nothing; got: {_service_kwargs(mock_service)!r}'
        )
        assert 'unverified_claim' not in result, f'Unexpected flag: {result!r}'
        assert [
            r for r in caplog.records
            if r.levelno >= logging.ERROR and 'completion_claim_gate' in r.getMessage()
        ], (
            'The swallowed defect must still be loud in the logs — a silent '
            f'except is how a gate quietly stops gating; got: {caplog.text!r}'
        )


# esc-3085-1 instance (2), verbatim: a reify-authored claim about a
# dark_factory ticket that did not exist.
_INSTANCE_2_CONTENT = (
    'reify task 5638 was reported unactionable and re-filed into '
    f"dark_factory's task tree as ticket {_TICKET_ID}"
)
_CROSS_PROJECTS = {'reify': '/reify-root', 'dark_factory': '/df-root'}


def _cross_project_server(mock_service, *, statuses=None, ticket_row=None):
    task_interceptor = MagicMock()
    task_interceptor.get_statuses = AsyncMock(return_value=statuses or {})
    task_interceptor.get_ticket_row = AsyncMock(return_value=ticket_row)
    server = create_mcp_server(
        mock_service,
        task_interceptor=task_interceptor,
        known_projects=_CROSS_PROJECTS,
    )
    return server, task_interceptor


class TestCrossProjectClaims:
    """esc-3085-1: the claim and the writer need not share a project, and
    checking the WRITER's registry would produce a confidently WRONG verdict in
    both directions — reporting a real dark_factory ticket as absent, and a
    dark_factory task's status as whatever reify's tree happens to say.
    """

    @pytest.mark.asyncio
    async def test_instance_2_absent_ticket_is_tagged(self):
        """The incident, end to end. A reify agent claims work was re-filed as a
        dark_factory ticket; the registry has no such ticket. Tagged, with the
        flag naming the absent id.
        """
        mock_service = _episode_service()
        server, task_interceptor = _cross_project_server(mock_service, ticket_row=None)

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _INSTANCE_2_CONTENT,
                'agent_id': 'claude-task-5638-implementer',
                'project_id': 'reify',
            },
        )

        entry = _assert_tagged(result, mock_service, ref=_TICKET_ID)
        assert entry.get('subject') == 'ticket', (
            f'The most specific ref wins — this is a ticket claim; got: {entry!r}'
        )
        assert entry.get('kind') == 'filing_dispatch', f'Got: {entry!r}'
        assert entry.get('status') == 'mismatch', (
            f'The registry ANSWERED, and it said no such ticket; got: {entry!r}'
        )
        assert _TICKET_ID in entry.get('observed', ''), (
            f'The flag must name the absent ticket; got: {entry!r}'
        )

        # The verdict must not depend on the writer's project in ANY way: a
        # globally unique PK lookup, and a claim that carries no project at all.
        task_interceptor.get_ticket_row.assert_awaited_once_with(_TICKET_ID)
        assert entry.get('project_id') is None, (
            f'A ticket claim resolves project-agnostically; got: {entry!r}'
        )

    @pytest.mark.asyncio
    async def test_qualified_task_claim_reads_the_claimed_projects_root(self):
        """"dark_factory task 3142 has landed", written by a reify agent. The
        status read must target dark_factory's root — reading reify's tree would
        answer a question nobody asked.
        """
        mock_service = _episode_service()
        server, task_interceptor = _cross_project_server(
            mock_service, statuses={'3142': 'in-progress'},
        )

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'dark_factory task 3142 has landed',
                'agent_id': 'claude-task-5638-implementer',
                'project_id': 'reify',
            },
        )

        task_interceptor.get_statuses.assert_awaited_once()
        kwargs = task_interceptor.get_statuses.call_args.kwargs
        assert kwargs.get('project_root') == '/df-root', (
            "The status read must target the CLAIMED project's root, not the "
            f'writer\'s; got: {kwargs!r}'
        )
        entry = _assert_tagged(result, mock_service, ref='3142')
        assert entry.get('project_id') == 'dark_factory', f'Got: {entry!r}'

    @pytest.mark.asyncio
    async def test_multi_project_episode_batches_one_read_per_project(self):
        """Two claimed projects → exactly two status reads, each batched over
        that project's own ids. One read per claim would multiply authority
        traffic by the claim count for no gain.
        """
        mock_service = _episode_service()
        server, task_interceptor = _cross_project_server(
            mock_service, statuses={'3142': 'in-progress', '5638': 'in-progress'},
        )

        await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': (
                    'dark_factory task 3142 has landed. reify task 5638 has landed'
                ),
                'agent_id': 'claude-task-5638-implementer',
                'project_id': 'reify',
            },
        )

        reads = {
            call.kwargs.get('project_root'): sorted(call.kwargs.get('ids') or [])
            for call in task_interceptor.get_statuses.await_args_list
        }
        assert reads == {'/df-root': ['3142'], '/reify-root': ['5638']}, (
            f'Expected one batched read per claimed project; got: {reads!r}'
        )

    @pytest.mark.asyncio
    async def test_claim_naming_an_unregistered_project_is_tagged(self):
        """A qualifier that is not in the registry falls back to the writer's
        project (an arbitrary preceding word is not a project name). If the
        WRITER's project is itself unregistered, no root resolves and the claim
        is tagged — never silently passed.
        """
        mock_service = _episode_service()
        server, task_interceptor = _cross_project_server(
            mock_service, statuses={'3142': 'done'},
        )

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': 'the merge task 3142 has landed',
                'agent_id': 'claude-task-5638-implementer',
                'project_id': 'reify',
            },
        )

        kwargs = task_interceptor.get_statuses.call_args.kwargs
        assert kwargs.get('project_root') == '/reify-root', (
            f"An unrecognised qualifier falls back to the writer's project; got: {kwargs!r}"
        )
        assert 'unverified_claim' not in result, (
            f'reify task 3142 is done, so the claim verifies; got: {result!r}'
        )


class TestTaggedIngestionFilesAnEscalation:
    """The tag labels the corpus; the escalation reaches an operator. Both, or
    the finding lives only in a log line nobody greps (INV-4).
    """

    @pytest.mark.asyncio
    async def test_tagged_ingestion_invokes_the_emitter_for_the_writers_root(
        self, monkeypatch
    ):
        import fused_memory.server.tools as tools_mod

        emitter = MagicMock(return_value='esc-unverified-claim-1')
        monkeypatch.setattr(tools_mod, 'emit_unverified_claim_escalation', emitter)

        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'in-progress'})

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        emitter.assert_called_once()
        args, _ = emitter.call_args
        assert args[0] == _KNOWN_PROJECTS[_PROJECT_ID], (
            f'the emitter must be given a filesystem root, not a project id: {args!r}'
        )
        assert (args[1].get('claims') or [])[0].get('ref') == '5422', (
            f'the emitter must receive the flag payload; got: {args!r}'
        )
        assert result['unverified_claim'].get('escalation_id') == 'esc-unverified-claim-1', (
            f'the filed id must be echoed onto the response flag; got: {result!r}'
        )

    @pytest.mark.asyncio
    async def test_verified_ingestion_files_nothing(self, monkeypatch):
        import fused_memory.server.tools as tools_mod

        emitter = MagicMock(return_value=None)
        monkeypatch.setattr(tools_mod, 'emit_unverified_claim_escalation', emitter)

        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'done'})

        await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        emitter.assert_not_called()

    @pytest.mark.asyncio
    async def test_emitter_raising_does_not_change_the_writes_outcome(self, monkeypatch):
        """The emitter is built never to raise — but a call site that RELIED on
        that promise would turn a future regression there into an outage here.
        Same reasoning as the markup gate's wrapping of its own emitter.
        """
        import fused_memory.server.tools as tools_mod

        emitter = MagicMock(side_effect=RuntimeError('escalation queue exploded'))
        monkeypatch.setattr(tools_mod, 'emit_unverified_claim_escalation', emitter)

        mock_service = _episode_service()
        server = _server(mock_service, statuses={'5422': 'in-progress'})

        result = await server._tool_manager.call_tool(
            'add_episode',
            {
                'content': _APPLIED_CONTENT,
                'agent_id': 'claude-task-5422-implementer',
                'project_id': _PROJECT_ID,
            },
        )

        entry = _assert_tagged(result, mock_service, ref='5422')
        assert entry.get('observed') == 'in-progress', f'Got: {entry!r}'
        assert 'escalation_id' not in result['unverified_claim'], (
            f'no id was filed, so none may be echoed; got: {result!r}'
        )
