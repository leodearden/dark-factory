"""Contract tests for the shared escalation HTTP call helper (``_escalation_http``).

``_escalation_http`` is the SINGLE construction site of the
``X-Escalation-Levels`` / ``X-Escalation-Identity`` capability headers across
``escalation/tests/`` (INV-5; task 3736 deduped the server-lifecycle half into
``conftest.py``, task 4345 this call half). Because every other module reaches
the wire protocol only through it, a silent regression here would flip the
meaning of ~25 capability-guard tests and the C1-C4 status-authority cells at
once while they all stayed green — so the helper gets its own pins.

Two kinds of test live here, and the split is deliberate:

* PURE SEAM (``TestCapabilityHeaders``) — ``capability_headers`` in isolation.
  These pin the ``None``-omits / ``''``-is-sent distinction, which is the half a
  naive ``if levels:`` rewrite silently breaks.
* BEHAVIOURAL WIRING (``TestHeaderReachesServer``) — one end-to-end drive
  through a REAL server, so a ``capability_headers`` that is correct in
  isolation but whose result never reaches the transport cannot pass.

There is deliberately NO ``levels``-forwarding behavioural test here:
``test_capability_guard_http.py`` drives ~25 tests' worth of level scenarios
through this same helper, and ``test_status_authority_gate.py``'s C1-C4 cells
drive the rest. A copy here would reintroduce exactly the lockstep duplication
task 4345 removes.
"""

from __future__ import annotations

import pytest
from _escalation_http import capability_headers, escalation_http_call

from escalation.models import Escalation

# ---------------------------------------------------------------------------
# Pure seam: capability_headers
# ---------------------------------------------------------------------------


class TestCapabilityHeaders:
    """``capability_headers`` — the one place the wire header names are built.

    Every assertion below is on RUNTIME BEHAVIOUR (the returned mapping), never
    on the function's source or signature, so the helper stays free to be
    rewritten as long as the wire contract holds.
    """

    def test_no_args_builds_no_headers(self) -> None:
        """A header-less call sends NO capability headers at all.

        Asserted as exact ``== {}`` rather than a falsiness check: the
        header-less path must exercise the server's genuine default-open branch,
        which it would not if either key were present carrying a falsy value.
        """
        assert capability_headers() == {}

    def test_levels_only_omits_identity_key(self) -> None:
        """*levels* alone sends only ``X-Escalation-Levels``."""
        headers = capability_headers(levels='0,1')
        assert headers == {'X-Escalation-Levels': '0,1'}
        assert 'X-Escalation-Identity' not in headers

    def test_identity_only_omits_levels_key(self) -> None:
        """*identity* alone sends only ``X-Escalation-Identity``."""
        headers = capability_headers(identity='agent:auto-watcher')
        assert headers == {'X-Escalation-Identity': 'agent:auto-watcher'}
        assert 'X-Escalation-Levels' not in headers

    def test_both_set_sends_both_verbatim(self) -> None:
        """Both set -> both keys present, both values sent verbatim."""
        assert capability_headers(levels='0,1,2', identity='agent:steward') == {
            'X-Escalation-Levels': '0,1,2',
            'X-Escalation-Identity': 'agent:steward',
        }

    @pytest.mark.parametrize(
        ('kwargs', 'expected'),
        [
            ({'levels': ''}, {'X-Escalation-Levels': ''}),
            ({'identity': ''}, {'X-Escalation-Identity': ''}),
            ({'levels': '', 'identity': ''}, {
                'X-Escalation-Levels': '',
                'X-Escalation-Identity': '',
            }),
        ],
        ids=['levels', 'identity', 'both'],
    )
    def test_empty_string_is_sent_not_omitted(
        self, kwargs: dict[str, str], expected: dict[str, str],
    ) -> None:
        """An explicitly-EMPTY header is a sendable value, distinct from omission.

        This is the other half of the contract and the one most at risk: the
        gate is ``is not None``, so a "tidy-up" to ``if levels:`` would silently
        collapse ``''`` (send an empty header) into ``None`` (send nothing).
        Those are different requests on the wire and the server distinguishes
        them — ``server.py::stamp_triage`` does ``if identity is not None``, so
        an empty identity header stamps ``triaged_by == ''`` while an omitted
        one leaves the tool arg intact.
        """
        assert capability_headers(**kwargs) == expected


# ---------------------------------------------------------------------------
# Behavioural wiring: the headers actually reach a running server
# ---------------------------------------------------------------------------


class TestHeaderReachesServer:
    """The built headers reach the transport and are read by a REAL server.

    ``stamp_triage`` is the probe tool specifically because it is UNGATED by
    ``X-Escalation-Levels`` (a triage-ack annotation, not a state transition),
    so identity forwarding is observable here without entangling the level
    gate. Its ``triaged_by`` attribution makes the contract a visible PRODUCT
    OUTCOME rather than a claim about a mock's call args:
    ``server.py::stamp_triage`` reads ``get_http_headers().get(...)`` and
    overrides *triaged_by* only ``if identity is not None``.
    """

    @pytest.fixture
    def seeded(self, tmp_path, serve_escalation_mcp):
        """A running server plus one pending escalation to annotate.

        FUNCTION-scoped (``serve_escalation_mcp``, not the ``_module`` variant),
        per that fixture's own docstring: these tests are function-scoped, so
        the module variant would only defer teardown and misattribute the
        bounded-join hung-thread assert to the module rather than to the test
        that actually hung.
        """
        base_url, _port, queue = serve_escalation_mcp(tmp_path / 'esc')

        def _seed(task_id: str) -> Escalation:
            esc = Escalation(
                id=queue.make_id(task_id),
                task_id=task_id,
                agent_role='implementer',
                level=0,
                severity='blocking',
                category='scope_violation',
                summary='capability-header wiring probe (task 4345)',
            )
            queue.submit(esc)
            return esc

        return base_url, queue, _seed

    @pytest.mark.asyncio
    async def test_identity_header_reaches_server_and_overrides_tool_arg(
        self, seeded,
    ) -> None:
        """``identity='...'`` arrives as a real request header, server-side.

        Proves the built mapping is actually handed to the transport: the
        server can only override the spoofed *triaged_by* arg if it saw the
        header on the wire.
        """
        base_url, queue, seed = seeded
        esc = seed('probe-4345-identity')

        result = await escalation_http_call(
            base_url,
            'stamp_triage',
            identity='agent:probe-4345',
            escalation_id=esc.id,
            triaged_by='spoofed',
        )

        assert 'error' not in result, f'Unexpected error: {result}'
        assert result['triaged_by'] == 'agent:probe-4345', (
            f'Expected the identity header to reach the server and win over the '
            f"spoofed tool arg, got: {result['triaged_by']!r}"
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.triaged_by == 'agent:probe-4345'

    @pytest.mark.asyncio
    async def test_identity_none_omits_header_leaving_tool_arg_intact(
        self, seeded,
    ) -> None:
        """``identity=None`` sends NO header — never an empty one.

        The load-bearing assertion of this module. ``stamp_triage`` overrides
        *triaged_by* ``if identity is not None``, so a helper that sent ``''``
        instead of omitting the header would stamp ``triaged_by == ''`` here.
        That makes "omitted entirely, never an empty string" directly OBSERVABLE
        end-to-end instead of mock-asserted.
        """
        base_url, queue, seed = seeded
        esc = seed('probe-4345-omitted')

        result = await escalation_http_call(
            base_url,
            'stamp_triage',
            identity=None,
            escalation_id=esc.id,
            triaged_by='from-tool-arg',
        )

        assert 'error' not in result, f'Unexpected error: {result}'
        assert result['triaged_by'] == 'from-tool-arg', (
            f'Expected the tool arg to survive an OMITTED identity header; '
            f"got {result['triaged_by']!r} (=='' means the header was sent empty "
            f'rather than omitted)'
        )
        reread = queue.get(esc.id)
        assert reread is not None
        assert reread.triaged_by == 'from-tool-arg'
