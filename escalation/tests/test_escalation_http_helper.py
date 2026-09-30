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

import ast
from pathlib import Path

import pytest
from _escalation_http import capability_headers, escalation_http_call

from escalation.models import Escalation

# The capability headers whose construction must live in exactly one module.
_CAPABILITY_HEADER_NAMES = frozenset({'X-Escalation-Levels', 'X-Escalation-Identity'})

# Modules the INV-5 scan below skips, as paths relative to ``escalation/tests/``
# (for a top-level module that is just its basename).
#
# This module is exempt from its own scan -- not a loophole.
# ``TestCapabilityHeaders`` asserts exact equality against dict literals naming
# both headers (``== {'X-Escalation-Levels': '0,1'}``), which the scan cannot
# distinguish from construction. Those literals are the PIN, not a drift risk:
# they are the deliberate INDEPENDENT restatement of the wire contract.
# Rewriting them to reference the helper's own output to satisfy the scan would
# make the test follow a rename anywhere and go green on a break -- exactly the
# failure ``_escalation_http``'s docstring refuses when it keeps the header
# names as literals rather than importing the server constants.
#
# So the SET is the extension point, deliberately: a future module that also
# needs an independent restatement of the wire contract belongs here, with a
# comment saying why. It does not belong rewritten to assert against
# ``capability_headers``' own output -- that would couple the new pin to the
# implementation it exists to check. Every module that could carry a drifting
# SECOND COPY is still scanned.
_THIS_MODULE = Path(__file__).name
_EXEMPT_MODULES = frozenset({_THIS_MODULE})

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


# ---------------------------------------------------------------------------
# INV-5: exactly one module constructs the capability headers
# ---------------------------------------------------------------------------


def _constructs_capability_headers(source: str) -> bool:
    """True iff *source* NAMES a capability header anywhere but a READ subscript.

    The rule is deliberately BROAD: any ``ast.Constant`` whose value is exactly
    ``'X-Escalation-Levels'`` or ``'X-Escalation-Identity'`` counts, wherever it
    appears — with one structural exception, the constant SLICE of a
    Load-context ``ast.Subscript``, i.e. the
    ``_WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']`` read form.

    An earlier version matched only two construction SHAPES (an ``Assign`` onto
    a subscript with a constant slice, and a ``Dict`` with a constant key). That
    left the guard weaker than its own purpose, because a reintroduced second
    copy that binds the wire name to a local first::

        _LEVELS = 'X-Escalation-Levels'   # Name target, not a Subscript
        headers[_LEVELS] = levels         # Name slice, not a Constant

    matched neither shape and slipped through completely — as did
    ``headers.setdefault('X-Escalation-Levels', v)``. Since the ENTIRE value of
    this guard is that a *later* copy cannot go unnoticed, and aliasing is a
    natural thing to write when you think you are being tidy, matching on the
    name itself rather than on a catalogue of spellings is the only version that
    stays honest.

    Prose needs no special case, and that measurement is what makes an AST scan
    viable where a text grep is not. Comments never enter the AST at all; a
    docstring that DISCUSSES a header is a single Constant holding the whole
    paragraph, and a paragraph is not EQUAL to a bare header name. Measured
    across ``escalation/tests/``: 78 textual occurrences, but only 31 bare-name
    Constants — and outside this module and ``_escalation_http.py`` all 17 of
    those are read subscripts. Hence an exception list exactly one entry long.

    The breadth over-approximates on purpose. A legitimate READ spelled some
    other way — ``headers.get('X-Escalation-Identity')`` — would trip this and
    is not construction. That direction of error is the chosen one: a false
    positive is a loud failure closed by one documented ``_EXEMPT_MODULES``
    entry, whereas a false negative silently voids INV-5 and nobody finds out.
    """
    # Materialised so every node stays referenced while its ``id()`` is in use:
    # ids are unique only among LIVE objects.
    nodes = list(ast.walk(ast.parse(source)))
    read_slices = {
        id(node.slice)
        for node in nodes
        if isinstance(node, ast.Subscript)
        and isinstance(node.ctx, ast.Load)
        and isinstance(node.slice, ast.Constant)
    }
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in _CAPABILITY_HEADER_NAMES
        and id(node) not in read_slices
        for node in nodes
    )


class TestConstructionDetector:
    """Pin the detector itself — an unchecked guard is not a guard.

    ``_constructs_capability_headers`` is the whole mechanism behind INV-5, so
    its own true/false boundary is pinned here rather than left implied by the
    suite-wide scan below (which, on a healthy tree, passes whether or not the
    detector can see anything at all).
    """

    @pytest.mark.parametrize(
        'source',
        [
            pytest.param(
                "headers['X-Escalation-Levels'] = levels",
                id='subscript-assign',
            ),
            pytest.param(
                "headers = {'X-Escalation-Identity': identity}",
                id='dict-literal',
            ),
            pytest.param(
                "_LEVELS = 'X-Escalation-Levels'\nheaders[_LEVELS] = levels",
                id='aliased-to-a-local',
            ),
            pytest.param(
                "headers.setdefault('X-Escalation-Levels', levels)",
                id='setdefault',
            ),
            pytest.param(
                "send(headers={'X-Escalation-Identity': who})",
                id='inline-kwarg-dict',
            ),
        ],
    )
    def test_construction_spellings_are_detected(self, source: str) -> None:
        """Every way of naming the wire header to BUILD it counts as a site.

        The aliased and ``setdefault`` cells are the regression pins: both were
        invisible to the earlier shape-matching version, so a second copy
        written either way would have passed the suite-wide scan silently.
        """
        assert _constructs_capability_headers(source) is True

    @pytest.mark.parametrize(
        'source',
        [
            pytest.param(
                "expected = _WATCHER_ESCALATION_HEADERS['X-Escalation-Identity']",
                id='read-subscript',
            ),
            pytest.param(
                '"""Prose about the X-Escalation-Levels capability header."""',
                id='docstring-prose',
            ),
            pytest.param(
                '# X-Escalation-Levels mentioned in a comment\nx = 1',
                id='comment',
            ),
            pytest.param(
                "headers['X-Some-Other-Header'] = value",
                id='unrelated-header',
            ),
        ],
    )
    def test_reads_and_prose_are_not_construction(self, source: str) -> None:
        """Reading, discussing or mentioning a header is not constructing one.

        These are the false positives that would make the guard unusable: the
        read-subscript cell alone accounts for 17 of the 31 bare-name Constants
        in this suite, and prose accounts for the gap between 31 and the 78
        textual occurrences a grep would have had to triage.
        """
        assert _constructs_capability_headers(source) is False


def test_exactly_one_module_constructs_the_capability_headers() -> None:
    """INV-5: ``_escalation_http.py`` is the only construction site in this suite.

    The point of folding the two ``_call_over_http`` copies together is that the
    ``X-Escalation-Levels`` / ``X-Escalation-Identity`` wire protocol cannot
    drift between them. That property is only durable if it is asserted — a
    second copy reintroduced later would otherwise pass every existing test,
    since both copies would be individually correct on the day they were
    written. This is the assertion.
    """
    tests_dir = Path(__file__).parent
    sites: list[str] = []
    for path in sorted(tests_dir.rglob('*.py')):
        # rglob, not glob: ``escalation/tests/`` already has a subdirectory
        # (``fixtures/``), and this suite already keeps child-process modules
        # (``_concurrent_queue_child.py``) as plain siblings -- so a helper that
        # migrated one level down is a realistic way for a second construction
        # site to land somewhere a top-level-only scan would never look. The
        # assertion claims "this suite"; the scan now covers it.
        if '__pycache__' in path.parts:
            continue
        relpath = path.relative_to(tests_dir).as_posix()
        if relpath in _EXEMPT_MODULES:
            continue
        if _constructs_capability_headers(path.read_text(encoding='utf-8')):
            sites.append(relpath)

    assert set(sites) == {'_escalation_http.py'}, (
        f'Expected exactly one module under {tests_dir.name}/ to construct the '
        f'capability headers ({sorted(_CAPABILITY_HEADER_NAMES)}), found: {sites}.\n'
        'If that is a second COPY: put the construction in '
        '_escalation_http.capability_headers and call it (via '
        'escalation_http_call, or directly) instead of building the header dict '
        'locally -- that is the INV-5 property this task established.\n'
        'If it is instead an intentional INDEPENDENT restatement of the wire '
        'contract (as in this module\'s TestCapabilityHeaders, which asserts '
        'against literal header dicts on purpose), add it to _EXEMPT_MODULES '
        'with a comment saying why. Do NOT rewrite it to assert against '
        "capability_headers' own output: a pin that reads the wire name out of "
        'the code under test follows a rename and goes green on a break.'
    )
