"""The SINGLE construction site of a directly-submitted pending ``Escalation``.

Every test in ``escalation/tests/`` that puts a pending record into a queue by
bypassing the MCP tools — no ``escalate_*`` call, no wire, just
``queue.submit()`` — goes through here. Task 4997 folded the three near-twin
``_seed`` bodies (``test_capability_guard_http.py``,
``test_status_authority_gate.py``, ``test_server.py``'s
``TestGetPendingLevelFilter._seed_esc``) into this module (INV-5); each of the
three keeps a thin wrapper that supplies its own default summary and delegates
everything else, so the id minting, the severity/category defaults, the
``Escalation`` build and the ``submit`` all exist exactly once. Its pins live in
``test_escalation_seed_helper.py``.

WHY A SIBLING MODULE AND NOT A CONFTEST FIXTURE, and why not folded into
``_escalation_http.py``. ``conftest.py`` is for FIXTURES, and a fixture is
unreachable from the module-level helpers and class methods that call this. The
house convention for a non-fixture test helper is a uniquely-named,
underscore-prefixed sibling module (so pytest never collects it) —
``_fm_helpers.py``, ``_orch_helpers.py``, ``_dashboard_helpers.py`` — made
importable by the ``sys.path.insert(0, str(_TESTS_DIR))`` in the subproject
conftest. ``_escalation_http.py`` is the wrong home for the same reason it is a
good module: it is the CLIENT half of a two-way HTTP boundary contract, and
seeding is its exact opposite — it touches no wire at all, and ``test_server.py``
does no HTTP and should not import fastmcp's transport to call
``queue.submit()``.

THE DEFAULT SUMMARY IS THE ONE AXIS LEFT TO CALLERS, and that is deliberate.
Two of the three consumers share ONE module-scoped ``EscalationQueue`` across
every test in their module, so when a cross-test interference failure surfaces a
record, its summary is what names the module that seeded it. A single generic
default would erase that, which is why *summary* is an explicit parameter here
rather than a ``kw.setdefault``: it is the thing the wrappers exist to supply,
so a reader sees it in the signature. ``severity`` and ``category`` stay in
``**kw`` behind ``setdefault`` — unlike *summary* they are uniform across all
three consumers and vary only at individual call sites.

*task_id* DOUBLES AS THE ``queue.make_id`` ID-NAMESPACE KEY HERE, as a test-only
convenience. ``escalation/src/escalation/queue.py::make_id`` documents at length
that these are NOT the same thing in production — five production sites diverge
deliberately, and nothing may derive a task_id from a filename or an escalation
id. Do not read this module's convenience as an endorsement of that false
identity, and note the pin in ``test_escalation_seed_helper.py`` asserts only
that repeated seeds get DISTINCT ids, never an id format.
"""

from __future__ import annotations

from typing import Any

from escalation.models import Escalation
from escalation.queue import EscalationQueue


def seed_escalation(
    queue: EscalationQueue,
    *,
    level: int,
    task_id: str,
    agent_role: str = 'implementer',
    summary: str | None = None,
    **kw: Any,
) -> Escalation:
    """Submit a pending escalation at *level* directly via ``queue.submit()``.

    Bypasses the MCP tools entirely, so no dedupe guard, no capability header
    and no server is involved. Returns the submitted record, whose ``id`` is the
    handle every caller reads back through ``queue.get(...)``.

    *summary* defaults to a level-labelled string; the three consumer wrappers
    each override it with their own module's label (see the module docstring for
    why that matters). ``severity``/``category`` default to an innocuous
    ``'blocking'``/``'scope_violation'`` and, like any other ``Escalation``
    field, can be overridden through **kw.
    """
    if summary is None:
        summary = f'seeded test escalation (level={level})'
    kw.setdefault('severity', 'blocking')
    kw.setdefault('category', 'scope_violation')
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=agent_role,
        level=level,
        summary=summary,
        **kw,
    )
    queue.submit(esc)
    return esc
