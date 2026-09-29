"""The shared store-mutation-preflight test contract (task 4480).

Every `fused-memory/scripts/` module that mutates the shared store in-process
calls
`fused-memory/src/fused_memory/utils/store_mutation_preflight.py::assert_store_mutation_allowed`
before its first mutation AND before its scan (task 3686). Each of those
scripts' test suites then expresses the SAME four-part contract against it:
neutralise the guard for the suite at large, rig it to refuse for the
guard's own tests, pin the sentinel it refuses with, and pin the diagnosis it
logs on the way out. That scaffolding was hand-copied into 14 suites, so a
contract change -- adding a third pinned marker, say -- had to be applied in 14
places or it drifted. It lives here once instead, parameterised on the only two
axes that genuinely vary: the script module, and the script's logger name.

Lives outside conftest.py and is underscore-prefixed for the reasons
`_ast_guard.py` documents at length -- pytest must not collect it, and each
subproject exports its helpers under a unique module name to avoid the
`sys.modules['conftest']` collision when root-level pytest loads several
subprojects' conftests in one process. That rationale is inherited, not
restated.

WHY THE NEUTRALISING FIXTURE EXISTS AT ALL
------------------------------------------
The preflight's probe touches the REAL `~/.mem0`. Without an autouse fixture
neutralising it, every `--apply` test in every guarded suite would pass or fail
according to whether the machine running pytest happens to be able to write
mem0's history directory -- and inside an agent sandbox it genuinely cannot,
which is the whole reason the guard exists. These are deliberately MOCK-unit
suites; the environment must not be an input to them.

WHY NOT `raising=False`
-----------------------
`neutralise_fixture` does not pass `raising=False` to `monkeypatch.setattr`. If
the guard is ever removed from a script, the fixture must break loudly with an
`AttributeError` rather than silently no-op and leave that suite's `--apply`
tests asserting nothing.

WHY THE LOG MESSAGE'S CONTENT IS PINNED
---------------------------------------
`fail_closed_records` filters on the message CONTENT, which is the narrow and
deliberate exception to the repo's don't-pin-guard-message-prose norm (task
3799). The record these assertions are about is defined BY its content: in most
of the guarded scripts `main` has no handler at all, so the refusal exits as an
uncaught traceback and this ERROR record is the ONLY place the operator is told
what was refused and what to do instead. Where `main` DOES have a generic
handler, its own "fatal error during ..." record shares this logger and this
level, so neither the level nor the logger name can tell the two apart -- only
the markers can. Mere record-existence would still pass if the whole diagnosis
were replaced by "boom", precisely the regression these assertions exist to
catch. Verified non-vacuous: mutating the marker in the script turns the
assertions red (task 4127 amendment).

Pinned on those two clauses ONLY -- the fail-closed marker and the remedy noun
-- so every other word of the message stays free to reword.
"""

from __future__ import annotations

import logging

import pytest

# The message the rigged preflight refuses with. A single obviously-synthetic
# token, so a test that accidentally asserts against it reads as a test artefact
# rather than as a plausible real diagnosis.
SENTINEL = 'SENTINEL-store-unwritable'

# The clauses a guard site's own ERROR record must carry: the fail-closed
# marker, and the remedy noun that tells the operator where to route the
# mutation instead. Exported as a TUPLE that every consumer ITERATES rather
# than unpacking at a fixed arity, so adding a third clause is a one-line
# change here rather than a sweep across the 12 guarded suites:
# `fail_closed_records` below requires `all(...)` of it, and the contract
# module's probe messages compose themselves from it (the positive probe
# carries every clause; one exclusion case is generated per clause by
# omission).
#
# ONE deliberate exception, and it is not an oversight: drift guard #3 sweeps
# `FAIL_CLOSED_MARKERS[0]` ONLY. The fail-closed marker is a distinctive
# sentence that appears as a string constant nowhere else under `tests/`; the
# remedy noun is ordinary English and appears in 48 unrelated test modules
# (measured 2026-09-04), so sweeping it would buy 48 false positives rather
# than a guard. That guard's own comment carries the same reason.
FAIL_CLOSED_MARKERS = ('NOT started (fail-closed)', 'MCP server')

_SHARED_NEUTRALISE_RATIONALE = """Keep this MOCK-unit suite independent of the REAL ``~/.mem0``.

    The preflight probe touches the real filesystem, so without this fixture
    every ``--apply`` test would pass or fail according to whether the machine
    running pytest happens to be able to write mem0's history directory -- and
    it genuinely cannot inside an agent sandbox, which is the whole reason the
    guard exists. This suite is deliberately MOCK-unit, so the environment must
    not be an input to it.

    Deliberately NOT ``raising=False``: if the guard is ever removed from the
    script this fixture must break loudly rather than silently no-op.

    See ``_store_mutation_preflight_contract`` for the full rationale. This
    suite's own seam:"""


def neutralise_fixture(mod, *, note):
    """Build the autouse fixture that neutralises *mod*'s preflight guard.

    *mod* is the script module under test -- the object the suite loaded and
    whose `assert_store_mutation_allowed` attribute is to be replaced.

    *note* is REQUIRED and keyword-only, carrying the two facts the shared
    rationale above cannot: this suite's seam (which entry point runs the
    preflight, before which phase, and under which originating task) and its
    mock substrate. It is composed into the generated fixture's `__doc__`, so it
    travels with the object and shows up in `pytest --fixtures -v` rather than
    drifting away as a comment. The `-v` is load-bearing: pytest hides
    underscore-prefixed fixtures from a bare `--fixtures`, and the suites bind
    this one as `_neutralise`, so plain `--fixtures` prints the
    `fixtures defined from _store_mutation_preflight_contract` section header
    with nothing under it. Making it required is the forcing function: a
    conversion that forgets to carry a suite's per-script rationale forward is a
    `TypeError` at collection time, not a silent prose deletion.
    """

    def _neutralise(monkeypatch):
        # Deliberately no `raising=False` -- see the module docstring.
        monkeypatch.setattr(mod, 'assert_store_mutation_allowed', lambda **_kw: None)

    _neutralise.__doc__ = f'{_SHARED_NEUTRALISE_RATIONALE}\n\n    {note}\n    '
    return pytest.fixture(autouse=True)(_neutralise)


def deny(mod, monkeypatch):
    """Rig *mod*'s preflight to refuse, as it would inside an agent sandbox.

    Raises the module's OWN `StoreMutationUnavailable`, so the exception the
    script's handlers see is the real type rather than a stand-in, and carries
    `SENTINEL` as its message. Accepts arbitrary positional and keyword
    arguments: call sites invoke the guard as `assert_store_mutation_allowed(
    operation=...)`, but the raiser must not constrain the signature it stands
    in for.
    """

    def _raise(*_args, **_kwargs):
        raise mod.StoreMutationUnavailable(SENTINEL)

    monkeypatch.setattr(mod, 'assert_store_mutation_allowed', _raise)


def fail_closed_records(caplog, logger_name):
    """The records that are a guard site's OWN fail-closed diagnosis.

    Filters *caplog* down to records from *logger_name* at ERROR or above whose
    message carries every one of `FAIL_CLOSED_MARKERS`. *logger_name* is passed
    rather than derived, because two of the guarded scripts use a logger name
    that is NOT their module name (`clear_false_dependency_invalidations` logs
    as `clear_false_dep_invalidations`; `invalidate_fabricated_shipping_edges`
    logs as `invalidate_shipping_edges`) -- deriving it would silently match
    nothing and make every assertion vacuous.

    The level test is `levelno >= logging.ERROR` rather than
    `levelname == 'ERROR'`. Both spellings were in use before this helper
    existed, and the superset is chosen deliberately: every call site is a
    POSITIVE assertion (`assert fail_closed_records(...), ...`) and none asserts
    emptiness, so a broader predicate can only add matches and can never turn a
    green assertion red. The two are extensionally equal on today's tree anyway,
    since every guard site logs at exactly `logger.error`.
    """
    return [
        rec for rec in caplog.records
        if rec.name == logger_name
        and rec.levelno >= logging.ERROR
        and all(marker in rec.getMessage() for marker in FAIL_CLOSED_MARKERS)
    ]
