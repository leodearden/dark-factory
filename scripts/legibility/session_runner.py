#!/usr/bin/env python3
"""scripts/legibility/session_runner.py — the legibility invocation boundary.

Every LLM call the legibility trickle and census make crosses this boundary,
and the exceptions here are what crossing it can raise: ``InvocationFailed``
for an invocation that produced no usable reply, and its subclass
``NoHeadroom`` for one that no pool account could take.

THE PACKAGE SPELLING, ``from legibility import session_runner``, is the only
way consumers reach this module, and that is what makes the exception classes
unique. scripts/legibility/ sits on sys.path beside scripts/, so census's bare
``import coder`` and nightly's ``from legibility import coder`` build two coder
module objects. Defined here rather than in coder, the classes exist once
however coder itself was imported, so ``coder.code_digest`` catches exactly
what the invoker raises (task 6042).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute -- same reasoning and same form as coder.py
# and census.py (tasks 2881/2882/3329). An editable install puts the MAIN
# checkout's shared/src on sys.path for a bare `python3`, so without this a
# copy running from a worktree would invoke through the MAIN checkout's runner.
_SHARED_SRC = Path(__file__).resolve().parents[2] / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))


class InvocationFailed(Exception):
    """An invocation produced no usable reply — the CLI failed, timed out,
    or never started. Never silently swallowed: ``coder.code_digest`` turns
    it into a per-digest failure, never a fabricated record.

    The message carries a tail of BOTH output streams, each labelled, because
    of what happened on 2026-08-24: the claude CLI wrote its usage-cap banner
    to STDOUT and exited 1, and the error of the day embedded only stderr.
    With stderr empty, the reason that reached the journal, the escalation and
    ``run.failures`` was ``claude CLI exited 1 (model='haiku', ...): `` —
    nothing after the colon — on 17 of 20 digests. A diagnostic the process
    EMITTED must never be dropped because it arrived on the less-expected
    stream.

    The two tails are ALSO carried as structured ``stdout``/``stderr``
    attributes, defaulting to ``''`` for the arms that have no streams.
    """

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class NoHeadroom(InvocationFailed):
    """No pool account could take this invocation; deferral-eligible.

    A SUBCLASS, not a sibling, and that is load-bearing: every site that
    catches ``InvocationFailed`` keeps catching this, so a missing headroom
    can never escape as an uncaught crash that takes down a whole batch.
    ``coder.code_digest`` catches it FIRST and labels the digest ``capped``.

    **This is a NORMAL operating condition, never a defect.** Leo's standing
    directive (task 4503): an all-accounts-capped night is expected weather.
    Before the label existed, 2026-08-24 presented as 17 of 20 hard per-digest
    failures, tripped the >50% storm threshold, and became an ERROR-level
    escalation for a condition ruled routine. Which exhaustions qualify — and
    which, like an all-auth-failed pool, must stay a loud ``InvocationFailed``
    — is task 5947's table.

    ``marker`` names the signal that fired, so a deferral reason can say WHICH.
    Never fabricated into a verdict: a capped digest yields no record at all.
    """

    def __init__(
        self, message: str, *, marker: str, stdout: str = "", stderr: str = "",
    ) -> None:
        super().__init__(message, stdout=stdout, stderr=stderr)
        self.marker = marker
