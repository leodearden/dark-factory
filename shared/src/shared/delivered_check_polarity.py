"""shared.delivered_check_polarity — authoring-time validation of
``metadata.delivered_checks`` (task 3500).

WHAT THIS EXISTS TO PREVENT. A ``delivered_check`` is a dep-gate: the
scheduler refuses to dispatch a dependent until the producer's check
reports DELIVERED. A check that can NEVER report delivered therefore
wedges its dependent forever, and — the task's opening complaint — a
malformed check is indistinguishable at runtime from a genuinely
undelivered capability, so the wedge reads as normal gating. The
measured specimen: task 5799 authored ``expect=present`` checks for a
pattern its own diff was scoped to REMOVE, and 5919 sat blocked behind
them.

THE ONE RULE. A sound delivered_check must FAIL at the authoring tree
and PASS after its producer lands (a 0→N or N→0 transition across the
task). At authoring time the reference tree is free — the task has not
been implemented yet, so HEAD *is* the pre-task tree — which collapses
the whole classification into a single measured predicate:

    ================  ===============  ==============
                      expect: present  expect: absent
    ================  ===============  ==============
    matches HEAD      REJECT           healthy
    no match at HEAD  healthy          REJECT
    ================  ===============  ==============

That subsumes the three fold-in modes (esc-4545-2) without parsing
English or guessing at author intent from the check's NAME:

  - MODE 1 — polarity inversion (5799): a pattern the task will remove
    is *necessarily present* at authoring time (that is what makes it
    removable), so the ``expect=present`` check already passes → the
    vacuity rule rejects it.
  - MODE 2 — over-broad ``expect=absent`` (task 3534's pattern matched
    inside the very file it owned). Genuinely undecidable at authoring
    time, so it lands as a WARN, not a reject.
  - MODE 3 — a ``kind=grep`` ``expect=present`` pattern whose only repo
    matches are FILENAMES rather than file contents (a test module that
    never mentions its own name). Invisible to the vacuity rule (at
    authoring time the file does not exist yet, so nothing matches and
    the check looks healthy), so it gets its own structural rule.

Comment-only and self-referential matches are not separate gates: at
authoring time they are sub-species of "a check that already matches",
so they sharpen the REJECTION MESSAGE rather than adding a predicate.

WHY THE REFERENCE TREE IS ``HEAD``, NOT A HISTORICAL SHA. The 2x2's axis
is "matches at the AUTHORING tree", and at authoring time that tree is
free: ``commit_planning``/stamping run BEFORE the task is implemented, so
whatever HEAD points at *is* the pre-task tree. No history, no
commit-ordering premise, no ``done``-time SHA to recover. That is the
whole reason a gate this cheap is sound, and it does not generalize
backwards: for an ALREADY-LANDED task the same descriptor's verdict
inverts — an ``expect=present`` check that matches is the SUCCESS state
of a landed producer, not vacuity — and recovering the tree it was
authored against needs exactly the commit-ordering premise this gate
avoids. Re-measured for task 3500: running the 2x2 with the reference
tree set to main-today flags 313/548 checked-in descriptors (57%),
overwhelmingly false positives of that shape. Hence the retroactive
corpus sweep is a separate, STATUS-AWARE tool
(``scripts/audit_delivered_checks.py``) that reads task status from
``tasks.db`` to tell "defect on a done task" from "healthy
forward-looking check on a pending one", rather than reusing this
predicate. Filed as esc-3500-1.

THE PARITY CONTRACT. :func:`build_grep_argv` and :func:`interpret_grep_rc`
are the SINGLE SOURCE OF TRUTH for grep-check semantics.
``orchestrator.delivered_checks._run_grep_check`` delegates to them, and
that delegation is the point: the authoring gate and the runtime gate
must agree exactly, or this module becomes a new source of the very
defect it prevents. A check the lint judges healthy but the runtime later
fails still wedges a dependent; a check the lint rejects that the runtime
would have accepted blocks legitimate planning. The semantics are
non-obvious and easy to diverge on:

  - ``git grep -E`` is POSIX EXTENDED regex — not Python ``re``, not PCRE.
  - The explicit ``-e`` separator exists so a pattern beginning with
    ``'-'`` is passed as the literal search pattern instead of being
    parsed by ``git grep`` as another option.
  - ``paths`` are git PATHSPECS, appended after a literal ``--`` only
    when non-empty.
  - ``rc >= 2`` is a git ERROR, not a no-match, and is its own outcome:
    an unresolvable *ref* is the common one (rc 128). Measured on git
    2.43.0 against every form this module builds: a PATHSPEC matching
    nothing in the tree exits **1**, a clean no-match. That distinction
    is load-bearing here — it means a forward-looking check whose
    ``paths`` name a file the task has not created yet reads as an
    ordinary healthy FAIL at the authoring tree, not as a reported
    ERRORED, so the gate stays quiet on the majority case.

One builder plus one interpreter makes divergence structurally
impossible.

IMPORT-LIGHT BY DESIGN. stdlib only — no pydantic, no ``fused_memory``,
no ``orchestrator``. ``shared/`` is the only package both fused-memory
and orchestrator already depend on (fused-memory must not import
orchestrator), so this is the sole placement that lets the authoring-time
and runtime call sites share code at all.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

__all__ = [
    'CheckFinding',
    'CheckOutcome',
    'build_grep_argv',
    'evaluate_grep_at_tree',
    'interpret_grep_rc',
    'lint_delivered_checks',
]

#: Wall-clock ceiling for ONE authoring-time ``git grep``. Generous
#: relative to a real grep (milliseconds on this repo) because exceeding it
#: is not a verdict — it degrades to ``ERRORED``, which is REPORTED rather
#: than blocking, so a slow disk delays a commit_planning call instead of
#: rejecting a healthy batch.
GREP_TIMEOUT_SECS: float = 30.0


class CheckOutcome(Enum):
    """Three-valued verdict for one evaluated check.

    Deliberately NOT ``orchestrator.delivered_checks.DeliveredCheckResult``:
    that enum's ``DELIVERED``/``FAILED`` vocabulary describes a
    *capability* at merge time, whereas this one describes whether a
    *predicate* held against some tree — the same three outcomes seen
    from the lint's side. ``_run_grep_check`` maps between them.

    ``ERRORED`` is its own disposition, never folded into either verdict:
    an unevaluable check is REPORTED (fail-open on infrastructure), never
    silently passed and never rejected.
    """

    PASS = 'pass'
    FAIL = 'fail'
    ERRORED = 'errored'


def build_grep_argv(
    pattern: str,
    paths: Sequence[str] | None,
    *,
    project_root: str | Path,
    ref: str,
) -> list[str]:
    """``['git','-C',<root>,'grep','-E','-e',<pattern>,<ref>[,'--',*paths]]``.

    The exact argv ``orchestrator.delivered_checks._run_grep_check`` has
    always built (it now calls this). See the module docstring's PARITY
    CONTRACT for why each element is load-bearing; in particular the
    ``-e`` separator is what keeps a pattern beginning with ``'-'`` from
    being parsed as a ``git grep`` option, and ``--`` is appended ONLY
    for a non-empty pathspec (an empty ``-- `` would be a pathspec
    matching nothing, turning every check into an rc>=2 ERRORED).

    ``paths`` accepts ``None`` as well as ``[]``: ``DeliveredCheckMeta``
    defaults it to ``[]``, but a raw metadata dict reaching the lint may
    carry ``None``, and both mean "no pathspec".
    """
    argv = ['git', '-C', str(project_root), 'grep', '-E', '-e', pattern, ref]
    if paths:
        argv.append('--')
        argv.extend(paths)
    return argv


def interpret_grep_rc(rc: int, expect: str | None) -> CheckOutcome:
    """Map a ``git grep`` exit code + expected polarity to a verdict.

    rc==0 (match) and rc==1 (no match) are both valid outcomes; rc>=2 is
    a git error. Which of match/no-match is a PASS depends on *expect*:
    ``'present'`` wants a match, ``'absent'`` wants no match.

    Any non-``'present'`` *expect* — including ``None`` — takes the
    absent arm. That is not a choice made here: it is
    ``_run_grep_check``'s pre-existing ``matched if expect == 'present'
    else not matched`` else-branch, preserved bit-for-bit so the
    delegation refactor is behaviour-preserving.
    """
    if rc >= 2:
        return CheckOutcome.ERRORED
    matched = rc == 0
    holds = matched if expect == 'present' else not matched
    return CheckOutcome.PASS if holds else CheckOutcome.FAIL


# ---------------------------------------------------------------------------
# The non-vacuity rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckFinding:
    """One thing the lint has to say about one delivered_check.

    *severity* is the disposition, and the three values are NOT a
    ranking of confidence — they are three different contracts:

    ``'reject'``
        A measured defect. The check is green at the authoring tree, so
        landing its producer cannot change its verdict and it gates
        nothing. ``commit_planning`` refuses the batch; the stamper
        refuses to copy the check.
    ``'warn'``
        Reported, never blocking. Reserved for classes that are genuinely
        undecidable at authoring time (an over-broad ``expect='absent'``
        pattern — task 3534's legitimately matched inside the very file it
        owned), where a reject would be a false positive on a hard gate.
    ``'errored'``
        The check could not be EVALUATED — git unavailable, root not a
        repo, ref unresolvable. Fail open on infrastructure: reported
        loudly, never silently passed, and never rejected (see the
        module docstring's IMPORT-LIGHT / fail-open note and
        :func:`evaluate_grep_at_tree`).

    *detail* carries the evidence a reader needs to act — sample match
    lines, offending paths — as a tuple so the finding stays hashable and
    frozen.
    """

    check_name: str
    severity: Literal['reject', 'warn', 'errored']
    code: str
    message: str
    detail: tuple[str, ...] = ()


def evaluate_grep_at_tree(
    pattern: str,
    paths: Sequence[str] | None,
    *,
    expect: str | None,
    repo_root: str | Path,
    ref: str = 'HEAD',
    timeout_secs: float = GREP_TIMEOUT_SECS,
) -> CheckOutcome:
    """Run one grep check against *ref* in *repo_root*. Never raises.

    ``build_grep_argv`` + ``subprocess.run`` + ``interpret_grep_rc`` and
    nothing else — which is what makes the verdict reached here provably
    the verdict ``orchestrator.delivered_checks._run_grep_check`` reaches
    against the same tree (module docstring, PARITY CONTRACT). This is
    the synchronous twin of that async runner: the authoring call sites
    (``commit_planning``, the capability-manifest stamper) are not on an
    event loop that would welcome one more await, and a grep against a
    committed tree is milliseconds.

    EVERY failure to evaluate collapses to :attr:`CheckOutcome.ERRORED`,
    never an exception and never a verdict:

    * ``OSError`` — ``git`` not on ``PATH`` (``FileNotFoundError``), or
      the exec itself failing.
    * ``subprocess.SubprocessError`` — chiefly ``TimeoutExpired``.
    * ``rc >= 2`` — git ran and reported an error (rc 128 for a
      non-repo *repo_root*, a missing directory, or an unresolvable
      *ref*); :func:`interpret_grep_rc` already maps that to ERRORED.

    Returning ERRORED rather than raising is the load-bearing half of
    "fail closed on a verdict, fail open on infrastructure": an
    availability failure must not be able to halt planning, but it must
    not read as a clean bill of health either.
    """
    argv = build_grep_argv(pattern, paths, project_root=repo_root, ref=ref)
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_secs
        )
    except (OSError, subprocess.SubprocessError):
        return CheckOutcome.ERRORED
    return interpret_grep_rc(completed.returncode, expect)


#: The invariant, in the author's terms. Appended to every rejection so the
#: message says what a SOUND check looks like, not merely that this one is
#: bad — a hard reject's message is the author's only feedback.
_INVARIANT = (
    'A sound delivered_check FAILS at the authoring tree and PASSES once its '
    'producer lands (a 0->N or N->0 transition across the task).'
)


def _vacuous_present_message(name: str, pattern: str, ref: str) -> str:
    return (
        f'delivered_check {name!r} (expect=present, pattern {pattern!r}) ALREADY '
        f'matches the authoring tree at {ref}, so landing this task cannot change '
        f'its verdict: it is green the day it is written and therefore gates '
        f'nothing. {_INVARIANT} Assert a symbol this task will INTRODUCE — or, if '
        f'this task REMOVES the pattern, flip the check to expect=absent.'
    )


def _vacuous_absent_message(name: str, pattern: str, ref: str) -> str:
    return (
        f'delivered_check {name!r} (expect=absent, pattern {pattern!r}) ALREADY has '
        f'no match at the authoring tree at {ref}: there is nothing left for this '
        f'task to remove, so the check is green the day it is written and therefore '
        f'gates nothing. {_INVARIANT} Assert a pattern this task will actually '
        f'REMOVE — or, if this task ADDS it, flip the check to expect=present.'
    )


def _unevaluable_message(name: str, ref: str, repo_root: str | Path) -> str:
    return (
        f'delivered_check {name!r} could not be EVALUATED against {ref} in '
        f'{repo_root} (git errored, timed out, or is unavailable; a non-repo root '
        f'lands here too). Reported as unvalidated rather than accepted or '
        f'rejected: an infrastructure failure must not block planning, but it must '
        f'not pass as a clean bill of health either.'
    )


def lint_delivered_checks(
    checks: Iterable[object],
    *,
    files: Sequence[str] | None,
    repo_root: str | Path,
    ref: str = 'HEAD',
) -> list[CheckFinding]:
    """Lint a whole ``metadata.delivered_checks`` list against the authoring tree.

    Returns one :class:`CheckFinding` per offending check (at most one per
    check) and an empty list for a clean batch. *files* is the task's
    declared ``metadata.files``, used by the refinements to tell a match
    inside the task's own scope from one outside it.

    Only ``kind == 'grep'`` is evaluated: the 2x2 is a statement about grep
    POLARITY, and a script check has no ``expect`` to invert. That
    short-circuit is deliberately BEFORE any subprocess, so a script-only
    batch costs nothing.
    """
    findings: list[CheckFinding] = []
    for check in checks:
        finding = _lint_one_check(check, files=files, repo_root=repo_root, ref=ref)
        if finding is not None:
            findings.append(finding)
    return findings


def _lint_one_check(
    check: Any,
    *,
    files: Sequence[str] | None,
    repo_root: str | Path,
    ref: str,
) -> CheckFinding | None:
    """The 2x2 for one check. ``None`` means healthy (or not our business).

    *check* is typed ``Any`` because it is a RAW metadata dict off the
    wire, not a validated ``DeliveredCheckMeta`` — the lint runs at
    authoring time precisely to catch entries a schema check cannot.
    """
    if check.get('kind') != 'grep':
        return None
    name = check.get('name')
    pattern = check.get('pattern')
    expect = check.get('expect')
    paths = check.get('paths')

    outcome = evaluate_grep_at_tree(
        pattern, paths, expect=expect, repo_root=repo_root, ref=ref
    )

    if outcome is CheckOutcome.ERRORED:
        return CheckFinding(
            check_name=name,
            severity='errored',
            code='unevaluable',
            message=_unevaluable_message(name, ref, repo_root),
        )

    if outcome is CheckOutcome.PASS:
        # The whole rule: a check that is ALREADY green at the authoring
        # tree can never signal anything, whichever polarity it claims.
        if expect == 'present':
            return CheckFinding(
                check_name=name,
                severity='reject',
                code='vacuous_present',
                message=_vacuous_present_message(name, pattern, ref),
            )
        return CheckFinding(
            check_name=name,
            severity='reject',
            code='vacuous_absent',
            message=_vacuous_absent_message(name, pattern, ref),
        )

    # CheckOutcome.FAIL — the healthy, forward-looking majority case.
    return None
