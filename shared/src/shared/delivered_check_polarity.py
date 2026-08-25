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
  - ``rc >= 2 → ERRORED`` carries the subtlety that a pathspec matching
    nothing in the tree exits >= 2, so it is ERRORED rather than FAIL.

One builder plus one interpreter makes divergence structurally
impossible.

IMPORT-LIGHT BY DESIGN. stdlib only — no pydantic, no ``fused_memory``,
no ``orchestrator``. ``shared/`` is the only package both fused-memory
and orchestrator already depend on (fused-memory must not import
orchestrator), so this is the sole placement that lets the authoring-time
and runtime call sites share code at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from pathlib import Path

__all__ = [
    'CheckOutcome',
    'build_grep_argv',
    'interpret_grep_rc',
]


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
