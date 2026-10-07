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
landed yet, so ``main`` *is* the pre-task tree — which collapses the
whole classification into a single measured predicate:

    ================  ===============  ==============
                      expect: present  expect: absent
    ================  ===============  ==============
    matches main      REJECT           healthy
    no match at main  healthy          REJECT
    ================  ===============  ==============

The rule is about POLARITY, so it covers every kind that carries an
``expect``: ``grep`` (does the pattern match) and ``path`` (does every
listed path exist). ``script`` has no ``expect`` and is not linted.

That subsumes the three fold-in modes (esc-4545-2) without parsing
English or guessing at author intent from the check's NAME:

  - MODE 1 — polarity inversion (5799): a pattern the task will remove
    is *necessarily present* at authoring time (that is what makes it
    removable), so the ``expect=present`` check already passes → the
    vacuity rule rejects it.
  - MODE 2 — over-broad ``expect=absent`` (task 3534's pattern matched
    inside the very file it owned). Genuinely undecidable at authoring
    time, so it lands as a WARN, not a reject.
  - MODE 3 — a ``kind=grep`` ``expect=present`` pattern that names a
    FILE rather than a symbol inside it (a test module that never
    mentions its own name). Invisible to the vacuity rule (at authoring
    time the file does not exist yet, so nothing matches and the check
    looks healthy), so it gets its own structural rule, read against the
    committed tree AND the task's declared ``metadata.files``, whose
    remedy is the ``kind='path'`` check that says what was meant.

Comment-only and self-referential matches are not separate gates: at
authoring time they are sub-species of "a check that already matches",
so they sharpen the REJECTION MESSAGE rather than adding a predicate.

WHY THE REFERENCE TREE IS :data:`GATE_REF`, NOT ``HEAD`` OR A HISTORICAL
SHA. The 2x2's axis is "matches at the AUTHORING tree", and at authoring
time that tree is free: ``commit_planning``/stamping run BEFORE the task
lands, so ``main`` *is* the pre-task tree. It is ``main`` and not
``HEAD`` because ``main`` is what the runtime gate reads, and the project
checkout is machine-operated and can sit on another branch: a branch that
already carries the capability would make a sound check look vacuous. A
root where ``main`` does not resolve yields ERRORED — reported, never a
reject. No history, no commit-ordering premise, no ``done``-time SHA to
recover. That is the
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
are the SINGLE SOURCE OF TRUTH for grep-check semantics, as
:func:`build_path_argv` and :func:`interpret_path_listing` are for
path checks. ``orchestrator.delivered_checks._run_grep_check`` and
``_run_path_check`` delegate to them, and that delegation is the point:
the authoring gate and the runtime gate must agree exactly, or this
module becomes a new source of the very defect it prevents. A check the lint judges healthy but the runtime later
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

One builder plus one interpreter per kind makes divergence structurally
impossible.

THE SECOND AXIS: SCOPE LIVENESS. Orthogonal to polarity, an
``expect=present`` check whose ``paths`` name a ``sys.modules`` alias shim or
a path the mainline deleted or moved away can never go green either (task
5036's ``merge_queue.py`` shim). That classification and its policy live in
``shared.delivered_check_scope``; the lint reports them as ``shim_path`` and
``removed_path`` rejects alongside any polarity finding.

NEVER RAISES. :func:`lint_delivered_checks` returns findings for any
iterable of entries — malformed check entries, values no argv can carry
(a NUL byte, a lone surrogate), a non-repo root, a missing ``git``. Both wire points depend on that
unconditionally and for OPPOSITE reasons: ``commit_planning`` would turn
an exception into a planning outage, and ``stamp_capability_manifests``
is contractually never-raising, so an exception there would abort the
very status flip it was called from. The three dispositions
(:class:`CheckFinding`'s ``severity``) are how a caller tells a measured
defect from an undecidable case from an availability failure.

IMPORT-LIGHT BY DESIGN. stdlib only — no pydantic, no ``fused_memory``,
no ``orchestrator``. ``shared/`` is the only package both fused-memory
and orchestrator already depend on (fused-memory must not import
orchestrator), so this is the sole placement that lets the authoring-time
and runtime call sites share code at all.
"""

from __future__ import annotations

import functools
import json
import logging
import posixpath
import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from shared.delivered_check_scope import (
    GIT_PROBE_FAILURES,
    GIT_TIMEOUT_SECS,
    STALE_PATH_CODES,
    PathState,
    ScopePath,
    classify_scope_paths,
    resolve_commit,
    stale_scope_paths,
)

__all__ = [
    'GATE_REF',
    'CheckFinding',
    'CheckOutcome',
    'build_grep_argv',
    'build_path_argv',
    'evaluate_grep_at_tree',
    'evaluate_path_at_tree',
    'extract_delivered_checks',
    'interpret_grep_rc',
    'interpret_path_listing',
    'lint_delivered_checks',
    'polarity_error',
]

logger = logging.getLogger(__name__)

#: The ref every delivered_check is judged against: the runtime gate's
#: (``orchestrator/src/orchestrator/scheduler.py::Scheduler._resolve_main_sha``
#: rev-parses it, and ``orchestrator.delivered_checks.run_delivered_check``
#: defaults to it). The authoring-time wire points evaluate against the same
#: name so the two gates judge one tree.
GATE_REF: str = 'main'


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


def build_path_argv(path: str, *, project_root: str | Path, ref: str) -> list[str]:
    """``['git','-C',<root>,'ls-tree','-r','--full-tree','--name-only',<ref>,'--',<path>]``.

    ONE path per probe: ``ls-tree`` answers with a flat filename list that
    cannot be attributed back to the pathspec that produced it without an
    ad-hoc parser over git output, so a multi-path check is one probe per
    entry. ``--full-tree`` makes the pathspec repo-root-relative whatever
    the subprocess cwd, matching the repo-relative ``paths`` the schema
    validator enforces for this kind.
    """
    return [
        'git',
        '-C',
        str(project_root),
        'ls-tree',
        '-r',
        '--full-tree',
        '--name-only',
        ref,
        '--',
        path,
    ]


def interpret_path_listing(rc: int, stdout: str, expect: str | None) -> CheckOutcome:
    """Map one ``ls-tree`` probe + expected polarity to a verdict.

    EXISTENCE IS READ FROM STDOUT, NOT FROM THE RETURN CODE — the opposite
    of :func:`interpret_grep_rc`, and the one thing that must not be carried
    across by analogy. ``ls-tree`` exits 0 whether or not the path exists; a
    missing path simply prints nothing. Reading ``rc == 0`` as "exists"
    would make every path check green.

    A non-zero rc is a genuine git error (a bad ref, or a pathspec outside
    the repository, both exit 128) and is checked FIRST: git prints nothing
    on an error and nothing on an absent path, so reading stdout first would
    turn "could not be evaluated" into a definitive verdict.

    Any non-``'present'`` *expect* takes the absent arm, exactly as
    :func:`interpret_grep_rc` does.
    """
    if rc != 0:
        return CheckOutcome.ERRORED
    exists = bool(stdout.strip())
    holds = exists if expect == 'present' else not exists
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
        A measured defect. Either the check is green at the authoring tree,
        so landing its producer cannot change its verdict and it gates
        nothing, or its scope is stale, so it can never go green.
        ``commit_planning`` refuses the batch; the stamper refuses to copy
        the check.
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
    ref: str = GATE_REF,
    timeout_secs: float = GIT_TIMEOUT_SECS,
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
    * ``ValueError`` — a NUL byte or a lone surrogate in the argv.
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
    except GIT_PROBE_FAILURES:
        return CheckOutcome.ERRORED
    return interpret_grep_rc(completed.returncode, expect)


def evaluate_path_at_tree(
    paths: Sequence[str],
    *,
    expect: str | None,
    repo_root: str | Path,
    ref: str = GATE_REF,
    timeout_secs: float = GIT_TIMEOUT_SECS,
) -> CheckOutcome:
    """Run one path check against *ref* in *repo_root*. Never raises.

    The synchronous twin of ``orchestrator.delivered_checks._run_path_check``
    and, like :func:`evaluate_grep_at_tree`, nothing but the shared builder,
    ``subprocess.run`` and the shared interpreter. CONJUNCTIVE and
    short-circuiting, as the runtime is: every path must hold, and the first
    probe that does not decides the outcome — ``ERRORED`` included, so an
    unevaluable probe is never outvoted by the paths after it.
    """
    for path in paths:
        argv = build_path_argv(path, project_root=repo_root, ref=ref)
        try:
            completed = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_secs
            )
        except GIT_PROBE_FAILURES:
            return CheckOutcome.ERRORED
        outcome = interpret_path_listing(completed.returncode, completed.stdout, expect)
        if outcome is not CheckOutcome.PASS:
            return outcome
    return CheckOutcome.PASS


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
        f'{repo_root} (git errored, timed out, or is unavailable, or the check '
        f'holds a value no command line can carry; a non-repo root lands here '
        f'too). Reported as unvalidated rather than accepted or '
        f'rejected: an infrastructure failure must not block planning, but it must '
        f'not pass as a clean bill of health either.'
    )


def _strings_only(values: object) -> list[str]:
    """The string entries of *values*, or ``[]`` for a non-sequence.

    Mirrors ``extract_files``' closing ``[f for f in files if
    isinstance(f, str)]`` — the same benign filtering, applied to the
    ``paths``/``files`` lists the lint is handed.
    """
    if not isinstance(values, (list, tuple)):
        return []
    return [v for v in values if isinstance(v, str)]


def lint_delivered_checks(
    checks: Iterable[object],
    *,
    files: Sequence[str] | None,
    repo_root: str | Path,
    ref: str = GATE_REF,
) -> list[CheckFinding]:
    """Lint a whole ``metadata.delivered_checks`` list against the authoring tree.

    Returns at most one polarity finding per check, plus at most one per
    stale-path code (the scope axis, see the module docstring), and an empty
    list for a clean batch. *files* is the task's declared
    ``metadata.files``, used by the refinements to tell a match inside the
    task's own scope from one outside it.

    The 2x2 is a statement about POLARITY, so it evaluates the kinds that
    carry an ``expect`` — ``grep`` and ``path``. A script check has none to
    invert and is skipped BEFORE any subprocess, so a script-only batch
    costs nothing.

    *ref* is resolved to a commit ONCE, on the first check that needs git,
    and every check in the call is judged at that one SHA. A ref that does
    not resolve makes every such check unevaluable: handed to ``git grep``
    unresolved, a name like ``main`` with no such branch would be read as a
    PATHSPEC and answered from the working tree.
    """
    tree = functools.cache(lambda: resolve_commit(repo_root, ref))
    findings: list[CheckFinding] = []
    for check in checks:
        finding = _lint_one_check(
            check, files=files, repo_root=repo_root, ref=ref, tree=tree
        )
        if finding is not None:
            findings.append(finding)
        findings.extend(
            _stale_scope_findings(check, repo_root=repo_root, ref=ref, tree=tree)
        )
    return findings


def _unevaluable_finding(name: str, ref: str, repo_root: str | Path) -> CheckFinding:
    return CheckFinding(
        check_name=name,
        severity='errored',
        code='unevaluable',
        message=_unevaluable_message(name, ref, repo_root),
    )


def _check_name(check: object) -> str | None:
    """The usable name a finding about *check* is addressed to, if any."""
    if not isinstance(check, dict):
        return None
    name = check.get('name')
    return name if isinstance(name, str) and name else None


def _lint_one_check(
    check: Any,
    *,
    files: Sequence[str] | None,
    repo_root: str | Path,
    ref: str,
    tree: Callable[[], str | None],
) -> CheckFinding | None:
    """Route one check to its kind's 2x2. ``None`` means healthy (or not
    our business).

    *check* is typed ``Any`` because it is a RAW metadata dict off the
    wire, not a validated ``DeliveredCheckMeta`` — the lint runs at
    authoring time precisely to catch entries a schema check cannot. Every
    field is therefore re-checked rather than trusted.

    Both skip guards here run BEFORE any subprocess:

    * no usable ``name`` — a finding is addressed to a check BY NAME;
      without one there is nothing a caller could report or an author could
      fix, so reporting it under a name that does not exist would be worse
      than skipping.
    * a kind with no entry in :data:`_POLARITY_LINTERS` — ``script`` (no
      ``expect`` to invert; a script-only batch must cost nothing), or a
      kind the schema does not know at all.
    """
    name = _check_name(check)
    if name is None:
        return None
    kind = check.get('kind')
    linter = _POLARITY_LINTERS.get(kind) if isinstance(kind, str) else None
    if linter is None:
        return None
    return linter(check, name=name, files=files, repo_root=repo_root, ref=ref, tree=tree)


def _lint_grep_check(
    check: dict[str, Any],
    *,
    name: str,
    files: Sequence[str] | None,
    repo_root: str | Path,
    ref: str,
    tree: Callable[[], str | None],
) -> CheckFinding | None:
    """The 2x2 for a ``kind='grep'`` check, plus the two rules it cannot see.

    A grep check without a usable ``pattern`` is a SCHEMA defect, not a
    polarity defect, and there is nothing to grep for; the metadata
    validator owns that diagnosis, so it is skipped before any subprocess.
    *ref* names the tree in messages; git reads the commit *tree* resolved.
    """
    pattern = check.get('pattern')
    if not isinstance(pattern, str) or not pattern:
        return None
    expect = check.get('expect')
    # A non-string entry in `paths` would be handed to subprocess as an argv
    # element and raise TypeError; in `files` it would reach `.rstrip`.
    # Dropping them keeps the check EVALUABLE against the usable entries,
    # degrading to a verdict rather than to an exception.
    paths = _strings_only(check.get('paths'))
    files = _strings_only(files)

    sha = tree()
    if sha is None:
        return _unevaluable_finding(name, ref, repo_root)
    outcome = evaluate_grep_at_tree(
        pattern, paths, expect=expect, repo_root=repo_root, ref=sha
    )

    if outcome is CheckOutcome.ERRORED:
        return _unevaluable_finding(name, ref, repo_root)

    if outcome is CheckOutcome.PASS:
        # The whole rule: a check that is ALREADY green at the authoring
        # tree can never signal anything, whichever polarity it claims.
        if expect != 'present':
            return CheckFinding(
                check_name=name,
                severity='reject',
                code='vacuous_absent',
                message=_vacuous_absent_message(name, pattern, ref),
            )
        # The REJECTION is already decided; only the CODE and the evidence
        # are still open. Every refinement below is therefore wrapped so a
        # git failure or a malformed record degrades to the unrefined code
        # — never to a crash, and never to a dropped rejection.
        try:
            code, sites = _refine_vacuous_present(
                _grep_matches(pattern, paths, repo_root=repo_root, ref=sha),
                manifest_path=check.get('manifest_path'),
            )
            message = _vacuous_present_refined_message(code, name, pattern, ref, sites)
        except Exception:  # noqa: BLE001 - a refinement must never lose the reject
            logger.warning(
                'delivered_check polarity: refinement failed for check %r '
                '(pattern=%r, ref=%s, repo_root=%s); reporting the unrefined '
                'vacuous_present rejection',
                name, pattern, ref, repo_root, exc_info=True,
            )
            code, sites = 'vacuous_present', ()
            message = _vacuous_present_message(name, pattern, ref)
        return CheckFinding(
            check_name=name,
            severity='reject',
            code=code,
            message=message,
            detail=sites,
        )

    # CheckOutcome.FAIL — healthy under the 2x2. The two classes the 2x2
    # cannot see live here, and BOTH are scoped to an explicit polarity so
    # a malformed `expect` can never be relabelled by one of them.
    try:
        if expect == 'present':
            return _filename_shaped_finding(
                name, pattern, paths, repo_root=repo_root, ref=sha
            ) or _declared_filename_finding(name, pattern, paths, files=files)
        if expect == 'absent':
            return _absent_overbroad_finding(
                name, pattern, paths, files=files, repo_root=repo_root, ref=sha
            )
    except Exception:  # noqa: BLE001 - these are advisory; never fail the lint
        # Degrading here loses only a diagnosis, never a rejection: the 2x2
        # has already declared this check healthy. Logged rather than
        # swallowed so the gate's own coverage gaps stay visible.
        logger.warning(
            'delivered_check polarity: advisory rule failed for check %r '
            '(expect=%s, pattern=%r, ref=%s, repo_root=%s); no finding emitted',
            name, expect, pattern, ref, repo_root, exc_info=True,
        )
        return None
    return None


def _vacuous_path_message(name: str, paths: Sequence[str], expect: str | None, ref: str) -> str:
    if expect == 'present':
        return (
            f"delivered_check {name!r} (kind='path', expect=present, paths {list(paths)!r}) "
            f'— every listed path ALREADY exists at the authoring tree at {ref}, so '
            f'landing this task cannot change its verdict: it is green the day it is '
            f'written and therefore gates nothing. {_INVARIANT} List a path this task '
            f'will CREATE — or, if this task DELETES it, flip the check to expect=absent.'
        )
    return (
        f"delivered_check {name!r} (kind='path', expect=absent, paths {list(paths)!r}) "
        f'— none of the listed paths exists at the authoring tree at {ref}: there is '
        f'nothing left for this task to delete, so the check is green the day it is '
        f'written and therefore gates nothing. {_INVARIANT} List a path this task will '
        f'actually DELETE — or, if this task CREATES it, flip the check to expect=present.'
    )


def _lint_path_check(
    check: dict[str, Any],
    *,
    name: str,
    files: Sequence[str] | None,
    repo_root: str | Path,
    ref: str,
    tree: Callable[[], str | None],
) -> CheckFinding | None:
    """The 2x2 for a ``kind='path'`` check, read as existence.

    "Matches the authoring tree" means every listed path already exists
    there — the runtime's own conjunctive reading, reached through the same
    primitive. The grep-only refinements and rules do not apply: a path has
    no comment lines, no self-matching ``pattern:`` line, and nothing to
    confuse with a filename. A check without a usable ``paths`` entry is a
    SCHEMA defect the metadata validator owns, skipped before any
    subprocess.
    """
    paths = _strings_only(check.get('paths'))
    if not paths:
        return None
    expect = check.get('expect')
    sha = tree()
    if sha is None:
        return _unevaluable_finding(name, ref, repo_root)
    outcome = evaluate_path_at_tree(paths, expect=expect, repo_root=repo_root, ref=sha)
    if outcome is CheckOutcome.ERRORED:
        return _unevaluable_finding(name, ref, repo_root)
    if outcome is CheckOutcome.FAIL:
        return None
    return CheckFinding(
        check_name=name,
        severity='reject',
        code='vacuous_present' if expect == 'present' else 'vacuous_absent',
        message=_vacuous_path_message(name, paths, expect, ref),
        detail=tuple(paths),
    )


#: The kinds the 2x2 evaluates: exactly the mechanical kinds that carry an
#: ``expect`` (``shared.capability_manifest`` forbids one on ``script``).
#: ``shared/tests/test_delivered_check_polarity.py::TestEveryPolarityKindIsLinted``
#: holds this against ``MECHANICAL_CHECK_KINDS`` so a new kind cannot be
#: added to the schema and silently skipped here.
_POLARITY_LINTERS: dict[str, Callable[..., CheckFinding | None]] = {
    'grep': _lint_grep_check,
    'path': _lint_path_check,
}


# ---------------------------------------------------------------------------
# The scope axis
# ---------------------------------------------------------------------------

_NEVER_GREEN = (
    'so the check can never go green, and its dependents wedge behind what '
    'reads as an undelivered capability.'
)


def _shim_path_message(name: str, entries: Sequence[ScopePath], ref: str) -> str:
    paths = [entry.path for entry in entries]
    return (
        f'delivered_check {name!r} (expect=present) is scoped to {paths!r}, which at '
        f'{ref} is a sys.modules alias shim: the file only rebinds '
        f'sys.modules[__name__] to the module it aliases, so the code the check '
        f'asserts lives elsewhere, {_NEVER_GREEN} Repath `paths` to the module the '
        f'shim aliases, where the code lives now.'
    )


def _removed_path_message(name: str, entries: Sequence[ScopePath], ref: str) -> str:
    sites = ', '.join(
        f'{entry.path} (removed in {entry.removed_in or "mainline history"})'
        for entry in entries
    )
    return (
        f'delivered_check {name!r} (expect=present) names {sites}: deleted or moved '
        f'away on the mainline and absent at {ref}, {_NEVER_GREEN} Repath `paths` to '
        f'where the code lives now. If this task genuinely re-creates the file, scope '
        f'a grep check to its parent directory instead.'
    )


_STALE_PATH_MESSAGES: dict[PathState, Callable[[str, Sequence[ScopePath], str], str]] = {
    PathState.SYS_MODULES_SHIM: _shim_path_message,
    PathState.REMOVED: _removed_path_message,
}


def _stale_scope_findings(
    check: Any,
    *,
    repo_root: str | Path,
    ref: str,
    tree: Callable[[], str | None],
) -> list[CheckFinding]:
    """One reject per stale-path code for *check*, each listing its paths.

    Skipped before any git call unless *check* is a named ``expect=present``
    check of a polarity kind with usable ``paths``. Fails open: an
    unresolvable tree is already reported by the polarity axis, and a
    classification failure loses only this diagnosis.
    """
    name = _check_name(check)
    if name is None:
        return []
    kind = check.get('kind')
    expect = check.get('expect')
    paths = list(dict.fromkeys(_strings_only(check.get('paths'))))
    if not (isinstance(kind, str) and kind in _POLARITY_LINTERS):
        return []
    if expect != 'present' or not paths:
        return []
    sha = tree()
    if sha is None:
        return []
    census = classify_scope_paths(paths, repo_root=repo_root, ref=sha)
    if census is None:
        logger.warning(
            'delivered_check scope: could not classify the paths of check %r '
            '(ref=%s, repo_root=%s); no scope finding emitted',
            name, ref, repo_root,
        )
        return []
    by_state: dict[PathState, list[ScopePath]] = {}
    for entry in stale_scope_paths(kind, expect, [census[p] for p in paths if p in census]):
        by_state.setdefault(entry.state, []).append(entry)
    return [
        CheckFinding(
            check_name=name,
            severity='reject',
            code=STALE_PATH_CODES[state],
            message=_STALE_PATH_MESSAGES[state](name, entries, ref),
            detail=tuple(entry.path for entry in entries),
        )
        for state, entries in by_state.items()
    ]


# ---------------------------------------------------------------------------
# Diagnostic refinements, and the two rules the 2x2 cannot see
# ---------------------------------------------------------------------------

#: Mirrors ``fused_memory.server.manifest_stamping._SIDECAR_SUFFIX`` and
#: ``shared/tests/capability_manifest_corpus.py::MANIFEST_SUFFIX``.
#: Re-declared rather than imported for the same reason the latter does it:
#: ``shared`` must not depend on ``fused_memory``.
_MANIFEST_SUFFIX = '.capability-manifest.yaml'

#: A match line starting (after leading whitespace) with any of these is
#: PROSE, not a capability. ``*`` covers both a C block-comment
#: continuation and a markdown bullet — both are prose for this purpose.
_COMMENT_MARKERS = ('#', '//', '*', '"""', "'''")


@dataclass(frozen=True)
class _GrepMatch:
    """One ``file:line:text`` site, so a classifier can inspect WHERE a
    check matches rather than only the pass/fail bit."""

    path: str
    line_no: int
    text: str


def _grep_matches(
    pattern: str,
    paths: Sequence[str] | None,
    *,
    repo_root: str | Path,
    ref: str,
    timeout_secs: float = GIT_TIMEOUT_SECS,
) -> list[_GrepMatch] | None:
    """The match SITES for a check. ``None`` means git could not answer.

    ``-n`` is INSERTED into the argv :func:`build_grep_argv` returns rather
    than a second argv being assembled here: the pattern's ``-e``
    separator, the ``--`` pathspec placement and the ref position all stay
    owned by the one builder, so a refinement can never search a different
    corpus than the verdict it is refining (module docstring, PARITY
    CONTRACT). ``argv.index('-E')`` finds the FLAG even when the pattern is
    itself the literal ``'-E'``, since the flag precedes the pattern.

    ``None`` (not ``[]``) for an unanswerable question is the load-bearing
    distinction: ``[]`` would read as "matches, but none of them are
    interesting", which is exactly how a refinement would silently drop a
    rejection.
    """
    argv = build_grep_argv(pattern, paths, project_root=repo_root, ref=ref)
    argv.insert(argv.index('-E'), '-n')
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_secs
        )
    except GIT_PROBE_FAILURES:
        return None
    if completed.returncode >= 2:
        return None

    prefix = f'{ref}:'
    matches: list[_GrepMatch] = []
    for record in completed.stdout.splitlines():
        rest = record[len(prefix):] if record.startswith(prefix) else record
        path, _, remainder = rest.partition(':')
        line_no, _, text = remainder.partition(':')
        if not path or not line_no.isdigit():
            # A record this parser cannot read degrades to "no refinement",
            # never to a wrong one.
            continue
        matches.append(_GrepMatch(path=path, line_no=int(line_no), text=text))
    return matches


def _tracked_paths(
    paths: Sequence[str] | None,
    *,
    repo_root: str | Path,
    ref: str,
    timeout_secs: float = GIT_TIMEOUT_SECS,
) -> list[str] | None:
    """Paths in *ref*'s tree under *paths* (whole tree when empty) — the
    committed tree the verdict was reached against, never the index.
    ``None`` on any git failure — same never-guess contract as
    :func:`_grep_matches`."""
    argv = ['git', '-C', str(repo_root), 'ls-tree', '-r', '-z', '--full-tree', '--name-only', ref]
    if paths:
        argv.append('--')
        argv.extend(paths)
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_secs
        )
    except GIT_PROBE_FAILURES:
        return None
    if completed.returncode != 0:
        return None
    return [record for record in completed.stdout.split('\0') if record]


def _descriptor_family(manifest_path: str) -> frozenset[str]:
    """The sidecar plus the PRD it annotates.

    ``plans/foo-prd.capability-manifest.yaml`` →
    ``{plans/foo-prd.capability-manifest.yaml, plans/foo-prd.md}``. Both
    count as "the descriptor talking about itself": the measured t2863
    specimen matched on the sidecar's own ``pattern:`` line, and a PRD
    naming the token it is specifying is the same non-evidence.
    """
    if manifest_path.endswith(_MANIFEST_SUFFIX):
        stem = manifest_path[: -len(_MANIFEST_SUFFIX)]
        return frozenset({manifest_path, f'{stem}.md'})
    return frozenset({manifest_path})


def _is_comment_line(text: str) -> bool:
    return text.lstrip().startswith(_COMMENT_MARKERS)


def _covered_by_declared_files(path: str, declared: Sequence[str]) -> bool:
    """Is *path* inside the task's declared ``metadata.files``?

    A declared entry may name a FILE or a DIRECTORY — tasks routinely
    declare scope coarsely — so a directory entry covers everything
    beneath it. Treating a directory as covering nothing would warn on the
    common shape.
    """
    return any(_is_under(path, entry) for entry in declared)


def _is_under(path: str, entry: str) -> bool:
    """*path* is *entry* itself, or lies beneath it as a directory."""
    return path == entry or path.startswith(entry.rstrip('/') + '/')


def _format_sites(matches: Sequence[_GrepMatch], *, limit: int = 5) -> tuple[str, ...]:
    """``path:line: text`` evidence lines, truncated so a pathological
    pattern cannot put thousands of lines into a reject payload."""
    shown = [f'{m.path}:{m.line_no}: {m.text.strip()}' for m in matches[:limit]]
    if len(matches) > limit:
        shown.append(f'... and {len(matches) - limit} more match(es)')
    return tuple(shown)


def _refine_vacuous_present(
    matches: list[_GrepMatch] | None, *, manifest_path: object
) -> tuple[str, tuple[str, ...]]:
    """Sharpen ``vacuous_present`` into a code that says WHY.

    Order is pinned: self-reference is the more specific diagnosis and wins
    when both hold (a markdown bullet in the sibling PRD is inside the
    descriptor family AND comment-shaped). Emitting both would double-count
    one defect in the reject payload.

    ``None`` matches — git could not answer — degrades to the unrefined
    code. The rejection itself was already decided by the 2x2 and is never
    at stake here.
    """
    if not matches:
        return 'vacuous_present', ()

    sites = _format_sites(matches)

    if isinstance(manifest_path, str) and manifest_path:
        family = _descriptor_family(manifest_path)
        if all(m.path in family for m in matches):
            return 'vacuous_present_self_referential', sites

    if all(_is_comment_line(m.text) for m in matches):
        return 'vacuous_present_comment_only', sites

    return 'vacuous_present', sites


def _vacuous_present_refined_message(
    code: str, name: str, pattern: str, ref: str, sites: Sequence[str]
) -> str:
    """The refined half of a ``vacuous_present`` message.

    Falls through to the unrefined message for the plain code, so the
    invariant and the concrete repair are stated exactly once.
    """
    if code == 'vacuous_present_self_referential':
        return (
            f'delivered_check {name!r} (expect=present, pattern {pattern!r}) matches '
            f'ONLY the descriptor that declares it and the PRD that describes it, at '
            f'{ref}: {"; ".join(sites)}. The check is satisfied by its own existence '
            f'— it was green before this task started and would stay green if the '
            f'capability were never built. {_INVARIANT} Assert a symbol the '
            f'IMPLEMENTATION will introduce, and scope `paths` to the code rather '
            f'than to plans/.'
        )
    if code == 'vacuous_present_comment_only':
        return (
            f'delivered_check {name!r} (expect=present, pattern {pattern!r}) matches '
            f'ONLY comment or docstring lines at {ref}: {"; ".join(sites)}. A comment '
            f'is not a capability, so this check asserts that someone wrote the word '
            f'down — true before the producer lands and still true if it never does. '
            f'{_INVARIANT} Assert a symbol in live code.'
        )
    return _vacuous_present_message(name, pattern, ref)


def _filename_shaped_finding(
    name: str,
    pattern: str,
    paths: Sequence[str] | None,
    *,
    repo_root: str | Path,
    ref: str,
) -> CheckFinding | None:
    """MODE 3: an ``expect='present'`` pattern that names a FILE, not a symbol.

    The one class the 2x2 cannot see. At authoring time the producer's file
    does not exist yet, so the check evaluates FAIL and looks like an
    ordinary healthy forward-looking check; it only reveals itself once the
    producer lands and the check STILL fails, because a test module does not
    mention its own name (the measured task-3536 specimen,
    ``test_workflow_merge_gating_strand``).

    This is SCOPE item 3's detective option. Its prescriptive twin,
    ``kind='path'``, has since landed (task 4743), so the rejection names it
    alongside the grep alternative of asserting a symbol inside the file.
    It does not prefill ``paths`` with the matched files: at the authoring
    tree those already exist, so a path check on them would itself be
    rejected as ``vacuous_present``.

    ``re.search`` — Python's engine, not the POSIX ERE ``git grep -E`` uses
    — is deliberate and safe HERE and nowhere else in this module: the
    subject is a path LIST that git offers no grep mode over, the result is
    a diagnostic rather than the verdict the runtime must reproduce, and any
    pattern Python cannot compile yields ``None`` (no finding) rather than a
    guess. A verdict is never decided this way.
    """
    tracked = _tracked_paths(paths, repo_root=repo_root, ref=ref)
    if not tracked:
        return None
    try:
        matcher = re.compile(pattern)
    except re.error:
        return None
    hits = [path for path in tracked if matcher.search(path)]
    if not hits:
        return None
    return CheckFinding(
        check_name=name,
        severity='reject',
        code='filename_shaped',
        message=(
            f'delivered_check {name!r} (expect=present, pattern {pattern!r}) has ZERO '
            f'content matches, but matches the FILENAME of a tracked path: '
            f'{", ".join(hits[:5])}. A grep check reads file CONTENTS, so this one '
            f'can never go green — the file existing is not something git grep can '
            f"see. If the capability IS a file's existence, declare it as "
            f"kind='path' with `paths` naming the file the producer creates; "
            f'otherwise assert a symbol defined INSIDE the file (a class, function '
            f'or constant the producer adds).'
        ),
        detail=tuple(hits[:5]),
    )


def _bracket_end(pattern: str, start: int) -> int:
    """Index just past the ERE bracket expression opening at *start*, or -1.

    A ``]`` first in the list (after an optional ``^``) is literal, and
    ``[:class:]``, ``[.coll.]`` and ``[=equiv=]`` carry their own ``]``.
    """
    i = start + 1
    if pattern.startswith('^', i):
        i += 1
    if pattern.startswith(']', i):
        i += 1
    while i < len(pattern):
        if pattern[i] == ']':
            return i + 1
        if pattern[i] == '[' and pattern[i + 1 : i + 2] in (':', '.', '='):
            close = pattern.find(pattern[i + 1] + ']', i + 2)
            if close < 0:
                return -1
            i = close + 2
            continue
        i += 1
    return -1


def _top_level_arms(pattern: str) -> list[str] | None:
    """The ERE alternatives of *pattern*: split on ``|`` outside groups,
    bracket expressions and escapes, after unwrapping a group that encloses
    the whole pattern. ``None`` when the pattern is not balanced.
    """
    arms: list[str] = []
    depth = 0
    start = 0
    wholly_grouped = pattern.startswith('(')
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == '\\':
            i += 2
            continue
        if char == '[':
            i = _bracket_end(pattern, i)
            if i < 0:
                return None
            continue
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                return None
            if depth == 0 and i != len(pattern) - 1:
                wholly_grouped = False
        elif char == '|' and depth == 0:
            arms.append(pattern[start:i])
            start = i + 1
        i += 1
    if depth:
        return None
    if wholly_grouped and not arms:
        return _top_level_arms(pattern[1:-1])
    arms.append(pattern[start:])
    return arms


def _declared_filename_finding(
    name: str,
    pattern: str,
    paths: Sequence[str],
    *,
    files: Sequence[str],
) -> CheckFinding | None:
    """MODE 3 in its real shape: the pattern IS the name of a file the task
    declares but has not created yet.

    :func:`_filename_shaped_finding` reads the committed tree, so it cannot
    see a file the producer will CREATE — which is the measured specimen:
    task 3536's check was authored before 3536 created the module it names.
    The task's ``metadata.files`` can.

    THE RULE, exactly: EVERY top-level alternative of the pattern (see
    :func:`_top_level_arms`), compiled by Python ``re``, FULLY matches the
    basename or the stem (basename minus its last suffix) of one declared
    file that EVERY entry of the check's ``paths`` contains (the whole tree
    when ``paths`` is empty). Full match, not search, so a symbol that
    merely shares text with a module name is not flagged; every arm, so an
    alternative naming a real symbol keeps the check a sound gate; every
    path, because a grep that also reads files other than the declared one
    can be a wiring check — a declared module named in ANOTHER file's
    import.
    A pattern the splitter cannot parse is judged whole, and only when it
    carries no ``|`` at all.
    """
    arms = _top_level_arms(pattern)
    if arms is None:
        if '|' in pattern:
            return None
        arms = [pattern]
    try:
        matchers = [re.compile(arm) for arm in arms]
    except re.error:
        return None
    for declared in files:
        if declared.endswith('/') or not all(_is_under(declared, scope) for scope in paths):
            continue
        base = posixpath.basename(declared)
        names = (base, posixpath.splitext(base)[0])
        if all(any(m.fullmatch(n) for n in names) for m in matchers):
            return CheckFinding(
                check_name=name,
                severity='reject',
                code='filename_shaped',
                message=(
                    f'delivered_check {name!r} (expect=present, pattern {pattern!r}) is '
                    f'the NAME of {declared!r}, a file this task declares, not a symbol '
                    f'inside it. A grep over file CONTENTS cannot state that a file '
                    f'exists: this check goes green only by the coincidence of some '
                    f'file — its own docstring or usage text, an importer — mentioning '
                    f"the name, and stays red if none does. If the capability IS that "
                    f"file's existence, declare it as kind='path', expect='present', "
                    f'paths=[{declared!r}]; otherwise assert a symbol defined INSIDE '
                    f'the file (a class, function or constant the producer adds).'
                ),
                detail=(declared,),
            )
    return None


def _absent_overbroad_finding(
    name: str,
    pattern: str,
    paths: Sequence[str] | None,
    *,
    files: Sequence[str] | None,
    repo_root: str | Path,
    ref: str,
) -> CheckFinding | None:
    """MODE 2/2b, as a WARN that is never promoted.

    An ``expect='absent'`` pattern that also matches files the task does not
    own will keep failing after the task lands, wedging its dependent. But
    it is genuinely UNDECIDABLE at authoring time — task 3534's pattern
    legitimately matched inside the very file it owned — and this gate is
    hard-blocking, so a false reject costs more than a missed catch.
    """
    matches = _grep_matches(pattern, paths, repo_root=repo_root, ref=ref)
    if not matches:
        return None
    declared = list(files or ())
    outside = [m for m in matches if not _covered_by_declared_files(m.path, declared)]
    if not outside:
        return None
    offenders = sorted({m.path for m in outside})
    return CheckFinding(
        check_name=name,
        severity='warn',
        code='absent_overbroad',
        message=(
            f'delivered_check {name!r} (expect=absent, pattern {pattern!r}) currently '
            f'matches {len(outside)} line(s) in {len(offenders)} file(s) OUTSIDE this '
            f'task\'s declared files: {", ".join(offenders[:5])}. If those matches '
            f'survive the task, the check stays red after the producer lands and its '
            f'dependent is blocked. Narrow `paths` (or the pattern) to the code this '
            f'task actually removes. Reported, NOT blocking: an over-broad-looking '
            f'pattern is sometimes correct, so this is a warning by design.'
        ),
        detail=_format_sites(outside),
    )


# ---------------------------------------------------------------------------
# Wire adapters
# ---------------------------------------------------------------------------


def extract_delivered_checks(metadata: object) -> list[dict[str, Any]]:
    """Pull ``metadata.delivered_checks`` out of whatever shape arrived.

    BENIGN-ABSENT: every malformed or missing shape resolves to ``[]``
    rather than raising, because a gate that can raise on a metadata shape
    it did not anticipate would take down planning for a defect it was not
    even built to catch.

    Rules, mirroring
    ``fused_memory.middleware.lock_charter_guard.extract_files`` case for
    case:

    * ``None`` → ``[]``
    * ``dict`` → used directly
    * ``''`` → ``[]`` (benign-absent, not a discard)
    * ``str`` → ``json.loads``; on failure, or a non-object result → ``[]``
    * anything else (list, int, ...) → ``[]``
    * no ``delivered_checks`` key, or a non-list value → ``[]``
    * non-dict entries inside the list are filtered out

    That function is the SPEC this is kept in sync with, and it is
    DUPLICATED rather than imported for one structural reason: it lives in
    ``fused_memory``, and ``shared`` must not import ``fused_memory`` (the
    dependency runs the other way — this module exists in ``shared``
    precisely so both ``fused_memory`` and ``orchestrator`` can reach it).
    ``capability_manifest_corpus.MANIFEST_SUFFIX`` re-declares a
    ``fused_memory`` constant for the same reason.

    The one deliberate divergence: ``extract_files`` routes its ``str``
    branch through ``shared.task_metadata.parse_metadata`` to emit a
    ``task_metadata.schema_warning``. That warning is the lock-charter
    guard's own contract, not this one's — here an unreadable metadata blob
    means "no checks to lint", and the caller's own metadata validation is
    what reports the blob itself.
    """
    parsed: dict[str, Any] | None = None
    if isinstance(metadata, dict):
        parsed = metadata
    elif isinstance(metadata, str) and metadata:
        try:
            decoded = json.loads(metadata)
        except (ValueError, TypeError):
            return []
        if isinstance(decoded, dict):
            parsed = decoded

    if parsed is None:
        return []

    checks = parsed.get('delivered_checks')
    if not isinstance(checks, list):
        return []
    return [c for c in checks if isinstance(c, dict)]


#: The remedy every measured repair actually used. Stated POSITIVELY on
#: purpose: all three specimens were fixed by asserting the symbol the
#: producer INTRODUCES rather than by banning the one it removes, and a
#: reject an author cannot act on is only noise.
_POLARITY_HINT = (
    'Replace each rejected check with one that asserts the NEW symbol the '
    'producer introduces (kind=grep, expect=present, pattern = a class, '
    'function or constant that does not exist yet), scoped with `paths` to the '
    'files this task actually writes. That is the shape every measured repair '
    'took, and it is the only shape the gate can observe going green. When the '
    "capability IS a file's existence, use kind='path' with `paths` naming the "
    'file(s) this task creates (expect=present) or deletes (expect=absent). '
    'The invariant: a sound delivered_check FAILS at the authoring tree and '
    'PASSES once its producer lands. A check that is already green the day it '
    'is written can never signal anything — it gates nothing, and its dependent '
    'is dispatched as if unguarded. A check that can NEVER go green is worse: '
    'it blocks its dependent forever, and at runtime that is indistinguishable '
    'from a genuinely undelivered capability. A check scoped to a sys.modules '
    'shim, or to a path the mainline removed, is that kind: repath `paths` to '
    'where the code lives now.'
)


def polarity_error(
    findings: Sequence[CheckFinding], *, task_id: str | None = None
) -> dict[str, Any]:
    """Build the ``DeliveredCheckPolarityViolation`` reject payload.

    Shape mirrors
    ``fused_memory.middleware.lock_charter_guard.lock_charter_error``'s
    ``{error, error_type, <detail>, hint}`` convention — including the
    ``(task N)`` parenthetical and its disappearance when *task_id* is
    ``None`` — so an MCP caller handles this through the same code path it
    already uses for ``LockCharterViolation``.

    ONLY ``severity='reject'`` findings enter the payload. A ``'warn'`` is
    undecidable at authoring time (MODE 2) and an ``'errored'`` is an
    availability failure; letting either in would silently convert a
    deliberately non-blocking disposition into a blocking one, which is the
    exact inversion the WARN and fail-open-on-infra choices exist to
    prevent. Those findings are still REPORTED by the caller, just not here.
    """
    rejects = [f for f in findings if f.severity == 'reject']
    rejected_checks = len({f.check_name for f in rejects})
    task_clause = f' (task {task_id})' if task_id else ''
    check_list = ', '.join(f'{f.check_name!r} [{f.code}]' for f in rejects)
    return {
        'error': (
            f'metadata.delivered_checks contains {rejected_checks} check(s) that are '
            f'already satisfied at the authoring tree, or can never be satisfied'
            f'{task_clause}: {check_list}. Such a check gates nothing (or gates '
            f'forever), and at runtime a mis-authored check is indistinguishable '
            f'from a genuinely undelivered capability.'
        ),
        'error_type': 'DeliveredCheckPolarityViolation',
        'checks': [
            {
                'name': f.check_name,
                'code': f.code,
                'message': f.message,
                'detail': list(f.detail),
            }
            for f in rejects
        ],
        'hint': _POLARITY_HINT,
    }
