#!/usr/bin/env python3
"""One-shot audit: cross-check found_on_main-provenance done tasks against
actual commit lineage on the audited ref (default: main).

Motivation: a task marked done with ``metadata.done_provenance.kind ==
'found_on_main'`` was not landed by the orchestrator's own merge pipeline —
its deliverable was independently discovered already present on main.  That
provenance kind is intentionally exempt from the stronger ``is_ancestor`` +
commit-citation checks task 2500 added for the ``merged`` kind, which leaves
two retrospective blind spots this script sweeps for:

  (a) post-hoc revert — the cited commit genuinely landed, but was later
      reverted (or its declared files quietly removed) without the task
      being reopened.
  (b) misattribution — the cited commit is real and on the ref, but its
      message cites a *different* task, so it was never actually this
      task's deliverable.

For every found_on_main task this script:
  1. Parses ``metadata.done_provenance`` (commit/note) and ``metadata.files``
     (declared deliverables) via ``shared.task_metadata.parse_metadata``.
  2. Gathers git facts about the cited commit: is it reachable from the
     audited ref, what its message says, what files it touched, whether it
     was later reverted, and whether the declared files are still present
     at the ref's HEAD.
  3. Classifies the task into exactly one verdict via a fixed precedence
     ladder: ``commit_not_on_main`` > ``misattributed`` > ``reverted`` >
     ``deliverable_absent`` > ``unverifiable`` > ``ok`` (first match wins;
     see :func:`classify`).

``--apply`` is intentionally non-destructive: it annotates every flagged
task's ``metadata.x_provenance_audit`` (an ``x_``-prefixed forward-compat
key, so it round-trips through ``parse_metadata`` with no ``SchemaWarning``)
and lists every flagged task under ``needs_human_review`` in the summary —
it NEVER reopens a done task back to pending.  That correction is a human
decision.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/audit_found_on_main_provenance.py --project-root /path/to/project

  # Annotate flagged tasks' metadata (still never reopens anything).
  python scripts/audit_found_on_main_provenance.py --project-root /path/to/project --apply

  # Audit lineage against a ref other than main.
  python scripts/audit_found_on_main_provenance.py --project-root /path/to/project \\
      --ref origin/main
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from shared.task_metadata import parse_metadata

logger = logging.getLogger('audit_found_on_main_provenance')

# Mirrors orchestrator.git_ops.DEFAULT_COMMIT_CITATION_PATTERN (git_ops.py:219)
# conventions (task 2870 widens that pattern with the same three forms below;
# this is a deliberately independent, self-contained copy — see the NOT
# imported note just below — so it does not depend on 2870's landing state).
# NOT imported from there — orchestrator.git_ops pulls in the whole GitOps
# stack, and a fused-memory -> orchestrator runtime import is architecturally
# backwards (see task 2645 design decisions). This local pattern is
# generalized to EXTRACT every cited task id from a message (named capture
# groups), rather than testing one already-known id via str.format
# interpolation as the orchestrator version does:
#   - conventional-commit subject: `impl(50): ...` / `fix(50): ...`
#   - `task/{id}` branch mention: `Merge task/50 into main`, `... task/50 ...`
#   - hash-paren / bare-paren: `(#50)` / `(50)` anywhere in the message
#   - task-word paren: `(task 50)` anywhere in the message
# `\d+` is greedy, so a captured id is always the FULL digit run — `task/339`
# never yields a citation match for id '3399', nor vice versa. The paren/hash
# forms need no `\b` guard: the literal parens (and optional `#`) are
# self-delimiting, so `(#2870)` can never yield a truncated '287'.
# Accepted false-positive tradeoff: unlike `(#N)`/`(task N)`, bare `(N)`
# matches ANY parenthesized integer (a step ref, a year, a count), not just
# genuine citations — so an incidental one can flip a correctly-attributed
# found_on_main task to `misattributed` (see
# TestClassifyMisattributed.test_bare_paren_incidental_number_is_an_accepted_false_positive
# in the test module). Accepted because the misattribution check only fires
# when the audited task_id is absent from the cited set, `--apply` never
# reopens a task, and the net effect is fewer missed true-positive citations
# — this offline audit trades a bounded false-positive rate (surfaced for
# human review, not auto-corrected) for fewer missed misattributions.
CITATION_PATTERN = re.compile(
    r'^(?:merge|impl|amend|fix|test|feat|chore|docs|refactor|style|build)'
    r'\(\s*(?P<conv_tid>\d+)\s*[):]'
    r'|\btask/(?P<branch_tid>\d+)\b'
    r'|\(#?(?P<paren_tid>\d+)\)'
    r'|\(task (?P<task_word_tid>\d+)\)',
    re.MULTILINE,
)

# Every verdict classify() can return, in precedence order. Used to seed
# build_audit_report's verdict_counts so every verdict is always present
# (0 when it doesn't occur) rather than silently absent from the report.
_ALL_VERDICTS: tuple[str, ...] = (
    'commit_not_on_main', 'misattributed', 'reverted', 'deliverable_absent',
    'unverifiable', 'ok',
)

# Verdicts that apply_audit_annotations treats as "flagged" (annotated +
# surfaced under needs_human_review). `ok` and `unverifiable` are left alone.
_FLAGGED_VERDICTS = frozenset({
    'commit_not_on_main', 'misattributed', 'reverted', 'deliverable_absent',
})


@dataclass
class TaskProvenanceAudit:
    """Per-task facts gathered for the found_on_main provenance audit.

    Task facts (``task_id``/``title``/``commit``/``note``/``declared_files``)
    are populated by :func:`select_found_on_main_tasks`. The git-fact fields
    default to benign placeholders until :func:`build_audit_report` fills
    them in from an injected git-facts dependency. ``verdict``/``reasons``
    are set by :func:`classify`.
    """

    task_id: str
    title: str
    commit: str | None
    note: str | None
    declared_files: list[str]
    is_ancestor: bool = True
    commit_subject: str = ''
    commit_message: str = ''
    commit_files: list[str] = field(default_factory=list)
    revert_commit: str | None = None
    declared_files_missing_on_main: list[str] = field(default_factory=list)
    declared_files_inconclusive: list[str] = field(default_factory=list)
    verdict: str = ''
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Selection (pure)
# ---------------------------------------------------------------------------

def _id_as_int(task_id: str, fallback: int = 0) -> int:
    """Numeric task id as int for deterministic sort; non-numeric -> fallback.

    Mirrors ``audit_duplicate_tasks._id_as_int``: dotted subtask ids (e.g.
    ``'1.2'``) and other non-decimal shapes fall back rather than raising
    ``ValueError`` from a bare ``int()`` call.
    """
    return int(task_id) if str(task_id).isdecimal() else fallback


def select_found_on_main_tasks(tasks: list[dict]) -> list[TaskProvenanceAudit]:
    """Filter *tasks* to found_on_main-provenance ones, parsed via ``parse_metadata``.

    Tolerates ``metadata`` given as a dict OR a JSON string (``parse_metadata``
    handles both) and never raises on malformed metadata — a blob that fails
    validation resolves ``done_provenance`` to ``None`` via ``parse_metadata``'s
    warn-and-omit path, which this function treats the same as "no
    provenance at all" and simply skips.

    Returns audits sorted by ``int(task_id)`` (see :func:`_id_as_int`) for a
    deterministic report ordering downstream.
    """
    selected: list[TaskProvenanceAudit] = []
    for task in tasks:
        meta, _warnings = parse_metadata(task.get('metadata'), direction='read')
        prov = meta.done_provenance
        if prov is None or prov.kind != 'found_on_main':
            continue
        task_id = str(task.get('id', ''))
        selected.append(TaskProvenanceAudit(
            task_id=task_id,
            title=task.get('title') or '',
            commit=prov.commit,
            note=prov.note,
            declared_files=list(meta.files),
        ))
    selected.sort(key=lambda a: _id_as_int(a.task_id))
    return selected


# ---------------------------------------------------------------------------
# Citation helpers (pure)
# ---------------------------------------------------------------------------

def extract_cited_task_ids(message: str) -> set[str]:
    """Return every task id cited in *message* (commit subject + body).

    Mirrors the citation conventions in ``CITATION_PATTERN`` above:
    conventional-commit ``type(id):`` subjects, ``task/{id}`` mentions
    (which subsumes the ``Merge task/{id} into <main>`` merge-commit
    subject), hash-paren/bare-paren ``(#id)``/``(id)``, and the task-word
    paren form ``(task id)``. Returns an empty set for a message with no
    citations — never raises.
    """
    ids: set[str] = set()
    for m in CITATION_PATTERN.finditer(message or ''):
        tid = (
            m.group('conv_tid') or m.group('branch_tid')
            or m.group('paren_tid') or m.group('task_word_tid')
        )
        if tid:
            ids.add(tid)
    return ids


# ---------------------------------------------------------------------------
# Classifier (pure)
# ---------------------------------------------------------------------------

def classify(audit: TaskProvenanceAudit) -> tuple[str, list[str]]:
    """Resolve *audit* to a single verdict via a fixed precedence ladder.

    ``commit_not_on_main`` > ``misattributed`` > ``reverted`` >
    ``deliverable_absent`` > ``unverifiable`` > ``ok`` — first matching rule
    wins (see module docstring for the rationale behind this ordering).

    A non-empty ``audit.declared_files_inconclusive`` never changes the
    verdict — it is purely advisory, appended as an extra reason so a human
    reviewer can tell "some declared file's presence on the ref couldn't be
    confirmed (transient git failure)" apart from a verdict actually earned
    by confirmed facts.
    """
    verdict, reasons = _classify_core(audit)
    if audit.declared_files_inconclusive:
        reasons = [*reasons, (
            'git check inconclusive for declared file(s) '
            f'{sorted(audit.declared_files_inconclusive)} — presence on the audited '
            'ref could not be confirmed (transient git failure), not asserted as missing'
        )]
    return verdict, reasons


def _classify_core(audit: TaskProvenanceAudit) -> tuple[str, list[str]]:
    """The precedence ladder itself, over confirmed facts only. See :func:`classify`."""
    if not audit.is_ancestor:
        return 'commit_not_on_main', [
            f'cited commit {audit.commit!r} is not an ancestor of the audited '
            f'ref (git merge-base --is-ancestor reported unreachable)',
        ]

    cited = extract_cited_task_ids(audit.commit_message)
    if cited and audit.task_id not in cited:
        others = ', '.join(sorted(cited))
        return 'misattributed', [
            f'commit message cites task(s) {others}, not task {audit.task_id} — '
            f'likely proof of a different task, not this one',
        ]

    if audit.revert_commit or audit.declared_files_missing_on_main:
        reasons: list[str] = []
        if audit.revert_commit:
            reasons.append(
                f'commit was later reverted by {audit.revert_commit} '
                f'(a "This reverts commit" marker was found on the audited ref)',
            )
        if audit.declared_files_missing_on_main:
            missing = ', '.join(audit.declared_files_missing_on_main)
            reasons.append(f'declared file(s) missing from the ref HEAD: {missing}')
        return 'reverted', reasons

    if audit.declared_files and not any(f in audit.commit_files for f in audit.declared_files):
        return 'deliverable_absent', [
            f'none of the declared file(s) {sorted(audit.declared_files)} appear in the '
            f'cited commit\'s diff',
        ]

    if not audit.declared_files and audit.task_id not in cited:
        return 'unverifiable', [
            'no declared files and the commit message does not cite this task — '
            'nothing to verify the found_on_main claim against',
        ]

    return 'ok', []


# ---------------------------------------------------------------------------
# Git I/O (injected) — thin async subprocess wrappers, each degrading to a
# safe/benign default on failure rather than raising (mirrors
# invalidate_fabricated_shipping_edges.py's _git_show_files shape).
# ---------------------------------------------------------------------------

async def _git_show_files(project_root: str, commit: str) -> list[str]:
    """Return the list of files in a commit diff, or [] on failure.

    Passes ``--first-parent -m`` so a no-ff merge commit reports the files
    it actually brought in. Plain ``git show --name-only`` reports an EMPTY
    list for a merge commit (verified empirically) — merge commits are
    exactly what found_on_main provenance is expected to cite (the
    orchestrator briefs agents to cite the merge/landing SHA on main), so
    without this flag ``deliverable_absent`` would false-positive on most
    legitimate found_on_main tasks. ``--first-parent -m`` is a no-op for an
    ordinary single-parent commit (verified empirically), so this is a
    strict improvement over the prior invocation.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', '-C', project_root, 'show', '--name-only', '--format=',
            '--first-parent', '-m', commit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            return []
    except FileNotFoundError:
        return []
    if proc.returncode != 0:
        return []
    return [
        ln.strip() for ln in stdout.decode('utf-8', errors='replace').splitlines()
        if ln.strip()
    ]


async def _git_is_ancestor(project_root: str, commit: str, ref: str) -> bool:
    """Return True iff *commit* is reachable from *ref*.

    Mirrors ``orchestrator.git_ops.GitOps.is_ancestor``'s rc==0 check over
    ``git merge-base --is-ancestor``. Any failure to confirm ancestry —
    unrelated commit, invalid object, missing git binary, or a timeout —
    degrades to False rather than raising. False is the conservative
    default here: it feeds classify()'s highest-precedence verdict
    (``commit_not_on_main``), so "couldn't confirm" and "confirmed not an
    ancestor" both surface as the loud outcome rather than silently passing.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', '-C', project_root, 'merge-base', '--is-ancestor', commit, ref,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            return False
    except FileNotFoundError:
        return False
    return proc.returncode == 0


async def _git_find_revert(project_root: str, commit: str, ref: str) -> str | None:
    """Return the sha of the commit on *ref* that reverted *commit*, or None.

    Searches ``git log <commit>..<ref>`` (commits reachable from *ref* but
    not from *commit* — i.e. everything that landed afterward) for the
    canonical ``This reverts commit <full-sha>`` trailer ``git revert``
    writes. Returns the newest match (git log's default order) when found;
    None on no match or any git failure.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', '-C', project_root, 'log', f'{commit}..{ref}',
            f'--grep=This reverts commit {commit}', '--format=%H',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            return None
    except FileNotFoundError:
        return None
    if proc.returncode != 0:
        return None
    lines = [
        ln.strip() for ln in stdout.decode('utf-8', errors='replace').splitlines()
        if ln.strip()
    ]
    return lines[0] if lines else None


async def _git_files_missing_on_ref(
    project_root: str, files: list[str], ref: str,
) -> tuple[list[str], list[str]]:
    """Return ``(confirmed_missing, inconclusive)`` subsets of *files* at *ref*.

    Batches the existence check into a single ``git ls-tree -r --name-only
    <ref>`` call and set-differences *files* against it, rather than
    spawning one ``cat-file -e`` subprocess per file — one subprocess for
    the whole declared-files list regardless of how many there are.

    This single call also cleanly separates two different signals a
    per-file check would conflate: a clean ``ls-tree`` run (rc==0) makes
    every non-listed file a *confirmed* absence (``confirmed_missing``). A
    git-invocation failure (timeout / missing binary / non-zero rc) can't
    confirm anything, so ALL of *files* come back in *inconclusive*
    instead — never silently folded into ``confirmed_missing``. This
    matters because ``confirmed_missing`` feeds the ``reverted`` verdict
    (see :func:`classify`); an audit-time infra flake must not be
    indistinguishable from a genuine revert.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', '-C', project_root, 'ls-tree', '-r', '--name-only', ref,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            return [], list(files)
    except FileNotFoundError:
        return [], list(files)
    if proc.returncode != 0:
        return [], list(files)
    present = {
        ln.strip() for ln in stdout.decode('utf-8', errors='replace').splitlines()
        if ln.strip()
    }
    missing = [path for path in files if path not in present]
    return missing, []


async def _git_commit_message(project_root: str, commit: str) -> str:
    """Return the full raw commit message (subject + body) for *commit*, or '' on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', '-C', project_root, 'show', '-s', '--format=%B', commit,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
        except TimeoutError:
            proc.kill()
            return ''
    except FileNotFoundError:
        return ''
    if proc.returncode != 0:
        return ''
    return stdout.decode('utf-8', errors='replace').strip()


class GitFacts:
    """Live git-inspection facade over a project's working tree.

    Thin async wrapper around the module-level ``_git_*`` subprocess
    helpers above, injected into :func:`build_audit_report` so its
    orchestration logic is unit-testable against a fake with canned
    per-commit facts (no live repo required).
    """

    def __init__(self, project_root: str) -> None:
        self.project_root = project_root

    async def gather(
        self, commit: str, ref: str, declared_files: list[str],
    ) -> dict[str, Any]:
        """Gather every git fact :func:`classify` needs about *commit* vs *ref*.

        Checks ``is_ancestor`` first and short-circuits the remaining four
        subprocess calls (commit message/diff, revert-log search, and the
        batched declared-files existence check) when it is False:
        ``classify()`` resolves to ``commit_not_on_main`` at its very first
        precedence check in that case and never consults any of the other
        facts, so gathering them would be pure wasted subprocess work.
        """
        is_ancestor = await _git_is_ancestor(self.project_root, commit, ref)
        if not is_ancestor:
            return {
                'is_ancestor': False,
                'commit_subject': '',
                'commit_message': '',
                'commit_files': [],
                'revert_commit': None,
                'declared_files_missing_on_main': [],
                'declared_files_inconclusive': [],
            }
        message = await _git_commit_message(self.project_root, commit)
        missing, inconclusive = await _git_files_missing_on_ref(
            self.project_root, declared_files, ref,
        )
        return {
            'is_ancestor': True,
            'commit_subject': message.splitlines()[0] if message else '',
            'commit_message': message,
            'commit_files': await _git_show_files(self.project_root, commit),
            'revert_commit': await _git_find_revert(self.project_root, commit, ref),
            'declared_files_missing_on_main': missing,
            'declared_files_inconclusive': inconclusive,
        }


# ---------------------------------------------------------------------------
# Orchestration (async)
# ---------------------------------------------------------------------------

async def build_audit_report(
    tasks: list[dict], git: Any, ref: str = 'main',
) -> dict[str, Any]:
    """Select found_on_main tasks, gather git facts for each, classify, aggregate.

    *git* is an injected facts provider exposing an async
    ``gather(commit, ref, declared_files) -> dict`` method (see
    :class:`GitFacts`) — all git access is routed through it, so this
    function is fully testable against a fake with canned per-commit facts.

    Returns a report dict:
      - ``ref``: the audited ref, echoed back.
      - ``dry_run``: always ``True`` — this function only ever reports; the
        separate :func:`apply_audit_annotations` call is what may mutate.
      - ``total``: number of found_on_main tasks audited.
      - ``verdict_counts``: dict with all of ``_ALL_VERDICTS`` as keys (0 for
        verdicts that didn't occur), so the shape never silently omits one.
      - ``tasks``: per-task detail list — ``{task_id, verdict, commit,
        commit_subject, reasons}`` — in the same deterministic (by
        ``int(task_id)``) order :func:`select_found_on_main_tasks` already
        produces. ``commit_subject`` is carried through for human triage of
        flagged (``misattributed``/``reverted``/etc.) verdicts even though
        ``classify()`` itself doesn't consult it.
    """
    audits = select_found_on_main_tasks(tasks)

    verdict_counts: dict[str, int] = dict.fromkeys(_ALL_VERDICTS, 0)
    task_details: list[dict[str, Any]] = []
    for audit in audits:
        facts = await git.gather(audit.commit, ref, audit.declared_files)
        # Conservative-False default: an unconfirmed/omitted is_ancestor fact
        # must surface as the loud commit_not_on_main verdict, mirroring
        # _git_is_ancestor's own fail-safe-False semantics, rather than
        # silently assuming "on main". GitFacts.gather always supplies the
        # key; this default only matters for an alternate facts provider.
        audit.is_ancestor = facts.get('is_ancestor', False)
        audit.commit_subject = facts.get('commit_subject', '')
        audit.commit_message = facts.get('commit_message', '')
        audit.commit_files = facts.get('commit_files') or []
        audit.revert_commit = facts.get('revert_commit')
        audit.declared_files_missing_on_main = facts.get('declared_files_missing_on_main') or []
        audit.declared_files_inconclusive = facts.get('declared_files_inconclusive') or []

        verdict, reasons = classify(audit)
        audit.verdict = verdict
        audit.reasons = reasons

        verdict_counts[verdict] += 1
        task_details.append({
            'task_id': audit.task_id,
            'verdict': verdict,
            'commit': audit.commit,
            'commit_subject': audit.commit_subject,
            'reasons': reasons,
        })

    return {
        'ref': ref,
        'dry_run': True,
        'total': len(audits),
        'verdict_counts': verdict_counts,
        'tasks': task_details,
    }


# ---------------------------------------------------------------------------
# Apply layer (non-destructive)
# ---------------------------------------------------------------------------

async def apply_audit_annotations(
    backend: Any,
    project_root: str,
    report: dict[str, Any],
    *,
    tag: str | None = None,
) -> dict[str, Any]:
    """Annotate every flagged task's ``metadata.x_provenance_audit``.

    Mirrors ``audit_duplicate_tasks.apply_changes``'s per-op try/except
    counter pattern: each ``update_task`` call is isolated so one failure
    does not abort the batch. Only verdicts in ``_FLAGGED_VERDICTS`` are
    touched — ``ok``/``unverifiable`` tasks are left completely alone.

    This function NEVER reopens a done task (no ``set_task_status`` call
    of any kind) — every flagged task id is instead surfaced under
    ``needs_human_review`` in the returned summary, regardless of whether
    its annotation write succeeded, so a human can decide whether to
    reopen it to pending or otherwise disposition it.

    Returns:
        ``{'annotated': int, 'errors': int, 'needs_human_review': [ids]}``.
    """
    annotated = 0
    errors = 0
    needs_human_review: list[str] = []
    ref = report.get('ref', 'main')

    for detail in report.get('tasks', []):
        verdict = detail.get('verdict')
        if verdict not in _FLAGGED_VERDICTS:
            continue
        task_id = detail['task_id']
        needs_human_review.append(task_id)
        annotation = {
            'verdict': verdict,
            'reasons': detail.get('reasons', []),
            'ref': ref,
            'audited_at': datetime.now(UTC).isoformat(),
        }
        try:
            await backend.update_task(
                task_id, project_root,
                metadata=json.dumps({'x_provenance_audit': annotation}),
                tag=tag,
            )
            logger.info('Annotated task %s (verdict=%s)', task_id, verdict)
            annotated += 1
        except Exception as exc:
            logger.error('Failed to annotate task %s: %s', task_id, exc)
            errors += 1

    return {
        'annotated': annotated,
        'errors': errors,
        'needs_human_review': needs_human_review,
    }


def _has_flagged_findings(report: dict[str, Any]) -> bool:
    """Return True iff *report*'s ``verdict_counts`` show any flagged verdict.

    Pure helper backing ``--fail-on-findings`` (see :func:`_run`): a report
    where every count sits in ``ok``/``unverifiable`` is "clean"; any count
    under one of ``_FLAGGED_VERDICTS`` means at least one task needs human
    disposition. Missing/absent counts are treated as 0, never as a finding.
    """
    counts = report.get('verdict_counts') or {}
    return any(counts.get(v, 0) > 0 for v in _FLAGGED_VERDICTS)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
    )

    import os  # noqa: PLC0415

    from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend  # noqa: PLC0415
    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415

    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    if config.taskmaster is None:
        logger.error('Task backend not configured in fused-memory config')
        return 1

    backend = SqliteTaskBackend(config.taskmaster)
    await backend.start()
    try:
        raw = await backend.get_tasks(args.project_root)
        tasks = raw.get('tasks') or []
        logger.info('Fetched %d task(s) from task backend', len(tasks))

        git = GitFacts(args.project_root)
        report = await build_audit_report(tasks, git, ref=args.ref)
        output = {'project': args.project, 'project_root': args.project_root, **report}
        print(json.dumps(output, indent=2, default=str))

        findings = _has_flagged_findings(report)
        fail_on_findings = args.fail_on_findings

        if not args.apply:
            logger.info(
                'Dry run — nothing was modified. Use --apply to annotate flagged tasks.',
            )
            if fail_on_findings and findings:
                logger.warning(
                    '--fail-on-findings: flagged verdict(s) present in dry-run report',
                )
                return 2
            return 0

        result = await apply_audit_annotations(backend, args.project_root, report)
        logger.info(
            'Applied: annotated %d flagged task(s); %d error(s); needs_human_review=%s',
            result['annotated'], result['errors'], result['needs_human_review'],
        )
        if result['errors'] > 0:
            return 1
        if fail_on_findings and findings:
            logger.warning(
                '--fail-on-findings: flagged verdict(s) present after apply '
                '(see needs_human_review)',
            )
            return 2
        return 0
    finally:
        await backend.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project', default='dark_factory',
        help='project_id label for the printed report (default: dark_factory)',
    )
    parser.add_argument(
        '--project-root', required=True,
        help='Absolute filesystem path to the project (required by Taskmaster + git)',
    )
    parser.add_argument(
        '--config', default=None,
        help='Path to fused-memory config file (sets CONFIG_PATH env var)',
    )
    parser.add_argument(
        '--ref', default='main',
        help='Git ref to audit commit lineage against (default: main)',
    )
    parser.add_argument(
        '--apply', action='store_true',
        help=(
            "Annotate flagged tasks' metadata.x_provenance_audit "
            '(default: dry-run, report only). Never reopens a done task.'
        ),
    )
    parser.add_argument(
        '--fail-on-findings', action='store_true',
        help=(
            'Exit 2 when verdict_counts shows any flagged verdict '
            '(commit_not_on_main/misattributed/reverted/deliverable_absent), '
            'in dry-run or --apply mode alike — distinct from the exit-1 '
            'apply-error code. Default: always exit 0 on a successful audit '
            'regardless of findings. Useful for wiring this script into CI '
            'as a guard so a clean exit only ever means "nothing flagged".'
        ),
    )
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == '__main__':
    sys.exit(main())
