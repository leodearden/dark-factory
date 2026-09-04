"""Repo-level guard: the consolidated atomic-write pattern cannot silently regrow.

WHAT THIS IS.  An AST sweep over SIX first-party source trees — ``shared/src``,
``orchestrator/src``, ``escalation/src``, ``fused-memory/src``,
``fused-memory/scripts`` and ``scripts`` — asserting that every
rename-into-place (the tmp+``os.replace`` shape task 3223 consolidated into
``shared.safe_io.atomic_write_text``) is a known, individually-reasoned
survivor rather than a fresh hand-rolled copy.

WHY IT LIVES HERE AND NOT IN A PACKAGE SUITE (task 3388).  It was born in
``shared/tests/test_safe_io.py``, beside the implementation it protects, and
that was defensible while it scanned three trees and shared owned the blessed
writer.  It stopped being defensible once the guard asserted on five packages
shared does not own: a guard that asserts on five packages cannot live inside a
sixth package's tests.  The concrete failure it created — not hypothetical, and
recorded in the allowlist itself — is that
``orchestrator/src/orchestrator/digest.py::write_digest_entry`` is allowlisted
AND flagged there as a prime migration candidate, so migrating it turned the
SHARED suite red, at a site no orchestrator author would think to look.  Moving
the guard to the repo level puts the failure where the fix is.

THE PLACEMENT IS THE ESTABLISHED PATTERN, not a stray file.  ``tests/scripts/``
is a registered module config (``tests/scripts/orchestrator.yaml``) and already
holds this exact class of repo-wide structural sweep:

  * ``test_nonmember_ruff_config.py`` — sweeps every non-member directory for
    ruff rule-set parity with a live workspace member.
  * ``test_pytest_workspace_collection.py`` — cross-checks every member's
    pytest markers.
  * ``test_module_verify_budgets.py`` — cross-checks every module yaml's
    verify budget.

Being repo-level is also what makes this module's two HARD assertions correct:
``assert root.is_dir()`` in ``_iter_source_files`` and the stale-entry assertion
in ``test_no_unapproved_renamers_in_source_trees``.  At the repo root every
declared tree is guaranteed present, so neither needs the ``pytest.skip`` /
warning-downgrade fallback that a package-local guard would have had to adopt.
Both carry a comment saying so at their site.

COLLECTED TWICE, DELIBERATELY.  ``tests/scripts/orchestrator.yaml`` collects
this directory, and so does ``scripts/orchestrator.yaml``
(``pytest tests/scripts/ scripts/tests/``) — so a diff touching ``scripts/``
runs the guard that scans ``scripts/``.  No yaml edit was needed to register
this file: all three commands there are directory-wide.

SCOPE LIMIT, stated because a green run here must not be over-read: relocating
this guard did NOT make ``shared/tests`` standalone-runnable.  Five other
cross-tree gates still live there; they are enumerated, with measured counts, in
``test_atomic_write_guard_does_not_scan_sibling_package_trees`` below.
"""
import ast
import functools
import re
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

# tests/scripts/<file>.py and shared/tests/<file>.py are both exactly two levels
# below the repo root, so this constant carried over verbatim from the guard's
# old home.  Shared by the relocated guard and by the boundary tests at the
# bottom of this module.
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Anti-regrowth guard (task 3223)
# ---------------------------------------------------------------------------
#
# Task 3223 consolidated ten hand-rolled tmp+rename writers into
# ``atomic_write_text`` above.  Nothing stops the eleventh from being written
# by hand next month — the pattern is short enough to look harmless, which is
# exactly how the first ten accumulated (two of them carried a docstring
# saying they were copied because "there is no atomic writer in shared/ to
# reuse").  These tests are the fence: a NEW rename-into-place inside the
# SIX TREES LISTED BELOW fails loudly and points the author at this module.
#
# SCOPE LIMIT — the fence is narrower than "the repo", and saying so matters
# more than the fence looking complete.  Task 3388 widened ``_SRC_TREES`` from
# three trees to six: ``shared/src``, ``orchestrator/src``, ``escalation/src``,
# ``fused-memory/src``, ``fused-memory/scripts`` and ``scripts``.
#
# THE COUNT, CORRECTED BY MEASUREMENT.  Widening surfaced SEVENTEEN unmigrated
# hand-rolled writers, not the six this block used to enumerate — and not the
# six the follow-up ticket that tracked the widening enumerated either.  Both
# of those counts were arrived at by reading; the detector disagreed with both.
# So: trust ``_find_renamers`` over any prose count in this file, this sentence
# included, and re-run it rather than re-trusting the sentence.  A fence sized
# from stale prose is not a fence, which is the whole reason 3388 existed.
#
# STILL NOT SCANNED, stated with COUNTS rather than bare directory names so the
# claim is falsifiable (measured at main 86db695984): ``cockpit/src`` 2
# (priority.py::save_priorities, ui_config.py::save_ui_config), filed as a
# follow-up; ``dashboard/src`` 0; ``sampler/src`` 0; ``skills`` 0; the
# repo-root ``tests/`` 0.
#
# TEST DIRECTORIES.  ``fused-memory/scripts`` and ``scripts`` are scanned
# WHOLESALE, tests included, rather than through an exclusion mirroring the
# ``*/src`` convention: a flat operator directory has no ``src/`` boundary to
# mirror, and the one place a regrown production writer could hide is precisely
# a directory somebody decided not to look at.  That is free today — the test
# dirs INSIDE the scanned trees are clean: ``shared/tests`` 0, ``fused-memory/
# tests`` 0, ``scripts/tests`` 0.  Stated in that scoped form deliberately: the
# general claim "test trees are clean" is FALSE at this base, because
# ``orchestrator/tests`` carries 2 renamers and ``escalation/tests`` 1.  Neither
# sits inside a scanned tree, so neither affects red/green — but they are live
# evidence that test directories do accumulate this pattern over time, and so
# that an exclusion would not have stayed harmless.

_SRC_TREES = (
    'shared/src',
    'orchestrator/src',
    'escalation/src',
    'fused-memory/src',
    'fused-memory/scripts',
    'scripts',
)

# Every (module, function) in the six trees that renames a path into place,
# with the reason it is not calling atomic_write_text.  Adding an entry is a
# deliberate act that needs a reason; growing this set silently is the failure
# mode the guard exists to prevent.
_ALLOWED_RENAMERS = {
    ('shared/src/shared/safe_io.py', 'atomic_write_text'):
        'THE consolidated implementation — the one blessed home for this pattern.',
    ('shared/src/shared/safe_io.py', 'load_json_or_warn'):
        'Quarantine rename of a corrupt file (<name>.corrupt); not a write path.',
    ('orchestrator/src/orchestrator/session_registry.py', '_atomic_write_text'):
        'DELIBERATE EXCEPTION to task 3223, not an oversight. The module is '
        'invoked by absolute path from skills/spawn/spawn-claude.sh under an '
        'interpreter with no venv/install/workspace packages, so its docstring '
        "declares it stdlib-only; a module-scope `from shared import safe_io` "
        'made it unimportable there (ModuleNotFoundError) and silently broke '
        'the spawn hook. Pinned by test_session_registry.py::'
        'TestStdlibOnlySelfContainment. Do not "finish" this migration.',
    ('orchestrator/src/orchestrator/session_hooks.py', '_run_install'):
        'Out of task 3223 enumerated scope; left alone deliberately.',
    ('orchestrator/src/orchestrator/verify_cancel.py', 'write_pgid_file'):
        'Out of task 3223 enumerated scope; left alone deliberately.',
    ('escalation/src/escalation/queue.py', 'EscalationQueue._atomic_write_path'):
        'Out of task 3223 enumerated scope. This is the helper 3223 was MODELLED '
        'on (its durable= flag became fsync=); a follow-up may migrate it.',
    ('escalation/src/escalation/queue.py', 'EscalationQueue._archive_resolved'):
        'Moves a resolved file into the dated archive dir; not a write path.',
    ('escalation/src/escalation/sweep.py', '_atomic_move'):
        'Out of task 3223 enumerated scope; moves an existing file, does not write.',
    ('orchestrator/src/orchestrator/digest.py', 'write_digest_entry'):
        'Out of task 3223 enumerated scope. Notable: b3_gate._save_state was '
        'originally documented as "modelled on digest.py" — this is where one '
        'of the ten copies came from, so it is a prime candidate for the '
        'follow-up migration.',
    ('orchestrator/src/orchestrator/evals/rereview.py', 'atomic_write_json'):
        'Out of task 3223 enumerated scope; left alone deliberately.',
    (
        'orchestrator/src/orchestrator/service_restart.py',
        'StaleServiceRestartCoordinator._persist_last_fire_wall',
    ):
        'Out of task 3223 enumerated scope; left alone deliberately.',
    ('orchestrator/src/orchestrator/agents/invoke.py', '_invoke_pi'):
        'Swaps .mcp.json with a backup and back again; renames existing files '
        'rather than writing new content, so atomic_write_text does not apply.',
    # --- Sites that landed on main AFTER this task branched. Both were triaged
    # on their merits when the branch was rebased; neither is "out of scope"
    # boilerplate, because each has a concrete semantic atomic_write_text
    # cannot express today. Stated here rather than migrated, because a
    # migration that quietly dropped one of these is exactly the silent
    # regression this task exists to avoid.
    ('shared/src/shared/transcript_archive.py', '_archive_one'):
        'COPIES an existing file rather than writing new text, so it is the '
        'same category as escalation sweep._atomic_move / queue._archive_resolved. '
        'Two specifics rule out atomic_write_text even as a rewrite: (1) the '
        'payload is a multi-MB agent-session JSONL moved with shutil.copyfile '
        'on the platform fast-copy path, so peak RSS stays flat regardless of '
        'transcript size, whereas atomic_write_text takes a str already fully '
        'in memory; (2) os.utime mirrors the SOURCE mtime onto the staging file '
        'BEFORE the replace, and atomic_write_text exposes no pre-replace seam '
        'for that — a now-stamped archive reads to gc_agent_transcripts as a '
        'reset retention age.',
    ('shared/src/shared/transcript_archive.py', 'restore_archived_transcript'):
        'The INVERSE of _archive_one above and in exactly its category: it '
        'COPIES an existing archived transcript back into a config dir rather '
        'than writing new text. The same two specifics rule out '
        'atomic_write_text: (1) the payload is a multi-MB agent-session JSONL '
        'moved with shutil.copyfile (or streamed through gzip.open + '
        'copyfileobj for the pre-3618 .jsonl.gz corpus) so peak RSS stays flat, '
        'whereas atomic_write_text takes a str already fully in memory; '
        '(2) os.utime mirrors the ARCHIVE mtime onto the staging file BEFORE '
        'the replace, and atomic_write_text exposes no pre-replace seam for '
        'that — a now-stamped restore reads to the next archival pass as newer '
        'than its own archive and is pointlessly re-archived over it. The '
        'staging+os.replace publish is load-bearing rather than decorative: the '
        'claude CLI PARSES the transcript rather than stat-ing it (a zero-byte '
        'file and a preamble-only file both yield `No conversation found with '
        'session ID`, measured on CLI 2.1.236), so a torn restore would arm '
        '--resume against a file the CLI then rejects. Added by task 3578.',
    ('shared/src/shared/transcript_archive.py', '_move_to_archive'):
        'MOVES an existing transcript rather than writing new text — the '
        'sibling of _archive_one above, and the same category as escalation '
        'sweep._atomic_move / queue._archive_resolved. atomic_write_text does '
        'not merely fit badly here, it has nothing to be handed: there is no '
        'content string, only an inode to relink. The os.rename IS the '
        'operation, not a rename-into-place finishing a write, so the '
        'structural detector that flags it is reading the shape correctly and '
        'the shape is correct. Added by task 3619, which made archival a '
        'precondition of config-dir deletion and needed the O(1) metadata move '
        'to keep that affordable inside a synchronous teardown. Two properties '
        'the copy path has to work for and this one gets from the rename '
        'itself: atomicity within a filesystem (no staging sibling, so a '
        'truncated transcript cannot appear at the canonical path) and mtime '
        'preservation via the surviving inode (a now-stamped archive would '
        'reset gc_agent_transcripts retention age and defeat the '
        'already-current skip). The EXDEV branch delegates to _archive_one '
        'when the rename is physically impossible, so the cross-device case '
        'is handled by the entry above rather than by a second writer here.',
    ('orchestrator/src/orchestrator/mcp/plan_tools.py', '_atomic_write_plan'):
        'Cannot delegate without losing three semantics atomic_write_text does '
        'not offer. (1) SYMLINK RESOLUTION: it writes to os.path.realpath(path) '
        'and refuses a dangling link; atomic_write_text replaces the path as '
        'given, so an os.replace onto the lane plan.json symlink would swap the '
        'LINK for a regular file and re-fork the lane/meta-root copies (the '
        'esc-5205-9 divergence that symlink exists to prevent). (2) PRE-REPLACE '
        'VERIFICATION: _verify_plan_json re-parses the TEMP file after the '
        'chmod and before the swap — a deliberately named seam a test injects '
        'into, at the last reversible checkpoint; atomic_write_text has no '
        'pre-replace inspection hook, and verifying after the swap is backwards. '
        '(3) FSYNC ASYMMETRY: it fsyncs the temp file but NOT the parent dir, '
        'while atomic_write_text does both under fsync=True and neither under '
        'fsync=False, so no setting reproduces it. It also funnels every '
        'failure into PlanWriteError naming both the original and resolved '
        'paths. A follow-up may widen atomic_write_text; it must not be forced '
        'through the current signature.',
    # --- Sites surfaced by task 3388's widening of _SRC_TREES to six trees.
    # Each reason below was written from READING that site, not pattern-matched
    # from a neighbour: an entry whose reason came from the entry above it is
    # indistinguishable from a triaged decision and is worse than no entry.
    # Where an argument already exists above, it is CITED rather than re-argued.
    # NOTE: this file's OUTER ``CuratorEscalator._persist_state`` used to need an
    # entry here too — a "DETECTOR-NESTING ARTIFACT" line stating outright that
    # it had nothing to migrate and no semantic justification to offer.  It is
    # gone because the detector was fixed rather than the allowlist widened
    # (amendment to task 3388): _find_renamers now attributes a rename to its
    # INNERMOST enclosing function only, so the closure below is reported and
    # its enclosing method is not.  Recorded rather than silently deleted,
    # because an artifact entry is not free: an _ALLOWED_RENAMERS key also
    # silences test_atomic_write_text_helpers_only_delegate for that (file,
    # name) pair, so every artifact entry widened a real hole while diluting
    # the list's stated invariant that each entry is an individually-reasoned
    # survivor.
    (
        'fused-memory/src/fused_memory/middleware/curator_escalator.py',
        'CuratorEscalator._persist_state._write',
    ):
        'Out of task 3223 enumerated scope; left alone deliberately. This is '
        'the real writer, and a migration must preserve two properties '
        'atomic_write_text does not offer today: a per-writer temp name '
        '``<state>.{pid}.{id(payload)}.tmp`` (mirroring event_queue.py\'s '
        'disjoint-path discipline, so two writers can never share a temp path '
        'even if they bypass _persist_lock), and the asyncio.to_thread offload '
        'that keeps the blocking I/O off the event loop under burst load.',
    ('fused-memory/src/fused_memory/reconciliation/backlog_policy.py', 'BacklogPolicy._restore_policy_keys'):
        'Out of task 3223 enumerated scope; left alone deliberately. Re-merges '
        'the policy-only keys that closing a halt strips (project_id, '
        'error_type, backlog, threshold) back onto the persisted record via '
        'tmp.write_text + tmp.replace. Best-effort by construction — every '
        'failure is logged and swallowed so a record that IS closed but lost '
        'its forensic keys is never misreported as un-closed.',
    ('fused-memory/src/fused_memory/reconciliation/event_queue.py', 'EventQueue._rotate_dead_letter'):
        'MOVES existing files: cascade-rotates dead_letter.jsonl -> .1 -> .2 '
        '-> ... There is no content string, only os.replace(src, dst) per '
        'slot, so this is the same class as escalation sweep._atomic_move / '
        'queue._archive_resolved and atomic_write_text has nothing to be '
        'handed.',
    ('fused-memory/src/fused_memory/reconciliation/event_queue.py', 'EventQueue.replay_dead_letters'):
        'MOVES an existing file: os.replace snapshots each dead-letter file '
        'onto a ``.replaying`` sibling as an atomic CLAIM — race-safe against '
        'concurrent _write_dead_letter appends and self-guarding against '
        're-replay, since a second call finds nothing left to snapshot. A '
        'claim, not a write; same class as escalation sweep._atomic_move / '
        'queue._archive_resolved.',
    ('fused-memory/src/fused_memory/server/manifest_stamping.py', '_stamp_capability_manifests_impl'):
        'Out of task 3223 enumerated scope; left alone deliberately. A genuine '
        'writer: yaml.safe_dump into a uniquely-named '
        '``<sidecar>.{pid}.{id(raw)}.tmp`` sibling, then os.replace. Its '
        'surrounding comment already documents the one tradeoff a migration '
        'must not silently change — a hard kill between write and replace can '
        'leave a .tmp sibling, which is harmless only because sidecar '
        'discovery matches the exact derived rel path and never a .tmp suffix.',
    ('fused-memory/scripts/bake_off_storage_shape.py', '_atomic_write_text'):
        'Out of task 3223 enumerated scope; left alone deliberately — but note '
        'what this entry COSTS, because leaving that implicit would hide a '
        'silenced assertion behind an allowlist line. '
        'test_atomic_write_text_helpers_only_delegate ``continue``s on '
        'anything in _ALLOWED_RENAMERS, so this entry also silences the '
        'delegate check for this name: a NON-delegating helper carrying the '
        'consolidated name — a full inlined mkstemp + os.replace body — '
        'survives here on purpose, in a one-off benchmark script. It is the '
        'single strongest candidate for the next migration, and it is the '
        'reason 3388 widened to fused-memory/scripts rather than stopping at '
        'fused-memory/src as its own ticket text proposed.',
    ('fused-memory/scripts/memory_eval_retrieval_probe.py', 'write_report_text'):
        'Out of task 3223 enumerated scope; left alone deliberately. Its '
        'docstring already argues the atomicity (the memory-eval leaves share '
        'one artifact root that the dashboard reads as plain files, so a '
        'truncated report beside a valid metrics artifact is the one state to '
        'exclude) and records that the mechanism is COPIED rather than '
        'imported because shared\'s own _atomic_write_text is module-private. '
        'That reason is retired by the PUBLIC safe_io.atomic_write_text, which '
        'is what makes this a migration candidate rather than an exception.',
    ('fused-memory/scripts/memory_eval_staleness_sweep.py', 'write_report_text'):
        'Out of task 3223 enumerated scope; left alone deliberately. The '
        'sibling of memory_eval_retrieval_probe.write_report_text above — its '
        'own docstring says the mechanism is copied \'(β does the same)\' — '
        'same shared artifact root, same mkstemp + os.replace, same candidacy. '
        'Migrate the two together or the copy-from-a-neighbour habit survives.',
    ('scripts/consume_redispatch_requests.py', 'archive_request'):
        'MOVES an existing file: os.replace of an APPLIED request into the '
        '``consumed/`` subdirectory, to keep an audit trail of what was '
        'actioned that the snapshot directory cannot provide. Same class as '
        'escalation sweep._atomic_move / queue._archive_resolved.',
    ('scripts/dashboard-watchdog.py', 'save_state'):
        'STDLIB-ONLY STANDALONE ENTRYPOINT — the same constraint already '
        'recorded above for session_registry._atomic_write_text, so that '
        'argument is not restated. This is a ``#!/usr/bin/env python3`` '
        'systemd/cron oneshot; verified by reading its imports (contextlib, '
        'json, os, subprocess, sys, tempfile, time, urllib) that it pulls in '
        'nothing outside the stdlib, so a module-scope `from shared import '
        'safe_io` would need an install or a sys.path graft it does not have. '
        'This is NOT a blanket \'scripts/ cannot import shared\': '
        'scripts/legibility/census.py does exactly that, via a '
        '__file__-relative sys.path insert.',
    ('scripts/legibility/census.py', 'advance_census_state'):
        'Out of task 3223 enumerated scope; left alone deliberately. '
        'mkstemp(prefix=\'.census-state-\') + os.replace. This module is '
        'census-state.json\'s SOLE writer, so a migration must keep the '
        'always-present ``last_census_done_count`` key — serialised as JSON '
        'null when the count could not be observed, never omitted as falsy — '
        'which is what lets zeta\'s compute_tasks_landed fail safe.',
    ('scripts/legibility/codebook.py', 'dump'):
        'Out of task 3223 enumerated scope; left alone deliberately. '
        'mkstemp(prefix=\'.codebook-\') + os.replace of a canonical '
        'block-style yaml document written behind a fixed HEADER comment. Sole '
        'writer of the legibility pipeline\'s canonical registry, and '
        'byte-stable given byte-stable input so a no-change night commits '
        'nothing — a migration must not perturb either property.',
    ('scripts/legibility/trickle_state.py', 'record_run'):
        'Out of task 3223 enumerated scope; left alone deliberately. '
        'Same-directory mkstemp + os.replace of the trickle-state document the '
        'function also returns; ordinary write path with no seam of its own.',
    ('scripts/migrate_transcript_archive_gunzip.py', 'gunzip_one'):
        'PRE-REPLACE SEAM, and the same one already recorded above for shared '
        'transcript_archive._archive_one — cited rather than re-argued. It '
        'decompresses to a staging sibling, corroborates the read-back, then '
        'os.utime mirrors the SOURCE mtime onto staging BEFORE the replace, '
        'because that mtime is the retention age gc_agent_transcripts keys on '
        'and the source is gone by the end. atomic_write_text exposes no '
        'pre-replace seam, and the payload is decompressed bytes on disk '
        'rather than a str already in memory.',
    ('scripts/orchestrator-watchdog.py', '_atomic_write_json'):
        'STDLIB-ONLY STANDALONE ENTRYPOINT — as dashboard-watchdog.save_state '
        'above, and session_registry._atomic_write_text before it. A '
        '``#!/usr/bin/env python3`` systemd/cron oneshot; verified by reading '
        'its imports that the only addition beyond dashboard-watchdog\'s set '
        'is shlex, i.e. still stdlib. Note this IS already a consolidation '
        '(task 3764 extracted it so the mkdir/mktemp/write/rename dance is '
        'defined once for every watchdog state file rather than per call '
        'site), so the duplication here is one deep, not per-writer.',
    ('scripts/sweep_toolcall_markup.py', 'write_repaired'):
        'PRE-REPLACE VERIFICATION SEAM — the same seam already recorded above '
        'for plan_tools._atomic_write_plan, whose ordering this function\'s '
        'docstring says it follows, so that argument is not restated. Between '
        'reading and swapping it RE-READS the target and confirms it is still '
        'the bytes the repair was computed from, as late as possible before '
        'the os.replace, returning a WriteFailure (REASON_CHANGED_UNDER_US) '
        'rather than silently reverting somebody else\'s concurrent write. '
        'atomic_write_text has no pre-replace inspection hook.',
}


def _find_renamers(source: str) -> list[str]:
    """Return the qualified names of functions in *source* that rename a path.

    Covers all four spellings the repo actually uses, because a scan for any
    one of them leaves a hole big enough to hide a copy in:

    * ``os.replace(tmp, dest)`` — eight of the ten sites 3223 consolidated.
    * ``os.rename(tmp, dest)`` — ``escalation.queue._atomic_write_path``.
    * ``tmp.replace(dest)`` — ``evals.rereview``, ``service_restart``.
    * ``tmp.rename(dest)`` — ``digest.write_digest_entry``, which is the
      writer ``b3_gate._save_state`` was documented as modelled on.

    The two ``Path``-method forms are matched by arity: ``Path.replace`` and
    ``Path.rename`` take exactly one positional argument, while the far more
    common ``str.replace(old, new)`` takes two, so requiring exactly one
    positional arg and no keywords separates them without a type-inference
    pass.

    AST-based rather than a text grep on purpose: six of the migrated modules
    still *mention* ``os.replace`` in a docstring describing what
    ``atomic_write_text`` does for them, and a text scan would flag all six as
    false positives.

    Limitation, stated rather than papered over: this finds the rename, which
    is the half of the pattern that cannot be omitted.  A copy that factored
    its rename out into a helper would be attributed to that helper instead —
    including a helper NESTED inside it, since attribution stops at the
    innermost enclosing function (see :func:`_own_body`).  That is a deliberate
    narrowing, not a hole: the nested function gets its own qualname from
    ``visit`` below and its own allowlist entry, so the rename is still
    reported, just at one site instead of a chain of them.
    """
    return _find_renamers_in_tree(ast.parse(source))


def _own_body(fn: ast.AST) -> Iterator[ast.AST]:
    """Yield the nodes of *fn* that belong to *fn* rather than to a nested def.

    ``ast.walk`` descends into nested ``FunctionDef`` nodes, so a rename inside
    a closure used to be attributed to the closure AND to every function
    lexically enclosing it.  That is not free: each spurious qualname needs its
    own ``_ALLOWED_RENAMERS`` key, an allowlist key ALSO silences
    ``test_atomic_write_text_helpers_only_delegate`` for that (file, name)
    pair, and an entry whose only honest reason is "the detector did this"
    dilutes the list's invariant that every entry is a reasoned survivor.

    ``Lambda`` is deliberately NOT cut here, and the asymmetry with
    ``FunctionDef`` is the whole point: ``visit`` assigns qualnames to
    ``FunctionDef``/``AsyncFunctionDef``/``ClassDef`` only, so a nested def has
    a name of its own to carry the attribution while a lambda has none.
    Cutting lambdas too would attribute ``f = lambda p: p.replace(dest)`` to
    nothing at all — trading a duplicate report for a missed one.  A nested
    ``ClassDef`` body IS walked (its statements execute in this function's
    frame) while its methods are not (they get ``outer.Inner.method``).
    """
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        stack.extend(ast.iter_child_nodes(node))


def _find_renamers_in_tree(tree: ast.AST) -> list[str]:
    """The already-parsed half of :func:`_find_renamers`, whose docstring documents
    every detector semantic below.

    Split out so the batch scan can parse each file ONCE and hand the same tree
    to both AST tests — see :func:`_scan_source_trees` for the measurement that
    made that worth doing.
    """
    found: list[str] = []

    def is_rename(sub) -> bool:
        if not (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)):
            return False
        if sub.func.attr not in ('replace', 'rename'):
            return False
        receiver = sub.func.value
        if isinstance(receiver, ast.Name) and receiver.id == 'os':
            return True
        # Path.replace(target) / Path.rename(target): exactly one positional
        # argument.  str.replace(old, new) takes two and is excluded here.
        return len(sub.args) == 1 and not sub.keywords

    def visit(node, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f'{prefix}{child.name}.')
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qualname = f'{prefix}{child.name}'
                if any(is_rename(sub) for sub in _own_body(child)):
                    found.append(qualname)
                visit(child, f'{qualname}.')
            else:
                visit(child, prefix)

    visit(tree, '')
    return found


def _find_write_helpers(source: str, tree: ast.AST) -> list[tuple[str, bool]]:
    """``(name, delegates)`` for every ``(_)atomic_write_text`` def in *tree*.

    The structural half of ``test_atomic_write_text_helpers_only_delegate``,
    split out for the same single-parse reason as
    :func:`_find_renamers_in_tree`.  Deliberately carries NO policy: the
    ``safe_io.py``-is-the-implementation skip and the ``_ALLOWED_RENAMERS``
    skip both stay in the test, where they can be read next to the invariant
    they qualify.
    """
    return [
        (node.name, 'safe_io.atomic_write_text(' in (ast.get_source_segment(source, node) or ''))
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name in ('_atomic_write_text', 'atomic_write_text')
    ]


@functools.cache
def _read_tree(tree: str, missing_ok: bool) -> tuple[tuple[str, str], ...]:
    """Read every ``*.py`` under ONE tree — once per process, then cached.

    Cached PER TREE rather than per tuple-of-trees on purpose: ``_SRC_TREES``
    and the pointer sweep's tree list overlap, and a per-tuple cache would
    re-read the six source trees for the sweep.

    *missing_ok* decides what an absent tree MEANS, and the two answers are
    genuinely different rather than a strictness knob — see the assert below.
    Note the warning fires once per process, since a cache hit skips this body.
    """
    root = _REPO_ROOT / tree
    if not root.is_dir():
        # HARD assert for a tree this guard SCANS, kept rather than downgraded
        # to pytest.skip, and that is only correct BECAUSE this guard is now
        # repo-level (task 3388): at the repo root every scanned tree is
        # guaranteed present, so a missing one means the fence silently covers
        # less.  The skip fallback was the price of NOT relocating; having
        # relocated, do not also pay it.  Paired with
        # test_every_declared_tree_contributes_scanned_files below, which
        # catches the case this assert cannot: a tree that IS a directory and
        # rglobs nothing.
        assert missing_ok, (
            f'{tree} not found under {_REPO_ROOT} — this guard walks fixed tree '
            f'names, so a moved/renamed package must update its tree list rather '
            f'than let the guard silently scan nothing.'
        )
        # SOFT for a tree this guard only READS (see _POINTER_OPTIONAL_TREES):
        # absence costs some documentation coverage, not fence coverage, so it
        # is reported rather than fatal.  Loud, not silent — and `assert
        # pointers` still stops the sweep from covering nothing at all.
        warnings.warn(
            f'{tree} not found under {_REPO_ROOT}; skipping it. This tree is '
            f'swept for stale documentation pointers only — it is not fenced '
            f'by this guard — so its absence narrows a doc check rather than '
            f'the atomic-write check. Update _POINTER_OPTIONAL_TREES if the '
            f'package is gone for good.',
            stacklevel=2,
        )
        return ()
    return tuple(
        (py.relative_to(_REPO_ROOT).as_posix(), py.read_text(encoding='utf-8'))
        for py in sorted(root.rglob('*.py'))
    )


def _walk_trees(
    trees: tuple[str, ...], missing_ok: bool = False
) -> tuple[tuple[str, str], ...]:
    """``(relpath, source)`` for every ``*.py`` under *trees*, in tree order.

    Duplicates are NOT removed here — ``scripts/tests`` sits inside the
    ``scripts`` scan root — because only the pointer sweep spans overlapping
    trees and it dedupes by relpath itself.
    """
    return tuple(item for tree in trees for item in _read_tree(tree, missing_ok))


class _ScannedModule(NamedTuple):
    """The derived facts one scanned module contributes — never its AST."""

    relpath: str
    #: Qualnames that rename a path into place (:func:`_find_renamers`).
    renamers: tuple[str, ...]
    #: ``(name, delegates)`` per ``(_)atomic_write_text`` def.
    write_helpers: tuple[tuple[str, bool], ...]


@functools.cache
def _scan_source_trees() -> tuple[_ScannedModule, ...]:
    """Parse every file in ``_SRC_TREES`` exactly ONCE and cache the derived facts.

    WHY THIS EXISTS AND WHY IT CACHES FACTS RATHER THAN TREES — measured on
    this worktree, because the answer is not the obvious one.  Walking and
    reading the six trees is 509 files / 21.2 MB / 0.36s; ``ast.parse``-ing
    the same 509 files is 10.05s.  So the dominant cost was never the I/O the
    ``_read_tree`` cache above removes — it was the two AST tests each parsing
    the whole corpus.  They now share one pass, which is worth ~10s of this
    module's wall clock against ~1s for de-duplicating the reads.

    Caching the parsed trees instead would save the same 10s and retain
    352.5 MB (tracemalloc, same 509 files) for the rest of the pytest session,
    on a box this suite already shares with concurrent verify legs.  The
    per-file facts below are a few KB.  Re-measure before trading that back.
    """
    scanned = []
    for relpath, source in _walk_trees(_SRC_TREES):
        tree = ast.parse(source)
        scanned.append(
            _ScannedModule(
                relpath,
                tuple(_find_renamers_in_tree(tree)),
                tuple(_find_write_helpers(source, tree)),
            )
        )
    return tuple(scanned)


def _iter_source_files():
    """Yield (repo-relative posix path, source text) for every scanned tree.

    "Every scanned tree" means ``_SRC_TREES``, whatever that currently holds —
    six at the time of writing.  Stated by reference rather than by count: the
    count in this sentence used to say "the three src trees" and stayed there
    through the widening to six, which is precisely the stale-prose failure
    this module exists to prevent (task 3388).

    A thin view over the cached ``_read_tree``, which carries the hard
    ``assert root.is_dir()`` and the argument for keeping it hard.  Kept as a
    named function because the tests and comments below refer to it, and
    because "the set of files this guard scans" is a concept worth a name.
    """
    yield from _walk_trees(_SRC_TREES)


#: The one helper ``test_atomic_write_text_helpers_only_delegate`` must always
#: still be looking at.  Same role as ``_CONTROL_MODULES`` further down, for the
#: same reason: that test's skip condition is membership in
#: ``_ALLOWED_RENAMERS``, so every future allowlist entry silences one more
#: name, and a bare "did we find any?" check degrades one entry at a time until
#: it is checking nothing.  prompt_artifact is the strongest available control —
#: it is the helper ``test_prompt_artifact.py`` monkeypatches at five sites, so
#: it is load-bearing for another suite and cannot quietly disappear.  If it
#: genuinely moves, REPOINT this at another delegating helper rather than
#: deleting the assertion, which would restore the vacuity it exists to prevent.
_DELEGATE_CONTROL = 'shared/src/shared/prompt_artifact.py::_atomic_write_text'


class TestNoRegrownAtomicWriters:
    """The consolidated pattern cannot silently re-duplicate."""

    def test_detector_fires_on_a_regrown_copy(self):
        """The detector has teeth: it flags a re-inlined tmp+rename block.

        Without this, a detector that silently matched nothing would make
        every other test in this class pass vacuously.  The sample below is
        the exact shape task 3223 removed from five call sites.
        """
        regrown = (
            'import json, os\n'
            'def _save_raw(self, state):\n'
            '    tmp = self._path.with_suffix(".json.tmp")\n'
            '    tmp.write_text(json.dumps(state), encoding="utf-8")\n'
            '    os.replace(str(tmp), str(self._path))\n'
        )
        assert _find_renamers(regrown) == ['_save_raw']

        os_rename_variant = (
            'import os\n'
            'class S:\n'
            '    def _write(self, path, text):\n'
            '        os.rename(tmp, path)\n'
        )
        assert _find_renamers(os_rename_variant) == ['S._write']

        # Path-method spellings: the hole that let digest.write_digest_entry,
        # rereview.atomic_write_json and service_restart sit unseen by an
        # os.replace-only scan.
        path_method_variant = (
            'def _write(path, text):\n'
            '    tmp = path.with_suffix(".tmp")\n'
            '    tmp.write_text(text)\n'
            '    tmp.rename(path)\n'
        )
        assert _find_renamers(path_method_variant) == ['_write']

        path_replace_variant = (
            'def _write(path, text):\n'
            '    tmp.replace(path)\n'
        )
        assert _find_renamers(path_replace_variant) == ['_write']

    def test_detector_attributes_a_rename_to_its_innermost_function(self):
        """A rename in a closure names the CLOSURE, not the chain enclosing it.

        The original detector used ``ast.walk``, which descends into nested
        defs, so a nested writer was reported once per enclosing function.
        Measured at the base this amendment was written against, that produced
        exactly one spurious qualname across all six trees
        (``curator_escalator.py::CuratorEscalator._persist_state``, whose only
        ``os.replace`` lives in its nested ``_write``) and it had to be bought
        off with an allowlist entry whose stated reason was that there was no
        reason — while ALSO silencing the delegate check for that name.

        The lambda case is the other half of the contract and the reason
        ``_own_body`` cuts at ``FunctionDef`` but not at ``Lambda``: a lambda
        gets no qualname of its own, so cutting there would report the rename
        NOWHERE.  A narrowing that loses a site is worse than the duplication
        it removes.
        """
        nested = (
            'import os\n'
            'class S:\n'
            '    def _persist(self, payload):\n'
            '        def _write():\n'
            '            os.replace(tmp, self._path)\n'
            '        return _write\n'
        )
        assert _find_renamers(nested) == ['S._persist._write']

        lambda_body = (
            'def outer(dest):\n'
            '    f = lambda p: p.replace(dest)\n'
            '    return f\n'
        )
        assert _find_renamers(lambda_body) == ['outer']

        # A nested class's METHOD gets its own qualname; a rename in the class
        # BODY executes in the enclosing function's frame and stays there.
        nested_class = (
            'import os\n'
            'def outer():\n'
            '    class Inner:\n'
            '        os.replace(a, b)\n'
            '        def m(self):\n'
            '            os.replace(c, d)\n'
        )
        assert _find_renamers(nested_class) == ['outer', 'outer.Inner.m']

    def test_detector_ignores_str_replace(self):
        """``str.replace(old, new)`` is not a rename.

        The Path-method branch keys on arity, so this is the false positive
        that would fire on half the repo if the arity check regressed.
        """
        two_arg = (
            'def normalise(s):\n'
            '    return s.replace("-", "_").replace(" ", "")\n'
        )
        assert _find_renamers(two_arg) == []

    def test_detector_ignores_docstring_mentions(self):
        """A docstring describing the pattern is not an implementation of it.

        Six migrated modules still reference ``os.replace`` in prose; flagging
        those would make the guard unusable and train people to disable it.
        """
        prose_only = (
            'def _save_raw(self, state):\n'
            '    """Delegates to safe_io.atomic_write_text (tmp + os.replace)."""\n'
            '    safe_io.atomic_write_text(self._path, state)\n'
        )
        assert _find_renamers(prose_only) == []

    def test_no_unapproved_renamers_in_source_trees(self):
        """Every rename-into-place in the scanned trees is a known, reasoned survivor.

        Scanned trees = ``_SRC_TREES`` (six at the time of writing).  Read the
        tuple, not this sentence — see _iter_source_files on why no count is
        written here.
        """
        actual = {
            (module.relpath, qualname)
            for module in _scan_source_trees()
            for qualname in module.renamers
        }

        unapproved = actual - set(_ALLOWED_RENAMERS)
        assert not unapproved, (
            'New hand-rolled rename-into-place found:\n  '
            + '\n  '.join(f'{f}::{q}' for f, q in sorted(unapproved))
            + '\nUse shared.safe_io.atomic_write_text instead. If this site '
            'genuinely cannot (it moves an existing file rather than writing '
            'one, say), add it to _ALLOWED_RENAMERS with the reason.'
        )

        # HARD assert on stale entries, kept rather than downgraded to a
        # warning, for the same reason as the is_dir assert above: repo-level,
        # every tree is present, so a stale entry means the site really is gone
        # or really stopped renaming.  When someone finally migrates the
        # allowlisted orchestrator digest.write_digest_entry this fires — and
        # the fix is the one-line allowlist deletion the message spells out, in
        # THIS repo-level module, not a red suite in a package that merely
        # happened to host the guard.
        stale = set(_ALLOWED_RENAMERS) - actual
        assert not stale, (
            'Stale _ALLOWED_RENAMERS entries (site is gone or no longer '
            f'renames): {sorted(stale)}. Remove them so the allowlist keeps '
            'describing reality.'
        )

    def test_atomic_write_text_helpers_only_delegate(self):
        """The surviving ``_atomic_write_text`` names must stay one-liners.

        Task 3223 kept these module-level names (test_prompt_artifact.py
        monkeypatches one at five sites) but emptied their bodies down to a
        delegation.  Re-inlining a real implementation under the old name is
        the most likely way the duplication comes back, because the name would
        still look consolidated from every call site.

        NO COUNT IS WRITTEN HERE, deliberately — this docstring said "the four
        surviving names" from before the widening to six trees and was wrong by
        the time anyone read it (task 3388).  For orientation only, measured at
        that widening: six such function names existed across ``_SRC_TREES``,
        of which this test actually checks THREE (shared/src/shared/
        {memory_eval_limits,memory_eval_metrics,prompt_artifact}.py).  The other
        three are skipped by construction rather than overlooked —
        shared/src/shared/safe_io.py::atomic_write_text is the blessed
        implementation (skipped by path), and session_registry's and
        bake_off_storage_shape's copies are allowlisted exceptions (skipped by
        _ALLOWED_RENAMERS).  Re-run the test to learn today's numbers.

        THAT NARROWNESS IS THE REASON FOR THE FENCE BELOW.  Three of six is
        already thin, the widening to six trees added zero delegate coverage,
        and one allowlist entry (bake_off) deliberately removed one — so the
        trend is downward and every future entry costs another name.  The
        docstring above USED to be the only thing recording that; it is now
        asserted, against a named control, so the drift reports itself.
        """
        offenders = []
        checked = []
        for module in _scan_source_trees():
            if module.relpath == 'shared/src/shared/safe_io.py':
                continue  # the blessed implementation lives here
            for name, delegates in module.write_helpers:
                if (module.relpath, name) in _ALLOWED_RENAMERS:
                    # Same documented exceptions as the sibling guard above.
                    # session_registry in particular MUST keep an inlined body:
                    # it is stdlib-only so spawn-claude.sh can run it with no
                    # venv, and importing shared makes it unimportable there.
                    continue
                checked.append(f'{module.relpath}::{name}')
                if not delegates:
                    offenders.append(f'{module.relpath}::{name} does not delegate')

        # ANTI-VACUITY, in the same shape as `assert pointers` in the pointer
        # sweep below and `_CONTROL_MODULES` in the tree check.  This test was
        # the only one in the module without such a fence, and it is the one
        # that most needs it: its skip condition is membership in
        # _ALLOWED_RENAMERS, so every entry added there silences one more name,
        # silently and by design.  Without this it could iterate all 509
        # scanned files, collect nothing, and report green forever.
        assert checked, (
            'The delegate check examined no helpers. Real ones exist '
            f'({_DELEGATE_CONTROL}), so an empty result means the walk or the '
            'name filter regressed, not that the repo got clean.'
        )
        assert _DELEGATE_CONTROL in checked, (
            f'{_DELEGATE_CONTROL} is no longer being checked. Either it moved '
            '(repoint _DELEGATE_CONTROL at another delegating helper) or it '
            'was added to _ALLOWED_RENAMERS, which SILENCES this test for that '
            f'name — an allowlist entry buys off both guards at once. Checked: '
            f'{sorted(checked)}'
        )

        assert not offenders, (
            'These helpers stopped delegating to shared.safe_io.atomic_write_text '
            'and re-inlined their own implementation:\n  ' + '\n  '.join(offenders)
        )


# ---------------------------------------------------------------------------
# Module boundary and anti-vacuity fences (task 3388)
# ---------------------------------------------------------------------------
#
# Two tests ABOUT the guard above rather than about the source trees: one keeps
# the guard out of the package suite it was relocated from, the other keeps it
# from reporting green while scanning nothing.

#: The ONE file this module's boundary test pins.  Deliberately a single file
#: rather than a walk of ``shared/tests/**/*.py`` — see the scope-limit block
#: in test_atomic_write_guard_does_not_scan_sibling_package_trees.
_GUARDED_FILE = 'shared/tests/test_safe_io.py'

#: Scan roots that name a package ``shared`` does not own.  Matched as exact
#: STRING LITERALS via ast, never as substrings of prose: a comment or
#: docstring in the guarded file may legitimately discuss another package (the
#: relocated guard's own scope block does), and only a scan root — a literal
#: fed to a directory walk — can make shared's suite depend on that package.
_SIBLING_SCAN_ROOTS = (
    'orchestrator/src',
    'escalation/src',
    'fused-memory/src',
    'fused-memory/scripts',
    'scripts',
)

#: One real, long-lived module per declared tree.  ``_SRC_TREES`` yielding
#: *something* is not enough — a tree that rglobs only a stray ``__init__.py``
#: would satisfy a bare non-empty check while scanning nothing that matters.
_CONTROL_MODULES = {
    'shared/src': 'shared/src/shared/safe_io.py',
    'orchestrator/src': 'orchestrator/src/orchestrator/digest.py',
    'escalation/src': 'escalation/src/escalation/queue.py',
    'fused-memory/src': 'fused-memory/src/fused_memory/reconciliation/event_queue.py',
    'fused-memory/scripts': 'fused-memory/scripts/bake_off_storage_shape.py',
    'scripts': 'scripts/legibility/codebook.py',
}


def test_atomic_write_guard_does_not_scan_sibling_package_trees():
    """``shared/tests/test_safe_io.py`` declares no sibling package as a scan root.

    THE INVARIANT THIS DELIVERS, stated at its true width.  The atomic-write
    anti-regrowth guard must not be turnable red by a refactor in a package
    ``shared`` does not own.  Task 3388's worked example: the guard allowlists
    ``orchestrator/src/orchestrator/digest.py::write_digest_entry`` and flags it
    there as a prime migration candidate — so while the guard lived in
    ``shared/tests``, migrating or renaming that orchestrator function turned
    the SHARED suite red, at a site no orchestrator author would think to look.
    Relocating the guard to this repo-level directory (step 4) is what removes
    that coupling, and this test is what keeps it removed.

    SCOPE LIMIT — READ THIS BEFORE TRUSTING A GREEN RUN.  This test pins ONE
    file, not ``shared/tests`` as a whole.  It does NOT establish that shared's
    suite is standalone-runnable against a lone ``dark-factory-shared``
    checkout, and after task 3388 that suite is **not** standalone-runnable.
    The five OTHER cross-tree gates below live in ``shared/tests`` and are
    deliberately NOT covered here; each carries its own comment arguing for its
    cross-tree reach, so narrowing them is a design question this task did not
    settle rather than an oversight it missed:

      * ``silent_fallthrough_scan.py`` — ``_SCOPE_ROOTS`` (7 roots:
        orchestrator/src, fused-memory/src, dashboard/src, escalation/src,
        shared/src, sampler/src, scripts) and a hard ``RuntimeError`` when the
        ``shared/src``/``orchestrator/src`` sentinels are absent, paired with
        ``silent_fallthrough_allowlist.py`` (13 sibling-path literals, as
        ``(path, qualname, hash, reason)`` tuples).
      * ``config_dir_archival_allowlist.py`` — 10 sibling-path literals, as
        ``{'path': ..., 'qualname': ...}`` dicts.
      * ``test_auth_failed.py`` — ``_PRODUCTION_SRC_ROOTS = ('shared/src',
        'orchestrator/src')`` with a hard ``assert root.is_dir()``.
      * ``test_silent_fallthrough_gate.py`` — 3 sibling-path literals,
        hard-asserted (``assert candidate in files``).
      * ``test_capability_manifest.py`` — 21 sibling-path literals; most are
        synthetic fixtures, but it hard-asserts on REAL files in ``scripts/``
        (``committed_file_mode('scripts/check_method_param_wiring.py') ==
        '100755'``, and a superset assertion naming
        ``scripts/gc_agent_transcripts.py``).

    Counts measured first-hand at commit 6b68a87fd6 by an ast sweep of
    ``shared/tests/**/*.py`` for non-docstring string constants ending in
    ``.py`` and beginning with a sibling package directory — stated as a method
    plus a number so the claim stays falsifiable rather than becoming the kind
    of stale prose task 3388 exists to eliminate.  Re-run the sweep; do not
    re-trust this sentence.

    WHY THE NARROW FORM.  The plan specified a walk of ``shared/tests/**/*.py``
    asserting the whole directory names no sibling tree.  Measured, that test
    is red forever: the gates above are the falsifying evidence, they are
    outside this task's file scope, and each is load-bearing where it sits.
    Ticket ``tkt_0RT7TDAAH2TS88BR88TZ1E3QMP`` tracks the real question — where
    this family of repo-wide gates should live, and what it should do when a
    tree it names is absent.  Until that lands, this test is a fence around one
    gate and must not be read as evidence about the others.
    """
    target = _REPO_ROOT / _GUARDED_FILE
    assert target.is_file(), (
        f'{_GUARDED_FILE} not found under {_REPO_ROOT}. This test pins one named '
        f'file, so a move or rename must update _GUARDED_FILE rather than let the '
        f'check silently pass over a file that is no longer there.'
    )

    offenders = [
        f'{_GUARDED_FILE}:{node.lineno}: {node.value!r}'
        for node in ast.walk(ast.parse(target.read_text(encoding='utf-8')))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in _SIBLING_SCAN_ROOTS
    ]

    assert not offenders, (
        f'{_GUARDED_FILE} declares a scan root in a package shared does not own:\n  '
        + '\n  '.join(offenders)
        + '\nA guard that walks sibling package trees cannot live inside one '
        'package\'s suite: a refactor in that sibling (migrating the allowlisted '
        'orchestrator digest.write_digest_entry, say) turns SHARED red, at a site '
        'no orchestrator author would think to look. Move the cross-tree guard to '
        'a repo-level suite — tests/scripts/test_atomic_write_regrowth.py is where '
        'the atomic-write one lives — rather than widening this exception.'
    )


def test_every_declared_tree_contributes_scanned_files():
    """Every tree in ``_SRC_TREES`` actually contributes files to the scan.

    THE ANTI-VACUITY FENCE, and the load-bearing one of the two.  The guard's
    own ``assert root.is_dir()`` catches a DELETED tree.  It does not catch a
    tree that is present and rglobs nothing useful — which is exactly how a
    fence that was just WIDENED reports green while silently scanning less than
    it did before, and is the failure mode task 3388 exists to close.  So this
    asserts two things per tree: that it yields at least one scanned file, and
    that one named, long-lived control module from it is in the scanned set.
    """
    scanned = {relpath for relpath, _ in _iter_source_files()}

    # Annotated set[str] rather than the bare set() calls: pyright infers
    # set[Literal['shared/src', ...]] from the _SRC_TREES tuple and then
    # rejects `-` against set[str] (reportOperatorIssue).
    declared: set[str] = set(_SRC_TREES)
    controlled: set[str] = set(_CONTROL_MODULES)

    assert controlled == declared, (
        'Every declared tree needs a named control module, or the anti-vacuity '
        'check degrades to "the tree yielded something" for the unpaired one.\n'
        f'  declared but uncontrolled: {sorted(declared - controlled)}\n'
        f'  controlled but undeclared: {sorted(controlled - declared)}'
    )

    empty = [tree for tree in _SRC_TREES
             if not any(p.startswith(f'{tree}/') for p in scanned)]
    assert not empty, (
        f'Declared scan trees contributed NO files: {empty}. The tree exists (the '
        'is_dir assertion passed) but rglobbed nothing, so the guard is scanning '
        'less than its _SRC_TREES claims while still reporting green.'
    )

    missing = sorted(
        control for control in _CONTROL_MODULES.values() if control not in scanned
    )
    assert not missing, (
        f'Control modules absent from the scanned set: {missing}. Either the guard '
        'stopped reaching into that tree, or the module genuinely moved — in which '
        'case repoint _CONTROL_MODULES at another long-lived module in the same '
        'tree rather than deleting the entry, which would restore the vacuity this '
        'test exists to prevent.'
    )


# ---------------------------------------------------------------------------
# Pointer resolution (task 3388)
# ---------------------------------------------------------------------------

#: Prose of the form ``_ALLOWED_RENAMERS`` in ``<path>.py`` — i.e. the path a
#: comment CLAIMS is the symbol's home.  Keyed on that ADJACENCY, deliberately,
#: and NOT on the obvious looser form "the file mentions the symbol and also
#: names some .py path".  Measured: the loose form matches 30+ .py paths inside
#: this module alone, because every _ALLOWED_RENAMERS KEY is a .py path sitting
#: beside the symbol — so the check would fail on its own definition site.
#:
#: Backticks are optional and the gaps span newlines AND the ``#`` of a wrapped
#: comment: session_registry.py wraps the phrase across a line break inside a
#: docstring (pure whitespace), while test_safe_io.py's pointer wraps inside a
#: ``#`` comment block.  A plain ``\s+`` matches the first and silently misses
#: the second — measured, when this module's own pointer went unseen — so a
#: pointer that merely got rewrapped would stop being checked without anything
#: reporting it.  A pointer written without RST markup is still a pointer.
_POINTER_RE = re.compile(r'_ALLOWED_RENAMERS`{0,2}[\s#]+in[\s#]+`{0,2}([\w./\-]+\.py)')

#: Trees swept for pointers IN ADDITION to _SRC_TREES.  A stale pointer is a
#: documentation defect, and documentation about this guard lives mostly in
#: test files — which _SRC_TREES only reaches for the two flat operator trees.
#: ``scripts/tests`` is already inside the ``scripts`` scan root and
#: ``tests/scripts`` inside ``tests``; the walk dedupes by path.
#:
#: REQUIRED (hard ``assert root.is_dir()``): the tests dir of every package
#: whose src tree this guard FENCES, plus the repo-root ``tests`` that hosts
#: this module.  If one of those is absent, either a fenced package or the
#: guard's own home is gone, and that is a real defect worth a red.
_POINTER_EXTRA_TREES = (
    'shared/tests',
    'orchestrator/tests',
    'escalation/tests',
    'fused-memory/tests',
    'tests',
)

#: OPTIONAL (warn and skip): sibling packages this guard does NOT fence and
#: reads only for documentation-pointer hygiene.
#:
#: WHY THESE THREE ARE SOFT WHILE _SRC_TREES STAYS HARD — the hard-assert
#: argument at the top of this module does not transfer, and applying it here
#: anyway was a real defect rather than a stylistic quibble.  That argument is
#: "a missing tree means the fence silently scans less".  For a tree the fence
#: never scans, a missing dir means only that fewer doc comments got checked —
#: yet it produced the identical red, so deleting or renaming an optional
#: sibling package turned the ATOMIC-WRITE REGROWTH GUARD red for a reason
#: with nothing to do with atomic writes.  The repo does not treat every
#: package as guaranteed either: ``dark-factory-orchestrator.yaml``'s
#: test_command still wraps cockpit in ``( [ -d cockpit ] || exit 0; ... )``.
#:
#: Soft is not silent: ``_read_tree`` emits a ``warnings.warn`` naming the
#: tree, and ``assert pointers`` in the sweep below still fails if the whole
#: sweep ends up covering nothing.  MEASURED at this base, which is why the
#: split falls here: the repo's three real pointers live in
#: ``orchestrator/src``, ``shared/tests`` and ``orchestrator/tests`` — all in
#: the required set — and these three trees carry zero.  Promote one to
#: _POINTER_EXTRA_TREES if it ever starts carrying a pointer worth pinning.
_POINTER_OPTIONAL_TREES = (
    'cockpit/tests',
    'dashboard/tests',
    'sampler/tests',
)


def _iter_pointer_candidates():
    """Yield (repo-relative posix path, source) over _SRC_TREES + the tests trees.

    Deduped by relpath because the tree lists overlap by construction
    (``scripts/tests`` sits inside the ``scripts`` scan root, ``tests/scripts``
    inside ``tests``).  The reads are shared with the source-tree walk through
    ``_read_tree``'s per-tree cache, so sweeping a tree this module already
    scans costs nothing beyond the dedupe.
    """
    seen: set[str] = set()
    for relpath, source in (
        _walk_trees(_SRC_TREES + _POINTER_EXTRA_TREES)
        + _walk_trees(_POINTER_OPTIONAL_TREES, missing_ok=True)
    ):
        if relpath in seen:
            continue
        seen.add(relpath)
        yield relpath, source


def test_allowlist_pointers_resolve_to_the_guard_module():
    """Every comment claiming to name the allowlist's home names a file that has it.

    A REFERENCE-RESOLUTION check, not a prose pin.  It asserts that a named path
    exists AND actually contains the assignment — nothing about wording — so it
    does not go stale when someone rewrites the surrounding sentence, and it is
    not satisfied by a file that still exists but no longer owns the symbol.
    That second half is the load-bearing one: it is exactly the state task 3388
    step 4 created by moving the guard out of ``shared/tests/test_safe_io.py``,
    a file that very much still exists.

    The guard module itself is excluded by identity — the file that DEFINES the
    symbol is the definition, not a pointer to it.  Belt-and-braces rather than
    load-bearing: the adjacency matcher self-matches 0 times here (verified),
    which is the whole reason it is keyed on adjacency.
    """
    guard_module = Path(__file__).resolve().relative_to(_REPO_ROOT).as_posix()

    pointers: list[str] = []
    broken: list[str] = []
    for relpath, source in _iter_pointer_candidates():
        if relpath == guard_module:
            continue
        for match in _POINTER_RE.finditer(source):
            claimed = match.group(1)
            site = f'{relpath}:{source.count(chr(10), 0, match.start()) + 1}'
            pointers.append(site)
            target = _REPO_ROOT / claimed
            if not target.is_file():
                broken.append(f'{site} names {claimed}, which does not exist')
            elif '_ALLOWED_RENAMERS = ' not in target.read_text(encoding='utf-8'):
                broken.append(
                    f'{site} names {claimed}, which exists but no longer defines '
                    f'the allowlist'
                )

    # Non-vacuity: pointers to this guard DO exist in the repo, so finding none
    # means the matcher regressed, not that the docs got tidy.
    assert pointers, (
        'The pointer sweep matched nothing. Real pointers exist (orchestrator '
        'session_registry.py and test_session_registry.py both carry one), so '
        'an empty result means _POINTER_RE stopped matching or '
        '_iter_pointer_candidates stopped walking — not that the repo is clean.'
    )

    assert not broken, (
        'Stale pointers to the atomic-write allowlist:\n  '
        + '\n  '.join(broken)
        + f'\nThe allowlist lives in {guard_module}. Repoint the path and leave '
        'the surrounding paragraph alone; the rationale it records is unchanged.'
    )
