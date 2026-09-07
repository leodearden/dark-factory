"""Drift guard over the merge-state vocabularies' hand-maintained prose copies.

Task 4829 (PRD ``plans/merge-status-durable-non-landed-prd.md``, decision D3,
behaviour B9). ``shared/src/shared/merge_state.py`` is the single normative copy
of the two merge wire vocabularies — ``MergeState`` (the POLL vocabulary of
``merge_status`` / ``merge_cancel``'s ``state``) and ``MergeSubmitStatus`` (the
SUBMIT vocabulary of ``merge_request``'s ``status``). Several committed
artifacts nonetheless restate parts of those vocabularies inline, and none of
them auto-extends when a member lands.

MEASURED, not predicted. Every restatement below was read off this tree before
this guard existed:

  * ``skills/escalation-watcher/SKILL.md`` documented the POLL terminal set as
    ``done | conflict | blocked | already_merged`` — three defects in one line.
    ``already_merged`` is SUBMIT-only (``_map_terminal_state`` collapses it to
    ``done``, as that same file says forty lines further down), and ``abandoned``
    and ``superseded`` were both missing even though the same file documents
    ``merge_status`` returning ``state: 'abandoned'``. The file contradicted
    itself.
  * ``escalation/src/escalation/server.py``'s ``merge_cancel`` docstring
    enumerated five states for a function that demonstrably returns six
    (``superseded`` was missing).
  * ``skills/merge-queue/SKILL.md`` and ``skills/unblock-low-risk/SKILL.md``
    both named a submit status ``failed`` that appears NOWHERE in the server
    (the real value is ``error``), while omitting six real ones.

This module derives both vocabularies ONCE from the normative module and
cross-checks every restatement against it. It never stores its own snapshot of
the members: a hardcoded list here would be one more lock-step copy, stale on
the next member exactly like the prose sites were.

THE PINNED SITES live in ``PINNED_SITES``. Each carries its value list INLINE —
a runbook an agent executes at run time must SHOW the values, not point at a
Python file — wrapped in a ``merge-state-vocab`` marked span naming the
partition the list must equal. Same convention as ``CONTRIBUTING.md``'s
``lint-command-mirror`` block and ``skills/prd/references/gates.md``'s
``inv-trigger-shapes`` span.

WHAT THIS GUARD DELIBERATELY DOES NOT PIN.

  * ARM RULES — a loop or branch that deliberately handles a NARROWED subset of
    a vocabulary. ``skills/merge-queue/SKILL.md``'s resumed-poll loop (which
    drops ``superseded`` from its terminal set on purpose) and
    ``skills/unblock/SKILL.md``'s ``terminal_resumed`` tuple are rules about
    which states a particular arm may stop on, not copies of a vocabulary.
    Pinning them to a partition would demand they re-add the very member they
    exist to exclude. PRD delta owns them.
  * ``skills/orchestrate/SKILL.md``'s halted-merge row (see
    ``_UNPINNED_RULE_SITES``) — the same shape, one file wider: it enumerates
    the five statuses ``orchestrator/src/orchestrator/merge_queue.py``'s
    ``_map_advance_failure`` can return that halt the queue, a rule owned by the
    orchestrator rather than a copy of ``SUBMIT_TERMINAL``.
  * REASONING PROSE around each list — which state means what, how to react.
    PRD delta rewrites it, and pinning wording would go red on an editorial
    edit while a genuinely stale VALUE list stayed green.
  * The mapping between the two vocabularies. That belongs to
    ``escalation/src/escalation/server.py::_map_terminal_state``; a copy here
    would be exactly the second home the normative module exists to remove.

PLACEMENT IS LOAD-BEARING. ``scripts/tests/`` modules must import NO first-party
package — that is what lets ``uv run --project shared pytest scripts/tests/``
(``scripts/orchestrator.yaml``'s ``test_command``) satisfy them on a freshly
synced verify worktree. This module is stdlib-only (``importlib.util``, ``os``,
``re``, ``subprocess``, ``pathlib``) plus ``pytest``, and it loads
``shared/src/shared/merge_state.py`` BY ABSOLUTE PATH rather than importing
``shared.merge_state``. That second point is load-bearing twice over: an agent
Bash session typically inherits the MAIN checkout's ``VIRTUAL_ENV`` (see
``CLAUDE.md``, "Locating installed code"), so a plain import could pin THIS
worktree's SKILL.md files against MAIN's vocabulary.

EXTRACTOR CONTRACT. Every extractor below raises a loud ``AssertionError``
naming its ``source`` rather than returning an empty result. An extractor that
silently yields nothing turns every downstream drift assertion green while
pinning nothing at all — strictly worse than no guard, because the check still
reports success. Extractors are unit-tested against HAND-WRITTEN fixture text,
never the live artifacts, so those tests stay stable under any future edit; the
live assertions re-read every committed artifact fresh.
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_VOCABULARY_MODULE = REPO_ROOT / "shared" / "src" / "shared" / "merge_state.py"

# Every partition name a `partition=` marker may cite. Stated here rather than
# derived from the vocabulary module's `__all__` because it is a CONTRACT, not an
# observation: a prose span may only pin a partition this guard knows how to
# compare, so adding a partition to the module is a deliberate act that must also
# teach the guard about it. The enum MEMBERS are never listed — those are derived
# from the module on every run, which is the whole point (a hardcoded roster here
# would be one more lock-step copy, stale on the next member).
_REQUIRED_PARTITIONS = (
    "LIVE_STATES",
    "TERMINAL_STATES",
    "OUTCOME_STATES",
    "EPISTEMIC_STATES",
    "POLL_STOP_STATES",
    "CANCEL_STATES",
    "SUBMIT_TERMINAL",
    "SUBMIT_NON_TERMINAL",
)

# The two vocabularies hold 22 distinct wire values today (13 poll + 14 submit,
# five spellings shared). A union this small would mean the module stopped
# parsing, not that the vocabularies shrank — merge states are only ever added.
# The floor is a NON-VACUITY guard on every comparison downstream, kept well
# below the live count so a deliberate retirement does not go red spuriously.
_MINIMUM_VOCABULARY_SIZE = 10


def load_vocabulary(*, path: Path = _VOCABULARY_MODULE) -> dict[str, frozenset[str]]:
    """The normative partitions, loaded from *path* BY FILE PATH, as plain strings.

    Never ``import shared.merge_state``. Two reasons, both load-bearing:

    * ``scripts/tests/`` modules must import no first-party package — that is what
      lets ``uv run --project shared pytest scripts/tests/``
      (``scripts/orchestrator.yaml``'s ``test_command``) satisfy them on a freshly
      synced verify worktree, where ``escalation``/``orchestrator`` are not
      installed.
    * An agent Bash session typically inherits the MAIN checkout's ``VIRTUAL_ENV``
      (``CLAUDE.md``, "Locating installed code"), so a plain import in a task
      worktree resolves to MAIN's tree. This guard would then pin THIS worktree's
      SKILL.md files against ANOTHER tree's vocabulary — green on drift, red on
      none, depending on ambient state.

    Values are coerced to ``str``. ``MergeState`` is a ``StrEnum``, so its members
    already compare equal to their wire spelling, but the spans carry TEXT and a
    caller building sets by hand must not have to know that.

    Every failure is a loud ``AssertionError`` naming *path*, never an empty dict:
    an empty vocabulary compares nothing against nothing, which PASSES — the guard
    would report its strongest verdict having read no vocabulary at all.
    """
    assert path.is_file(), (
        f"the merge-state vocabulary module {path} does not exist (task 4829). This "
        f"guard derives both wire vocabularies from it on every run, so without it "
        f"every span comparison below would compare an empty set against an empty "
        f"set and PASS. Either the module moved (update `_VOCABULARY_MODULE`) or "
        f"the worktree is incomplete."
    )

    spec = importlib.util.spec_from_file_location("_merge_state_vocabulary_by_path", path)
    assert spec is not None and spec.loader is not None, (
        f"{path} could not be turned into an importable module spec (task 4829)."
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 — re-raised as a named guard failure
        raise AssertionError(
            f"the merge-state vocabulary module {path} could not be executed (task "
            f"4829): {type(exc).__name__}: {exc}. It is loaded by absolute path under "
            f"a synthetic module name, so the bare traceback names neither this guard "
            f"nor the file — fix the module, or point `_VOCABULARY_MODULE` at it."
        ) from exc

    missing = [name for name in _REQUIRED_PARTITIONS if not hasattr(module, name)]
    assert not missing, (
        f"{path} defines no {missing!r} (task 4829). Every `merge-state-vocab` span "
        f"in the repo cites a partition BY NAME, so a renamed or deleted partition "
        f"leaves those spans pinned to nothing. Either restore the name or rename it "
        f"in `_REQUIRED_PARTITIONS` and in every span that cites it."
    )

    partitions = {
        name: frozenset(str(value) for value in getattr(module, name))
        for name in _REQUIRED_PARTITIONS
    }
    empty = sorted(name for name, values in partitions.items() if not values)
    assert not empty, (
        f"{path} defines {empty!r} as EMPTY (task 4829) — a span pinned to an empty "
        f"partition can only match an empty list, so the check would either pass "
        f"vacuously or send a reader to delete a correct list."
    )
    return partitions


def vocabulary_values(partitions: dict[str, frozenset[str]]) -> frozenset[str]:
    """Every wire value across BOTH vocabularies — the token filter for spans.

    The union spans ``MergeState`` AND ``MergeSubmitStatus`` deliberately. Filtering
    a span by its OWN partition instead would drop an intruder from the sibling
    vocabulary before the comparison ever ran: a POLL span claiming ``merge_status``
    returns ``wip_halted`` would read green, which is one of the three defects this
    task exists to correct.
    """
    values = frozenset().union(*partitions.values()) if partitions else frozenset()
    assert len(values) >= _MINIMUM_VOCABULARY_SIZE, (
        f"the loaded vocabulary holds only {len(values)} distinct values "
        f"{sorted(values)!r}, below the non-vacuity floor of "
        f"{_MINIMUM_VOCABULARY_SIZE} (task 4829). Both wire vocabularies are "
        f"expected; a union this small means the module was parsed but not read."
    )
    return values


# `failed` is NOT a member of either vocabulary — it is a value
# `skills/merge-queue/SKILL.md` and `skills/unblock-low-risk/SKILL.md` invented
# (the real value is `error`). It is listed here anyway so that correcting those
# files cannot accidentally start counting it.
_AMBIGUOUS_TOKENS = frozenset(
    {"done", "blocked", "queued", "conflict", "unknown", "error", "gate", "failed"}
)


def discriminating_tokens(partitions: dict[str, frozenset[str]]) -> frozenset[str]:
    """Vocabulary values minus the ones that double as task statuses or prose.

    `done`, `blocked`, `queued`, `conflict` and `unknown` are all Taskmaster task
    statuses; `error` and `gate` are ordinary English in this repo. Counting them
    would flag most of `skills/` as an enumeration site, which trains readers to
    register files to silence the guard — the failure mode that makes a guard stop
    meaning anything.
    """
    tokens = vocabulary_values(partitions) - _AMBIGUOUS_TOKENS
    assert tokens, (
        "the discriminating token set is empty (task 4829) — with nothing to count, "
        "no file can clear the threshold and the registry scan passes vacuously."
    )
    return tokens


_BEGIN_MARKER = "merge-state-vocab:begin"
_END_MARKER = "merge-state-vocab:end"

# The partition a span pins itself to. UPPER_SNAKE only: partition names are
# module-level constants, and a lowercase `partition=terminal_states` is a typo
# that must fail as "names no partition" rather than silently matching nothing.
_PARTITION_RE = re.compile(r"partition=([A-Z][A-Z0-9_]*)")

# Any lowercase identifier-shaped token. Word boundaries, not a bare `[a-z]+`:
# without the leading `\b` the pattern matches "erminal" inside "Terminal", which
# is harmless only because membership filters it — better not to generate it.
# This one regex serves both literal styles the pinned sites use (markdown
# backticks, Python string literals) and both container styles (`{a, b}` set
# notation, `a | b` pipes), because every one of them reduces to bare tokens once
# the punctuation is ignored.
_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9_]*\b")


def extract_spans(text: str, *, source: str) -> list[tuple[str, str]]:
    """Every ``merge-state-vocab`` span in *text*, as ``(partition_name, body)``.

    TWO CARRIERS, one scanner. In markdown the markers ride in HTML comments
    (``<!-- merge-state-vocab:begin partition=X ... -->`` … ``<!-- …:end -->``); in
    a Python docstring they are bare lines, because an HTML comment there is
    nonsense that renders literally in `--help` output and in every agent's
    context. Four pinned sites use the first carrier and one uses the second, so a
    scanner that handled only one would leave the other silently unpinned.

    The begin comment's own body is EXCLUDED from the returned span: it carries the
    source-of-truth pointer, this module's path and the partition name, any of
    which would join the extracted value set if swallowed.

    Every malformation raises loudly, naming *source*. The highest-value case is
    ZERO spans — that is what an edit which deletes the markers but keeps the list
    produces, and returning `[]` for it would turn the whole site's drift check
    into a vacuous pass while the list rots.
    """
    lines = text.splitlines()
    spans: list[tuple[str, str]] = []
    open_span: tuple[str, int, str, int] | None = None

    for index, line in enumerate(lines):
        line_number = index + 1
        if _BEGIN_MARKER in line:
            match = _PARTITION_RE.search(line)
            assert match is not None, (
                f"{source}: the `{_BEGIN_MARKER}` marker at line {line_number} names no "
                f"`partition=<NAME>` (task 4829). A span that does not say which "
                f"partition it must equal cannot be checked against anything; add "
                f"`partition=` naming one of {list(_REQUIRED_PARTITIONS)}."
            )
            partition_name = match.group(1)
            assert open_span is None, (
                f"{source}: span `{partition_name}` opens at line {line_number} while "
                f"span `{open_span[0]}` (opened at line {open_span[3]}) is still open "
                f"(task 4829) — spans do not nest. The inner list would be read as part "
                f"of the outer one, so both comparisons would be wrong. Close the outer "
                f"span with `{_END_MARKER}` before opening another."
            )

            marker_column = line.index(_BEGIN_MARKER)
            if "<!--" in line[:marker_column]:
                close_index = index
                while close_index < len(lines) and "-->" not in lines[close_index]:
                    close_index += 1
                assert close_index < len(lines), (
                    f"{source}: the `<!-- {_BEGIN_MARKER} partition={partition_name}` "
                    f"comment opened at line {line_number} is never closed with `-->` "
                    f"(task 4829) — the whole rest of the file would be read as the "
                    f"begin comment's body."
                )
                remainder = lines[close_index].split("-->", 1)[1]
                body_start = close_index + 1
            else:
                remainder = ""
                body_start = index + 1

            open_span = (partition_name, body_start, remainder, line_number)
        elif _END_MARKER in line:
            assert open_span is not None, (
                f"{source}: a `{_END_MARKER}` marker at line {line_number} closes no "
                f"open span (task 4829) — either its `{_BEGIN_MARKER}` was deleted, "
                f"leaving the list below it unpinned, or the two markers are inverted."
            )
            partition_name, body_start, remainder, _ = open_span
            body = "\n".join([remainder, *lines[body_start:index]])
            spans.append((partition_name, body))
            open_span = None

    if open_span is not None:
        raise AssertionError(
            f"{source}: span `{open_span[0]}` opened at line {open_span[3]} is never "
            f"closed by a `{_END_MARKER}` marker (task 4829). An unterminated span has "
            f"no boundary, so every vocabulary token in the rest of the file would join "
            f"its value set."
        )

    assert spans, (
        f"{source}: no `{_BEGIN_MARKER} partition=<NAME>` span found at all (task "
        f"4829). This file is in `PINNED_SITES` because it restates a merge-state "
        f"vocabulary inline; with the markers gone, its list is free to drift and "
        f"this guard would report success having checked nothing. Restore the span "
        f"around the list, or unregister the file if the list is gone."
    )

    unknown = sorted({name for name, _ in spans if name not in _REQUIRED_PARTITIONS})
    assert not unknown, (
        f"{source}: span(s) pinned to {unknown!r}, which name no partition of "
        f"{_repo_relative(_VOCABULARY_MODULE)} (task 4829). Known partitions are "
        f"{list(_REQUIRED_PARTITIONS)}. A misspelled partition pins the list to "
        f"nothing at all."
    )
    return spans


def extract_values(body: str, vocabulary_values: frozenset[str], *, source: str = "a span") -> frozenset[str]:
    """The vocabulary members named inside a span *body*.

    Handles every literal style the pinned sites use — markdown backticks
    (`` `done` ``), Python string literals (``"done"`` / ``'done'``), bare set
    notation (``{queued, verifying}``) and pipe notation (``done | conflict``) —
    because all four reduce to bare lowercase tokens once punctuation is ignored.
    The sites are deliberately not forced into one style: a SKILL.md renders
    backticks and a Python docstring does not.

    Membership filtering is what lets a span carry ordinary English (``— stop
    polling here``) without every noun becoming a phantom member. *vocabulary_values*
    must be the union of BOTH vocabularies — see ``vocabulary_values()``.

    Raises rather than returning an empty set: an empty set does compare unequal to
    every partition, so the drift would be caught — but reported as "the list lost
    all five members", sending the reader to fix a list that is fine when the real
    defect is a misplaced `:end` marker.
    """
    found = frozenset(token for token in _TOKEN_RE.findall(body) if token in vocabulary_values)
    assert found, (
        f"{source}: a `merge-state-vocab` span body carries no vocabulary member at "
        f"all (task 4829). Body was {body!r}. Almost always a misplaced marker — the "
        f"`:end` above the list it was meant to close, or the `:begin` below it — "
        f"rather than a list that genuinely lost every value."
    )
    return found


def assert_span_matches(
    values: frozenset[str],
    partition: frozenset[str],
    *,
    source: str,
    partition_name: str,
) -> None:
    """SET EQUALITY between a span's values and the partition it claims to mirror.

    Missing and extra are reported SEPARATELY, each as a repr, because the two
    have different fixes: a MISSING value means the prose list is stale (a member
    landed and this copy did not follow — the drift B9 binds), while an EXTRA one
    is usually a value pasted in from the sibling vocabulary (``already_merged`` in
    a POLL list) or a value that never existed (``failed``).

    The message quotes the offending values rather than a phrase about drift: a
    reader cannot act on "the list has drifted" without being told which value.
    """
    values = frozenset(str(value) for value in values)
    partition = frozenset(str(value) for value in partition)

    assert values, (
        f"{source}: the span pinned to `{partition_name}` produced an EMPTY value set "
        f"(task 4829) — nothing was compared, so this check would otherwise pass or "
        f"fail for reasons unrelated to the list's contents."
    )

    missing = sorted(partition - values)
    extra = sorted(values - partition)
    assert not missing and not extra, (
        f"{source}: the `merge-state-vocab` span pinned to `{partition_name}` has "
        f"drifted from {_repo_relative(_VOCABULARY_MODULE)}::{partition_name} (task "
        f"4829). missing={missing!r} extra={extra!r}; span has {sorted(values)!r}, "
        f"partition is {sorted(partition)!r}. A MISSING value means this list did not "
        f"follow a member that landed in the enum; an EXTRA one is usually a value "
        f"from the sibling vocabulary (`MergeSubmitStatus` values are not "
        f"`merge_status` states) or one that never existed. Fix the list — the "
        f"partition is the source of truth."
    )


# Every artifact that restates a merge-state vocabulary inline, and WHAT is pinned
# there. `test_registry_is_complete` checks this registry against a scan, so it
# cannot quietly fall behind the repo the way the prose sites did.
PINNED_SITES = {
    "escalation/src/escalation/server.py": (
        "the `merge_cancel` docstring's CANCEL_STATES span (bare markers — an HTML "
        "comment in a docstring would render literally)"
    ),
    "skills/merge-queue/SKILL.md": (
        "live set, poll terminal set, submit terminal + non-terminal sets"
    ),
    "skills/unblock/SKILL.md": (
        "the two poll tuples — branch arm (TERMINAL_STATES) and scoped arms "
        "(POLL_STOP_STATES)"
    ),
    "skills/unblock-low-risk/SKILL.md": (
        "live set (twice), submit terminal + non-terminal sets"
    ),
    "skills/escalation-watcher/SKILL.md": (
        "poll terminal set, submit terminal + non-terminal sets"
    ),
}

# Files inside the scan's domain that enumerate a deliberately NARROWED subset of a
# vocabulary — a RULE about which values a particular arm handles, not a copy of a
# vocabulary. Registering one would demand it re-add the very members it exists to
# exclude, so each is excluded BY NAME with its reason recorded here rather than
# left to fall under the threshold by luck.
#
# `test_declared_rule_sites_are_tracked_and_span_free` keeps this from becoming a
# silent escape hatch: an excluded file must still exist, and must carry NO
# `merge-state-vocab` span (if someone wraps a list there, it must be registered
# above instead).
_UNPINNED_RULE_SITES = {
    "skills/orchestrate/SKILL.md": (
        "the MERGE (halted) troubleshooting row enumerates the five statuses "
        "`orchestrator/src/orchestrator/merge_queue.py`'s `_map_advance_failure` can "
        "return that halt the queue (wip_halted, done_wip_recovery, "
        "wip_recovery_no_advance, unmerged_state, stash_failed) — an orchestrator-owned "
        "halt rule, not a copy of SUBMIT_TERMINAL. See esc-4829-3."
    ),
}

# Four DISTINCT discriminating tokens is an enumeration, not a discussion.
#
# MEASURED on this tree over `git ls-files -- 'skills/*.md'` (42 files):
# merge-queue 8, escalation-watcher 7, unblock-low-risk 6, orchestrate 6 (the rule
# site above), unblock 5, review-briefing 2, everything else <= 2. So the threshold
# sits one token below the lowest registered site and two above the highest
# unregistered one. (The plan for task 4829 predicted 6/5/5/5 and "<= 2 everywhere
# else"; the re-measured numbers above are what this tree actually shows, and the
# sixth site is why `_UNPINNED_RULE_SITES` exists — filed as esc-4829-3.)
_ENUMERATION_THRESHOLD = 4

# BOUNDARY, deliberate and worth stating so a reader does not read it as an
# oversight: this scan covers `skills/**` only — the domain of the five registered
# sites. `ARCHITECTURE.md` (~lines 700-702) and `OPERATIONS.md` (~lines 517-523)
# each carry a further enumeration interleaving BOTH vocabularies, and
# `skills/merge-queue/SKILL.md` documents a `needs_rebase` outcome that
# `merge_request` never returns (it is a `suffix_graph.py` internal). Those are out
# of task 4829's declared scope and are filed as follow-up work.
_SKILL_MARKDOWN_PATHSPEC = "skills/*.md"


def _repo_relative(path: Path) -> str:
    """*path* as a repo-relative label when it is in the repo, else absolute."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


_scan_label = _repo_relative


def _scrubbed_git_env() -> dict[str, str]:
    """``os.environ`` with every ``GIT_*`` override removed.

    ``GIT_DIR``, ``GIT_WORK_TREE`` and ``GIT_INDEX_FILE`` are inherited by default
    and any one of them silently retargets a git invocation at a different
    repository than its ``cwd`` implies — which would make this scan's verdict a
    property of ambient state rather than of repo content.
    """
    return {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}


def tracked_skill_markdown(*, root: Path = REPO_ROOT) -> list[Path]:
    """Every TRACKED markdown file under ``skills/`` in *root*.

    Sourced from ``git ls-files``, never a filesystem walk: an untracked file — a
    scratch draft, a gitignored digest — must not be able to flip the registry
    scan's verdict on nothing but local working-tree state.

    Raises rather than falling back when the oracle cannot run. The repository half
    is guaranteed (this module lives in one), but the ``git`` EXECUTABLE on ``PATH``
    is not: a verify subprocess's ``PATH`` is rewritten by
    ``orchestrator/src/orchestrator/verify.py::_target_subprocess_env``. A silent
    filesystem fallback would restore exactly the hazard this sourcing removes, in
    the situation nobody is watching.

    COST, stated rather than hidden: a file written but not yet ``git add``ed is
    invisible here. An author who writes a new runbook restating the vocabulary and
    runs this guard before staging it gets a GREEN verdict. Acceptable — every
    dispatched agent commits before verify runs, and pre-commit sees staged content
    — but stage a new file before trusting a green run.
    """
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z", "--", _SKILL_MARKDOWN_PATHSPEC],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env=_scrubbed_git_env(),
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        returncode = getattr(exc, "returncode", None)
        stderr = getattr(exc, "stderr", None)
        detail = stderr.strip() if stderr else str(exc)
        outcome = f"exited {returncode}" if returncode is not None else "could not be run"
        raise RuntimeError(
            f"the tracked-file scan of {root} failed (task 4829): `git ls-files -z -- "
            f"{_SKILL_MARKDOWN_PATHSPEC}` {outcome} — {detail}. This scan sources its "
            f"file list from TRACKED files only, by design, so {root} must be a git "
            f"repository (or a subdirectory of one) with `git` on PATH — there is no "
            f"filesystem-walk fallback."
        ) from exc

    found = set()
    for relative in completed.stdout.split("\0"):
        if not relative or not relative.endswith(".md"):
            continue
        path = root / relative
        # `git ls-files` reads the INDEX, not the worktree: a path can be listed
        # while missing on disk (routine mid-rebase), and it lists an UNMERGED path
        # once per merge stage. Dropping the absent ones and deduping here keeps a
        # conflicted tree from triplicating every finding downstream.
        if path.is_file():
            found.add(path)
    return sorted(found)


def find_unregistered_sites(
    *,
    files: list[Path] | None = None,
    registry: dict[str, str] | None = None,
    rule_sites: dict[str, str] | None = None,
    discriminating: frozenset[str] | None = None,
    threshold: int = _ENUMERATION_THRESHOLD,
) -> list[str]:
    """Files restating *threshold*+ DISTINCT discriminating values that nobody pinned.

    A drift guard whose site list is hand-maintained reproduces the exact defect it
    exists to prevent: it reads green while a newly written runbook restates the
    vocabulary and rots on the next member. So ``PINNED_SITES`` is not trusted — it
    is checked against a scan of every tracked markdown file under ``skills/``.

    DISTINCT tokens, not occurrences: a file discussing one state six times is
    discussing it, not enumerating the vocabulary.

    Every argument is a seam for this module's own unit tests, which scan fixture
    files under ``tmp_path``; the live call passes none of them. Loud on an empty
    scan rather than returning ``[]``, since ``[]`` is this scan's strongest
    possible verdict ("every site is registered") and an over-broad pathspec would
    report it having read nothing.
    """
    files = tracked_skill_markdown() if files is None else files
    registry = PINNED_SITES if registry is None else registry
    rule_sites = _UNPINNED_RULE_SITES if rule_sites is None else rule_sites
    if discriminating is None:
        discriminating = discriminating_tokens(load_vocabulary())

    assert files, (
        "the registry scan received no files to check (task 4829) — an empty scan "
        "returns an empty result, which is indistinguishable from `every enumeration "
        "site is registered`. Check the pathspec."
    )
    assert discriminating, (
        "the registry scan received an empty discriminating token set (task 4829) — "
        "with nothing to count, no file can clear the threshold and the scan passes "
        "vacuously."
    )

    unregistered: list[str] = []
    for path in sorted(files):
        label = _scan_label(path)
        if label in registry or label in rule_sites:
            continue
        text = path.read_text(encoding="utf-8")
        found = {token for token in _TOKEN_RE.findall(text) if token in discriminating}
        if len(found) >= threshold:
            unregistered.append(label)
    return unregistered


# ---------------------------------------------------------------------------
# Hand-written fixture text. NEVER the live artifacts: an extractor unit test
# that read a real SKILL.md would go red on any future edit to that file, which
# trains readers to edit the guard instead of the drift.
# ---------------------------------------------------------------------------

_FIXTURE_SOURCE = "fixture-site.md"

# The markdown carrier, with the marker form step-10 writes into the SKILL.md
# files: a multi-line begin comment carrying the source-of-truth pointer, and a
# bare end comment. Two decoys are deliberate — the begin comment names the
# partition (a token an over-broad span would swallow), and the trailing prose
# names `already_merged`, a real vocabulary member that sits OUTSIDE the span.
_MARKDOWN_SPAN = """\
# fixture site

<!-- merge-state-vocab:begin partition=TERMINAL_STATES
     Mirrors shared/src/shared/merge_state.py::TERMINAL_STATES. Pinned by
     scripts/tests/test_merge_state_vocabulary_consistency.py — extend the enum
     and this line goes red until it matches. -->
**Terminal states** (`done`, `conflict`, `blocked`, `abandoned`, `superseded`) — stop polling.
<!-- merge-state-vocab:end -->

Elsewhere in the file, outside every span, the prose discusses `already_merged`.
"""

# The Python-docstring carrier: bare marker lines inside a docstring, where an
# HTML comment would be nonsense. This is the shape
# `escalation/src/escalation/server.py::merge_cancel` carries.
_DOCSTRING_SPAN = '''\
def merge_cancel() -> dict:
    """Cancel a pending merge request.

    Returns:
        state (str) — Coarse terminal state; always a member of
                      ``shared.merge_state.MergeState``.
                      merge-state-vocab:begin partition=CANCEL_STATES
                      'done' | 'conflict' | 'blocked' | 'abandoned' | 'superseded' | 'unknown'
                      merge-state-vocab:end
                      Source of truth: shared/src/shared/merge_state.py::CANCEL_STATES.
    """
'''

# The third literal style, and the single-line begin-comment form: bare set
# notation inside a fenced block, markers outside the fence.
_SET_NOTATION_SPAN = """\
<!-- merge-state-vocab:begin partition=LIVE_STATES -->
```python
LIVE = {"queued", "verifying", "gate", "finalizing"}
```
<!-- merge-state-vocab:end -->
"""

_TWO_SPANS = """\
<!-- merge-state-vocab:begin partition=LIVE_STATES -->
Live: `queued`, `verifying`, `gate`, `finalizing`
<!-- merge-state-vocab:end -->

<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
<!-- merge-state-vocab:end -->
"""

# ── seeded violations ──────────────────────────────────────────────────────
# Each is a shape a prose list actually developed on this tree, or the shape a
# future edit would develop. B9 binds the REJECTION, so each must be OBSERVED
# to fail rather than asserted by reading the code.

# Today's `skills/escalation-watcher/SKILL.md` line, defect for defect.
_SPAN_MISSING_MEMBER = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`
<!-- merge-state-vocab:end -->
"""

_SPAN_EXTRA_MEMBER = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`, `already_merged`
<!-- merge-state-vocab:end -->
"""

# A value from the OTHER vocabulary. `wip_halted` is SUBMIT-only, so a POLL span
# naming it is claiming `merge_status` can return it — it cannot.
_SPAN_WRONG_VOCABULARY = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`, `wip_halted`
<!-- merge-state-vocab:end -->
"""

_SPAN_ABSENT = """\
# fixture site

A registered file whose span was deleted in an edit. It still discusses
`done`, `conflict` and `superseded` in prose, so a content heuristic would
think it was fine.
"""

_SPAN_UNTERMINATED = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
"""

_SPAN_NESTED = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATES -->
Terminal: `done`, `conflict`, `blocked`
<!-- merge-state-vocab:begin partition=LIVE_STATES -->
Live: `queued`, `verifying`, `gate`, `finalizing`
<!-- merge-state-vocab:end -->
"""

_SPAN_UNOPENED_END = """\
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
<!-- merge-state-vocab:end -->
"""

_SPAN_UNKNOWN_PARTITION = """\
<!-- merge-state-vocab:begin partition=TERMINAL_STATUSES -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
<!-- merge-state-vocab:end -->
"""

_SPAN_NO_PARTITION = """\
<!-- merge-state-vocab:begin -->
Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`
<!-- merge-state-vocab:end -->
"""


# ---------------------------------------------------------------------------
# load_vocabulary
# ---------------------------------------------------------------------------


def test_load_vocabulary_exposes_every_partition_the_spans_can_name() -> None:
    """The loader hands back every partition a `partition=` marker may cite.

    Named explicitly rather than derived from the module's `__all__`: the point
    of the guard is that a prose span can only pin a partition this module
    knows how to compare, so the set of citable names is part of the contract,
    not an accident of what the vocabulary module happens to export.
    """
    partitions = load_vocabulary()

    assert set(partitions) == {
        "LIVE_STATES",
        "TERMINAL_STATES",
        "OUTCOME_STATES",
        "EPISTEMIC_STATES",
        "POLL_STOP_STATES",
        "CANCEL_STATES",
        "SUBMIT_TERMINAL",
        "SUBMIT_NON_TERMINAL",
    }
    for name, values in partitions.items():
        assert values, f"{name} loaded empty — every downstream comparison would be vacuous"
        assert all(isinstance(value, str) for value in values), (
            f"{name} must load as plain `str` values: spans carry text, and a "
            f"comparison of text against enum members would never be equal"
        )


def test_load_vocabulary_values_covers_both_vocabularies() -> None:
    """The union of the partitions is the token filter `extract_values` uses.

    It must span BOTH vocabularies. If it covered only `MergeState`, a POLL span
    that named a SUBMIT-only value would have that value filtered out before the
    comparison and the drift would read GREEN — the seeded wrong-vocabulary case
    below is exactly that failure.
    """
    values = vocabulary_values(load_vocabulary())

    assert {"done", "abandoned", "superseded", "no_record"} <= values, "POLL members missing"
    assert {"already_merged", "wip_halted", "attached", "error"} <= values, "SUBMIT members missing"
    assert "failed" not in values, (
        "`failed` is a value the runbooks invented; it appears nowhere in "
        "escalation/src/escalation/server.py (the real value is `error`)"
    )


def test_load_vocabulary_raises_when_the_vocabulary_module_is_missing(tmp_path: Path) -> None:
    """A missing normative module RAISES rather than loading an empty vocabulary.

    An empty vocabulary makes every span comparison compare nothing against
    nothing, which passes — the guard would report its strongest verdict while
    having read no vocabulary at all.
    """
    missing = tmp_path / "merge_state.py"

    with pytest.raises(AssertionError) as excinfo:
        load_vocabulary(path=missing)

    assert str(missing) in str(excinfo.value)


def test_load_vocabulary_raises_when_a_partition_is_absent(tmp_path: Path) -> None:
    """A renamed or deleted partition names ITSELF in the failure.

    This is the drift shape a refactor of the vocabulary module produces: the
    spans still cite `TERMINAL_STATES` while the module no longer defines it.
    """
    stub = tmp_path / "merge_state.py"
    stub.write_text(
        "LIVE_STATES = frozenset({'queued'})\n"
        "OUTCOME_STATES = frozenset({'queued'})\n"
        "EPISTEMIC_STATES = frozenset({'unknown'})\n"
        "POLL_STOP_STATES = frozenset({'done'})\n"
        "CANCEL_STATES = frozenset({'done'})\n"
        "SUBMIT_TERMINAL = frozenset({'done'})\n"
        "SUBMIT_NON_TERMINAL = frozenset({'queued'})\n",
        encoding="utf-8",
    )

    with pytest.raises(AssertionError) as excinfo:
        load_vocabulary(path=stub)

    assert "TERMINAL_STATES" in str(excinfo.value)


def test_load_vocabulary_raises_when_the_vocabulary_module_will_not_execute(
    tmp_path: Path,
) -> None:
    """A broken normative module fails as a NAMED loader failure.

    Without the wrapper the reader gets a bare `SyntaxError` from a synthetic
    module name, with nothing tying it to this guard or to the file it tried to
    load.
    """
    broken = tmp_path / "merge_state.py"
    broken.write_text("LIVE_STATES = frozenset({\n", encoding="utf-8")

    with pytest.raises(AssertionError) as excinfo:
        load_vocabulary(path=broken)

    assert str(broken) in str(excinfo.value)


def test_load_vocabulary_does_not_import_the_shared_package(tmp_path: Path) -> None:
    """Loading is BY PATH: the stub above wins over any installed `shared`.

    The property that matters is that the guard reads the vocabulary from the
    tree it was pointed at. A `import shared.merge_state` would resolve through
    whatever editable install is on `sys.path` — in a task worktree, typically
    the MAIN checkout — and would silently pin this worktree's SKILL.md files
    against another tree's vocabulary.
    """
    stub = tmp_path / "merge_state.py"
    stub.write_text(
        "\n".join(
            f"{name} = frozenset({{'sentinel_value'}})"
            for name in (
                "LIVE_STATES",
                "TERMINAL_STATES",
                "OUTCOME_STATES",
                "EPISTEMIC_STATES",
                "POLL_STOP_STATES",
                "CANCEL_STATES",
                "SUBMIT_TERMINAL",
                "SUBMIT_NON_TERMINAL",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert load_vocabulary(path=stub)["TERMINAL_STATES"] == frozenset({"sentinel_value"})


# ---------------------------------------------------------------------------
# extract_spans
# ---------------------------------------------------------------------------


def test_extract_spans_reads_the_markdown_comment_carrier() -> None:
    """The HTML-comment carrier, with the begin comment's own body EXCLUDED.

    The begin comment names this module's path and the partition; a span that
    swallowed it would pin the pointer rather than the list. The trailing prose
    naming `already_merged` is outside the span and must not be read either —
    span placement is what keeps the extracted set honest.
    """
    spans = extract_spans(_MARKDOWN_SPAN, source=_FIXTURE_SOURCE)

    assert [name for name, _ in spans] == ["TERMINAL_STATES"]
    body = spans[0][1]
    assert "superseded" in body
    assert "Mirrors shared/src/shared/merge_state.py" not in body, "begin comment leaked in"
    assert "already_merged" not in body, "out-of-span prose leaked in"


def test_extract_spans_reads_the_bare_marker_carrier_in_a_docstring() -> None:
    """Bare marker lines inside a Python docstring, where `<!-- -->` is nonsense.

    Both carriers matter: four pinned sites are markdown and one is a docstring
    in `escalation/src/escalation/server.py`, and a guard that handled only one
    carrier would leave the other unpinned.
    """
    spans = extract_spans(_DOCSTRING_SPAN, source="server.py")

    assert [name for name, _ in spans] == ["CANCEL_STATES"]
    body = spans[0][1]
    assert "'superseded'" in body
    assert "Source of truth" not in body, "trailing pointer prose leaked in"


def test_extract_spans_reads_a_single_line_begin_comment() -> None:
    """The begin comment may close on its own line; the body starts after `-->`."""
    spans = extract_spans(_SET_NOTATION_SPAN, source=_FIXTURE_SOURCE)

    assert [name for name, _ in spans] == ["LIVE_STATES"]
    assert "finalizing" in spans[0][1]


def test_extract_spans_reads_every_span_in_order() -> None:
    """A file carries several spans; each is returned with its own partition."""
    spans = extract_spans(_TWO_SPANS, source=_FIXTURE_SOURCE)

    assert [name for name, _ in spans] == ["LIVE_STATES", "TERMINAL_STATES"]
    assert "verifying" in spans[0][1]
    assert "abandoned" in spans[1][1]


@pytest.mark.parametrize(
    ("text", "case", "expected_fragment"),
    [
        pytest.param(_SPAN_ABSENT, "no span at all", _FIXTURE_SOURCE, id="no-span"),
        pytest.param(_SPAN_UNTERMINATED, "unterminated span", "TERMINAL_STATES", id="unterminated"),
        pytest.param(_SPAN_NESTED, "nested span", "LIVE_STATES", id="nested"),
        pytest.param(_SPAN_UNOPENED_END, "end without begin", _FIXTURE_SOURCE, id="unopened-end"),
        pytest.param(
            _SPAN_UNKNOWN_PARTITION,
            "partition= names no partition",
            "TERMINAL_STATUSES",
            id="unknown-partition",
        ),
        pytest.param(_SPAN_NO_PARTITION, "no partition= at all", _FIXTURE_SOURCE, id="no-partition"),
    ],
)
def test_extract_spans_rejects_every_malformed_span(
    text: str, case: str, expected_fragment: str
) -> None:
    """Each way a span can be malformed fails LOUDLY, naming the offender.

    A missing span is the highest-value case: it is what an edit that deletes
    the markers while keeping the list produces, and returning `[]` for it would
    turn the site's whole drift check into a vacuous pass.
    """
    with pytest.raises(AssertionError) as excinfo:
        extract_spans(text, source=_FIXTURE_SOURCE)

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message or "server.py" in message, f"{case}: {message!r}"
    assert expected_fragment in message, f"{case}: message must name the offender: {message!r}"


# ---------------------------------------------------------------------------
# extract_values
# ---------------------------------------------------------------------------


_TERMINAL_LITERAL_STYLES = [
    pytest.param(
        "Terminal: `done`, `conflict`, `blocked`, `abandoned`, `superseded`",
        "markdown backticks",
        id="backticks",
    ),
    pytest.param(
        "TERMINAL = {\"done\", \"conflict\", 'blocked', 'abandoned', \"superseded\"}",
        "python string literals",
        id="python-literals",
    ),
    pytest.param(
        "state is done | conflict | blocked | abandoned | superseded",
        "bare pipe notation",
        id="pipes",
    ),
    pytest.param(
        "{done, conflict, blocked, abandoned, superseded}",
        "bare set notation",
        id="bare-set",
    ),
]


@pytest.mark.parametrize(("body", "case"), _TERMINAL_LITERAL_STYLES)
def test_extract_values_reads_every_literal_style_these_files_use(body: str, case: str) -> None:
    """All four styles present across the pinned sites yield the same set.

    The sites are not uniform and must not be forced to be: a SKILL.md renders
    backticks, a Python docstring writes quoted literals, and a fenced code
    block carries bare set notation. A guard that read only one style would
    silently pin nothing in the other files.
    """
    values = extract_values(body, vocabulary_values(load_vocabulary()))

    assert values == {"done", "conflict", "blocked", "abandoned", "superseded"}, case


def test_extract_values_keeps_only_vocabulary_members() -> None:
    """Surrounding prose words are dropped; only declared members survive.

    This is what lets the spans stay readable — a marked list may carry ordinary
    English inside it without every noun becoming a phantom member.
    """
    body = "**Terminal states** (`done`, `conflict`) — the poll loop stops here."

    assert extract_values(body, vocabulary_values(load_vocabulary())) == {"done", "conflict"}


def test_extract_values_keeps_a_member_of_the_OTHER_vocabulary() -> None:
    """A SUBMIT-only value inside a POLL span SURVIVES extraction.

    Load-bearing, and the reason the filter is the union of both vocabularies
    rather than the span's own partition: filtering by partition would drop the
    intruder before the comparison, and a span claiming `merge_status` returns
    `wip_halted` would read green. Surviving extraction is what makes it show up
    as an EXTRA value in `assert_span_matches`.
    """
    body = "Terminal: `done`, `wip_halted`"

    assert extract_values(body, vocabulary_values(load_vocabulary())) == {"done", "wip_halted"}


def test_extract_values_raises_when_it_extracts_nothing() -> None:
    """An empty extraction RAISES rather than returning an empty set.

    An empty set compares unequal to every partition, so this case would be
    caught downstream — but as "the list lost all five members", which sends the
    reader to fix a list that is fine. The real defect is a misplaced `:end`
    marker, and the message must say so.
    """
    with pytest.raises(AssertionError):
        extract_values("prose with no members at all", vocabulary_values(load_vocabulary()))


# ---------------------------------------------------------------------------
# assert_span_matches
# ---------------------------------------------------------------------------


def test_assert_span_matches_accepts_an_exact_match() -> None:
    """The green case: the span's set equals its partition exactly."""
    partitions = load_vocabulary()

    assert_span_matches(
        partitions["TERMINAL_STATES"],
        partitions["TERMINAL_STATES"],
        source=_FIXTURE_SOURCE,
        partition_name="TERMINAL_STATES",
    )


@pytest.mark.parametrize(
    ("text", "case", "expected_repr"),
    [
        pytest.param(
            _SPAN_MISSING_MEMBER,
            "a member the list forgot — today's escalation-watcher line",
            repr(["superseded"]),
            id="missing-member",
        ),
        pytest.param(
            _SPAN_EXTRA_MEMBER,
            "a SUBMIT-only value in a POLL list — also today's escalation-watcher line",
            repr(["already_merged"]),
            id="extra-member",
        ),
        pytest.param(
            _SPAN_WRONG_VOCABULARY,
            "a value from the other vocabulary entirely",
            repr(["wip_halted"]),
            id="wrong-vocabulary",
        ),
    ],
)
def test_assert_span_matches_rejects_every_drift_shape(
    text: str, case: str, expected_repr: str
) -> None:
    """Each drift shape fires, quoting the REPR of the offending values.

    A repr is contractually load-bearing: a reader cannot act on "the list has
    drifted" without being told WHICH value, so a message that stopped carrying
    it would have genuinely regressed. Pinning wording instead makes an
    editorial reword red while a message degraded to uselessness stays green.

    Missing and extra are reported SEPARATELY because the fixes differ: a
    missing value means the list is stale, an extra one usually means a value
    from the sibling vocabulary was pasted in.
    """
    partitions = load_vocabulary()
    (partition_name, body), = extract_spans(text, source=_FIXTURE_SOURCE)
    values = extract_values(body, vocabulary_values(partitions))

    with pytest.raises(AssertionError) as excinfo:
        assert_span_matches(
            values,
            partitions[partition_name],
            source=_FIXTURE_SOURCE,
            partition_name=partition_name,
        )

    message = str(excinfo.value)
    assert _FIXTURE_SOURCE in message, f"{case}: {message!r}"
    assert partition_name in message, f"{case}: {message!r}"
    assert expected_repr in message, f"{case}: message must quote the offending value: {message!r}"


def test_assert_span_matches_rejects_an_empty_value_set() -> None:
    """An empty set is refused up front, as a non-vacuity backstop.

    `extract_values` already raises on an empty extraction; this is the second
    line of defence for any future caller that builds a value set another way.
    """
    partitions = load_vocabulary()

    with pytest.raises(AssertionError):
        assert_span_matches(
            frozenset(),
            partitions["TERMINAL_STATES"],
            source=_FIXTURE_SOURCE,
            partition_name="TERMINAL_STATES",
        )


# ---------------------------------------------------------------------------
# find_unregistered_sites
# ---------------------------------------------------------------------------


def _write_fixture_site(directory: Path, name: str, tokens: list[str]) -> Path:
    """A throwaway markdown file naming `tokens`, for the registry scan's tests.

    Written under `tmp_path` rather than into the repo: the scan's own fixtures
    must not be discoverable BY the live scan, or `test_registry_is_complete`
    would go red on this module's test data.
    """
    body = "\n".join(f"- `{token}` — restated here" for token in tokens)
    path = directory / name
    path.write_text(f"# fixture site\n\n{body}\n", encoding="utf-8")
    return path


_DISCRIMINATING_SAMPLE = ["abandoned", "superseded", "verifying", "finalizing", "already_merged"]

# Every one of these is a real vocabulary member AND an ordinary English word or
# a task status. A file naming six of them is discussing merges, not enumerating
# the vocabulary — which is why they are excluded from the count.
_AMBIGUOUS_SAMPLE = ["done", "blocked", "queued", "conflict", "unknown", "error"]


def test_find_unregistered_sites_is_empty_when_every_site_is_registered(tmp_path: Path) -> None:
    """The green case, including the two shapes that must NOT be reported.

    A registered file over the threshold is fine (that is the point of the
    registry), and an unregistered file UNDER the threshold is fine too — it
    mentions states without enumerating them.
    """
    enumerating = _write_fixture_site(tmp_path, "enumerating.md", _DISCRIMINATING_SAMPLE)
    under = _write_fixture_site(tmp_path, "under-threshold.md", _DISCRIMINATING_SAMPLE[:3])

    assert (
        find_unregistered_sites(
            files=[enumerating, under],
            registry={str(enumerating): "enumerates the poll vocabulary"},
            threshold=4,
        )
        == []
    )


def test_find_unregistered_sites_reports_a_new_unpinned_site(tmp_path: Path) -> None:
    """A NEW enumeration site nobody registered is exactly what this scan is for.

    A drift guard whose site list is hand-maintained reproduces the defect it
    exists to prevent: it reads green while a freshly written runbook restates
    the vocabulary and drifts on the next member.
    """
    registered = _write_fixture_site(tmp_path, "registered.md", _DISCRIMINATING_SAMPLE)
    newcomer = _write_fixture_site(tmp_path, "newcomer.md", _DISCRIMINATING_SAMPLE[:4])

    assert find_unregistered_sites(
        files=[registered, newcomer],
        registry={str(registered): "enumerates the poll vocabulary"},
        threshold=4,
    ) == [str(newcomer)]


def test_find_unregistered_sites_counts_distinct_tokens_only(tmp_path: Path) -> None:
    """A file naming one state six times is discussing it, not enumerating.

    Counting occurrences would flag every doc that treats a single state in
    depth — noise that trains readers to register files to silence the guard,
    which is how a guard stops meaning anything.
    """
    repeated = _write_fixture_site(tmp_path, "repeated.md", ["superseded"] * 6)

    assert find_unregistered_sites(files=[repeated], registry={}, threshold=4) == []


def test_find_unregistered_sites_ignores_the_ambiguous_tokens(tmp_path: Path) -> None:
    """Six AMBIGUOUS members do not make an enumeration.

    `done`, `blocked`, `queued`, `conflict`, `unknown` and `error` are all task
    statuses or ordinary prose in this repo, so counting them would flag most of
    `skills/` — which is why the count runs over the DISCRIMINATING subset.
    """
    ambiguous = _write_fixture_site(tmp_path, "ambiguous.md", _AMBIGUOUS_SAMPLE)

    assert find_unregistered_sites(files=[ambiguous], registry={}, threshold=4) == []


def test_find_unregistered_sites_skips_the_declared_rule_sites(tmp_path: Path) -> None:
    """An explicitly declared ARM-RULE site is not reported as unregistered.

    A rule site enumerates a deliberately NARROWED subset (which statuses halt
    the queue, which states a particular loop may stop on). Pinning one to a
    partition would demand it re-add the very members it exists to exclude, so
    it is excluded by name, with its reason recorded beside it, rather than
    silently falling under the threshold.
    """
    rule_site = _write_fixture_site(tmp_path, "rule-site.md", _DISCRIMINATING_SAMPLE)

    assert (
        find_unregistered_sites(
            files=[rule_site],
            registry={},
            rule_sites={str(rule_site): "a narrowed halt-status rule, not a vocabulary copy"},
            threshold=4,
        )
        == []
    )


def test_find_unregistered_sites_raises_on_an_empty_scan() -> None:
    """An empty file list RAISES rather than returning `[]`.

    `[]` from an empty scan is indistinguishable from "every site is
    registered" — an over-broad pathspec would report the guard's strongest
    possible result while having read nothing at all.
    """
    with pytest.raises(AssertionError):
        find_unregistered_sites(files=[], registry={}, threshold=4)


def test_tracked_skill_markdown_fails_loudly_without_the_git_oracle(tmp_path: Path) -> None:
    """The tracked-file oracle raises when it cannot run, with no filesystem fallback.

    The scan's verdict must come from TRACKED files only, so that an untracked
    scratch file cannot flip it. Falling back to a filesystem walk on a failed
    oracle would restore that exact hazard in precisely the situation nobody is
    watching for it. `git` on PATH is not guaranteed either: a verify
    subprocess's PATH is rewritten by
    `orchestrator/src/orchestrator/verify.py::_target_subprocess_env`.
    """
    with pytest.raises(RuntimeError) as excinfo:
        tracked_skill_markdown(root=tmp_path)

    assert str(tmp_path) in str(excinfo.value)


# ---------------------------------------------------------------------------
# LIVE assertions. Every one re-reads the committed artifact fresh — never a
# snapshot taken at import time — and is parametrised over `PINNED_SITES` so a
# failure names the file that drifted rather than "some site".
# ---------------------------------------------------------------------------


def _is_tracked(relative_path: str) -> bool:
    """Is *relative_path* in the repo's index?

    Uses the same tracked-file oracle as the registry scan, so a registered site
    and a scanned one cannot disagree about what "exists" means.
    """
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--", relative_path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
        env=_scrubbed_git_env(),
    )
    return bool(completed.stdout.strip("\0").strip())


@pytest.mark.parametrize("relative_path", sorted(PINNED_SITES))
def test_pinned_sites_all_exist(relative_path: str) -> None:
    """Every registered path is tracked and present.

    A renamed or deleted site must not silently drop out of the registry: with
    the path gone, its span assertions would either error for an unrelated
    reason or — worse, if the registry were ever made forgiving — quietly stop
    checking a file that still restates the vocabulary under a new name.
    """
    assert _is_tracked(relative_path), (
        f"{relative_path} is in PINNED_SITES but is not tracked by git (task 4829) — "
        f"{PINNED_SITES[relative_path]}. Either the file was renamed (update the "
        f"registry) or it was deleted (drop the entry)."
    )
    assert (REPO_ROOT / relative_path).is_file(), (
        f"{relative_path} is tracked but missing from the worktree (task 4829)."
    )


@pytest.mark.parametrize("relative_path", sorted(PINNED_SITES))
def test_every_pinned_site_carries_at_least_one_span(relative_path: str) -> None:
    """Each registered site still carries its `merge-state-vocab` markers.

    Separated from the equality check below because the two failures mean
    different things: a MISSING span means the pin itself was lost (the list is
    now free to drift with nothing red), while a mismatched span means the pin
    is working and caught something.
    """
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    spans = extract_spans(text, source=relative_path)

    assert spans, f"{relative_path}: no spans — {PINNED_SITES[relative_path]}"


@pytest.mark.parametrize("relative_path", sorted(PINNED_SITES))
def test_every_span_matches_its_partition(relative_path: str) -> None:
    """THE drift assertion (PRD B9): every pinned list equals its partition exactly.

    This is the mechanism the capability manifest binds. Because every partition
    in `shared/src/shared/merge_state.py` is DERIVED rather than hand-listed, and
    `OUTCOME_STATES | EPISTEMIC_STATES` is asserted exhaustive and disjoint over
    `MergeState` in `shared/tests/test_merge_state.py`, adding a member to either
    enum forces a partition assignment — which in turn reddens every prose span
    pinned to that partition. The chain runs through the partitions, so no
    hand-copied master list sits anywhere in it.
    """
    partitions = load_vocabulary()
    values_filter = vocabulary_values(partitions)
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")

    for partition_name, body in extract_spans(text, source=relative_path):
        values = extract_values(body, values_filter, source=relative_path)
        assert_span_matches(
            values,
            partitions[partition_name],
            source=relative_path,
            partition_name=partition_name,
        )


def test_registry_is_complete() -> None:
    """No tracked `skills/` markdown enumerates the vocabulary without being pinned.

    The registry is not trusted — it is checked against a scan. This is what stops
    this guard from developing the very defect it exists to prevent: a new runbook
    restating the vocabulary, green today, stale on the next member.
    """
    unregistered = find_unregistered_sites()

    if unregistered:
        discriminating = discriminating_tokens(load_vocabulary())
        detail = []
        for label in unregistered:
            text = (REPO_ROOT / label).read_text(encoding="utf-8")
            found = sorted({t for t in _TOKEN_RE.findall(text) if t in discriminating})
            detail.append(f"{label}: {len(found)} distinct — {found!r}")
        raise AssertionError(
            "tracked `skills/` markdown enumerating >= "
            f"{_ENUMERATION_THRESHOLD} distinct discriminating merge-state values "
            f"without being registered (task 4829):\n  " + "\n  ".join(detail) + "\n"
            "Either wrap each list in a `merge-state-vocab` span and add the file to "
            "PINNED_SITES, or — if the list is a deliberately NARROWED arm rule "
            "rather than a copy of a vocabulary — add it to _UNPINNED_RULE_SITES "
            "with the reason, as `skills/orchestrate/SKILL.md` is."
        )


@pytest.mark.parametrize("relative_path", sorted(_UNPINNED_RULE_SITES))
def test_declared_rule_sites_are_tracked_and_span_free(relative_path: str) -> None:
    """The rule-site exclusion cannot silently absorb a real pinned list.

    Two ways an escape hatch rots, both closed here: the file is renamed and the
    exclusion starts covering nothing (checked as trackedness), or somebody wraps
    a genuine vocabulary list in that file and the exclusion keeps the registry
    scan from ever noticing (checked by requiring NO span — a file with a span
    belongs in PINNED_SITES, where its span is compared against its partition).
    """
    assert _is_tracked(relative_path), (
        f"{relative_path} is in _UNPINNED_RULE_SITES but is not tracked (task 4829) — "
        f"{_UNPINNED_RULE_SITES[relative_path]}. A stale exclusion covers nothing "
        f"while reading as though the file were reviewed."
    )
    text = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
    assert _BEGIN_MARKER not in text, (
        f"{relative_path} is excluded from the registry scan as an arm-rule site, but "
        f"now carries a `{_BEGIN_MARKER}` span (task 4829). A file with a span must be "
        f"in PINNED_SITES so the span is actually compared against its partition — "
        f"otherwise the span is decoration. Move the entry."
    )
