"""Lock-charter directory-vs-file predicate guard (γ).

Backstop at the fused-memory task-creation path (``submit_task`` /
``commit_planning``) that REJECTS a directory string in ``metadata.files``
with a clear error, while ACCEPTING a file-level set or ``[]`` (the
deliberate defer-to-architect value).

This catches every creation path the /prd decompose guard (β) misses —
most importantly human-decompose (``submit_task(planning_mode=True)``  →
``commit_planning``), the dominant origin of over-wide directory locks
migrated verbatim from ``metadata.modules`` by commit fabfa367f5.

## Predicate contract (C-P1..C-P4, PRD §4.1)

The α/γ shared predicate is a *pure string* classification — no filesystem
stat, no model call (C-P3).  C-P3 determinism means the Python
implementation here and reify's Bash ``_is_file_path`` in
``scripts/lock-charter-guard.sh`` (reify task 4676, status=done) are
equivalent *by construction*: same extension allowlist, same extension-less
filename allowlist, same strip/segment/ext algorithm.

The drift-guard test in ``tests/test_lock_charter_guard.py`` pins
``sorted(FILE_EXTENSIONS)`` to the shared α/γ canonical vector printed by
``lock-charter-guard.sh --list-extensions``, so divergence is caught at CI
time.

STATUS — cross-repo agreement on the extension-less vector (#3248)
-------------------------------------------------------------------
``EXTENSIONLESS_FILENAMES`` (below) is NOT a dark_factory-only lead, and this
paragraph exists because #3248 was planned on the opposite assumption.  reify
got there FIRST: ``scripts/lock-charter-guard.sh`` already carries the same
8-name vector inside its bash ``_is_file_path`` and exposes it via
``--list-extensionless`` (reify #5890).  Measured 2026-08-09 on the live
script — the two vectors are identical, and ``classify hooks/project-checks``
returns ACCEPT, not REJECT.  So this change CONVERGED the seam rather than
opening one; no reify follow-up is owed.

The extension-half Tier-2 drift guard
(``test_extension_drift_guard_vs_reify_script``) compares
``sorted(FILE_EXTENSIONS)`` against ``--list-extensions``, i.e. an EXTENSION
vector only, and is structurally blind to this frozenset.  That gap is closed
by its extension-less sibling
(``test_extensionless_drift_guard_vs_reify_script``), which pins this vector
against ``--list-extensionless`` so three-way α/γ/bash agreement is ENFORCED
rather than merely asserted in prose.

Extension allowlist rationale (reify PRD §11 Q2):
- PRD-explicit: rs ri toml cpp c h hpp md json yaml yml lock py sh ts tsx js txt step stl
- Corpus-evidenced: css mjs html jsonc gcode service
- Common source siblings: cc cxx hh mts cts cjs jsx scss svg png
- git-ls-files sweep 2026-07-28 (reify #5726 / dark_factory #3117) — 22
  tracked-file extensions across reify + dark-factory that this list
  misclassified as directories; supersedes #4676's "OQ#2 resolved" completeness
  claim FOR THIS LIST, which was a 22-extension undercount:
  conf diff envrc example example-systemd-config gitattributes gitignore gitkeep
  gitmodules golden grammar icns ico jq jsonl log manifest npmrc python-version
  template timer typed

Extension-LESS filename rationale (dark_factory #3248):
- The extension allowlist above can never reach a final segment with no dot at
  all — there is no token to look up — so every extension-less tracked FILE
  (``LICENSE``, ``Dockerfile``, ``hooks/pre-commit``, ``hooks/project-checks``)
  classified as a DIRECTORY.  Two distinct faults followed: γ rejected such a
  declaration outright with a LockCharterViolation, and α — silently — stripped
  it, so a task declaring ONLY such paths got an empty charter and fell through
  to a per-task ``task-<id>`` synthetic lock that conflicts with nothing.
- Fixed with a SECOND enumerated frozenset rather than a general rule, for the
  same reason the extension list stays enumerated: "an extension-less segment
  naming an executable is a file" is not evaluable as pure string (C-P3) and
  would re-admit real directories.
- Vector measured by ``git ls-files -s`` over both repos; submodule gitlinks
  (mode 160000) deliberately excluded.  Full evidence and the admission rule
  are on the ``EXTENSIONLESS_FILENAMES`` definition below.

## Predicate canonical location

The α/γ shared predicate (``FILE_EXTENSIONS``, ``is_file_path``,
``directory_locks``) is canonical in ``shared.locking`` so that
``orchestrator.scheduler`` (α) can import it without a
``fused_memory`` → ``orchestrator`` cross-package edge (no-cross-package-edge
convention documented at ``agents/triage.py:78-95``; orchestrator's pyright
``extraPaths`` does not include ``../fused-memory/src``).

The definitions are **duplicated here verbatim** (not re-exported from
``shared.locking``) so this module remains self-contained in the fused-memory
virtual environment (the editable shared package installed in ``.venv`` may be
pinned to a release that predates the relocation).  These two copies are
**not** linked by a re-export.  Drift is caught by two explicit equality
tests:

- ``tests/test_lock_charter_guard.py::test_extension_drift_guard``
  (pins this copy against ``_CANONICAL_EXTENSIONS``)
- ``shared/tests/test_locking.py::TestFileExtensionsDriftGuard``
  (pins the ``shared.locking`` copy against the same canonical vector)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from shared.task_metadata import parse_metadata

logger = logging.getLogger(__name__)

# SchemaWarning codes (shared.task_metadata) that mean the WHOLE metadata
# value was discarded — parse_metadata could not produce any dict at all.
# Only these are loud here; 'unknown_key'/'invalid_field'/'invalid_submodel'
# etc. are expected on legitimate curator-internal metadata and belong to
# the write boundary (W3-β), not this read-path guard.
_DISCARD_CODES = frozenset({'unparseable_json', 'not_an_object'})

# ---------------------------------------------------------------------------
# Canonical extension allowlist — kept in sync with shared.locking.FILE_EXTENSIONS.
# Drift guard: tests/test_lock_charter_guard.py::test_extension_drift_guard
#
# NAMING — the contract.  This is the set of recognised FILE-EXTENSION TOKENS
# whose presence after the last dot of a path's final segment is evidence that
# the segment names a FILE rather than a DIRECTORY — which is all any caller
# (is_file_path, directory_locks, strip_directory_locks,
# module_charter.derive_modules) relies on.  It is NOT a code-file allowlist, so
# its non-code members are correct and load-bearing: png svg ico icns log lock
# diff golden manifest timer typed service template example python-version
# gitignore gitkeep gitmodules gitattributes npmrc envrc conf.
# ADMISSION RULE: judge a candidate addition by "does a real tracked file end in
# this?", NEVER by "is this code?".  Reading this list's former, CODE_-prefixed
# name as if it meant "code" is exactly what let it sit at a 22-extension
# undercount until #3117 — the history this rule exists to prevent repeating,
# and the reason the name now says FILE.
# Renamed to FILE_EXTENSIONS by #3248, the dedicated follow-up #3117 deferred
# that rename to.
# ---------------------------------------------------------------------------

__all__ = [
    'EXTENSIONLESS_FILENAMES',
    'FILE_EXTENSIONS',
    'is_file_path',
    'directory_locks',
    'extract_files',
    'lock_charter_error',
]

FILE_EXTENSIONS: frozenset[str] = frozenset({
    'c', 'cc', 'cjs', 'conf', 'cpp', 'css', 'csv', 'cts', 'cxx',
    'diff', 'envrc', 'example', 'example-systemd-config',
    'gcode', 'gitattributes', 'gitignore', 'gitkeep', 'gitmodules', 'golden', 'grammar',
    'h', 'hh', 'hpp', 'html',
    'icns', 'ico',
    'jq', 'js', 'json', 'jsonc', 'jsonl', 'jsx',
    'lock', 'log',
    'manifest', 'md', 'mjs', 'mts',
    'npmrc', 'png', 'py', 'python-version',
    'ri', 'rs', 'scss', 'service', 'sh', 'step', 'stl', 'svg',
    'template', 'timer', 'toml', 'ts', 'tsx', 'txt', 'typed',
    'yaml', 'yml',
})

# ---------------------------------------------------------------------------
# Canonical extension-LESS filename allowlist (dark_factory #3248).
#
# CONTRACT: recognised final-segment NAMES that carry no extension at all yet
# denote a FILE.  FILE_EXTENSIONS above can never reach these — there is no dot,
# so there is no token to look up — which is why they need a second enumerated
# vector rather than another entry in the first.  Matched against the final
# segment BEFORE the "no dot ⇒ directory" reject in is_file_path.
#
# EVIDENCE: `git ls-files -s` over BOTH repos, re-measured 2026-08-09 (identical
# to the 2026-07-31 measurement) — every tracked path whose final segment
# contains no '.'.  dark-factory contributes Dockerfile LICENSE pre-commit
# pre-merge-commit project-checks; reify adds cargo cargo-audit-orphans
# reference-transaction.  Union = the 8 below.
#
# GITLINK EXCLUSION — load-bearing, and a deliberate decision rather than an
# accident of the sweep: `git ls-files -s` entries with mode 160000 are submodule
# mount points (`graphiti`, `mem0` in dark-factory).  They are extension-less and
# would otherwise qualify, but admitting them would let a task declare an ENTIRE
# VENDORED SUBMODULE as its lock charter — strictly worse than the bug being
# fixed.  They stay DIRECTORIES.  The corpus sweeps assert this filter is doing
# real work rather than passing vacuously.
#
# CASE-SENSITIVITY: exact tracked spelling only.  `LICENSE` matches; `license`
# and `License` do not.  Same rule as the (lowercase) extension allowlist: a
# tracked file whose case differs is a naming problem, not an allowlist gap.
#
# ADMISSION RULE, as for FILE_EXTENSIONS: judge a candidate by "is this a real
# tracked file?", never by "does it look like a file?".  The list must stay
# ENUMERATED — a general rule such as "an extension-less segment naming an
# executable is a file" cannot be evaluated as pure string (C-P3) and would
# re-admit directories.
#
# CROSS-PROJECT SCOPE — the limitation this enumeration knowingly accepts, and
# the reason the admission rule above is stricter than it looks.  Members are
# matched as BARE FINAL SEGMENTS and several are generic: `cargo`, `Dockerfile`,
# `LICENSE`, `pre-commit`.  The collision guard that bounds this risk
# (test_no_canonical_extensionless_name_is_also_a_directory) sweeps only the
# dark-factory and reify checkouts — but this predicate governs lock charters
# for EVERY project the orchestrator targets.  In a third project containing
# `tools/cargo/` or `docker/Dockerfile/` as a real DIRECTORY, that directory now
# classifies as a FILE: directory_locks stops rejecting it and derive_modules
# RETAINS it, yielding a prefix lock over the entire subtree (modules_conflict
# is prefix-based) — the reify-3468 over-wide-charter failure the strip exists
# to prevent, and now with NO INFO diagnostic, because nothing was stripped.
# Admitting a generic basename is therefore a CROSS-PROJECT COMMITMENT, not a
# local one: weigh a candidate against every targeted repo, not only the two
# that are actually swept.
#
# Drift guard (this γ copy): fused-memory/tests/test_lock_charter_guard.py::test_extensionless_drift_guard
# Drift guard (shared copy in shared/locking.py): shared/tests/test_locking.py::TestExtensionlessFilenamesDriftGuard
# Self-audit (both copies): test_every_tracked_extensionless_file_is_allowlisted
#   sweeps both repos' tracked corpora, so the 9th extension-less file surfaces
#   as a red test rather than as another LockCharterViolation incident.
# ---------------------------------------------------------------------------

EXTENSIONLESS_FILENAMES: frozenset[str] = frozenset({
    'Dockerfile',
    'LICENSE',
    'cargo',
    'cargo-audit-orphans',
    'pre-commit',
    'pre-merge-commit',
    'project-checks',
    'reference-transaction',
})


def is_file_path(path: str) -> bool:
    """Return True iff *path* is a file-level declaration (not a directory).

    Mirrors ``_is_file_path`` in reify's ``scripts/lock-charter-guard.sh``
    exactly — pure string, no filesystem access, no model call (C-P3).

    Algorithm (C-P1..C-P3):
    1. Strip ALL trailing ``/`` from *path*.
    2. Final segment = substring after last ``/``.
       Empty segment (path was all slashes) → directory → return False.
    3. If the segment is in ``EXTENSIONLESS_FILENAMES`` → file → return True.
       Checked BEFORE the no-dot reject below, because these names carry no
       extension and would otherwise be rejected at step 4 (dark_factory #3248).
    4. If the segment contains no ``.`` → extension-less → return False.
    5. ext = substring after the last ``.`` in the segment.
       Return ``ext in FILE_EXTENSIONS`` (case-sensitive, matching α's
       ``[ "$ext" = "$e" ]`` against a lowercase allowlist).

    C-P3 is PRESERVED by step 3: it is a set membership test on a string, so
    the predicate remains pure string — no stat, no filesystem access, no model
    call.  A ghost path that is not on disk still classifies deterministically,
    exactly as before.
    """
    # Strip all trailing slashes.
    p = path
    while p.endswith('/'):
        p = p[:-1]

    # Final segment (everything after the last '/').
    seg = p.rsplit('/', 1)[-1] if '/' in p else p

    # Empty segment → path was all slashes → directory.
    if not seg:
        return False

    # Recognised extension-less FILE name → file.  Must precede the no-dot
    # reject below, which would otherwise swallow every one of these.
    if seg in EXTENSIONLESS_FILENAMES:
        return True

    # No dot in segment → extension-less → treat as directory (REJECT).
    if '.' not in seg:
        return False

    # Extension = substring after the last dot.
    ext = seg.rsplit('.', 1)[1]

    # A dotted segment is a FILE iff its post-last-dot substring is allowlisted.
    # There is NO dotfile special case: a leading dot is just a dot, so a
    # leading-dot segment reaches this lookup like any other and its "extension"
    # is the whole name after that dot.  Both outcomes are load-bearing:
    #   '.gitignore' → ext='gitignore' — in allowlist → True (a real file).
    #   '.worktrees' → ext='worktrees' — NOT in allowlist → False (a real
    #                  directory).  This is why the allowlist must stay
    #                  ENUMERATED rather than growing a "leading dot ⇒ file"
    #                  rule: that rule would make '.worktrees' (the whole
    #                  worktree pool) declarable as a lock charter.
    # 'f.PY' → ext='PY' — not in (lowercase) allowlist → False.  Correct.
    return ext in FILE_EXTENSIONS


# ---------------------------------------------------------------------------
# List-gate helpers
# ---------------------------------------------------------------------------


def directory_locks(files: list[Any]) -> list[str]:
    """Return the ordered, de-duplicated list of directory-like entries in *files*.

    Non-string, empty, and whitespace-only tokens are silently skipped
    (mirrors ``_extract_metadata_files`` benign-absent convention).
    Order is preserved from the input; duplicates are collapsed to the first
    occurrence.
    """
    seen: set[str] = set()
    result: list[str] = []
    for f in files:
        if not isinstance(f, str) or not f.strip():
            continue
        if not is_file_path(f) and f not in seen:
            seen.add(f)
            result.append(f)
    return result


def extract_files(metadata: str | dict[str, Any] | None) -> list[str]:
    """Parse *metadata* (dict, JSON string, or None) and return ``metadata.files``.

    Mirrors the benign-absent logic of ``_extract_metadata_files`` in
    ``task_interceptor.py``: malformed or absent metadata is treated as
    no-files (return []) rather than raising.

    Relationship to ``_extract_metadata_files``:
        ``_extract_metadata_files`` takes a full task dict (``{'metadata': {...}}``
        shape) and emits logger.WARNING on unexpected shapes.  This function is
        intentionally self-contained: it accepts the *metadata value* directly
        (dict, JSON string, or None) so the guard can handle both the
        submit_task path (inline metadata arg) and the commit_planning path
        (get_task result's ``metadata`` value) without wrapping.  The JSON-string
        and None normalization are the only genuinely new behaviours; the dict-path
        logic is equivalent and kept local to avoid importing a private function
        across module boundaries.

    Rules:
    - ``None`` → []
    - ``dict`` → use directly
    - ``str`` → ``json.loads``; on failure or non-dict result → []
    - If the resolved dict has no ``files`` key → []
    - If ``files`` is not a list → []
    - Return only the string entries of the list.

    The ``str`` branch delegates its malformed-input policy to
    :func:`shared.task_metadata.parse_metadata` (``direction='read'``) and
    emits a ``task_metadata.schema_warning`` WARNING when *metadata* cannot
    be resolved to a JSON object at all (unparseable JSON / non-object) — I4:
    this replaces a silent ``[]`` coercion. The dict-path extraction below is
    untouched, so mixed-type-list filtering and non-list-``files`` behaviour
    are unchanged.
    """
    if metadata is None:
        return []

    parsed: dict[str, Any] | None = None
    if isinstance(metadata, dict):
        parsed = metadata
    elif isinstance(metadata, str) and not metadata:
        # Empty string is benign-absent, not a discard — mirrors the
        # pre-collapse `isinstance(raw, str) and raw` guard.
        return []
    elif isinstance(metadata, str):
        _, warnings = parse_metadata(metadata, direction='read')
        discard = [w for w in warnings if w.code in _DISCARD_CODES]
        if discard:
            reason = '; '.join(w.message for w in discard) or 'unrecognised shape'
            logger.warning(
                'task_metadata.schema_warning source=lock_charter_guard.extract_files '
                'error=%s (type=%s); metadata discarded',
                reason,
                type(metadata).__name__,
            )
            return []
        # parse_metadata already confirmed this parses to a JSON object —
        # reparse independently for the raw dict (see docstring above: never
        # model_dump()).
        parsed = json.loads(metadata)

    if parsed is None:
        return []

    files = parsed.get('files')
    if not isinstance(files, list):
        return []

    return [f for f in files if isinstance(f, str)]


def lock_charter_error(
    directories: list[str],
    task_id: str | None = None,
) -> dict[str, Any]:
    """Return a structured LockCharterViolation error dict.

    Shape mirrors ``_done_gate_error`` / ``PathGuardVerdict.to_error_dict``
    so MCP callers handle it uniformly.

    Args:
        directories: The offending directory-like path strings.
        task_id: Optional task id to name in the error message (used by
            ``commit_planning`` to name the first offending task).
    """
    task_clause = f' (task {task_id})' if task_id else ''
    dir_list = ', '.join(repr(d) for d in directories)
    return {
        'error': (
            f'metadata.files contains directory declarations{task_clause}: '
            f'{dir_list}. '
            f'Declare individual file paths (e.g. "src/foo.py") or use '
            f'files=[] to defer scope to the architect.'
        ),
        'error_type': 'LockCharterViolation',
        'directory_paths': directories,
        'hint': (
            'Replace each directory with the specific file paths it contains, '
            'or use files=[] (the defer-to-architect value) to leave scope '
            'unspecified. Directory strings in metadata.files are rejected '
            'because they create over-wide lock charters that block unrelated '
            'work in the same directory. '
            'Note: extension matching is case-sensitive against a lowercase '
            'allowlist (e.g. "README.MD" or "config.YAML" is classified as a '
            'directory — use lowercase extensions like "README.md" instead). '
            'If the offending path has NO extension at all, "replace it with '
            'the files it contains" may be unfollowable because it IS a file '
            '(e.g. "hooks/project-checks"): an extension-less final segment '
            'counts as a file only when its exact name is listed in '
            'EXTENSIONLESS_FILENAMES. If it is a genuinely new extension-less '
            'tracked file, add its name to EXTENSIONLESS_FILENAMES in BOTH '
            'shared/src/shared/locking.py and '
            'fused-memory/src/fused_memory/middleware/lock_charter_guard.py, '
            'AND to reify scripts/lock-charter-guard.sh _EXTLESS (all three '
            'must agree — the Tier-2 --list-extensionless drift guard goes RED '
            'otherwise), plus the _CANONICAL_EXTENSIONLESS vector in both test '
            'copies.'
        ),
    }
