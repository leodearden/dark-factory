"""Module path normalization for scheduler lock keys and task-corpus module matching.

The orchestrator's scheduler uses fixed-depth path prefixes as lock keys so that two
tasks touching `crates/reify-compiler/src/foo.rs` and `crates/reify-compiler/src/bar.rs`
serialize on the same lock. The task curator reuses the same normalization to find
tasks whose module footprint overlaps a candidate.

Directory/file path classifier (α/γ shared predicate)
------------------------------------------------------
``FILE_EXTENSIONS``, ``is_file_path``, ``directory_locks``, and
``strip_directory_locks`` live here so the orchestrator scheduler (α
enforcement point) can import them without a ``fused_memory`` dependency
(the orchestrator must not import fused_memory; see
``agents/triage.py:78-95``).

``lock_charter_guard.py`` keeps **verbatim duplicate definitions** of these
names so it remains self-contained in the fused-memory virtual environment
(the editable shared package in ``.venv`` may be pinned to a release that
predates this relocation).  The two copies are **not** connected by a
re-export; drift between them is caught by explicit equality tests:

- ``shared/tests/test_locking.py::TestFileExtensionsDriftGuard``
  (pins this copy)
- ``fused-memory/tests/test_lock_charter_guard.py::test_extension_drift_guard``
  (pins the lock_charter_guard.py copy)

STATUS — deliberate cross-repo divergence (dark_factory #3248)
--------------------------------------------------------------
``EXTENSIONLESS_FILENAMES`` (below) is a dark_factory-only lead.  reify's
``scripts/lock-charter-guard.sh`` still conservative-rejects every one of
those 8 names, per its PRD §5.2 ``files=[]`` escape hatch, so the two
predicates now DISAGREE on exactly those names and only those names.  This
is the mirror image of #3117's one-sided lead, and it is recorded loudly
here rather than left to be rediscovered.

The Tier-2 cross-source drift guard
(``test_extension_drift_guard_vs_reify_script``) is UNAFFECTED, and that is
measured rather than assumed: it compares ``sorted(FILE_EXTENSIONS)`` against
``lock-charter-guard.sh --list-extensions``, i.e. an EXTENSION vector only.  A
separate frozenset is invisible to it.  So this divergence will NOT surface as
a red test — which is exactly why it is written down.  Mirroring it into reify
is filed as a follow-up.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = [
    'EXTENSIONLESS_FILENAMES',
    'FILE_EXTENSIONS',
    'directory_locks',
    'files_to_modules',
    'is_file_path',
    'modules_conflict',
    'normalize_lock',
    'strip_directory_locks',
]

# ---------------------------------------------------------------------------
# Canonical extension allowlist — authoritative Python copy for the α enforcement point.
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
#
# Copied verbatim from reify's scripts/lock-charter-guard.sh _EXTS (reify:4676).
# git-ls-files sweep 2026-07-28 (reify #5726 / dark_factory #3117) — 22 tracked-file
# extensions across reify + dark-factory that this list misclassified as directories;
# supersedes #4676's "OQ#2 resolved" completeness claim FOR THIS LIST, which was a
# 22-extension undercount:
#   conf diff envrc example example-systemd-config gitattributes gitignore gitkeep
#   gitmodules golden grammar icns ico jq jsonl log manifest npmrc python-version
#   template timer typed
# Widened 58 -> 59 on 2026-08-06: `csv` — caught by the corpus->allowlist guard
# (test_every_tracked_extension_is_allowlisted) once plans/evidence/
# scheduler-scoring-2026-08-06/*.csv landed on dark-factory main (e1c51efa8d).
# Drift guard (this shared copy): shared/tests/test_locking.py::TestFileExtensionsDriftGuard
# Drift guard (γ copy in lock_charter_guard.py): fused-memory/tests/test_lock_charter_guard.py::test_extension_drift_guard
# ---------------------------------------------------------------------------

FILE_EXTENSIONS: frozenset[str] = frozenset(
    {
        'c',
        'cc',
        'cjs',
        'conf',
        'cpp',
        'css',
        'csv',
        'cts',
        'cxx',
        'diff',
        'envrc',
        'example',
        'example-systemd-config',
        'gcode',
        'gitattributes',
        'gitignore',
        'gitkeep',
        'gitmodules',
        'golden',
        'grammar',
        'h',
        'hh',
        'hpp',
        'html',
        'icns',
        'ico',
        'jq',
        'js',
        'json',
        'jsonc',
        'jsonl',
        'jsx',
        'lock',
        'log',
        'manifest',
        'md',
        'mjs',
        'mts',
        'npmrc',
        'png',
        'py',
        'python-version',
        'ri',
        'rs',
        'scss',
        'service',
        'sh',
        'step',
        'stl',
        'svg',
        'template',
        'timer',
        'toml',
        'ts',
        'tsx',
        'txt',
        'typed',
        'yaml',
        'yml',
    }
)

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
# Drift guard (this shared copy): shared/tests/test_locking.py::TestExtensionlessFilenamesDriftGuard
# Drift guard (γ copy in lock_charter_guard.py): fused-memory/tests/test_lock_charter_guard.py::test_extensionless_drift_guard
# Self-audit (both copies): test_every_tracked_extensionless_file_is_allowlisted
#   sweeps both repos' tracked corpora, so the 9th extension-less file surfaces
#   as a red test rather than as another LockCharterViolation incident.
# ---------------------------------------------------------------------------

EXTENSIONLESS_FILENAMES: frozenset[str] = frozenset(
    {
        'Dockerfile',
        'LICENSE',
        'cargo',
        'cargo-audit-orphans',
        'pre-commit',
        'pre-merge-commit',
        'project-checks',
        'reference-transaction',
    }
)


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


def strip_directory_locks(files: list[Any]) -> list[str]:
    """Return only the file-level entries from *files* (inverse of directory_locks).

    Non-string, empty, and whitespace-only tokens are silently skipped.
    Used as a pre-filter before lock derivation so directory charters do not
    produce subtree-wide prefix locks.
    """
    return [f for f in files if isinstance(f, str) and f.strip() and is_file_path(f)]


def modules_conflict(a: str, b: str) -> bool:
    """Two module locks conflict if one is a prefix of the other (or exact match).

    This is the canonical hierarchical-conflict rule used by the orchestrator's
    ``ModuleLockTable`` (which delegates to it) and by the dashboard's holder
    lookup. Keeping it here means the prefix semantics live in exactly one place.
    """
    return a == b or a.startswith(b + '/') or b.startswith(a + '/')


def normalize_lock(module: str, depth: int = 2) -> str:
    """Normalize a module path to a fixed depth for lock granularity.

    e.g. normalize_lock('crates/reify-types/src/persistent.rs') -> 'crates/reify-types'
    """
    if not module:
        return module
    parts = module.strip('/').split('/')
    return '/'.join(parts[:depth])


def files_to_modules(files: Iterable[str], depth: int) -> list[str]:
    """Derive unique module locks from a list of file paths.

    Each file path is normalized to ``depth`` components, then deduplicated and
    returned sorted so callers get a stable ordering.
    """
    modules: set[str] = set()
    for f in files:
        normalized = normalize_lock(f, depth)
        if normalized:
            modules.add(normalized)
    return sorted(modules)
