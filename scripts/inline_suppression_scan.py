"""Find every suppression in the tracked Python of a checkout.

``git ls-files`` gives the corpus; a byte prefilter over the kind table's marker
substrings drops the files that cannot hold a marker; the survivors are
tokenized and only their COMMENT tokens are read.  A file the scan cannot read
or tokenize is an :class:`~inline_suppression_refusal.InstrumentFailure`, never
a skip.
"""

from __future__ import annotations

import subprocess
import tokenize
from dataclasses import dataclass
from io import StringIO
from pathlib import Path, PurePosixPath

from inline_suppression_kinds import KIND_SPECS, Comment, Site
from inline_suppression_refusal import InstrumentFailure

#: Generous, and not a performance assertion: `git ls-files` over this tree
#: takes well under a second, so anything approaching this has hung.
_GIT_TIMEOUT_SECS = 60


def scan_source(source: str, *, path: str) -> tuple[Comment, ...]:
    """Every COMMENT token in *source*, each carrying the Sites it holds.

    TOKENIZE, NEVER REGEX OVER THE SOURCE TEXT: a string literal that merely
    mentions ``#`` is not a comment, and no regex gets that right.

    THE CATCH IS A TRIO, not ``TokenError`` alone: a file with broken
    indentation raises ``IndentationError`` and one with a bad statement raises
    ``SyntaxError``, and either would otherwise escape as a bare traceback
    instead of the named exit 2 boundary scenario 8 requires.

    Raises:
        InstrumentFailure: *source* could not be tokenized.  Never a skip —
            see :class:`InstrumentFailure` for why the polarity matters.
    """
    comments: list[Comment] = []
    try:
        for token in tokenize.generate_tokens(StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                continue
            comments.append(_comment_at(token, path=path))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        raise InstrumentFailure(
            f'{path}: could not be tokenized -- {type(exc).__name__}: {exc}'
        ) from exc
    return tuple(comments)


def _comment_at(token: tokenize.TokenInfo, *, path: str) -> Comment:
    """Build the :class:`Comment` for one COMMENT *token*.

    The kind table's marker substring is tested here as well as in the
    pre-decode prefilter, and for the same reason one scale down: a marker-free
    comment is the overwhelming majority, and a substring test is far cheaper
    than the five pattern searches it stands in front of.  One table, two
    granularities — not a second policy.

    THE PHYSICAL LINE IS STRIPPED ONLY FOR A COMMENT THAT MATCHED, which is the
    same economy one step further in.  The strip feeds nothing but a
    :class:`Site`, so every marker-free comment would otherwise allocate a
    string that is discarded on the next line — paid inside the one function
    the ten-second budget rests on.  The marker test
    already decides whether it is wanted, so nothing new is being consulted.
    """
    line = token.start[0]
    text = token.string
    matched = [
        (kind, spec.codes(match))
        for kind, spec in KIND_SPECS.items()
        if spec.may_match(text) and (match := spec.pattern.search(text)) is not None
    ]
    stripped = token.line.strip() if matched else ''
    sites = tuple(
        Site(path=path, line=line, kind=kind, codes=codes, text=stripped)
        for kind, codes in matched
    )
    return Comment(path=path, line=line, text=text, sites=sites)


@dataclass(frozen=True)
class Scan:
    """One sweep of a tree: what it found, and how much work it did.

    Attributes:
        comments: Every COMMENT token in every file that survived the
            prefilter, in enumeration order.
        files_enumerated: How many tracked paths this scan set out to read —
            what ``git ls-files`` listed, narrowed to the scope when the run is
            a scoped one, and INCLUDING any whose worktree file has since been
            deleted.
        files_tokenized: How many of those were actually decoded and tokenized
            — the ones whose raw bytes carried a marker substring.

    THE TWO COUNTS ARE THE BUDGET'S ENFORCEMENT POINT.  The PRD gives this scan
    ten seconds and the prefilter is the only reason it fits, so the guard
    asserts these two numbers rather than the clock;
    ``scripts/tests/test_inline_suppression_ratchet.py::test_the_live_scan_tokenizes_exactly_the_marker_bearing_files``
    says why.
    """

    comments: tuple[Comment, ...]
    files_enumerated: int
    files_tokenized: int

    @property
    def sites(self) -> tuple[Site, ...]:
        """Every :class:`Site` held by every comment, flattened.

        A property rather than a field, so it cannot go stale against
        ``comments`` — the two would otherwise be a redundant pair that a
        future edit could desynchronise (heuristic 11).
        """
        return tuple(site for comment in self.comments for site in comment.sites)


#: The case-sensitive kinds' prefilter substrings as raw bytes, for the
#: pre-decode pass.  Derived from the one table rather than written twice; the
#: markers are ASCII, so the encoding is exact.
_EXACT_MARKER_BYTES: tuple[bytes, ...] = tuple(
    spec.marker.encode('utf-8') for spec in KIND_SPECS.values() if not spec.folds_case
)

#: The case-folding kinds' substrings, tested against the lower-cased bytes.
_FOLDED_MARKER_BYTES: tuple[bytes, ...] = tuple(
    spec.marker.encode('utf-8') for spec in KIND_SPECS.values() if spec.folds_case
)


def _may_hold_marker(raw: bytes) -> bool:
    """The byte prefilter: could any kind's pattern match somewhere in *raw*?"""
    if any(marker in raw for marker in _EXACT_MARKER_BYTES):
        return True
    folded = raw.lower()
    return any(marker in folded for marker in _FOLDED_MARKER_BYTES)


def _tracked_python_files(root: Path) -> tuple[str, ...]:
    """Every TRACKED ``*.py`` under *root*, repo-relative, sorted and unique.

    REFUSES RATHER THAN DEGRADING TO ``[]``.  An empty corpus and a clean
    corpus are indistinguishable in a violation count, and only one of them is
    good news — the argument
    ``scripts/audit_manifest_descriptor_drift.py::ManifestDiscoveryUnavailable``
    makes.  It is sharper here: an empty scan handed to ``--seed`` or
    ``--tighten`` writes an empty baseline, which compares clean against
    everything and opens the gate for good.

    The return code is inspected by hand rather than with ``check=True``, so
    the refusal can carry git's own stderr.  Deduped through a set because an
    UNMERGED path is listed once per merge stage, and sorted so a scan's
    enumeration order — and therefore a report's — is reproducible.
    """
    try:
        completed = subprocess.run(
            ['git', '-C', str(root), 'ls-files', '-z', '--', '*.py'],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InstrumentFailure(
            f'could not run `git ls-files` in {root} -- {type(exc).__name__}: {exc}. '
            'The tracked corpus is this scan\'s only source; there is no filesystem-walk '
            'fallback, because a walk would silently restore untracked working-tree state '
            'as an input to the gate.'
        ) from exc

    if completed.returncode != 0:
        raise InstrumentFailure(
            f'`git ls-files` failed in {root} (rc={completed.returncode}): '
            f'{completed.stderr.strip() or "no stderr"}'
        )

    return tuple(sorted({path for path in completed.stdout.split('\0') if path}))


def _source_of(root: Path, relative: str) -> str | None:
    """Decode *relative*'s bytes, or ``None`` when the prefilter rejects it.

    Reading the raw BYTES and testing them for a marker substring before
    decoding is the whole prefilter: the files that cannot hold a suppression
    are never decoded and never tokenized.

    ``FileNotFoundError`` is the one read failure that is not a fault:
    ``git ls-files`` reads the INDEX, so it lists a path whose worktree file
    has been deleted mid-edit, and that is an ordinary state rather than a
    broken instrument.  Every OTHER ``OSError``, and any decode failure,
    raises — a file that exists and cannot be read is exactly the case where
    carrying on would measure low.
    """
    location = root / relative
    try:
        raw = location.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise InstrumentFailure(
            f'{relative}: could not be read -- {type(exc).__name__}: {exc}'
        ) from exc

    if not _may_hold_marker(raw):
        return None

    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise InstrumentFailure(
            f'{relative}: could not be decoded as UTF-8 -- {exc}. A tracked Python file '
            'this scanner cannot decode is never skipped: it might hold the very '
            'suppression the gate exists to find.'
        ) from exc


def _within(relative: str, scope: tuple[str, ...]) -> bool:
    """Whether *relative* is one of *scope*'s entries, or sits under one.

    COMPARED AS PATH COMPONENTS, never as a string prefix: ``scripts`` must not
    scope ``scripts_old/`` and ``sc`` must not scope anything, which is the same
    mistake in the same shape as the string-prefix one
    ``scripts/inline_suppressions.py::selects`` refuses.
    Both sides are ``PurePosixPath``, because ``git ls-files`` emits forward
    slashes on every platform and a scope typed as ``./pkg`` should mean ``pkg``.
    """
    wanted = tuple(PurePosixPath(entry).parts for entry in scope)
    parts = PurePosixPath(relative).parts
    return any(parts[: len(entry)] == entry for entry in wanted)


def scan_tree(root: Path, *, scope: tuple[str, ...] = ()) -> Scan:
    """Scan every tracked ``*.py`` under *root*, or only those under *scope*.

    The prefilter means ``files_tokenized`` is strictly smaller than
    ``files_enumerated`` on any real tree, and a file it rejects is reported
    honestly as enumerated-not-tokenized rather than silently vanishing.

    A SCOPE NARROWS THE ENUMERATION, not the report of it: a scoped run reads
    fewer files and says so in both counts, which is what makes it the cheap
    early-feedback run D12 keeps it for.  Every verb that a partial view could
    mislead refuses a scope outright rather than relying on this being noticed.

    A SCOPE THAT SELECTS NOTHING IS REFUSED, for the reason
    :func:`_tracked_python_files` refuses an empty corpus one step earlier —
    and reaching it needs no broken git, only a mistyped path on the
    early-feedback run.  The ``0 of 0`` the report line prints is a tell only a
    reader who looks will catch, and this module chooses refusal over that.

    Raises:
        InstrumentFailure: the tracked corpus is empty, or *scope* selected
            none of it.
    """
    tracked = _tracked_python_files(root)
    if scope:
        tracked = tuple(path for path in tracked if _within(path, scope))
        if not tracked:
            raise InstrumentFailure(
                f'scope {", ".join(scope)} matched no tracked *.py file under {root}. '
                'An empty scan and a clean scan are indistinguishable in a violation '
                'count, and only one of them is good news -- the same argument '
                '`_tracked_python_files` makes for an empty corpus.'
            )
    comments: list[Comment] = []
    tokenized = 0
    for relative in tracked:
        source = _source_of(root, relative)
        if source is None:
            continue
        tokenized += 1
        comments.extend(scan_source(source, path=relative))
    return Scan(
        comments=tuple(comments),
        files_enumerated=len(tracked),
        files_tokenized=tokenized,
    )
