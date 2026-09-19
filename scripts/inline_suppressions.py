"""INV-12's inline-suppression scanner, consumer model and multiset ratchet.

**What this is.**  Every ``# type: ignore``, ``# noqa``, ``# pyright: ignore``,
``# pragma: no cover`` and ``# nosec`` in the tracked Python of this repository
silences a detector.  INV-12 ("exceptions are owned or ratified") says each one
carries a *disposition* — a named owner who will remove it, or a recorded
operator ruling that it stays.  This module finds them, decides which are
owned, and ratchets the rest so the population can shrink but never grow.
``plans/inv12-exceptions-owned-or-ratified-prd.md``, decisions **D6**-**D9**
and **D12**.

**The four layers**, each reading only the one below it:

1. *Scan.*  ``git ls-files`` gives the tracked corpus; a byte prefilter over
   the kind table's marker substrings drops the files that cannot hold a
   marker; the survivors are tokenized and only COMMENT tokens are read.
2. *Consumer model* (D8).  Which tool, if any, actually honours this marker.
3. *Classify.*  Owned by an inline disposition, ratified by class, or unowned.
4. *Ratchet.*  ``shared.ratchet`` does all the arithmetic and all the baseline
   I/O; this module supplies the keys and renders the violations.

**The exit-code ladder**, which is the PRD Contract's and is restated in the
argparse epilog:

* **0** — clean.  A scoped run says ``partial``, and a run with no baseline yet
  says ``advisory``; both are green, and the label says which green it is.
* **1** — violations, one per line: site, kind, codes, the reason, and the
  accepted forms.  A refusal to act is still 1.
* **2** — instrument failure.  A file that cannot be read or tokenized, a
  baseline that exists but cannot be compared against, a config key this model
  does not implement, or a missing import.  Never a finding.

**What it deliberately does NOT do.**

* *It commits no baseline.*  ``scripts/inline_suppression_baseline.json`` is
  seeded once, on main, by the operator step κ1 — not by this module's author
  and not by a test.  Until it exists every run is green and says ``advisory``.
* *It rules on nothing.*  ``RATIFIED_SUPPRESSION_CLASSES`` ships EMPTY; the
  rows are the operator's (δ) and are applied by ζb.
* *It parses no disposition grammar of its own.*  D6's forms live in
  ``shared.governed_exceptions`` and are reached only through
  ``parse_disposition_marker``.
* *It models no suppression kind outside the table below.*
  ``pytest.mark.skip`` / ``xfail`` and ``shellcheck disable`` are named in the
  PRD's out-of-scope list; :data:`KIND_SPECS` is the one place a later task
  adds them.

**Consumers.**  γ2's exception register reads ``--json`` for the closed-world
check of inline ``ratified:`` ids and for the per-``(kind, code)`` table it
renders; κ1 runs ``--seed`` once at the cutover; ζb runs ``--tighten`` after
the rulings land; θ's integration gate reads the report.  ``--json`` is the
report's only data source, so nothing downstream re-implements the scan.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from types import MappingProxyType


class InstrumentFailure(Exception):
    """The scanner could not take a measurement it was asked for — exit 2.

    THE NAME STATES THE CONSUMER-RELEVANT FACT RATHER THAN A CAUSE, the same
    choice ``shared.ratchet.BaselineUnusable`` argues for.  A file that cannot
    be read, a file that cannot be tokenized, a ``git ls-files`` that did not
    run, a config key this model does not implement and a missing ``shared``
    import all mean one identical thing to a caller — *this run measured
    nothing you can trust* — and all map to the same exit 2.  Splitting them by
    cause would hand ``main`` five ``except`` clauses that all do one thing.

    Categorically apart from a VIOLATION, which is exit 1.  That separation is
    the whole point: a broken instrument reported as a finding sends an agent
    to fix code that was never the problem, and a finding reported as a broken
    instrument is an INV-12 breach nobody is told about.  Every message names
    the file, path or key at fault, because that is the one thing the operator
    cannot derive from the rest of it.
    """


#: Generous, and not a performance assertion: `git ls-files` over this tree
#: takes well under a second, so anything approaching this has hung.
_GIT_TIMEOUT_SECS = 60


class Kind(Enum):
    """The suppression kinds this scanner models.

    The value is the label rendered in violations, in ``--json`` and in D9's
    class keys — one spelling, published from one place, so a report and a
    class row can never disagree about what a kind is called.
    """

    TYPE_IGNORE = 'type: ignore'
    NOQA = 'noqa'
    PYRIGHT_IGNORE = 'pyright: ignore'
    PRAGMA_NO_COVER = 'pragma: no cover'
    NOSEC = 'nosec'


@dataclass(frozen=True)
class Site:
    """One suppression: which kind, where, and the line it rides on.

    Attributes:
        path: Repo-relative path of the file holding it.
        line: 1-based physical line number.
        kind: Which row of :data:`KIND_SPECS` matched.
        codes: The rule codes the marker names, in SOURCE order.  Empty for a
            bare marker and for a kind that carries none.  Sorting is D7's
            key's job, not the scan's — a report that renders a site verbatim
            should show what the author wrote.
        text: The whole physical line, stripped.  This is what D7 digests, so
            editing anything on the line — the code, the disposition, a
            neighbouring argument — changes the key.  It is NOT the comment:
            see :class:`Comment`.
    """

    path: str
    line: int
    kind: Kind
    codes: tuple[str, ...]
    text: str


@dataclass(frozen=True)
class Comment:
    """One COMMENT token, and the suppressions it holds.

    THE SCAN'S UNIT IS THE COMMENT, NOT THE SITE, and that is load-bearing.
    D6 names two disposition violations, and the second — a marker on a line
    with NO suppression — is a property of a comment that produced zero Sites.
    Nothing downstream could reconstruct it from the Sites alone, because such
    a comment contributes none.

    Attributes:
        path: Repo-relative path of the file holding it.
        line: 1-based physical line number.
        text: The comment token verbatim, ``#`` included.  This is what
            ``shared.governed_exceptions.parse_disposition_marker`` reads.
        sites: One :class:`Site` per kind present, in :data:`KIND_SPECS` order.
    """

    path: str
    line: int
    text: str
    sites: tuple[Site, ...]


@dataclass(frozen=True)
class KindSpec:
    """How one :class:`Kind` is found, and how its codes are read.

    Attributes:
        marker: The substring that must be present for :attr:`pattern` to have
            any chance of matching, tested TWICE — against a file's raw bytes
            before it is decoded, and against a comment token before the
            pattern is run.  It must therefore appear in EVERY form
            :attr:`pattern` accepts, so it is the longest run of the pattern
            that no whitespace can interrupt: ``'type:'`` and not
            ``'type: ignore'``, because ``# type:ignore`` is a marker too.  The
            direction of error decides this — a substring that is too short
            costs only scan time, while one that is not invariant silently
            drops real sites.
        pattern: Searched against a whole COMMENT token.  Anchoring each form
            to its own ``#`` is what stops ``# see the noqa convention`` from
            registering, and matches how ruff and mypy read their own
            directives.
        codes: Reads the rule codes out of that match.
    """

    marker: str
    pattern: re.Pattern[str]
    codes: Callable[[re.Match[str]], tuple[str, ...]]


def _no_codes(match: re.Match[str]) -> tuple[str, ...]:
    """For the kinds that name no rule: ``pragma: no cover`` and ``nosec``."""
    del match
    return ()


def _bracketed_codes(match: re.Match[str]) -> tuple[str, ...]:
    """``[attr-defined, arg-type]`` — mypy's and pyright's shared spelling.

    The bracket must follow the directive with no space, which is what both
    tools require; ``# type: ignore [arg-type]`` is a bare ignore to mypy and
    is read as one here too.
    """
    inside = match.group(1)
    if inside is None:
        return ()
    return tuple(code.strip() for code in inside.split(',') if code.strip())


#: A ruff rule code: a linter's letters followed by its number.
_RUFF_CODE_RE = re.compile(r'[A-Za-z]+[0-9]+')

#: A first-party checker's code, e.g. ``bare-magicmock``.  Kebab-case carries
#: no numeric run at all, which is why the consumer model's ``(linter,
#: number)`` split has to tolerate one.
_KEBAB_CODE_RE = re.compile(r'[a-z][a-z0-9]*(?:-[a-z0-9]+)+')


def _noqa_codes(match: re.Match[str]) -> tuple[str, ...]:
    """``# noqa: E402,F401`` — read left to right, STOPPING at the first token
    that is not code-shaped.

    MEASURED, NOT ASSUMED (this tree, 2026-09-19).  ``noqa`` markers here carry
    232 distinct tails and the prose-bearing ones are ordinary: ``# noqa: F401
    — the binding IS the wiring``, ``# noqa: E402  (import after path fix)``.
    Splitting a whole tail on whitespace manufactures 'the', 'never', 'a' and
    'IS' as rule codes.  Stopping at the first non-code reproduces ruff's own
    reading and yields 40 distinct codes over the tree, every one of them real.

    KEBAB-CASE IS A CODE ONLY IN FIRST POSITION, which is the union of the two
    consumers' own grammars and nothing wider: ruff takes a
    ``<letters><digits>`` code anywhere in the list, while
    ``fused-memory/scripts/check_bare_magicmock_config.py`` anchors its
    kebab-case code immediately after ``noqa:``.  Admitting kebab-case anywhere
    would read ``# noqa: F401  re-export shim`` as naming a second code
    ``re-export`` (9 such sites measured), which is neither consumer's rule.
    """
    tail = match.group(1)
    if tail is None:
        return ()
    codes: list[str] = []
    for position, token in enumerate(token for token in re.split(r'[,\s]+', tail.strip()) if token):
        first = position == 0
        if _RUFF_CODE_RE.fullmatch(token) or (first and _KEBAB_CODE_RE.fullmatch(token)):
            codes.append(token)
        else:
            break
    return tuple(codes)


#: The optional ``[code, code]`` suffix mypy and pyright share.
_BRACKET_SUFFIX = r'(?:\[([^\]]*)\])?'

KIND_SPECS: MappingProxyType[Kind, KindSpec] = MappingProxyType(
    {
        Kind.TYPE_IGNORE: KindSpec(
            marker='type:',
            pattern=re.compile(r'#\s*type:\s*ignore' + _BRACKET_SUFFIX),
            codes=_bracketed_codes,
        ),
        Kind.NOQA: KindSpec(
            marker='noqa',
            # The tail stops at the next `#`, so a disposition riding in the
            # same comment can never be read as a rule code.
            pattern=re.compile(r'#\s*noqa(?::\s*([^#]*))?'),
            codes=_noqa_codes,
        ),
        Kind.PYRIGHT_IGNORE: KindSpec(
            marker='pyright:',
            pattern=re.compile(r'#\s*pyright:\s*ignore' + _BRACKET_SUFFIX),
            codes=_bracketed_codes,
        ),
        Kind.PRAGMA_NO_COVER: KindSpec(
            marker='pragma:',
            pattern=re.compile(r'#\s*pragma:\s*no\s+cover\b'),
            codes=_no_codes,
        ),
        Kind.NOSEC: KindSpec(
            # `\b` on both sides, because 'nanosecond' CONTAINS 'nosec' and
            # 'nosecret' starts with it.
            marker='nosec',
            pattern=re.compile(r'#\s*nosec\b'),
            codes=_no_codes,
        ),
    }
)
"""The one table the prefilter, the token scan and the consumer model all read.

SPOT (heuristic 11), and the PRD's named extension point: ``pytest.mark.skip`` /
``xfail`` and ``shellcheck disable`` are out of scope for this batch and are
added HERE when they come in scope, not as a second scanner.  A row supplies a
prefilter substring, a detection pattern and a code rule; the consumer model
maps the :class:`Kind` — not the row — to a consumer family, because that is a
question about tools rather than about syntax.
"""


def scan_source(source: str, *, path: str) -> tuple[Comment, ...]:
    """Every COMMENT token in *source*, each carrying the Sites it holds.

    TOKENIZE, NEVER REGEX OVER THE SOURCE TEXT.  A string literal that merely
    mentions ``#`` is not a comment, and no regex gets that right — the same
    argument ``scripts/merge_lane_metrics.py::_comment_lines`` makes, and it
    bites harder here: this repository's test modules are full of suppression
    markers quoted as data, including every fixture in this scanner's own
    suite.

    ``token.type`` rather than ``token.exact_type``: COMMENT has no exact-type
    refinement, and the coarse field is what the existing consumer uses.

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
    comment is the overwhelming majority (measured over this tree: 140,237
    comments hold 6,047 sites), and a substring test is far cheaper than the
    five pattern searches it stands in front of.  One table, two granularities
    — not a second policy.
    """
    line = token.start[0]
    text = token.string
    stripped = token.line.strip()
    sites = tuple(
        Site(path=path, line=line, kind=kind, codes=spec.codes(match), text=stripped)
        for kind, spec in KIND_SPECS.items()
        if spec.marker in text and (match := spec.pattern.search(text)) is not None
    )
    return Comment(path=path, line=line, text=text, sites=sites)


@dataclass(frozen=True)
class Scan:
    """One sweep of a tree: what it found, and how much work it did.

    Attributes:
        comments: Every COMMENT token in every file that survived the
            prefilter, in enumeration order.
        files_enumerated: How many tracked paths ``git ls-files`` listed,
            INCLUDING any whose worktree file has since been deleted.  It is
            the size of the corpus the scan set out to read.
        files_tokenized: How many of those were actually decoded and tokenized
            — the ones whose raw bytes carried a marker substring.

    THE TWO COUNTS ARE THE BUDGET'S ENFORCEMENT POINT.  The PRD gives this scan
    ten seconds, and the prefilter is the only reason it fits; asserting the
    CLOCK on the box the merge gate runs on would plant a flake (measured: ~5
    CPU-seconds of work took 6.5-6.8 s of wall clock at load 105), so the guard
    test asserts these two numbers instead.  They pin the mechanism the budget
    rests on — this is a prefiltered scan, not a whole-tree tokenize — and
    cannot flake.
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


#: Every kind's prefilter substring as raw bytes, for the pre-decode pass.
#: Derived from the one table rather than written twice; the markers are ASCII,
#: so the encoding is exact.
_MARKER_BYTES: tuple[bytes, ...] = tuple(
    spec.marker.encode('utf-8') for spec in KIND_SPECS.values()
)


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
    are never decoded and never tokenized.  Measured over this tree, that is
    1,212 of 1,933 files skipped.

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

    if not any(marker in raw for marker in _MARKER_BYTES):
        return None

    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError as exc:
        raise InstrumentFailure(
            f'{relative}: could not be decoded as UTF-8 -- {exc}. A tracked Python file '
            'this scanner cannot decode is never skipped: it might hold the very '
            'suppression the gate exists to find.'
        ) from exc


def scan_tree(root: Path) -> Scan:
    """Scan every tracked ``*.py`` under *root*.

    The prefilter means ``files_tokenized`` is strictly smaller than
    ``files_enumerated`` on any real tree, and a file it rejects is reported
    honestly as enumerated-not-tokenized rather than silently vanishing.
    """
    tracked = _tracked_python_files(root)
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


# ---------------------------------------------------------------------------
# D7 — the multiset key.

#: How many hex characters of the sha256 the key keeps.  Twelve is the PRD's,
#: and it is part of the ratchet's ``params`` block: changing it makes two
#: baselines not two measurements of the same thing, which is exactly what
#: ``shared.ratchet.ParamsMismatch`` exists to refuse.
_DIGEST_HEX = 12


@dataclass(frozen=True)
class SuppressionKey:
    """D7's multiset key: ``(kind, sorted codes, digest of the stripped line)``.

    AND DELIBERATELY NO PATH.  That single omission is what buys the ratchet
    its two best properties: a file rename or split moves every marker without
    inventing a single new key, and an ordinary task therefore never has to
    touch the baseline.  Its cost is stated in D7 rather than hidden — two
    identical lines share one key, so an un-tightened baseline lets an
    identical line back in where one was removed, which is what ``slack``
    measures and what D11's sweep files a tighten task for.

    Attributes:
        kind: Which suppression this is.
        codes: The rule codes, SORTED at construction — ``[b, a]`` and
            ``[a, b]`` are the same suppression, and a multiset that
            disagreed would ratchet on the author's typing order.
        digest: ``sha256`` of the stripped physical line, truncated.  Digesting
            the LINE rather than the comment is what makes an edit anywhere on
            it a touch, which is D7's whole conversion mechanism.
    """

    kind: Kind
    codes: tuple[str, ...]
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'codes', tuple(sorted(self.codes)))

    def render(self) -> str:
        """The key as one string, for JSON and for a violation line.

        A RENDERING EXISTS; AN INVERSE DOES NOT, and that asymmetry is the
        point.  JSON object keys are strings by the format's definition, so
        something has to render — ``shared.ratchet``'s docstring makes the same
        argument for the same reason.  What must not exist is a reader that
        parses one back: ``shared.ratchet`` treats every key as an opaque
        identity token, so a parser would be the ad-hoc parser of an internal
        value heuristic 12 forbids, and it would silently promote this spelling
        to a wire format nobody could ever change.

        The codes group is omitted entirely when there are none, because
        ``pragma: no cover[]`` reads as a missing code rather than as a kind
        that never has one.  The digest is separated by a SPACE, never by
        ``@`` — ``@`` is :meth:`SuppressionClass.render`'s separator, and two
        key spellings a reader could confuse is the one thing worth spending a
        character to avoid.
        """
        codes = f'[{",".join(self.codes)}]' if self.codes else ''
        return f'{self.kind.value}{codes} {self.digest}'


def key_for(site: Site) -> SuppressionKey:
    """The D7 key *site* contributes to the multiset."""
    return SuppressionKey(
        kind=site.kind,
        codes=site.codes,
        digest=hashlib.sha256(site.text.encode('utf-8')).hexdigest()[:_DIGEST_HEX],
    )
