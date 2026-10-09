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
3. *Classify.*  Consumer first, then ratified class, then inline disposition
   — THAT fixed order is what makes a marker no tool reads un-rescuable by
   either of the other two; a site none of them owns is unowned.
4. *Ratchet.*  ``shared.ratchet`` does all the arithmetic and all the baseline
   I/O; this module supplies the keys and renders the violations.

**The key** (D7).  An unowned site contributes ``(kind, sorted codes, a 12-hex
sha256 of its stripped physical line)`` to a multiset — and no path.  A rename
or a file split therefore invents no key and needs no baseline diff, while
editing the line a marker rides on does invent one, which is how an ordinary
edit converts a grandfathered suppression into one somebody has to own.

**The exit-code ladder**, which is the PRD Contract's and is restated in the
argparse epilog:

* **0** — clean.  A scoped run says ``partial``, and a run with no baseline yet
  says ``advisory``; both are green, and the label says which green it is.
* **1** — violations, one per line: site, kind, codes, the reason, and the
  accepted forms.
* **2** — instrument failure or a refusal to act.  A file that cannot be read
  or tokenized, a baseline that exists but cannot be compared against, a config
  key this model does not implement, a missing import, a scope that matches no
  tracked file, a scoped ``--seed`` or ``--tighten``, or ``--seed`` over an
  existing baseline.  Never a finding.

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

**Why this is one file, measured rather than asserted.**  ~2,050 raw lines, of
which ~760 are code and ~46% is prose — and ``docs/code-quality.md`` says in
terms not to steer by raw line count, citing a 55%-prose precedent.  Heuristic
14's 2,000-line alarm is nevertheless crossed on the figure it names (one
default ``Read``), so the alternatives were measured rather than waved off: the
docstrings carry no redundancy to cut (the largest is this one at ~60 lines and
the rest are flat, with each load-bearing phrase appearing once), and a split
is illegitimate because all four layers read :data:`KIND_SPECS` and pass
:class:`Site` — satellites over that shared state would be the function-bags
heuristic 14 names as cheating, failing 13 to pass 14.  So: one file, alarm
acknowledged, and this paragraph is the record that it was weighed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import re
import subprocess
import sys
import tokenize
import tomllib
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from io import StringIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared.governed_exceptions import Debt, Disposition, Policy
    from shared.ratchet import Enumeration

# The shared/src bootstrap, resolved from __file__ and inserted at sys.path[0] —
# the idiom and the precedence argument of
# scripts/scan_plan_decision_pairing.py. A run inside a task worktree must read
# THAT checkout's disposition grammar and ratchet kernel, not whichever editable
# install happens to be on the path; the editable install is an ordinary .pth
# entry, so sys.path ORDER decides the winner. It sits BELOW every import in this
# file rather than above them, which is the whole reason no import here needs a
# suppression for E402: every `shared` name is fetched lazily through
# :func:`_shared`, so there is no module-level import left to sit after this
# statement.
_SHARED_SRC = Path(__file__).resolve().parents[1] / 'shared' / 'src'
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))


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

def _shared(name: str) -> ModuleType:
    """Import ``shared.<name>`` lazily, as exit 2 rather than exit 1.

    THE ONLY PLACE THIS MODULE IMPORTS ``shared``, and the reason is the PRD
    Contract: an ImportError must be 2, never 1.  A top-level ``from shared… import
    …`` cannot satisfy that — it raises while this module is still executing, so
    ``main`` is never defined, the ``__main__`` block never runs, and Python's own
    uncaught-exception exit is 1, the exact code the ladder reserves for a FINDING.
    A gate reporting a broken environment as an INV-12 breach sends an agent to
    fix code that was never the problem.

    Returning the module rather than the names is what keeps the conversion in one
    place, following ``scripts/merge_lane_metrics.py::_import_complexipy``.  The
    cost is measured and bounded: attributes come back as ``Any``, so a caller that
    needs ``isinstance`` NARROWING binds the class through a ``type[X]``
    annotation first (verified: ``isinstance(x, module.Debt)`` does not narrow,
    ``debt_type: type[Debt] = module.Debt`` then ``isinstance(x, debt_type)``
    does).  A caller that only needs the runtime class — an ``except`` clause, a
    branch predicate — uses the attribute directly.
    """
    try:
        return importlib.import_module(f'shared.{name}')
    except ImportError as exc:
        raise InstrumentFailure(
            f'`shared.{name}` could not be imported, so this scanner has no disposition '
            f'grammar and no ratchet kernel to work with -- {exc}. It is a workspace '
            'member of this repository: run the scanner as `uv run --project shared '
            'python scripts/inline_suppressions.py`, which is how the merge gate and '
            'every declared check invoke it.'
        ) from exc


#: This checkout, resolved from ``__file__`` rather than from the working
#: directory, so a run inside a task worktree measures THAT worktree's tracked
#: corpus and reads THAT worktree's configuration.  The argument
#: ``scripts/scan_plan_decision_pairing.py`` makes for the same resolution.
_REPO_ROOT = Path(__file__).resolve().parents[1]

#: How this scanner names itself in a refusal, so a reader of a bare exit 2 in
#: a merge log knows which instrument spoke.
_PROG = 'inline_suppressions.py'


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
            directives.  Compiled with ``re.IGNORECASE`` exactly when the
            consuming tool reads the directive case-insensitively; that flag is
            the one statement of the kind's case rule, and both marker tests
            follow it through :attr:`folds_case`.
        codes: Reads the rule codes out of that match.
    """

    marker: str
    pattern: re.Pattern[str]
    codes: Callable[[re.Match[str]], tuple[str, ...]]

    def __post_init__(self) -> None:
        if self.folds_case and self.marker != self.marker.lower():
            raise ValueError(
                f'case-folding kind marker {self.marker!r} must be lower-case, '
                'because it is tested against lower-cased text'
            )

    @cached_property
    def folds_case(self) -> bool:
        """Whether the directive is read case-insensitively, from the pattern's flags."""
        return bool(self.pattern.flags & re.IGNORECASE)

    def may_match(self, text: str) -> bool:
        """The cheap substring test that stands in front of :attr:`pattern`."""
        return self.marker in (text.lower() if self.folds_case else text)


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
            # Case-insensitive because ruff reads the directive in any case.
            pattern=re.compile(r'#\s*noqa(?::\s*([^#]*))?', re.IGNORECASE),
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
"""The one table the prefilter and the token scan read.

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

    THE PHYSICAL LINE IS STRIPPED ONLY FOR A COMMENT THAT MATCHED, which is the
    same economy one step further in.  The strip feeds nothing but a
    :class:`Site`, so over this tree the ~134,000 marker-free comments would
    otherwise each allocate a string that is discarded on the next line — paid
    inside the one function the ten-second budget rests on.  The marker test
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
    mistake in the same shape as the selector-matching one decision 2 records.
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


# ---------------------------------------------------------------------------
# D8 — the consumer model: which tool, if any, actually reads a marker.


class Consumer(Enum):
    """Who honours a suppression.

    :attr:`NONE` is the finding D8 exists to surface, not an absence of
    information: a marker nothing reads is not protecting anything, so it is
    deleted rather than dispositioned.
    """

    PYRIGHT = 'pyright'
    RUFF = 'ruff'
    FIRST_PARTY = 'first-party'
    NONE = 'none'


#: Which consumer a kind resolves to WITHOUT consulting the code or any
#: config.  ``noqa`` is deliberately absent — it is the one kind whose answer
#: depends on both.  Totality over :class:`Kind` is enforced at the one read
#: site rather than assumed, so a sixth row added to :data:`KIND_SPECS` with no
#: consumer row is a loud exit 2 instead of silently inheriting ``noqa``'s
#: behaviour.
_KIND_CONSUMERS: Mapping[Kind, Consumer] = MappingProxyType(
    {
        # pyright runs over every package as a declared gate, so both of its
        # marker spellings are read wherever they sit.
        Kind.TYPE_IGNORE: Consumer.PYRIGHT,
        Kind.PYRIGHT_IGNORE: Consumer.PYRIGHT,
        # Nothing in this repository reads either one today: no coverage gate
        # is configured, and bandit is not installed — which is why the live
        # `nosec` count is zero.  Both rows are statements about TOOLS, and
        # they change when a tool is adopted, not when code changes.  (Neither
        # marker is spelled out here with its leading `#`: inside a comment
        # that would BE a marker, and this scanner would read its own prose.)
        Kind.PRAGMA_NO_COVER: Consumer.NONE,
        Kind.NOSEC: Consumer.NONE,
    }
)

_LEADING_LETTERS = re.compile(r'^[A-Za-z]+')
_TRAILING_DIGITS = re.compile(r'[0-9]+$')


@dataclass(frozen=True)
class RuleCode:
    """A ruff rule code or selector, split at the boundary into its two parts.

    Attributes:
        linter: The leading run of letters — ``E``, ``BLE``, ``PLC``.  This is
            the linter ruff resolves the selector against.
        number: The trailing run of digits, which may be empty for a bare
            linter selector like ``B`` and for a first-party kebab-case code
            like ``bare-magicmock``.  An empty number is tolerated rather than
            raising: a code this split cannot read is simply one no selector
            can match, which is the correct answer for it.
    """

    linter: str
    number: str

    def render(self) -> str:
        """The code as the model READ it, for ``--json``.

        Deliberately the parsed form rather than the source string: the report
        exists so a reader can audit what the consumer model used, and a
        selector the model mis-read is exactly what they are looking for.
        """
        return f'{self.linter}{self.number}'


def parse_rule_code(raw: str) -> RuleCode:
    """Split *raw* into its leading letters and trailing digits."""
    linter = _LEADING_LETTERS.match(raw)
    number = _TRAILING_DIGITS.search(raw)
    return RuleCode(
        linter=linter.group() if linter is not None else '',
        number=number.group() if number is not None else '',
    )


def selects(selector: RuleCode, code: RuleCode) -> bool:
    """Whether *selector* selects *code*, ruff's way and never by string prefix.

    MEASURED AGAINST REAL RUFF, not assumed: ``ruff check --isolated --select
    B`` does not flag ``BLE001`` while ``--select BLE`` does, and within one
    linter ``--select E4`` flags E402 only while ``--select E5`` flags E501
    only.  A selector is therefore a (linter, code-prefix) PAIR: the linter
    must be EQUAL — ``B`` (flake8-bugbear) never reaches ``BLE``
    (flake8-blind-except) — and the number is a prefix within it.

    A naive ``code.startswith(selector)`` would report this tree's 202
    ``# noqa: BLE001`` markers as ruff-consumed, defeating D8 for its
    second-largest population.

    KNOWN LIMIT, recorded rather than hidden: ruff's meta-prefix selectors
    (``PL`` covering PLC/PLE/PLR/PLW) read as unselected under this rule.  No
    ``pyproject.toml`` in this repository uses one, and ``--json`` publishes
    the resolved select lists so a reader can see exactly what the model used.
    """
    return selector.linter == code.linter and code.number.startswith(selector.number)


@dataclass(frozen=True)
class RuffConfig:
    """The ``select``/``ignore`` pair one ``pyproject.toml`` declares.

    Attributes:
        path: The declaring file, repo-relative, so ``--json`` can name it.
        select: The selectors, parsed.
        ignore: The suppressors, parsed.  Read because it is the same tomllib
            call and is strictly more honest: all eight pyprojects here set
            ``ignore = ["E501"]``, so ruff provably never emits E501 and every
            ``# noqa: E501`` in the tree is dead — which is exactly what D8
            exists to say.
    """

    path: str
    select: tuple[RuleCode, ...]
    ignore: tuple[RuleCode, ...]

    def consumes(self, code: RuleCode) -> bool:
        """Whether ruff would emit *code* under this config."""
        return any(selects(selector, code) for selector in self.select) and not any(
            selects(suppressor, code) for suppressor in self.ignore
        )


#: Keys this model does not implement that could WIDEN the selected set.  The
#: split is by DIRECTION OF ERROR, which is the only thing that matters for a
#: gate: under-reading the selected set rejects a marker ruff genuinely
#: honours — a false red on a legitimate suppression, the expensive failure —
#: so the scanner refuses to guess.  Keys that only ever SUBTRACT
#: (`extend-ignore`, `per-file-ignores`) are tolerated and deliberately not
#: modelled, because over-reading only grandfathers a dead marker.
#:
#: `extend` is in the list because config INHERITANCE widens in exactly that
#: expensive direction: the inherited file carries its own `select` /
#: `extend-select`, which this model does not follow, so the local list read
#: here would be a subset of the rules ruff actually runs.  Nothing in this
#: tree uses it today; completeness on the expensive side is the whole purpose
#: of the list, so it is not left to be discovered by the first pyproject that
#: does.
#:
#: An ABSENT `select` belongs in this family for the same reason and is
#: enforced with it: ruff then applies its BUILT-IN default rule set, which is
#: wider than the nothing this model would otherwise infer and which drifts
#: with the ruff version.  All eight pyprojects here declare `select`
#: explicitly, so nothing in this tree reaches that refusal.
_WIDENING_KEYS: tuple[str, ...] = ('extend-select', 'extend')


def _ruff_config_at(location: Path, *, relative: str) -> RuffConfig | None:
    """Read *location*'s ruff config, or ``None`` when it declares none.

    ``None`` covers two cases ruff itself treats identically: the file is not
    there, and the file carries no ``[tool.ruff]`` section at all.  Ruff SKIPS
    such a manifest and keeps looking upward, so a packaging-only
    ``pyproject.toml`` must not shadow the config above it — stated in this
    repository's own root ``pyproject.toml``.

    ``select`` AND ``ignore`` ARE RESOLVED INDEPENDENTLY, each from the last
    table that declares it, because that is what ruff does: a deprecated
    top-level lint key is honoured unless ``[tool.ruff.lint]`` declares that
    same key.  Taking both off the one table that happened to declare
    ``select`` would drop an ``ignore`` sitting beside it, and an ignored code
    is precisely the dead marker :class:`RuffConfig` reads ``ignore`` to name.
    """
    try:
        raw = tomllib.loads(location.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise InstrumentFailure(
            f'{relative}: could not be read as TOML -- {type(exc).__name__}: {exc}'
        ) from exc

    tool = raw.get('tool')
    ruff = tool.get('ruff') if isinstance(tool, dict) else None
    if not isinstance(ruff, dict):
        return None

    lint = ruff.get('lint')
    tables = [('tool.ruff', ruff)]
    if isinstance(lint, dict):
        tables.append(('tool.ruff.lint', lint))
    for name, table in tables:
        for key in _WIDENING_KEYS:
            if key in table:
                raise InstrumentFailure(
                    f'{relative}: [{name}] carries {key!r}, which this consumer '
                    'model does not implement and which could WIDEN the selected rule '
                    'set. Refusing to guess: under-reading the selected set would reject '
                    'a marker ruff genuinely honours.'
                )

    def declared(key: str) -> tuple[RuleCode, ...] | None:
        """*key*'s parsed list from the last table declaring it, else ``None``."""
        for _, table in reversed(tables):
            if key in table:
                return tuple(parse_rule_code(code) for code in table[key])
        return None

    select = declared('select')
    if select is None:
        raise InstrumentFailure(
            f'{relative}: has a [tool.ruff] section but declares no `select`, so ruff '
            "applies its BUILT-IN default rule set. This consumer model does not "
            'implement those defaults — they are wider than the nothing it would '
            'otherwise infer, and they drift with the ruff version. Declare `select` '
            'explicitly.'
        )

    return RuffConfig(path=relative, select=select, ignore=declared('ignore') or ())


#: The rule codes first-party checkers define, each mapped to the checker that
#: defines it.  These are not ruff codes at all — no selector can ever match a
#: kebab-case token — so they are resolved AHEAD of the ruff path.
#:
#: A SECOND COPY OF A PRIVATE CONSTANT, knowingly.  The source of truth is
#: `_RULE_A_CODE` / `_RULE_B_CODE` / `_RULE_C_CODE` in a stdlib-only script in
#: another package, which offers no public seam to import; a copy is
#: unavoidable, so heuristic 11 asks that its DRIFT be made loud rather than
#: that the copy be hidden.  The guard is behavioural and bidirectional:
#: `scripts/tests/test_inline_suppression_ratchet.py` runs that checker over one
#: fixture per rule and asserts SET EQUALITY between the codes it emits and this
#: table.  Equality, not containment — a table that merely holds real codes can
#: still MISS one, which is the expensive direction (the scanner would tell an
#: author to delete a marker a live checker reads).
#:
#: Rule C's `wall-clock-deadline` is named in neither the PRD's D8 prose nor
#: this task's plan, both of which list the first two; it is here because the
#: checker honours it and its violation message tells authors to write it
#: (measured, esc-5601-1).
FIRST_PARTY_CODES: Mapping[str, str] = MappingProxyType(
    {
        'bare-magicmock': 'fused-memory/scripts/check_bare_magicmock_config.py',
        'bare-dataclass-double': 'fused-memory/scripts/check_bare_magicmock_config.py',
        'wall-clock-deadline': 'fused-memory/scripts/check_bare_magicmock_config.py',
    }
)


class ConsumerModel:
    """Resolves each site's consumer, reading the nearest ``pyproject.toml``.

    STATEFUL FOR ONE REASON ONLY — the memo.  Resolution walks from a site's
    directory up to the scan root, and a tree of 6,000 sites in 800 files would
    otherwise re-read and re-parse the same eight manifests thousands of times.
    The memo is keyed by DIRECTORY rather than by file, so one read serves every
    file in a package, and it is private to one scan.
    """

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._by_directory: dict[Path, RuffConfig | None] = {}

    def consumer_for(self, site: Site) -> Consumer:
        """Who honours *site*'s marker — :attr:`Consumer.NONE` if nobody does."""
        fixed = _KIND_CONSUMERS.get(site.kind)
        if fixed is not None:
            return fixed
        if site.kind is not Kind.NOQA:
            raise InstrumentFailure(
                f'{site.path}:{site.line}: kind {site.kind.value!r} is in KIND_SPECS but '
                'has no row in the consumer model, so this scanner cannot say whether '
                'anything reads it. Add its row beside the others.'
            )
        return self._noqa_consumer(site)

    def _noqa_consumer(self, site: Site) -> Consumer:
        """Resolve a ``noqa`` site against the nearest config.

        THE FIRST-PARTY TABLE IS CONSULTED FIRST, because those codes are not
        ruff codes at all and no config could ever answer for them.  A site is
        consumed if ANY of its codes is — the direction that over-reads, which
        decision 3 establishes is the cheap failure (it grandfathers a dead
        marker) where under-reading would falsely reject a live one.

        A CODELESS marker silences whatever ruff would have said, so it is
        consumed exactly when ruff has something to say at all — and a config
        selecting nothing therefore leaves it dead.

        EVERY MARKER THIS MODULE MENTIONS IS MENTIONED IN A DOCSTRING, never in
        a ``#`` comment, and that is a rule rather than a habit.  A ``#``
        comment quoting one is a real marker: ruff parses it and warns
        ``Invalid # noqa directive``, and this very scanner reads it as a site
        in its own corpus.  A docstring is a STRING, which is precisely what
        the token walk exists to tell apart.
        """
        if any(code in FIRST_PARTY_CODES for code in site.codes):
            return Consumer.FIRST_PARTY
        config = self._nearest_config(Path(site.path).parent)
        if config is None:
            return Consumer.NONE
        if not site.codes:
            return Consumer.RUFF if config.select else Consumer.NONE
        if any(config.consumes(parse_rule_code(code)) for code in site.codes):
            return Consumer.RUFF
        return Consumer.NONE

    def resolved(self) -> tuple[RuffConfig, ...]:
        """Every config this model actually consulted, sorted by declaring path.

        THE REPORT'S AUDIT TRAIL, and the reason it is the model that publishes
        it: what a reader needs is not every ``pyproject.toml`` in the tree but
        the ones whose ``select``/``ignore`` decided an answer here — including,
        by their absence, the meta-prefix selectors decision 2 records as a known
        limit.  Deduped by path, because the memo is keyed by directory and one
        manifest serves a whole subtree.
        """
        return tuple(
            sorted(
                {config.path: config for config in self._by_directory.values() if config}.values(),
                key=lambda config: config.path,
            )
        )

    def _nearest_config(self, directory: Path) -> RuffConfig | None:
        """The config governing *directory*, which is relative to the scan root.

        The walk stops AT the root and never climbs out of the tree under
        measurement; otherwise a scan of a fixture tree would silently read
        this repository's own configuration and report a stranger's answer.
        """
        if directory in self._by_directory:
            return self._by_directory[directory]
        relative = str(directory / 'pyproject.toml') if str(directory) != '.' else 'pyproject.toml'
        config = _ruff_config_at(self._root / relative, relative=relative)
        if config is None and str(directory) != '.':
            config = self._nearest_config(directory.parent)
        self._by_directory[directory] = config
        return config


# ---------------------------------------------------------------------------
# D9 — ratified classes, the pressure valve the OPERATOR holds.


class Scope(Enum):
    """Where a ratified class applies.

    ``SRC`` is defined as "not tests" rather than as its own predicate, so the
    two are a PARTITION and no path can fall outside both.  Two independent
    predicates could each miss a path, and a class row that silently covered
    nothing is the failure nobody would notice.
    """

    SRC = 'src'
    TESTS = 'tests'
    ANY = 'any'

    def covers(self, path: str) -> bool:
        """Whether *path* is in this scope.

        A path is ``tests`` iff one of its COMPONENTS is ``tests`` — not a
        filename pattern.  Verified complete for this repository: every tracked
        test module lives under a ``tests`` component, so no ``test_*.py`` rule
        is needed beside this one.
        """
        if self is Scope.ANY:
            return True
        in_tests = 'tests' in Path(path).parts
        return in_tests if self is Scope.TESTS else not in_tests


@dataclass(frozen=True)
class SuppressionClass:
    """D9's key: one kind, one code, one scope.

    The table below is keyed by this TYPE rather than by its rendering, so the
    operator's rows are type-checked at import and a malformed row cannot
    masquerade as a class nobody happens to match.  :meth:`render` exists only
    because the report publishes the key as a string.

    Attributes:
        kind: The suppression kind the row covers.
        code: The single rule code, or ``''`` for the codeless form — which
            matches a BARE marker rather than matching everything, because a
            row covering every code of a kind is a far bigger valve than D9
            describes and should be written out if it is ever wanted.
        scope: Where it applies.
    """

    kind: Kind
    code: str
    scope: Scope

    def render(self) -> str:
        """D9's published spelling, ``kind[code]@scope``."""
        return f'{self.kind.value}[{self.code}]@{self.scope.value}'

    def covers(self, site: Site) -> bool:
        """Whether this row ratifies *site*."""
        matches_code = self.code in site.codes if self.code else not site.codes
        return self.kind is site.kind and matches_code and self.scope.covers(site.path)


RATIFIED_SUPPRESSION_CLASSES: Mapping[SuppressionClass, Policy] = MappingProxyType({})
"""Blanket rulings, keyed by class — SHIPPED EMPTY, and that is D9's design.

The valve is the OPERATOR's: δ rules the rows against the inflow table so the
valve is sized against the filing rate each row prevents, and ζb applies them.
An implementer adding a row here would be ratifying a blanket exception on the
operator's behalf, which is the one thing D9 reserves.

Rows added AFTER the baseline is seeded turn their sites' baseline keys into
slack, which the next ``--tighten`` removes; removing a row makes its sites
new.  Both directions are ordinary ratchet arithmetic, which is why no row
needs a migration.
"""


# ---------------------------------------------------------------------------
# Layer 3 — classification.


class Ownership(Enum):
    """How a site is accounted for.

    :attr:`UNOWNED` is the only value that contributes a key to the multiset;
    the other three are the three ways of being answered for.
    """

    DEBT = 'debt'
    POLICY = 'policy'
    CLASS = 'class'
    UNOWNED = 'unowned'


@dataclass(frozen=True)
class Classified:
    """One site, and everything the pipeline decided about it."""

    site: Site
    consumer: Consumer
    ownership: Ownership
    disposition: Disposition | None = None
    suppression_class: SuppressionClass | None = None


@dataclass(frozen=True)
class Violation:
    """One exit-1 finding, rendered as one line.

    Attributes:
        path: The file, repo-relative.
        line: 1-based physical line.
        kind: The suppression kind, or ``None`` when the finding is about a
            comment that carries no suppression at all.
        codes: The rule codes, for the reader who needs to find the marker.
        reason: What is wrong, in one clause.
        forms: The accepted disposition forms to publish, rendered from
            ``shared.governed_exceptions.INLINE_MARKER_FORMS`` rather than
            retyped.  EMPTY when the finding accepts no disposition — a
            consumer-less marker is deleted, not dispositioned, so offering
            forms there would be advice that does not work.
    """

    path: str
    line: int
    kind: Kind | None
    codes: tuple[str, ...]
    reason: str
    forms: tuple[str, ...] = ()

    def render(self) -> str:
        """One line: where, what, why, and how to fix it."""
        codes = f'[{",".join(self.codes)}]' if self.codes else ''
        marker = f'{self.kind.value}{codes}' if self.kind is not None else 'no suppression'
        accepted = f' Accepted forms: {" | ".join(self.forms)}' if self.forms else ''
        return f'{self.path}:{self.line}: {marker} -- {self.reason}{accepted}'

    def record(self) -> dict[str, object]:
        """The same finding as structured data, for ``--json``.

        Not the rendered line re-parsed, and not a second statement of what a
        finding IS: both come off the same fields, so a consumer reads values
        where a human reads a sentence and neither has to parse the other
        (heuristic 12).
        """
        return {
            'path': self.path,
            'line': self.line,
            'kind': None if self.kind is None else self.kind.value,
            'codes': list(self.codes),
            'reason': self.reason,
            'forms': list(self.forms),
        }


@dataclass(frozen=True)
class Classification:
    """The whole tree, classified.

    Attributes:
        classified: One entry per site, in scan order.
        violations: The exit-1 findings that do not depend on the baseline —
            the two disposition faults D6 names.  Ratchet violations are the
            ``--check`` verb's and are computed against the baseline.
        unowned: Rendered key to the entries that produced it, sorted by key.
            The COUNTS are derived from this rather than tracked beside it
            (heuristic 11): ``--check`` needs the entries behind an excess key
            in order to name their lines and to say why each one is a finding,
            and a count kept separately could drift from the list.
    """

    classified: tuple[Classified, ...]
    violations: tuple[Violation, ...]
    unowned: Mapping[str, tuple[Classified, ...]]

    @property
    def counts(self) -> dict[str, int]:
        """The multiset :class:`shared.ratchet.Enumeration` compares."""
        return {key: len(entries) for key, entries in self.unowned.items()}


def classify(scan: Scan, model: ConsumerModel) -> Classification:
    """Decide every site's ownership, and collect D6's two disposition faults.

    THE ORDER IS FIXED — consumer, then ratified class, then inline
    disposition — and it is the single mechanism that makes D8's "accepts no
    disposition" true.  A site nothing consumes is unowned whatever is written
    beside it, so boundary scenario 9 ("a disposition does not rescue it")
    falls out of the pipeline instead of needing a special case beside the
    disposition check.  It simultaneously satisfies D8's "grandfathered dead
    markers stay counted": such a site is in the multiset, so a baseline that
    already holds it yields no excess and the gate is green, while the report
    still counts it.

    The class table is second rather than first only because D8's prohibition
    is categorical whereas D9's valve is the operator's.  Since the table ships
    EMPTY no site is affected by that relative order today; recording it now is
    what stops a later reader flipping it by accident.

    A MALFORMED MARKER IS A VIOLATION AND LEAVES ITS SITES UNOWNED.  Both are
    true at once and neither substitutes for the other: the broken marker is a
    fault at the site whatever the baseline says, and the sites it failed to
    disposition are genuinely undisposed.

    THE PREFILTER BOUNDS THE SUPPRESSION-FREE-DISPOSITION FINDING, and the
    bound is stated here rather than left to be found.  This layer sees only
    the comments of files whose raw bytes carried a KIND marker, so a stray
    ``# debt: …`` alone in a file with no suppressions anywhere is never read.
    That limit falls on the harmless side: with nothing silenced in the file,
    the marker answers for nothing and misleads nobody about a live detector.
    The dangerous case — an author who believes a REAL marker in this file is
    now dispositioned when it is not — is precisely the one that IS caught,
    because such a file carries a marker by definition.  Widening the prefilter
    to the disposition keywords would put a second copy of D6's grammar in this
    module, which is the one thing ``shared.governed_exceptions`` exists to
    prevent.
    """
    governed = _shared('governed_exceptions')

    classified: list[Classified] = []
    violations: list[Violation] = []
    unowned: dict[str, list[Classified]] = {}

    for comment in scan.comments:
        try:
            disposition = governed.parse_disposition_marker(comment.text)
        except governed.MalformedDisposition as exc:
            disposition = None
            violations.append(
                Violation(
                    path=comment.path,
                    line=comment.line,
                    kind=None,
                    codes=(),
                    reason=(
                        f'the disposition marker in {comment.text!r} does not parse, so '
                        f'nothing here is dispositioned ({exc.__class__.__name__}).'
                    ),
                    forms=governed.INLINE_MARKER_FORMS,
                )
            )
        if disposition is not None and not comment.sites:
            violations.append(
                Violation(
                    path=comment.path,
                    line=comment.line,
                    kind=None,
                    codes=(),
                    reason=(
                        f'{comment.text!r} carries a disposition but the line holds no '
                        'suppression for it to answer for. A disposition names why a '
                        'silenced detector stays silent; with nothing silenced it says '
                        'nothing, and it will not be followed up. Remove it, or put it '
                        'on the line that carries the marker.'
                    ),
                )
            )

        for site in comment.sites:
            entry = _classify_site(site, disposition, model)
            classified.append(entry)
            if entry.ownership is Ownership.UNOWNED:
                unowned.setdefault(key_for(site).render(), []).append(entry)

    return Classification(
        classified=tuple(classified),
        violations=tuple(violations),
        unowned=MappingProxyType(
            {key: tuple(entries) for key, entries in sorted(unowned.items())}
        ),
    )


def _classify_site(
    site: Site, disposition: Disposition | None, model: ConsumerModel
) -> Classified:
    """One site, through the fixed order :func:`classify` documents."""
    consumer = model.consumer_for(site)
    if consumer is Consumer.NONE:
        return Classified(site=site, consumer=consumer, ownership=Ownership.UNOWNED)

    row = next(
        (row for row in RATIFIED_SUPPRESSION_CLASSES if row.covers(site)),
        None,
    )
    if row is not None:
        return Classified(
            site=site,
            consumer=consumer,
            ownership=Ownership.CLASS,
            suppression_class=row,
        )

    if disposition is None:
        return Classified(site=site, consumer=consumer, ownership=Ownership.UNOWNED)

    return Classified(
        site=site,
        consumer=consumer,
        ownership=(
            Ownership.DEBT
            if isinstance(disposition, _shared('governed_exceptions').Debt)
            else Ownership.POLICY
        ),
        disposition=disposition,
    )


# ---------------------------------------------------------------------------
# Layer 4 — the ratchet: the baseline, the verbs and the exit ladder.

#: Where the committed baseline lives, relative to the repository root.  Named
#: here and nowhere else, so ``--baseline`` has a default that cannot drift from
#: the path κ1 seeds and the merge gate reads.
BASELINE_PATH = 'scripts/inline_suppression_baseline.json'

#: The ``--json`` report's own schema version, and deliberately NOT
#: ``shared.ratchet.SCHEMA_VERSION`` — that one versions the BASELINE FILE's
#: format, which changes for entirely different reasons.  Two orthogonal schemas
#: get two numbers, so a report-shape change never reads as a baseline-format
#: change to the consumer that sees it.
REPORT_SCHEMA_VERSION = 1

#: The key scheme, as one opaque token in the ratchet's ``params`` block.  It is
#: a NAME rather than a description: params are compared for equality, so its
#: only job is to differ when the keys mean something different.
_KEY_SCHEME = 'kind+codes+sha256-of-stripped-line'


def _params() -> dict[str, object]:
    """This scan's measurement parameters, for the baseline's ``params`` block.

    ONLY WHAT CHANGES THE MEANING OF THE MULTISET.  A params mismatch is exit 2
    and the only way out of it is re-seeding, which is itself the widening move
    D7 built the verbs to prevent — so a field belongs here exactly when a
    change to it makes two baselines two measurements of DIFFERENT things.  The
    scanned kinds and the key scheme qualify: add a kind, or change the hash, and
    every count is about something else.

    The resolved ruff ``select``/``ignore`` lists deliberately do NOT qualify,
    and that omission is the decision worth recording.  Putting them here would
    convert an ordinary, reviewable ``pyproject.toml`` edit into a forced
    baseline regeneration — punishing a legitimate config change by demanding
    the one operation nobody should perform casually.  They are published in
    ``--json`` instead, so the consumer model stays auditable without arming
    that tripwire.
    """
    return {
        'kinds': [kind.value for kind in Kind],
        'key_scheme': _KEY_SCHEME,
        'digest_hex': _DIGEST_HEX,
    }


class Status(Enum):
    """Which green a zero exit is — three states that are not interchangeable.

    :attr:`CLEAN` is the only one that means the WHOLE tree was measured against
    a real baseline and nothing was in excess.  The other two are green with a
    stated limit, and saying which limit applies is the point: a reader who sees
    a bare zero over a tree of undisposed markers concludes the scanner is
    broken, and a reader who sees no label at all concludes the gate is live when
    it is enforcing nothing.

    Red has no label here, and deliberately so: a run with violations reports
    how many, because a status word beside a finding would read as a verdict on
    the tree rather than on the run's reach.
    """

    CLEAN = 'clean'
    PARTIAL = 'partial'
    ADVISORY = 'advisory'


@dataclass(frozen=True)
class Request:
    """One invocation's resolved inputs, shared by every verb.

    Attributes:
        root: The checkout under measurement.
        baseline: The baseline file this run compares against or writes.
        scope: The positional ``PATH`` arguments, empty for a whole-tree run.
    """

    root: Path
    baseline: Path
    scope: tuple[str, ...] = ()

    @property
    def scoped(self) -> bool:
        """Whether this run measured only part of the tree."""
        return bool(self.scope)


def _status(request: Request) -> Status:
    """Which green *request*'s zero would be.

    ABSENCE IS DECIDED HERE, BY AN EXPLICIT EXISTENCE CHECK, and never by
    catching ``shared.ratchet.BaselineUnusable``.  That kernel refusal collapses
    absent, undecodable, unparseable, misshapen and wrong-schema into one case
    because they mean one thing to its callers; this consumer is the one place
    where they do not.  D12 makes absence a legitimate pre-κ1 state, while a
    baseline that exists and cannot be read is a broken instrument.  Reaching the
    advisory path by catching the refusal would report a corrupt or truncated
    baseline as a clean tree, which is the silent fail-soft an empty baseline
    causes (INV-11).

    ABSENCE OUTRANKS SCOPE.  A scoped run with no baseline is enforcing nothing
    at all, which is the stronger of the two limits and therefore the one worth
    the label.
    """
    if not request.baseline.exists():
        return Status.ADVISORY
    return Status.PARTIAL if request.scoped else Status.CLEAN


@dataclass(frozen=True)
class Measurement:
    """One run's scan, its classification, and the model that resolved it.

    The model travels with the other two because ``--json`` publishes the
    configs it consulted; a report that re-resolved them could publish a
    different answer than the one the classification used.
    """

    scan: Scan
    classification: Classification
    model: ConsumerModel


def _measure(request: Request) -> Measurement:
    """Scan and classify *request*'s tree — the work every verb starts with."""
    scan = scan_tree(request.root, scope=request.scope)
    model = ConsumerModel(request.root)
    return Measurement(scan=scan, classification=classify(scan, model), model=model)


def _enumeration(classification: Classification, kernel: ModuleType) -> Enumeration:
    """The multiset the kernel compares, under this scan's params.

    ``complete=True`` is asserted rather than computed because this scanner has
    no partial-success mode to report: every unreadable file has already been
    raised as an :class:`InstrumentFailure`, so a scan that returns at all read
    everything it enumerated.  The kernel's ``unreadable`` list is therefore
    always empty here, and the field that would carry names has none to carry.
    """
    return kernel.Enumeration(counts=classification.counts, params=_params(), complete=True)


#: Why an undisposed suppression is a finding, in one clause.  The forms are
#: rendered beside it from the published grammar, never retyped.
_UNDISPOSED_REASON = (
    'this suppression is not in the baseline and names no owner. INV-12 requires every '
    'silenced detector to carry a disposition -- who will remove it, or the operator '
    'ruling that keeps it'
)

#: D8's rejection arm, in the same clause shape.  It names no remedy but
#: deletion, because there is no other one that works.
_DEAD_REASON = (
    'delete this marker -- no tool reads it, so it silences nothing and protects '
    'nothing. A disposition does not answer for it either: with no detector silenced '
    'there is nothing for an owner to own or for the operator to rule on'
)


def _violation_for(entry: Classified) -> Violation:
    """The exit-1 finding one excess *entry* renders.

    THE TWO REASONS ARE NOT INTERCHANGEABLE ADVICE.  An undisposed live
    suppression is fixed by writing a disposition, so the line publishes the
    accepted forms; a marker nothing consumes is fixed only by deleting it, so
    the line publishes none.  Handing an author forms that would not clear the
    finding is worse than handing them nothing: they would write one, rerun,
    and see the identical red with no idea why.

    Which of the two applies is read off the entry's consumer rather than
    recomputed, because the classification already resolved it — in the fixed
    order that made the site unowned in the first place.
    """
    dead = entry.consumer is Consumer.NONE
    return Violation(
        path=entry.site.path,
        line=entry.site.line,
        kind=entry.site.kind,
        codes=entry.site.codes,
        reason=_DEAD_REASON if dead else _UNDISPOSED_REASON,
        forms=() if dead else _shared('governed_exceptions').INLINE_MARKER_FORMS,
    )


def _excess_violations(
    over: Mapping[str, int], classification: Classification
) -> tuple[Violation, ...]:
    """One violation per SITE of every key the baseline does not cover.

    EVERY SITE, NOT ``over[key]`` OF THEM, and the reason is structural: a key
    IS the content of its line, so two sites sharing one are indistinguishable
    by construction and the scanner cannot say which is the new one.  Naming
    them all is the only honest rendering — the alternative picks an arbitrary
    line and sends the reader to code that may have been there for a year.

    "ABSENT FROM THE BASELINE" IS MADE PRECISE AS "IN EXCESS", and that one
    substitution is what folds D8's rejection arm INTO the ratchet instead of
    standing it beside one.  A dead marker is not a separate rule with its own
    exemption list: it is an ordinary member of the multiset, so the baseline
    grandfathers the existing population, a second copy of a grandfathered line
    is over budget, and removing one shows up as slack — all without a second
    mechanism to keep in step with this one.

    Sorted by key so a rerun over one tree prints the same lines in the same
    order; within a key, scan order, which is path then line.
    """
    return tuple(
        _violation_for(entry) for key in sorted(over) for entry in classification.unowned[key]
    )


@dataclass(frozen=True)
class Verdict:
    """What the gate would say about this run.

    Attributes:
        status: Which green a zero exit would be.
        excess: Keys over budget, or ``None`` when no baseline was compared.
        slack: Headroom, or ``None`` when this run's view cannot honestly
            measure it — an advisory run has no baseline, and a scoped one would
            read every unscanned key as headroom.
        violations: Every exit-1 finding, disposition faults first.

    COMPUTED ONCE AND READ TWICE.  ``--check`` turns it into an exit code and
    ``--json`` publishes it, so a report reader sees the gate's answer without
    running the gate — and cannot see a DIFFERENT answer, which is what two
    scans of a tree that changed in between would give them.
    """

    status: Status
    excess: Mapping[str, int] | None
    slack: Mapping[str, int] | None
    violations: tuple[Violation, ...]

    @property
    def exit_code(self) -> int:
        """1 when this run found something, else 0."""
        return 1 if self.violations else 0


def _verdict(
    request: Request, classification: Classification, kernel: ModuleType
) -> Verdict:
    """What the gate would say about *classification*.

    The disposition faults come first and are baseline-INDEPENDENT: a marker that
    does not parse is a fault at the site whatever any baseline holds, so an
    advisory run still reports one.  Only the ratchet's half waits for a baseline.
    """
    status = _status(request)
    if status is Status.ADVISORY:
        return Verdict(
            status=status, excess=None, slack=None, violations=classification.violations
        )

    baseline = kernel.load(request.baseline)
    current = _enumeration(classification, kernel)
    over = dict(kernel.excess(current, baseline))
    return Verdict(
        status=status,
        excess=over,
        slack=None if request.scoped else dict(kernel.slack(current, baseline)),
        violations=classification.violations + _excess_violations(over, classification),
    )


def _total(counts: Mapping[str, int] | None) -> int | None:
    """A ratchet block's multiplicity total, preserving "not measured"."""
    return None if counts is None else sum(counts.values())


def _count(total: int, noun: str) -> str:
    """*total* and *noun*, pluralised by the only rule this report needs."""
    return f'{total} {noun}' if total == 1 else f'{total} {noun}s'


def _figure(total: int | None) -> str:
    """A ratchet figure, or ``n/a`` when this run's view cannot honestly give one.

    THE DISTINCTION IS NOT COSMETIC.  Zero means *measured, and nothing there*;
    ``n/a`` means *not measured*, which is what an advisory run's excess and a
    scoped run's slack both are.  Printing 0 for either would be a number that is
    wrong in the direction that reassures.
    """
    return 'n/a' if total is None else str(total)


def _headline(status: Status, violations: tuple[Violation, ...]) -> str:
    """The report's first words: which green this is, or how big the red is."""
    if not violations:
        return status.value
    return _count(len(violations), 'violation')


def _report_line(
    measured: Measurement,
    *,
    headline: str,
    excess_total: int | None,
    slack_total: int | None,
) -> str:
    """The one-line summary every verb prints to stdout.

    The counted-work figures lead because they are what a reader checks first
    when a run comes back suspiciously clean: a scan that enumerated nothing
    and a tree with nothing wrong are otherwise the same output.
    """
    counts = measured.classification.counts
    scan = measured.scan
    return (
        f'{headline}: {_count(len(scan.sites), "suppression site")} in '
        f'{scan.files_tokenized} of {scan.files_enumerated} tracked files; '
        f'{sum(counts.values())} unowned in {_count(len(counts), "key")}; '
        f'excess {_figure(excess_total)}, slack {_figure(slack_total)}'
    )


def _check(request: Request, kernel: ModuleType) -> int:
    """Compare *request*'s tree against its baseline — the merge gate's verb.

    The violations are the two independent kinds added together: the disposition
    faults D6 names, which are faults at the site whatever any baseline says,
    and the ratchet's excess.  Either alone is exit 1.
    """
    measured = _measure(request)
    verdict = _verdict(request, measured.classification, kernel)
    print(
        _report_line(
            measured,
            headline=_headline(verdict.status, verdict.violations),
            excess_total=_total(verdict.excess),
            slack_total=_total(verdict.slack),
        )
    )
    if verdict.status is Status.ADVISORY:
        print(_ADVISORY_NOTICE.format(baseline=request.baseline))
    for violation in verdict.violations:
        print(violation.render(), file=sys.stderr)
    return verdict.exit_code


def _require_whole_tree(request: Request, verb: str) -> None:
    """Refuse a scoped run of a baseline-WRITING verb, before any scan work.

    THE KERNEL'S ARITHMETIC CANNOT REFUSE THIS ONE, which is why the refusal
    lives here.  A scoped scan is honestly ``complete=True`` for its scope, so it
    passes ``_require_comparable`` and the tighten goes through — writing
    ``baseline ∩ scope`` and deleting every key outside it.  The kernel's own
    docstring names regenerating a baseline from a partial view as the single hole
    its arithmetic leaves; this is the consumer closing it.

    BEFORE THE SCAN, not after, for two reasons that both matter.  A refusal is
    not worth minutes of tokenizing, and — the load-bearing one — a tree that
    also holds an unreadable file would otherwise report THAT as the exit 2,
    sending the reader to fix a file when the invocation was the problem.

    A scoped ``--check`` is deliberately not refused: ``Counter`` subtraction is
    per-key and saturating, so a narrower current view can only lower a key's
    count and therefore only UNDER-report violations.  It can never manufacture
    one, which is what the ``partial`` label is telling the reader.
    """
    if request.scoped:
        raise InstrumentFailure(
            f'refusing a scoped {verb}: {" ".join(request.scope)} is part of a tree, and a '
            'baseline written from a partial view drops every key outside it -- which makes '
            'each of those suppressions a fresh violation on the next whole-tree run. Re-run '
            'without PATH arguments. A scoped --check is the supported early-feedback run.'
        )


def _seed(request: Request, kernel: ModuleType) -> int:
    """Write *request*'s tree as a fresh baseline — κ1's one-time verb.

    BOTH REFUSALS PRECEDE ``dump``, and they have to: ``dump`` writes the
    enumeration it is handed and checks nothing about the file already at the
    path, because seeding a new baseline and carrying an honestly incomplete one
    across a file boundary are both legitimate uses of it.  Re-seeding an
    EXISTING baseline is therefore the one call in the kernel that widens the
    gate — by every key the current scan added — so the guard belongs to whoever
    calls it.

    The status is :attr:`Status.CLEAN` by construction rather than by
    measurement: a baseline written from this very scan has no excess over it,
    and the run is whole-tree because a scoped seed is refused.  Both ratchet
    figures are ``n/a``, because nothing was compared — printing 0 for an excess
    that was never computed would be the reassuring-direction error
    :func:`_figure` exists to refuse.
    """
    _require_whole_tree(request, '--seed')
    if request.baseline.exists():
        raise InstrumentFailure(
            f'refusing to seed over the baseline already at {request.baseline}: a baseline is '
            "seeded ONCE, as the operator's one-time cutover on main, and tightened "
            'thereafter. Re-seeding an existing one silently widens the gate by every key '
            'this scan added. Use --tighten to remove what the tree no longer needs.'
        )
    measured = _measure(request)
    kernel.dump(_enumeration(measured.classification, kernel), request.baseline)
    print(
        _report_line(
            measured,
            headline=f'{Status.CLEAN.value} -- seeded {request.baseline}',
            excess_total=None,
            slack_total=None,
        )
    )
    return 0


#: What a run with no baseline tells the reader, so the next question — "then
#: why is this green?" — is answered in the same output.
#:
#: IT SCOPES ITS CLAIM TO THE RATCHET, because the run it accompanies may not be
#: green at all: a disposition fault is a fault at the site whatever any baseline
#: holds, so an advisory run still reports one and still exits 1.
_ADVISORY_NOTICE = (
    'no baseline at {baseline}, so the RATCHET is not enforced yet: every suppression '
    'here is reported and none of them counts as excess. Disposition faults are '
    'reported regardless, because a marker that does not parse is a fault whatever a '
    'baseline holds. The baseline is seeded once, on main, with --seed'
)


def _tighten(request: Request, kernel: ModuleType) -> int:
    """Spend the baseline's headroom — ζb's verb, run after the rulings land.

    THROUGH ``tighten_into``, NEVER ``tighten`` PLUS ``dump``.  The kernel makes
    the composition the SHORT way to write the safe call for a reason: persisting
    ``tighten``'s bare ``Counter`` by hand means re-supplying ``params``,
    ``complete`` and ``unreadable``, three chances to commit a baseline claiming
    more than the scan measured — sitting next to ``dump(current, path)``, a
    shorter line that widens the gate by every key this scan added.

    THE REMOVALS COME FROM THE RETURN VALUE, not from reading the file back.  The
    loaded pre-image and the tightened counter are both in hand, so the difference
    is exact; re-reading would report what the file says rather than what this run
    did, which are the same thing only when nothing went wrong.

    Excess is measured BEFORE the write and is not stale: tightening lowers a
    baseline key only where the current count is lower, which is precisely where
    excess is already zero, so ``excess`` is invariant under it.  Slack afterwards
    is zero by the same arithmetic — a pointwise minimum is elementwise ≤ the
    current scan — so the figure is measured, not assumed.

    It reports NO status word, deliberately.  ``--tighten`` does not gate: it
    returns 0 having done its maintenance even when the tree is still over
    budget, and a green label beside a non-zero excess would read as a verdict it
    never made.  The excess figure in the same line is what tells the reader the
    gate is still red.
    """
    _require_whole_tree(request, '--tighten')
    measured = _measure(request)
    baseline = kernel.load(request.baseline)
    current = _enumeration(measured.classification, kernel)
    excess_total = sum(kernel.excess(current, baseline).values())
    tightened = kernel.tighten_into(current, baseline, request.baseline)

    removed = {
        key: count - tightened.get(key, 0)
        for key, count in baseline.counts.items()
        if tightened.get(key, 0) < count
    }
    print(
        _report_line(
            measured,
            headline=f'tightened {request.baseline}',
            excess_total=excess_total,
            slack_total=0,
        )
    )
    for key, count in sorted(removed.items()):
        print(f'  -{count} {key}')
    return 0


def _site_record(site: Site) -> dict[str, object]:
    """One site, as every block of the report names it."""
    return {
        'path': site.path,
        'line': site.line,
        'kind': site.kind.value,
        'codes': list(site.codes),
    }


def _owner(debt: Debt) -> str:
    """A debt's owner as D6 spells it — ``task 5601`` or ``ticket tkt_…``.

    The same tail the inline marker carries, so a report entry and the line it
    came from read alike and a consumer following the owner up needs no second
    vocabulary.  The two nouns come from the two ref TYPES rather than from a
    string test, which is why a ticket can never be reported as a task.
    """
    noun = (
        'task'
        if isinstance(debt.owner, _shared('governed_exceptions').TaskRef)
        else 'ticket'
    )
    return f'{noun} {debt.owner.id}'


def _tally(values: Iterable[str], vocabulary: Iterable[str]) -> dict[str, int]:
    """Count *values*, with every word of *vocabulary* present even at zero.

    THE ZERO ROWS ARE THE POINT.  ``nosec`` reading 0 is a measurement — this
    repository configures no bandit — and a schema that omitted it would make "no
    bandit markers in the tree" and "the scanner stopped looking for them" the
    same output.  Every kind, consumer and ownership state therefore appears in
    every report, which is also what lets the live guard assert a zero.
    """
    counted = Counter(values)
    return {word: counted.get(word, 0) for word in vocabulary}


def _by_kind_code(scan: Scan) -> list[dict[str, object]]:
    """The per-``(kind, code)`` table the exception register renders.

    A site with two codes contributes to BOTH rows, because the question this
    answers is "how many suppressions silence E402", and a marker silencing E402
    and F401 silences each of them.  The totals therefore need not sum to the
    site count, which is why they are published beside ``kind_totals`` rather
    than instead of it.

    A codeless marker gets ``None`` rather than an empty string: ``""`` is a
    value a reader could filter on and silently match nothing.
    """
    tally: Counter[tuple[str, str | None]] = Counter()
    for site in scan.sites:
        for code in site.codes or (None,):
            tally[(site.kind.value, code)] += 1
    return [
        {'kind': kind, 'code': code, 'sites': sites}
        for (kind, code), sites in sorted(
            tally.items(), key=lambda row: (row[0][0], row[0][1] or '')
        )
    ]


def _owned_blocks(classification: Classification) -> dict[str, object]:
    """The three site-level ownership blocks, built in one pass.

    GROUPED BY WHAT EACH ENTRY CARRIES rather than by its :class:`Ownership`
    label, and the two agree by construction: :func:`_classify_site` sets exactly
    one of ``disposition`` and ``suppression_class``, and sets NEITHER on an
    unowned site — including one whose comment did carry a disposition that D8
    refused.  Keying off the field removes the unreachable branch a label-first
    grouping would need, and the report then says what is actually written down.

    THESE ARE THE ONLY SITE-LEVEL BLOCKS, deliberately.  Each is bounded by what
    somebody wrote down, while the corpus itself is thousands of sites and is
    published as counts; a report enumerating every one would be a megabyte
    nobody reads and a second copy of the tree.  ``ratified`` carries its sites
    because the register's closed-world check cannot report an id it has no
    sites to point at, and ``classes`` carries its own so a blanket ruling cannot
    quietly absorb a growing population.
    """
    governed = _shared('governed_exceptions')
    debt_type: type[Debt] = governed.Debt
    policy_type: type[Policy] = governed.Policy

    debt: list[dict[str, object]] = []
    ratified: dict[str, list[dict[str, object]]] = {}
    classes: dict[str, list[dict[str, object]]] = {}
    for entry in classification.classified:
        record = _site_record(entry.site)
        if isinstance(entry.disposition, debt_type):
            debt.append(record | {'owner': _owner(entry.disposition)})
        elif isinstance(entry.disposition, policy_type):
            ratified.setdefault(entry.disposition.ratified, []).append(record)
        if entry.suppression_class is not None:
            classes.setdefault(entry.suppression_class.render(), []).append(record)
    return {
        'debt': debt,
        'ratified': [{'id': key, 'sites': sites} for key, sites in sorted(ratified.items())],
        'classes': [{'class': key, 'sites': sites} for key, sites in sorted(classes.items())],
    }


def _report_dict(request: Request, measured: Measurement, verdict: Verdict) -> dict[str, object]:
    """The whole report, as plain JSON-expressible data.

    ONE BUILDER, so ``--json`` is the only place the report exists and nothing
    downstream re-implements the scan.  Every mapping is dumped with sorted keys
    and every list is built in a sorted order — the site-level ones in scan
    order, which is sorted path then ascending line — so two runs over one
    unchanged tree emit identical BYTES.  A report that reordered between runs
    would make its every consumer's diff pure noise.
    """
    classification = measured.classification
    return {
        'schema_version': REPORT_SCHEMA_VERSION,
        'status': verdict.status.value,
        'params': _params(),
        'files_enumerated': measured.scan.files_enumerated,
        'files_tokenized': measured.scan.files_tokenized,
        'sites': len(measured.scan.sites),
        'kind_totals': _tally(
            (site.kind.value for site in measured.scan.sites),
            (kind.value for kind in Kind),
        ),
        'consumers': _tally(
            (entry.consumer.value for entry in classification.classified),
            (consumer.value for consumer in Consumer),
        ),
        'ownership': _tally(
            (entry.ownership.value for entry in classification.classified),
            (ownership.value for ownership in Ownership),
        ),
        'by_kind_code': _by_kind_code(measured.scan),
        **_owned_blocks(classification),
        'baseline': {
            'path': str(request.baseline),
            'present': verdict.status is not Status.ADVISORY,
            'excess': None if verdict.excess is None else dict(verdict.excess),
            'slack': None if verdict.slack is None else dict(verdict.slack),
        },
        'ruff_config': [
            {
                'pyproject': config.path,
                'select': [code.render() for code in config.select],
                'ignore': [code.render() for code in config.ignore],
            }
            for config in measured.model.resolved()
        ],
        'violations': [violation.record() for violation in verdict.violations],
    }


def _json(request: Request, kernel: ModuleType) -> int:
    """Publish the report on stdout — a REPORT verb, never a gate.

    IT EXITS 0 WHENEVER THE SCAN SUCCEEDED, including over a tree that is
    currently in breach, and that is the contract rather than an oversight: a
    consumer parsing this report must not also be gated by it, and the state the
    exception register most needs to read is precisely the one ``--check``
    refuses.  Only a broken instrument makes it 2, because then there is no
    report to publish.
    """
    measured = _measure(request)
    verdict = _verdict(request, measured.classification, kernel)
    print(json.dumps(_report_dict(request, measured, verdict), indent=2, sort_keys=True))
    return 0


_EPILOG = """exit codes:
  0  clean. A scoped run says `partial`; a run with no baseline yet says
     `advisory`. All three are green, and the label says which green it is.
  1  violations, one per line on stderr: site, kind, codes, the reason and the
     accepted disposition forms.
  2  instrument failure or a refusal to act -- a file that could not be read
     or tokenized, a baseline that exists but cannot be compared against, a
     config key this model does not implement, a missing import, a scope that
     matches no tracked file, a scoped --seed or --tighten, or --seed over an
     existing baseline. Never a finding.
"""


def _build_parser() -> argparse.ArgumentParser:
    """The CLI: one verb, a tree, a baseline, and an optional scope."""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "INV-12's inline-suppression scanner and multiset ratchet: every tracked "
            'suppression is owned by a disposition, ratified by class, or held by the '
            'baseline, and the population can shrink but never grow.'
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verbs = parser.add_mutually_exclusive_group()
    verbs.add_argument(
        '--check',
        action='store_true',
        help='compare the tree against the baseline (the default verb)',
    )
    verbs.add_argument(
        '--seed',
        action='store_true',
        help='write the tree as a fresh baseline; refuses if one already exists',
    )
    verbs.add_argument(
        '--tighten',
        action='store_true',
        help='remove the baseline headroom the tree no longer uses',
    )
    verbs.add_argument(
        '--json',
        action='store_true',
        help=(
            'publish the whole report on stdout and exit 0 unless the instrument broke. '
            'Mutually exclusive with the other verbs: --json --seed has two defensible '
            'readings and no way to say which was meant'
        ),
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=_REPO_ROOT,
        help='the checkout to measure (default: the one holding this script)',
    )
    parser.add_argument(
        '--baseline',
        type=Path,
        default=None,
        help=f'the baseline file (default: <root>/{BASELINE_PATH})',
    )
    parser.add_argument(
        'paths',
        nargs='*',
        metavar='PATH',
        help=(
            'limit the scan to these files or directories. A scoped --check is early '
            'feedback and labels its green `partial`; the baseline-writing verbs refuse '
            'a scope outright'
        ),
    )
    return parser


def _refuse(exc: Exception) -> int:
    """Print *exc* as this scanner's exit-2 refusal and return 2.

    ONE PRINT SITE FOR EVERY BROKEN-INSTRUMENT MESSAGE, so the prefix a log
    reader greps for cannot differ between causes — which is the same argument
    :class:`InstrumentFailure` makes for there being one exception type.
    """
    print(f'{_PROG}: {exc}', file=sys.stderr)
    return 2


def _run(args: argparse.Namespace, kernel: ModuleType) -> int:
    """Perform the verb *args* selected, as the exit code it returns.

    The kernel's whole error family is converted to this module's own
    instrument failure HERE, at the one place the kernel is reachable, because
    to a caller they mean the identical thing: nothing was compared.  That is
    the single ``except`` clause ``shared.ratchet``'s docstring says a consumer
    wants, spent once rather than at every call site.
    """
    root = Path(args.root)
    baseline = Path(args.baseline) if args.baseline is not None else root / BASELINE_PATH
    request = Request(root=root, baseline=baseline, scope=tuple(args.paths))
    try:
        if args.seed:
            return _seed(request, kernel)
        if args.tighten:
            return _tighten(request, kernel)
        if args.json:
            return _json(request, kernel)
        return _check(request, kernel)
    except kernel.RatchetError as exc:
        raise InstrumentFailure(
            f'the ratchet kernel refused this run -- {exc}'
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """The 0/1/2 entry point, with every broken-instrument path landing on 2."""
    args = _build_parser().parse_args(argv)
    try:
        return _run(args, _shared('ratchet'))
    except InstrumentFailure as exc:
        return _refuse(exc)


if __name__ == '__main__':
    sys.exit(main())
