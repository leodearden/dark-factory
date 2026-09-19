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

import re
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from types import MappingProxyType


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
        marker: The byte substring the prefilter tests a file's raw bytes for.
            It must appear in EVERY form :attr:`pattern` accepts, so it is the
            longest run of :attr:`pattern` that no whitespace can interrupt —
            ``b'type:'`` and not ``b'type: ignore'``, because ``# type:ignore``
            is a marker too.  The direction of error decides this: a substring
            that is too short costs only scan time, while one that is not
            invariant silently drops real sites.
        pattern: Searched against a whole COMMENT token.  Anchoring each form
            to its own ``#`` is what stops ``# see the noqa convention`` from
            registering, and matches how ruff and mypy read their own
            directives.
        codes: Reads the rule codes out of that match.
    """

    marker: bytes
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
            marker=b'type:',
            pattern=re.compile(r'#\s*type:\s*ignore' + _BRACKET_SUFFIX),
            codes=_bracketed_codes,
        ),
        Kind.NOQA: KindSpec(
            marker=b'noqa',
            # The tail stops at the next `#`, so a disposition riding in the
            # same comment can never be read as a rule code.
            pattern=re.compile(r'#\s*noqa(?::\s*([^#]*))?'),
            codes=_noqa_codes,
        ),
        Kind.PYRIGHT_IGNORE: KindSpec(
            marker=b'pyright:',
            pattern=re.compile(r'#\s*pyright:\s*ignore' + _BRACKET_SUFFIX),
            codes=_bracketed_codes,
        ),
        Kind.PRAGMA_NO_COVER: KindSpec(
            marker=b'pragma:',
            pattern=re.compile(r'#\s*pragma:\s*no\s+cover\b'),
            codes=_no_codes,
        ),
        Kind.NOSEC: KindSpec(
            # `\b` on both sides, because 'nanosecond' CONTAINS 'nosec' and
            # 'nosecret' starts with it.
            marker=b'nosec',
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
    """
    comments: list[Comment] = []
    for token in tokenize.generate_tokens(StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        comments.append(_comment_at(token, path=path))
    return tuple(comments)


def _comment_at(token: tokenize.TokenInfo, *, path: str) -> Comment:
    """Build the :class:`Comment` for one COMMENT *token*."""
    line = token.start[0]
    text = token.string
    stripped = token.line.strip()
    sites = tuple(
        Site(path=path, line=line, kind=kind, codes=spec.codes(match), text=stripped)
        for kind, spec in KIND_SPECS.items()
        if (match := spec.pattern.search(text)) is not None
    )
    return Comment(path=path, line=line, text=text, sites=sites)
