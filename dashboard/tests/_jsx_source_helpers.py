"""Source-text probes over the dashboard's served JS, JSX and HTML, for the suites asserting on it.

An underscore-prefixed sibling module, importable through conftest.py's sys.path insert
exactly as `_dashboard_helpers.py` is.  Its contract lives in test_jsx_source_helpers.py.
"""

from __future__ import annotations

import html.parser
import re
from collections.abc import Callable, Sequence
from typing import NamedTuple

# ---------------------------------------------------------------------------
# JSX source slicing.
#
# These helpers serve the Python suites that assert structural contracts
# against the *served* .jsx text.  Nearly every such assertion must first
# scope itself to one function's body — otherwise a token appearing anywhere
# else in the file satisfies it and the test proves nothing.
#
# These helpers used to be private copies in nine test modules (task 3549),
# under two names covering FOUR distinct implementations, so a fix had to be
# applied nine times or not at all.  Their contract lives in
# test_jsx_source_helpers.py.
#
# There are three of them now.  `find_function_params` is the paren-depth walk
# that locates a declaration's parameter list; `extract_function_body` resumes
# from where it stops to take the body, and test_charts_axis_labels.py's
# `_extract_signature` takes the params themselves.  Those two return DISJOINT,
# adjacent slices of the same declaration, so neither can be built on the
# other — but the walk that finds the boundary between them is one walk, and it
# lives here rather than in each of them.
#
# All three are built on ONE quote-aware scanner, `_scan_js`.  Giving them a
# scanner each would re-create in miniature exactly the duplication this
# consolidation removed — and they need the same answer to the same question:
# which stretches of this text are NOT code?
# ---------------------------------------------------------------------------


class _JsSpan(NamedTuple):
    """One stretch of JS source that is not code: a string literal or a comment.

    ``end`` is exclusive.  ``opener`` is the quote character for a string span
    and ``'//'`` / ``'/*'`` for a comment.  ``closed`` is False when the source
    ran out before the span was terminated.
    """

    start: int
    end: int
    kind: str  # 'string' | 'comment'
    opener: str
    closed: bool


def _string_literal_end(source: str, quote_at: int) -> tuple[int, bool]:
    """Return ``(end, closed)`` for the literal whose opening quote is at *quote_at*.

    ``end`` is exclusive and clamped to ``len(source)``; ``closed`` is False
    when the source ran out first.  A backslash skips the next character, so
    an escaped quote cannot close the literal.
    """
    quote = source[quote_at]
    i, n = quote_at + 1, len(source)
    while i < n:
        if source[i] == '\\':
            i += 2
            continue
        if source[i] == quote:
            return i + 1, True
        i += 1
    return n, False


def _scan_js(source: str) -> list[_JsSpan]:
    """Return the string-literal and comment spans of *source*, in order.

    Deliberately not a full JS lexer: a regex literal containing ``//`` or a
    quote (``str.replace(/'/g, '')``), or a quote nested inside a template's
    ``${...}``, would confuse it.  Neither occurs in the assets these helpers
    are used on, and `_assert_js_lexable` turns the failure mode that WOULD
    reach them — an apostrophe in JSX prose — into a loud error.
    """
    spans: list[_JsSpan] = []
    i, n = 0, len(source)

    while i < n:
        ch = source[i]

        if ch in '\'"`':
            end, closed = _string_literal_end(source, i)
            spans.append(_JsSpan(i, end, 'string', ch, closed))
            i = end
            continue

        if source[i : i + 2] == '//':
            end = source.find('\n', i)
            end = n if end == -1 else end
            spans.append(_JsSpan(i, end, 'comment', '//', True))
            i = end
            continue

        if source[i : i + 2] == '/*':
            end = source.find('*/', i + 2)
            spans.append(_JsSpan(i, n if end == -1 else end + 2, 'comment', '/*', end != -1))
            i = n if end == -1 else end + 2
            continue

        i += 1

    return spans


def _assert_js_lexable(source: str, spans: Sequence[_JsSpan]) -> None:
    """Raise if the scan hit a state that means it almost certainly misparsed.

    THE FAILURE THIS EXISTS FOR is an apostrophe in JSX *text* — ``don't`` in a
    label — which the scanner reads as an opening quote.  Everything up to the
    next ``'`` anywhere later in the file is then treated as string contents,
    so real comments inside it are left unstripped (prose leaks into what the
    consumers call "code") and real braces inside it are not counted (a body is
    mis-scoped).  Both are silent: the caller gets a plausible string back.

    Two states betray it, and neither can occur in well-formed source that this
    scanner actually understands:

    * a literal still open at end-of-input;
    * a non-template literal spanning a newline — JS forbids a raw newline in a
      ``'``/``"`` literal, so this is a misparse, not a long string.

    A template literal (backtick) legitimately spans lines and is exempt; there
    is one in tabs.jsx.
    """
    def _line(index: int) -> int:
        return source.count('\n', 0, index) + 1

    _APOSTROPHE_HINT = (
        'The likeliest cause is an apostrophe in JSX text (a label reading '
        '`don\'t`), which this scanner reads as an opening quote — everything '
        'after it is then treated as string contents, so comments inside it are '
        'left unstripped and braces inside it are not counted, and the caller '
        'gets a plausible-looking wrong answer. Reword the prose, or write the '
        'apostrophe as `&apos;`. (A regex literal containing a quote, e.g. '
        '`/\'/`, would also do it — see `_scan_js`.)'
    )

    for span in spans:
        if not span.closed:
            what = 'string literal' if span.kind == 'string' else 'block comment'
            raise AssertionError(
                f'Cannot scan this JS source: a {span.opener} {what} opened at '
                f'line {_line(span.start)} is never closed. {_APOSTROPHE_HINT}'
            )
        if span.kind == 'string' and span.opener != '`' and '\n' in source[span.start : span.end]:
            raise AssertionError(
                f'Cannot scan this JS source: a {span.opener} literal opened at '
                f'line {_line(span.start)} spans a newline (it appears to close at '
                f'line {_line(span.end - 1)}). A raw newline is illegal inside a '
                f'{span.opener} literal in JS, so this is a misparse rather than a '
                f'long string. {_APOSTROPHE_HINT}'
            )


def _mask_js(source: str, spans: Sequence[_JsSpan]) -> str:
    """Return *source* with every non-code span blanked, LENGTH PRESERVED.

    Equal length is the whole point: the caller searches and walks the mask but
    slices the ORIGINAL with the indices it finds, so the returned text is the
    real source rather than a blanked copy.  Newlines survive so a line number
    computed from the mask still means something.
    """
    chars = list(source)
    for span in spans:
        for k in range(span.start, span.end):
            if chars[k] != '\n':
                chars[k] = ' '
    masked = ''.join(chars)
    assert len(masked) == len(source), 'the mask must be index-aligned with the source'
    return masked


def find_function_params(
    source: str,
    func_name: str,
    miss: Callable[[str], BaseException] | None = None,
) -> tuple[str, int, int]:
    """Locate ``function <func_name>(``'s parameter list.

    Returns ``(masked, params_start, params_end)`` where *masked* is the
    `_mask_js` copy the caller keeps walking, ``source[params_start:params_end]``
    is the parameter-list text with the parens EXCLUDED, and *params_end* is the
    index OF the matching ``)``.  A caller wanting the body resumes with
    ``masked.find('{', params_end + 1)``.

    The search and the depth walk run over the mask, so a ``(``, ``)`` or the
    word ``function`` inside a STRING LITERAL OR A COMMENT is not counted.  The
    mask is length-preserving and index-aligned, so both indices address the
    ORIGINAL source and the slice a caller takes is the real text.

    Paren-DEPTH rather than "find the next ``)``": a destructured parameter
    (``function Foo({ a, b }) {``) carries its own ``{``/``}`` pair inside the
    parameter list, so a caller that took the first ``{`` after the opening
    paren would get the destructuring pattern instead of the body.

    The search regex is deliberately NOT line-anchored, so a declaration NESTED
    inside another function is found (the real instance is
    ``function searchMatches(t) {`` indented inside ``TasksTab`` in
    tab_tasks.jsx).  Its trailing ``\\s*\\(`` is equally load-bearing in the
    other direction: without it a prefix sibling declared earlier would shadow
    the target — ``function TaskGraphEdges(`` at tab_tasks.jsx:33 precedes
    ``function TaskGraph(`` at :151.

    RAISES on a miss rather than returning a sentinel, because every consumer
    slices the source by the returned indices and a sentinel would hand them a
    silently wrong — or empty — slice, over which absence assertions pass
    vacuously.  *miss* lets a caller supply the exception: `extract_function_body`
    threads its own four-way wording through it, and test_charts_axis_labels.py
    keeps a file-specific message.  The default raises ``AssertionError``.
    """
    def _default_miss(what: str) -> BaseException:
        return AssertionError(
            f'Could not locate the `function {func_name}(` parameter list: '
            f'{what}. Either the function was removed or renamed, or it was '
            f'rewritten as an arrow function or a class method — neither is '
            f'matched, only a named `function` declaration is.'
        )

    _miss = miss if miss is not None else _default_miss

    spans = _scan_js(source)
    _assert_js_lexable(source, spans)
    masked = _mask_js(source, spans)

    match = re.search(rf'\bfunction\s+{re.escape(func_name)}\s*\(', masked)
    if match is None:
        raise _miss('no such declaration in this source')

    paren_depth = 1
    i = match.end()
    while i < len(masked) and paren_depth > 0:
        if masked[i] == '(':
            paren_depth += 1
        elif masked[i] == ')':
            paren_depth -= 1
        i += 1
    if paren_depth != 0:
        raise _miss('its parameter list is never closed')

    return masked, match.end(), i - 1


def extract_function_body(source: str, func_name: str) -> str:
    """Return the brace-delimited body of a ``function <func_name>(`` declaration.

    The returned slice starts at the body's opening ``{`` and ends at its
    matching ``}``, both included — the SIGNATURE AND PARAMETER LIST ARE
    EXCLUDED.  Only named ``function`` declarations are matched: an arrow
    function bound to a const, and a class method spelled ``Foo(a) {``, carry
    no ``function`` keyword and are misses.

    The search and both depth walks run over a `_mask_js` copy of the source, so
    a brace, paren or ``function`` keyword inside a STRING LITERAL OR A COMMENT
    is not counted.  Without that, ``const s = '}'`` inside a body ends the
    brace walk early and the caller gets a truncated, unbalanced slice — every
    absence assertion over which then passes vacuously, which is the same
    permanent false GREEN the raise-on-miss rule below exists to prevent.  The
    mask is index-aligned with the source, so the slice returned is the real
    text, comments and literals intact.

    Paren-depth walks past the parameter list before looking for the body's
    opening ``{`` — a destructured parameter (``function Foo({ a, b }) {``)
    contains its own ``{``/``}`` pair *inside* the parameter list, so naively
    taking the first ``{`` after the opening ``(`` would return just the
    destructuring pattern (e.g. ``{ a, b }``) instead of the function body.

    The search regex is deliberately NOT line-anchored, so a declaration
    NESTED inside another function is found and scoped to its own body (the
    real instance is ``function searchMatches(t) {`` indented inside
    ``TasksTab`` in tab_tasks.jsx).  Its trailing ``\\s*\\(`` is equally
    load-bearing in the other direction: without it a prefix sibling declared
    earlier would shadow the target — ``function TaskGraphEdges(`` at
    tab_tasks.jsx:33 precedes ``function TaskGraph(`` at :151.

    RAISES ``AssertionError`` on any miss rather than returning ``''``.  An
    empty body makes every downstream ABSENCE assertion pass vacuously, which
    is a permanent false GREEN that no amount of care at the call site can
    detect; a loud failure naming the function is strictly better.
    """
    def _miss(what: str) -> AssertionError:
        return AssertionError(
            f'Could not locate the `function {func_name}(` body: {what}. Either '
            f'the function was removed or renamed, or it was rewritten as an '
            f'arrow function or a class method — neither is matched, only a '
            f'named `function` declaration is. This cannot silently return an '
            f'empty body: an absence assertion over one would pass vacuously.'
        )

    masked, _params_start, params_end = find_function_params(
        source, func_name, miss=_miss,
    )

    start = masked.find('{', params_end + 1)
    if start == -1:
        raise _miss('no opening brace follows its parameter list')

    depth = 0
    for j in range(start, len(masked)):
        char = masked[j]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return source[start : j + 1]
    raise _miss('its body brace is never closed')


def strip_js_comments(source: str) -> str:
    """Return *source* with every JS/JSX comment blanked, string literals intact.

    The probes built on these helpers are plain substring/regex searches, so
    without this they also match PROSE.  That coupling deformed the production
    source once already: charts.jsx carried a comment whose content was an
    apology for what it could not say, because naming the very expression the
    component had just stopped using would fail CI with a message claiming the
    component "still contains hole-blind scale/path arithmetic" — pointing at a
    comment.  A comment is exactly where that expression SHOULD be quotable.

    Quote-aware rather than a bare regex: blanking from a ``//`` inside a string
    literal (a URL, say) to end-of-line would delete real CODE, and an absence
    assertion over deleted code is a permanent false GREEN.  All three quote
    styles (``'``, ``"``, backtick) are tracked, and an escaped character inside
    a literal cannot close it.

    Each comment is replaced by a SINGLE SPACE rather than removed outright, so
    two previously separated tokens can never be spliced into a new match.

    Deliberately not a full JS lexer (see `_scan_js` for the exact blind spots).
    The one it is most likely to meet — an apostrophe in JSX text, which reads
    as an opening quote and leaves every comment up to the next ``'`` in the
    file unstripped — is not silently tolerated: `_assert_js_lexable` RAISES on
    it.  A missed comment is a false RED for an absence probe and a false GREEN
    for a presence one, and neither is visible at the call site.
    """
    spans = _scan_js(source)
    _assert_js_lexable(source, spans)

    out: list[str] = []
    prev = 0
    for span in spans:
        if span.kind != 'comment':
            continue  # a string literal is passed through verbatim
        out.append(source[prev : span.start])
        out.append(' ')
        prev = span.end
    out.append(source[prev:])

    return ''.join(out)


# ---------------------------------------------------------------------------
# window.DF_CHARTS namespace destructure/export parsing.
#
# charts.jsx publishes its components as `window.DF_CHARTS = { ... }` and each
# consumer picks them up with `const { ... } = window.DF_CHARTS`.  Several
# suites parse those two lines rather than hardcoding a component list, so the
# list they check against can never drift from what the files actually say.
#
# `destructure_bindings` returns (canonical, local) PAIRS, and each caller
# projects the half it needs.  This is not fussiness — its two DF_CHARTS
# consumers answer OPPOSITE questions over the identical line:
#   test_charts_consumer_bindings wants the LOCAL/alias name  — what the file
#       must actually reference, since the alias is what it renders by;
#   test_tab_burndown wants BOTH, as an alias -> canonical map, plus the
#       CANONICAL/source names of the export literal — what must actually
#       exist on the namespace object.
# On tabs.jsx's real `HistBar: HB` those are 'HB' and 'HistBar'.  A primitive
# that picked one side would silently INVERT one of the two suites, which is
# precisely the canonical-vs-alias slip test_charts_consumer_bindings.py
# freezes a negative-control fixture against.
#
# The list shape is load-bearing for the same reason: order is preserved and
# duplicates are NOT collapsed, because the callers' own collection shapes
# differ (dict last-wins / ordered list keeping duplicates / deduped set).
# Share the parser, not the policy — each consumer also keeps its own
# search-vs-finditer choice and its own miss behaviour.
#
# The `[^{}]*` class in both patterns is brace-HOSTILE ON PURPOSE and must NOT
# be widened.  Two independent reasons, from the two suites that documented it:
#   - Widening to swallow the other DF_CHARTS access shapes is actively wrong,
#     not merely extra work: a namespace binding's own name IS used, so a naive
#     extension flags it as a false positive; and member reads off a namespace
#     object are not statically enumerable the way a destructure list is.  The
#     defect these suites exist to catch can only exist in the destructure
#     shape anyway.
#   - A NESTED brace must yield NO match rather than a half-read binding list:
#     an empty result turns a downstream assertion red, where a half-read list
#     would let a sweep go quietly wrong.  No call site names the coupling on
#     that miss (test_tab_burndown.py's readers return an empty result), so the
#     property is pinned in the contract below.
#
# Contract: test_jsx_source_helpers.py::TestDfChartsDestructure.
# ---------------------------------------------------------------------------

DF_CHARTS_DESTRUCTURE_RE = re.compile(r'const\s*\{([^{}]*)\}\s*=\s*window\.DF_CHARTS')
"""The CONSUMER shape: `const { Foo, Bar: B } = window.DF_CHARTS`."""

DF_CHARTS_EXPORT_RE = re.compile(r'window\.DF_CHARTS\s*=\s*\{([^{}]*)\}')
"""The PROVIDER shape: `window.DF_CHARTS = { Foo, Bar }` in charts.jsx."""


def destructure_bindings(brace_body: str) -> list[tuple[str, str]]:
    """Split a destructure/object-literal brace body into (canonical, local) pairs.

    *brace_body* is the inside of the braces — typically ``m.group(1)`` from one
    of the two patterns above, though the surrounding braces are harmless.

    ``{ StackedAreaChart, HistBar: HB }`` yields
    ``[('StackedAreaChart', 'StackedAreaChart'), ('HistBar', 'HB')]``: a bare
    name is BOTH canonical and local; an aliased one splits on the FIRST colon,
    canonical left and local right.  Both halves are whitespace-stripped, empty
    parts (a trailing comma, say) produce no entry, and source ORDER is
    preserved with DUPLICATES INTACT so each caller can impose its own
    collection shape.
    """
    pairs: list[tuple[str, str]] = []
    for part in brace_body.strip().strip('{}').split(','):
        part = part.strip()
        if not part:
            continue
        canonical, _, alias = part.partition(':')
        canonical = canonical.strip()
        pairs.append((canonical, alias.strip() or canonical))
    return pairs


# ---------------------------------------------------------------------------
# Balanced-delimiter walking, and window.DF_DATA seed-block extraction.
#
# The dashboard's served data.js carries its fixture payload as a
# `window.DF_DATA = { KEY: { ... }, ... }` literal, and several suites need to
# scope an assertion to ONE key's object so a token elsewhere in the file
# cannot satisfy it.
#
# This used to be a private copy in three test modules (test_tab_escalations,
# test_tab_memory_evals, test_tab_escalation_analytics) whose code was
# byte-identical.  Its contract lives in
# test_jsx_source_helpers.py::TestExtractDfDataBlock.
#
# The WALK is separated from the ANCHOR for the same reason `extract_function_body`
# was rebuilt on `find_function_params`: two helpers needed the identical
# depth loop with DIFFERENT anchors, so keeping the loop in both meant the
# string-literal blind spot below was documented — and would have to be
# fixed — in two places.  `walk_balanced` is the loop;
# `extract_df_data_block` anchors it on data.js's `key: {` seed form and
# test_tab_memory_evals.py's `_extract_const_object` anchors it on a
# module-scope `const NAME = {`/`[` declaration.  Its own contract lives in
# test_jsx_source_helpers.py::TestWalkBalanced.
#
# Two behaviours differ from the sibling `extract_function_body` and are
# deliberately kept as they were rather than changed in the move: these return
# `''` SILENTLY on a miss where the other RAISES (every call site already
# asserts on the returned value, so raising would only relocate their
# failures), and the depth walk is NOT quote-aware where the other is.  Both
# are pinned as current behaviour, so upgrading either later is a visible
# contract edit rather than silent drift.
# ---------------------------------------------------------------------------


def walk_balanced(
    src: str, start: int, open_char: str = '{', close_char: str = '}'
) -> str:
    """Return ``src`` from ``start`` through the delimiter matching ``src[start]``.

    ``start`` must be the index OF the opening delimiter; both callers get it
    from a regex whose pattern ends on that delimiter (``m.end() - 1``).  The
    walk counts ``open_char``/``close_char`` so a NESTED pair does not
    terminate it early — which is the whole reason a ``[^}]*`` regex was
    rejected for this job.

    Returns the delimited text INCLUDING both delimiters, or the empty string
    if the opening delimiter is never closed.  Returning ``''`` rather than
    raising is the deliberate policy of this family (see the banner above).

    Note: the depth walk does not skip delimiters inside JS string literals,
    so a quoted ``{`` or ``}`` miscounts.  This is the single place that
    limitation now lives; the callers document what makes it acceptable for
    the sources they read.
    """
    depth = 0
    for i in range(start, len(src)):
        c = src[i]
        if c == open_char:
            depth += 1
        elif c == close_char:
            depth -= 1
            if depth == 0:
                return src[start : i + 1]
    return ''


def extract_df_data_block(src: str, key: str) -> str:
    """Return the body of the ``<key>: { ... }`` seed object, braces included.

    Locates ``<key>:`` followed by ``{`` (allowing arbitrary whitespace), then
    hands off to ``walk_balanced`` to find the matching close brace.
    This is brace-aware: a simple regex ``[^}]*`` would stop at the first
    nested ``}`` and miss later keys.
    Returns the empty string if no matching block is found.

    Note: ``walk_balanced`` does not skip ``{``/``}`` inside JS string
    literals.  This is acceptable because the data.js seed block uses simple
    numeric/array values and does not embed brace characters inside quoted
    strings.
    """
    m = re.search(rf'{re.escape(key)}\s*:\s*\{{', src)
    if m is None:
        return ''
    return walk_balanced(src, m.end() - 1)  # m.end() - 1 is the opening `{`


# ---------------------------------------------------------------------------
# JSX opening-tag walk.
#
# Spans one opening tag, attributes included, so a probe can confine itself to
# that element.  `jsx_open_tag_end` is the walk from one known `<`;
# `jsx_open_tags` reads every opening tag a pattern finds, comment-stripped;
# `jsx_prop_expr` reads one prop's expression out of such a tag.
# Contracts: test_jsx_source_helpers.py::TestJsxOpenTagEnd,
# test_jsx_source_helpers.py::TestJsxOpenTags and
# test_jsx_source_helpers.py::TestJsxPropExpr.
# ---------------------------------------------------------------------------


def jsx_open_tag_end(src: str, start: int) -> int | None:
    """Return the index JUST PAST the ``>`` closing the opening tag at *start*.

    *start* must index the tag's ``<``, else ``ValueError``.  ``src[start:end]``
    is the whole tag; ``src[end - 2:end] == '/>'`` tells a self-closing one.  A
    ``<`` or ``>`` inside a ``{...}`` attribute expression or a string/template
    literal does not end the tag.

    ``None``, never a raise, when *start* begins no closed opening tag: it is
    never closed, a literal is never terminated, a bare ``<`` opens at depth 0
    first, or a ``}`` drives brace depth negative.

    Not comment-aware (pass `strip_js_comments` output), and it shares
    `_scan_js`'s lexer blind spots.
    """
    if src[start : start + 1] != '<':
        raise ValueError(
            f"jsx_open_tag_end: start={start} must index the tag's `<`, "
            f'found {src[start : start + 1]!r}'
        )
    depth = 0
    i = start + 1
    while i < len(src):
        c = src[i]
        if c in '\'"`':
            i, closed = _string_literal_end(src, i)
            if not closed:
                return None
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0 and c == '>':
            return i + 1
        elif depth == 0 and c == '<':
            return None
        i += 1
    return None


def jsx_open_tags(src: str, pattern: re.Pattern[str]) -> list[str]:
    """Return every opening tag *pattern* finds in comment-stripped *src*, in order.

    A tag with children is returned as its opening tag, which carries every
    attribute; ``tag[-2:] == '/>'`` tells a self-closing one.  Each match must
    start at the tag's ``<`` (else `jsx_open_tag_end`'s ``ValueError``), and a
    tag the walk cannot close raises rather than being dropped.
    """
    code = strip_js_comments(src)
    tags: list[str] = []
    for m in pattern.finditer(code):
        end = jsx_open_tag_end(code, m.start())
        if end is None:
            raise AssertionError(
                f'jsx_open_tags: a {pattern.pattern!r} tag never closes, or a '
                f'bare `<` or a stray `}}` comes first, so no probe over its span '
                f'would mean anything: {code[m.start() : m.start() + 80]!r}'
            )
        tags.append(code[m.start() : end])
    return tags


def jsx_prop_expr(opening_tag: str, prop: str) -> str | None:
    """Return the expression inside *opening_tag*'s own ``prop={...}``, stripped.

    ``None`` when the tag passes no braced *prop*.  The search and the brace
    walk run over the `_mask_js` text, so a quoted brace or a quoted
    ``prop={`` is not code, and only a prop at brace depth 0 counts: a nested
    element's ``prop={...}`` is not this tag's.  Raises when the expression
    never closes.  *opening_tag* is a `jsx_open_tags` result.
    """
    spans = _scan_js(opening_tag)
    _assert_js_lexable(opening_tag, spans)
    code = _mask_js(opening_tag, spans)
    for m in re.finditer(rf'\b{re.escape(prop)}=\{{', code):
        if code.count('{', 0, m.start()) != code.count('}', 0, m.start()):
            continue
        open_at = m.end() - 1
        braced = walk_balanced(code, open_at)
        assert braced, (
            f'jsx_prop_expr: {prop}={{...}} never closes in {opening_tag[:80]!r}'
        )
        return opening_tag[open_at + 1 : open_at + len(braced) - 1].strip()
    return None


# ---------------------------------------------------------------------------
# Served-HTML script order.
#
# index.html loads its scripts as classic synchronous tags, so DOCUMENT order
# is EXECUTION order — which is what lets a text-level test assert that a
# provider script loads before the consumer that dereferences it at module
# scope.  That equivalence is fragile: `defer`, `async` or `type="module"` on
# either tag breaks it, and a position comparison over a deferred pair is a
# false pass, not a failure.  Hence the guard below runs BEFORE the comparison.
#
# These three used to be private copies in FIVE test modules
# (test_esc_flow_diagram, test_index_html, test_tab_escalation_analytics,
# test_tab_escalations, test_tab_memory_evals), so a fix to the false-pass
# guard had to be applied five times or not at all.  Their contract lives in
# test_jsx_source_helpers.py::TestScriptOrderHelpers.
#
# The five copies agreed byte-for-byte except in one place, resolved here in
# favour of the canonical 4-of-5 form: the ORDERING failure message ends with
# the caller's `consumer_note`, where test_index_html.py alone substituted a
# fixed two-sentence string.  Nothing pins that message — the only tests that
# match this helper's failure text (test_index_html.py:303 and :338) pin the
# GUARD phrases, which are identical across all five — so both variants were
# green, and this one strictly carries more information: it surfaces the note
# at index_html's 13 note-passing call sites instead of discarding the notes
# at the other four modules' 14 sites.  The guard messages themselves are
# carried over verbatim and MUST stay that way; index_html's
# `_DEFERRED_CDN_CASES` / `_DEFERRED_TAB_TASKS_CASES` turn them into
# `pytest.raises(match=...)` patterns.
# ---------------------------------------------------------------------------


class ScriptTagCollector(html.parser.HTMLParser):
    """Collects the attribute dicts for every <script> start-tag encountered."""

    def __init__(self) -> None:
        super().__init__()
        self.script_attrs: list[dict[str, str | None]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == 'script':
            self.script_attrs.append(dict(attrs))


def find_script_position(
    body: str, src_prefix: str
) -> tuple[int, dict[str, str | None]] | None:
    """Return ``(index, attrs)`` for the first <script> tag whose ``src``
    starts with ``src_prefix``, or ``None`` if no such tag exists.

    ``index`` is the tag's 0-based position in ``ScriptTagCollector.script_attrs``
    (document order, since the list preserves insertion order).  Returning attrs
    alongside the position avoids a second parse when the caller also needs the
    src or other attributes.
    """
    collector = ScriptTagCollector()
    collector.feed(body)
    for i, attrs in enumerate(collector.script_attrs):
        if (attrs.get('src') or '').startswith(src_prefix):
            return i, attrs
    return None


def assert_script_loads_before(
    body: str,
    before_src_prefix: str,
    after_src_prefix: str,
    before_label: str,
    after_label: str,
    consumer_note: str = '',
) -> None:
    """Assert that the script for ``before_src_prefix`` loads BEFORE the
    script for ``after_src_prefix`` in ``body``.  Combines a
    defer/async/type=module false-pass guard with the document-order
    position comparison.
    """
    before_result = find_script_position(body, before_src_prefix)
    assert before_result is not None, (
        f'No <script src="{before_src_prefix}..."> tag found in index.html. '
        f'{consumer_note}'
    )
    before_pos, before_attrs = before_result
    before_src = before_attrs.get('src')

    after_result = find_script_position(body, after_src_prefix)
    assert after_result is not None, (
        f'<script src="{after_src_prefix}..."> not found in index.html — '
        f'cannot verify load-order invariant for {before_label}.'
    )
    after_pos, after_attrs = after_result

    # Both tags must be classic synchronous scripts — otherwise document order
    # diverges from execution order and the position comparison below is moot.
    for _label, _attrs in [
        (before_label, before_attrs),
        (after_label, after_attrs),
    ]:
        assert 'defer' not in _attrs, (
            f'{_label} has a defer attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert 'async' not in _attrs, (
            f'{_label} has an async attribute; document order no longer implies '
            f'execution order, so the load-order check below may give a false pass.'
        )
        assert (_attrs.get('type') or '').lower() != 'module', (
            f'{_label} has type="module"; ES modules are deferred by default, '
            f'so document order no longer implies execution order.'
        )

    assert before_pos < after_pos, (
        f'{before_label} (position {before_pos}, src={before_src!r}) must load '
        f'BEFORE {after_label} (position {after_pos}). '
        f'{consumer_note}'
    )
