"""Contract for the shared JSX source-slicing helpers in `_dashboard_helpers`.

The dashboard suite asserts structural contracts against the *served* .jsx
text (there is no JS runtime in this project), and nearly every such
assertion first needs to scope itself to one function's body — otherwise a
token appearing anywhere else in the file satisfies it and the test proves
nothing.  That scoping helper used to be copied into each consuming module,
which meant a fix to it had to be applied nine times or not at all.  This
file owns its contract so the single shared implementation can be changed
with confidence.

`extract_function_body` is the brace-walk variant: it returns exactly the
brace-delimited block of a named `function` declaration, signature excluded,
and it RAISES on a miss.  Raising rather than returning `''` is load-bearing:
a silently-empty body makes every downstream ABSENCE assertion pass
vacuously, which is a permanent false GREEN.

`strip_js_comments` is the companion: every probe in these suites is a plain
substring or regex search, so without it they also match PROSE — and a
component is then forbidden from documenting the very expression it stopped
using.  Its over-reach direction is the dangerous one and is pinned hardest
here: blanking real code would disarm every absence assertion downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from _dashboard_helpers import (
    DF_CHARTS_DESTRUCTURE_RE,
    DF_CHARTS_EXPORT_RE,
    ScriptTagCollector,
    assert_script_loads_before,
    destructure_bindings,
    extract_df_data_block,
    extract_function_body,
    find_function_params,
    find_script_position,
    strip_js_comments,
)


class TestExtractFunctionBody:
    """The brace-walk extractor's contract, pinned against synthetic sources."""

    def test_returns_the_brace_delimited_block(self) -> None:
        """The result is exactly `{...}` — balanced, and the signature is excluded."""
        src = 'function Foo(a, b) { const x = 1; }'
        body = extract_function_body(src, 'Foo')

        assert body.startswith('{'), f'body must start at the opening brace, got {body!r}'
        assert body.endswith('}'), f'body must end at the matching close brace, got {body!r}'
        assert body.count('{') == body.count('}'), f'braces must balance in {body!r}'
        assert 'const x = 1;' in body
        assert 'function Foo' not in body, 'the signature is NOT part of the returned slice'

    def test_destructured_parameter_list_is_walked_past(self) -> None:
        """`function Foo({ a, b }) {` — the pattern's own braces are not the body.

        This is what the paren-depth walk buys: naively taking the first `{`
        after the opening `(` would return the destructuring pattern
        (`{ a, b }`) and every probe against it would then be answered by the
        parameter names instead of the code.
        """
        src = 'function Foo({ a, b }) { const x = 1; }'
        body = extract_function_body(src, 'Foo')

        assert 'const x = 1;' in body, f'the BODY must be returned, got {body!r}'
        assert 'a, b' not in body, (
            f'the destructuring pattern was returned instead of the body: {body!r}'
        )

    def test_nested_declaration_is_scoped_to_its_own_body(self) -> None:
        """A function declared INSIDE another is extracted on its own.

        This is the limitation the retired top-level-slice variant carried: it
        anchored its regex at column 0 and sliced to the next top-level
        `function`, so an indented inner declaration could only ever be
        reached by returning the whole enclosing function.  The real instance
        is `function statusMatches(s) {` nested inside `TasksTab` in
        tab_tasks.jsx.
        """
        src = "function Outer() {\n  function inner(s) { return s === 'x'; }\n  return 1;\n}"
        body = extract_function_body(src, 'inner')

        assert "return s === 'x';" in body, f"inner's own body must be returned, got {body!r}"
        assert 'return 1;' not in body, (
            f'the extractor fell back to the ENCLOSING function body: {body!r}'
        )

    def test_prefix_sibling_does_not_shadow_the_target(self) -> None:
        """`function FooEdges(` declared first must not answer a request for `Foo`.

        The trailing `\\s*\\(` in the regex is the only thing separating them.
        Real instance: `function TaskGraphEdges(` at tab_tasks.jsx:33 precedes
        `function TaskGraph(` at :151.
        """
        src = 'function FooEdges(a) { const edges = 1; }\nfunction Foo(a) { const own = 2; }'
        body = extract_function_body(src, 'Foo')

        assert 'const own = 2;' in body, f"Foo's own body must be returned, got {body!r}"
        assert 'const edges = 1;' not in body, (
            f'the prefix-sibling FooEdges shadowed Foo: {body!r}'
        )

    def test_a_brace_inside_a_string_literal_does_not_end_the_body(self) -> None:
        """`const s = '}';` must not be read as the body's closing brace.

        A depth walk that counts every `{`/`}` character stops at the literal
        and hands back a TRUNCATED, unbalanced slice — silently.  Everything
        after the literal (which is most of a real component) then simply is
        not there, so every ABSENCE probe over the result passes vacuously
        against a two-line stub: the same permanent false GREEN that
        raise-on-miss exists to prevent, reached by a different route.
        """
        src = "function Foo(a) {\n  const s = '}';\n  const banned = (v / max) * 100;\n}"
        body = extract_function_body(src, 'Foo')

        assert '(v / max) * 100' in body, (
            f'the walk stopped at the `}}` inside the string literal, so the rest '
            f'of the body is missing from {body!r}'
        )
        assert body.endswith('\n}'), (
            f'the body must end at the real closing brace, got {body!r}'
        )

    def test_a_brace_inside_a_comment_does_not_end_the_body(self) -> None:
        """The same hazard spelled as a comment rather than a literal."""
        src = 'function Foo(a) { /* } */ const kept = 1; }'
        body = extract_function_body(src, 'Foo')

        assert 'const kept = 1;' in body, (
            f'the walk stopped at the `}}` inside the block comment: {body!r}'
        )

    def test_an_unbalanced_body_raises_rather_than_returning_a_truncated_slice(self) -> None:
        """When the body cannot be closed, the answer is an error, never a prefix.

        The literal's `}` is the ONLY `}` in this source, and it is not a
        closing brace.  Returning the slice up to it would be a well-formed
        looking string — which is exactly why it has to raise instead.
        """
        src = "function Foo(a) {\n  const s = '}';\n  const banned = (v / max) * 100;"

        with pytest.raises(AssertionError, match='Foo'):
            extract_function_body(src, 'Foo')

    def test_a_declaration_inside_a_comment_is_not_the_declaration(self) -> None:
        """A commented-out `function Foo(` must not answer a request for `Foo`.

        Real instance: tweaks-panel.jsx documents its edit-mode protocol with a
        commented-out `function App() {` block, and a scan that does not know
        about comments slices that prose as if it were the component.
        """
        src = '// function Foo(a) { const commented = 1; }\nconst x = 2;'

        with pytest.raises(AssertionError, match='Foo'):
            extract_function_body(src, 'Foo')

    def test_missing_function_raises_with_a_naming_diagnostic(self) -> None:
        """A miss is LOUD, and the message names the function that missed.

        Returning `''` here (the retired behaviour) made every downstream
        absence assertion pass vacuously — a permanent false GREEN that no
        amount of downstream care can detect.

        Only the NAME is pinned.  The message also spells out the known limits
        (arrow function, class method) and that prose is worth having, but
        pinning it would turn a rewording into a red suite while proving
        nothing further about behaviour.
        """
        src = 'function Foo(a) { const x = 1; }'

        with pytest.raises(AssertionError, match='NotDeclared'):
            extract_function_body(src, 'NotDeclared')

    def test_arrow_function_binding_is_not_matched(self) -> None:
        """`const Foo = (a) => {...}` is not a named `function` declaration."""
        src = 'const Foo = (a) => { const x = 1; };'

        with pytest.raises(AssertionError, match='Foo'):
            extract_function_body(src, 'Foo')

    def test_class_method_is_not_matched(self) -> None:
        """A method spelled `Foo(a) {` inside a class carries no `function` keyword."""
        src = 'class C {\n  Foo(a) { const x = 1; }\n}'

        with pytest.raises(AssertionError, match='Foo'):
            extract_function_body(src, 'Foo')

    def test_unclosed_body_raises_rather_than_returning_a_partial_slice(self) -> None:
        """A truncated source must not yield an unbalanced fragment.

        A partial slice would silently drop the tail of the function, so a
        presence assertion on anything past the truncation point would fail
        for the wrong reason and an absence assertion would pass for the wrong
        reason.
        """
        src = 'function Foo(a) { const x = 1;'

        with pytest.raises(AssertionError, match='Foo'):
            extract_function_body(src, 'Foo')

    def test_unclosed_parameter_list_raises(self) -> None:
        """A `function Foo(` whose parameter list never closes is also a miss."""
        src = 'function Foo(a, b'

        with pytest.raises(AssertionError, match='Foo'):
            extract_function_body(src, 'Foo')


class TestStripJsComments:
    """The quote-aware comment stripper's contract.

    Mirrors and extends the contract previously pinned by
    test_charts_null_samples.py's own `test_strip_comments_*` tests, so
    nothing that file guarantees today is lost by the consolidation.
    """

    def test_line_comment_is_blanked_and_the_code_either_side_survives(self) -> None:
        stripped = strip_js_comments(
            'const a = 1; // st.values[i] || 0 is the old scrub\nconst b = 2;'
        )

        assert 'st.values[i] || 0' not in stripped, 'a line comment must not be probed'
        assert 'const a = 1;' in stripped, 'code before the comment survives'
        assert 'const b = 2;' in stripped, 'code after the comment survives'

    def test_block_comment_is_blanked_and_the_code_either_side_survives(self) -> None:
        stripped = strip_js_comments('const a = 1; /* Math.max(...values, 1) */ const b = 2;')

        assert 'Math.max(...values' not in stripped, 'a block comment must not be probed'
        assert 'const a = 1;' in stripped
        assert 'const b = 2;' in stripped

    def test_jsx_comment_is_blanked_and_the_markup_survives(self) -> None:
        stripped = strip_js_comments('<g>{/* (v / max) * 100 */}<rect /></g>')

        assert '(v / max) * 100' not in stripped, 'a JSX comment must not be probed'
        assert '<rect />' in stripped

    @pytest.mark.parametrize(
        'literal',
        ['"https://x/y"', "'https://x/y'", '`https://x/y`'],
        ids=['double', 'single', 'template'],
    )
    def test_a_slash_slash_inside_a_string_literal_is_not_a_comment(self, literal: str) -> None:
        """The OVER-REACH direction — the one that produces a false GREEN.

        Blanking from a `//` inside a string literal to end-of-line deletes
        real CODE, and an absence assertion over deleted code passes for the
        wrong reason, permanently and silently.  All three quote styles are
        exercised because the retired two-regex variant guarded only the
        `https:` case and would fail the single-quoted and template rows.
        """
        stripped = strip_js_comments(f'const u = {literal}; const h = (v / max) * chartH;')

        assert '(v / max) * chartH' in stripped, (
            f'a `//` inside the string literal {literal} was treated as a comment, '
            f'blanking the code after it — that silently disarms every probe on '
            f'the rest of the line'
        )

    def test_a_comment_does_not_splice_the_tokens_it_separated(self) -> None:
        """Each comment becomes a single space, never nothing.

        Removing it outright would join `a` and `b` into `ab` — a token that
        appears nowhere in the source, so a presence assertion could be
        satisfied by text the file does not contain.
        """
        assert 'ab' not in strip_js_comments('a/* */b')

    def test_an_escaped_quote_does_not_terminate_the_literal_early(self) -> None:
        r"""`'a\'b // c'` stays one literal, so its `//` is still not a comment.

        Were the escape mishandled, the scanner would consider the string
        closed at the escaped quote and treat the rest as code — blanking from
        the `//` onwards, which is the over-reach direction again.
        """
        stripped = strip_js_comments(r"const s = 'a\'b // c'; const keep = 1;")

        assert 'const keep = 1;' in stripped, (
            'an escaped quote closed the literal early, so the `//` inside it '
            'was treated as a comment and the following code was blanked'
        )

    def test_an_apostrophe_in_jsx_text_raises_instead_of_swallowing_comments(self) -> None:
        """`don't` in a label is read as an opening quote — that must be LOUD.

        This is the scanner's likeliest real-world blind spot, and its silent
        failure is nasty in both directions: everything from the apostrophe to
        the next `'` anywhere later in the file is treated as string contents,
        so real comments inside that stretch are handed back UNSTRIPPED (prose
        the consumers then probe as if it were code — a false RED for an
        absence assertion and a false GREEN for a presence one) and real braces
        inside it are not counted.
        """
        src = (
            'function Foo(a) {\n'
            "  return <div>don't panic</div>; // Math.max(...values, 1) was the old scrub\n"
            '}\n'
        )

        with pytest.raises(AssertionError, match='never closed'):
            strip_js_comments(src)

    def test_a_quote_literal_spanning_a_newline_raises(self) -> None:
        """The same misparse when a later `'` does eventually "close" it.

        JS forbids a raw newline inside a `'`/`"` literal, so a literal that
        spans one is a misparse rather than a long string — and unlike the
        unterminated case it leaves the scan looking perfectly healthy.  Here
        the apostrophe in `don't` pairs with the quote of `'ok'` two lines
        later, blanking nothing and swallowing the comment between them.
        """
        src = "const a = 1; // keep\nconst label = <b>don't panic</b>;\nconst b = 'ok';"

        with pytest.raises(AssertionError, match='spans a newline'):
            strip_js_comments(src)

    def test_a_template_literal_may_span_newlines(self) -> None:
        """A backtick literal legitimately spans lines and must NOT raise.

        tabs.jsx contains one, so this is the case the newline rule above has
        to leave alone.
        """
        stripped = strip_js_comments('const t = `line one\nline two`; // Math.max(...v, 1)\n')

        assert 'line two' in stripped, 'the template literal must survive intact'
        assert 'Math.max(...v' not in stripped, 'the trailing comment is still blanked'

    def test_composes_with_the_extractor_over_the_same_source(self) -> None:
        """Stripping first must not cost the extractor the function it needs.

        This is how the real consumers use the pair — extract, then strip, or
        strip, then extract — so the stripper has to leave declarations, brace
        structure and line breaks intact.
        """
        src = (
            'function Foo(a) {\n'
            '  // Math.max(...values, 1) was the old scrub\n'
            '  const kept = 1;\n'
            '}\n'
        )

        body = extract_function_body(strip_js_comments(src), 'Foo')

        assert 'const kept = 1;' in body, 'the code inside the stripped body survives'
        assert 'Math.max(...values' not in body, 'the comment inside the body is blanked'


# The served assets whose per-module fixture copies conftest.py now owns.
# One row per asset: adding a tenth shared asset is a one-line change here and
# a one-line fixture in conftest.py — and until BOTH land,
# `test_every_conftest_body_fixture_has_a_row` below fails, so this table
# cannot silently fall behind conftest.
_SHARED_ASSET_FIXTURES = {
    'index_html_body': '/static/redux/index.html',
    'data_js_body': '/static/redux/data.js',
    'app_jsx_body': '/static/redux/app.jsx',
    'shell_jsx_body': '/static/redux/shell.jsx',
    'tabs_jsx_body': '/static/redux/tabs.jsx',
    'charts_jsx_body': '/static/redux/charts.jsx',
    'tab_analytics_jsx_body': '/static/redux/tab_escalation_analytics.jsx',
    'tab_escalations_jsx_body': '/static/redux/tab_escalations.jsx',
    'tab_tasks_jsx_body': '/static/redux/tab_tasks.jsx',
}

_CONFTEST_PATH = Path(__file__).with_name('conftest.py')


def _dashboard_conftest():
    """Return the loaded `dashboard/tests/conftest.py` module, found BY PATH.

    Deliberately not `import conftest`: a root-level pytest run loads several
    subprojects' conftests in one process and they all compete for
    `sys.modules['conftest']` — the very collision `_dashboard_helpers.py`
    exists to sidestep (see its module docstring).  Matching on `__file__`
    names the right one unambiguously, whatever it was registered as.
    """
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path) == _CONFTEST_PATH:
            return module
    raise AssertionError(
        f'{_CONFTEST_PATH} is not loaded — pytest must import it to collect '
        f'this file, so either the tests moved or the import mechanism changed'
    )


def _conftest_body_fixture_names() -> set[str]:
    """Every served-asset fixture conftest defines, by name.

    Name-based ON PURPOSE.  Asking pytest "is this object a fixture?" is the
    version-unstable question — `@pytest.fixture` marked the function itself
    with `_pytestfixturefunction` through 8.3 and returns a
    `FixtureFunctionDefinition` wrapper on 9.x — whereas the `_body` suffix is
    this suite's own convention for a served-asset fixture and conftest defines
    nothing else that ends in it.  A future non-fixture `*_body` name there
    fails this loudly rather than going uncovered, which is the right direction.
    """
    return {name for name in vars(_dashboard_conftest()) if name.endswith('_body')}


def _resolved_client_scope(request) -> str | None:
    """The scope of the `_client` fixture definition in effect, or None.

    Both routes into pytest's fixture registry are private API, so this tries
    them in order and reports None if they have ALL moved, letting the caller
    xfail with an explanation rather than fail the run for a non-defect:

    1. `FixtureManager.getfixturedefs` — the definition actually in effect,
       which is the stronger answer.  Its signature changed inside the 8.x line
       (nodeid string before 8.1, node after), so both are attempted.
    2. the marker on the fixture object conftest defines — reachable under both
       the pre-9 (`_pytestfixturefunction`) and 9.x (`_fixture_function_marker`)
       spellings.  Weaker: it reads the DECLARATION rather than the resolution.
    """
    getfixturedefs = getattr(getattr(request, '_fixturemanager', None), 'getfixturedefs', None)
    if getfixturedefs is not None:
        for node_arg in (request.node, request.node.nodeid):
            try:
                fixturedefs = getfixturedefs('_client', node_arg)
            except Exception:
                continue
            if fixturedefs:
                # Least-specific first, so the definition in effect is last.
                return getattr(fixturedefs[-1], 'scope', None)

    fixture = getattr(_dashboard_conftest(), '_client', None)
    for attr in ('_fixture_function_marker', '_pytestfixturefunction'):
        marker = getattr(fixture, attr, None)
        if marker is not None:
            return getattr(marker, 'scope', None)
    return None


class TestFindFunctionParams:
    """The paren-depth-walk primitive shared by the two signature consumers.

    `extract_function_body` and test_charts_axis_labels.py's
    `_extract_signature` both had to locate `function NAME(` and walk to the
    matching `)`.  They return DISJOINT, adjacent slices of that declaration —
    the params between the parens, and the body from the brace — so neither
    could be built on the other, but the walk itself was duplicated.  This
    primitive is that walk, and it returns the two indices plus the masked
    source so each caller can take the slice it actually needs.
    """

    def test_returns_the_parameter_list_slice_parens_excluded(self) -> None:
        """`source[params_start:params_end]` is exactly the parameter text."""
        src = 'function Foo(a, b = 2) { const x = 1; }'

        masked, params_start, params_end = find_function_params(src, 'Foo')

        assert src[params_start:params_end] == 'a, b = 2'
        assert src[params_start - 1] == '(', 'params_start is just past the open paren'
        assert src[params_end] == ')', 'params_end is the index OF the matching paren'

    def test_the_body_brace_is_found_past_params_end(self) -> None:
        """`masked.find('{', params_end + 1)` is the body's opening brace."""
        src = 'function Foo(a) { const x = 1; }'

        masked, _params_start, params_end = find_function_params(src, 'Foo')
        start = masked.find('{', params_end + 1)

        assert src[start:] == '{ const x = 1; }'

    def test_a_destructured_parameter_does_not_capture_the_body(self) -> None:
        """`function Foo({ a, b }) {` must yield the BODY, never `{ a, b }`.

        The destructuring pattern carries its own `{`/`}` pair INSIDE the
        parameter list, so taking the first `{` after the opening paren returns
        the pattern instead of the body — a truncated slice every downstream
        absence assertion then passes vacuously over.
        """
        src = 'function Foo({ a, b }) { const x = 1; }'

        masked, params_start, params_end = find_function_params(src, 'Foo')

        assert src[params_start:params_end] == '{ a, b }'
        start = masked.find('{', params_end + 1)
        assert src[start:] == '{ const x = 1; }'

    def test_a_paren_inside_a_string_literal_is_not_counted(self) -> None:
        """The walk runs over the mask, so a `)` in a string cannot close it."""
        src = 'function Foo(a = ")") { const x = 1; }'

        masked, params_start, params_end = find_function_params(src, 'Foo')

        assert src[params_start:params_end] == 'a = ")"'
        start = masked.find('{', params_end + 1)
        assert src[start:] == '{ const x = 1; }'

    def test_the_function_keyword_inside_a_comment_is_not_matched(self) -> None:
        """A commented-out declaration must not shadow the real one."""
        src = '/* function Foo(decoy) {} */\nfunction Foo(real) { const x = 1; }'

        _masked, params_start, params_end = find_function_params(src, 'Foo')

        assert src[params_start:params_end] == 'real'

    def test_the_mask_is_index_aligned_with_the_source(self) -> None:
        """`len(masked) == len(source)`, so the indices slice the REAL text."""
        src = 'function Foo(a /* note */, b) { const s = "x"; }'

        masked, params_start, params_end = find_function_params(src, 'Foo')

        assert len(masked) == len(src)
        assert src[params_start:params_end] == 'a /* note */, b', (
            'the slice is taken from the original source, comments intact'
        )

    def test_finds_a_declaration_nested_inside_another_function(self) -> None:
        """The regex is deliberately NOT line-anchored.

        The real instance is `function statusMatches(s) {` indented inside
        `TasksTab` in tab_tasks.jsx.
        """
        src = (
            'function Outer(a) {\n'
            '    function Inner(b) { return b; }\n'
            '    return Inner(a);\n'
            '}'
        )

        _masked, params_start, params_end = find_function_params(src, 'Inner')

        assert src[params_start:params_end] == 'b'

    def test_a_prefix_sibling_does_not_shadow_the_target(self) -> None:
        """The trailing `\\s*\\(` stops `TaskGraphEdges(` matching `TaskGraph`.

        `function TaskGraphEdges(` at tab_tasks.jsx:33 precedes
        `function TaskGraph(` at :151, so without the anchor the earlier
        declaration wins and the wrong slice is returned.
        """
        src = 'function TaskGraphEdges(edges) { }\nfunction TaskGraph(nodes) { }'

        _masked, params_start, params_end = find_function_params(src, 'TaskGraph')

        assert src[params_start:params_end] == 'nodes'

    def test_raises_when_there_is_no_such_declaration(self) -> None:
        """A miss RAISES — never a sentinel, so no caller can go vacuously GREEN."""
        src = 'const Foo = (a) => a;'

        with pytest.raises(AssertionError):
            find_function_params(src, 'Foo')

    def test_raises_when_the_parameter_list_is_never_closed(self) -> None:
        src = 'function Foo(a, b'

        with pytest.raises(AssertionError):
            find_function_params(src, 'Foo')

    def test_the_miss_reason_is_available_to_the_caller(self) -> None:
        """Callers can supply their own exception factory.

        `extract_function_body` needs its four-way `_miss` wording preserved
        exactly, and test_charts_axis_labels.py keeps a file-specific message;
        neither can be served by one fixed string.
        """
        src = 'const Foo = (a) => a;'
        sentinel = 'CALLER SPECIFIC MESSAGE'

        with pytest.raises(AssertionError, match=sentinel):
            find_function_params(
                src, 'Foo', miss=lambda what: AssertionError(f'{sentinel}: {what}'),
            )


class TestDfChartsDestructure:
    """The shared `window.DF_CHARTS` destructure/export parser.

    The primitive returns (canonical, local) PAIRS rather than one side,
    because the three consumers it replaces project OPPOSITE halves of the
    same line: test_charts_consumer_bindings wants the LOCAL/alias name (what
    the file must actually reference), test_charts_axis_labels wants the
    CANONICAL/source name (what must exist on the namespace object), and
    test_tab_burndown wants both at once.  On `HistBar: HB` those are `'HB'`
    and `'HistBar'` — disjoint — so a primitive that picked a side would
    silently invert one suite.
    """

    def test_a_bare_name_is_both_canonical_and_local(self) -> None:
        assert destructure_bindings('{ StackedAreaChart }') == [
            ('StackedAreaChart', 'StackedAreaChart'),
        ]

    def test_an_alias_splits_canonical_left_local_right(self) -> None:
        """`Canonical: alias` — the source name binds to the local name."""
        assert destructure_bindings('{ StackedAreaChart, HistBar: HB }') == [
            ('StackedAreaChart', 'StackedAreaChart'),
            ('HistBar', 'HB'),
        ]

    def test_it_splits_on_the_first_colon_only(self) -> None:
        assert destructure_bindings('a: b: c') == [('a', 'b: c')]

    def test_both_halves_are_whitespace_stripped(self) -> None:
        assert destructure_bindings('\n   HistBar   :   HB   \n') == [
            ('HistBar', 'HB'),
        ]

    def test_empty_parts_and_trailing_commas_produce_no_entry(self) -> None:
        assert destructure_bindings('A, , B,') == [('A', 'A'), ('B', 'B')]

    def test_source_order_is_preserved_and_duplicates_are_not_collapsed(self) -> None:
        """The LIST shape is load-bearing — each caller does its own projection.

        consumer_bindings' `_unused_bindings` must report a repeated binding
        twice; burndown's dict collapses last-wins; axis_labels' set dedupes.
        Collapsing here would take that choice away from all three.
        """
        assert destructure_bindings('A, B, A') == [
            ('A', 'A'), ('B', 'B'), ('A', 'A'),
        ]

    def test_the_two_projections_are_opposite_halves(self) -> None:
        """The `HistBar: HB` case both consumer suites hinge on."""
        pairs = destructure_bindings('{ StackedAreaChart, HistBar: HB }')

        assert [local for _canonical, local in pairs] == ['StackedAreaChart', 'HB']
        assert {canonical for canonical, _local in pairs} == {
            'StackedAreaChart', 'HistBar',
        }

    def test_destructure_re_matches_the_consumer_shape(self) -> None:
        src = "const { StackedAreaChart, HistBar: HB } = window.DF_CHARTS;"

        m = DF_CHARTS_DESTRUCTURE_RE.search(src)

        assert m is not None
        assert destructure_bindings(m.group(1)) == [
            ('StackedAreaChart', 'StackedAreaChart'), ('HistBar', 'HB'),
        ]

    def test_export_re_matches_the_provider_shape(self) -> None:
        src = "window.DF_CHARTS = { StackedAreaChart, LineChart };"

        m = DF_CHARTS_EXPORT_RE.search(src)

        assert m is not None
        assert {c for c, _ in destructure_bindings(m.group(1))} == {
            'StackedAreaChart', 'LineChart',
        }

    def test_a_nested_brace_yields_no_match(self) -> None:
        """The `[^{}]*` class is brace-HOSTILE by design; do not widen it.

        Three call sites turn this miss into a loud, self-naming failure that
        points at the nested brace.  Widening it would instead let the parser
        return a half-read binding list, and a sweep built on that list would
        go quietly wrong rather than loudly red.
        """
        nested = "const { StackedAreaChart, opts: { a: 1 } } = window.DF_CHARTS;"

        assert DF_CHARTS_DESTRUCTURE_RE.search(nested) is None

    def test_export_re_is_equally_brace_hostile(self) -> None:
        nested = "window.DF_CHARTS = { StackedAreaChart, opts: { a: 1 } };"

        assert DF_CHARTS_EXPORT_RE.search(nested) is None


class TestExtractDfDataBlock:
    """The `window.DF_DATA` seed-block extractor's contract.

    Deliberately pins the CURRENT behaviour of the three private copies this
    replaces (test_tab_escalations, test_tab_memory_evals,
    test_tab_escalation_analytics), whose code was byte-identical.  Two of
    those behaviours differ from its sibling `extract_function_body` and are
    kept as-is rather than "improved" in the same change that moves them: the
    silent `''` on a miss, and the string-literal blind spot.  Unifying and
    changing semantics at once is exactly what makes a consolidation unsafe.
    """

    def test_returns_the_block_with_both_braces(self) -> None:
        src = "window.DF_DATA = { ESCALATIONS: { open: 3 }, OTHER: 1 };"

        block = extract_df_data_block(src, 'ESCALATIONS')

        assert block == '{ open: 3 }'
        assert block.startswith('{') and block.endswith('}')

    def test_the_walk_is_brace_depth_aware(self) -> None:
        """A NESTED object does not terminate the block early.

        This is the whole reason a `[^}]*` regex was rejected: it would stop at
        the first nested `}` and silently truncate.
        """
        src = (
            "ESCALATIONS: { summary: { open: 3, closed: 1 }, rows: [] }, "
            "MEMORY_EVALS: { n: 9 }"
        )

        block = extract_df_data_block(src, 'ESCALATIONS')

        assert block == '{ summary: { open: 3, closed: 1 }, rows: [] }'
        assert 'MEMORY_EVALS' not in block, 'a later sibling key must be excluded'
        assert block.count('{') == block.count('}')

    def test_the_key_is_regex_escaped(self) -> None:
        """A key carrying regex metacharacters is matched literally."""
        src = "a.b: { x: 1 }, aXb: { x: 2 }"

        assert extract_df_data_block(src, 'a.b') == '{ x: 1 }'
        assert extract_df_data_block(src, 'a[b') == ''

    def test_arbitrary_whitespace_is_allowed_around_the_colon(self) -> None:
        src = "ESCALATIONS   :\n    {\n  open: 3\n}"

        block = extract_df_data_block(src, 'ESCALATIONS')

        assert block.startswith('{') and 'open: 3' in block

    def test_returns_empty_string_when_the_key_is_absent(self) -> None:
        """A miss is SILENT — the opposite policy to `extract_function_body`.

        Kept deliberately: all four call sites already assert on the returned
        value themselves, so raising here would merely relocate their failures.
        """
        assert extract_df_data_block('OTHER: { x: 1 }', 'ESCALATIONS') == ''

    def test_returns_empty_string_when_the_brace_is_never_closed(self) -> None:
        assert extract_df_data_block('ESCALATIONS: { open: 3', 'ESCALATIONS') == ''

    def test_is_re_entrant_over_its_own_output(self) -> None:
        """Feeding the result back in extracts a nested key.

        test_tab_escalations.py relies on exactly this: it pulls the
        ESCALATIONS seed block, then pulls `summary` back out of it.
        """
        src = "ESCALATIONS: { summary: { open: 3 }, rows: [] }, OTHER: 1"

        seed_block = extract_df_data_block(src, 'ESCALATIONS')
        summary_block = extract_df_data_block(seed_block, 'summary')

        assert summary_block == '{ open: 3 }'

    def test_brace_inside_a_string_literal_miscounts(self) -> None:
        """KNOWN LIMITATION, pinned as current behaviour, not endorsed.

        Unlike `extract_function_body`, this walk is NOT quote-aware: a `{` or
        `}` inside a quoted string is counted, so the block ends early.  Pinned
        so that making it quote-aware later is a deliberate, visible contract
        edit rather than silent drift.
        """
        src = 'ESCALATIONS: { label: "a } b", open: 3 }'

        block = extract_df_data_block(src, 'ESCALATIONS')

        assert block == '{ label: "a }', (
            'if this now returns the full block the walk became quote-aware — '
            'a real improvement, but update this pin deliberately'
        )
        assert 'open: 3' not in block


class TestScriptOrderHelpers:
    """The served-HTML script-order helper trio's contract.

    These three used to be copied verbatim into five modules
    (test_esc_flow_diagram, test_index_html, test_tab_escalation_analytics,
    test_tab_escalations, test_tab_memory_evals), so a fix to the false-pass
    guard had to be applied five times or not at all.  Everything pinned below
    was measured off those copies before the move, so the consolidation cannot
    silently change an outcome.

    The three guard phrases in particular are pinned VERBATIM because
    test_index_html.py's `_DEFERRED_CDN_CASES` / `_DEFERRED_TAB_TASKS_CASES`
    turn them into `pytest.raises(match=...)` patterns; a reworded message
    there would make those parametrizations match nothing.
    """

    # -- ScriptTagCollector ------------------------------------------------

    def test_collector_records_one_attrs_dict_per_script_start_tag(self) -> None:
        """One entry per <script> START tag, in document order."""
        collector = ScriptTagCollector()
        collector.feed(
            '<html><head>'
            '<script src="/a.js"></script>'
            '<script src="/b.js" defer></script>'
            '</head></html>'
        )

        assert [a.get('src') for a in collector.script_attrs] == ['/a.js', '/b.js']
        assert collector.script_attrs[1].get('defer') is not None or (
            'defer' in collector.script_attrs[1]
        ), 'valueless attributes must still be recorded as keys'

    def test_collector_counts_inline_scripts(self) -> None:
        """An inline (src-less) <script> still consumes an index.

        Load-order positions are indices into this list, so a tag that did not
        consume one would shift every later position and could invert a
        comparison.
        """
        collector = ScriptTagCollector()
        collector.feed(
            '<script src="/a.js"></script>'
            '<script>window.X = 1;</script>'
            '<script src="/b.js"></script>'
        )

        assert len(collector.script_attrs) == 3
        assert 'src' not in collector.script_attrs[1]
        assert [a.get('src') for a in collector.script_attrs] == [
            '/a.js', None, '/b.js',
        ]

    def test_collector_ignores_non_script_tags(self) -> None:
        collector = ScriptTagCollector()
        collector.feed('<link rel="stylesheet" href="/a.css"><div><p>hi</p></div>')

        assert collector.script_attrs == []

    # -- find_script_position ----------------------------------------------

    def test_find_returns_index_and_attrs_for_the_first_prefix_match(self) -> None:
        """`(index, attrs)` for the FIRST tag whose src startswith the prefix."""
        body = (
            '<script src="/static/vendor/react.js"></script>'
            '<script src="/static/redux/store.js"></script>'
            '<script src="/static/redux/tabs.jsx"></script>'
        )

        result = find_script_position(body, '/static/redux/')

        assert result is not None
        index, attrs = result
        assert index == 1, 'index is the 0-based position in document order'
        assert attrs.get('src') == '/static/redux/store.js'

    def test_find_index_counts_inline_scripts(self) -> None:
        """The index is a document-order position over ALL script tags."""
        body = (
            '<script>window.DF = {};</script>'
            '<script src="/static/app.js"></script>'
        )

        result = find_script_position(body, '/static/app.js')

        assert result is not None
        assert result[0] == 1

    def test_find_returns_none_when_no_tag_matches(self) -> None:
        body = '<script src="/static/app.js"></script>'

        assert find_script_position(body, '/static/missing') is None

    def test_find_treats_a_srcless_tag_as_empty_string(self) -> None:
        """No `src` coerces to `''`, so it matches only an empty prefix.

        The `or ''` is what stops `None.startswith` blowing up on an inline
        script, and the empty string only ever prefix-matches `''`.
        """
        body = '<script>window.X = 1;</script>'

        assert find_script_position(body, '/static/') is None

        result = find_script_position(body, '')
        assert result is not None
        assert result[0] == 0

    # -- assert_script_loads_before ----------------------------------------

    def test_passes_when_before_precedes_after(self) -> None:
        body = (
            '<script src="/static/vendor/react.js"></script>'
            '<script src="/static/redux/tabs.jsx"></script>'
        )

        assert_script_loads_before(
            body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
        )

    def test_raises_naming_the_missing_before_prefix(self) -> None:
        body = '<script src="/static/redux/tabs.jsx"></script>'

        with pytest.raises(AssertionError, match=r'No <script src="/static/vendor/'):
            assert_script_loads_before(
                body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
            )

    def test_raises_distinctly_when_the_after_tag_is_absent(self) -> None:
        """A missing AFTER tag gets its own message, not the missing-BEFORE one."""
        body = '<script src="/static/vendor/react.js"></script>'

        with pytest.raises(AssertionError) as exc:
            assert_script_loads_before(
                body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
            )

        message = str(exc.value)
        assert 'cannot verify load-order invariant for react' in message
        assert not message.startswith('No <script src=')

    def test_raises_on_out_of_order_pair(self) -> None:
        body = (
            '<script src="/static/redux/tabs.jsx"></script>'
            '<script src="/static/vendor/react.js"></script>'
        )

        with pytest.raises(AssertionError, match=r'must load\s+BEFORE'):
            assert_script_loads_before(
                body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
            )

    def test_out_of_order_message_interpolates_the_consumer_note(self) -> None:
        """The ordering failure carries the caller's `consumer_note`.

        This is the ONE place the five copies diverged — test_index_html.py
        alone dropped the note for a fixed two-sentence string.  The unified
        helper adopts the canonical 4-of-5 form, which makes the note visible
        at index_html's 13 note-passing call sites instead of discarding the
        notes at the other four modules' 14 sites.  No test anywhere matches
        the ordering message, so either variant satisfies the suite; this one
        strictly carries more information.
        """
        body = (
            '<script src="/static/redux/tabs.jsx"></script>'
            '<script src="/static/vendor/react.js"></script>'
        )
        note = 'tabs.jsx dereferences React at module scope.'

        with pytest.raises(AssertionError) as exc:
            assert_script_loads_before(
                body,
                '/static/vendor/',
                '/static/redux/',
                'react',
                'tabs',
                note,
            )

        assert note in str(exc.value)

    # -- the defer/async/type=module false-pass guard ----------------------

    @pytest.mark.parametrize('offender', ['before', 'after'])
    def test_defer_on_either_tag_trips_the_guard(self, offender: str) -> None:
        """`defer` divorces document order from execution order on EITHER tag."""
        before_attr = ' defer' if offender == 'before' else ''
        after_attr = ' defer' if offender == 'after' else ''
        body = (
            f'<script src="/static/vendor/react.js"{before_attr}></script>'
            f'<script src="/static/redux/tabs.jsx"{after_attr}></script>'
        )

        with pytest.raises(AssertionError, match='defer attribute'):
            assert_script_loads_before(
                body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
            )

    @pytest.mark.parametrize('offender', ['before', 'after'])
    def test_async_on_either_tag_trips_the_guard(self, offender: str) -> None:
        before_attr = ' async' if offender == 'before' else ''
        after_attr = ' async' if offender == 'after' else ''
        body = (
            f'<script src="/static/vendor/react.js"{before_attr}></script>'
            f'<script src="/static/redux/tabs.jsx"{after_attr}></script>'
        )

        with pytest.raises(AssertionError, match='async attribute'):
            assert_script_loads_before(
                body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
            )

    @pytest.mark.parametrize('offender', ['before', 'after'])
    def test_type_module_on_either_tag_trips_the_guard(self, offender: str) -> None:
        before_attr = ' type="module"' if offender == 'before' else ''
        after_attr = ' type="module"' if offender == 'after' else ''
        body = (
            f'<script src="/static/vendor/react.js"{before_attr}></script>'
            f'<script src="/static/redux/tabs.jsx"{after_attr}></script>'
        )

        with pytest.raises(
            AssertionError, match=r'type="module".*deferred by default'
        ):
            assert_script_loads_before(
                body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
            )

    def test_type_is_compared_case_insensitively(self) -> None:
        """`type="MODULE"` is the same module semantics; the guard lowercases."""
        body = (
            '<script src="/static/vendor/react.js" type="MODULE"></script>'
            '<script src="/static/redux/tabs.jsx"></script>'
        )

        with pytest.raises(AssertionError, match='deferred by default'):
            assert_script_loads_before(
                body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
            )

    def test_a_non_module_type_does_not_trip_the_guard(self) -> None:
        """`type="text/javascript"` is a classic script — order still holds."""
        body = (
            '<script src="/static/vendor/react.js" type="text/javascript"></script>'
            '<script src="/static/redux/tabs.jsx"></script>'
        )

        assert_script_loads_before(
            body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
        )

    def test_the_guard_runs_before_the_order_comparison(self) -> None:
        """A deferred pair fails with the GUARD message even when in order.

        Otherwise a `defer` regression would pass silently: document order
        would still be right while execution order was not.
        """
        body = (
            '<script src="/static/vendor/react.js" defer></script>'
            '<script src="/static/redux/tabs.jsx"></script>'
        )

        with pytest.raises(AssertionError, match='defer attribute'):
            assert_script_loads_before(
                body, '/static/vendor/', '/static/redux/', 'react', 'tabs',
            )


class TestSharedServedAssetFixtures:
    """The `_client` and served-asset fixtures resolve from conftest.py.

    This module defines none of them locally, so these tests can only pass
    once conftest owns them — which is the point: nine modules used to carry
    byte-identical copies.
    """

    @pytest.mark.parametrize(
        ('name', 'path'),
        sorted(_SHARED_ASSET_FIXTURES.items()),
        ids=sorted(_SHARED_ASSET_FIXTURES),
    )
    def test_each_shared_fixture_serves_what_the_app_serves(
        self, request, _client, name: str, path: str
    ) -> None:
        """Every row resolves, is served with HTTP 200, and matches that response.

        THE STATUS CHECK IS THE LOAD-BEARING ONE — do not weaken it back to a
        non-emptiness check.  A fixture wired to a mistyped path does not yield
        an EMPTY body: it yields the app's 404 body, `{"detail":"Not Found"}`,
        which is 22 non-empty characters and compares equal to itself.  So both
        `assert body` and `body == _client.get(path).text` pass for a mis-wired
        row, and every downstream absence assertion built on that fixture then
        passes vacuously against a 22-character JSON blob.  `status_code == 200`
        is the only one of these assertions a path typo cannot satisfy.

        Non-emptiness is still asserted, but for a DIFFERENT failure that the
        status cannot catch: an asset genuinely served, yet empty.

        PARAMETRIZED, not a loop over the table: two mis-wired rows must report
        as two failures.  Inside one test the first would mask the rest and the
        second typo would survive the fix of the first.
        """
        body = request.getfixturevalue(name)
        # One request per row, reused by both assertions below.
        resp = _client.get(path)

        assert isinstance(body, str), f'fixture {name!r} must yield the response TEXT'
        assert resp.status_code == 200, (
            f'fixture {name!r} is wired to {path}, which the app does not '
            f'serve (HTTP {resp.status_code}) — its consumers would be probing '
            f'a {len(resp.text)}-character error body, so every absence '
            f'assertion built on it would pass vacuously'
        )
        assert body, (
            f'fixture {name!r} is empty — {path} is served, but with no '
            f'content, so every probe built on it would pass vacuously'
        )
        assert body == resp.text, (
            f'fixture {name!r} does not serve {path} — it is wired to the '
            f'wrong asset, so its consumers are asserting against the wrong file'
        )

    def test_every_conftest_body_fixture_has_a_row(self) -> None:
        """`_SHARED_ASSET_FIXTURES` must mirror conftest exactly, both ways.

        Without this the table is a hand-maintained copy of conftest, and the
        failure it is meant to make impossible reappears one level up: a tenth
        `*_body` fixture added to conftest and not listed here is served to its
        consumers with NOTHING checking that it is wired to the asset its name
        claims — silently uncovered by the very test that exists to cover it.

        The other direction is checked by the same equality: a row naming a
        fixture conftest no longer defines would otherwise error only as an
        obscure fixture-lookup failure in the parametrized test above.
        """
        rows = set(_SHARED_ASSET_FIXTURES)
        defined = _conftest_body_fixture_names()

        assert rows == defined, (
            f'the shared-asset table and conftest have drifted apart — '
            f'in conftest but unlisted here (so UNCOVERED): '
            f'{sorted(defined - rows) or "none"}; listed here but not defined '
            f'in conftest: {sorted(rows - defined) or "none"}'
        )

    def test_client_is_module_scoped_not_function_scoped(self, request) -> None:
        """One app lifespan per consuming MODULE, not per test.

        The nine local copies being retired are all `scope='module'`, while
        conftest's pre-existing `client` fixture is function-scoped.  Silently
        satisfying `_client` from a function-scoped definition would stand up
        and tear down `TestClient(app)` once per test across a ~1900-test
        suite — correct, but a large and gratuitous slowdown that nothing
        would report.

        PRIVATE-API PIN, DEGRADING TO XFAIL.  Every route to a fixture's scope
        is pytest-internal (see `_resolved_client_scope`), so a pytest bump can
        move all of them.  That is not a defect in this repo and must not turn
        the suite red for whoever does the bump — but it must not silently stop
        checking either, so it xfails with instructions instead of skipping.
        """
        scope = _resolved_client_scope(request)

        if scope is None:
            pytest.xfail(
                'pytest moved every known route to a fixture definition '
                '(FixtureManager.getfixturedefs and the fixture-marker '
                'attribute) — re-verify this pin against the new API and update '
                '_resolved_client_scope; `_client` is UNCHECKED until then'
            )

        assert scope == 'module', (
            f'`_client` is {scope}-scoped; it must be module-scoped so each '
            f'consuming module pays exactly one app lifespan'
        )


class TestNestedDeclarationAgainstRealSource:
    """The charter's "known limitation", pinned against the served asset.

    This may already pass the moment `extract_function_body` exists — the
    brace-walk variant never had the limitation.  Its job is not to catch a
    bug today but to make the newly-acquired capability an explicit,
    permanent contract: the retired variant's line-anchored
    `^function NAME\\(` regex plus slice-to-next-top-level-`function` looks
    like a simpler spelling of the same thing, and a future simplification
    back to it must fail here rather than quietly re-lose the ability to
    scope an indented declaration.

    Ground truth in tab_tasks.jsx: `function statusMatches(s) {` is declared
    INSIDE `TasksTab`, and `flipFilter` is a sibling const in the same
    enclosing function — so a fallback to the enclosing body is detectable.
    """

    def test_status_matches_is_scoped_out_of_its_enclosing_component(
        self, tab_tasks_jsx_body: str
    ) -> None:
        body = extract_function_body(tab_tasks_jsx_body, 'statusMatches')

        assert body, 'the nested `statusMatches` declaration could not be sliced out'
        for status in ('in-progress', 'blocked', 'merge-deferred'):
            assert status in body, (
                f'{status!r} is missing from the extracted `statusMatches` body — '
                f'either the status disjunction changed or the extractor returned '
                f'the wrong slice'
            )
        assert 'flipFilter' not in body, (
            'the extracted body contains `flipFilter`, a sibling const declared in '
            'the ENCLOSING `TasksTab` — so the extractor fell back to the whole '
            'enclosing component instead of scoping to the nested declaration, '
            'which is exactly the limitation this task fixed'
        )
