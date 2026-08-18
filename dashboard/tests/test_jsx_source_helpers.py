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

import pytest
from _dashboard_helpers import extract_function_body, strip_js_comments


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

    def test_missing_function_raises_with_a_naming_diagnostic(self) -> None:
        """A miss is LOUD, and the message names the function and the known limits.

        Returning `''` here (the retired behaviour) made every downstream
        absence assertion pass vacuously — a permanent false GREEN that no
        amount of downstream care can detect.
        """
        src = 'function Foo(a) { const x = 1; }'

        with pytest.raises(AssertionError, match='NotDeclared'):
            extract_function_body(src, 'NotDeclared')

        with pytest.raises(AssertionError, match='arrow function'):
            extract_function_body(src, 'NotDeclared')

        with pytest.raises(AssertionError, match='class method'):
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
# a one-line fixture in conftest.py.
SHARED_ASSET_FIXTURES = {
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


class TestSharedServedAssetFixtures:
    """The `_client` and served-asset fixtures resolve from conftest.py.

    This module defines none of them locally, so these tests can only pass
    once conftest owns them — which is the point: nine modules used to carry
    byte-identical copies.
    """

    def test_each_shared_fixture_serves_what_the_app_serves(self, request, _client) -> None:
        """Every row resolves, is non-empty, and matches the live response.

        Non-empty is asserted separately from the equality: a fixture wired to
        a mistyped path would return a 404 body, and `body == _client.get(...)`
        would compare that 404 to itself and pass.
        """
        for name, path in SHARED_ASSET_FIXTURES.items():
            body = request.getfixturevalue(name)

            assert isinstance(body, str), f'fixture {name!r} must yield the response TEXT'
            assert body, (
                f'fixture {name!r} is empty — {path} is probably not served under '
                f'that exact path, and every probe built on it would then pass '
                f'vacuously'
            )
            assert body == _client.get(path).text, (
                f'fixture {name!r} does not serve {path} — it is wired to the '
                f'wrong asset, so its consumers are asserting against the wrong file'
            )

    def test_client_is_module_scoped_not_function_scoped(self, request) -> None:
        """One app lifespan per consuming MODULE, not per test.

        The nine local copies being retired are all `scope='module'`, while
        conftest's pre-existing `client` fixture is function-scoped.  Silently
        satisfying `_client` from a function-scoped definition would stand up
        and tear down `TestClient(app)` once per test across a ~1900-test
        suite — correct, but a large and gratuitous slowdown that nothing
        would report.

        PRIVATE-API PIN.  `FixtureManager.getfixturedefs` is not public API,
        so the lookup is asserted to have RESOLVED before its result is read:
        a moved attribute must fail loudly here rather than degrade into a
        skipped assertion, which would leave the scope unchecked forever.
        """
        fixturedefs = request._fixturemanager.getfixturedefs('_client', request.node)

        assert fixturedefs, (
            'could not resolve a `_client` fixture definition — either conftest '
            'no longer defines it, or pytest moved '
            'FixtureManager.getfixturedefs and this pin needs re-verifying'
        )
        # Least-specific first, so the definition actually in effect is last.
        assert fixturedefs[-1].scope == 'module', (
            f'`_client` is {fixturedefs[-1].scope}-scoped; it must be module-scoped '
            f'so each consuming module pays exactly one app lifespan'
        )
