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
"""

from __future__ import annotations

import pytest
from _dashboard_helpers import extract_function_body


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
