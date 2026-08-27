"""AST detector for MCP URLs built with a trailing slash.

The fused-memory server 307-redirects ``/mcp/`` to ``/mcp``. A bare
``httpx.AsyncClient`` does not follow that redirect, so a POST to the slashed
form is a perfectly successful HTTP exchange that discards the payload. The
sweep guard in ``test_mcp_post_transport.py`` exists to stop a future call site
from reintroducing the slash; this module is how it looks.

WHY PARSING, NOT GREP. The guard originally matched the substring ``}/mcp/'``.
A bare ``/mcp/`` was rejected up front because it also matches unrelated prose
in ``merge_queue.py`` docstrings (``scheduler/mcp/usage_gate/cost_store`` at
:1163 and :2467), which would have made the assertion permanently
unsatisfiable — a correct fix could never turn it green. But the narrower
anchor bought that immunity by hard-coding one spelling of one construct: it
required a closing brace immediately before the slash and a single quote
immediately after, so the double-quoted, plain-literal, percent-format and
concatenation spellings of the identical defect all passed silently.
``fused-memory/src/fused_memory/server/main.py::run_server`` carries the
percent-format spelling today, and the old anchor did not see it.

Matching on the parsed literal's TAIL gets both properties at once: it catches
every spelling because it looks at the value rather than the syntax, and it is
immune to the docstring prose because that string merely *contains* ``/mcp/``
without *ending* in it.

"EVERY SPELLING" IS A CLAIM THAT HAS TO BE EARNED, and the first cut did not
earn it. ``'%s/mcp/' % base`` — percent formatting via the OPERATOR rather than
deferred to ``logger``'s own args — returned no hits, because two halves
disagreed: the suppression pass treated every ``BinOp`` as owning its
children's tails, while ``_literal_tail`` modelled only ``Add``. The site was
suppressed by one half and unmodelled by the other, which is the same
one-spelling-hard-coded hole this module was written to close. Both halves now
name the operators they mean (see :func:`_owns_child_tail`), so an unmodelled
operator can only ever cause a DOUBLE report, never a miss.

MEASURED, against real historical source rather than a fabricated sample. Run
over ``git show 2633a244a6:<path>`` — the commit this task branched from — the
detector fires on exactly the 8 pre-fix defect sites and nothing else:
``orchestrator/src/orchestrator/workflow.py`` 15173, 15215, 15247, 15286;
``orchestrator/src/orchestrator/merge_queue.py`` 16031;
``scripts/migrate_metadata_modules_to_files.py`` 86;
``scripts/trial_module_tagger_haiku.py`` 658;
``fused-memory/scripts/strip_leaked_control_keys.py`` 89. Run over the
post-fix tree across the seven sweep directories it flags only the two
deliberately-excluded lines (``reconciliation/stages/base.py::_build_mcp_config``,
an MCP config entry for the Claude CLI's own redirect-following client, and two
display/log strings in ``server/main.py::run_server``). Non-vacuous in both
directions, which is the property the guard needs and the property a
constant-pinned-against-itself meta-test cannot have.

A ``_``-prefixed, uniquely-named sibling module (like
``_verify_config_corpus.py``, ``_orch_helpers.py``): ``conftest.py`` inserts
``_TESTS_DIR`` on ``sys.path`` at import time, which is what makes a bare
``from _mcp_url_scan import ...`` resolve. It is NOT in ``conftest.py``
because non-fixture helpers imported from a conftest collide across sibling
subprojects under ``sys.modules['conftest']`` — see that file's docstring.
"""

from __future__ import annotations

import ast

__all__ = ['SLASHED_TAIL', 'find_trailing_slash_mcp_urls']

#: The tail that makes a URL get redirected. Compared against the parsed
#: literal's value, never searched for in raw source text.
SLASHED_TAIL = '/mcp/'


def _literal_tail(node: ast.expr) -> str:
    """Return the trailing string literal *node* evaluates to end with, or ``''``.

    Only the tail matters: a URL is redirected because of how it ENDS, so a
    string whose slash sits anywhere else (docstring prose, a path fragment in
    the middle of a sentence) is correctly invisible here.

    ``''`` means "no statically-known tail" — an f-string ending in a
    ``{placeholder}`` cannot end in the slash no matter what the placeholder
    holds, and anything this function does not model is treated the same way.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else ''
    if isinstance(node, ast.JoinedStr):
        # The f-string's own trailing element. If it is a FormattedValue the
        # string ends with an interpolation, so no static tail exists.
        if node.values and isinstance(node.values[-1], ast.Constant):
            value = node.values[-1].value
            return value if isinstance(value, str) else ''
        return ''
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Add):
            # ``base + '/mcp/'`` — concatenation's tail is its right operand's.
            return _literal_tail(node.right)
        if isinstance(node.op, ast.Mod):
            # ``'%s/mcp/' % base`` — percent-format interpolates INTO the left
            # operand, so the tail is the left operand's unless the template
            # itself ends in a placeholder.  ``'%s/mcp/' % base`` ends in the
            # slash for the same reason ``f'{base}/mcp/'`` does, and the
            # spelling is live in the tree today
            # (``fused_memory/server/main.py::run_server`` defers it to logging).
            return _literal_tail(node.left)
    return ''


def _owns_child_tail(node: ast.AST) -> bool:
    """True when :func:`_literal_tail` derives *node*'s tail from a child's.

    Exactly the node kinds whose descendants must be suppressed so a site is
    reported once, at the outermost expression that owns it.  Deliberately NOT
    "every ``BinOp``": under that broader rule an unmodelled operator (say
    ``'/mcp/' * n``) suppressed its own operands while contributing no tail of
    its own, so the site vanished from BOTH ends — suppressed by one half,
    unmodelled by the other.  That is precisely the hole the ``Mod`` spelling
    fell through.
    """
    if isinstance(node, ast.JoinedStr):
        return True
    return isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod))


def find_trailing_slash_mcp_urls(source: str, *, filename: str) -> list[tuple[int, str]]:
    """Return ``(lineno, stripped_source_line)`` for every ``/mcp/``-tailed literal.

    Raises ``AssertionError`` naming *filename* if *source* does not parse.
    Swallowing ``SyntaxError`` into an empty result is the one failure mode
    that would let the sweep guard silently stop covering a file while staying
    green, so every failure at this boundary IS a finding — the same reasoning
    ``_verify_config_corpus.load_config_scalar`` re-raises on.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        raise AssertionError(
            f'cannot parse {filename} while sweeping for trailing-slash MCP URLs: '
            f'{exc.__class__.__name__}: {exc}.\n'
            f'FIX: repair the file. An unparseable file is a real failure, not an '
            f'absent one — a scanner that skipped it would keep reporting a clean '
            f'sweep while silently covering one file less.'
        ) from exc

    # Report each site ONCE, at the outermost expression that owns the tail.
    # A naive walk matches both a JoinedStr and its inner Constant('/mcp/'),
    # and both a BinOp and its right operand. De-duplicating by line does not
    # fix it: implicit concatenation puts an f-string's JoinedStr and its
    # trailing Constant on DIFFERENT lines, so the duplicate has to be
    # prevented structurally.
    #
    # ONE pass, not a nested ``ast.walk`` per candidate: ``ast.walk`` is
    # breadth-first (it popleft()s a deque), so a parent is always visited
    # before its children and suppression can simply be propagated down one
    # generation at a time. The sweep guard parses ~500 files, so the
    # quadratic re-walk this replaces was the dominant cost of the whole test.
    nested: set[int] = set()
    for node in ast.walk(tree):
        # A suppressed node's descendants are suppressed too; a tail-owning
        # node's children are suppressed because the PARENT reports the site.
        if id(node) in nested or _owns_child_tail(node):
            for child in ast.iter_child_nodes(node):
                nested.add(id(child))

    lines = source.splitlines()
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in nested:
            continue
        if not isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            continue
        if not _literal_tail(node).endswith(SLASHED_TAIL):
            continue
        lineno = node.lineno
        text = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ''
        hits.append((lineno, text))

    return sorted(hits)
