"""FalkorDB index normal form: what SHOULD exist, and what DOES (task 3706, α).

Why this module exists
----------------------
Index provisioning needs to compare two things that arrive in completely
different shapes: the CREATE statements graphiti emits, and the records
``CALL db.indexes()`` returns.  Comparing them ad hoc is how indices go missing
without anyone noticing.  This module defines a single **normal form** and maps
both sides onto it::

    IndexSpec = (label, entity_type, field, index_type)
                 entity_type in {'NODE', 'RELATIONSHIP'}
                 index_type  in {'RANGE', 'FULLTEXT'}

* EXPECTED side — :func:`expected_index_set` parses what graphiti itself emits.
* ACTUAL side — :func:`normalize_index_records` projects ``list_indices()``
  records onto the same form.

A set difference in either direction is then a plain set operation, and a
multi-property index can no longer be misread as "missing" because one side
carried a list where the other carried a scalar.

The four measured FalkorDB statement forms
------------------------------------------
Measured verbatim 2026-08-06 against the installed **graphiti-core 0.28.2**
(pinned open-ended at ``pyproject.toml`` as ``graphiti-core[falkordb]>=0.28.1``)::

    R-node   CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)
    R-edge   CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid, e.group_id, ...)
    FT-node  CALL db.idx.fulltext.createNodeIndex({label: 'Entity', stopwords: [...]},
                                                  'name', 'summary', 'group_id')
    FT-edge  CREATE FULLTEXT INDEX FOR ()-[e:RELATES_TO]-() ON (e.name, e.fact, e.group_id)

No FalkorDB form carries ``IF NOT EXISTS`` (it is a FalkorDB syntax error) and
none carries an index-name token.

Why the strict parse is load-bearing
------------------------------------
``GraphProvider`` is a plain ``Enum``, so ``GraphProvider.FALKORDB == 'falkordb'``
is **False**, and ``get_range_indices('falkordb')`` returns the 27-statement
**NEO4J** set with no error at all — the type annotation is not enforced at
runtime.  The regexes below deliberately reject an index-name token and
``IF NOT EXISTS``, so every neo4j statement is unparseable.  That turns the
plain-Enum gotcha into a loud failure: if a future edit regressed to the bare
string, the parse would blow up rather than silently yield the wrong expected
set.  Do not relax the patterns to "be tolerant" — the intolerance IS the check.

This module deliberately imports nothing from ``graphiti_client`` so the rules
stay unit-testable with a plain import and no client/LLM/embedder stack.  That
isolation is also a HAZARD control: ``FalkorDriver.__init__`` fire-and-forgets
``build_indices_and_constraints()`` when an event loop is running, so a module
that could pull a driver into scope risks creating indices merely by being
imported — destroying esc-3375-1's protected evidence (the current absence of
indices).  Importing this module performs zero I/O.
"""

from __future__ import annotations

import re

# The PRD normal form: (label, entity_type, field, index_type).
IndexSpec = tuple[str, str, str, str]

#: The graphiti-core release these patterns were measured against.  Named in
#: failure messages so an operator can tell a version drift apart from a
#: provider mix-up without going digging.
MEASURED_GRAPHITI_VERSION = '0.28.2'


class UnparsedIndexStatementError(ValueError):
    """A statement from graphiti's index generators did not match any known form.

    This RAISES rather than warning-and-skipping, on purpose.  The two sides of
    the seam fail in opposite directions, so they want opposite defaults, and
    this is the side where silence is dangerous: a skipped statement makes the
    expected set SHORT, the diff then reports "nothing missing", and an index
    that was never created is quietly blessed — precisely the failure mode this
    PRD exists to remove (PRD Open Question 1, closed as "fail"; INV-4).

    The message embeds the offending statement verbatim (INV-2 — emit the fact
    the caller actually has) plus the graphiti-core version the patterns were
    measured against, so the reader can immediately tell "graphiti changed its
    syntax" apart from "the wrong provider was passed".
    """


# --- Statement patterns ----------------------------------------------------
#
# Both patterns admit an optional FULLTEXT keyword so the node/edge matcher is
# parameterised on index type rather than duplicated -- a future node-side
# `CREATE FULLTEXT INDEX FOR (n:L) ON (n.p)` is then handled uniformly.
#
# Neither admits an index-name token between INDEX and FOR, nor `IF NOT EXISTS`.
# That rejection is the tripwire documented in the module docstring; it is not
# an oversight.

_RANGE_OR_FULLTEXT_NODE_RE = re.compile(
    r'^\s*CREATE\s+(?P<fulltext>FULLTEXT\s+)?INDEX\s+'
    r'FOR\s*\(\s*(?P<var>\w+)\s*:\s*(?P<label>\w+)\s*\)\s*'
    r'ON\s*\((?P<props>[^)]*)\)\s*$',
    re.IGNORECASE | re.DOTALL,
)

_RANGE_OR_FULLTEXT_EDGE_RE = re.compile(
    r'^\s*CREATE\s+(?P<fulltext>FULLTEXT\s+)?INDEX\s+'
    r'FOR\s*\(\s*\)\s*-\s*\[\s*(?P<var>\w+)\s*:\s*(?P<label>\w+)\s*\]\s*-\s*\(\s*\)\s*'
    r'ON\s*\((?P<props>[^)]*)\)\s*$',
    re.IGNORECASE | re.DOTALL,
)


def _fail(statement: str, why: str) -> UnparsedIndexStatementError:
    """Build the loud error, naming the statement verbatim and the measured version."""
    return UnparsedIndexStatementError(
        f'Unparsed FalkorDB index statement ({why}). Refusing to skip it: a '
        'skipped statement silently shortens the expected index set and turns a '
        'missing index into "nothing to do". Patterns were measured against '
        f'graphiti-core {MEASURED_GRAPHITI_VERSION}; if graphiti changed its '
        'syntax, extend parse_index_statement. If the statement carries an '
        "index name or 'IF NOT EXISTS' it is the NEO4J form, which means a "
        "bare 'falkordb' string reached get_range_indices instead of "
        f'GraphProvider.FALKORDB.\nStatement: {statement}'
    )


_FULLTEXT_CALL_PREFIX = 'CALL db.idx.fulltext.createNodeIndex('
_CONFIG_LABEL_RE = re.compile(r"\blabel\s*:\s*'(?P<label>[^']*)'")
_QUOTED_ARG_RE = re.compile(r"'([^']*)'")


def _parse_fulltext_call(statement: str) -> list[IndexSpec]:
    """Parse ``CALL db.idx.fulltext.createNodeIndex(<config-map>, 'p1', 'p2', ...)``.

    The config map is located by BRACE MATCHING rather than by a regex, and the
    property arguments are taken only from AFTER its closing ``}``.  That split
    point is the whole trick: the map embeds a 33-word quoted, comma-separated
    ``stopwords`` list, so any pattern that scans the statement as a whole picks
    the stopwords up as properties — roughly 36 bogus entries, each of which
    would be reported as a permanently missing index that can never be created.
    """
    open_idx = statement.find('{')
    if open_idx == -1:
        raise _fail(statement, 'fulltext CALL has no { config map')

    depth = 0
    close_idx = -1
    for i in range(open_idx, len(statement)):
        if statement[i] == '{':
            depth += 1
        elif statement[i] == '}':
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx == -1:
        raise _fail(statement, 'fulltext CALL config map has unbalanced braces')

    config = statement[open_idx:close_idx + 1]
    label_match = _CONFIG_LABEL_RE.search(config)
    if label_match is None or not label_match.group('label'):
        raise _fail(statement, "fulltext CALL config map has no non-empty 'label:' key")

    # Properties come strictly from AFTER the config map -- never from inside it.
    tail = statement[close_idx + 1:]
    props = [p for p in _QUOTED_ARG_RE.findall(tail) if p]
    if not props:
        raise _fail(statement, 'fulltext CALL listed no property arguments after the config map')

    label = label_match.group('label')
    return [(label, 'NODE', prop, 'FULLTEXT') for prop in props]


def _split_properties(statement: str, var: str, props: str) -> list[str]:
    """Split an ``ON (n.a, n.b)`` property clause into bare property names."""
    names: list[str] = []
    for raw in props.split(','):
        token = raw.strip()
        if not token:
            continue
        prefix = f'{var}.'
        if not token.startswith(prefix):
            raise _fail(statement, f'property {token!r} is not prefixed with {prefix!r}')
        name = token[len(prefix):].strip()
        if not name:
            raise _fail(statement, f'empty property name in {token!r}')
        names.append(name)
    if not names:
        raise _fail(statement, 'the ON (...) clause listed no properties')
    return names


def parse_index_statement(statement: str) -> list[IndexSpec]:
    """Fan one graphiti index statement out to one :data:`IndexSpec` per property.

    Args:
        statement: A single CREATE/CALL statement as emitted by
            ``graphiti_core.graph_queries`` for ``GraphProvider.FALKORDB``.

    Returns:
        One spec per indexed property.  Never an empty list — a statement is
        either fanned out or raises.

    Raises:
        UnparsedIndexStatementError: The statement matched no known FalkorDB
            form.  See the class docstring for why this is not a warn-and-skip.
    """
    # Statement-form dispatch is exhaustive-with-else-raise: no branch may fall
    # through to an empty return (see UnparsedIndexStatementError).
    if statement.lstrip().startswith(_FULLTEXT_CALL_PREFIX):
        return _parse_fulltext_call(statement)

    for pattern, entity_type in (
        (_RANGE_OR_FULLTEXT_NODE_RE, 'NODE'),
        (_RANGE_OR_FULLTEXT_EDGE_RE, 'RELATIONSHIP'),
    ):
        match = pattern.match(statement)
        if match is None:
            continue
        index_type = 'FULLTEXT' if match.group('fulltext') else 'RANGE'
        label = match.group('label')
        props = _split_properties(statement, match.group('var'), match.group('props'))
        return [(label, entity_type, prop, index_type) for prop in props]

    raise _fail(statement, 'matched no known FalkorDB index form')
