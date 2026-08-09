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

Why the two sides differ in strictness (and why that is not a fail-soft)
-----------------------------------------------------------------------
The expected side RAISES on anything it does not recognise; the actual side
PROJECTS onto the normal form and emits no tuple for an index type the form
cannot represent (today: ``VECTOR``).  They fail in opposite directions, so they
want opposite defaults:

* An unparsed UPSTREAM statement makes the expected set SHORT — under-provisioning,
  this PRD's own failure mode recurring.  Must be loud.
* An ACTUAL index the normal form cannot represent is not a surprise and not drift
  to repair.  The PRD defines ``index_type in {RANGE, FULLTEXT}``, VECTOR indices
  legitimately exist on real graphs (``reindex.py`` drops them), and the PRD says
  ``unexpected`` is reported but never acted on.  Raising there would turn the
  detector into a false-alarm generator on every real graph.

The projection is NOT a blanket skip.  Every genuine SHAPE surprise still raises
:class:`IndexRecordShapeError`: a ``label`` that is not a non-empty string, an
``entity_type`` outside ``{NODE, RELATIONSHIP}``, a ``field`` that is neither a
string nor a non-empty list, or a property present in ``field`` with no entry in
``type``.  The unrepresentable index TYPE is the *only* thing projected away;
every way a record can be structurally wrong is loud, because each of those means
a column was bound to the wrong thing — the defect this module was written on.

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
from collections.abc import Sequence

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices

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


def expected_index_set() -> set[IndexSpec]:
    """The SINGLE home for "what indices should exist" on a FalkorDB graph.

    Derived by parsing what graphiti itself emits, never by restating it (INV-5,
    PRD D3).  β (``ensure_indices``) and δ (``summarize_index_health``) both
    consume this rather than keeping their own copy, so a graphiti upgrade that
    changes the index set is picked up automatically, and one that changes the
    statement SYNTAX fails loudly via :class:`UnparsedIndexStatementError`
    instead of silently shortening the set.

    Deliberately PARAMETERLESS and hard-bound to ``GraphProvider.FALKORDB``.
    MEASURED 2026-08-06: ``GraphProvider`` is a plain ``Enum``, so
    ``GraphProvider.FALKORDB == 'falkordb'`` is **False** and
    ``get_range_indices('falkordb')`` returns 27 NEO4J statements with **no
    error** — the ``(provider: GraphProvider)`` annotation is not enforced at
    runtime.  A ``provider=`` argument would therefore exist only to let a caller
    pass the one value that silently produces the wrong answer.  This repo
    targets FalkorDB, so the parameter buys nothing and costs a live footgun.

    Results are intentionally NOT cached: the call is a handful of regex passes,
    and a cached set would complicate a graphiti upgrade taking effect.

    Returns:
        Every ``(label, entity_type, field, index_type)`` that should exist.

    Raises:
        UnparsedIndexStatementError: graphiti emitted a statement form this
            module does not recognise — see the class docstring.
    """
    statements = list(get_range_indices(GraphProvider.FALKORDB)) + list(
        get_fulltext_indices(GraphProvider.FALKORDB)
    )
    return {spec for statement in statements for spec in parse_index_statement(statement)}


# --- The ACTUAL side: the CALL db.indexes() header -------------------------


class IndexHeaderShapeError(ValueError):
    """``CALL db.indexes()`` did not expose the columns a caller resolves by name.

    Every reader of ``CALL db.indexes()`` must resolve its columns BY NAME, never
    positionally: the live header carries ``options`` at index 3 and
    ``entitytype`` at index 6, and reading the former as the latter is exactly
    the defect task 3706 was opened on — it degraded silently, returning an
    ``OrderedDict`` where ``'NODE'``/``'RELATIONSHIP'`` belonged, and no test
    noticed because they asserted key presence rather than value.

    Subclasses ``ValueError`` so the historical ``list_indices()`` contract
    (``raise ValueError`` on a missing column) is preserved for callers that
    catch it.  The test-side ``_fm_helpers.IndexHeaderError`` deliberately
    subclasses ``AssertionError`` instead, because there a header surprise must
    read as a failed test rather than as an error the suite might swallow.
    """


def resolve_header_positions(header, wanted: dict[str, str]) -> dict[str, int]:
    """Map each wanted output key to the position of its named FalkorDB column.

    The single home for by-name column resolution over a ``CALL db.indexes()``
    result header, so ``list_indices`` (and later β's ``ensure_indices`` /
    δ's ``summarize_index_health``) share one implementation instead of each
    re-forking the build-names / check-missing / ``.index()`` sequence.  A
    re-forked copy is how the positional read survived in the first place.

    Args:
        header: The result header — the measured live shape is a list of
            ``(type_code, column_name)`` pairs, e.g. ``[[1, 'label'], ...]``.
        wanted: ``{output_key: column_name}``, e.g. ``{'entity_type': 'entitytype'}``.

    Returns:
        ``{output_key: position}`` for every entry in *wanted*.

    Raises:
        IndexHeaderShapeError: A header entry is not a ``(type, name)`` pair, or
            a required column name is absent.  Both fail closed and name what was
            actually seen (INV-2) rather than letting the caller read a column it
            never verified.
    """
    names: list[object] = []
    for position, entry in enumerate(header or []):
        # A bare str is a Sequence whose [1] is a character, so it must be
        # rejected explicitly -- otherwise a header of ['label', 'properties']
        # would resolve to nonsense column names instead of failing.
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) < 2:
            raise IndexHeaderShapeError(
                f'CALL db.indexes() header entry at position {position} is '
                f'{entry!r}, expected a (type, name) pair. FalkorDB changed its '
                'result shape; refusing to guess which column is which. '
                f'Header: {header!r}'
            )
        names.append(entry[1])

    missing = [column for column in wanted.values() if column not in names]
    if missing:
        raise IndexHeaderShapeError(
            f'CALL db.indexes() is missing required column(s) {missing}; '
            f'FalkorDB changed its result shape (header={names}). '
            'Refusing to return index records with silently-wrong values.'
        )
    return {key: names.index(column) for key, column in wanted.items()}


# --- The ACTUAL side: the records ------------------------------------------

#: The index types the PRD's normal form can represent.  Anything else (VECTOR
#: today) is projected away rather than raised on -- see the module docstring's
#: "why the two sides differ in strictness".
_REPRESENTABLE_INDEX_TYPES = frozenset({'RANGE', 'FULLTEXT'})

_VALID_ENTITY_TYPES = frozenset({'NODE', 'RELATIONSHIP'})


class IndexRecordShapeError(ValueError):
    """A ``list_indices()`` record did not have the measured live shape.

    Distinct from :class:`UnparsedIndexStatementError` because the two name
    different problems with different fixes.  An unparsed STATEMENT means
    graphiti changed its syntax (or the wrong provider was passed).  A bad
    RECORD means FalkorDB's ``CALL db.indexes()`` result changed shape, or a
    caller bound its columns wrongly — which is exactly how this was found:
    ``list_indices()`` bound ``entity_type`` to the ``options`` column and
    handed over an ``OrderedDict`` where ``'NODE'``/``'RELATIONSHIP'`` belonged.

    Raised only for genuine SHAPE violations.  An unrepresentable index TYPE is
    not one — that is projected away deliberately (see the module docstring).
    """


def normalize_index_record(record) -> list[IndexSpec]:
    """Project one ``list_indices()`` record onto the normal form.

    Fan-out is driven by the record's ``field`` list (the authoritative property
    membership), with each property's index types read from the ``type`` mapping.
    Driving it off ``type.keys()`` instead would silently under-report if the two
    ever disagreed — the same class of silent-omission bug this module exists to
    catch.  Measured live 2026-08-06, the two are keyed identically today; the
    raise below makes any future divergence loud.

    Because FalkorDB merges every index on a label into ONE record, emission is
    one spec per (property, index_type) pair: an ``Entity`` record carrying
    ``name: ['RANGE', 'FULLTEXT']`` correctly yields both tuples.

    Args:
        record: A mapping with ``label``, ``entity_type``, ``field`` and ``type``
            keys, as returned by ``GraphitiBackend.list_indices()``.

    Returns:
        One spec per (property, representable index type) pair.  May legitimately
        be empty — e.g. a VECTOR-only record.

    Raises:
        IndexRecordShapeError: ``label`` is not a non-empty string,
            ``entity_type`` is not ``'NODE'``/``'RELATIONSHIP'``, ``field`` is
            neither a string nor a non-empty list, or a property in ``field``
            has no entry in ``type``.
    """
    label = record.get('label')
    entity_type = record.get('entity_type')

    # Symmetric with the entity_type check below, and for the same reason: a
    # column mis-binding is the failure class this module exists to catch, and
    # `label` is just as bindable to the wrong column as `entity_type` was.
    # Without this, a dict label sails through and the emitted tuple is only
    # rejected later by `normalize_index_records`' set comprehension, as
    # `TypeError: unhashable type: 'dict'` -- a traceback that names neither the
    # record nor the column, from a function that documents raising
    # IndexRecordShapeError.
    if not isinstance(label, str) or not label:
        raise IndexRecordShapeError(
            f'index record has label {label!r}, expected a non-empty string. '
            'CALL db.indexes() changed shape, or the caller bound the wrong '
            'column. '
            f'Record: {record!r}'
        )

    # Catches the options-column mis-binding: an OrderedDict is not a valid
    # entity_type, and tolerating it would yield a set missing every record.
    # The isinstance check is load-bearing, not defensive noise -- the value this
    # is most likely to receive IS a dict (the options column), and a bare
    # `not in frozenset` would raise TypeError: unhashable type instead of the
    # IndexRecordShapeError that names what actually went wrong.
    if not isinstance(entity_type, str) or entity_type not in _VALID_ENTITY_TYPES:
        raise IndexRecordShapeError(
            f'index record for label {label!r} has entity_type {entity_type!r}, '
            f'expected one of {sorted(_VALID_ENTITY_TYPES)}. CALL db.indexes() '
            'changed shape, or the caller bound the wrong column (the options '
            'column is a dict; entitytype is the string one). '
            f'Record: {record!r}'
        )

    fields = record.get('field')
    # A bare str is accepted as a single-element list so a future FalkorDB scalar
    # shape degrades to "one property" rather than silently to zero tuples.
    if isinstance(fields, str):
        fields = [fields]
    # Anything else RAISES, including a missing/None field list.  Projecting it
    # to zero tuples would be strictly worse than the divergence the next block
    # already refuses to tolerate: that one drops ONE property, this drops ALL of
    # them, and the consequence flows straight into β -- a fully-provisioned
    # label comes back as entirely missing and β re-creates indices that already
    # exist.  MEASURED 2026-08-06: the live `properties` column is always a list,
    # even on a graph with zero indices, so tolerating None buys nothing real.
    elif not isinstance(fields, Sequence) or not fields:
        raise IndexRecordShapeError(
            f'index record for label {label!r} has field {fields!r}, expected a '
            'non-empty list of property names (or a single property string). '
            'Refusing to project it to zero tuples: that silently reports every '
            'index on this label as missing. '
            f'Record: {record!r}'
        )

    types = record.get('type') or {}

    specs: list[IndexSpec] = []
    for prop in fields:
        if prop not in types:
            raise IndexRecordShapeError(
                f'property {prop!r} is listed in the field list of the {label!r} '
                f'index record but has no entry in its type mapping '
                f'({sorted(types)}). Refusing to drop it: a dropped property '
                'silently under-reports what is actually indexed. '
                f'Record: {record!r}'
            )
        raw_types = types[prop]
        if isinstance(raw_types, str):
            raw_types = [raw_types]
        for index_type in raw_types:
            # Unrepresentable types (VECTOR) are projected away, NOT raised on.
            if index_type in _REPRESENTABLE_INDEX_TYPES:
                specs.append((label, entity_type, prop, index_type))
    return specs


def normalize_index_records(records) -> set[IndexSpec]:
    """Union :func:`normalize_index_record` over an iterable of records.

    Returns a set so the caller can diff it directly against
    :func:`expected_index_set`; duplicates arriving from two records collapse.
    """
    return {spec for record in records for spec in normalize_index_record(record)}
