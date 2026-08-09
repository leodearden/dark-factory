"""Unit tests for ``fused_memory.backends.falkor_indices`` (task 3706, α).

What this module pins
---------------------
Both halves of the PRD's normal-form seam, consumed later by β (``ensure_indices``)
and δ (``summarize_index_health``):

* the EXPECTED side — ``expected_index_set()``, derived by parsing the statements
  graphiti itself emits, never a hand-copied list (INV-5);
* the ACTUAL side — ``normalize_index_records()``, which fans a ``list_indices()``
  record out to one tuple per (property, index_type) pair.

Since task 3707 (β) it additionally pins β's PURE planner — ``IndexProvisionResult``,
``range_create_statement`` and ``plan_index_statements`` — which lives in the same
module precisely because it is zero-I/O.  That placement makes D1's "β never issues
the upstream composite range statement" guarantee assertable as a fast unit test
rather than only as an observation against a live graph.

The four FalkorDB statement forms, measured verbatim 2026-08-06 against the
installed **graphiti-core 0.28.2** (pinned open-ended at ``pyproject.toml`` as
``graphiti-core[falkordb]>=0.28.1``)::

    R-node   CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)
    R-edge   CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid, e.group_id, ...)
    FT-node  CALL db.idx.fulltext.createNodeIndex({label: 'Entity', stopwords: [...]},
                                                  'name', 'summary', 'group_id')
    FT-edge  CREATE FULLTEXT INDEX FOR ()-[e:RELATES_TO]-() ON (e.name, e.fact, e.group_id)

Note that NO FalkorDB form carries ``IF NOT EXISTS`` — it is a FalkorDB syntax
error — and none carries an index-name token.  Both facts are load-bearing: they
are what makes the neo4j forms unparseable, which is the tripwire described in
``TestPlainEnumGotcha``.

HAZARD compliance: every test here is a pure unit test.  No live graph, no
``select_graph`` call, and no ``FalkorDriver`` / ``_MultiTenantFalkorDriver`` /
``GraphitiBackend``-with-real-driver construction anywhere — ``FalkorDriver.__init__``
fire-and-forgets ``build_indices_and_constraints()`` when an event loop is running,
so merely constructing one would create indices and destroy esc-3375-1's protected
evidence (the current absence of indices on real graphs).
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict, defaultdict
from unittest.mock import MagicMock

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices

from fused_memory.backends.falkor_indices import (
    IndexHeaderShapeError,
    IndexProvisionResult,
    IndexRecordShapeError,
    UnparsedIndexStatementError,
    expected_index_set,
    normalize_index_record,
    normalize_index_records,
    parse_index_statement,
    range_create_statement,
    resolve_header_positions,
)


def _upstream_statements() -> list[str]:
    """Every index statement the INSTALLED graphiti emits for FalkorDB.

    Built here in the test, independently of the module under test, so the
    zero-unparsed-remainder assertion compares the parser against upstream
    rather than against itself.
    """
    # A list DISPLAY, not list() + list(): graphiti annotates both getters as
    # list[LiteralString], and list[LiteralString] is not assignable to list[str]
    # because list is invariant.  Unpacking into a display lets the declared
    # return type drive inference, so the elements widen to str as intended.
    return [
        *get_range_indices(GraphProvider.FALKORDB),
        *get_fulltext_indices(GraphProvider.FALKORDB),
    ]

# --- The measured FalkorDB RANGE forms, verbatim ---------------------------

RANGE_NODE_ENTITY = 'CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)'
RANGE_NODE_COMMUNITY = 'CREATE INDEX FOR (n:Community) ON (n.uuid)'
RANGE_EDGE_RELATES_TO = (
    'CREATE INDEX FOR ()-[e:RELATES_TO]-() ON '
    '(e.uuid, e.group_id, e.name, e.created_at, e.expired_at, e.valid_at, e.invalid_at)'
)

# --- The measured FalkorDB FULLTEXT forms, verbatim ------------------------
#
# Copied character-for-character from get_fulltext_indices(GraphProvider.FALKORDB)
# under graphiti-core 0.28.2 — including the 48/52-space indentation and the full
# 33-word stopwords list.  The stopwords list is precisely WHY these are copied
# verbatim rather than paraphrased: it is a quoted, comma-separated list living
# INSIDE the config map, so a naive comma- or quote-split over the whole
# statement reports ~36 bogus "properties".  Newlines are explicit (rather than
# a triple-quoted block) so the measured leading whitespace on each line cannot
# be silently reflowed by an editor or a formatter.

_STOPWORDS = (
    "                                                    stopwords: ['a', 'is', 'the', 'an', "
    "'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 'it', 'no', "
    "'not', 'of', 'on', 'or', 'such', 'that', 'their', 'then', 'there', 'these', 'they', "
    "'this', 'to', 'was', 'will', 'with']"
)

FULLTEXT_NODE_ENTITY = '\n'.join((
    'CALL db.idx.fulltext.createNodeIndex(',
    '                                                {',
    "                                                    label: 'Entity',",
    _STOPWORDS,
    '                                                },',
    "                                                'name', 'summary', 'group_id'",
    '                                                )',
))

FULLTEXT_NODE_COMMUNITY = '\n'.join((
    'CALL db.idx.fulltext.createNodeIndex(',
    '                                                {',
    "                                                    label: 'Community',",
    _STOPWORDS,
    '                                                },',
    "                                                'name', 'group_id'",
    '                                                )',
))

FULLTEXT_EDGE_RELATES_TO = (
    'CREATE FULLTEXT INDEX FOR ()-[e:RELATES_TO]-() ON (e.name, e.fact, e.group_id)'
)

# --- The neo4j forms, measured verbatim from the BARE-STRING provider call --
# These must NOT parse.  See TestFailsLoudly for why that is the contract.

NEO4J_RANGE = 'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)'
NEO4J_FULLTEXT = (
    'CREATE FULLTEXT INDEX episode_content IF NOT EXISTS\n'
    '  FOR (e:Episodic) ON EACH [e.content, e.source]'
)


class TestParseRangeStatements:
    """``parse_index_statement`` fans a FalkorDB RANGE statement out to one spec per property."""

    def test_node_form_fans_out_to_one_spec_per_property(self):
        """The 4-property node form yields exactly 4 ('Entity', 'NODE', <prop>, 'RANGE') tuples."""
        specs = parse_index_statement(RANGE_NODE_ENTITY)
        assert set(specs) == {
            ('Entity', 'NODE', 'uuid', 'RANGE'),
            ('Entity', 'NODE', 'group_id', 'RANGE'),
            ('Entity', 'NODE', 'name', 'RANGE'),
            ('Entity', 'NODE', 'created_at', 'RANGE'),
        }
        assert len(specs) == 4

    def test_relationship_form_yields_relationship_entity_type(self):
        """The 7-property edge form yields ('RELATES_TO', 'RELATIONSHIP', <prop>, 'RANGE')."""
        specs = parse_index_statement(RANGE_EDGE_RELATES_TO)
        assert set(specs) == {
            ('RELATES_TO', 'RELATIONSHIP', prop, 'RANGE')
            for prop in (
                'uuid', 'group_id', 'name', 'created_at',
                'expired_at', 'valid_at', 'invalid_at',
            )
        }
        assert len(specs) == 7

    def test_single_property_form_yields_exactly_one_spec(self):
        """Arity is driven by the statement, not hardcoded — a 1-property index gives 1 spec."""
        assert parse_index_statement(RANGE_NODE_COMMUNITY) == [
            ('Community', 'NODE', 'uuid', 'RANGE'),
        ]

    @pytest.mark.parametrize(
        'statement',
        [RANGE_NODE_ENTITY, RANGE_NODE_COMMUNITY, RANGE_EDGE_RELATES_TO],
        ids=['node-4prop', 'node-1prop', 'edge-7prop'],
    )
    def test_never_returns_an_empty_list_for_a_recognised_form(self, statement):
        """A statement is either fanned out or raises — there is no skip-and-warn path.

        An empty return would be exactly the silent under-provisioning this PRD
        exists to remove (INV-4): the expected set would come back short and the
        diff would report "nothing missing" for an index that was never created.
        """
        assert parse_index_statement(statement) != []


class TestParseFulltextStatements:
    """The two upstream FULLTEXT syntaxes — a CALL procedure and a CREATE statement.

    graphiti emits fulltext indices in two entirely different shapes depending on
    node vs edge, so this is not one form with a keyword difference: the node side
    is a stored-procedure CALL whose properties are positional arguments AFTER a
    config map, and the edge side is an ordinary ``CREATE FULLTEXT INDEX ... ON (...)``.
    """

    def test_node_fulltext_call_form_fans_out_to_one_spec_per_property(self):
        specs = parse_index_statement(FULLTEXT_NODE_ENTITY)
        assert set(specs) == {
            ('Entity', 'NODE', 'name', 'FULLTEXT'),
            ('Entity', 'NODE', 'summary', 'FULLTEXT'),
            ('Entity', 'NODE', 'group_id', 'FULLTEXT'),
        }

    def test_stopwords_never_leak_into_the_property_set(self):
        """THE comma hazard: properties come from AFTER the config map's closing brace.

        The config map embeds a 33-word quoted, comma-separated stopwords list, so
        a naive comma- or quote-split over the whole statement yields ~36 bogus
        "properties" — every one of which would then be reported as a MISSING
        index forever, and none of which could ever be created.  This assertion is
        what pins the brace-matching requirement.
        """
        props = {spec[2] for spec in parse_index_statement(FULLTEXT_NODE_ENTITY)}
        assert props == {'name', 'summary', 'group_id'}
        for stopword in ('a', 'is', 'the', 'and', 'with'):
            assert stopword not in props

    def test_node_fulltext_arity_follows_the_statement(self):
        """Community indexes 2 properties, Entity 3 — the arity is not hardcoded."""
        specs = parse_index_statement(FULLTEXT_NODE_COMMUNITY)
        assert set(specs) == {
            ('Community', 'NODE', 'name', 'FULLTEXT'),
            ('Community', 'NODE', 'group_id', 'FULLTEXT'),
        }
        assert len(specs) == 2

    def test_edge_fulltext_create_form(self):
        specs = parse_index_statement(FULLTEXT_EDGE_RELATES_TO)
        assert set(specs) == {
            ('RELATES_TO', 'RELATIONSHIP', 'name', 'FULLTEXT'),
            ('RELATES_TO', 'RELATIONSHIP', 'fact', 'FULLTEXT'),
            ('RELATES_TO', 'RELATIONSHIP', 'group_id', 'FULLTEXT'),
        }

    def test_config_map_without_a_label_key_raises_rather_than_yielding_an_empty_label(self):
        """Strictness survives the new branch — no silent ('', 'NODE', 'name', 'FULLTEXT')."""
        statement = "CALL db.idx.fulltext.createNodeIndex({stopwords: []}, 'name')"
        with pytest.raises(UnparsedIndexStatementError) as excinfo:
            parse_index_statement(statement)
        assert statement in str(excinfo.value)


class TestFailsLoudly:
    """An unrecognised statement RAISES; it is never skipped, warned about, or dropped.

    PRD Open Question 1 is closed as "fail".  The direction of the failure is what
    forces this: an unparsed UPSTREAM statement means the expected set is SHORT,
    which means under-provisioning — precisely this PRD's own failure mode
    recurring.  A skip-and-warn would make the detector quietly agree that a
    missing index is fine.
    """

    @pytest.mark.parametrize(
        'statement',
        [NEO4J_RANGE, NEO4J_FULLTEXT, 'MATCH (n) RETURN n'],
        ids=['neo4j-range', 'neo4j-fulltext', 'not-an-index-statement'],
    )
    def test_unrecognised_statement_raises(self, statement):
        with pytest.raises(UnparsedIndexStatementError):
            parse_index_statement(statement)

    @pytest.mark.parametrize(
        'statement',
        [NEO4J_RANGE, NEO4J_FULLTEXT, 'MATCH (n) RETURN n'],
        ids=['neo4j-range', 'neo4j-fulltext', 'not-an-index-statement'],
    )
    def test_message_carries_the_offending_statement_verbatim(self, statement):
        """INV-2 — emit the fact the caller actually has, not a generic complaint.

        The operator reading this failure needs the statement text to tell a
        graphiti syntax change apart from a provider mix-up.
        """
        with pytest.raises(UnparsedIndexStatementError) as excinfo:
            parse_index_statement(statement)
        assert statement in str(excinfo.value)


class TestExpectedIndexSet:
    """``expected_index_set()`` consumes EVERY statement the installed graphiti emits.

    This is the task's primary signal, and it is deliberately expressed as
    "nothing was left unparsed" rather than as a count.  A count would have to be
    edited in lock-step with every graphiti upgrade — the hand-copied duplication
    INV-5 / PRD D3 exist to prevent — and a wrong count is indistinguishable from
    a wrong parse.  The remainder assertion is derived from whatever upstream
    emits, so it stays true across upgrades and fails only when a genuinely new
    statement form appears.
    """

    def test_every_upstream_statement_parses_with_zero_remainder(self):
        statements = _upstream_statements()
        # Guards against a vacuous pass if a future graphiti returns [].
        assert statements, 'graphiti emitted no FalkorDB index statements at all'

        for statement in statements:
            specs = parse_index_statement(statement)  # raises if unparseable
            assert specs, f'statement parsed to zero specs: {statement}'

        assert expected_index_set() == {
            spec for statement in statements for spec in parse_index_statement(statement)
        }

    def test_expected_set_is_the_union_of_its_parts_and_well_typed(self):
        """Every tuple is in the normal form the PRD defines — no stray shapes."""
        for label, entity_type, field, index_type in expected_index_set():
            assert isinstance(label, str) and label
            assert entity_type in {'NODE', 'RELATIONSHIP'}
            assert isinstance(field, str) and field
            assert index_type in {'RANGE', 'FULLTEXT'}

    @pytest.mark.parametrize(
        ('label', 'entity_type', 'fields', 'index_type'),
        [
            ('Entity', 'NODE', ('uuid', 'group_id', 'name', 'created_at'), 'RANGE'),
            ('Entity', 'NODE', ('name', 'summary', 'group_id'), 'FULLTEXT'),
            (
                'RELATES_TO', 'RELATIONSHIP',
                ('uuid', 'group_id', 'name', 'created_at', 'expired_at', 'valid_at', 'invalid_at'),
                'RANGE',
            ),
            ('RELATES_TO', 'RELATIONSHIP', ('name', 'fact', 'group_id'), 'FULLTEXT'),
        ],
        ids=['entity-range', 'entity-fulltext', 'relates_to-range', 'relates_to-fulltext'],
    )
    def test_fan_out_content_by_label(self, label, entity_type, fields, index_type):
        """The substantive shape, checked by label rather than against a magic total."""
        expected = expected_index_set()
        assert {(label, entity_type, f, index_type) for f in fields} <= expected

    def test_both_index_types_are_represented(self):
        """A shape check, deliberately NOT a count.

        Measured 2026-08-06 under graphiti-core 0.28.2 the totals were 38 specs
        (26 RANGE + 12 FULLTEXT), and an earlier revision asserted exactly that.
        Those literals were removed: they go red on any legitimate graphiti
        upgrade that adds or drops an index, which is the lock-step hand-copied
        duplication INV-5 / PRD D3 exist to prevent, and a red count is
        indistinguishable from a real parse regression.
        ``test_every_upstream_statement_parses_with_zero_remainder`` already
        covers the same ground derived FROM upstream, so the counts added
        upgrade friction and no failure mode.

        What survives is the property that cannot silently rot: both halves of
        the union are non-empty, so a generator returning ``[]`` (or a fan-out
        that collapsed to one index type) is still caught.
        """
        expected = expected_index_set()
        assert {spec for spec in expected if spec[3] == 'RANGE'}
        assert {spec for spec in expected if spec[3] == 'FULLTEXT'}
        assert {spec[3] for spec in expected} == {'RANGE', 'FULLTEXT'}


class TestPlainEnumGotcha:
    """The bare string ``'falkordb'`` silently returns the NEO4J statement set.

    MEASURED 2026-08-06 under graphiti-core 0.28.2: ``GraphProvider`` is a plain
    ``Enum``, so ``GraphProvider.FALKORDB == 'falkordb'`` is **False**, and
    ``get_range_indices('falkordb')`` returns 27 NEO4J statements with **no
    error** — the ``(provider: GraphProvider)`` annotation is not enforced at
    runtime.  Nothing in graphiti catches this.

    What catches it is the strict parser: no neo4j form is parseable (they carry
    an index-name token and ``IF NOT EXISTS``, neither of which appears in any
    FalkorDB form, and ``IF NOT EXISTS`` is itself a FalkorDB syntax error).  So
    a future edit that regressed to the bare string would BLOW UP here rather
    than silently produce a wrong 27-statement expected set.  These tests pin the
    measurement so the tripwire's premise cannot rot unnoticed.
    """

    def test_enum_member_does_not_equal_its_string_value(self):
        assert GraphProvider.FALKORDB != 'falkordb'

    def test_bare_string_silently_returns_the_neo4j_set(self):
        # The suppression below is load-bearing, not cleanup debt.  pyright is
        # RIGHT that 'falkordb' is the wrong type -- and that static-only
        # rejection is precisely the gotcha this class pins: nothing rejects it
        # at RUNTIME, where it quietly yields the neo4j set instead.  The
        # deliberate mis-call has to survive type-checking to be exercised.
        neo4j_statements = get_range_indices('falkordb')  # pyright: ignore[reportArgumentType]
        # The DISCRIMINATING property, not a count.  An earlier revision asserted
        # `len(...) == 27`; that pinned an upstream total in lock-step for no
        # extra signal, since what makes the gotcha real is the SYNTAX the bare
        # string yields, not how many statements of it there are.
        assert neo4j_statements
        assert all('IF NOT EXISTS' in s for s in neo4j_statements)

    def test_enum_member_returns_the_falkordb_set(self):
        falkor_statements = get_range_indices(GraphProvider.FALKORDB)
        assert falkor_statements
        # IF NOT EXISTS is a FalkorDB syntax error, so its absence is load-bearing.
        # (Also formerly a `len(...) == 9` assertion — dropped for the same reason.)
        assert not any('IF NOT EXISTS' in s for s in falkor_statements)

    def test_the_two_providers_do_not_return_the_same_statements(self):
        """The gotcha in one line, and it survives any upstream count change.

        If a future graphiti made ``GraphProvider`` a ``StrEnum``, the bare
        string would start returning the FalkorDB set and this would go red —
        which is the moment to revisit the tripwire premise, not to silently
        keep relying on it.
        """
        bare_string = get_range_indices('falkordb')  # pyright: ignore[reportArgumentType]
        enum_member = get_range_indices(GraphProvider.FALKORDB)
        assert list(bare_string) != list(enum_member)

    def test_the_neo4j_set_is_unparseable_which_is_what_makes_it_a_tripwire(self):
        # Deliberate mis-call again -- see test_bare_string_silently_returns_the_neo4j_set.
        for statement in get_range_indices('falkordb'):  # pyright: ignore[reportArgumentType]
            with pytest.raises(UnparsedIndexStatementError):
                parse_index_statement(statement)


class TestNormalizeIndexRecords:
    """The ACTUAL side: project a ``list_indices()`` record onto the normal form.

    Fixtures here are SYNTHETIC dicts built to the shape measured live on
    2026-08-06 via ``GRAPH.RO_QUERY dark_factory "CALL db.indexes()"``: ``field``
    is a LIST of property names and ``type`` is a DICT of property -> list of
    index-type strings.  ``TestListIndicesColumnBinding`` below then drives the
    same normalizer from a mock shaped like the real 9-column response, so the
    two halves are checked against a real-shaped row and not only against these
    hand-written dicts.

    Why this side is a PROJECTION while the expected side RAISES: the two fail in
    opposite directions.  An unparsed UPSTREAM statement shortens the expected
    set (under-provisioning — must be loud).  An ACTUAL index the normal form
    cannot represent is not drift to repair: the PRD defines
    ``index_type in {RANGE, FULLTEXT}``, VECTOR indices legitimately exist on
    real graphs, and the PRD says ``unexpected`` is reported but never acted on.
    Raising there would make the detector a false-alarm generator on every real
    graph.  Genuine SHAPE surprises still raise — see the IndexRecordShapeError
    tests.
    """

    def test_boundary_4_a_four_property_record_yields_four_distinct_tuples(self):
        """THE named signal: one multi-property record fans out to one tuple per property.

        The caveat this guards: a naive comparison that treated the record's
        ``field`` list as a single value would report EVERY multi-property index
        as missing, forever.
        """
        record = {
            'label': 'Entity',
            'field': ['uuid', 'group_id', 'name', 'created_at'],
            'type': {
                'uuid': ['RANGE'],
                'group_id': ['RANGE'],
                'name': ['RANGE'],
                'created_at': ['RANGE'],
            },
            'entity_type': 'NODE',
        }
        specs = normalize_index_record(record)
        assert set(specs) == {
            ('Entity', 'NODE', 'uuid', 'RANGE'),
            ('Entity', 'NODE', 'group_id', 'RANGE'),
            ('Entity', 'NODE', 'name', 'RANGE'),
            ('Entity', 'NODE', 'created_at', 'RANGE'),
        }
        # Tied to the record, not to a literal 4.
        assert len(specs) == len(record['field'])
        assert len(set(specs)) == len(specs)

    def test_the_naive_whole_list_as_one_field_reading_is_dead(self):
        """Regression guard: no emitted tuple carries a LIST where a property belongs."""
        record = {
            'label': 'Entity',
            'field': ['uuid', 'group_id', 'name', 'created_at'],
            'type': {p: ['RANGE'] for p in ('uuid', 'group_id', 'name', 'created_at')},
            'entity_type': 'NODE',
        }
        specs = normalize_index_record(record)
        assert all(isinstance(spec[2], str) for spec in specs)
        assert ('Entity', 'NODE', ['uuid', 'group_id', 'name', 'created_at'], 'RANGE') not in specs

    def test_merged_range_and_fulltext_record_emits_one_tuple_per_property_type_pair(self):
        """FalkorDB merges per-label indices into ONE record, so a property can carry both types."""
        record = {
            'label': 'Entity',
            'field': ['uuid', 'group_id', 'name', 'created_at', 'summary'],
            'type': {
                'uuid': ['RANGE'],
                'group_id': ['RANGE', 'FULLTEXT'],
                'name': ['RANGE', 'FULLTEXT'],
                'created_at': ['RANGE'],
                'summary': ['FULLTEXT'],
            },
            'entity_type': 'NODE',
        }
        specs = normalize_index_record(record)
        assert set(specs) == {
            ('Entity', 'NODE', 'uuid', 'RANGE'),
            ('Entity', 'NODE', 'group_id', 'RANGE'),
            ('Entity', 'NODE', 'group_id', 'FULLTEXT'),
            ('Entity', 'NODE', 'name', 'RANGE'),
            ('Entity', 'NODE', 'name', 'FULLTEXT'),
            ('Entity', 'NODE', 'created_at', 'RANGE'),
            ('Entity', 'NODE', 'summary', 'FULLTEXT'),
        }
        assert len(specs) == 7

    def test_relationship_entity_type_passes_through_unchanged(self):
        record = {
            'label': 'RELATES_TO',
            'field': ['uuid', 'group_id'],
            'type': {'uuid': ['RANGE'], 'group_id': ['RANGE']},
            'entity_type': 'RELATIONSHIP',
        }
        assert set(normalize_index_record(record)) == {
            ('RELATES_TO', 'RELATIONSHIP', 'uuid', 'RANGE'),
            ('RELATES_TO', 'RELATIONSHIP', 'group_id', 'RANGE'),
        }

    def test_vector_only_record_projects_to_zero_tuples(self):
        """PROJECTION, not fail-soft — the normal form cannot represent VECTOR.

        A vector index added by an operator or by graphiti is not drift to
        repair; the PRD reports ``unexpected`` but never acts on it.  Raising
        here would make the detector alarm on every real graph.
        """
        record = {
            'label': 'Entity',
            'field': ['name_embedding'],
            'type': {'name_embedding': ['VECTOR']},
            'entity_type': 'NODE',
        }
        assert normalize_index_record(record) == []

    def test_mixed_vector_and_range_record_keeps_only_the_representable_tuple(self):
        record = {
            'label': 'Entity',
            'field': ['name_embedding', 'uuid'],
            'type': {'name_embedding': ['VECTOR'], 'uuid': ['RANGE']},
            'entity_type': 'NODE',
        }
        assert normalize_index_record(record) == [('Entity', 'NODE', 'uuid', 'RANGE')]

    def test_unknown_entity_type_raises(self):
        record = {
            'label': 'Entity',
            'field': ['uuid'],
            'type': {'uuid': ['RANGE']},
            'entity_type': 'NODE_OR_SOMETHING',
        }
        with pytest.raises(IndexRecordShapeError):
            normalize_index_record(record)

    def test_the_mis_bound_options_dict_raises(self):
        """DIRECT pin on the step-9/10 defect.

        Today's ``list_indices()`` binds ``entity_type`` to ``row[3]`` — the
        ``options`` column — so it hands the normalizer an OrderedDict like
        ``{'uuid': {}}`` where ``'NODE'``/``'RELATIONSHIP'`` belongs.  That must
        raise here rather than being coerced or skipped, because a normalizer
        that tolerated it would produce a set silently missing every record.
        """
        record = {
            'label': 'RELATES_TO',
            'field': ['uuid'],
            'type': {'uuid': ['RANGE']},
            'entity_type': OrderedDict({'uuid': OrderedDict()}),
        }
        with pytest.raises(IndexRecordShapeError):
            normalize_index_record(record)

    @pytest.mark.parametrize(
        'label',
        [OrderedDict({'uuid': OrderedDict()}), None, '', 42],
        ids=['options-dict', 'none', 'empty-string', 'int'],
    )
    def test_non_string_label_raises_the_documented_error_not_an_unhashable_typeerror(
        self, label,
    ):
        """``label`` is exactly as bindable to the wrong column as ``entity_type`` was.

        Without the guard a dict label sails through ``normalize_index_record``
        and emits tuples like ``({'uuid': {}}, 'NODE', 'uuid', 'RANGE')``; the
        set comprehension in ``normalize_index_records`` then dies with
        ``TypeError: unhashable type: 'dict'`` — a traceback naming neither the
        record nor the column, from a function whose contract says
        ``IndexRecordShapeError``.  Asserting through the PLURAL entry point is
        what pins that, since the singular one never raised the TypeError.
        """
        record = {
            'label': label,
            'field': ['uuid'],
            'type': {'uuid': ['RANGE']},
            'entity_type': 'NODE',
        }
        with pytest.raises(IndexRecordShapeError):
            normalize_index_record(record)
        with pytest.raises(IndexRecordShapeError):
            normalize_index_records([record])

    @pytest.mark.parametrize(
        'field',
        [None, [], {}, 42],
        ids=['missing-none', 'empty-list', 'dict', 'int'],
    )
    def test_absent_or_empty_field_list_raises_rather_than_projecting_to_zero_tuples(
        self, field,
    ):
        """Dropping ALL properties must be at least as loud as dropping one.

        ``test_property_present_in_field_but_absent_from_type_raises`` refuses the
        far milder surprise of ONE property going missing, on the grounds that it
        under-reports what is actually indexed.  A ``None``/empty ``field`` drops
        every property on the label, and the consequence flows into β: the ACTUAL
        set comes back short, a fully-provisioned label reads as entirely missing,
        and β re-creates indices that already exist.

        MEASURED 2026-08-06: the live ``properties`` column is always a list, even
        on a graph with zero indices, so tolerating ``None`` bought nothing.
        """
        record = {
            'label': 'Entity',
            'field': field,
            'type': {'uuid': ['RANGE']},
            'entity_type': 'NODE',
        }
        with pytest.raises(IndexRecordShapeError) as excinfo:
            normalize_index_record(record)
        assert 'Entity' in str(excinfo.value)

    def test_property_present_in_field_but_absent_from_type_raises(self):
        """A genuine SHAPE surprise — dropping it would silently under-report."""
        record = {
            'label': 'Entity',
            'field': ['uuid', 'group_id'],
            'type': {'uuid': ['RANGE']},
            'entity_type': 'NODE',
        }
        with pytest.raises(IndexRecordShapeError) as excinfo:
            normalize_index_record(record)
        assert 'group_id' in str(excinfo.value)

    def test_normalize_index_records_unions_and_deduplicates(self):
        records = [
            {
                'label': 'Entity', 'field': ['uuid'],
                'type': {'uuid': ['RANGE']}, 'entity_type': 'NODE',
            },
            {
                'label': 'Entity', 'field': ['uuid'],
                'type': {'uuid': ['RANGE']}, 'entity_type': 'NODE',
            },
            {
                'label': 'RELATES_TO', 'field': ['uuid'],
                'type': {'uuid': ['RANGE']}, 'entity_type': 'RELATIONSHIP',
            },
        ]
        result = normalize_index_records(records)
        assert isinstance(result, set)
        assert result == {
            ('Entity', 'NODE', 'uuid', 'RANGE'),
            ('RELATES_TO', 'RELATIONSHIP', 'uuid', 'RANGE'),
        }

    def test_seam_round_trip_a_fully_provisioned_graph_diffs_to_empty(self):
        """The α↔β seam: derive records mechanically FROM the expected set and normalize back.

        Records are DERIVED, never hand-copied, so this cannot drift from the
        parser.  A green here means the diff β and δ compute is empty for a
        fully-provisioned graph — i.e. the two halves genuinely speak the same
        normal form rather than merely looking similar.
        """
        expected = expected_index_set()

        grouped: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for label, entity_type, field, index_type in expected:
            grouped[(label, entity_type)][field].append(index_type)

        derived = [
            {
                'label': label,
                'entity_type': entity_type,
                'field': list(types_by_field),
                'type': {f: list(ts) for f, ts in types_by_field.items()},
            }
            for (label, entity_type), types_by_field in grouped.items()
        ]

        assert normalize_index_records(derived) == expected


# --- The MEASURED live CALL db.indexes() shape -----------------------------
#
# Measured read-only 2026-08-06 via
#   docker exec docker-falkordb-1 redis-cli GRAPH.RO_QUERY dark_factory "CALL db.indexes()"
# and via a raw falkordb.asyncio client on the ro path.  No graphiti driver was
# constructed and no index was created, dropped or modified anywhere.
#
# The header is 9 two-tuples; note that `entitytype` is index 6 and `options` is
# index 3.  That gap is the defect these tests pin: list_indices() bound
# entity_type to row[3].

LIVE_HEADER = [
    [1, 'label'], [1, 'properties'], [1, 'types'], [1, 'options'], [1, 'language'],
    [1, 'stopwords'], [1, 'entitytype'], [1, 'status'], [1, 'info'],
]

LIVE_ROW_RELATES_TO = [
    'RELATES_TO',
    ['uuid'],
    OrderedDict({'uuid': ['RANGE']}),
    OrderedDict({'uuid': OrderedDict()}),
    'english',
    [],
    'RELATIONSHIP',
    'OPERATIONAL',
    {},
]

LIVE_ROW_ENTITY = [
    'Entity',
    ['uuid', 'group_id', 'name', 'created_at'],
    OrderedDict({
        'uuid': ['RANGE'], 'group_id': ['RANGE'],
        'name': ['RANGE'], 'created_at': ['RANGE'],
    }),
    OrderedDict({'uuid': OrderedDict()}),
    'english',
    [],
    'NODE',
    'OPERATIONAL',
    {},
]


class TestResolveHeaderPositions:
    """The SINGLE home for by-name ``CALL db.indexes()`` column resolution.

    Extracted so ``list_indices`` — and later β's ``ensure_indices`` and δ's
    ``summarize_index_health`` — share one implementation rather than each
    re-forking the build-names / check-missing / ``.index()`` sequence.  A
    re-forked copy is how the positional read survived in the first place.
    (``_fm_helpers.await_index_operational`` still carries its own copy; it is
    outside task 3706's locked scope and is filed as a follow-up.)
    """

    def test_resolves_every_wanted_key_to_its_named_column(self):
        positions = resolve_header_positions(
            LIVE_HEADER,
            {'label': 'label', 'field': 'properties', 'entity_type': 'entitytype'},
        )
        assert positions == {'label': 0, 'field': 1, 'entity_type': 6}

    def test_positions_follow_the_header_not_the_measured_order(self):
        reordered = [[1, 'entitytype'], [1, 'label'], [1, 'properties']]
        positions = resolve_header_positions(
            reordered,
            {'label': 'label', 'field': 'properties', 'entity_type': 'entitytype'},
        )
        assert positions == {'label': 1, 'field': 2, 'entity_type': 0}

    def test_missing_column_raises_naming_it_and_the_header_it_saw(self):
        with pytest.raises(IndexHeaderShapeError) as excinfo:
            resolve_header_positions(
                [c for c in LIVE_HEADER if c[1] != 'entitytype'],
                {'label': 'label', 'entity_type': 'entitytype'},
            )
        message = str(excinfo.value)
        assert 'entitytype' in message
        assert 'label' in message  # the header it actually saw is named

    @pytest.mark.parametrize(
        'header',
        [
            ['label', 'properties'],
            [[1, 'label'], [1]],
            [[1, 'label'], None],
            [[1, 'label'], 42],
        ],
        ids=['bare-strings', 'one-element-pair', 'none-entry', 'int-entry'],
    )
    def test_malformed_header_entry_raises_the_named_shape_error(self, header):
        """A bare-string header is the trap: ``'label'[1]`` is ``'a'``, not an error.

        Indexing ``col[1]`` unguarded turns a shape surprise into either an
        IndexError/TypeError with no context or — worse, for the bare-string case
        — a silently wrong set of column names.  Both are the silent degradation
        by-name resolution exists to remove, so both must be the named error.
        """
        with pytest.raises(IndexHeaderShapeError):
            resolve_header_positions(header, {'label': 'label'})

    @pytest.mark.parametrize('header', [None, []], ids=['none', 'empty'])
    def test_absent_header_raises_rather_than_resolving_nothing(self, header):
        with pytest.raises(IndexHeaderShapeError):
            resolve_header_positions(header, {'label': 'label'})

    def test_error_is_a_valueerror_preserving_the_list_indices_contract(self):
        """``list_indices`` historically raised ``ValueError``; callers may catch it."""
        assert issubclass(IndexHeaderShapeError, ValueError)


class TestListIndicesColumnBinding:
    """``list_indices()`` must resolve ``CALL db.indexes()`` columns BY NAME.

    MEASURED 2026-08-06: the live header is
    ``[label, properties, types, options, language, stopwords, entitytype, status, info]``,
    so ``entitytype`` sits at index 6 while ``options`` sits at index 3.
    ``list_indices()`` bound ``'entity_type': row[3]`` — the OPTIONS column — and
    therefore returned ``OrderedDict({'uuid': OrderedDict()})`` where
    ``'NODE'``/``'RELATIONSHIP'`` belonged.  The PRD's normal form REQUIRES
    ``entity_type``, so the normalizer cannot be built on top of that.

    The defect survived because the existing live test asserts only key
    PRESENCE, never the value.

    By-name resolution — not a corrected index — is the fix, mirroring
    ``_fm_helpers.await_index_operational``, which resolves ``status`` by name
    precisely so a FalkorDB column reorder fails loudly instead of silently
    reading the wrong column.  ``test_by_name_resolution_survives_a_column_reorder``
    is what a merely-positional fix cannot pass.

    HAZARD: driven entirely through ``make_graph_mock``.  No live FalkorDB, no
    driver construction.
    """

    @pytest.mark.asyncio
    async def test_entity_type_binds_to_the_entitytype_column(
        self, mock_config, make_backend, make_graph_mock,
    ):
        backend = make_backend(mock_config)
        graph = make_graph_mock([LIVE_ROW_RELATES_TO], header=LIVE_HEADER)
        backend._driver._get_graph = MagicMock(return_value=graph)

        records = await backend.list_indices(group_id='test')

        assert len(records) == 1
        assert records[0]['entity_type'] == 'RELATIONSHIP'

    @pytest.mark.asyncio
    async def test_by_name_resolution_survives_a_column_reorder(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """A positional fix (row[3] -> row[6]) passes the test above by accident; not this one."""
        reordered_header = [
            [1, 'status'], [1, 'entitytype'], [1, 'label'], [1, 'info'],
            [1, 'types'], [1, 'language'], [1, 'properties'], [1, 'stopwords'],
            [1, 'options'],
        ]
        reordered_row = [
            'OPERATIONAL',
            'RELATIONSHIP',
            'RELATES_TO',
            {},
            OrderedDict({'uuid': ['RANGE']}),
            'english',
            ['uuid'],
            [],
            OrderedDict({'uuid': OrderedDict()}),
        ]
        backend = make_backend(mock_config)
        graph = make_graph_mock([reordered_row], header=reordered_header)
        backend._driver._get_graph = MagicMock(return_value=graph)

        records = await backend.list_indices(group_id='test')

        assert records[0]['label'] == 'RELATES_TO'
        assert records[0]['field'] == ['uuid']
        assert records[0]['type'] == {'uuid': ['RANGE']}
        assert records[0]['entity_type'] == 'RELATIONSHIP'

    @pytest.mark.asyncio
    async def test_missing_required_column_raises_naming_it_and_the_header(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Never return a record with a silently-wrong or absent value (INV-2/INV-4).

        Mirrors ``_fm_helpers.IndexHeaderError``: a shape change means the caller
        cannot be trusted at all, so it fails closed and names what it saw.
        """
        header_without_entitytype = [c for c in LIVE_HEADER if c[1] != 'entitytype']
        backend = make_backend(mock_config)
        graph = make_graph_mock([LIVE_ROW_RELATES_TO], header=header_without_entitytype)
        backend._driver._get_graph = MagicMock(return_value=graph)

        with pytest.raises(ValueError) as excinfo:
            await backend.list_indices(group_id='test')

        message = str(excinfo.value)
        assert 'entitytype' in message
        assert 'label' in message  # the header it actually saw is named

    @pytest.mark.asyncio
    async def test_composition_with_the_normalizer_over_real_shaped_rows(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """What β and δ actually call — proving the two halves compose.

        Checked against a REAL-shaped response rather than only against
        hand-written synthetic dicts, which is the whole point: a normalizer
        validated only against fiction would be validated against the exact
        failure shape this PRD exists to remove.
        """
        backend = make_backend(mock_config)
        graph = make_graph_mock(
            [LIVE_ROW_RELATES_TO, LIVE_ROW_ENTITY], header=LIVE_HEADER,
        )
        backend._driver._get_graph = MagicMock(return_value=graph)

        actual = normalize_index_records(await backend.list_indices(group_id='test'))

        assert actual == {
            ('RELATES_TO', 'RELATIONSHIP', 'uuid', 'RANGE'),
            ('Entity', 'NODE', 'uuid', 'RANGE'),
            ('Entity', 'NODE', 'group_id', 'RANGE'),
            ('Entity', 'NODE', 'name', 'RANGE'),
            ('Entity', 'NODE', 'created_at', 'RANGE'),
        }

    @pytest.mark.asyncio
    async def test_read_only_path_is_unchanged(
        self, mock_config, make_backend, make_graph_mock,
    ):
        """Guards the read-only guarantee test_list_indices_integration.py pins live."""
        backend = make_backend(mock_config)
        graph = make_graph_mock([LIVE_ROW_RELATES_TO], header=LIVE_HEADER)
        backend._driver._get_graph = MagicMock(return_value=graph)

        await backend.list_indices(group_id='test')

        assert graph.ro_query.called
        assert not graph.query.called


# --- β's pure planner (task 3707) ------------------------------------------


def _expected_range_specs() -> list:
    """Every RANGE spec in the expected set, DERIVED — never hard-coded."""
    return sorted(spec for spec in expected_index_set() if spec[3] == 'RANGE')


class TestIndexProvisionResult:
    """The structured result β returns instead of a bare "did something" log.

    The PRD contract names four fields (``created`` / ``already_present`` /
    ``failed`` / ``expected_total``); β adds a fifth, ``statements`` — the
    statements actually issued, in order — because two obligations need it and
    neither is served by the four.  Boundary test 1 asserts that NO composite
    statement appears in the issued-statement log, and D7 requires the WARNING to
    name each FAILED STATEMENT, which cannot be reconstructed from an IndexSpec
    (a fulltext spec's statement is upstream's verbatim string).
    """

    def test_field_set_is_exactly_the_contract_plus_statements(self):
        """No accidental extra field, and none of the five silently dropped."""
        assert dataclasses.is_dataclass(IndexProvisionResult)
        names = [f.name for f in dataclasses.fields(IndexProvisionResult)]
        assert set(names) == {
            'created', 'already_present', 'failed', 'expected_total', 'statements',
        }

    def test_constructs_with_the_contract_field_types(self):
        spec = ('Entity', 'NODE', 'name', 'RANGE')
        result = IndexProvisionResult(
            created=[spec],
            already_present=2,
            failed=[(spec, 'boom')],
            expected_total=38,
            statements=['CREATE INDEX FOR (n:Entity) ON (n.name)'],
        )
        assert result.created == [spec]
        assert result.already_present == 2
        assert result.failed == [(spec, 'boom')]
        assert result.expected_total == 38
        assert result.statements == ['CREATE INDEX FOR (n:Entity) ON (n.name)']

    @pytest.mark.parametrize(
        ('created', 'failed', 'expected'),
        [
            ([('Entity', 'NODE', 'name', 'RANGE')], [], True),
            ([], [(('Entity', 'NODE', 'name', 'RANGE'), 'boom')], True),
            ([], [], False),
        ],
        ids=['created-only', 'failed-only', 'neither'],
    )
    def test_changed_is_true_when_anything_was_created_or_failed(
        self, created, failed, expected,
    ):
        """``changed`` is the predicate D7's INFO-on-change log keys on.

        A FAILED statement counts as changed on purpose: a run that tried and
        could not provision must not be indistinguishable from a run that found
        nothing to do (INV-2 — emit the fact the caller actually has).
        """
        result = IndexProvisionResult(
            created=created,
            already_present=0,
            failed=failed,
            expected_total=38,
            statements=[],
        )
        assert result.changed is expected


class TestRangeCreateStatement:
    """β synthesizes RANGE indices PER-PROPERTY, never as upstream's composite.

    MEASURED against the seeded trap state: upstream's
    ``CREATE INDEX FOR (n:Entity) ON (n.uuid, n.group_id, n.name, n.created_at)``
    is rejected wholesale with ``Attribute 'uuid' is already indexed`` when ANY
    listed property already exists — and ``falkordb_driver.py``'s
    ``execute_query`` swallows that rejection, so all four properties are lost
    silently.  Per-property statements converge to the identical index state
    while degrading one property at a time.
    """

    def test_node_form_is_the_single_property_shape(self):
        assert (
            range_create_statement(('Entity', 'NODE', 'name', 'RANGE'))
            == 'CREATE INDEX FOR (n:Entity) ON (n.name)'
        )

    def test_relationship_form_is_the_single_property_edge_shape(self):
        assert (
            range_create_statement(('RELATES_TO', 'RELATIONSHIP', 'uuid', 'RANGE'))
            == 'CREATE INDEX FOR ()-[e:RELATES_TO]-() ON (e.uuid)'
        )

    def test_fulltext_spec_raises_because_fulltext_is_issued_verbatim(self):
        """D1: fulltext is emitted as upstream emits it, never synthesized here.

        Synthesizing it would mean re-deriving the node-side
        ``CALL db.idx.fulltext.createNodeIndex`` form — including its 33-word
        stopwords map — which was measured to succeed VERBATIM against the same
        trap state.  Rebuilding a form that already works is unmeasured churn.
        """
        with pytest.raises(ValueError) as excinfo:
            range_create_statement(('Entity', 'NODE', 'name', 'FULLTEXT'))
        assert 'FULLTEXT' in str(excinfo.value)

    @pytest.mark.parametrize('spec', _expected_range_specs(), ids=lambda s: f'{s[0]}.{s[2]}')
    def test_never_emits_if_not_exists(self, spec):
        """``IF NOT EXISTS`` is a FalkorDB SYNTAX ERROR — adding it breaks every write."""
        assert 'IF NOT EXISTS' not in range_create_statement(spec).upper()

    @pytest.mark.parametrize('spec', _expected_range_specs(), ids=lambda s: f'{s[0]}.{s[2]}')
    def test_alpha_beta_seam_round_trip(self, spec):
        """What β WRITES must parse back, through α's parser, to exactly what it meant.

        This is the seam guard: β's synthesized form and α's ``parse_index_statement``
        are the two halves of "what should exist", and a divergence between them
        would make the diff permanently non-empty (β re-issuing statements forever
        for indices it just created).  Specs are derived from
        ``expected_index_set()``, so a graphiti upgrade widens this sweep
        automatically instead of leaving a stale hard-coded list passing.
        """
        assert parse_index_statement(range_create_statement(spec)) == [spec]
