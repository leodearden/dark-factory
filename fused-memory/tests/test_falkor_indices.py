"""Unit tests for ``fused_memory.backends.falkor_indices`` (task 3706, α).

What this module pins
---------------------
Both halves of the PRD's normal-form seam, consumed later by β (``ensure_indices``)
and δ (``summarize_index_health``):

* the EXPECTED side — ``expected_index_set()``, derived by parsing the statements
  graphiti itself emits, never a hand-copied list (INV-5);
* the ACTUAL side — ``normalize_index_records()``, which fans a ``list_indices()``
  record out to one tuple per (property, index_type) pair.

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

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices

from fused_memory.backends.falkor_indices import (
    UnparsedIndexStatementError,
    expected_index_set,
    parse_index_statement,
)


def _upstream_statements() -> list[str]:
    """Every index statement the INSTALLED graphiti emits for FalkorDB.

    Built here in the test, independently of the module under test, so the
    zero-unparsed-remainder assertion compares the parser against upstream
    rather than against itself.
    """
    return list(get_range_indices(GraphProvider.FALKORDB)) + list(
        get_fulltext_indices(GraphProvider.FALKORDB)
    )

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

    def test_count_is_a_change_detector_not_the_contract(self):
        """CHANGE-DETECTOR ONLY — NOT the contract.

        Measured against **graphiti-core 0.28.2** on 2026-08-06: 38 specs, 26
        RANGE + 12 FULLTEXT.

        On a LEGITIMATE upstream change (graphiti adds or drops an index), UPDATE
        these numbers — they are a cheap "something moved upstream, go look"
        signal and nothing more.  The assertion that must NEVER be weakened is
        ``test_every_upstream_statement_parses_with_zero_remainder``, which is
        derived from upstream rather than copied from it.  Asserting a
        hand-copied total AS the contract is exactly the lock-step duplication
        this task exists to prevent (INV-5 / PRD D3).
        """
        expected = expected_index_set()
        assert len(expected) == 38
        assert len([s for s in expected if s[3] == 'RANGE']) == 26
        assert len([s for s in expected if s[3] == 'FULLTEXT']) == 12


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
        neo4j_statements = get_range_indices('falkordb')
        assert len(neo4j_statements) == 27
        assert all('IF NOT EXISTS' in s for s in neo4j_statements)

    def test_enum_member_returns_the_falkordb_set(self):
        falkor_statements = get_range_indices(GraphProvider.FALKORDB)
        assert len(falkor_statements) == 9
        # IF NOT EXISTS is a FalkorDB syntax error, so its absence is load-bearing.
        assert not any('IF NOT EXISTS' in s for s in falkor_statements)

    def test_the_neo4j_set_is_unparseable_which_is_what_makes_it_a_tripwire(self):
        for statement in get_range_indices('falkordb'):
            with pytest.raises(UnparsedIndexStatementError):
                parse_index_statement(statement)
