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

from fused_memory.backends.falkor_indices import (
    UnparsedIndexStatementError,
    parse_index_statement,
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
