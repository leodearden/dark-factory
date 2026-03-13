"""Enumerations for the fused memory system."""

from enum import Enum


class MemoryCategory(str, Enum):
    """Six taxonomy categories from DESIGN.md."""

    entities_and_relations = 'entities_and_relations'
    temporal_facts = 'temporal_facts'
    decisions_and_rationale = 'decisions_and_rationale'
    preferences_and_norms = 'preferences_and_norms'
    procedural_knowledge = 'procedural_knowledge'
    observations_and_summaries = 'observations_and_summaries'


class SourceStore(str, Enum):
    """Which backend store a memory lives in."""

    graphiti = 'graphiti'
    mem0 = 'mem0'


class QueryType(str, Enum):
    """Read-router query classification."""

    entity_lookup = 'entity_lookup'
    temporal = 'temporal'
    relational = 'relational'
    preference = 'preference'
    procedural = 'procedural'
    broad = 'broad'


# Categories whose primary store is Graphiti
GRAPHITI_PRIMARY: set[MemoryCategory] = {
    MemoryCategory.entities_and_relations,
    MemoryCategory.temporal_facts,
    MemoryCategory.decisions_and_rationale,
}

# Categories whose primary store is Mem0
MEM0_PRIMARY: set[MemoryCategory] = {
    MemoryCategory.preferences_and_norms,
    MemoryCategory.procedural_knowledge,
    MemoryCategory.observations_and_summaries,
}
