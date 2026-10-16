"""Guarded deletion of an arm's scratch graph and replica collection, one name per call.

INV-3: each helper re-checks the scratch guard at deletion time, even though ArmSpec
already validated the name, because a deletion path must not trust a value's
history. Neither helper accepts a list or a pattern, so there is no bulk-delete surface.
"""

from typing import Protocol

from fused_memory.arm_harness.arm_spec import EmbeddingArmSpec, LlmArmSpec
from fused_memory.arm_harness.scratch_guard import GuardCheckpoint, require_scratch_name


class DeletableGraph(Protocol):
    async def delete(self) -> None: ...


class GraphClient(Protocol):
    """The slice of falkordb's async FalkorDB client a graph teardown uses."""

    def select_graph(self, graph_id: str, /) -> DeletableGraph: ...


class CollectionClient(Protocol):
    """The slice of qdrant's AsyncQdrantClient a collection teardown uses."""

    async def delete_collection(self, collection_name: str, /) -> object: ...


async def delete_scratch_graph(falkor_client: GraphClient, name: str) -> None:
    require_scratch_name(name, checkpoint=GuardCheckpoint.TEARDOWN_GRAPH)
    await falkor_client.select_graph(name).delete()


async def delete_replica_collection(qdrant_client: CollectionClient, name: str) -> None:
    require_scratch_name(name, checkpoint=GuardCheckpoint.TEARDOWN_COLLECTION)
    await qdrant_client.delete_collection(name)


async def teardown_arm(
    falkor_client: GraphClient,
    qdrant_client: CollectionClient | None,
    spec: LlmArmSpec | EmbeddingArmSpec,
) -> None:
    """Delete the arm's scratch graph and, given a Qdrant client, its same-named replica."""
    await delete_scratch_graph(falkor_client, spec.scratch_group_id)
    if qdrant_client is not None:
        await delete_replica_collection(qdrant_client, spec.scratch_group_id)
