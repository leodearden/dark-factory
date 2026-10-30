"""An embedder for graphiti bulk writes whose every embedding is preset.

``graphiti_core.utils.bulk_utils.add_nodes_and_edges_bulk_tx`` takes an
``EmbedderClient`` and embeds each entity node or edge whose embedding is
``None``.  Tests that preset every embedding pass ``UnusedEmbedder`` so that a
missing one fails the test instead of reaching an embedding provider.
"""

from __future__ import annotations

from typing import Any

from graphiti_core.embedder import EmbedderClient


class UnusedEmbedder(EmbedderClient):
    """Every embedding is preset, so graphiti must never embed."""

    async def create(self, input_data: Any) -> list[float]:
        raise AssertionError('embeddings are preset; graphiti must not call the embedder')
