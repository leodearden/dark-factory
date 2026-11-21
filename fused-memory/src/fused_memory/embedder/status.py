"""The get_status blocks that report the embedder identity.

``embedder`` is the startup endpoint probe's cached result; ``config_overlay``
is the live config's overlay provenance, present only while an overlay is
active.
"""

from typing import Any

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.embedder.endpoint_probe import EmbedderEndpointProbe


def embedder_identity_status(
    config: FusedMemoryConfig, probe: EmbedderEndpointProbe,
) -> dict[str, Any]:
    status: dict[str, Any] = {'embedder': probe.status_block()}
    if config.config_overlay is not None:
        status['config_overlay'] = config.config_overlay.model_dump(mode='json')
    return status
