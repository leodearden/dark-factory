"""The one home of the scratch-namespace invariant.

Invariant: every graph or collection an arm writes, indexes, reads for topology,
or tears down is named ``evalmem_<lowercase-alnum-or-underscore>``. The guard is
an allow-list only, so the protected live graphs listed in
plans/local-memory-models-eval-prd.md §Hazards are unreachable by construction
and are never restated here. Every enforcement point calls
``require_scratch_name``; none re-derives the pattern.
"""

import re
from enum import StrEnum

SCRATCH_NAME_PATTERN = re.compile(r'evalmem_[a-z0-9_]+')
"""Use only via ``fullmatch``: an ``re.match`` with ``$`` accepts a trailing newline."""

_PATTERN_DISPLAY = f'^{SCRATCH_NAME_PATTERN.pattern}$'


class GuardCheckpoint(StrEnum):
    ARM_SPEC = 'arm-spec'
    REPLAY = 'replay'
    INDEX_BUILD = 'index-build'
    INDEX_PROBE = 'index-probe'
    TOPOLOGY_READ = 'topology-read'
    TEARDOWN_GRAPH = 'teardown-graph'
    TEARDOWN_COLLECTION = 'teardown-collection'
    GRAPH_COPY = 'graph-copy'
    REEMBED = 'reembed'
    INDEX_DROP = 'index-drop'
    REPLICA_BUILD = 'replica-build'
    SEARCH = 'search'


class ScratchGuardError(Exception):
    """A non-scratch name reached a guarded checkpoint.

    Deliberately not a ValueError: pydantic wraps a ValueError raised inside a
    validator into ValidationError, but lets any other exception propagate raw.
    """

    def __init__(self, name: object, checkpoint: GuardCheckpoint) -> None:
        self.name = name
        self.checkpoint = checkpoint
        super().__init__(
            f'scratch guard refused {name!r} at checkpoint {checkpoint.value!r}: '
            f'arm-harness names must match {_PATTERN_DISPLAY}'
        )


def require_scratch_name(name: object, *, checkpoint: GuardCheckpoint) -> str:
    if isinstance(name, str) and SCRATCH_NAME_PATTERN.fullmatch(name):
        return name
    raise ScratchGuardError(name, checkpoint)
