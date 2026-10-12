"""δ's corpus manifest, resolved against fetched episode bodies, into replay items.

The manifest records only episode uuids and content hashes; the bodies are read from
the store at replay time. Every manifest episode must be present and byte-identical
to what was hashed, or a run would measure a different corpus than the one its
``corpus_sha`` names. The hash function is injected so δ's definition
(``scripts/local_memory_models_eval/build_corpus.py::content_hash``) stays its one home.
"""

import hashlib
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol

from fused_memory.arm_harness.replay_types import ReplayItem


def corpus_sha(manifest_bytes: bytes) -> str:
    """The ``ArmSpec.corpus_sha`` of a manifest: sha256 hex of the file's exact bytes."""
    return hashlib.sha256(manifest_bytes).hexdigest()


class EpisodeLike(Protocol):
    """The fields of δ's ``EpisodeRecord`` a replay item is built from."""

    @property
    def uuid(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def source_description(self) -> str: ...

    @property
    def created_at(self) -> str: ...

    @property
    def content(self) -> str: ...


class CorpusIntegrityError(RuntimeError):
    """Manifest episodes were absent from the population, or their bodies drifted."""

    def __init__(self, missing_ids: tuple[str, ...], drifted_ids: tuple[str, ...]) -> None:
        self.missing_ids = missing_ids
        self.drifted_ids = drifted_ids
        super().__init__(
            f'corpus integrity failed: missing episodes {list(missing_ids)}, '
            f'content drifted from the manifest hash {list(drifted_ids)}'
        )


def select_replay_items(
    manifest: Mapping[str, Any],
    population: Iterable[EpisodeLike],
    *,
    content_hash: Callable[[str], str],
) -> tuple[ReplayItem, ...]:
    """The manifest's episodes, in manifest order; population records outside it are ignored."""
    entries = manifest['episodes']
    bodies = {episode.uuid: episode for episode in population}
    missing = tuple(entry['uuid'] for entry in entries if entry['uuid'] not in bodies)
    drifted = tuple(
        entry['uuid']
        for entry in entries
        if entry['uuid'] in bodies
        and content_hash(bodies[entry['uuid']].content) != entry['content_hash']
    )
    if missing or drifted:
        raise CorpusIntegrityError(missing, drifted)
    return tuple(_replay_item(bodies[entry['uuid']]) for entry in entries)


def _replay_item(episode: EpisodeLike) -> ReplayItem:
    return ReplayItem(
        episode_id=episode.uuid,
        name=episode.name,
        content=episode.content,
        source_description=episode.source_description,
        reference_time=_aware_utc(episode.created_at),
    )


def _aware_utc(created_at: str) -> datetime:
    """``created_at`` as aware UTC; a naive store value is read as UTC, as every writer stamps it."""
    parsed = datetime.fromisoformat(created_at)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
