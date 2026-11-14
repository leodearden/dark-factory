"""One reconciliation run's finding index: its stage session and its dedup keys.

Invariant: every key names a finding held by the reachable entry that owns it,
every finding a reachable entry holds is owned by that entry, and each finding's
signature, description and cited-task keys name it.  :meth:`RunFindingIndex.violations`
reports breaches.  Contract: plans/review-all-s13-recon-sweeps-flag-identity-and-recon-report-state-prd.md
§Contracts C4.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar

from fused_memory.models.reconciliation import StageId
from fused_memory.reconciliation.flag_identity import (
    description_digest,
    normalize_description,
    parse_flag_subject,
    recon_signature_task_id,
    signature_field,
)

Signature = tuple[str | None, str | None]


class FindingRecord(Protocol):
    @property
    def finding_id(self) -> str: ...
    @property
    def task_id(self) -> str | None: ...
    @property
    def flag_type(self) -> str | None: ...
    @property
    def description(self) -> str: ...
    @property
    def cited_tasks(self) -> Sequence[Mapping[str, Any]]: ...


class EntryRecord(Protocol):
    @property
    def run_id(self) -> str: ...
    @property
    def stage(self) -> str: ...
    @property
    def findings(self) -> Sequence[FindingRecord]: ...


EntryT = TypeVar('EntryT', bound=EntryRecord)


@dataclass(frozen=True)
class FindingKeys:
    signature: Signature | None = None
    desc_hash: str | None = None
    cited_task: str | None = None
    derived_signature: tuple[str, str | None] | None = None


@dataclass(frozen=True)
class PersistedSlice:
    """One entry's owned keys, in the four maps its ``entry_json`` row carries."""

    signature_to_finding: Mapping[Signature, str] = field(default_factory=dict)
    deschash_to_finding: Mapping[str, str] = field(default_factory=dict)
    sig_anchor_slice: Mapping[Signature, str] = field(default_factory=dict)
    cited_task_slice: Mapping[str, str] = field(default_factory=dict)


def cited_task_key(project_id: object, task_id: object) -> str:
    return f'{project_id}:{task_id}'


def _derive_keys(
    task_id: str | None,
    flag_type: str | None,
    description: str,
    primary_citation: Mapping[str, Any] | None,
    owner_stage: str | None,
) -> FindingKeys:
    subject = parse_flag_subject(task_id)
    c_task_id = recon_signature_task_id(subject)
    c_flag_type = signature_field(flag_type)
    is_null_null = c_task_id is None and c_flag_type is None
    signature = None if is_null_null else (c_task_id, c_flag_type)
    desc_hash = (
        description_digest(description)
        if is_null_null and normalize_description(description)
        else None
    )
    cited_task = None
    derived_signature = None
    if primary_citation is not None:
        if (
            c_task_id is None
            and owner_stage is not None
            and owner_stage != StageId.memory_consolidator
        ):
            cited_task = cited_task_key(
                primary_citation['project_id'], primary_citation['task_id']
            )
        cited_task_id = signature_field(primary_citation['task_id'])
        if (
            c_flag_type is not None
            and cited_task_id is not None
            and (c_task_id is None or cited_task_id in subject.components)
        ):
            derived_signature = (cited_task_id, c_flag_type)
    return FindingKeys(signature, desc_hash, cited_task, derived_signature)


class RunFindingIndex(Generic[EntryT]):
    """The session (which stages are registered, resident and active) and the
    dedup maps of one run.  Keys have one owner each: the first registrant wins
    and nothing here overwrites an owner."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id
        self._registered: dict[str, EntryT] = {}
        self._resident: dict[str, EntryT] = {}
        self._resident_view: Mapping[str, EntryT] = MappingProxyType(self._resident)
        self._active_stage: str | None = None
        self._by_signature: dict[Signature, str] = {}
        self._by_desc: dict[str, str] = {}
        self._by_cited_task: dict[str, str] = {}
        self._owners: dict[str, EntryT] = {}

    @property
    def run_id(self) -> str:
        return self._run_id

    # -- session ----------------------------------------------------------

    def install(self, entry: EntryT) -> EntryT | None:
        """Register *entry* as its stage's entry and make it resident.

        Returns the different entry it displaced from that stage, if any.
        """
        if entry.run_id != self._run_id:
            raise ValueError(
                f'run {self._run_id!r}: cannot install an entry of run {entry.run_id!r}'
            )
        displaced = self._registered.get(entry.stage)
        self._registered[entry.stage] = entry
        self._resident[entry.stage] = entry
        return displaced if displaced is not entry else None

    def activate(self, stage: str) -> None:
        if stage not in self._resident:
            raise ValueError(
                f'run {self._run_id!r}: cannot activate stage {stage!r}, which is not resident'
            )
        self._active_stage = stage

    @property
    def active_stage(self) -> str | None:
        return self._active_stage

    def evict(self, stage: str) -> None:
        """Make *stage* non-resident; its entry stays reachable."""
        if stage not in self._resident:
            raise ValueError(
                f'run {self._run_id!r}: cannot evict stage {stage!r}, which is not resident'
            )
        del self._resident[stage]
        if self._active_stage == stage:
            self._active_stage = None

    @property
    def quiescent(self) -> bool:
        return not self._resident

    def reachable_entries(self) -> list[EntryT]:
        """Each stage's latest entry, live or evicted: what persistence writes."""
        return list(self._registered.values())

    def resident_entry(self, stage: str) -> EntryT | None:
        return self._resident.get(stage)

    @property
    def resident_entries(self) -> Mapping[str, EntryT]:
        return self._resident_view

    # -- keys -------------------------------------------------------------

    def keys_for(self, finding: FindingRecord) -> FindingKeys:
        owner = self._owners.get(finding.finding_id)
        return _derive_keys(
            finding.task_id,
            finding.flag_type,
            finding.description,
            finding.cited_tasks[0] if finding.cited_tasks else None,
            owner.stage if owner is not None else None,
        )

    def citation_keys(
        self, finding: FindingRecord, *, project_id: str, task_id: str
    ) -> FindingKeys:
        """The keys *finding* would hold if this citation became its primary.

        A finding that already has a primary citation gets no citation keys.
        """
        if finding.cited_tasks:
            return replace(self.keys_for(finding), cited_task=None, derived_signature=None)
        owner = self._owners.get(finding.finding_id)
        return _derive_keys(
            finding.task_id,
            finding.flag_type,
            finding.description,
            {'project_id': project_id, 'task_id': task_id},
            owner.stage if owner is not None else None,
        )

    # -- registration -----------------------------------------------------

    def register(self, finding: FindingRecord, entry: EntryT) -> FindingKeys:
        """Make *entry*, which holds *finding*, its owner and register its keys."""
        self._require_held(finding, entry, 'register')
        owner = self._owners.setdefault(finding.finding_id, entry)
        if owner is not entry:
            raise ValueError(
                f'run {self._run_id!r}: cannot register finding {finding.finding_id!r} '
                f'to stage {entry.stage!r}; stage {owner.stage!r} owns it'
            )
        keys = self.keys_for(finding)
        for _name, index, key, _required in self._keys_by_map(keys):
            if key is not None:
                index.setdefault(key, finding.finding_id)
        return keys

    def register_citation(self, finding: FindingRecord) -> None:
        """Register the keys *finding*'s primary citation adds.  *finding* must
        be registered and still held by its owner."""
        owner = self._owners.get(finding.finding_id)
        if owner is None:
            raise ValueError(
                f'run {self._run_id!r}: cannot register a citation of finding '
                f'{finding.finding_id!r}, which is not registered'
            )
        self._require_held(finding, owner, 'register a citation of')
        keys = self.keys_for(finding)
        if keys.cited_task is not None:
            self._by_cited_task.setdefault(keys.cited_task, finding.finding_id)
        if keys.derived_signature is not None:
            self._by_signature.setdefault(keys.derived_signature, finding.finding_id)

    def unregister(self, finding: FindingRecord) -> None:
        if finding.finding_id not in self._owners:
            return
        for _name, index, key, _required in self._keys_by_map(self.keys_for(finding)):
            if key is not None and index.get(key) == finding.finding_id:
                del index[key]
        del self._owners[finding.finding_id]

    def _require_held(self, finding: FindingRecord, entry: EntryT, operation: str) -> None:
        if not any(held is finding for held in entry.findings):
            raise ValueError(
                f'run {self._run_id!r}: cannot {operation} finding {finding.finding_id!r}, '
                f'which stage {entry.stage!r} does not hold'
            )

    def _named_maps(self) -> tuple[tuple[str, dict[Any, str]], ...]:
        return (
            ('signature', self._by_signature),
            ('desc', self._by_desc),
            ('cited-task', self._by_cited_task),
        )

    def _keys_by_map(self, keys: FindingKeys) -> tuple[tuple[str, dict[Any, str], Any, bool], ...]:
        """``(name, map, key, required)`` per kind of *keys*.

        A live finding need not own its derived signature: that key is
        projectless, so it has one owner per run and the first registrant wins.
        """
        return (
            ('signature', self._by_signature, keys.signature, True),
            ('derived signature', self._by_signature, keys.derived_signature, False),
            ('desc', self._by_desc, keys.desc_hash, True),
            ('cited-task', self._by_cited_task, keys.cited_task, True),
        )

    # -- lookups ----------------------------------------------------------

    def by_signature(self, signature: Signature | None) -> str | None:
        return None if signature is None else self._by_signature.get(signature)

    def by_desc(self, desc_hash: str | None) -> str | None:
        return None if desc_hash is None else self._by_desc.get(desc_hash)

    def by_cited_task(self, key: str | None) -> str | None:
        return None if key is None else self._by_cited_task.get(key)

    def owner_of(self, finding_id: str) -> EntryT | None:
        return self._owners.get(finding_id)

    def flag_types_for_task(self, task_id: str) -> frozenset[str]:
        return frozenset(
            flag_type
            for (sig_task_id, flag_type) in self._by_signature
            if sig_task_id == task_id and flag_type is not None
        )

    # -- persistence ------------------------------------------------------

    def owned_slice(self, entry: EntryT) -> PersistedSlice:
        held = {
            f.finding_id: f for f in entry.findings if self._owners.get(f.finding_id) is entry
        }
        ordinary = {fid: self.keys_for(f).signature for fid, f in held.items()}
        signature_to_finding: dict[Signature, str] = {}
        sig_anchor_slice: dict[Signature, str] = {}
        for sig, fid in self._by_signature.items():
            if fid in held:
                target = signature_to_finding if sig == ordinary[fid] else sig_anchor_slice
                target[sig] = fid
        return PersistedSlice(
            signature_to_finding=signature_to_finding,
            deschash_to_finding={h: fid for h, fid in self._by_desc.items() if fid in held},
            sig_anchor_slice=sig_anchor_slice,
            cited_task_slice={k: fid for k, fid in self._by_cited_task.items() if fid in held},
        )

    def absorb(self, entry: EntryT, persisted: PersistedSlice) -> None:
        """Adopt *entry*'s findings and the persisted keys that name them."""
        held = {f.finding_id for f in entry.findings}
        for fid in held:
            self._owners.setdefault(fid, entry)

        def adopt(index: dict[Any, str], persisted_keys: Mapping[Any, str]) -> None:
            for key, fid in persisted_keys.items():
                if fid in held:
                    index.setdefault(key, fid)

        adopt(self._by_signature, persisted.signature_to_finding)
        adopt(self._by_signature, persisted.sig_anchor_slice)
        adopt(self._by_desc, persisted.deschash_to_finding)
        adopt(self._by_cited_task, persisted.cited_task_slice)

    # -- invariant --------------------------------------------------------

    def violations(self) -> list[str]:
        """One message per breach of the module invariant; empty when it holds."""
        live = {
            f.finding_id: f
            for entry in self._registered.values()
            for f in entry.findings
            if self._owners.get(f.finding_id) is entry
        }
        breaches = [
            *self._session_breaches(),
            *self._ownership_breaches(live),
            *self._key_breaches({fid: self.keys_for(f) for fid, f in live.items()}),
        ]
        return [f'run {self._run_id!r}: {breach}' for breach in breaches]

    def _session_breaches(self) -> Iterator[str]:
        for stage, entry in self._resident.items():
            if self._registered.get(stage) is not entry:
                yield f'resident stage {stage!r} is not its registered entry'
        if self._active_stage is not None and self._active_stage not in self._resident:
            yield f'active stage {self._active_stage!r} is not resident'

    def _ownership_breaches(self, live: Mapping[str, FindingRecord]) -> Iterator[str]:
        for entry in self._registered.values():
            for f in entry.findings:
                if self._owners.get(f.finding_id) is not entry:
                    yield (
                        f'finding {f.finding_id!r} held by stage {entry.stage!r} '
                        'is not registered to it'
                    )
        for fid, owner in self._owners.items():
            if fid not in live:
                yield (
                    f'finding {fid!r} is registered to stage {owner.stage!r} '
                    'but no reachable entry holds it'
                )

    def _key_breaches(self, derived: Mapping[str, FindingKeys]) -> Iterator[str]:
        for name, index in self._named_maps():
            for key, fid in index.items():
                if fid not in derived:
                    yield f'{name} key {key!r} names {fid!r}, which no reachable entry holds'
                elif not any(
                    owned is index and owned_key == key
                    for _n, owned, owned_key, _r in self._keys_by_map(derived[fid])
                ):
                    yield f'stale {name} key {key!r} names {fid!r}'
        for fid, keys in derived.items():
            for name, index, key, required in self._keys_by_map(keys):
                if required and key is not None and index.get(key) != fid:
                    yield f'{name} key {key!r} of finding {fid!r} is missing'
