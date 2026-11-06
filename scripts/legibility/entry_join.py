"""Joining a pending candidate onto an entry that already stands, instead of
promoting it to an entry of its own: the codebook transform the census key
rule, the similarity attach and a queue member following its representative
share (plans/census-incremental-prd.md L6, L7)."""
from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any

from legibility import census_findings, codebook


def _record_by_id(records: Iterable[dict], record_id: str, *, kind: str) -> dict:
    for record in records:
        if record.get("id") == record_id:
            return record
    raise ValueError(f"join_entry: no {kind} {record_id!r}")


def join_entry(
    cb: dict,
    candidate_id: str,
    entry_id: str,
    *,
    candidate_fields: Mapping[str, Any],
    sighting_fields: Mapping[str, Any],
) -> dict:
    """A new codebook in which pending *candidate_id* is promoted onto entry
    *entry_id* instead of to an entry of its own, carrying *candidate_fields*.

    The candidate takes the entry's finding identity when it has one; each of
    its sightings from a session the entry does not carry is appended to the
    entry with *sighting_fields*."""
    result = copy.deepcopy(cb)
    candidate = _record_by_id(result.get("candidates") or [], candidate_id, kind="candidate")
    if candidate.get("disposition") != "pending":
        raise ValueError(
            f"join_entry: candidate {candidate_id!r} is {candidate.get('disposition')!r}, not pending",
        )
    target = _record_by_id(result.get("entries") or [], entry_id, kind="entry")
    identity = census_findings.FindingIdentity.from_record(target)
    if identity is not None:
        candidate.update(census_findings.stamped(candidate, identity))
    candidate.update(copy.deepcopy(dict(candidate_fields)), disposition="promoted", promoted_to=target["id"])
    sightings = target.setdefault("sightings", [])
    carried = set(codebook.sessions_of(target))
    for sighting in candidate.get("sightings") or []:
        if sighting.get("session") in carried:
            continue
        carried.add(sighting.get("session"))
        sightings.append({**copy.deepcopy(sighting), **copy.deepcopy(dict(sighting_fields))})

    codebook.assert_no_deletion(cb, result)
    return result
