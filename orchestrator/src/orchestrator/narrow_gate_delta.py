"""Read a commit range's tree delta from git and classify it for the narrow merge gate (rationale: docs/narrow-merge-gate.md).

The docs lane's class-A entry point (task 10996) is classify_branch_change combined with classify_docs_change.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from shared.git_async import run_git

from orchestrator.narrow_gate_class import (
    ChangedPath,
    ChangeStatus,
    GateClassification,
    MissReason,
    classify_change,
    full_gate,
    is_docs_path,
)

logger = logging.getLogger(__name__)

GIT_READ_TIMEOUT_SECS = 60.0

_STDERR_TAIL_CHARS = 300
# A typechange is a mode change, so it reads as 'M' with differing modes; any other letter is unreadable.
_STATUS_BY_GIT_LETTER: Mapping[str, ChangeStatus] = {'A': 'A', 'M': 'M', 'D': 'D', 'T': 'M'}
_OBJECT_ID = re.compile(r'[0-9a-f]{40}|[0-9a-f]{64}')
_BLOB_HEADER = re.compile(rb'blob (\d+)')
# The null object id is never a real object, so git answers it 'missing'. Requested last, it keeps
# run_git's stdout strip off the final blob.
_BATCH_END = '0' * 40

BlobPair = tuple[str | None, str | None]


async def read_tree_delta(repo: Path, base: str, head: str) -> tuple[ChangedPath, ...] | None:
    """Every path that differs between *base* and *head*, renames split into a deletion and an addition; None when git cannot say."""
    out = await _git_stdout(repo, ['diff', '--raw', '--no-renames', '--no-color', '-z', base, head])
    return None if out is None else _parse_raw_delta(out)


async def read_head(repo: Path) -> str | None:
    """The commit *repo*'s HEAD names; None when git cannot say."""
    head = await _git_stdout(repo, ['rev-parse', '--verify', 'HEAD'])
    return head if head is not None and _OBJECT_ID.fullmatch(head) else None


async def list_tree_paths(repo: Path, rev: str) -> tuple[str, ...] | None:
    """Every file path in *rev*'s tree; None when git cannot say."""
    out = await _git_stdout(repo, ['ls-tree', '-r', '-z', '--name-only', rev])
    return None if out is None else tuple(path for path in out.split('\0') if path)


async def classify_tree_delta(repo: Path, base: str, head: str, docs_paths: Sequence[str]) -> GateClassification:
    """The gate the change *base*..*head* qualifies for, with both sides' blobs read from git, never the working tree."""
    delta = await read_tree_delta(repo, base, head)
    if delta is None:
        return full_gate(MissReason.DELTA_UNREADABLE)
    to_read = [c.path for c in delta if c.status == 'M' and not is_docs_path(c.path, docs_paths)]
    pairs = await _read_blob_pairs(repo, base, head, to_read) if to_read else {}
    return classify_change(delta, docs_paths, lambda path: pairs.get(path, (None, None)))


async def classify_branch_change(
    repo: Path, main_ref: str, branch_ref: str, docs_paths: Sequence[str],
) -> GateClassification:
    """The gate *branch_ref*'s own change qualifies for: merge-base(*main_ref*, *branch_ref*)..*branch_ref*."""
    merge_base = await _git_stdout(repo, ['merge-base', main_ref, branch_ref])
    if merge_base is None or _OBJECT_ID.fullmatch(merge_base) is None:
        return full_gate(MissReason.DELTA_UNREADABLE)
    return await classify_tree_delta(repo, merge_base, branch_ref, docs_paths)


async def narrow_member_ids(
    repo: Path, main_ref: str, branch_refs: Mapping[str, str], docs_paths: Sequence[str],
) -> frozenset[str]:
    """The ids in *branch_refs* whose branch qualifies for a narrow gate; one whose classification raised is not narrow."""
    ids = list(branch_refs)
    results = await asyncio.gather(
        *(classify_branch_change(repo, main_ref, branch_refs[member_id], docs_paths) for member_id in ids),
        return_exceptions=True,
    )
    narrow: set[str] = set()
    for member_id, result in zip(ids, results, strict=True):
        if isinstance(result, GateClassification):
            if result.resolution.is_narrow:
                narrow.add(member_id)
        elif isinstance(result, Exception):
            logger.warning('narrow gate: classifying %s (%s) raised; it is not narrow', member_id,
                           branch_refs[member_id], exc_info=result)
        else:
            raise result
    return frozenset(narrow)


async def _git_stdout(repo: Path, args: Sequence[str], *, input_text: str | None = None) -> str | None:
    argv = ['git', *args]
    try:
        result = await run_git(argv, cwd=repo, input_text=input_text, timeout=GIT_READ_TIMEOUT_SECS)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning('narrow gate: %s could not run in %s: %s', ' '.join(argv[:3]), repo, exc)
        return None
    if not result.ok:
        logger.warning(
            'narrow gate: %s failed in %s (rc=%s, timed_out=%s): %s', ' '.join(argv[:3]), repo,
            result.returncode, result.timed_out, result.stderr[-_STDERR_TAIL_CHARS:],
        )
        return None
    return result.stdout


def _parse_raw_delta(out: str) -> tuple[ChangedPath, ...] | None:
    fields = out.split('\0')
    if fields[-1] == '':
        fields.pop()
    if len(fields) % 2:
        return None
    changes = [_changed_path(meta, path) for meta, path in zip(fields[0::2], fields[1::2], strict=True)]
    if any(change is None for change in changes):
        return None
    return tuple(change for change in changes if change is not None)


def _changed_path(meta: str, path: str) -> ChangedPath | None:
    parts = meta.removeprefix(':').split(' ')
    if not meta.startswith(':') or len(parts) != 5:
        return None
    old_mode, new_mode, _old_oid, _new_oid, letter = parts
    status = _STATUS_BY_GIT_LETTER.get(letter)
    return None if status is None else ChangedPath(path, status, old_mode, new_mode)


async def _read_blob_pairs(repo: Path, base: str, head: str, paths: Sequence[str]) -> dict[str, BlobPair]:
    readable = [path for path in paths if '\n' not in path]
    requests = [f'{rev}:{path}' for path in readable for rev in (base, head)]
    batch = ''.join(f'{request}\n' for request in (*requests, _BATCH_END))
    out = await _git_stdout(repo, ['cat-file', '--batch=%(objecttype) %(objectsize)'], input_text=batch)
    blobs = None if out is None else _parse_blob_batch(out.encode(), requests)
    if blobs is None:
        return {}
    return {path: (blobs[2 * i], blobs[2 * i + 1]) for i, path in enumerate(readable)}


def _parse_blob_batch(data: bytes, requests: Sequence[str]) -> list[str | None] | None:
    blobs: list[str | None] = []
    pos = 0
    for request in requests:
        eol = data.find(b'\n', pos)
        if eol < 0:
            return None
        header, pos = data[pos:eol], eol + 1
        if header == f'{request} missing'.encode():
            blobs.append(None)
            continue
        match = _BLOB_HEADER.fullmatch(header)
        end = pos + int(match.group(1)) if match else -1
        if match is None or data[end:end + 1] != b'\n':
            return None
        blobs.append(data[pos:end].decode())
        pos = end + 1
    return blobs if data[pos:] == f'{_BATCH_END} missing'.encode() else None
