"""How a task's new tip relates, by content, to the tip it already has in the lane.

PRD ``plans/one-merge-request-per-task-prd.md`` D6 names four relations against
a target: SAME (one sha), OLDER (the new tip is an ancestor of the old one),
REBASED (both ranges beyond the target carry the same patch ids and neither
holds a merge commit) and DIFFERENT (anything else).  A question git cannot
answer raises :class:`ContentUnclassifiable` rather than returning a relation,
so the caller decides what an unanswered question means.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

from orchestrator.merge_lane.git_read import (
    GitReadFailed,
    is_ancestor_or_raise,
    resolve_commit,
)
from orchestrator.merge_lane.landing_evidence import (
    RangePatchIds,
    is_valid_sha_40,
    range_patch_ids,
)

if TYPE_CHECKING:
    from orchestrator.git_ops import GitOps


class ContentRelation(enum.StrEnum):
    SAME = 'same'
    REBASED = 'rebased'
    OLDER = 'older'
    DIFFERENT = 'different'


class ContentUnclassifiable(RuntimeError):
    """Git could not answer a read :func:`classify_content` needed."""

    def __init__(self, *, old_tip: str, new_tip: str, target: str, stderr: str) -> None:
        super().__init__(
            f'cannot relate new tip {new_tip} to old tip {old_tip} over {target}: '
            f'{stderr.strip()}'
        )
        self.old_tip = old_tip
        self.new_tip = new_tip
        self.target = target
        self.stderr = stderr


async def classify_content(
    git_ops: GitOps, *, target: str, old_tip: str, new_tip: str,
) -> ContentRelation:
    """The D6 relation of *new_tip* to *old_tip*, cheapest question first.

    Both tips must be full commit shas, because SAME is their string equality;
    any other spelling raises ``ValueError``.  *target* may be any commit-ish:
    it is resolved once, so both range reads measure against one snapshot of it.
    """
    _require_full_sha('old_tip', old_tip)
    _require_full_sha('new_tip', new_tip)
    if old_tip == new_tip:
        return ContentRelation.SAME
    try:
        if await is_ancestor_or_raise(git_ops, ancestor=new_tip, descendant=old_tip):
            return ContentRelation.OLDER
        target_sha = await resolve_commit(git_ops, target)
        old_range = await range_patch_ids(git_ops, target=target_sha, tip=old_tip)
        new_range = await range_patch_ids(git_ops, target=target_sha, tip=new_tip)
    except GitReadFailed as exc:
        raise ContentUnclassifiable(
            old_tip=old_tip, new_tip=new_tip, target=target, stderr=exc.stderr,
        ) from exc
    return _relation_of_ranges(old_range, new_range)


def _require_full_sha(name: str, value: str) -> None:
    if not is_valid_sha_40(value):
        raise ValueError(
            f'classify_content needs {name} as a full 40-hex commit sha; got {name}={value!r}',
        )


def _relation_of_ranges(old: RangePatchIds, new: RangePatchIds) -> ContentRelation:
    if old.has_merge_commit or new.has_merge_commit or old.patch_ids != new.patch_ids:
        return ContentRelation.DIFFERENT
    return ContentRelation.REBASED
