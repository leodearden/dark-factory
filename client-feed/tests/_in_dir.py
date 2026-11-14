"""The portal's half of the boundary, in/engagement.json and in/inbox.jsonl, written through the contract models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from client_feed.contract import (
    ENGAGEMENT_FILENAME,
    INBOX_FILENAME,
    EngagementProjection,
    RevealRequested,
)


def write_engagement(
    in_dir: Path,
    *,
    project_id: str,
    version: int = 1,
    stage: str = 'building',
    delivered: str | None = None,
) -> Path:
    in_dir.mkdir(parents=True, exist_ok=True)
    path = in_dir / ENGAGEMENT_FILENAME
    projection = EngagementProjection(
        engagement_version=version, project_id=project_id, stage=stage, delivered_commit=delivered
    )
    path.write_text(projection.model_dump_json())
    return path


def _requests_so_far(inbox: Path) -> int:
    if not inbox.exists():
        return 0
    return len(inbox.read_bytes().splitlines())


def request_reveals(in_dir: Path, *paths: str, at: datetime) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    inbox = in_dir / INBOX_FILENAME
    first = _requests_so_far(inbox) + 1
    lines = ''.join(
        RevealRequested(seq=seq, at=at, path=path).model_dump_json() + '\n'
        for seq, path in enumerate(paths, first)
    )
    with inbox.open('a') as handle:
        handle.write(lines)
