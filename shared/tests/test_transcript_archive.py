"""Tests for shared.transcript_archive — best-effort per-task transcript archival."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from shared import transcript_archive as transcript_archive_module
from shared.transcript_archive import archive_task_transcripts
