"""The docs/legibility files a legibility project keeps."""
from __future__ import annotations

from pathlib import Path

LEGIBILITY_DIR = Path('docs') / 'legibility'
CONFIG_RELPATH = LEGIBILITY_DIR / 'legibility.yaml'
CODEBOOK_RELPATH = LEGIBILITY_DIR / 'confusion-codebook.yaml'
CENSUS_STATE_RELPATH = LEGIBILITY_DIR / 'census-state.json'


def config_path(project_root: str | Path) -> Path:
    return Path(project_root) / CONFIG_RELPATH


def codebook_path(project_root: str | Path) -> Path:
    return Path(project_root) / CODEBOOK_RELPATH


def census_state_path(project_root: str | Path) -> Path:
    return Path(project_root) / CENSUS_STATE_RELPATH
