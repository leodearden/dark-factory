"""The shipped ``orchestrator/defaults.yaml``, parsed for tests that pin to it.

Tests asserting that a config field, a prose bound or a derived budget agrees
with the packaged defaults read the file through this module, so they track
it rather than restating its values.
"""

from __future__ import annotations

import importlib.resources
from typing import Any

import yaml


def load_packaged_defaults() -> dict[str, Any]:
    defaults_file = importlib.resources.files('orchestrator') / 'defaults.yaml'
    defaults = yaml.safe_load(defaults_file.read_text())
    assert isinstance(defaults, dict), (
        f'{defaults_file} must parse to a mapping, got {type(defaults).__name__}'
    )
    return defaults
