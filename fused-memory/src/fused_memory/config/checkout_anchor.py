"""Where a relative fused-memory data path is anchored: the checkout this package was imported from, never the process CWD (task 5462)."""

import os
from pathlib import Path

#: config -> fused_memory -> src -> fused-memory -> checkout.
CHECKOUT_ROOT: Path = Path(__file__).resolve().parents[4]


def anchor_at_checkout(path: str) -> str:
    if os.path.isabs(path):
        return path
    return str(CHECKOUT_ROOT / path)
