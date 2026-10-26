"""How a census report section's lines quote a codebook record's title."""
from __future__ import annotations


def quoted_title(title: str) -> str:
    """*title* on one line, in double quotes."""
    return '"' + " ".join(title.split()) + '"'
