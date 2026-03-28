"""Shared test utility helpers for dashboard tests.

This module provides helper functions used across multiple test files.
It is intentionally separate from conftest.py, which is reserved for
pytest fixtures and hooks.
"""

from __future__ import annotations

import re


def _get_opening_tag(html: str, marker: str) -> str:
    """Return the opening HTML tag that contains *marker*.

    Uses a regex pattern ``<[^>]+<marker>[^>]*>`` so that:

    * Multi-line tags are handled correctly — ``[^>]`` matches any character
      except ``>``, including newlines, without needing ``re.DOTALL``.
    * Tag boundaries are respected — the match cannot span past a ``>``
      character, so it cannot accidentally merge two tags.
    * Special regex characters in *marker* are safely escaped via
      ``re.escape``.

    Args:
        html: The HTML string to search.
        marker: A literal substring that must appear inside the opening tag
            (e.g. ``'data-updated-for="orchestrators"'``).

    Returns:
        The full opening tag string, from the leading ``<`` to the closing
        ``>``, inclusive.

    Raises:
        ValueError: If no opening tag containing *marker* is found.
    """
    pattern = r'<[^>]*' + re.escape(marker) + r'[^>]*>'
    match = re.search(pattern, html)
    if match is None:
        raise ValueError(
            f'No opening tag containing {marker!r} found in HTML'
        )
    return match.group(0)
