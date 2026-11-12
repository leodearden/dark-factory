"""Assert a prose clip from the outside, so a site test needs no knowledge of the site's cap."""

from os.path import commonprefix

from fused_memory.utils.clip_disclosure import elision_marker


def assert_elided(rendered: str, original: str) -> None:
    """*rendered* is a non-empty head of *original* followed by a marker counting exactly what was cut."""
    head = commonprefix([rendered, original])
    assert head, f'nothing of {original[:40]!r} survives in {rendered[:80]!r}'
    assert rendered[len(head):] == elision_marker(len(original) - len(head)), rendered[-80:]
