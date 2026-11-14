"""The live payload predicate's summary must fit the deterministic runner's note cap.

``dashboard.payload_check`` ends its output with one compact JSON summary,
which the deterministic runner keeps as the link's done note only if it fits
``PREDICATE_NOTE_MAX_PAYLOAD_CHARS``; a longer one is replaced wholesale by an
"elided" marker. The two packages do not import each other, so this directory,
where both are importable, is the one place their caps can be compared.
"""

from dashboard.payload_check import SUMMARY_MAX_CHARS
from orchestrator.predicate_output import PREDICATE_NOTE_MAX_PAYLOAD_CHARS


def test_the_payload_summary_fits_the_runners_note_cap() -> None:
    assert SUMMARY_MAX_CHARS <= PREDICATE_NOTE_MAX_PAYLOAD_CHARS, (
        f'dashboard.payload_check.SUMMARY_MAX_CHARS is {SUMMARY_MAX_CHARS}, but the '
        f'deterministic runner keeps a done note of at most '
        f'{PREDICATE_NOTE_MAX_PAYLOAD_CHARS} chars, so every summary near the cap '
        'would be elided. Lower SUMMARY_MAX_CHARS to the runner\'s cap.'
    )
