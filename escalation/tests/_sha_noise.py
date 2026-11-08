"""SHA-shaped tokens that name no commit, as an escalation's prose carries them.

``escalation/src/escalation/sha_citations.py::CITED_SHA`` matches every one, so
a record full of them must still leave room under
``escalation/src/escalation/sha_citations.py::MAX_CITED_SHAS_RESOLVED`` for the
SHAs that do name commits.
"""

from __future__ import annotations

# Each contributes two SHA-shaped tokens (its 8- and 12-hex segments).
UUIDS = (
    '3f2a9c1e-4b7d-4e21-9a0c-5c8d0b7f1a33',
    '7b1d2e3f-0a9c-4d8e-b7f6-1e2d3c4b5a69',
    'c0ffee12-3456-4789-abcd-ef0123456789',
    'deadbeef-cafe-4bab-8e5e-0123456789ab',
    '1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d',
    '9f8e7d6c-5b4a-4392-8170-6f5e4d3c2b1a',
)
NOISE = f"{' '.join(UUIDS)} pid 4194304 on 20261023 hash 9c1e7b4d5a609e21"
