"""Wiring tests for the escalation-analytics endpoint + data.js registration.

Follows the source-assertion idiom established in test_tab_escalations.py:
static text checks against data.js (no JS runtime in this project) plus a
TestClient-driven route test against the real FastAPI app.
"""

from __future__ import annotations
