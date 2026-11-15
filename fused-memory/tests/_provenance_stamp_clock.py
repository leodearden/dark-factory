"""A done_provenance stamp clock pinned so assertions are exact, not range checks.

Shared by the direct-call stamp tests (test_done_provenance.py) and the
interceptor-seam stamp tests (test_task_interceptor.py), which inject it as
``TaskInterceptor(provenance_stamp_clock=...)``.
"""

FIXED_STAMP = '2026-08-06T01:23:45.678901+00:00'


def fixed_stamp_clock() -> str:
    return FIXED_STAMP
