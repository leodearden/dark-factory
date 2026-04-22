"""Ticket persistence store for two-phase add_task submit/resolve flow.

Tickets survive fused-memory restarts via SQLite (sibling DB to reconciliation.db).
On startup, any tickets left in 'pending' state from a prior run are marked as
'failed' with reason='server_restart'.
"""
