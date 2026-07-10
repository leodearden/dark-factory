"""Tests for orchestrator.verify_cmd — structured VerifyCmd command model.

Task β of the verify-plan PRD (plans/verify-plan-prd.md §Contract·VerifyCmd).
Replaces verify.py's raw-shell-string find/replace-surgery command model with
``parse_config_command`` (raw string -> VerifyCmd), ``render`` (VerifyCmd ->
raw string), and a set of pure VerifyCmd -> VerifyCmd mutators.

No source stub exists yet — every test in this module is RED until
orchestrator/src/orchestrator/verify_cmd.py is created (step-2).
"""

from __future__ import annotations

from orchestrator.verify_cmd import ToolKind, VerifyCmd
