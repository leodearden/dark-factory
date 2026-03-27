"""Stage and judge system prompts."""

# Shared template for the project_id usage guideline embedded in each stage's system prompt.
# Each stage calls .format(tools=<stage-specific tool list>) to produce the final string.
_PROJECT_ID_GUIDELINE = (
    "- Always include the `project_id` from the Reconciliation Context block"
    " in every fused-memory MCP call ({tools})."
)
