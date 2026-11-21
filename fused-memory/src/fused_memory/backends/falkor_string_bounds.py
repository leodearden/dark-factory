"""FalkorDB's RANGE index returns a superset when it serves ``<`` or ``<=`` on a string property; ``>`` and ``>=`` are exact.

Measurements and the per-site audit are in plans/falkordb-resultset-cap-audit.md,
section 'String < / <= on RANGE-indexed properties (task 7543)'.
"""


def exact_string_upper_bound(prop: str, param: str, *, inclusive: bool) -> str:
    """Return the Cypher predicate ``<prop> <= $<param>`` (``<`` when not inclusive) in a form no RANGE index can serve.

    Concatenating ``''`` turns the operand into an expression, so the planner
    evaluates the comparison in a Filter instead of handing it to the index.
    """
    if not param.isidentifier():
        raise ValueError(f'exact_string_upper_bound needs a bare Cypher parameter name, got {param!r}')
    operator = '<=' if inclusive else '<'
    return f"({prop} + '') {operator} ${param}"
