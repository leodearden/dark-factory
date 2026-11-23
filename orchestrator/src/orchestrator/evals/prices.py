"""The per-config price table a composite report echoes: built from config
prices (input, output and any set cache rates), with an explicit 'unpriced'
marker for unlisted models, and rendered as a fixed-width block.
"""

from __future__ import annotations

from typing import Any

from orchestrator.agents.token_pricing import optional_rate, rate

from .table_render import render_fixed_table

_CACHE_RATE_KEYS = ('cached_input_per_1m', 'cache_write_per_1m')


def _price_role_entry(model: str, prices: dict[str, Any] | None) -> dict[str, Any]:
    """One role's price cell for *model*: its ``input/output_per_1m`` rates plus
    each cache rate the entry sets, or the EXPLICIT ``{'source': 'unpriced'}``
    marker for an unlisted model — never a fabricated default (loud-over-silent).
    Shared by :func:`build_price_table` (individual-config keys) and
    :func:`build_pairwise_price_table` (the μ end-to-end combined ``arch+impl``
    keys) so both build cells identically.
    """
    entry = prices.get(model) if prices else None
    if entry is None:
        return {'source': 'unpriced'}
    cache_rates = {key: optional_rate(entry, key) for key in _CACHE_RATE_KEYS}
    return {
        'input_per_1m': rate(entry, 'input_per_1m'),
        'output_per_1m': rate(entry, 'output_per_1m'),
        **{key: value for key, value in cache_rates.items() if value is not None},
    }


def build_price_table(
    configs: list[Any], prices: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Build the C4 per-config price table ``{config_name: {role: entry}}``.

    Each config's ``model`` is looked up in *prices* (a ``config.prices``-shaped
    map; ``PriceEntry`` objects and plain ``{'input_per_1m','output_per_1m'}``
    dicts are both accepted via ``agents.token_pricing.rate``). A listed model
    yields ``{role: {'input_per_1m', 'output_per_1m'}}`` plus whichever of
    ``cached_input_per_1m`` / ``cache_write_per_1m`` it sets; an UNLISTED model
    yields the explicit ``{role: {'source': 'unpriced'}}`` marker — never a
    fabricated default (loud-over-silent). Pure over ``(configs, prices)``; the caller
    (μ / CLI) seeds *prices* from ``default_price_table()`` / ``config.prices``.
    Config keys are inserted in sorted order so the table is byte-deterministic.
    """
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for config in sorted(configs, key=lambda c: c.name):
        table[config.name] = {config.role: _price_role_entry(config.model, prices)}
    return table


def build_pairwise_price_table(
    pairs: list[Any], prices: dict[str, Any] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Combined-name price table for the μ end-to-end stages (matrix / confirm).

    Each ``(architect_cfg, implementer_cfg)`` pair keys ONE entry under the
    combined ``f'{arch.name}+{impl.name}'`` config name — byte-identical to the
    ``config_name`` :func:`run_end_to_end` stamps on that pair's ``EvalResult`` —
    carrying BOTH roles' per-model rates. Keying by the combined name keeps the
    rendered price-table section's ``config`` column ALIGNED with an end-to-end
    composite report's combined rows; individual-config :func:`build_price_table`
    keys never lined up with those rows (reviewer: correctness). Pure over
    ``(pairs, prices)``; keys inserted in sorted order for byte-determinism.
    """
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for arch_cfg, impl_cfg in sorted(pairs, key=lambda p: f'{p[0].name}+{p[1].name}'):
        table[f'{arch_cfg.name}+{impl_cfg.name}'] = {
            arch_cfg.role: _price_role_entry(arch_cfg.model, prices),
            impl_cfg.role: _price_role_entry(impl_cfg.model, prices),
        }
    return table


_RATE_COLUMNS = ('input_per_1m', 'output_per_1m', *_CACHE_RATE_KEYS)
_PRICE_TABLE_COLUMNS = ('config', 'role', *_RATE_COLUMNS)


def _rate_cells(entry: dict[str, Any]) -> dict[str, str]:
    """Each rate column's cell: the rate at 4 dp, 'none' for a listed entry's
    absent cache rate, or an unpriced entry's source marker in every cell."""
    if 'input_per_1m' not in entry:
        marker = str(entry.get('source', 'unpriced'))
        return dict.fromkeys(_RATE_COLUMNS, marker)
    return {
        column: f'{float(entry[column]):.4f}' if column in entry else 'none'
        for column in _RATE_COLUMNS
    }


def format_price_table_section(price_table: dict[str, Any]) -> list[str]:
    """Render the ``{config: {role: entry}}`` price table as a deterministic
    block. A listed entry shows ``input_per_1m`` / ``output_per_1m`` /
    ``cached_input_per_1m`` / ``cache_write_per_1m`` at 4 dp, with 'none' for
    an absent cache rate so the report reveals it; an UNPRICED entry
    (``{'source': 'unpriced'}``) shows the explicit marker in every rate cell —
    never a blank or fabricated price. Configs and roles are sorted.
    """
    rendered: list[dict[str, str]] = []
    for cfg in sorted(price_table):
        roles = price_table[cfg] or {}
        for role in sorted(roles):
            rendered.append({
                'config': str(cfg),
                'role': str(role),
                **_rate_cells(roles[role] or {}),
            })
    return render_fixed_table(
        _PRICE_TABLE_COLUMNS, rendered, header='price table:',
    )
