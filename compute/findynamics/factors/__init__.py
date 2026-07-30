"""Layer 0 — shared risk factors.

Global financial variables owned by no engine. Computed once per run from
``series.yaml``, PIT-correct, scored 0-100, and served to every engine through
``WorldState`` (``01-target-architecture.md`` §5). An engine never recomputes a
shared factor privately.
"""

from findynamics.factors.definitions import FACTORS, factor_specs

__all__ = ["FACTORS", "factor_specs"]
