"""Factor definitions — the initial factor set and its config-backed specs.

The nine v1 "forces" of FINDYN_V1_SPEC.md §2.2 become the initial factor set
(``02-migration-map.md`` §1). They are defined once, in
:mod:`findynamics.core.contracts.vocab`, because ``core.config`` validates
against them and ``core`` may not import this layer; this module is the import
path the rest of the stack uses and the place P1 extends with ``real_rate``,
``usd_strength`` and the curve inputs.

Scoring (winsorized z → expanding-window percentile → 0-100) is Phase 1 work and
lands in ``factors/compute.py``; P0 deliberately ships only the definitions.
"""

from __future__ import annotations

from findynamics.core.config import FactorSpec, SeriesConfig, get_series_config
from findynamics.core.contracts.vocab import FACTORS

__all__ = ["FACTORS", "factor_series_ids", "factor_specs"]


def factor_specs(config: SeriesConfig | None = None) -> dict[str, FactorSpec]:
    """The configured factor specs, keyed by factor name.

    ``core.config`` has already checked that the set is exactly :data:`FACTORS`
    and that every factor has at least one series, so this is a plain accessor —
    the validation lives at load time where a bad edit fails fastest.
    """
    return dict((config or get_series_config()).factors)


def factor_series_ids(config: SeriesConfig | None = None) -> tuple[str, ...]:
    """Every series id the factor layer needs, de-duplicated and sorted."""
    specs = factor_specs(config)
    return tuple(sorted({s.id for spec in specs.values() for s in spec.series}))
