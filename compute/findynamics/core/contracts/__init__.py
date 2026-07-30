"""Data contracts crossing layer boundaries.

Everything here is frozen, plain-Python and pandas-free: a value that travels
between the data layer, an engine and the portfolio layer must be comparable and
serialisable without dragging a DataFrame along (``03-contracts.md`` §1).
"""

from findynamics.core.contracts.pit import PITAccessor
from findynamics.core.contracts.state import (
    AssetState,
    EngineOutput,
    FactorState,
    Signal,
    WorldState,
)
from findynamics.core.contracts.vocab import (
    ASSETS,
    EDUCATIONAL_HORIZONS,
    FACTORS,
    FREQUENCIES,
    HORIZONS,
    QUANTILES,
)

__all__ = [
    "ASSETS",
    "EDUCATIONAL_HORIZONS",
    "FACTORS",
    "FREQUENCIES",
    "HORIZONS",
    "QUANTILES",
    "AssetState",
    "EngineOutput",
    "FactorState",
    "PITAccessor",
    "Signal",
    "WorldState",
]
