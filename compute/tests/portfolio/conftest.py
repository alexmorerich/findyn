"""Shared helpers for the portfolio suite: synthetic states and panels.

Nothing here touches the network or a real engine — the portfolio layer consumes
``AssetState`` objects, so a test builds them directly and never runs a model.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from findynamics.core.contracts.state import AssetState, FactorState, WorldState
from findynamics.data.accessor import PandasPITAccessor
from findynamics.portfolio.config import load_portfolio_config

_EMPTY_COLUMNS = [
    "series_id",
    "provider",
    "frequency",
    "unit",
    "obs_date",
    "release_date",
    "revision_date",
    "value",
]


def make_state(
    asset: str,
    as_of: date,
    *,
    expected_return: float | None,
    risk_score: float,
    confidence: float,
    regime: str,
    components: dict[str, float] | None = None,
) -> AssetState:
    return AssetState(
        asset=asset,
        as_of=as_of,
        regime=regime,
        expected_return=expected_return,
        risk_score=risk_score,
        confidence=confidence,
        signals=(),
        model_version=f"{asset}-1.0.0",
        components=components,
    )


def make_world(as_of: date, factors: dict[str, float] | None = None) -> WorldState:
    accessor = PandasPITAccessor(pd.DataFrame(columns=_EMPTY_COLUMNS), as_of)
    factor_states = {
        name: FactorState(name=name, as_of=as_of, score=score, components={})
        for name, score in (factors or {}).items()
    }
    return WorldState(as_of=as_of, factors=factor_states, series=accessor)


def full_panel_states(as_of: date) -> dict[str, AssetState]:
    """A complete, fresh set of states across the four non-experimental engines."""
    return {
        "money": make_state(
            "money",
            as_of,
            expected_return=0.045,
            risk_score=2.0,
            confidence=0.8,
            regime="normal",
            components={"discount_1y": 0.956, "discount_3y": 0.87, "discount_10y": 0.62},
        ),
        "rates": make_state(
            "rates", as_of, expected_return=0.052, risk_score=38.0, confidence=0.65,
            regime="re_steepening",
        ),
        "equity": make_state(
            "equity", as_of, expected_return=0.085, risk_score=55.0, confidence=0.6,
            regime="normal_expansion",
        ),
        "gold": make_state(
            "gold", as_of, expected_return=0.030, risk_score=40.0, confidence=0.5,
            regime="hedge_bid",
        ),
    }


def load_config():
    return load_portfolio_config()
