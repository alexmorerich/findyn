"""FinCrypto vocabulary — names this engine owns.

Mirrored in ``serving/src/domain.ts`` as ``CRYPTO_REGIMES``;
``tests/test_domain.py`` fails on drift.

The three states describe **how much speculation is in the price**, and that is
the only question this engine claims to answer. There is no cash flow to
discount, no issuer to assess and — unlike gold — not even a four-thousand-year
record of being treated as money. What is left that can be measured honestly is
the character of the price process itself and its relationship to the money
supply, so the regimes name three states of that process rather than three
theories about what bitcoin is worth.

Evaluated by :mod:`findynamics.engines.crypto.regime`:

``winter``
    Deep in a drawdown from the trailing-year peak. 2014-15, 2018, mid-2022.
    A statement about depth, not duration: March 2020 spent a fortnight here on
    a 50% crash and left again, and that is the correct reading of a fortnight
    in which the asset had halved.
``normal``
    Neither. The honest label for "nothing in particular is happening", which is
    more of the record than either of the other two.
``frenzy``
    Price far above where it was a year ago *and* realized volatility elevated.
    Late 2017 and 2021. Both conditions are required because either alone is
    ordinary for this asset: bitcoin has had 100% years without a blowoff and
    volatile years without a trend, and it is the combination that has
    historically preceded the drawdowns.

Order is the wire order: ``engine_output`` publishes the state as its index here
(``regime_code``), because that table stores REAL values. Listed in increasing
order of speculation, so a chart of ``regime_code`` reads upwards as more of the
price is momentum.
"""

from __future__ import annotations

from typing import Final

#: The three states, in wire order. Reordering this silently relabels every
#: ``regime_code`` row the engine has ever written.
CRYPTO_REGIMES: Final[tuple[str, ...]] = (
    "winter",
    "normal",
    "frenzy",
)

#: The wide metrics FinCrypto publishes per date (``engine_output.metric``).
CRYPTO_METRICS: Final[tuple[str, ...]] = (
    "price",
    "realized_vol",
    "drawdown",
    "return_12m",
    "speculation_index",
    "liquidity_beta",
    "liquidity_beta_r2",
    "liquidity_residual",
    "jump_intensity",
    "issued_supply",
    "issuance_rate",
    "stock_to_flow",
    "tx_volume_trend",
    # 1.0 on dates the daily-average history role supplied, 0.0 on dates that
    # came from a close. Published as a series rather than summarised, because a
    # reader looking at 2012 is looking at a different statistic from a reader
    # looking at 2022 and the chart is where that has to be visible.
    "price_is_daily_average",
    # The on-chain measurements as published. Only `tx_volume_usd` feeds the
    # model; the other three are charted so a feed that stops arriving is
    # visible rather than silently absent.
    "tx_volume_usd",
    "active_addresses",
    "transactions",
    "hash_rate",
    "regime_code",
)


def regime_code(regime: str) -> int:
    """Index of ``regime`` in :data:`CRYPTO_REGIMES`."""
    try:
        return CRYPTO_REGIMES.index(regime)
    except ValueError:
        raise ValueError(
            f"unknown crypto regime {regime!r}; expected one of {CRYPTO_REGIMES}"
        ) from None


__all__ = ["CRYPTO_METRICS", "CRYPTO_REGIMES", "regime_code"]
