"""FinGold vocabulary — names this engine owns.

Mirrored in ``serving/src/domain.ts`` as ``GOLD_REGIMES``; ``tests/test_domain.py``
fails on drift.

The three states describe **why gold is being bid, or why it is not**. Gold has no
cash flow, so there is nothing to value and no earnings to be right or wrong
about: what changes is the demand for a monetary asset with no counterparty. The
regimes name the three forces that move that demand, and they are not degrees of
one thing — a rate headwind and a crisis bid can and do arrive in the same month
(1981-82 had both), which is why the state is published as a posterior over all
three rather than as a winner.

Evaluated by :mod:`findynamics.engines.gold.regime`:

``hedge_bid``
    The ordinary state, and the most common. Gold is held as portfolio
    insurance: no crisis is being priced, real rates are not moving against it,
    and it trades on the dollar and on flows. The honest label for "nothing in
    particular is happening to gold", which is most of the time.
``carry_headwind``
    Real rates are rising. This is the one genuine cost of holding gold — a
    non-yielding asset competes with a real yield, and when that yield rises the
    competition is arithmetic rather than sentimental. 1981-82 is the archetype
    (real 10y from -5% to +8%, gold down 60%); 2013's taper tantrum and 2022's
    tightening cycle are the modern ones.
``crisis_bid``
    Stress is being paid for. Financial conditions are tight or dislocating and
    gold is bid as the asset with no issuer. 1979-80, 2008H2 and March 2020.
    Note this is a statement about *demand*, not about direction: the first
    weeks of a crisis usually see gold sold for liquidity before it is bought
    for safety, and 2008 and 2020 both did exactly that.

Order is the wire order: ``engine_output`` publishes the state as its index here
(``regime_code``), because that table stores REAL values. Listed
least-to-most eventful, so a chart of ``regime_code`` reads upwards as the
monetary story gets louder.
"""

from __future__ import annotations

from typing import Final

#: The three states, in wire order. Reordering this silently relabels every
#: ``regime_code`` row the engine has ever written.
GOLD_REGIMES: Final[tuple[str, ...]] = (
    "hedge_bid",
    "carry_headwind",
    "crisis_bid",
)

#: The wide metrics FinGold publishes per date (``engine_output.metric``).
#:
#: The three ``regime_posterior_*`` rows are published individually rather than
#: as one composite, for the same reason the equity engine publishes its crash
#: decomposition as three bars: a 0.5/0.5 split between a hedge bid and a rate
#: headwind is a completely different statement from a 1.0 hedge bid, and a
#: single winning label cannot express the difference.
GOLD_METRICS: Final[tuple[str, ...]] = (
    "price",
    "hedge_score",
    "jump_intensity",
    "crisis_premium",
    "real_rate_10y",
    "real_rate_change_12m",
    "usd_trend",
    "stress_score",
    "regime_code",
    "regime_posterior_hedge_bid",
    "regime_posterior_carry_headwind",
    "regime_posterior_crisis_bid",
)

#: Prefix under which each regime's posterior is published to ``engine_output``.
POSTERIOR_PREFIX: Final[str] = "regime_posterior_"


def posterior_metric(regime: str) -> str:
    """``'crisis_bid'`` -> ``'regime_posterior_crisis_bid'``."""
    if regime not in GOLD_REGIMES:
        raise ValueError(f"unknown gold regime {regime!r}; expected one of {GOLD_REGIMES}")
    return f"{POSTERIOR_PREFIX}{regime}"


def regime_code(regime: str) -> int:
    """Index of ``regime`` in :data:`GOLD_REGIMES`."""
    try:
        return GOLD_REGIMES.index(regime)
    except ValueError:
        raise ValueError(
            f"unknown gold regime {regime!r}; expected one of {GOLD_REGIMES}"
        ) from None
