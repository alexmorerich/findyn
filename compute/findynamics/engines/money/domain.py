"""FinMoney vocabulary — names this engine owns.

Mirrored in ``serving/src/domain.ts`` as ``MONEY_REGIMES``;
``tests/test_domain.py`` fails on drift.

The four states describe the **condition of the funding market**, not the level
of rates. A 5% short rate is not "tight" here and a 0% one is not "abundant" —
that is FinRates' subject. What this engine reads is whether cash is changing
hands normally, which is a different question with a different answer: March 2020
had a falling policy rate and a dislocated funding market at the same time.

Evaluated in the order listed in :mod:`findynamics.engines.money.liquidity`:

``stressed``
    The money market is not clearing. Either the overnight rate has spiked clear
    of the 3m bill (collateral and reserve scarcity — September 2019), or bills
    are being scrambled for far below the overnight rate and moving fast (a dash
    for cash — March 2020). Both show up as the bill-SOFR spread going sharply
    negative, from opposite directions.
``tightening``
    The buffer is draining: reverse-repo balances are falling well off their own
    trailing level, or bills are persistently through the overnight rate. The
    ordinary reading during quantitative tightening and ahead of easing cycles.
``abundant``
    Reverse-repo take-up is large and not shrinking — more cash in the system
    than it has uses for, parked at the Fed. 2021-2023, and the weeks after the
    March 2020 intervention.
``normal``
    None of the above. The honest label when the funding market is unremarkable,
    which is most of the time.
"""

from __future__ import annotations

from typing import Final

#: Order is the wire order: ``engine_output`` publishes the state as its index
#: here (``liquidity_code``), because that table stores REAL values.
#:
#: Listed least-to-most constrained so the code is monotone in tightness — a
#: chart of ``liquidity_code`` reads upwards as conditions worsen.
MONEY_REGIMES: Final[tuple[str, ...]] = (
    "abundant",
    "normal",
    "tightening",
    "stressed",
)

#: The wide metrics FinMoney publishes per date (``engine_output.metric``).
MONEY_METRICS: Final[tuple[str, ...]] = (
    "wealth_index",
    "short_rate",
    "carry_1m",
    "carry_3m",
    "carry_12m",
    "discount_1y",
    "discount_3y",
    "discount_10y",
    "bill_sofr_spread",
    "liquidity_code",
)

#: Trailing carry windows, in calendar days, and the metric each is published as.
#: Calendar rather than trading days because the accrual is calendar-based: cash
#: earns over a weekend, it just earns at Friday's rate.
CARRY_WINDOWS: Final[dict[str, int]] = {
    "carry_1m": 30,
    "carry_3m": 91,
    "carry_12m": 365,
}


def liquidity_code(state: str) -> int:
    """Index of ``state`` in :data:`MONEY_REGIMES`."""
    try:
        return MONEY_REGIMES.index(state)
    except ValueError:
        raise ValueError(
            f"unknown liquidity state {state!r}; expected one of {MONEY_REGIMES}"
        ) from None
