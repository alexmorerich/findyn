"""The supply schedule, from the halving calendar.

This is the one part of FinCrypto that is not an estimate. Bitcoin's issuance is
fixed by consensus rules: 50 BTC per block at genesis, halved every 210,000
blocks, and no participant can change it without changing the asset. The four
halvings that have happened are historical facts with block heights and dates;
everything after them is arithmetic on those facts.

Hard-coded dates are correct here, and only here
------------------------------------------------

Everywhere else in this system, a number in Python instead of ``series.yaml`` is
a bug — a rule nobody can recalibrate without a deploy. The halving dates are
the exception the rule is written against: they are not measurements of a market
that could be revised, re-based or restated by a publisher. They are the dates on
which a deterministic counter crossed a multiple of 210,000, and they are as
fixed as the block heights themselves. Putting them in config would invite
someone to "correct" them.

The projected halvings are a different kind of number and are labelled as one.

What this module does NOT do
----------------------------

It does not build a stock-to-flow price model. ``stock_to_flow`` is published
because it is a supply statistic with a definition — issued stock divided by
annual issuance — and because a page about scarcity that omits the number
everyone arrives looking for is being coy rather than careful. It is published
as a **supply statistic and nothing else**: the engine's ``predict`` never reads
it, no regime branches on it, and it appears in no forecast. The S2F *price*
model that made the ratio famous regressed price on it in levels across a sample
of two halvings, projected the fit forward, and was falsified in 2022 by roughly
an order of magnitude. Reproducing it here would contradict §0's first non-goal
in the most direct way available.

Precision
---------

Supply within a completed epoch is exact: 210,000 blocks at a known subsidy.
Within the *current* epoch it is interpolated linearly in time between the last
halving and the projected next one, because block height is not one of this
engine's inputs. Historically blocks have arrived a few percent faster than the
nominal ten minutes, so the projection runs late and the interpolation
understates supply — by at most ~0.2% of total issued supply at the end of an
epoch, which is smaller than the difference between issued supply and any
plausible estimate of *spendable* supply.

Issued, not circulating. Coins in blocks whose miner claimed less than the full
subsidy, and coins whose keys are lost, are counted here and are not spendable.
Nobody knows the second number; publishing an estimate of it would be publishing
a guess in a module whose whole point is that it contains none.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

#: Blocks between halvings, from the consensus rules.
HALVING_INTERVAL_BLOCKS = 210_000

#: Initial block subsidy, BTC.
INITIAL_SUBSIDY = 50.0

#: Target seconds per block. The protocol's difficulty adjustment aims here.
TARGET_BLOCK_SECONDS = 600

#: Nominal days between halvings at the target block time. Used only to project
#: *future* halvings; every past one below is an observed date.
NOMINAL_EPOCH_DAYS = HALVING_INTERVAL_BLOCKS * TARGET_BLOCK_SECONDS / 86_400.0

#: Blocks per year at the target block time — the denominator of the issuance rate.
BLOCKS_PER_YEAR = 365.25 * 86_400.0 / TARGET_BLOCK_SECONDS

#: The genesis block. Issuance starts here.
GENESIS = date(2009, 1, 3)

#: Observed halving dates, by the block height each one occurred at. Facts, not
#: parameters — see the module docstring on why these are not in config.
HALVINGS: tuple[tuple[int, date], ...] = (
    (210_000, date(2012, 11, 28)),
    (420_000, date(2016, 7, 9)),
    (630_000, date(2020, 5, 11)),
    (840_000, date(2024, 4, 20)),
)


@dataclass(frozen=True)
class Epoch:
    """One subsidy era: when it began, what it pays, and how it ends."""

    index: int
    start: date
    #: ``None`` for the epoch that has not ended yet as of the caller's cutoff.
    end: date | None
    subsidy: float
    #: True when ``end`` is a projection rather than an observed halving.
    end_is_projected: bool
    #: Issued supply at ``start``.
    supply_at_start: float


def subsidy_for_epoch(index: int) -> float:
    """Block subsidy in epoch ``index`` (0 = genesis era)."""
    if index < 0:
        raise ValueError(f"epoch index must be >= 0, got {index}")
    return INITIAL_SUBSIDY / (2.0**index)


def project_halving(previous: date, epochs_ahead: int = 1) -> date:
    """Projected date of a future halving, at the nominal block time.

    Late by construction: every observed epoch has run 4-10% short of nominal
    because hash rate grows within an epoch and the difficulty adjustment only
    catches up after the fact. The bias is documented rather than fitted out —
    correcting it would mean estimating future hash-rate growth, which is not a
    consensus constant and does not belong in this module.
    """
    return previous + timedelta(days=NOMINAL_EPOCH_DAYS * epochs_ahead)


def epochs(through: date) -> list[Epoch]:
    """Every subsidy epoch from genesis through ``through``.

    The final entry is the epoch containing ``through``; its ``end`` is a
    projection when the next halving has not happened yet.
    """
    boundaries = [day for _, day in HALVINGS]
    result: list[Epoch] = []
    supply = 0.0
    start = GENESIS

    for index in range(len(boundaries) + 1):
        subsidy = subsidy_for_epoch(index)
        if index < len(boundaries):
            end: date | None = boundaries[index]
            projected = False
        else:
            end = project_halving(start)
            projected = True

        result.append(
            Epoch(
                index=index,
                start=start,
                end=end,
                subsidy=subsidy,
                end_is_projected=projected,
                supply_at_start=supply,
            )
        )
        if end is not None and end > through:
            break
        # A completed epoch issues exactly 210,000 blocks at its subsidy.
        supply += HALVING_INTERVAL_BLOCKS * subsidy
        start = end if end is not None else start

    return result


def current_epoch(day: date) -> Epoch:
    """The subsidy epoch ``day`` falls in."""
    if day < GENESIS:
        raise ValueError(f"{day} precedes the genesis block ({GENESIS})")
    return epochs(day)[-1]


def issued_supply(day: date) -> float:
    """Issued BTC as of ``day``. Exact at a halving, interpolated between them."""
    epoch = current_epoch(day)
    if day <= epoch.start:
        return epoch.supply_at_start
    end = epoch.end if epoch.end is not None else project_halving(epoch.start)
    span = (end - epoch.start).days
    if span <= 0:
        return epoch.supply_at_start
    fraction = min(max((day - epoch.start).days / span, 0.0), 1.0)
    return epoch.supply_at_start + fraction * HALVING_INTERVAL_BLOCKS * epoch.subsidy


def annual_issuance(day: date) -> float:
    """BTC issued per year at ``day``'s subsidy."""
    return current_epoch(day).subsidy * BLOCKS_PER_YEAR


def issuance_rate(day: date) -> float:
    """Annual issuance as a percentage of issued supply — the inflation rate."""
    supply = issued_supply(day)
    if supply <= 0.0:
        return float("nan")
    return 100.0 * annual_issuance(day) / supply


def stock_to_flow(day: date) -> float:
    """Issued stock divided by annual issuance. **A supply statistic only.**

    See the module docstring: this number is published for the page and is read
    by nothing in ``predict``. It is not a valuation, not an input to one, and
    the model that made it famous has been falsified.
    """
    flow = annual_issuance(day)
    if flow <= 0.0:
        return float("nan")
    return issued_supply(day) / flow


def days_to_next_halving(day: date) -> tuple[float, bool]:
    """``(days, is_projected)`` until the halving that ends ``day``'s epoch."""
    epoch = current_epoch(day)
    end = epoch.end if epoch.end is not None else project_halving(epoch.start)
    return float((end - day).days), epoch.end_is_projected


def schedule(index: pd.DatetimeIndex) -> pd.DataFrame:
    """The supply schedule on a date index — one row per date, four columns.

    A pure function of the calendar. No market data enters this frame, which is
    why it needs no point-in-time treatment: a run on any cutoff computes the
    same supply for the same past date, because the halvings that had happened by
    that date had happened.
    """
    if len(index) == 0:
        return pd.DataFrame(
            index=pd.DatetimeIndex([], name=index.name),
            columns=["issued_supply", "issuance_rate", "stock_to_flow", "subsidy"],
            dtype=float,
        )

    days = [ts.date() for ts in index]
    frame = pd.DataFrame(index=index)
    frame["issued_supply"] = [issued_supply(day) if day >= GENESIS else np.nan for day in days]
    frame["issuance_rate"] = [issuance_rate(day) if day >= GENESIS else np.nan for day in days]
    frame["stock_to_flow"] = [stock_to_flow(day) if day >= GENESIS else np.nan for day in days]
    frame["subsidy"] = [current_epoch(day).subsidy if day >= GENESIS else np.nan for day in days]
    return frame


__all__ = [
    "BLOCKS_PER_YEAR",
    "GENESIS",
    "HALVINGS",
    "HALVING_INTERVAL_BLOCKS",
    "INITIAL_SUBSIDY",
    "NOMINAL_EPOCH_DAYS",
    "TARGET_BLOCK_SECONDS",
    "Epoch",
    "annual_issuance",
    "current_epoch",
    "days_to_next_halving",
    "epochs",
    "issuance_rate",
    "issued_supply",
    "project_halving",
    "schedule",
    "stock_to_flow",
    "subsidy_for_epoch",
]
