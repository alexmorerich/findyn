"""The canonical bitcoin price record, stitched from three configured roles.

Bitcoin has no single authoritative price. It trades continuously on dozens of
venues with real spreads between them, so every "the" bitcoin price is some
vendor's composite, and the composites differ. This module turns the three
configured roles into one record and — more importantly — keeps saying which
source produced which date, because the answer changes what some of the numbers
downstream mean.

The three roles, in precedence order
------------------------------------

``price``
    ``STOOQ:BTCUSD``, the configured primary. A daily close. Unreachable from
    every automated egress this project has (the proof-of-work interstitial that
    also blocks ``^SPX``), so in practice it supplies nothing — but it stays the
    declared primary because that is what it is, and the day the egress changes
    it takes over without a config edit.
``price_fallback``
    ``YAHOO:BTC-USD``, a daily close from a composite venue. Reachable, keyless,
    and starts **2014-09-17** — which is the whole problem this module exists to
    solve, because it cuts 2011 and 2013 out of a sample that only has six
    cycles in it.
``price_history``
    ``BLOCKCHAIN:MARKET_PRICE``, a volume-weighted daily **average** across
    exchanges, from **2010-08-18**. Extends the record backwards by four years
    and two full cycles.

Closes take precedence over averages wherever both exist, and the history role
only ever supplies dates *before* the closes begin. A spliced record can
therefore lengthen the history and can never restate a figure already published
from a close.

Why this splice is validated differently from the equity one
------------------------------------------------------------

``engines/equity/prices.py`` splices two vendors' copies of *one index* and
tests that they agree to a tenth of a percent — over their shared decade
``FRED:SP500`` and ``YAHOO:^GSPC`` differ by a median 2e-8, so a 1e-3 limit sits
four orders of magnitude above the noise.

That test applied here would refuse this splice, and for the wrong reason. These
two series are not two copies of one statistic: one is a close, the other is a
daily average. Measured over their 4,341 shared dates they differ by a median
1.5%, 63% of days differ by more than 1%, and 2020-03-12 differs by 60% — a
$4,971 close against a $7,937 daily average, on a day bitcoin fell by a third
intraday. Every one of the worst days is a large-intraday-range day. The
disagreement is the definition, not an error.

So the validity test is built from the three properties that actually decide
whether joining them is honest, each with a measured value behind its limit:

1. **No step at the seam.** The splice must not manufacture a price move.
   Measured: the implied return across 2014-09-16 → 2014-09-17 is -2.9%, against
   a median absolute daily move of 1.4% and a 99th percentile of 11.9%. An
   ordinary day.
2. **No systematic level bias.** One being persistently above the other would
   mean a rebasing rather than a spread. Measured: mean signed gap -0.18%.
3. **Comparable volatility.** This engine's regime, risk score and jump detector
   are all functions of the return process, so an extension whose returns are
   quieter would publish a calmer 2011 than happened. Measured: annualized
   volatility 67.9% from the averages against 66.8% from the closes, a ratio of
   1.016.

What is still not equivalent, and is published rather than hidden
-----------------------------------------------------------------

A daily average and a daily close are different statistics on any *single* day
even though their return processes match in aggregate. The dates the extension
supplied are therefore flagged per date, exactly as ``engines/gold/drivers.py``
flags the dates its real rate came from trailing CPI rather than a breakeven —
"a different quantity, so the splice is recorded per date rather than blended
silently". The engine publishes the flag as a series, carries the average-based
share on the state, and takes a confidence penalty for it.

The jump detector is the consumer most affected: it asks whether one day stood
out against its neighbours, and an average suppresses exactly the intraday range
that makes a day stand out. It self-scales — the threshold is set by each date's
own trailing bipower volatility, so a quieter window raises fewer flags rather
than the wrong ones — but a jump date before 2014-09-17 and one after are not
strictly the same measurement, and nothing here pretends otherwise.

The design note behind all of this — why an average is accepted, why closes take
precedence, how the confidence penalty propagates, and the rule for changing a
threshold — is ``docs/design/crypto-price-record.md``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger("findynamics.engines.crypto.prices")

#: Roles that carry a daily **close**, in precedence order. The configured
#: primary first; the reachable fallback behind it.
CLOSE_ROLES: tuple[str, ...] = ("price", "price_fallback")

#: The role that extends the record backwards with a daily **average**.
HISTORY_ROLE = "price_history"

# ---------------------------------------------------------------------------
# The four splice thresholds.
#
# READ THIS BEFORE CHANGING ANY OF THEM.
#
# These are **empirical guardrails, not tuned parameters**, and the difference is
# not a matter of taste — it decides what you are allowed to do when one fires.
#
# A tuned parameter is chosen by looking at the data you have and picking the
# value that gives the answer you want. A guardrail is chosen by asking "what
# would a *broken* splice look like?" and drawing the line there, before looking
# at whether the shipped pair clears it. Every limit below was fixed by the
# second question. The measured values are recorded beside each one as
# *evidence that the guardrail is not binding*, never as the reason for its
# value — which is why every one of them sits an order of magnitude away from
# what the shipped pair actually does, rather than comfortably just past it.
#
# The practical consequence, and the reason it is worth this many words: if one
# of these ever fires on a new vendor or a regenerated snapshot, **the correct
# response is to investigate the data, not to widen the limit.** A threshold
# moved to make a red check go green has stopped being a check. If after
# investigating you conclude the limit itself was wrong, change it in a commit
# that does nothing else, says which failure mode the new value still catches,
# and updates the measured figures in docs/design/crypto-price-record.md.
#
# `tests/engines/crypto/test_prices.py` builds a synthetic pair per branch that
# trips exactly one check, because the shipped pair passes all four and a suite
# that only ran against it would be testing that a splice happens — never that a
# bad one is refused.
# ---------------------------------------------------------------------------

#: Shared dates the two records need before their agreement can be judged.
#:
#: Guards: a comparison so short it is anecdote. Three of the four checks below
#: are averages over the overlap, and an average over a fortnight of a 60%-vol
#: asset says nothing at all.
#:
#: Chosen as roughly a year of observations — the same figure the equity engine
#: uses, for the same reason, and long enough to span a full seasonal cycle
#: rather than one market mood. Measured on the shipped pair: **4,340 shared
#: dates**, 17x the floor.
MIN_SPLICE_OVERLAP = 250

#: Largest absolute log return tolerated across the join date.
#:
#: Guards: a **manufactured price move**. The seam is the one return in the
#: record that no market produced — it is an artefact of two series being placed
#: end to end — so if the two legs are on different scales (a rebasing, a
#: different currency, a units error) this is where it shows up, as a single
#: enormous day sitting in the return series that feeds the jump detector and
#: the regime.
#:
#: Chosen from what it must *not* reject: bitcoin has had genuine single sessions
#: past 25% (2013-04-10 fell about half), so a tighter limit would refuse a sound
#: splice whenever the join happened to land on a real crash. 25% is therefore
#: deliberately permissive — it cannot catch a small scale error, and is not
#: meant to; a scale error large enough to matter is an order of magnitude
#: bigger. Measured on the shipped pair: **-2.9%**, against a 1.4% median daily
#: move. 8.6x of headroom.
#:
#: Known failure direction: if a join *does* land on a genuine >25% day the
#: splice is refused and the record is shortened. That is the conservative
#: outcome — history is lost, never corrupted — and it is reported rather than
#: silent (`declined_reason`).
MAX_SEAM_RETURN = 0.25

#: Largest mean signed relative gap tolerated over the overlap.
#:
#: Guards: a **persistent offset**, which means the two legs are not the same
#: quantity. Venue spreads are noise that averages out; a series quoted in
#: another currency, on another asset, or against a different base does not
#: average out, and would splice into a record whose early years are simply the
#: wrong number.
#:
#: Chosen from the gap between the two: bitcoin cross-venue spreads have run to a
#: few percent in stressed windows (and far more in the Mt. Gox era), while any
#: genuine mismatch of quantity is tens of percent or more. 5% sits in the empty
#: space between those two populations. Signed rather than absolute on purpose —
#: symmetric day-to-day disagreement is exactly what a close-versus-average pair
#: should show, and cancels; a real offset does not.
#: Measured on the shipped pair: **-0.18%**, 28x of headroom.
MAX_LEVEL_BIAS = 0.05

#: Bounds on (extension volatility / close volatility) over the overlap.
#:
#: Guards: an extension whose **return process is not the same market**. This is
#: the check that matters most for this particular engine, because essentially
#: everything it publishes — the regime's vol leg, the risk score, the jump
#: intensity, the speculation index's first term — is a function of the return
#: process rather than of the level. A smoothed, weekly, or interpolated
#: extension would look perfectly reasonable on a price chart and would publish a
#: calm 2011 that did not happen.
#:
#: Chosen from what a smoothed series does: a 30-day rolling mean of the closes
#: lands near 0.2 and a weekly average near 0.4, so a floor of 0.75 catches
#: anything meaningfully smoothed while leaving room for the genuine
#: average-versus-close difference. Roughly symmetric in ratio terms (1/0.75 =
#: 1.33 against the 1.35 ceiling), because an extension that is *more* volatile
#: than the closes is equally not the same series.
#: Measured on the shipped pair: **1.016**, near the centre of the band.
VOLATILITY_RATIO_BOUNDS = (0.75, 1.35)

#: Bitcoin has no exchange calendar.
CALENDAR_DAYS = 365


# Deliberately not named *Error (hence the noqa), matching
# core.engine.StateUnavailable: an information set with no price in it is not a
# crash, it is the correct answer for a cutoff before bitcoin had a market. The
# engine catches this and declines to publish rather than failing the run.
class PriceUnavailable(RuntimeError):  # noqa: N818
    """No configured role produced a usable price series."""


@dataclass(frozen=True)
class PriceRecord:
    """The stitched daily price, and an honest account of where it came from."""

    #: Daily price, oldest first, strictly positive.
    price: pd.Series
    #: ``True`` on dates supplied by the daily-average history role.
    is_daily_average: pd.Series
    #: Role that supplied the close leg (``price`` or ``price_fallback``).
    close_role: str
    #: Series id behind the close leg.
    close_series_id: str
    #: Role and id behind the extension, or ``None`` when nothing was spliced.
    history_role: str | None = None
    history_series_id: str | None = None
    #: Why the extension was declined, when it was. ``None`` on success and when
    #: no extension was configured at all.
    declined_reason: str | None = None

    @property
    def empty(self) -> bool:
        return self.price.empty

    @property
    def spliced(self) -> bool:
        return self.history_series_id is not None and self.declined_reason is None

    @property
    def average_share(self) -> float:
        """Fraction of the record that is average-based rather than close-based."""
        if self.price.empty:
            return 0.0
        return float(self.is_daily_average.mean())

    @property
    def from_fallback(self) -> bool:
        """``True`` when the configured primary was absent and the fallback ran."""
        return self.close_role != CLOSE_ROLES[0]

    @property
    def series_id(self) -> str:
        """Composite identity naming every vendor that contributed.

        Composite rather than "still YAHOO:BTC-USD", for the reason the equity
        engine gives: a 2011 price did not come from a series that starts in
        2014, and a reader tracing a number back to its source is entitled to
        both names.
        """
        if not self.spliced:
            return self.close_series_id
        return f"{self.close_series_id}+{self.history_series_id}"


def _role_series(frame: pd.DataFrame, series_id: str | None) -> pd.Series:
    """Positive closes for one configured id, or an empty series."""
    if not series_id or series_id not in frame:
        return pd.Series(dtype=float)
    values = frame[series_id].dropna()
    return values[values > 0]


def _annualized_volatility(price: pd.Series) -> float:
    """Annualized standard deviation of daily log returns, in percent."""
    returns = np.log(price).diff().dropna()
    if len(returns) < 2:
        return float("nan")
    return float(returns.std() * math.sqrt(CALENDAR_DAYS) * 100.0)


@dataclass(frozen=True)
class SpliceCheck:
    """The three measurements the splice decision is made from."""

    overlap: int
    seam_return: float
    level_bias: float
    volatility_ratio: float

    def refusal(self) -> str | None:
        """Why the splice must be declined, or ``None`` when it may proceed."""
        if self.overlap < MIN_SPLICE_OVERLAP:
            return (
                f"the two records share only {self.overlap} date(s), below the "
                f"{MIN_SPLICE_OVERLAP} needed to judge whether they agree"
            )
        if not math.isfinite(self.seam_return) or abs(self.seam_return) > MAX_SEAM_RETURN:
            return (
                f"joining them implies a {self.seam_return:+.1%} move across the seam, "
                f"past the {MAX_SEAM_RETURN:.0%} limit — that is a rebasing, not a market day"
            )
        if not math.isfinite(self.level_bias) or abs(self.level_bias) > MAX_LEVEL_BIAS:
            return (
                f"the extension sits {self.level_bias:+.1%} from the closes on average, "
                f"past the {MAX_LEVEL_BIAS:.0%} limit — a persistent offset is a different "
                "series, not a spread between venues"
            )
        low, high = VOLATILITY_RATIO_BOUNDS
        if not math.isfinite(self.volatility_ratio) or not low <= self.volatility_ratio <= high:
            return (
                f"the extension's volatility is {self.volatility_ratio:.2f}x the closes', "
                f"outside [{low}, {high}] — splicing it would publish a differently-behaved "
                "market under the same name, and the regime and risk score would inherit it"
            )
        return None


def measure_splice(closes: pd.Series, extension: pd.Series) -> SpliceCheck:
    """The three validity measurements, over the records' shared dates.

    Separate from the decision so a test — and a person — can read the numbers
    without reading the branch that acts on them.
    """
    shared = closes.index.intersection(extension.index)
    if len(shared) == 0:
        return SpliceCheck(0, float("nan"), float("nan"), float("nan"))

    relative = (extension[shared] - closes[shared]) / closes[shared]

    # The seam is the last extension date strictly before the closes begin, and
    # the first close. That join is the one price move the splice invents.
    prefix = extension.loc[extension.index < closes.index[0]]
    seam = float("nan")
    if not prefix.empty:
        seam = float(np.log(closes.iloc[0] / prefix.iloc[-1]))

    close_vol = _annualized_volatility(closes[shared])
    extension_vol = _annualized_volatility(extension[shared])
    ratio = extension_vol / close_vol if close_vol > 0 else float("nan")

    return SpliceCheck(
        overlap=len(shared),
        seam_return=seam,
        level_bias=float(relative.mean()),
        volatility_ratio=float(ratio),
    )


def build(frame: pd.DataFrame, ids: dict[str, str]) -> PriceRecord:
    """Resolve the configured roles into one canonical daily price record.

    ``frame`` is ``world.series.wide(...)`` — every price here arrived through
    the point-in-time gateway like any other observation, which is what makes
    the record replayable. Nothing in this module reads a file or a network.
    """
    close_role, closes = "", pd.Series(dtype=float)
    for role in CLOSE_ROLES:
        candidate = _role_series(frame, ids.get(role))
        if not candidate.empty:
            close_role, closes = role, candidate
            break

    history_id = ids.get(HISTORY_ROLE)
    history = _role_series(frame, history_id)

    if closes.empty:
        # No close anywhere. The average-based record alone is better than
        # nothing and is labelled as entirely average-based.
        if history.empty:
            raise PriceUnavailable(
                "no configured price role produced an observation; expected one of "
                f"{list(CLOSE_ROLES)} or {HISTORY_ROLE}"
            )
        log.warning(
            "crypto prices: no daily close is knowable; the whole record is %s, a daily average",
            history_id,
        )
        return PriceRecord(
            price=history,
            is_daily_average=pd.Series(True, index=history.index),
            close_role=HISTORY_ROLE,
            close_series_id=str(history_id),
            history_role=HISTORY_ROLE,
            history_series_id=str(history_id),
        )

    close_id = str(ids[close_role])
    unspliced = PriceRecord(
        price=closes,
        is_daily_average=pd.Series(False, index=closes.index),
        close_role=close_role,
        close_series_id=close_id,
    )

    if history.empty:
        if history_id:
            log.info(
                "crypto prices: %s has no knowable observations; the record starts where %s does",
                history_id,
                close_id,
            )
        return unspliced

    prefix = history.loc[history.index < closes.index[0]]
    if prefix.empty:
        log.info("crypto prices: %s reaches no further back than %s", history_id, close_id)
        return unspliced

    check = measure_splice(closes, history)
    refusal = check.refusal()
    if refusal is not None:
        log.error(
            "crypto prices: refusing to splice %s in front of %s — %s. Publishing the "
            "close record alone; the sample loses %d day(s) of history back to %s.",
            history_id,
            close_id,
            refusal,
            len(prefix),
            prefix.index[0].date(),
        )
        return PriceRecord(
            price=closes,
            is_daily_average=pd.Series(False, index=closes.index),
            close_role=close_role,
            close_series_id=close_id,
            history_role=HISTORY_ROLE,
            history_series_id=str(history_id),
            declined_reason=refusal,
        )

    price = pd.concat([prefix, closes]).sort_index()
    # Trailing-window statistics read this index positionally, so a duplicate
    # date would quietly shift every window by one. `prefix` is strictly before
    # `closes` by construction, which makes duplicates impossible — asserted
    # rather than assumed, because the cost of being wrong is silent.
    if price.index.has_duplicates:  # pragma: no cover - defensive
        raise PriceUnavailable(
            "the spliced record has duplicate dates; the prefix and the closes overlap"
        )

    is_average = pd.Series(False, index=price.index)
    is_average.loc[prefix.index] = True

    log.info(
        "crypto prices: %d observations %s → %s (%d from %s as a daily average, %d from %s "
        "as a close; seam %+.1f%%, level bias %+.2f%%, volatility ratio %.3f over %d "
        "shared dates)",
        len(price),
        price.index[0].date(),
        price.index[-1].date(),
        len(prefix),
        history_id,
        len(closes),
        close_id,
        check.seam_return * 100.0,
        check.level_bias * 100.0,
        check.volatility_ratio,
        check.overlap,
    )

    return PriceRecord(
        price=price,
        is_daily_average=is_average,
        close_role=close_role,
        close_series_id=close_id,
        history_role=HISTORY_ROLE,
        history_series_id=str(history_id),
    )


__all__ = [
    "CALENDAR_DAYS",
    "CLOSE_ROLES",
    "HISTORY_ROLE",
    "MAX_LEVEL_BIAS",
    "MAX_SEAM_RETURN",
    "MIN_SPLICE_OVERLAP",
    "VOLATILITY_RATIO_BOUNDS",
    "PriceRecord",
    "PriceUnavailable",
    "SpliceCheck",
    "build",
    "measure_splice",
]
