"""Liquidity state — is the funding market clearing?

Two observables, both in ``config/engines/money.yaml``:

* **the bill-SOFR spread**, 3m constant-maturity bill minus the overnight rate.
  Negative means bills are rich to repo. It goes sharply negative for two
  opposite reasons — repo blowing out above bills (September 2019) or bills being
  scrambled for below repo (March 2020) — and both are funding stress.
* **reverse-repo take-up**, the cash the system has parked at the Fed because it
  has nothing better to do with it. The drainable buffer; its trend says whether
  that buffer is filling or emptying.

**The hard part, stated plainly.** The spread also goes to about -0.5 in an
ordinary easing cycle, for a completely benign reason: bills price the cuts that
are coming while the overnight rate has not moved yet. In September 2024 it
reached -0.49 with nothing whatsoever wrong. In March 2020 it reached -0.87 with
a great deal wrong. The *level* of the spread cannot separate those, and any rule
that tries will either miss March 2020 or call every easing cycle a crisis.

What separates them is what else is moving. A telegraphed cut moves bills slowly
and leaves repo alone; stress moves one of them violently. So ``stressed``
requires a sustained dislocation **and** a fast mover behind it — either bills
collapsing (flight to quality) or the overnight rate climbing (funding squeeze).
On the real series 2018-2026 that combination fires on eleven sessions: 17-18
September 2019 and 4-16 March 2020. Nothing else, and specifically not September
2024, which fails both legs independently.

One honest caveat, recorded because a future recalibration should know it: the
turn-of-year squeeze of 31 December 2018 (SOFR 3.00% against a 2.40% bill) misses
``stressed`` by three basis points of the sustained-spread threshold and reads
``tightening``. It is a genuine borderline case, not a comfortable margin. Widen
``stressed_spread_pp`` by 0.05 and it flips.

Before SOFR (2018) the spread cannot be formed at all, and the state is decided
by the reverse-repo trend alone — which was near zero for most of that history,
so the honest answer there is usually ``normal`` with reduced confidence. The
engine says so through ``confidence`` rather than inventing a reading.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from findynamics.engines.money.domain import MONEY_REGIMES

log = logging.getLogger("findynamics.engines.money.liquidity")


@dataclass(frozen=True)
class LiquidityRules:
    """Every threshold the classifier branches on. All of it configuration."""

    #: How stale the newest usable bill-SOFR spread may be before the state falls
    #: back to reverse-repo alone. Calendar days. Needed because SOFR is published
    #: a day ahead of the constant-maturity bill, so the newest row of the rate
    #: path routinely has no spread on it (see MoneyEngine._latest_inputs).
    max_spread_staleness_days: int = 5

    #: Observations averaged for the "sustained" spread read.
    spread_window: int = 3
    #: Observations spanned by the bill / overnight trend reads.
    trend_window: int = 10
    #: Observations in the reverse-repo short and long trend averages.
    rrp_fast_window: int = 5
    rrp_slow_window: int = 63

    #: A single-session spread this negative is a repo blowout on its own — the
    #: September 2019 case, where the overnight rate tripled for one day.
    stressed_spike_pp: float = -1.50
    #: Sustained dislocation gate. See the module docstring on why this is not a
    #: comfortable distance from an ordinary easing cycle.
    stressed_spread_pp: float = -0.55
    #: ...confirmed by bills collapsing over the trend window (dash for cash)...
    stressed_bill_drop_pp: float = -0.50
    #: ...or by the overnight rate climbing over it (funding squeeze).
    stressed_sofr_jump_pp: float = 0.25

    #: Persistently through the overnight rate: the buffer is being used.
    tightening_spread_pp: float = -0.20
    #: Reverse-repo fast average this far below its slow average, while the
    #: balance is material enough for the ratio to mean anything.
    tightening_rrp_ratio: float = 0.75

    #: Reverse-repo take-up (fast average, $bn) that counts as a real buffer
    #: rather than rounding error. Below this the trend carries no information —
    #: 2019 ran at $3bn, where a 50% swing is one counterparty.
    rrp_material_bn: float = 100.0
    #: ...and above this, plus a buffer that is not shrinking, is `abundant`.
    abundant_rrp_bn: float = 250.0
    abundant_rrp_ratio: float = 0.90

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> LiquidityRules:
        raw = params.get("liquidity") or {}
        if not isinstance(raw, dict):
            raise ValueError("engines/money.yaml: 'liquidity' must be a mapping")
        known = {f: getattr(cls, f) for f in cls.__dataclass_fields__}
        unknown = set(raw) - set(known)
        if unknown:
            raise ValueError(
                f"engines/money.yaml liquidity: unknown threshold(s) {sorted(unknown)}; "
                f"expected some of {sorted(known)}"
            )
        values: dict[str, Any] = {}
        for name, default in known.items():
            value = raw.get(name, default)
            values[name] = int(value) if isinstance(default, int) else float(value)
        return cls(**values)


@dataclass(frozen=True)
class LiquidityInputs:
    """What the classifier reads on one date. ``None`` where unobservable."""

    as_of: pd.Timestamp
    #: 3m bill minus overnight, percentage points.
    spread: float | None
    #: Mean spread over ``spread_window`` observations.
    spread_mean: float | None
    #: Change in the bill yield over ``trend_window`` observations, pp.
    bill_change: float | None
    #: Change in the overnight rate over the same window, pp.
    overnight_change: float | None
    #: Reverse-repo fast average, $bn.
    rrp_level: float | None
    #: Fast average divided by slow average. 1.0 is flat.
    rrp_ratio: float | None

    @property
    def has_spread(self) -> bool:
        return self.spread is not None and self.spread_mean is not None


def build_inputs(
    frame: pd.DataFrame,
    rules: LiquidityRules,
    *,
    bill_id: str,
    overnight_id: str,
    rrp_id: str,
) -> pd.DataFrame:
    """Per-date classifier inputs over the whole history in ``frame``.

    Every transform is trailing — rolling, never centred, and shifted where a
    difference is taken — so the row for a date depends on that date and earlier
    ones only. A centred average here would let next week decide whether this week
    was stressed, which is the failure mode §14.1 rule 3 exists to prevent.
    """
    if frame.empty:
        return pd.DataFrame()

    bill = _numeric(frame, bill_id)
    overnight = _numeric(frame, overnight_id)
    rrp = _numeric(frame, rrp_id)

    out = pd.DataFrame(index=frame.index.copy())
    out["bill"] = bill
    out["overnight"] = overnight

    if bill is not None and overnight is not None:
        out["spread"] = bill - overnight
        out["spread_mean"] = (
            out["spread"].rolling(rules.spread_window, min_periods=rules.spread_window).mean()
        )
    else:
        out["spread"] = np.nan
        out["spread_mean"] = np.nan

    # Differences over the trend window, on each series' own observations.
    out["bill_change"] = (
        bill.diff(rules.trend_window) if bill is not None else pd.Series(np.nan, index=out.index)
    )
    out["overnight_change"] = (
        overnight.diff(rules.trend_window)
        if overnight is not None
        else pd.Series(np.nan, index=out.index)
    )

    if rrp is not None:
        fast = rrp.rolling(rules.rrp_fast_window, min_periods=1).mean()
        slow = rrp.rolling(rules.rrp_slow_window, min_periods=rules.rrp_fast_window).mean()
        out["rrp_level"] = fast
        # A slow average at or near zero makes the ratio meaningless rather than
        # infinite; the `rrp_material_bn` gate is what actually protects the rule,
        # but NaN here keeps a division from deciding anything.
        out["rrp_ratio"] = (fast / slow.where(slow > 1e-9)).replace([np.inf, -np.inf], np.nan)
    else:
        out["rrp_level"] = np.nan
        out["rrp_ratio"] = np.nan

    return out


def _numeric(frame: pd.DataFrame, series_id: str) -> pd.Series | None:
    if series_id not in frame.columns:
        return None
    values = pd.to_numeric(frame[series_id], errors="coerce")
    return None if values.dropna().empty else values


def inputs_on(row: pd.Series, key: pd.Timestamp) -> LiquidityInputs:
    """One row of :func:`build_inputs` as a :class:`LiquidityInputs`."""
    return LiquidityInputs(
        as_of=pd.Timestamp(key),
        spread=_opt(row.get("spread")),
        spread_mean=_opt(row.get("spread_mean")),
        bill_change=_opt(row.get("bill_change")),
        overnight_change=_opt(row.get("overnight_change")),
        rrp_level=_opt(row.get("rrp_level")),
        rrp_ratio=_opt(row.get("rrp_ratio")),
    )


def _opt(value: object) -> float | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def is_stressed(inputs: LiquidityInputs, rules: LiquidityRules) -> bool:
    """Funding stress: a spike, or a sustained dislocation with a fast mover.

    See the module docstring for why the second leg needs the confirmation. In
    short: the sustained-spread gate alone would also catch every easing cycle.
    """
    if inputs.spread is not None and inputs.spread <= rules.stressed_spike_pp:
        return True

    if inputs.spread_mean is None or inputs.spread_mean > rules.stressed_spread_pp:
        return False

    bills_collapsing = (
        inputs.bill_change is not None and inputs.bill_change <= rules.stressed_bill_drop_pp
    )
    repo_climbing = (
        inputs.overnight_change is not None
        and inputs.overnight_change >= rules.stressed_sofr_jump_pp
    )
    return bills_collapsing or repo_climbing


def is_tightening(inputs: LiquidityInputs, rules: LiquidityRules) -> bool:
    """The buffer is being consumed: bills through repo, or reverse repo draining."""
    if inputs.spread_mean is not None and inputs.spread_mean <= rules.tightening_spread_pp:
        return True
    return (
        inputs.rrp_level is not None
        and inputs.rrp_level >= rules.rrp_material_bn
        and inputs.rrp_ratio is not None
        and inputs.rrp_ratio <= rules.tightening_rrp_ratio
    )


def is_abundant(inputs: LiquidityInputs, rules: LiquidityRules) -> bool:
    """More cash than the system has uses for, and not shrinking."""
    return (
        inputs.rrp_level is not None
        and inputs.rrp_level >= rules.abundant_rrp_bn
        and inputs.rrp_ratio is not None
        and inputs.rrp_ratio >= rules.abundant_rrp_ratio
    )


def classify(inputs: LiquidityInputs, rules: LiquidityRules) -> str:
    """The liquidity state on one date. Order is the priority order.

    Mutually exclusive by construction: the first matching rule wins, and stress
    outranks a draining buffer because a market that is not clearing is the more
    important fact about it.
    """
    if is_stressed(inputs, rules):
        return "stressed"
    if is_tightening(inputs, rules):
        return "tightening"
    if is_abundant(inputs, rules):
        return "abundant"
    return "normal"


def classify_history(frame: pd.DataFrame, rules: LiquidityRules) -> pd.Series:
    """Every date in :func:`build_inputs`' output, classified causally.

    Each date is judged from its own trailing window, so this is the sequence of
    calls a daily run standing on each of those dates would have made — not a
    relabelling of history with today's knowledge.
    """
    if frame.empty:
        return pd.Series(dtype=object)
    labels = [classify(inputs_on(row, key), rules) for key, row in frame.iterrows()]
    return pd.Series(labels, index=frame.index, name="liquidity")


def explain(inputs: LiquidityInputs, rules: LiquidityRules) -> dict[str, float]:
    """The numbers the classification was made from, for the explainability trace."""
    components: dict[str, float] = {}
    for name, value in (
        ("bill_sofr_spread", inputs.spread),
        ("bill_sofr_spread_mean", inputs.spread_mean),
        ("bill_change", inputs.bill_change),
        ("overnight_change", inputs.overnight_change),
        ("rrp_level_bn", inputs.rrp_level),
        ("rrp_ratio", inputs.rrp_ratio),
    ):
        if value is not None:
            components[name] = round(value, 6)
    components["stressed_spread_threshold_pp"] = rules.stressed_spread_pp
    components["tightening_spread_threshold_pp"] = rules.tightening_spread_pp
    return components


__all__ = [
    "MONEY_REGIMES",
    "LiquidityInputs",
    "LiquidityRules",
    "build_inputs",
    "classify",
    "classify_history",
    "explain",
    "inputs_on",
    "is_abundant",
    "is_stressed",
    "is_tightening",
]
