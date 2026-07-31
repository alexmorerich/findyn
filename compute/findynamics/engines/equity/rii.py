"""The Regime Instability Index (FINDYN_V1_SPEC.md §3.2).

Snap — the fourth derivative of price — is **replaced entirely** by this. §3.1 is
explicit about why: financial series are noisy, each differentiation amplifies
that noise, and a fourth derivative of a daily index is essentially all
microstructure. So the quantity that answers "how close is this system to a
state transition" is built as a composite of things that are individually
measurable, not as a derivative nobody can estimate.

Seven components, §3.2's table:

=========================  =================================================
posterior entropy          −Σ p·log p over the regime posterior. High means
                           the model cannot tell which regime this is, and
                           model uncertainty *is* instability.
confidence deficit         1 − max posterior. The same idea read the other
                           way; both are published because a two-way tie and
                           a five-way smear are different situations.
jerk                       |z| of Δacceleration, from §3.1. Already a
                           thresholded indicator, never a raw derivative.
vol of vol                 realized volatility of realized volatility. A
                           market whose volatility is itself unstable is
                           closer to a transition than a calmly volatile one.
correlation breakdown      rolling stock-bond correlation against its own
                           3-year baseline. When the hedge stops hedging,
                           the system has changed state.
credit velocity            rate of change of the high-yield spread. Credit
                           usually moves first.
liquidity stress           NFCI level and its 13-week change.
=========================  =================================================

Every component is mapped to 0-100 by the **same causal percentile the shared
factors use** (``factors.compute.score_series``): winsorized z, then expanding
percentile. Expanding and never centred, so a 2008 reading is ranked against
1960-2008 and not against a future that had not happened. Reusing that function
rather than writing a second scoring pipeline is deliberate — two implementations
of "0-100 score" would eventually disagree about what 50 means.

**Missing components do not silently become zero.** A run without a credit
spread has six components, not seven with one scored 0, because 0 on this axis
means "maximally stable" and absence means nothing of the kind. The weights are
renormalized over what is present and the omissions are published.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from findynamics.factors.compute import score_series

log = logging.getLogger("findynamics.engines.equity.rii")

#: §3.2 — equal weights by default, and configurable because the spec says so.
#: Equal is a real choice rather than a placeholder: nothing in the record
#: justifies asserting that credit leads liquidity by a factor of 1.4, and a
#: weight vector fitted on eight episodes would be fitted on eight observations.
DEFAULT_WEIGHTS: dict[str, float] = {
    "posterior_entropy": 1.0,
    "confidence_deficit": 1.0,
    "jerk": 1.0,
    "vol_of_vol": 1.0,
    "correlation_breakdown": 1.0,
    "credit_velocity": 1.0,
    "liquidity_stress": 1.0,
}

#: Windows, in months. Quoted in months and converted against each series' own
#: frequency, so the monthly deep-history path means the same thing.
DEFAULT_VOL_OF_VOL_MONTHS = 3.0
DEFAULT_CORRELATION_MONTHS = 3.0
DEFAULT_CORRELATION_BASELINE_YEARS = 3.0
DEFAULT_CREDIT_VELOCITY_MONTHS = 1.0
DEFAULT_LIQUIDITY_CHANGE_WEEKS = 13.0

#: Below this many scored observations a component is noise, not a percentile.
MIN_COMPONENT_OBSERVATIONS = 60


@dataclass(frozen=True)
class RiiResult:
    """The index, its parts, and what it could not see."""

    #: 0-100 per date. Higher is closer to a transition.
    index: pd.Series
    #: Component name -> its own 0-100 score, for the explanation trace.
    components: dict[str, pd.Series]
    #: Weights actually used, renormalized over the components present.
    weights: dict[str, float]
    #: Components the information set could not supply.
    missing: tuple[str, ...] = ()

    @property
    def latest(self) -> float | None:
        usable = self.index.dropna()
        return None if usable.empty else float(usable.iloc[-1])

    def latest_components(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, series in self.components.items():
            usable = series.dropna()
            if not usable.empty and np.isfinite(usable.iloc[-1]):
                out[name] = float(usable.iloc[-1])
        return out

    def explain(self) -> dict[str, float]:
        """Per-component score and its contribution to the published index."""
        latest = self.latest_components()
        trace = {f"rii_{name}": value for name, value in latest.items()}
        for name in latest:
            trace[f"rii_{name}_weight"] = round(self.weights.get(name, 0.0), 6)
        return trace


def _periods(months: float, periods_per_year: float, floor: int = 2) -> int:
    return max(int(round(months * periods_per_year / 12.0)), floor)


def posterior_entropy(posteriors: pd.DataFrame) -> pd.Series:
    """−Σ p·log p, per date. Zero is certainty; log(5) is a uniform guess."""
    values = posteriors.to_numpy(dtype=float)
    safe = np.clip(values, 1e-12, 1.0)
    return pd.Series(
        -(safe * np.log(safe)).sum(axis=1), index=posteriors.index, name="posterior_entropy"
    )


def vol_of_vol(
    realized: pd.Series,
    *,
    periods_per_year: float,
    months: float = DEFAULT_VOL_OF_VOL_MONTHS,
) -> pd.Series:
    """Realized volatility of the realized-volatility series itself.

    §3.2 asks for "realized vol of rolling realized vol (63d window of 21d RV)".
    ``realized`` is already the 21-day series, so this is the outer window only.
    """
    window = _periods(months, periods_per_year)
    return realized.rolling(window=window, min_periods=window).std().rename("vol_of_vol")


def correlation_breakdown(
    equity_returns: pd.Series,
    bond_yield: pd.Series,
    *,
    periods_per_year: float,
    months: float = DEFAULT_CORRELATION_MONTHS,
    baseline_years: float = DEFAULT_CORRELATION_BASELINE_YEARS,
) -> pd.Series:
    """How far the stock-bond correlation has moved from its own baseline.

    Bonds are given as a *yield*, so the bond return proxy is the negative change
    in yield — a falling yield is a rising bond. Getting that sign wrong would
    invert the whole component and it would still look plausible, which is why it
    is stated here rather than left in the arithmetic.

    The published value is the absolute deviation from the trailing baseline, not
    the correlation itself: the instability signal is the hedge *changing*, in
    either direction, and a correlation that has simply always been positive is
    not news.
    """
    window = _periods(months, periods_per_year)
    baseline_window = _periods(baseline_years * 12.0, periods_per_year)

    bond_returns = -bond_yield.diff()
    # sort=True explicitly. The two series rarely share a calendar exactly — an
    # equity close and a Treasury yield miss different holidays — so the union
    # must come back in date order. pandas is deprecating the implicit sort, and
    # the future default would silently put the rolling correlation out of order.
    aligned = pd.concat([equity_returns, bond_returns], axis=1, sort=True).dropna()
    if len(aligned) < window * 2:
        return pd.Series(dtype=float, name="correlation_breakdown")

    rolling = aligned.iloc[:, 0].rolling(window=window, min_periods=window).corr(aligned.iloc[:, 1])
    baseline = rolling.rolling(
        window=baseline_window, min_periods=max(baseline_window // 2, window)
    ).mean()
    return (rolling - baseline).abs().rename("correlation_breakdown")


def rate_of_change(
    series: pd.Series,
    *,
    periods_per_year: float,
    months: float,
) -> pd.Series:
    """Absolute change over a trailing window. Backward-looking by construction."""
    return series.diff(_periods(months, periods_per_year)).abs()


def compute_rii(
    posteriors: pd.DataFrame,
    *,
    jerk_z: pd.Series | None = None,
    realized_vol: pd.Series | None = None,
    equity_returns: pd.Series | None = None,
    bond_yield: pd.Series | None = None,
    credit_spread: pd.Series | None = None,
    liquidity: pd.Series | None = None,
    periods_per_year: float = 252.0,
    weights: dict[str, float] | None = None,
) -> RiiResult:
    """The §3.2 composite over the dates ``posteriors`` covers.

    Everything except the posterior is optional. A run that cannot see the credit
    spread still publishes an index — from six components, with the seventh named
    — because an instability reading that disappears whenever one FRED series is
    late is worse than one that says what it was built from.
    """
    resolved = dict(DEFAULT_WEIGHTS if weights is None else weights)

    entropy = posterior_entropy(posteriors)
    raw: dict[str, pd.Series] = {
        "posterior_entropy": entropy,
        "confidence_deficit": 1.0 - posteriors.max(axis=1),
    }

    if jerk_z is not None:
        # §3.1: jerk enters as a magnitude. Direction is the market's business;
        # instability is about the size of the change in trend, either way.
        raw["jerk"] = jerk_z.abs()

    if realized_vol is not None:
        raw["vol_of_vol"] = vol_of_vol(realized_vol, periods_per_year=periods_per_year)

    if equity_returns is not None and bond_yield is not None:
        raw["correlation_breakdown"] = correlation_breakdown(
            equity_returns, bond_yield, periods_per_year=periods_per_year
        )

    if credit_spread is not None:
        raw["credit_velocity"] = rate_of_change(
            credit_spread,
            periods_per_year=periods_per_year,
            months=DEFAULT_CREDIT_VELOCITY_MONTHS,
        )

    if liquidity is not None:
        # Level and 13-week change together, as §3.2 asks. The level says how
        # tight conditions are; the change says whether they are still tightening.
        change = rate_of_change(
            liquidity,
            periods_per_year=periods_per_year,
            months=DEFAULT_LIQUIDITY_CHANGE_WEEKS / 4.345,
        )
        raw["liquidity_stress"] = liquidity.add(change, fill_value=0.0)

    scored: dict[str, pd.Series] = {}
    missing: list[str] = []
    for name in resolved:
        series = raw.get(name)
        if series is None:
            missing.append(name)
            continue
        # direction=+1: a higher raw value is a *more unstable* reading, and the
        # percentile is taken on that axis, so 100 is maximally unstable. This is
        # the opposite orientation to the shared factors, where 100 is maximally
        # supportive — the axis is named by what the number measures.
        component = score_series(series.reindex(posteriors.index).dropna(), 1)
        if len(component) < MIN_COMPONENT_OBSERVATIONS:
            missing.append(name)
            continue
        scored[name] = component.reindex(posteriors.index)

    if not scored:
        raise ValueError("the RII needs at least one usable component and has none")

    total = sum(resolved[name] for name in scored)
    used = {name: resolved[name] / total for name in scored}

    frame = pd.DataFrame(scored)
    # Renormalized per date over the components that actually have a value there,
    # so the warm-up of a slow component does not drag the whole index toward
    # zero while it fills.
    weight_row = pd.Series(used)
    mask = frame.notna()
    weighted = (frame.fillna(0.0) * weight_row).sum(axis=1)
    coverage = (mask * weight_row).sum(axis=1)
    index = (weighted / coverage.replace(0.0, np.nan)).clip(0.0, 100.0)

    if missing:
        log.info(
            "rii: built from %d of %d components; missing %s",
            len(scored),
            len(resolved),
            ", ".join(sorted(missing)),
        )

    return RiiResult(
        index=index.rename("rii"),
        components=scored,
        weights=used,
        missing=tuple(sorted(missing)),
    )


__all__ = [
    "DEFAULT_CORRELATION_BASELINE_YEARS",
    "DEFAULT_CORRELATION_MONTHS",
    "DEFAULT_CREDIT_VELOCITY_MONTHS",
    "DEFAULT_LIQUIDITY_CHANGE_WEEKS",
    "DEFAULT_VOL_OF_VOL_MONTHS",
    "DEFAULT_WEIGHTS",
    "MIN_COMPONENT_OBSERVATIONS",
    "RiiResult",
    "compute_rii",
    "correlation_breakdown",
    "posterior_entropy",
    "rate_of_change",
    "vol_of_vol",
]
