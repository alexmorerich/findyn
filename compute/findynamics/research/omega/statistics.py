"""The statistics that decide whether any KK3 number means anything.

Four rules from ``docs/research/kk-omega-design.md`` §8 live here, and each one
exists because its absence produces a flattering answer:

**Out-of-sample R² is measured against the training-window mean.** Against the
*test*-window mean it is lookahead — the benchmark would know the average of the
period being predicted — and the resulting number is reliably positive for a
model with no skill at all.

**Overlapping horizons destroy naive standard errors.** At ``h = 63`` each
observation shares 62 days with its neighbour, so the effective sample is a
fraction of the nominal one. t-statistics are Newey–West with ``lag >= h - 1``;
an uncorrected ``t = 3.0`` at that horizon is approximately noise.

**Multiplicity.** Six arms × four horizons × five questions is a lot of chances
to be lucky. Every p-value is reported beside a Benjamini–Hochberg q-value and
the family is named. A result that survives only unadjusted is reported as not
surviving.

**Performance metrics are the repository's, not a second set.**
``findynamics/backtest/portfolio.py::_max_drawdown`` is imported rather than
reimplemented — it anchors the curve at 1.0 so a first-period decline counts,
which a fresh implementation would get wrong once and then disagree forever.
Its neighbour ``_metrics`` is **not** reused: it hard-codes a monthly cadence
(``12.0 / n``, ``sqrt(12)``) because the P6 backtest rebalances monthly, and
this track runs daily. Rather than edit a module the portfolio backtest depends
on, the annualization is a parameter here and the design note records the
choice. Sortino, Calmar, hit rate and turnover do not exist there at all.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# The repository's definition of a drawdown. Private, and imported anyway: a
# second definition of "worst peak-to-trough" is how two reports come to
# disagree about the same strategy.
from findynamics.backtest.portfolio import _max_drawdown

log = logging.getLogger("findynamics.research.omega.statistics")

#: Trading days per year, for annualizing a daily path.
PERIODS_PER_YEAR = 252.0


def newey_west_lag(n: int, horizon: int = 1) -> int:
    """``max(floor(4·(n/100)^0.25), h - 1)``.

    The Newey-West rule, floored at the overlap the horizon itself creates. At
    ``h = 63`` the rule alone would pick 11 on a 6,000-row sample and leave 51
    lags of mechanical autocorrelation uncorrected.
    """
    if n <= 0:
        # No observations, no lags — and the horizon floor cannot invent them.
        # `max(n, 1)` below would otherwise report a lag for an empty sample.
        return max(horizon - 1, 0)
    rule = max(int(math.floor(4.0 * (n / 100.0) ** 0.25)), 0)
    return max(rule, horizon - 1)


def hac_tstat(values: pd.Series, horizon: int = 1) -> tuple[float, float, int]:
    """``(t, p, lag)`` for the mean of ``values`` under Newey-West.

    Used on differences — of squared errors, of returns — where the question is
    "is this mean different from zero" and the observations overlap.
    """
    clean = np.asarray(values.dropna(), dtype=float)
    n = len(clean)
    if n < 3:
        return float("nan"), float("nan"), 0
    lag = newey_west_lag(n, horizon)
    centred = clean - clean.mean()

    variance = float(centred @ centred) / n
    for j in range(1, min(lag, n - 1) + 1):
        gamma = float(centred[j:] @ centred[:-j]) / n
        variance += 2.0 * (1.0 - j / (lag + 1.0)) * gamma
    if not np.isfinite(variance) or variance <= 0.0:
        return float("nan"), float("nan"), lag

    t = float(clean.mean() / math.sqrt(variance / n))
    p = float(2.0 * stats.norm.sf(abs(t)))
    return t, p, lag


def oos_r2(actual: pd.Series, predicted: pd.Series, training_mean: float) -> float:
    """``1 − SSE / SST`` with ``SST`` taken against the **training** mean.

    The one line in this module most likely to be "simplified" into
    ``actual.mean()``, which is the lookahead form and reliably produces small
    positive numbers for a model with no skill. ``training_mean`` is a scalar
    the caller computed on data the model was allowed to see.
    """
    pair = pd.concat([actual.rename("y"), predicted.rename("f")], axis=1, sort=True).dropna()
    if pair.empty:
        return float("nan")
    errors = pair["y"] - pair["f"]
    baseline = pair["y"] - training_mean
    sse = math.fsum(errors**2)
    sst = math.fsum(baseline**2)
    if sst <= 0.0:
        return float("nan")
    return 1.0 - sse / sst


def delta_r2_test(
    actual: pd.Series,
    challenger: pd.Series,
    baseline: pd.Series,
    *,
    horizon: int,
) -> dict[str, float]:
    """Is the challenger's squared error smaller than the baseline's?

    A Diebold-Mariano test on the loss differential, with Newey-West standard
    errors. Reported as a one-sided reading, because "does adding Ω help" has a
    direction and a two-sided p-value would call a *worse* model significant.
    """
    frame = pd.concat(
        [actual.rename("y"), challenger.rename("c"), baseline.rename("b")],
        axis=1,
        sort=True,
    ).dropna()
    if len(frame) < 30:
        return {"dm_t": float("nan"), "dm_p": float("nan"), "lag": 0.0, "n": float(len(frame))}

    differential = (frame["y"] - frame["b"]) ** 2 - (frame["y"] - frame["c"]) ** 2
    t, _, lag = hac_tstat(differential, horizon)
    # One-sided: the alternative is "the challenger has smaller loss".
    p = float(stats.norm.sf(t)) if np.isfinite(t) else float("nan")
    return {"dm_t": t, "dm_p": p, "lag": float(lag), "n": float(len(frame))}


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """BH-adjusted q-values, aligned to the input index.

    NaNs pass through as NaN and are excluded from the family size, so a test
    that could not be run does not make the surviving ones look better.
    """
    clean = p_values.dropna()
    if clean.empty:
        return pd.Series(np.nan, index=p_values.index, dtype=float)

    ordered = clean.sort_values()
    m = len(ordered)
    ranks = np.arange(1, m + 1, dtype=float)
    raw = ordered.to_numpy(dtype=float) * m / ranks
    # Enforce monotonicity from the largest p downwards — the step-up procedure.
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    return pd.Series(np.clip(adjusted, 0.0, 1.0), index=ordered.index).reindex(p_values.index)


def rank_ic(feature: pd.Series, target: pd.Series) -> float:
    """Spearman correlation of a feature against a forward target."""
    pair = pd.concat([feature.rename("x"), target.rename("y")], axis=1, sort=True).dropna()
    if len(pair) < 30:
        return float("nan")
    return float(pair["x"].corr(pair["y"], method="spearman"))


def auc(labels: pd.Series, scores: pd.Series) -> float:
    """Area under the ROC curve, from ranks rather than from a sweep.

    The Mann-Whitney form: exact, tie-aware, and one pass. A threshold sweep
    would agree to three decimals and disagree on the fourth depending on how
    many thresholds it tried.
    """
    pair = pd.concat([labels.rename("y"), scores.rename("s")], axis=1, sort=True).dropna()
    positives = pair["y"] > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pair["s"].rank()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def brier(labels: pd.Series, probabilities: pd.Series) -> float:
    """Mean squared error of a probability forecast."""
    pair = pd.concat([labels.rename("y"), probabilities.rename("p")], axis=1, sort=True).dropna()
    if pair.empty:
        return float("nan")
    return float(math.fsum((pair["p"] - pair["y"]) ** 2) / len(pair))


@dataclass(frozen=True)
class Performance:
    """What a decision rule did, after costs."""

    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    turnover: float
    cost_adjusted_cagr: float
    #: Cost in basis points per unit turnover at which the edge over holding
    #: cash disappears. A finding, not a footnote — an edge that dies at 3 bps
    #: is not an edge.
    break_even_bps: float
    periods: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cagr": self.cagr,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "hit_rate": self.hit_rate,
            "turnover": self.turnover,
            "cost_adjusted_cagr": self.cost_adjusted_cagr,
            "break_even_bps": self.break_even_bps,
            "periods": self.periods,
        }


def performance(
    exposure: pd.Series,
    returns: pd.Series,
    *,
    cost_bps: float,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> Performance:
    """Grade a decision rule. ``exposure`` at *t* earns ``returns`` at *t*.

    The caller is responsible for having lagged the exposure — this function
    multiplies the two as given, and a caller that passes today's signal against
    today's return is measuring its own alignment error. The walk-forward's
    exposures come off predictions made strictly before the return they earn.
    """
    frame = pd.concat([exposure.rename("w"), returns.rename("r")], axis=1, sort=True).dropna()
    if len(frame) < 2:
        empty = float("nan")
        return Performance(empty, empty, empty, empty, empty, empty, empty, 0.0, empty, empty, 0.0)

    weights = frame["w"]
    gross = weights * frame["r"]
    traded = weights.diff().abs().fillna(weights.abs().iloc[:1].reindex(weights.index).fillna(0.0))
    turnover = float(math.fsum(traded))
    net = gross - traded * (cost_bps / 10_000.0)

    n = len(frame)
    years = n / periods_per_year
    growth = float(np.exp(math.fsum(np.log1p(np.clip(net, -0.999999, None)))))
    gross_growth = float(np.exp(math.fsum(np.log1p(np.clip(gross, -0.999999, None)))))

    volatility = float(net.std(ddof=1) * math.sqrt(periods_per_year))
    downside = net[net < 0.0]
    downside_vol = (
        float(downside.std(ddof=1) * math.sqrt(periods_per_year)) if len(downside) > 1 else 0.0
    )
    cagr = growth ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    gross_cagr = gross_growth ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    drawdown = _max_drawdown(net)

    # The cost per unit turnover that eats the gross growth entirely. Reported
    # in basis points so it can be read against a real commission schedule.
    total_gross = math.fsum(gross)
    break_even = (total_gross / turnover * 10_000.0) if turnover > 0 else float("inf")

    return Performance(
        cagr=gross_cagr,
        volatility=volatility,
        sharpe=(float(net.mean() / net.std(ddof=1) * math.sqrt(periods_per_year)))
        if net.std(ddof=1) > 0
        else float("nan"),
        sortino=(float(net.mean() / downside_vol * math.sqrt(periods_per_year)))
        if downside_vol > 0
        else float("nan"),
        max_drawdown=drawdown,
        calmar=(cagr / abs(drawdown)) if drawdown < 0 else float("nan"),
        hit_rate=float((gross > 0).mean()),
        turnover=turnover,
        cost_adjusted_cagr=cagr,
        break_even_bps=break_even,
        periods=float(n),
    )


def _placement_values(positives: np.ndarray, negatives: np.ndarray) -> np.ndarray:
    """DeLong structural components: each positive's share of wins over negatives."""
    order = np.argsort(negatives)
    ordered = negatives[order]
    below = np.searchsorted(ordered, positives, side="left")
    equal = np.searchsorted(ordered, positives, side="right") - below
    return (below + 0.5 * equal) / len(negatives)


def delong_auc_test(
    labels: pd.Series,
    left: pd.Series,
    right: pd.Series,
) -> dict[str, float]:
    """Paired test of ``AUC(left) − AUC(right)`` on the dates both cover.

    DeLong's method: the variance of an AUC difference comes from the structural
    components — how each positive ranks against the negatives, and vice versa —
    rather than from a bootstrap. Deterministic, which a bootstrap is not, and
    this is a number the pre-registered Q3 rule depends on.

    Paired on purpose. Two AUCs computed on different samples are not
    comparable, and their difference has no standard error worth quoting.
    """
    frame = pd.concat(
        [labels.rename("y"), left.rename("l"), right.rename("r")], axis=1, sort=True
    ).dropna()
    positives = frame["y"] > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    empty = {"delta_auc": float("nan"), "z": float("nan"), "p": float("nan"), "n": 0.0}
    if n_pos < 5 or n_neg < 5:
        return empty

    out: dict[str, np.ndarray] = {}
    aucs: dict[str, float] = {}
    for key in ("l", "r"):
        scores = frame[key].to_numpy(dtype=float)
        pos, neg = scores[positives.to_numpy()], scores[~positives.to_numpy()]
        v10 = _placement_values(pos, neg)
        v01 = 1.0 - _placement_values(neg, pos)
        out[f"{key}_10"], out[f"{key}_01"] = v10, v01
        aucs[key] = float(v10.mean())

    s10 = np.cov(np.vstack([out["l_10"], out["r_10"]]))
    s01 = np.cov(np.vstack([out["l_01"], out["r_01"]]))
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ s10 @ contrast) / n_pos + float(contrast @ s01 @ contrast) / n_neg
    delta = aucs["l"] - aucs["r"]
    if not np.isfinite(variance) or variance <= 0.0:
        return {**empty, "delta_auc": delta, "n": float(len(frame))}

    z = delta / math.sqrt(variance)
    # One-sided: the alternative is "the challenger discriminates better".
    return {
        "delta_auc": delta,
        "z": float(z),
        "p": float(stats.norm.sf(z)),
        "n": float(len(frame)),
    }


__all__ = [
    "PERIODS_PER_YEAR",
    "Performance",
    "auc",
    "delong_auc_test",
    "benjamini_hochberg",
    "brier",
    "delta_r2_test",
    "hac_tstat",
    "newey_west_lag",
    "oos_r2",
    "performance",
    "rank_ic",
]
