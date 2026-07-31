"""M4 diagnostics: does the tail fit, does the RII discriminate, does the MC drift?

The M3 backtest answers "did the regime model see the crises". This answers the
three questions M4 raises, and it is a *separate* report because the answers are
not all favourable and burying them in a longer document is how they stop being
read.

What each section is for:

**GPD fit.** §4 requires the shock factor to come from extreme-value theory on
the deep history, and an EVT fit is only as good as its independence assumption.
The section reports the declustered episode count beside the raw exceedance
count, because the gap between them (1039 → 46) is the difference between a
fitted tail and a fitted autocorrelation. Return levels are printed against the
empirical record so a reader can see where the fit disagrees with what happened.

**RII discrimination.** The index is measured at its peak over each episode
against a calm baseline, per component. Measured on the *calibration* path, not
the published one: every component is an expanding percentile, the published path
is ten years long, and grading a century-scale index on it produced a 1.3-point
separation that was read as a flaw in §3.2 for longer than it should have been
(open issue 12). The per-component table is what makes that visible.

**Monte Carlo calibration.** A simulation whose median 12-year path implies a
return the market has never delivered is wrong regardless of how good its tails
look. The section compares simulated annualized drift and drawdown frequencies
against the realized record over the same span.

Reproducible from the committed snapshot with no API keys, like the M3 report.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from findynamics.engines.equity import crash as crash_mod
from findynamics.engines.equity import simulate as simulate_mod
from findynamics.engines.equity.backtest import EPISODES

log = logging.getLogger("findynamics.engines.equity.diagnostics")

#: Calm reference windows for the RII comparison. Chosen as years with no
#: episode in `EPISODES` and no drawdown past 10% — not as years the index
#: happens to score low, which would be choosing the answer.
CALM_YEARS: tuple[tuple[str, date, date], ...] = (
    ("1995", date(1995, 1, 1), date(1995, 12, 31)),
    ("2005", date(2005, 1, 1), date(2005, 12, 31)),
    ("2017", date(2017, 1, 1), date(2017, 12, 31)),
    ("2021", date(2021, 1, 1), date(2021, 12, 31)),
)

#: Return periods, in years, to report severities for.
RETURN_PERIODS: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0, 100.0)


@dataclass(frozen=True)
class TailDiagnostics:
    fit: crash_mod.TailFit
    #: Exceedances before declustering — the number an unguarded fit would use.
    raw_exceedances: int
    #: Fitted vs empirical severity at each return period.
    return_levels: pd.DataFrame
    #: The declustered episode depths themselves, worst first.
    episodes: pd.Series


def tail_diagnostics(
    log_price: pd.Series,
    *,
    fit: crash_mod.TailFit,
) -> TailDiagnostics:
    """Fit quality for the GPD behind ``p_shock``."""
    drawdowns = crash_mod.drawdown_magnitudes(log_price).dropna()
    raw = int((drawdowns > fit.threshold).sum())
    episodes = crash_mod.decluster(drawdowns, fit.threshold).sort_values(ascending=False)

    rows = []
    for years in RETURN_PERIODS:
        # The severity exceeded once per `years`, from the fitted tail: solve
        # survival(x) * rate * periods_in(years) = 1.
        observations = years * fit.periods_per_year
        target = 1.0 / (fit.exceedance_rate * observations)
        rows.append(
            {
                "return_period_years": years,
                "fitted_severity": _inverse_survival(fit, target),
                "empirical_severity": _empirical_return_level(episodes, fit, years),
            }
        )

    return TailDiagnostics(
        fit=fit,
        raw_exceedances=raw,
        return_levels=pd.DataFrame(rows),
        episodes=episodes,
    )


def _inverse_survival(fit: crash_mod.TailFit, probability: float) -> float:
    """Severity whose conditional exceedance probability is ``probability``."""
    if probability <= 0.0 or probability >= 1.0:
        return float("nan")
    if abs(fit.shape) < 1e-8:
        excess = -fit.scale * math.log(probability)
    else:
        excess = fit.scale / fit.shape * (probability**-fit.shape - 1.0)
    return fit.threshold + excess


def _empirical_return_level(episodes: pd.Series, fit: crash_mod.TailFit, years: float) -> float:
    """The observed severity exceeded once per ``years`` in the record.

    Nan when the record is shorter than the return period, rather than an
    extrapolation — the point of printing this column is to have something the
    fit can be wrong against, and an extrapolated "empirical" value is just the
    fit again under another name.
    """
    span_years = fit.observations / fit.periods_per_year
    expected_count = span_years / years
    if expected_count < 1.0 or episodes.empty:
        return float("nan")
    rank = int(round(expected_count)) - 1
    if rank >= len(episodes):
        return float("nan")
    return float(episodes.iloc[rank])


@dataclass(frozen=True)
class RiiDiagnostics:
    #: Episode name -> RII at the trough.
    episode_readings: pd.Series
    #: Calm window name -> mean RII over that window.
    calm_readings: pd.Series
    #: Component -> (episode mean − calm mean). The evidence for issue 12.
    component_gaps: pd.Series
    #: Episode mean − calm mean for the composite.
    gap: float


def rii_diagnostics(index: pd.Series, components: dict[str, pd.Series]) -> RiiDiagnostics:
    """How far the RII separates the episodes from calm years, component by component."""
    usable = index.dropna()
    if usable.empty:
        raise ValueError("the RII is empty over this window")

    episode_readings: dict[str, float] = {}
    for episode in EPISODES:
        window = _window(usable, episode.peak, episode.trough)
        if not window.empty:
            # The maximum over peak-to-trough, not the value at the trough: an
            # instability index earns its keep by being high *going in*, and by
            # the trough the transition has already happened.
            episode_readings[episode.name] = float(window.max())

    calm_readings = {
        name: float(_window(usable, start, end).mean())
        for name, start, end in CALM_YEARS
        if not _window(usable, start, end).empty
    }

    if not episode_readings or not calm_readings:
        raise ValueError("not enough overlap with the episodes or the calm windows")

    episode_mean = float(np.mean(list(episode_readings.values())))
    calm_mean = float(np.mean(list(calm_readings.values())))

    gaps: dict[str, float] = {}
    for name, series in components.items():
        usable_component = series.dropna()
        if usable_component.empty:
            continue
        episode_values = [
            float(_window(usable_component, e.peak, e.trough).max())
            for e in EPISODES
            if not _window(usable_component, e.peak, e.trough).empty
        ]
        calm_values = [
            float(_window(usable_component, start, end).mean())
            for _, start, end in CALM_YEARS
            if not _window(usable_component, start, end).empty
        ]
        if episode_values and calm_values:
            gaps[name] = float(np.mean(episode_values) - np.mean(calm_values))

    return RiiDiagnostics(
        episode_readings=pd.Series(episode_readings),
        calm_readings=pd.Series(calm_readings),
        component_gaps=pd.Series(gaps).sort_values(ascending=False),
        gap=episode_mean - calm_mean,
    )


def _window(series: pd.Series, start: date, end: date) -> pd.Series:
    return series.loc[pd.Timestamp(start) : pd.Timestamp(end)]


@dataclass(frozen=True)
class SimulationDiagnostics:
    #: Horizon -> annualized drift implied by the simulated median.
    simulated_drift: pd.Series
    #: The realized annualized drift of the calibration record, same units.
    realized_drift: float
    #: Horizon -> P(max drawdown > 20%) from the simulation.
    simulated_drawdown: pd.Series
    #: The realized frequency of a 20% drawdown starting per horizon-length window.
    realized_drawdown: pd.Series


def simulation_diagnostics(
    result: simulate_mod.SimulationResult,
    log_price: pd.Series,
    *,
    periods_per_year: float,
) -> SimulationDiagnostics:
    """Simulated drift and drawdown frequency against the realized record."""
    drift: dict[str, float] = {}
    drawdown: dict[str, float] = {}
    for name, forecast in result.forecasts.items():
        implied = (forecast.median_log_level - result.start_log_level) / forecast.years
        drift[name] = float(math.expm1(implied))
        drawdown[name] = float(forecast.drawdown_probabilities.get(0.20, float("nan")))

    span_years = len(log_price) / periods_per_year
    realized = float(math.expm1((log_price.iloc[-1] - log_price.iloc[0]) / span_years))

    # Realized frequency: over rolling windows of the horizon's length, how often
    # did a 20% drawdown from the window's running peak occur? Overlapping
    # windows, so this is a frequency and not an independent-trial estimate —
    # stated because the simulated column *is* an independent-trial estimate and
    # comparing them without saying so would overstate the agreement.
    realized_drawdown: dict[str, float] = {}
    for name, forecast in result.forecasts.items():
        window = int(round(forecast.years * periods_per_year))
        if window < 2 or window > len(log_price):
            continue
        peak = log_price.rolling(window=window, min_periods=window).max()
        depth = 1.0 - np.exp(log_price - peak)
        realized_drawdown[name] = float((depth > 0.20).mean())

    return SimulationDiagnostics(
        simulated_drift=pd.Series(drift),
        realized_drift=realized,
        simulated_drawdown=pd.Series(drawdown),
        realized_drawdown=pd.Series(realized_drawdown),
    )


# --------------------------------------------------------------------- report


def render_report(
    *,
    tail: TailDiagnostics | None,
    rii: RiiDiagnostics | None,
    simulation: SimulationDiagnostics | None,
    calibration_series: str,
    generated_at: str,
) -> str:
    lines: list[str] = [
        "# FinEquity M4 — instability diagnostics",
        "",
        f"Generated {generated_at} from the committed price snapshot. "
        f"Calibration series `{calibration_series}`.",
        "",
        "Regenerate with `python -m jobs.diagnostics`. This report is committed "
        "deliberately and not on a schedule: it is an artifact of a model version, "
        "and a file that changed under a cron would make every number in it "
        "un-anchored.",
        "",
        "> Read this beside `equity-open-issues.md`: this report is the measurement,",
        "> that one is the interpretation and what is still wrong.",
        "",
    ]

    lines += _tail_section(tail)
    lines += _rii_section(rii)
    lines += _simulation_section(simulation)
    return "\n".join(lines) + "\n"


def _tail_section(tail: TailDiagnostics | None) -> list[str]:
    if tail is None:
        return [
            "## 1. Tail fit (GPD)",
            "",
            "No fit. The deep-history series was unavailable or held too few "
            "drawdown episodes to fit, and `p_shock` falls back to a flagged "
            "unconditional base rate — `tail_fitted = 0` in the published detail.",
            "",
        ]

    fit = tail.fit
    lines = [
        "## 1. Tail fit (GPD)",
        "",
        f"Peaks-over-threshold on `{fit.series_id}`, {fit.observations} observations "
        f"at {fit.periods_per_year:g}/year.",
        "",
        "| | |",
        "|---|---|",
        f"| threshold | {fit.threshold:.0%} drawdown |",
        f"| exceedances, raw | {tail.raw_exceedances} |",
        f"| exceedances, declustered | {fit.exceedances} |",
        f"| shape ξ | {fit.shape:+.4f} |",
        f"| scale β | {fit.scale:.4f} |",
        f"| exceedance rate | {fit.exceedance_rate:.4%} per observation |",
        "",
        f"**Declustering removes {tail.raw_exceedances - fit.exceedances} of "
        f"{tail.raw_exceedances} exceedances.** Every period a drawdown stays below "
        "the threshold is an exceedance under a naive count, so one 2008 arrives as "
        "hundreds of 'independent' tail events. Fitting a GPD to that fits the "
        "autocorrelation of drawdowns, not their tail, and the fitted rate is the "
        "fraction of history spent underwater rather than the frequency of crashes.",
        "",
    ]

    if fit.shape > 0:
        lines += [
            f"ξ = {fit.shape:+.4f} > 0 — a heavy (Fréchet-domain) tail with no finite "
            "upper bound. That is the expected sign for equity drawdowns and it is "
            "what makes the severity at long return periods extrapolate rather than "
            "saturate.",
            "",
        ]
    else:
        lines += [
            f"ξ = {fit.shape:+.4f} ≤ 0 — a bounded tail, implying a finite worst "
            f"possible drawdown of {fit.threshold - fit.scale / fit.shape:.1%}. Treat "
            "that bound as an artifact of the sample rather than a fact about markets.",
            "",
        ]

    lines += [
        "### Return levels",
        "",
        "| return period | fitted | empirical |",
        "|---|---|---|",
    ]
    for row in tail.return_levels.itertuples():
        empirical = (
            "—" if not np.isfinite(row.empirical_severity) else f"{row.empirical_severity:.1%}"
        )
        lines.append(
            f"| 1 in {row.return_period_years:g}y | {row.fitted_severity:.1%} | {empirical} |"
        )

    span = fit.observations / fit.periods_per_year
    lines += [
        "",
        f"The record is {span:.0f} years long, so the empirical column stops where "
        "the record does. Nothing is extrapolated into it — a fit is only testable "
        "against something that is not the fit.",
        "",
        "### The declustered episodes, worst first",
        "",
        "| rank | depth |",
        "|---|---|",
    ]
    for rank, depth in enumerate(tail.episodes.head(10), start=1):
        lines.append(f"| {rank} | {depth:.1%} |")
    lines += ["", ""]
    return lines


def _rii_section(rii: RiiDiagnostics | None) -> list[str]:
    if rii is None:
        return [
            "## 2. RII discrimination",
            "",
            "Not computed — the index was empty over the evaluation window.",
            "",
        ]

    lines = [
        "## 2. RII discrimination",
        "",
        "Peak RII over each episode's peak-to-trough window, against the mean over "
        "calm years. The maximum rather than the trough reading: an instability "
        "index earns its keep by being high *going in*, and by the trough the "
        "transition has already happened.",
        "",
        "| episode | peak RII |",
        "|---|---|",
    ]
    for name, value in rii.episode_readings.items():
        lines.append(f"| {name} | {value:.1f} |")
    lines += ["", "| calm year | mean RII |", "|---|---|"]
    for name, value in rii.calm_readings.items():
        lines.append(f"| {name} | {value:.1f} |")

    verdict = "weak" if rii.gap < 10 else "moderate" if rii.gap < 25 else "strong"
    lines += [
        "",
        f"**Separation: {rii.gap:+.1f} points on a 0–100 scale — {verdict}.**",
        "",
        "### Per component",
        "",
        "Episode mean minus calm mean, per component. A negative row is a component "
        "that reads *calmer* during crises than during calm years.",
        "",
        "| component | episode − calm |",
        "|---|---|",
    ]
    for name, value in rii.component_gaps.items():
        lines.append(f"| {name} | {value:+.1f} |")
    lines.append("")

    negative = [name for name, value in rii.component_gaps.items() if value < 0]
    if negative:
        lines += [
            "",
            f"**{len(negative)} of {len(rii.component_gaps)} components move against the "
            f"composite**: {', '.join(str(n) for n in negative)}.",
            "",
            "For the posterior-entropy and confidence-deficit terms this is structural "
            "rather than a defect. §3.2 reads model uncertainty as instability, on the "
            "theory that a model which cannot tell which regime this is has identified "
            "an ambiguous moment. That holds at regime *boundaries* and fails at regime "
            "*cores* — in a crash the HMM is highly confident it is in the crisis "
            "state, so its entropy collapses exactly when the market is least stable.",
            "",
            "The composite is published with every component's own score beside it "
            "rather than reweighted to hide this. Fitting six weights to four episodes "
            "is not a model, and whether §3.2's formulation should change is a spec "
            "question. See open issue 12.",
            "",
        ]
    else:
        lines += [
            "Every component moves with the composite. That is the result §3.2 "
            "predicts and it is not what an earlier build measured — see open issue "
            "12 for what changed and why the earlier reading was taken on too short "
            "a window to mean anything.",
            "",
        ]
    return lines


def _simulation_section(simulation: SimulationDiagnostics | None) -> list[str]:
    if simulation is None:
        return [
            "## 3. Monte Carlo calibration",
            "",
            "Not computed — no simulation was available for this run.",
            "",
        ]

    lines = [
        "## 3. Monte Carlo calibration",
        "",
        "Annualized drift implied by the simulated median at each horizon, against "
        "the realized drift of the calibration record. A simulation whose median "
        "12-year path implies a return the market has never delivered is wrong "
        "however good its tails look.",
        "",
        f"Realized: **{simulation.realized_drift:+.2%}/yr** over the calibration record.",
        "",
        "| horizon | simulated median drift | P(drawdown > 20%) | realized frequency |",
        "|---|---|---|---|",
    ]
    for name, value in simulation.simulated_drift.items():
        probability = simulation.simulated_drawdown.get(name, float("nan"))
        realized = simulation.realized_drawdown.get(name, float("nan"))
        lines.append(
            f"| {name} | {value:+.2%}/yr | "
            f"{'—' if not np.isfinite(probability) else f'{probability:.1%}'} | "
            f"{'—' if not np.isfinite(realized) else f'{realized:.1%}'} |"
        )

    drift_gap = float(
        np.mean([abs(v - simulation.realized_drift) for v in simulation.simulated_drift])
    )
    lines += [
        "",
        f"Mean absolute drift gap: **{drift_gap:.2%}/yr**.",
        "",
        "The two right-hand columns are not directly comparable and the difference "
        "matters: the simulated probability is over independent paths, the realized "
        "frequency is over overlapping rolling windows of the same record. Read them "
        "for order of magnitude, not for calibration.",
        "",
        "The shock overlay retraces fully (`SHOCK_RETRACE = 1.0`). The permanent loss "
        "from historical crashes is already inside the fitted regime means; a partial "
        "retrace subtracts it a second time, which measured out at 2.7%/yr against a "
        "6.1% realized. The overlay contributes the *shape* of a tail event — a "
        "discontinuity over days — and the regimes contribute the level.",
        "",
    ]
    return lines


__all__ = [
    "CALM_YEARS",
    "RETURN_PERIODS",
    "RiiDiagnostics",
    "SimulationDiagnostics",
    "TailDiagnostics",
    "render_report",
    "rii_diagnostics",
    "simulation_diagnostics",
    "tail_diagnostics",
]
