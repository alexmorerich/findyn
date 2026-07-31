"""Walk-forward event study for the regime engine (FINDYN_V1_SPEC.md §15).

Every number in the committed report comes from here, and every one of them is
**out of sample**: the model that judges 2008 was fitted on data ending before
2008. That is the only version of this measurement worth having. A model refitted
on the whole history and then asked how it did in 2008 will always look
prescient, because the episode it is being graded on is part of what taught it
what a crisis looks like.

The mechanics
-------------

Refits walk forward on a fixed cadence over an expanding window. Between refits,
the model in force is the one fitted at the last refit date, applied to features
the run could actually have seen. The resulting regime path is stitched together
from those out-of-sample segments and is what the metrics below are computed on.

What is measured, and against what
----------------------------------

* **Lead/lag** of the first bear-or-crisis call against the episode's market
  peak, in trading days. Negative is a warning before the peak.
* **Drawdown warning rate** — the share of episodes flagged before the drawdown
  reached 20%.
* **False-alarm rate** — how often an elevated call was *not* followed by a
  material drawdown. This is the number that keeps the others honest: a model
  that shouts permanently has perfect lead time and no value.
* **Brier score and a reliability table** for the calibrated transition
  probabilities.

NBER dates are carried for the three episodes that have them. They are a
recession chronology, not a market one, and are published with a lag of up to a
year — so the market peak is the primary reference and NBER is reported beside
it rather than used as the target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from findynamics.engines.equity.domain import REGIMES
from findynamics.engines.equity.regime.calibrate import (
    ADVERSE_REGIMES,
    TransitionModels,
    build_training_frame,
    fit_all_horizons,
    transition_labels,
)
from findynamics.engines.equity.regime.design import RegimeDesign
from findynamics.engines.equity.regime.hmm import RegimeModel, fit_hmm

log = logging.getLogger("findynamics.engines.equity.backtest")


@dataclass(frozen=True)
class Episode:
    """One historical stress episode, with the dates the metrics reference."""

    name: str
    #: Evaluation window — wide enough to contain the run-up and the recovery.
    start: date
    end: date
    #: The market's peak before the drawdown. The primary reference: it is a
    #: market fact, known the day it happens, and needs no committee.
    peak: date
    #: The drawdown trough.
    trough: date
    #: NBER recession start, where one exists. 2022 has none — it was a rate
    #: shock and a bear market without a recession, which is exactly why a
    #: recession chronology cannot be the target.
    nber_start: date | None = None
    note: str = ""


#: §15's four daily-resolution events. Dates are S&P 500 closing peaks and
#: troughs; NBER peaks are the official business-cycle dates.
EPISODES: tuple[Episode, ...] = (
    Episode(
        name="2000 dot-com",
        start=date(1998, 1, 1),
        end=date(2003, 12, 31),
        peak=date(2000, 3, 24),
        trough=date(2002, 10, 9),
        nber_start=date(2001, 3, 1),
        note="on the NASDAQ 100 calibration series this drawdown reached 83%",
    ),
    Episode(
        name="2008 GFC",
        start=date(2006, 1, 1),
        end=date(2010, 12, 31),
        peak=date(2007, 10, 9),
        trough=date(2009, 3, 9),
        nber_start=date(2007, 12, 1),
    ),
    Episode(
        name="2020 COVID",
        start=date(2019, 1, 1),
        end=date(2021, 6, 30),
        peak=date(2020, 2, 19),
        trough=date(2020, 3, 23),
        nber_start=date(2020, 2, 1),
        note="23 trading days peak to trough — the hardest case for a trailing-window model",
    ),
    Episode(
        name="2022 rate shock",
        start=date(2021, 1, 1),
        end=date(2023, 12, 31),
        peak=date(2022, 1, 3),
        trough=date(2022, 10, 12),
        nber_start=None,
        note="no NBER recession — a bear market without one, so the peak is the only reference",
    ),
)

#: A drawdown this deep is what "a material drawdown" means throughout.
MATERIAL_DRAWDOWN = 0.20

#: A transition probability at or above this counts as an elevated warning.
ELEVATED_THRESHOLD = 0.50

#: Refit cadence for the walk-forward, in months. Two years rather than the
#: production one month: this runs 20 HMM fits and 60 boosters as it is, and the
#: parameters of a five-state HMM over an expanding multi-decade window do not
#: move enough month to month to change any metric here.
DEFAULT_REFIT_MONTHS = 24

#: Observations required before the first out-of-sample judgement.
DEFAULT_MIN_TRAIN = 1260


@dataclass(frozen=True)
class WalkForward:
    """The stitched out-of-sample path and the fits that produced it."""

    #: Argmax regime per date, each from a model fitted strictly before it.
    regimes: pd.Series
    #: Filtered posteriors per date, same provenance.
    posteriors: pd.DataFrame
    #: Calibrated transition probability per horizon, columns ``p_3m`` etc.
    transitions: pd.DataFrame
    #: (refit date, rows judged by that fit).
    segments: tuple[tuple[date, int], ...]
    series_id: str
    is_proxy: bool

    @property
    def span(self) -> tuple[date, date]:
        return self.regimes.index[0].date(), self.regimes.index[-1].date()


def walk_forward(
    design: RegimeDesign,
    *,
    refit_months: int = DEFAULT_REFIT_MONTHS,
    min_train: int = DEFAULT_MIN_TRAIN,
    is_proxy: bool = False,
    with_transitions: bool = True,
    seed: int | None = None,
) -> WalkForward:
    """Expanding-window refits, each judging only the dates that follow it.

    The one rule this function exists to enforce: a date is judged by a model
    that could not have seen it. Everything else here is bookkeeping.
    """
    index = design.frame.index
    if len(index) < min_train + 250:
        raise ValueError(
            f"{design.series_id if hasattr(design, 'series_id') else design.series.series_id}: "
            f"{len(index)} rows cannot support a walk-forward with {min_train} training rows"
        )

    refit_dates: list[pd.Timestamp] = []
    cursor = index[min_train]
    while cursor < index[-1]:
        refit_dates.append(cursor)
        cursor = cursor + pd.DateOffset(months=refit_months)

    regime_parts: list[pd.Series] = []
    posterior_parts: list[pd.DataFrame] = []
    transition_parts: list[pd.DataFrame] = []
    segments: list[tuple[date, int]] = []

    for number, refit_at in enumerate(refit_dates):
        train = design.slice(end=refit_at)
        stop = refit_dates[number + 1] if number + 1 < len(refit_dates) else None

        # The judged block: strictly after the refit date, up to the next one.
        judged_index = index[index > refit_at]
        if stop is not None:
            judged_index = judged_index[judged_index <= stop]
        if judged_index.empty:
            continue

        try:
            fit = fit_hmm(train, is_proxy=is_proxy, **({"seed": seed} if seed else {}))
        except ValueError as err:
            log.warning("walk-forward: no fit at %s (%s)", refit_at.date(), err)
            continue
        model = RegimeModel(fit)

        # Inference runs on the whole history up to the end of the judged block —
        # the forward recursion needs the run-up to have the right prior at the
        # block's first date — and only the judged rows are kept.
        visible = design.slice(end=judged_index[-1])
        posteriors = model.posteriors(visible).loc[judged_index]
        posterior_parts.append(posteriors)
        regime_parts.append(posteriors.idxmax(axis=1))

        if with_transitions:
            transition_parts.append(
                _transition_block(train, visible, model, judged_index, is_proxy=is_proxy)
            )

        segments.append((refit_at.date(), len(judged_index)))
        log.info(
            "walk-forward: fit through %s judges %d rows to %s",
            refit_at.date(),
            len(judged_index),
            judged_index[-1].date(),
        )

    if not regime_parts:
        raise ValueError("walk-forward produced no out-of-sample segments")

    return WalkForward(
        regimes=pd.concat(regime_parts).rename("regime"),
        posteriors=pd.concat(posterior_parts),
        transitions=(pd.concat(transition_parts) if transition_parts else pd.DataFrame()),
        segments=tuple(segments),
        series_id=design.series.series_id,
        is_proxy=is_proxy,
    )


def _transition_block(
    train: RegimeDesign,
    visible: RegimeDesign,
    model: RegimeModel,
    judged_index: pd.DatetimeIndex,
    *,
    is_proxy: bool,
) -> pd.DataFrame:
    """Calibrated transition probabilities for one out-of-sample block."""
    train_posteriors = model.posteriors(train)
    train_states = train_posteriors.idxmax(axis=1)
    models = fit_all_horizons(
        train,
        train_posteriors,
        train_states,
        fitted_on=train.series.series_id,
        is_proxy=is_proxy,
    )
    if not models:
        return pd.DataFrame(index=judged_index)

    features = build_training_frame(visible, model.posteriors(visible))
    usable = features.reindex(judged_index).dropna()
    if usable.empty:
        return pd.DataFrame(index=judged_index)

    return pd.DataFrame(
        {f"p_{months}m": m.predict(usable) for months, m in sorted(models.models.items())}
    ).reindex(judged_index)


# --- metrics -----------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeResult:
    """What the model did on one episode."""

    episode: Episode
    #: First bear-or-crisis call inside the window, if any.
    first_warning: date | None
    #: Trading days from the warning to the market peak. Negative is early.
    lead_days: int | None
    #: Drawdown already suffered when the warning fired. Lower is better.
    drawdown_at_warning: float | None
    #: True when the warning came before the drawdown reached 20%.
    warned_before_material: bool
    #: Share of the peak-to-trough window called bear or crisis.
    coverage: float
    #: Share called ``late_cycle`` or worse.
    #:
    #: Reported beside the strict number because the two answer different
    #: questions, and conflating them produced a badly wrong reading of the
    #: deep-history cross-check. Labels are relative to *each series' own* return
    #: distribution: monthly returns are far less fat-tailed than daily ones, so
    #: the monthly fit's extreme states sit at milder values and a 50% drawdown
    #: can land in ``late_cycle``. Strict coverage then reads 0% and looks like
    #: blindness, when the model in fact moved down the risk ordering and stayed
    #: there. "Did it deteriorate" is the question that survives a change of
    #: frequency; "did it say crisis" is not.
    deteriorated_coverage: float
    #: Mean severity rank over the fall, 0 (bull_expansion) to 4 (crisis).
    mean_severity: float
    #: Days from the trough to the first non-adverse call — recovery lag.
    recovery_lag_days: int | None
    nber_lead_days: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode.name,
            "first_warning": None if self.first_warning is None else self.first_warning.isoformat(),
            "lead_days": self.lead_days,
            "drawdown_at_warning": self.drawdown_at_warning,
            "warned_before_material": self.warned_before_material,
            "coverage": self.coverage,
            "deteriorated_coverage": self.deteriorated_coverage,
            "mean_severity": self.mean_severity,
            "recovery_lag_days": self.recovery_lag_days,
            "nber_lead_days": self.nber_lead_days,
        }


#: Where each regime sits on the risk-on ordering: 0 is the most risk-on.
SEVERITY: dict[str, int] = {label: rank for rank, label in enumerate(REGIMES)}

#: ``late_cycle`` and worse. The looser reading of "the engine deteriorated".
DETERIORATED_REGIMES: frozenset[str] = frozenset({"late_cycle", "bear", "crisis"})


def drawdown_path(log_price: pd.Series) -> pd.Series:
    """Fractional drawdown from the running maximum, as a positive number."""
    return 1.0 - np.exp(log_price - log_price.cummax())


def evaluate_episode(
    walk: WalkForward,
    log_price: pd.Series,
    episode: Episode,
    *,
    adverse: frozenset[str] = ADVERSE_REGIMES,
) -> EpisodeResult | None:
    """Lead/lag and coverage for one episode. ``None`` if it is out of span."""
    window = walk.regimes.loc[str(episode.start) : str(episode.end)]
    if window.empty:
        return None

    peak = pd.Timestamp(episode.peak)
    trough = pd.Timestamp(episode.trough)
    prices = log_price.loc[window.index[0] : window.index[-1]]
    if prices.empty:
        return None

    # Drawdown from the *running* maximum, which is the only version knowable on
    # the day. Measuring distance below the episode's eventual peak would report
    # a warning fired months *before* that peak as arriving 20% into a drawdown
    # that had not started — the price was simply lower then than it later got.
    drawdown = drawdown_path(prices)

    adverse_calls = window[window.isin(adverse)]
    # Warnings are only interesting from the run-up onwards; a call two years
    # early during an unrelated wobble is a false alarm, not foresight.
    run_up = adverse_calls.loc[str(peak - pd.DateOffset(months=12)) :]

    first_warning: date | None = None
    lead_days: int | None = None
    drawdown_at_warning: float | None = None
    if not run_up.empty:
        stamp = run_up.index[0]
        first_warning = stamp.date()
        peak_position = window.index.get_indexer([peak], method="nearest")[0]
        lead_days = int(window.index.get_loc(stamp) - peak_position)
        value = drawdown.asof(stamp)
        drawdown_at_warning = float(value) if np.isfinite(value) else None

    warned_before_material = bool(
        drawdown_at_warning is not None and drawdown_at_warning < MATERIAL_DRAWDOWN
    )

    fall = window.loc[peak:trough] if peak <= trough else window.iloc[:0]
    coverage = float(fall.isin(adverse).mean()) if not fall.empty else 0.0
    deteriorated = float(fall.isin(DETERIORATED_REGIMES).mean()) if not fall.empty else 0.0
    severity = float(fall.map(SEVERITY).mean()) if not fall.empty else float("nan")

    after = window.loc[trough:]
    benign = after[~after.isin(adverse)]
    recovery_lag = int(after.index.get_loc(benign.index[0])) if not benign.empty else None

    nber_lead = None
    if episode.nber_start is not None and first_warning is not None:
        nber_lead = (first_warning - episode.nber_start).days

    return EpisodeResult(
        episode=episode,
        first_warning=first_warning,
        lead_days=lead_days,
        drawdown_at_warning=drawdown_at_warning,
        warned_before_material=warned_before_material,
        coverage=coverage,
        deteriorated_coverage=deteriorated,
        mean_severity=severity,
        recovery_lag_days=recovery_lag,
        nber_lead_days=nber_lead,
    )


def false_alarm_rate(
    walk: WalkForward,
    log_price: pd.Series,
    *,
    horizon_months: int = 12,
    periods_per_year: float = 252.0,
    threshold: float = MATERIAL_DRAWDOWN,
    adverse: frozenset[str] = ADVERSE_REGIMES,
) -> dict[str, float]:
    """P(no material drawdown within h | the engine called bear or crisis).

    §15's fourth metric, and the one that stops the others being gamed. Measured
    from a *running* maximum rather than an episode peak, because a false alarm
    is by definition not part of an episode anyone has named.
    """
    horizon = max(int(round(horizon_months * periods_per_year / 12.0)), 1)
    prices = log_price.reindex(walk.regimes.index).ffill()

    # Worst drawdown from today's level over the next `horizon` observations.
    forward_min = prices[::-1].rolling(window=horizon, min_periods=1).min()[::-1].shift(-1)
    forward_drawdown = 1.0 - np.exp(forward_min - prices)

    called = walk.regimes.isin(adverse)
    usable = forward_drawdown.notna()
    alarms = called & usable
    if not alarms.any():
        return {"alarms": 0.0, "false_alarm_rate": float("nan"), "hit_rate": float("nan")}

    followed = forward_drawdown[alarms] >= threshold
    return {
        "alarms": float(alarms.sum()),
        "false_alarm_rate": float(1.0 - followed.mean()),
        "hit_rate": float(followed.mean()),
    }


def reliability_table(
    probabilities: pd.Series,
    labels: pd.Series,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    """Predicted versus realized frequency per probability decile.

    The reliability diagram §15 asks for, as the table behind it. A model can
    rank well and still be badly calibrated, and only this shows it.
    """
    joined = pd.concat([probabilities.rename("p"), labels.rename("y")], axis=1).dropna()
    if joined.empty:
        return pd.DataFrame(columns=["bin", "count", "predicted", "observed"])

    edges = np.linspace(0.0, 1.0, bins + 1)
    joined["bin"] = np.clip(np.digitize(joined["p"], edges[1:-1]), 0, bins - 1)

    rows = []
    for index, group in joined.groupby("bin"):
        rows.append(
            {
                "bin": f"{edges[int(index)]:.1f}-{edges[int(index) + 1]:.1f}",
                "count": int(len(group)),
                "predicted": float(group["p"].mean()),
                "observed": float(group["y"].mean()),
            }
        )
    return pd.DataFrame(rows)


def brier(probabilities: pd.Series, labels: pd.Series) -> float:
    joined = pd.concat([probabilities.rename("p"), labels.rename("y")], axis=1).dropna()
    if joined.empty:
        return float("nan")
    return float(((joined["p"] - joined["y"]) ** 2).mean())


@dataclass
class BacktestResult:
    """Everything the committed report is rendered from."""

    walk: WalkForward
    episodes: list[EpisodeResult] = field(default_factory=list)
    false_alarms: dict[str, float] = field(default_factory=dict)
    brier_by_horizon: dict[int, float] = field(default_factory=dict)
    base_rate_by_horizon: dict[int, float] = field(default_factory=dict)
    reliability: dict[int, pd.DataFrame] = field(default_factory=dict)
    #: The monthly deep-history cross-check, when one was run.
    cross_check: dict[str, Any] | None = None


def run_backtest(
    design: RegimeDesign,
    log_price: pd.Series,
    *,
    refit_months: int = DEFAULT_REFIT_MONTHS,
    min_train: int = DEFAULT_MIN_TRAIN,
    is_proxy: bool = False,
    reference_regimes: pd.Series | None = None,
) -> BacktestResult:
    """Walk forward, then compute every §15 metric on the out-of-sample path.

    ``reference_regimes`` supplies the *labels*: what the regime actually turned
    out to be, from a single model fitted on the whole history. Predictions must
    be out of sample; labels are ex-post history and should not be.

    Taking labels from the stitched walk-forward path instead is a trap worth
    naming, because it looks more rigorous and is badly wrong: each refit can
    disagree slightly with the last, so every one of the eighteen segment
    boundaries can manufacture a regime "entry" that never happened. Against 52
    genuine entries in 39 years, that contaminates a third of the label set — and
    it inflates the measured base rate, which is what a calibrated probability is
    graded against.
    """
    walk = walk_forward(design, refit_months=refit_months, min_train=min_train, is_proxy=is_proxy)

    episodes = [
        result
        for episode in EPISODES
        if (result := evaluate_episode(walk, log_price, episode)) is not None
    ]

    if reference_regimes is None:
        reference_regimes = RegimeModel(fit_hmm(design, is_proxy=is_proxy)).states(design)

    brier_by_horizon: dict[int, float] = {}
    base_rate: dict[int, float] = {}
    reliability: dict[int, pd.DataFrame] = {}
    if not walk.transitions.empty:
        for column in walk.transitions.columns:
            months = int(column.removeprefix("p_").removesuffix("m"))
            labels = (
                transition_labels(reference_regimes, months, design.periods_per_year)
                .reindex(walk.regimes.index)
                .dropna()
            )
            probabilities = walk.transitions[column]
            brier_by_horizon[months] = brier(probabilities, labels)
            aligned = labels.reindex(probabilities.dropna().index).dropna()
            base_rate[months] = float(aligned.mean()) if not aligned.empty else float("nan")
            reliability[months] = reliability_table(probabilities, labels)

    return BacktestResult(
        walk=walk,
        episodes=episodes,
        false_alarms=false_alarm_rate(walk, log_price, periods_per_year=design.periods_per_year),
        brier_by_horizon=brier_by_horizon,
        base_rate_by_horizon=base_rate,
        reliability=reliability,
    )


def cross_check_deep_history(
    deep: RegimeDesign,
    log_price: pd.Series,
    *,
    refit_months: int = 60,
    min_train: int = 360,
) -> dict[str, Any]:
    """The monthly 1871+ cross-check, and what it licenses.

    The calibration series is the NASDAQ 100 — a proxy, and a more volatile one.
    The deep-history series is genuinely the S&P composite across all four
    episodes. So running the same pipeline on it, at monthly resolution, asks the
    question the proxy cannot answer about itself: *does a model built this way
    identify these episodes on the real index?*

    Agreement is what licenses the transfer. Disagreement is a stop sign for
    sub-milestone C, not a footnote — it would mean the episodes the proxy taught
    the model to recognise are the proxy's, not the market's.
    """
    walk = walk_forward(
        deep,
        refit_months=refit_months,
        min_train=min_train,
        is_proxy=False,
        with_transitions=False,
    )
    results = [
        result
        for episode in EPISODES
        if (result := evaluate_episode(walk, log_price, episode)) is not None
    ]
    return {
        "series": deep.series.series_id,
        "frequency": deep.series.frequency,
        "span": [walk.span[0].isoformat(), walk.span[1].isoformat()],
        "episodes": [r.as_dict() for r in results],
        "flagged": sorted(r.episode.name for r in results if r.coverage > 0.25),
        "deteriorated": sorted(
            r.episode.name for r in results if r.deteriorated_coverage > 0.5 and r.coverage <= 0.25
        ),
        # An episode that reached bear/crisis cannot also be "missed" — the
        # first version of this compared only against the looser threshold and
        # listed 2000 as missed while also listing it as called.
        "missed": sorted(
            r.episode.name for r in results if r.coverage <= 0.25 and r.deteriorated_coverage <= 0.5
        ),
    }


def _pct(value: float | None, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def render_report(
    result: BacktestResult,
    *,
    calibration_series: str,
    is_proxy: bool,
    generated_at: str,
) -> str:
    """The committed markdown artifact (§15).

    Written to be read by someone deciding whether to trust the engine, so the
    failures are in the summary rather than in a footnote.
    """
    walk = result.walk
    lines: list[str] = [
        "# FinEquity — walk-forward backtest (P3-B)",
        "",
        f"Generated {generated_at} · FINDYN_V1_SPEC.md §15",
        "",
        "## What was measured",
        "",
        f"- **Calibration series**: `{calibration_series}`"
        + ("  ⚠️ **a proxy, not the S&P 500**" if is_proxy else ""),
        f"- **Out-of-sample span**: {walk.span[0]} → {walk.span[1]}",
        f"- **Refits**: {len(walk.segments)} expanding-window fits; every date is",
        "  judged by a model fitted strictly before it.",
        "",
    ]

    if is_proxy:
        lines += [
            "> Every fitted parameter below comes from the NASDAQ 100, because daily",
            "> S&P history before 2016 is not reachable (the Stooq endpoint",
            "> bot-filters CI as well as developer networks). The NASDAQ is a more",
            "> volatile, tech-heavy index. The transfer is defensible only because",
            "> the design matrix is dimensionless — see the deep-history cross-check",
            "> at the end, which runs the same pipeline on the real S&P composite.",
            "",
        ]

    lines += [
        "## Episode detection",
        "",
        "Lead is in trading days against the market peak; negative is a warning",
        "*before* the peak. Drawdown-at-warning is measured from the running",
        "maximum, which is the only version knowable on the day.",
        "",
        "| Episode | First warning | Lead (d) | Drawdown at warning | Before −20%? |"
        " Bear/crisis | late_cycle+ | Recovery lag (d) |",
        "|---|---|---:|---:|:-:|---:|---:|---:|",
    ]
    for item in result.episodes:
        lines.append(
            f"| {item.episode.name} | {item.first_warning or '—'} | "
            f"{'—' if item.lead_days is None else item.lead_days} | "
            f"{_pct(item.drawdown_at_warning)} | "
            f"{'yes' if item.warned_before_material else 'no'} | "
            f"{_pct(item.coverage)} | {_pct(item.deteriorated_coverage)} | "
            f"{'—' if item.recovery_lag_days is None else item.recovery_lag_days} |"
        )

    nber = [i for i in result.episodes if i.nber_lead_days is not None]
    if nber:
        lines += [
            "",
            "Against the NBER recession start, for the three episodes that have one",
            "(2022 was a bear market without a recession):",
            "",
            "| Episode | NBER start | Warning lead vs NBER (calendar days) |",
            "|---|---|---:|",
        ]
        for item in nber:
            lines.append(
                f"| {item.episode.name} | {item.episode.nber_start} | {item.nber_lead_days} |"
            )

    alarms = result.false_alarms
    lines += [
        "",
        "## False alarms",
        "",
        "P(no drawdown ≥ 20% within 12 months | the engine called bear or crisis),",
        "measured per session over the whole out-of-sample span.",
        "",
        f"- Sessions called bear or crisis: **{int(alarms.get('alarms', 0)):,}**",
        f"- False-alarm rate: **{_pct(alarms.get('false_alarm_rate'))}**",
        f"- Hit rate: **{_pct(alarms.get('hit_rate'))}**",
        "",
        "## Transition probabilities",
        "",
        "Brier score of the calibrated probability against what the regime",
        "actually did, out of sample. The reference is the Brier score of always",
        "predicting the base rate — the score to beat.",
        "",
        "| Horizon | Brier | Base rate | Reference Brier | Skill |",
        "|---|---:|---:|---:|---:|",
    ]
    for months in sorted(result.brier_by_horizon):
        score = result.brier_by_horizon[months]
        base = result.base_rate_by_horizon.get(months, float("nan"))
        reference = base * (1 - base)
        skill = 1 - score / reference if reference > 0 else float("nan")
        lines.append(
            f"| {months}m | {score:.4f} | {_pct(base)} | {reference:.4f} | {_pct(skill)} |"
        )

    for months in sorted(result.reliability):
        table = result.reliability[months]
        if table.empty:
            continue
        lines += [
            "",
            f"### Reliability, {months}m",
            "",
            "| Predicted band | Observations | Mean predicted | Observed frequency |",
            "|---|---:|---:|---:|",
        ]
        for _, row in table.iterrows():
            lines.append(
                f"| {row['bin']} | {int(row['count']):,} | "
                f"{row['predicted']:.3f} | {row['observed']:.3f} |"
            )

    if result.cross_check:
        check = result.cross_check
        lines += [
            "",
            "## Deep-history cross-check",
            "",
            f"The same pipeline on `{check['series']}` ({check['frequency']}, "
            f"{check['span'][0]} → {check['span'][1]}) — genuinely the S&P composite",
            "across all four episodes, where the calibration series is a proxy.",
            "",
            "| Episode | First warning | Bear/crisis | late_cycle or worse | Mean severity |",
            "|---|---|---:|---:|---:|",
        ]
        for item in check["episodes"]:
            lines.append(
                f"| {item['episode']} | {item['first_warning'] or '—'} | "
                f"{_pct(item['coverage'])} | {_pct(item['deteriorated_coverage'])} | "
                f"{item['mean_severity']:.2f} |"
            )
        lines += [
            "",
            f"Called bear or crisis: {', '.join(check['flagged']) or 'none'}.",
            f"Deteriorated to late_cycle or worse: {', '.join(check['deteriorated']) or 'none'}.",
            f"Missed entirely: {', '.join(check['missed']) or 'none'}.",
        ]

    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_MIN_TRAIN",
    "DEFAULT_REFIT_MONTHS",
    "ELEVATED_THRESHOLD",
    "EPISODES",
    "MATERIAL_DRAWDOWN",
    "BacktestResult",
    "Episode",
    "EpisodeResult",
    "TransitionModels",
    "WalkForward",
    "brier",
    "cross_check_deep_history",
    "drawdown_path",
    "evaluate_episode",
    "false_alarm_rate",
    "reliability_table",
    "render_report",
    "run_backtest",
    "walk_forward",
]
