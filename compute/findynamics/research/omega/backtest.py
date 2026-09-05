"""The walk-forward harness — where leakage would actually enter.

Everything KK3 reports must come from a model that had never seen the
observation it is being graded on. The unit tests in KK1 and KK2 cover the
transforms; they do not cover the harness, and the harness is where a leak is
both easiest to introduce and hardest to see.

**One PIT gateway, reused.** ``findynamics/backtest/replay.py::world_at`` binds a
``PandasPITAccessor`` to the cutoff, so nothing downstream can widen the
information set. A second gateway written here would be a second place for the
release-date filter to be subtly wrong, and the first one is already tested.

What is refit per window and what is frozen — and why
-----------------------------------------------------

Refitting *everything* at every rebalance date is the obviously safe design and
it is not affordable: an unfrozen ``compute_features`` on the century price path
costs 4.7 seconds, so 259 monthly rebalances would be twenty minutes before a
single arm is fitted. Freezing everything is affordable and wrong. The split
below is drawn on one question — **does this quantity, computed once and read at
date t, depend on data after t?**

Refit at every rebalance, because the answer is yes:

``Ω``
    A PCA is a projection onto axes chosen from the window, and the axes move.
    This is the thing under test, so it gets the expensive treatment: scaler and
    loadings fitted on data through ``t``, then applied to ``(t, t+cadence]``
    and never re-applied backwards.
regime posteriors
    ``RegimeModel.posteriors`` is forward-backward, which conditions on the
    whole sequence handed to it. Computed once at the end it would smooth
    2008 using 2026. Recomputed per window it smooths ``[0, t]``, which is what
    a run at *t* would have published — and it costs 30 ms.

Frozen at the **first training window**, because with the parameters fixed each
of these is a forward recursion whose value at *t* depends on ``[0, t]`` alone:

Kalman variances and FFD ``d`` for the price path
    Frozen exactly as ``FrozenFeatureParams`` is frozen in production — a
    monthly refit persists them and the daily run reads them back. Freezing once
    at the start is *more* conservative than production's monthly refit, not
    less.
the HMM fit
    Same argument: production refits monthly and infers daily. One fit at the
    start of the out-of-sample path has strictly less information than that.
Kalman variances for the Ω path
    Without this, Ω̇ at date *t* would depend on the MLE having seen the whole
    Ω path. With it, the filter is a pure recursion.
ridge and logistic regularization strengths
    Chosen on an inner expanding split of the **first** training window and
    frozen. Re-selecting per window is more faithful and costs a factor of
    fifteen; the design note records the trade and the report states it.

Computed once on the stitched path, because they are causal by construction:
the coupling's expanding OLS accumulates with ``np.cumsum`` and the curvature's
z-scores and percentile are ``expanding()``, so row *t* of either is a function
of rows ``0..t``. The harness truncation test is what checks that claim rather
than trusting it.

The arms
--------

Identical in everything but the feature set. ``P`` enters as ``ffd_price``
rather than as the filtered level: the level is I(1) and a ridge of a forward
return on it is a spurious regression by construction, and the fractionally
differenced price is the stationary representation ``FINDYN_V1_SPEC.md`` §8.2
builds for exactly this purpose.

**A′ is mandatory.** Without it, "Ω adds information over price alone" is a
claim about a strawman — ``rii.py`` already publishes a composite built from
nearly the same observables. Any incremental result that survives A but not A′
is reported as *no incremental information over what the engine already
publishes*.

Ridge and logistic rather than a gradient-boosted tree: the arms differ by one
to five columns, and a high-variance learner would make the comparison a study
of its own tuning rather than of the columns.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from findynamics.backtest.replay import world_at
from findynamics.core.config import SeriesConfig, get_series_config
from findynamics.engines.equity import rii as rii_mod
from findynamics.engines.equity.crash import ADVERSE
from findynamics.engines.equity.features.kalman import KalmanParams, fit_params
from findynamics.engines.equity.features.pipeline import FrozenFeatureParams, compute_features
from findynamics.engines.equity.prices import PERIODS_PER_YEAR, PriceSeries, price_path
from findynamics.engines.equity.regime.design import build_design
from findynamics.engines.equity.regime.hmm import RegimeModel, fit_hmm
from findynamics.research.omega import targets as targets_mod
from findynamics.research.omega.config import OmegaConfig
from findynamics.research.omega.contracts import OmegaSpec
from findynamics.research.omega.coupling import CouplingParams, all_couplings, primary_coupling
from findynamics.research.omega.curvature import CurvatureParams, financial_curvature
from findynamics.research.omega.diagnostics import RII_SERIES
from findynamics.research.omega.dynamics import omega_dynamics
from findynamics.research.omega.estimator import DEFAULT_SEED, build_estimator
from findynamics.research.omega.features import (
    FeatureParams,
    available_columns,
    build_feature_frame,
)
from findynamics.research.omega.statistics import auc

log = logging.getLogger("findynamics.research.omega.backtest")

#: Arm -> its feature columns. ``A`` is the baseline every other arm extends.
BASELINE: tuple[str, ...] = ("ffd_price", "velocity", "acceleration", "jerk_z")

#: The RII and its seven components, as the engine publishes them. Arm A′ gets
#: the composite *and* its parts, because "Ω beats the published composite" and
#: "Ω beats everything that went into it" are different claims and the second is
#: the harder one.
RII_COLUMNS: tuple[str, ...] = (
    "rii",
    "rii_posterior_entropy",
    "rii_confidence_deficit",
    "rii_jerk",
    "rii_vol_of_vol",
    "rii_correlation_breakdown",
    "rii_credit_velocity",
    "rii_liquidity_stress",
)

ARMS: dict[str, tuple[str, ...]] = {
    "A": BASELINE,
    "A_prime": BASELINE + RII_COLUMNS,
    "B": BASELINE + ("omega",),
    "C": BASELINE + ("omega", "omega_velocity", "omega_acceleration"),
    "D": BASELINE + ("coupling",),
    "E": BASELINE + ("omega", "omega_velocity", "omega_acceleration", "coupling", "curvature"),
}

#: Columns the panel carries that are never arm features — targets are built
#: from ``log_price`` and ``regime``, and ``window`` records provenance.
PANEL_METADATA: tuple[str, ...] = ("log_price", "regime", "window")

#: Continuous targets, by name. The binary one is handled separately because it
#: needs a classifier and a different score.
CONTINUOUS_TARGETS: tuple[str, ...] = ("forward_return", "forward_vol")


class OmegaBacktestError(ValueError):
    """Raised when the walk-forward cannot run as configured."""


@dataclass(frozen=True)
class WalkForwardParams:
    """The walk-forward half of ``omega.yaml``."""

    cadence_months: int = 1
    min_train_years: float = 5.0
    horizons: tuple[int, ...] = (1, 5, 21, 63)
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    logistic_c: tuple[float, ...] = (0.01, 0.1, 1.0)
    inner_splits: int = 3
    cost_bps: float = 5.0
    sub_periods: tuple[tuple[str, date, date], ...] = (
        ("2005-2009", date(2005, 1, 1), date(2009, 12, 31)),
        ("2010-2014", date(2010, 1, 1), date(2014, 12, 31)),
        ("2015-2019", date(2015, 1, 1), date(2019, 12, 31)),
        ("2020-2024", date(2020, 1, 1), date(2024, 12, 31)),
        ("2025-", date(2025, 1, 1), date(2100, 12, 31)),
    )
    shuffle_seed: int = DEFAULT_SEED
    auc_ceiling: float = 0.55
    #: Multiples of Ω's own trailing standard deviation a refit boundary may
    #: move it by before the window-boundary test calls it a discontinuity.
    boundary_jump_sigma: float = 8.0
    #: Trailing window for that standard deviation, in observations.
    boundary_window: int = 63

    @classmethod
    def from_config(cls, config: OmegaConfig) -> WalkForwardParams:
        block = config.block("walk_forward")
        defaults = cls()

        cadence = int(block.get("cadence_months", defaults.cadence_months))
        if cadence < 1:
            raise OmegaBacktestError(f"walk_forward.cadence_months must be >= 1, got {cadence}")
        min_train = float(block.get("min_train_years", defaults.min_train_years))
        if min_train < 0.0:
            raise OmegaBacktestError(f"walk_forward.min_train_years must be >= 0, got {min_train}")
        horizons = tuple(int(h) for h in (block.get("horizons") or defaults.horizons))
        if not horizons or any(h < 1 for h in horizons):
            raise OmegaBacktestError(f"walk_forward.horizons must all be >= 1, got {horizons}")
        cost = float(block.get("cost_bps", defaults.cost_bps))
        if cost < 0.0:
            raise OmegaBacktestError(f"walk_forward.cost_bps must be >= 0, got {cost}")

        raw_periods = block.get("sub_periods")
        if raw_periods:
            sub_periods = tuple(
                (
                    str(entry["name"]),
                    date.fromisoformat(entry["start"]),
                    date.fromisoformat(entry["end"]),
                )
                for entry in raw_periods
            )
        else:
            sub_periods = defaults.sub_periods

        return cls(
            cadence_months=cadence,
            min_train_years=min_train,
            horizons=horizons,
            ridge_alphas=tuple(
                float(a) for a in (block.get("ridge_alphas") or defaults.ridge_alphas)
            ),
            logistic_c=tuple(float(c) for c in (block.get("logistic_c") or defaults.logistic_c)),
            inner_splits=int(block.get("inner_splits", defaults.inner_splits)),
            cost_bps=cost,
            sub_periods=sub_periods,
            shuffle_seed=int(block.get("shuffle_seed", defaults.shuffle_seed)),
            auc_ceiling=float(block.get("auc_ceiling", defaults.auc_ceiling)),
            boundary_jump_sigma=float(
                block.get("boundary_jump_sigma", defaults.boundary_jump_sigma)
            ),
            boundary_window=int(block.get("boundary_window", defaults.boundary_window)),
        )


@dataclass(frozen=True)
class WindowRecord:
    """What one rebalance froze, and which dates it went on to explain."""

    rebalance: date
    spec: OmegaSpec
    n_train: int
    block_start: date | None
    block_end: date | None

    def as_dict(self) -> dict[str, object]:
        return {
            "rebalance": self.rebalance.isoformat(),
            "n_train": self.n_train,
            "block_start": None if self.block_start is None else self.block_start.isoformat(),
            "block_end": None if self.block_end is None else self.block_end.isoformat(),
            "spec": self.spec.as_dict(),
        }


@dataclass(frozen=True)
class WalkForwardResult:
    """The stitched out-of-sample path and everything computed on it."""

    #: One row per out-of-sample date: arm features, targets, metadata.
    panel: pd.DataFrame
    #: One column per ``(arm, target, horizon)``, holding the prediction made
    #: from a model that had not seen that row.
    predictions: pd.DataFrame
    #: The mean of each target over each window's training data — what
    #: :func:`statistics.oos_r2` measures against.
    training_means: pd.DataFrame
    #: Arm -> the feature columns availability dropped from it. Reported rather
    #: than implied: an arm quietly missing a column is a different arm.
    dropped_columns: dict[str, tuple[str, ...]]
    windows: tuple[WindowRecord, ...]
    params: WalkForwardParams
    config_hash: str
    model_version: str
    diagnostics: dict[str, float] = field(default_factory=dict)

    def target_name(self, kind: str, horizon: int) -> str:
        return f"{kind}_{horizon}"

    def arm_columns(self, arm: str) -> tuple[str, ...]:
        """The columns the arm was actually fitted on, after availability."""
        dropped = set(self.dropped_columns.get(arm, ()))
        return tuple(c for c in ARMS[arm] if c in self.panel.columns and c not in dropped)


def config_hash(config: OmegaConfig) -> str:
    """A stable digest of the parameters a run used.

    ``sort_keys`` so two dicts that differ only in insertion order hash the
    same, and a digest rather than the blob so it fits in a CSV column. Recorded
    per run because a stored artifact that cannot say which configuration
    produced it is not reproducible.
    """
    payload = json.dumps(
        {"enabled": config.enabled, "experimental": config.experimental, **config.params},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def rebalance_dates(
    index: pd.DatetimeIndex,
    params: WalkForwardParams,
) -> list[pd.Timestamp]:
    """Month-end trading dates, warm-up discarded.

    The first ``min_train_years`` are dropped **entirely** rather than run with
    a short window: an Ω fitted on 200 observations is a statement about its own
    start-up, exactly as ``kalman_burn_in_years`` is for velocity.
    """
    if index.empty:
        return []
    month_ends = (
        pd.Series(index, index=index).groupby([index.year, index.month]).last().sort_values()
    )
    selected = list(month_ends.to_numpy())[:: params.cadence_months]
    cutoff = index[0] + pd.DateOffset(days=int(round(params.min_train_years * 365.25)))
    return [pd.Timestamp(d) for d in selected if pd.Timestamp(d) >= cutoff]


def _price_series(series_id: str, frequency: str) -> PriceSeries:
    """The research price path, described the way the equity feature path wants.

    ``role="publication"`` is a label on a dataclass, not a registration. The
    equity engine's own ``resolve_from`` cannot be used here: it requires
    ``FRED:SP500``, which begins in 2016 and therefore raises at every
    rebalance date before then.
    """
    return PriceSeries(
        role="publication",
        source_role="backfill",
        series_id=series_id,
        frequency=frequency,
        observations=0,
    )


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge with an unpenalized intercept.

    Written out rather than taken from scikit-learn so the answer is the normal
    equations and not a solver's stopping rule — the harness truncation test
    compares coefficients bit-for-bit.
    """
    centre = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    z = (x - centre) / scale
    gram = z.T @ z + alpha * np.eye(z.shape[1])
    beta = np.linalg.solve(gram, z.T @ (y - y.mean()))
    return np.concatenate([[y.mean() - float(beta @ (centre / scale))], beta / scale])


def _predict(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return coefficients[0] + x @ coefficients[1:]


def _select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    params: WalkForwardParams,
) -> float:
    """Pick a ridge strength on an inner **expanding** split of the training data.

    Expanding, not k-fold: a random fold would train on data after the rows it
    scores, which is the leak this whole module is built to avoid — inside the
    model selection, where it is least visible.
    """
    n = len(y)
    if n < 4 * params.inner_splits:
        return float(params.ridge_alphas[len(params.ridge_alphas) // 2])

    edges = [int(n * (i + 1) / (params.inner_splits + 1)) for i in range(params.inner_splits)]
    scores: dict[float, list[float]] = {alpha: [] for alpha in params.ridge_alphas}
    for edge in edges:
        width = max(n // (params.inner_splits + 1), 1)
        train_x, train_y = x[:edge], y[:edge]
        test_x, test_y = x[edge : edge + width], y[edge : edge + width]
        if len(test_y) < 5 or train_x.shape[0] <= train_x.shape[1] + 2:
            continue
        baseline = math.fsum((test_y - train_y.mean()) ** 2)
        if baseline <= 0.0:
            continue
        for alpha in params.ridge_alphas:
            residual = test_y - _predict(test_x, _ridge(train_x, train_y, alpha))
            scores[alpha].append(1.0 - math.fsum(residual**2) / baseline)

    usable = {alpha: values for alpha, values in scores.items() if values}
    if not usable:
        return float(params.ridge_alphas[len(params.ridge_alphas) // 2])
    # max() over a dict comprehension would depend on insertion order at a tie;
    # sorting by (-score, alpha) makes the choice total and reproducible.
    ranked = sorted(
        ((float(np.mean(values)), alpha) for alpha, values in usable.items()),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return float(ranked[0][1])


def _select_logistic_c(x: np.ndarray, y: np.ndarray, params: WalkForwardParams) -> float:
    """The same expanding-split selection for the classifier."""
    n = len(y)
    middle = float(params.logistic_c[len(params.logistic_c) // 2])
    if n < 4 * params.inner_splits or len(np.unique(y)) < 2:
        return middle

    edges = [int(n * (i + 1) / (params.inner_splits + 1)) for i in range(params.inner_splits)]
    scores: dict[float, list[float]] = {value: [] for value in params.logistic_c}
    width = max(n // (params.inner_splits + 1), 1)
    for edge in edges:
        train_x, train_y = x[:edge], y[:edge]
        test_x, test_y = x[edge : edge + width], y[edge : edge + width]
        if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
            continue
        for value in params.logistic_c:
            model = _fit_logistic(train_x, train_y, value)
            if model is None:
                continue
            probability = model.predict_proba(test_x)[:, 1]
            score = auc(pd.Series(test_y), pd.Series(probability))
            if np.isfinite(score):
                scores[value].append(score)

    usable = {value: got for value, got in scores.items() if got}
    if not usable:
        return middle
    ranked = sorted(
        ((float(np.mean(got)), value) for value, got in usable.items()),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return float(ranked[0][1])


def _fit_logistic(x: np.ndarray, y: np.ndarray, c: float) -> LogisticRegression | None:
    """A pinned, deterministic logistic fit, or ``None`` on a degenerate sample."""
    if len(np.unique(y)) < 2:
        return None
    centre, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    model = LogisticRegression(
        C=c,
        max_iter=1_000,
        solver="lbfgs",
        random_state=DEFAULT_SEED,
    )
    with threadpool_limits(limits=1):
        model.fit((x - centre) / scale, y)
    model._omega_centre = centre  # noqa: SLF001 - carried so predict can reuse it
    model._omega_scale = scale  # noqa: SLF001
    return model


def _logistic_probability(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    centre = getattr(model, "_omega_centre", 0.0)
    scale = getattr(model, "_omega_scale", 1.0)
    return model.predict_proba((x - centre) / scale)[:, 1]


@dataclass(frozen=True)
class _Frozen:
    """What the first training window froze, and the identity of that window."""

    features: FrozenFeatureParams
    regime: RegimeModel
    omega_kalman: KalmanParams
    series: PriceSeries
    fitted_through: date


def _freeze(
    observations: pd.DataFrame,
    cutoff: pd.Timestamp,
    config: OmegaConfig,
    series_config: SeriesConfig,
    feature_params: FeatureParams,
) -> _Frozen:
    """Fit the reusable machinery on the first training window and freeze it.

    Everything frozen here becomes a forward recursion afterwards, which is what
    makes reading it at date *t* safe. See the module docstring for the split.
    """
    accessor = world_at(
        observations, cutoff.date(), config=series_config, with_factors=False
    ).series
    series = _price_series(feature_params.series["price"], feature_params.frequency)
    path = price_path(accessor, series)
    if path.empty:
        raise OmegaBacktestError(
            f"{series.series_id}: no knowable closes at {cutoff.date()}; nothing to freeze"
        )

    reference = compute_features(path, series)
    regime = RegimeModel(fit_hmm(build_design(reference)))

    estimator = build_estimator(config)
    block = build_feature_frame(accessor, config)
    spec = estimator.fit(block)
    # The variances are estimated on this window's own Ω — in-sample for the
    # *training* window, which is where a frozen parameter is supposed to come
    # from — and every later Ω̇ is then a pure forward recursion.
    omega = estimator.transform(block, spec).dropna()
    kalman, _ = fit_params(omega, label="omega-walk-forward")

    log.info(
        "omega walk-forward: froze feature params, the regime model and Ω's filter "
        "variances on %s (%d price rows, %d Ω rows)",
        cutoff.date(),
        len(path),
        len(omega),
    )
    return _Frozen(
        features=reference.frozen(),
        regime=regime,
        omega_kalman=kalman,
        series=series,
        fitted_through=cutoff.date(),
    )


def _reference_block(
    accessor,
    frozen: _Frozen,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """The engine's published quantities at one cutoff, on frozen parameters.

    Returns ``(features_frame, log_price, regime_labels)`` plus the RII columns
    folded into the frame. Cheap — 30 ms against 4.7 seconds unfrozen — which is
    the whole reason the walk-forward is affordable.
    """
    path = price_path(accessor, frozen.series)
    features = compute_features(path, frozen.series, frozen=frozen.features)
    design = build_design(features)
    posteriors = frozen.regime.posteriors(design)
    labels = frozen.regime.states(design)

    wide = accessor.wide(list(RII_SERIES))
    try:
        rii = rii_mod.compute_rii(
            posteriors,
            jerk_z=features.frame.get("jerk_z"),
            realized_vol=design.realized_vol,
            equity_returns=features.log_price.diff(),
            bond_yield=wide.get("FRED:DGS10"),
            credit_spread=wide.get("FRED:BAMLH0A0HYM2"),
            liquidity=wide.get("FRED:NFCI"),
            periods_per_year=frozen.series.periods_per_year,
        )
        # Forward-filled onto the daily calendar, exactly as
        # `engines/equity/engine.py` does when it publishes them. Without this,
        # `liquidity_stress` exists only on NFCI's weekly observation dates and
        # `correlation_breakdown` only where both legs traded, so arm A′ — whose
        # whole job is to be a fair control — was fitted on 19 rows out of 5,407.
        # The ffill carries a released value forward and never backwards.
        rii_columns = {"rii": rii.index.ffill()}
        rii_columns.update(
            {f"rii_{name}": series.ffill() for name, series in rii.components.items()}
        )
    except ValueError as err:
        # A window too early for any RII component still produces price
        # kinematics; arm A′ simply has nothing to add there and the panel says
        # so with NaN rather than with a zero.
        log.info("omega walk-forward: no RII at this cutoff (%s)", err)
        rii_columns = {}

    frame = features.frame[["ffd_price", "velocity", "acceleration", "jerk_z"]].copy()
    for name, series in rii_columns.items():
        frame[name] = series.reindex(frame.index)
    return frame, features.log_price, labels


def walk_forward(
    observations: pd.DataFrame,
    config: OmegaConfig,
    *,
    params: WalkForwardParams | None = None,
    series_config: SeriesConfig | None = None,
    shuffle_targets: bool = False,
) -> WalkForwardResult:
    """The stitched out-of-sample path, one spec per window, one row per date.

    ``shuffle_targets`` permutes every target under the configured seed and is
    the acceptance gate of this phase: if any arm shows skill against a permuted
    target, the harness is leaking and every other number here is void.
    """
    resolved = params or WalkForwardParams.from_config(config)
    series_config = series_config or get_series_config()
    feature_params = FeatureParams.from_config(config)
    estimator = build_estimator(config)

    calendar = _calendar(observations, feature_params)
    dates = rebalance_dates(calendar, resolved)
    if len(dates) < 3:
        raise OmegaBacktestError(
            f"{len(dates)} rebalance date(s) after discarding {resolved.min_train_years:g} "
            "year(s) of warm-up; a walk-forward needs at least three"
        )

    frozen = _freeze(observations, dates[0], config, series_config, feature_params)

    blocks: list[pd.DataFrame] = []
    records: list[WindowRecord] = []
    previous_spec: OmegaSpec | None = None
    previous_date: pd.Timestamp | None = None

    for cutoff in dates:
        accessor = world_at(
            observations, cutoff.date(), config=series_config, with_factors=False
        ).series
        try:
            z = build_feature_frame(accessor, config)
            spec = estimator.fit(z)
        except ValueError as err:
            log.info("omega walk-forward: no Ω spec at %s (%s)", cutoff.date(), err)
            previous_date = cutoff
            continue

        if previous_spec is not None and previous_date is not None:
            window = z.loc[(z.index > previous_date) & (z.index <= cutoff)]
            if not window.empty:
                reference, log_price, labels = _reference_block(accessor, frozen)
                block = reference.reindex(window.index)
                block["omega"] = estimator.transform(window, previous_spec)
                block["log_price"] = log_price.reindex(window.index)
                block["regime"] = labels.reindex(window.index)
                block["window"] = len(records)
                blocks.append(block)
                records.append(
                    WindowRecord(
                        rebalance=previous_date.date(),
                        spec=previous_spec,
                        n_train=previous_spec.n_observations,
                        block_start=window.index[0].date(),
                        block_end=window.index[-1].date(),
                    )
                )
        previous_spec, previous_date = spec, cutoff

    if not blocks:
        raise OmegaBacktestError("the walk-forward produced no out-of-sample block")

    panel = pd.concat(blocks).sort_index()
    panel = panel[~panel.index.duplicated(keep="first")]
    panel.index.name = "obs_date"

    panel = _derive(panel, config, frozen, resolved, previous_spec)
    panel = _attach_targets(panel, resolved, feature_params, shuffle_targets)

    predictions, training_means, dropped_by_arm = _fit_arms(
        panel, records, resolved, shuffle_targets, feature_params
    )

    diagnostics = {
        "windows": float(len(records)),
        "out_of_sample_rows": float(len(panel)),
        "shuffled": float(shuffle_targets),
        "frozen_through": float(frozen.fitted_through.toordinal()),
    }
    log.info(
        "omega walk-forward: %d window(s), %d out-of-sample row(s) %s → %s%s",
        len(records),
        len(panel),
        panel.index[0].date(),
        panel.index[-1].date(),
        " (SHUFFLED TARGETS — control run)" if shuffle_targets else "",
    )
    return WalkForwardResult(
        panel=panel,
        predictions=predictions,
        training_means=training_means,
        dropped_columns=dropped_by_arm,
        windows=tuple(records),
        params=resolved,
        config_hash=config_hash(config),
        model_version=records[-1].spec.model_version,
        diagnostics=diagnostics,
    )


def _calendar(observations: pd.DataFrame, feature_params: FeatureParams) -> pd.DatetimeIndex:
    """Trading dates the research window covers, from the price series alone."""
    price_id = feature_params.series["price"]
    rows = observations[observations["series_id"] == price_id]
    if rows.empty:
        raise OmegaBacktestError(f"{price_id}: absent from the observation frame")
    dates = pd.DatetimeIndex(sorted(rows["obs_date"].unique()))
    if feature_params.start is not None:
        dates = dates[dates >= pd.Timestamp(feature_params.start)]
    return dates


def _derive(
    panel: pd.DataFrame,
    config: OmegaConfig,
    frozen: _Frozen,
    params: WalkForwardParams,
    spec: OmegaSpec,
) -> pd.DataFrame:
    """Ω̇, Ω̈, ``C`` and ``K`` from the stitched path.

    Computed once rather than per window because each is causal by construction
    — the filter runs on frozen variances, the coupling accumulates with
    ``np.cumsum`` and the curvature's z-scores are ``expanding()`` — so row *t*
    of every one of them is a function of rows ``0..t``. The harness truncation
    test is what checks that claim rather than trusting it.
    """
    path = omega_dynamics(
        panel["omega"],
        spec,
        periods_per_year=PERIODS_PER_YEAR[frozen.series.frequency],
        config=config,
        kalman_params=frozen.omega_kalman,
    )
    out = panel.copy()
    out["omega_velocity"] = path.omega_velocity
    out["omega_acceleration"] = path.omega_acceleration
    out["omega_zscore"] = path.omega_zscore
    out["omega_regime"] = path.omega_regime

    coupling_params = CouplingParams.from_config(config)
    couplings = all_couplings(
        out["velocity"],
        out["acceleration"],
        out["omega"],
        out["omega_velocity"],
        out["omega_acceleration"],
        params=coupling_params,
        periods_per_year=PERIODS_PER_YEAR[frozen.series.frequency],
    )
    out["coupling"] = primary_coupling(couplings, coupling_params).reindex(out.index)

    curvature = financial_curvature(
        {
            "acceleration": out["acceleration"],
            "jerk": out["jerk_z"],
            "omega_velocity": out["omega_velocity"],
            "omega_acceleration": out["omega_acceleration"],
            "coupling": out["coupling"],
        },
        params=CurvatureParams.from_config(config),
        periods_per_year=PERIODS_PER_YEAR[frozen.series.frequency],
    )
    out["curvature"] = curvature.curvature.reindex(out.index)
    return out


def _attach_targets(
    panel: pd.DataFrame,
    params: WalkForwardParams,
    feature_params: FeatureParams,
    shuffle: bool,
) -> pd.DataFrame:
    """Forward return, forward volatility and the transition label, per horizon.

    Shuffling permutes the *values* and keeps the index, so the control run has
    the same missingness pattern and the same sample sizes as the real one. A
    control that also changed the sample would not be a control.
    """
    out = panel.copy()
    log_price = panel["log_price"]
    rng = np.random.default_rng(params.shuffle_seed)

    for horizon in params.horizons:
        columns = {
            f"forward_return_{horizon}": targets_mod.forward_return(log_price, horizon),
            f"forward_vol_{horizon}": targets_mod.forward_volatility(
                log_price, horizon, periods_per_year=feature_params.periods_per_year
            ),
            f"forward_drawdown_{horizon}": targets_mod.forward_drawdown(log_price, horizon),
            f"transition_{horizon}": targets_mod.transition_within(
                panel["regime"], horizon, ADVERSE
            ),
        }
        for name, values in columns.items():
            if shuffle:
                finite = values.dropna()
                permuted = pd.Series(
                    rng.permutation(finite.to_numpy()), index=finite.index, name=name
                )
                values = permuted.reindex(values.index)
            out[name] = values
    return out


def arm_columns(
    panel: pd.DataFrame,
    arm: str,
    min_coverage: float,
    min_observations: int,
) -> tuple[list[str], tuple[str, ...]]:
    """One arm's usable feature columns, and the ones availability dropped.

    The same check Ω's own feature block gets, applied to the arms — and it is
    not decorative. ``rii_credit_velocity`` inherits ``FRED:BAMLH0A0HYM2``'s
    2023 start, so an arm that required it would be fitted on the last three
    years of a twenty-one-year path and every comparison against it would be a
    comparison of sample sizes.

    A dropped column is **named**, never imputed. Arm A′ with six of eight RII
    columns is still the control the design note asks for; arm A′ silently
    truncated to 2023 is not.
    """
    present = [c for c in ARMS[arm] if c in panel.columns]
    if not present:
        return [], ()
    surviving, dropped = available_columns(panel[present], min_observations, min_coverage)
    # `available_columns` sorts; the arm's own order is what the report prints.
    ordered = [c for c in present if c in surviving]
    return ordered, dropped


def _fit_arms(
    panel: pd.DataFrame,
    records: list[WindowRecord],
    params: WalkForwardParams,
    shuffle: bool,
    feature_params: FeatureParams,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[str, ...]]]:
    """Fit every arm on each window's training data and predict its block.

    Regularization is selected on an inner expanding split of the **first**
    window that can support one, then frozen — re-selecting per window is more
    faithful and costs a factor of fifteen, and the design note records the
    trade. Every fit after that sees only its own training rows.
    """
    predictions = pd.DataFrame(index=panel.index, dtype=float)
    means = pd.DataFrame(index=panel.index, dtype=float)
    alphas: dict[tuple[str, str], float] = {}
    dropped_by_arm: dict[str, tuple[str, ...]] = {}

    ordered = sorted(records, key=lambda record: record.rebalance)
    for arm in ARMS:
        usable, dropped = arm_columns(
            panel, arm, feature_params.min_column_coverage, feature_params.min_column_observations
        )
        dropped_by_arm[arm] = dropped
        if dropped:
            log.info(
                "omega walk-forward: arm %s drops %s on availability; %d column(s) remain",
                arm,
                ", ".join(dropped),
                len(usable),
            )
        if not usable:
            continue
        for horizon in params.horizons:
            for kind in CONTINUOUS_TARGETS:
                name = f"{kind}_{horizon}"
                if name not in panel.columns:
                    continue
                _fit_continuous_arm(
                    panel, ordered, usable, arm, name, horizon, params, alphas, predictions, means
                )
            _fit_binary_arm(
                panel, ordered, usable, arm, f"transition_{horizon}", horizon, params, predictions
            )
    log.info(
        "omega walk-forward: fitted %d arm/target column(s)%s",
        len(predictions.columns),
        " on permuted targets" if shuffle else "",
    )
    return predictions, means, dropped_by_arm


def _training_slice(
    panel: pd.DataFrame,
    columns: list[str],
    target: str,
    cutoff: pd.Timestamp,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex] | None:
    """Rows dated ``<= cutoff`` whose target had already been realized there."""
    realized = targets_mod.realized_by(panel[target], cutoff, horizon)
    if realized.empty:
        return None
    frame = panel.loc[realized.index, columns].dropna()
    if len(frame) < max(50, 5 * len(columns)):
        return None
    aligned = realized.reindex(frame.index)
    return frame.to_numpy(dtype=float), aligned.to_numpy(dtype=float), frame.index


def _fit_continuous_arm(
    panel: pd.DataFrame,
    records: list[WindowRecord],
    columns: list[str],
    arm: str,
    target: str,
    horizon: int,
    params: WalkForwardParams,
    alphas: dict[tuple[str, str], float],
    predictions: pd.DataFrame,
    means: pd.DataFrame,
) -> None:
    column = f"{arm}::{target}"
    predictions[column] = np.nan
    means[column] = np.nan

    for record in records:
        if record.block_start is None or record.block_end is None:
            continue
        cutoff = pd.Timestamp(record.rebalance)
        training = _training_slice(panel, columns, target, cutoff, horizon)
        if training is None:
            continue
        x, y, _ = training

        key = (arm, target)
        if key not in alphas:
            alphas[key] = _select_alpha(x, y, params)
        coefficients = _ridge(x, y, alphas[key])

        block = panel.loc[
            pd.Timestamp(record.block_start) : pd.Timestamp(record.block_end), columns
        ].dropna()
        if block.empty:
            continue
        predictions.loc[block.index, column] = _predict(block.to_numpy(dtype=float), coefficients)
        # The training mean travels with the prediction: OOS R² is measured
        # against it, and against the test-window mean it would be lookahead.
        means.loc[block.index, column] = float(y.mean())


def _fit_binary_arm(
    panel: pd.DataFrame,
    records: list[WindowRecord],
    columns: list[str],
    arm: str,
    target: str,
    horizon: int,
    params: WalkForwardParams,
    predictions: pd.DataFrame,
) -> None:
    if target not in panel.columns:
        return
    column = f"{arm}::{target}"
    predictions[column] = np.nan
    chosen: float | None = None

    for record in records:
        if record.block_start is None or record.block_end is None:
            continue
        training = _training_slice(panel, columns, target, pd.Timestamp(record.rebalance), horizon)
        if training is None:
            continue
        x, y, _ = training
        if chosen is None:
            chosen = _select_logistic_c(x, y, params)
        model = _fit_logistic(x, y, chosen)
        if model is None:
            continue
        block = panel.loc[
            pd.Timestamp(record.block_start) : pd.Timestamp(record.block_end), columns
        ].dropna()
        if block.empty:
            continue
        predictions.loc[block.index, column] = _logistic_probability(
            model, block.to_numpy(dtype=float)
        )


def shuffled_target_control(
    observations: pd.DataFrame,
    config: OmegaConfig,
    *,
    params: WalkForwardParams | None = None,
    series_config: SeriesConfig | None = None,
) -> WalkForwardResult:
    """The acceptance gate: the same pipeline against permuted targets.

    If any arm shows skill here, the harness is leaking and every other number
    in this phase is void — no partial credit, no "but the real result was
    stronger". The permutation keeps the index and the missingness, so the
    control run has the same sample sizes as the real one and a difference
    cannot be explained by one having more data.
    """
    return walk_forward(
        observations,
        config,
        params=params,
        series_config=series_config,
        shuffle_targets=True,
    )


__all__ = [
    "ARMS",
    "BASELINE",
    "CONTINUOUS_TARGETS",
    "PANEL_METADATA",
    "RII_COLUMNS",
    "OmegaBacktestError",
    "WalkForwardParams",
    "WalkForwardResult",
    "WindowRecord",
    "arm_columns",
    "config_hash",
    "rebalance_dates",
    "shuffled_target_control",
    "walk_forward",
]
