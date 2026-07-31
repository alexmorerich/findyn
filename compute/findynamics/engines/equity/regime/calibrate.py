"""P(transition into bear or crisis within h) — §9 layer L3.

The HMM says where the market is. This says where it is likely to go, at 3, 6
and 12 months, as a **calibrated** probability: when the model says 20%, it
should happen about one time in five. A raw gradient-boosted score is a ranking,
not a probability — it separates good cases from bad ones well and is
systematically overconfident about both — so an isotonic regression is fitted on
held-out folds and applied on top.

Why the cross-validation is not ordinary
----------------------------------------

The label at date *t* is "does the regime become bear or crisis anywhere in the
next h months". Two dates a week apart therefore share almost all of their label
horizon, and a random K-fold would put one in train and the other in test and
score the model on a question it had already been shown the answer to. The
result is a Brier score that looks excellent and a model that is worthless live.

López de Prado's two corrections, both implemented in :func:`purged_folds`:

* **Purging** removes training observations whose label horizon overlaps the
  test fold at all.
* **An embargo** removes a further six months *after* the test fold, because
  features are built from trailing windows and a training point just past the
  fold boundary still carries the fold's data inside its own moving averages.

Together they cost roughly a fifth of the sample and they are not optional; the
same code without them reports a materially better score for a materially worse
model.

The classifier is fitted on the **calibration** series, like the HMM, and for
the same reason: 3-month transition labels over a ten-year window contain a
handful of genuine positive episodes, which is not a training set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from findynamics.engines.equity.regime.design import RegimeDesign

log = logging.getLogger("findynamics.engines.equity.regime.calibrate")

#: §9 L3 — the horizons a transition probability is published for, in months.
HORIZON_MONTHS: tuple[int, ...] = (3, 6, 12)

#: The states a transition *into* is worth warning about.
ADVERSE_REGIMES: frozenset[str] = frozenset({"bear", "crisis"})

#: §14.1 rule 4 — six months of embargo after every test fold.
DEFAULT_EMBARGO_MONTHS = 6

DEFAULT_FOLDS = 5

#: Deliberately shallow. The feature count is small, the effective sample size is
#: far below the row count (adjacent days are nearly the same observation), and
#: the output is a probability that has to stay calibrated — none of which
#: rewards depth. Fixed rather than tuned, because tuning against a purged CV
#: score on ~190 independent regime episodes overfits the tuner.
DEFAULT_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_lambda": 2.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
}

DEFAULT_SEED = 20260731


class CalibrationError(ValueError):
    """Raised when a usable calibrated classifier cannot be produced."""


@dataclass(frozen=True)
class FoldReport:
    """One purged fold's out-of-sample performance."""

    fold: int
    train_rows: int
    test_rows: int
    positives: int
    brier: float
    auc: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "positives": self.positives,
            "brier": self.brier,
            "auc": self.auc,
        }


@dataclass(frozen=True)
class TransitionModel:
    """A fitted, calibrated classifier for one horizon."""

    horizon_months: int
    feature_names: tuple[str, ...]
    #: The booster, serialized. XGBoost's own JSON round-trip, not pickle.
    booster_json: str
    #: Isotonic calibration knots, fitted on out-of-fold predictions only.
    calibration_x: tuple[float, ...]
    calibration_y: tuple[float, ...]
    fitted_on: str
    fitted_on_proxy: bool
    observations: int
    positives: int
    base_rate: float
    folds: tuple[FoldReport, ...]
    #: Out-of-fold Brier score of the *calibrated* probability. The honest one.
    brier: float
    auc: float | None
    seed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon_months": self.horizon_months,
            "feature_names": list(self.feature_names),
            "booster_json": self.booster_json,
            "calibration_x": list(self.calibration_x),
            "calibration_y": list(self.calibration_y),
            "fitted_on": self.fitted_on,
            "fitted_on_proxy": self.fitted_on_proxy,
            "observations": self.observations,
            "positives": self.positives,
            "base_rate": self.base_rate,
            "folds": [f.as_dict() for f in self.folds],
            "brier": self.brier,
            "auc": self.auc,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TransitionModel:
        return cls(
            horizon_months=int(raw["horizon_months"]),
            feature_names=tuple(raw["feature_names"]),
            booster_json=str(raw["booster_json"]),
            calibration_x=tuple(float(v) for v in raw["calibration_x"]),
            calibration_y=tuple(float(v) for v in raw["calibration_y"]),
            fitted_on=str(raw["fitted_on"]),
            fitted_on_proxy=bool(raw.get("fitted_on_proxy", False)),
            observations=int(raw["observations"]),
            positives=int(raw["positives"]),
            base_rate=float(raw["base_rate"]),
            folds=tuple(
                FoldReport(
                    fold=int(f["fold"]),
                    train_rows=int(f["train_rows"]),
                    test_rows=int(f["test_rows"]),
                    positives=int(f["positives"]),
                    brier=float(f["brier"]),
                    auc=None if f.get("auc") is None else float(f["auc"]),
                )
                for f in raw.get("folds", [])
            ),
            brier=float(raw["brier"]),
            auc=None if raw.get("auc") is None else float(raw["auc"]),
            seed=int(raw.get("seed", DEFAULT_SEED)),
        )

    def _booster(self):
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(bytearray(self.booster_json, "utf-8"))
        return booster

    def raw_scores(self, features: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        ordered = features[list(self.feature_names)]
        return self._booster().predict(xgb.DMatrix(ordered.to_numpy(dtype=float)))

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Calibrated P(transition within the horizon), indexed like ``features``."""
        raw = self.raw_scores(features)
        calibrated = np.interp(raw, np.asarray(self.calibration_x), np.asarray(self.calibration_y))
        return pd.Series(
            np.clip(calibrated, 0.0, 1.0),
            index=features.index,
            name=f"p_transition_{self.horizon_months}m",
        )

    def contributions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Per-prediction SHAP values (§14.3), in log-odds units.

        Taken from the booster's own exact tree SHAP rather than the ``shap``
        explainer wrapper: it is the same algorithm, needs no background dataset,
        and returns in milliseconds for a single row — which is what the daily
        run actually asks for.

        These explain the *uncalibrated* score. Isotonic calibration is monotone,
        so it changes the number but never the sign or the ranking of a
        contribution, which is what an explanation is claiming.
        """
        import xgboost as xgb

        ordered = features[list(self.feature_names)]
        values = self._booster().predict(
            xgb.DMatrix(ordered.to_numpy(dtype=float)), pred_contribs=True
        )
        # The final column is the bias term, which is not a feature.
        return pd.DataFrame(values[:, :-1], index=features.index, columns=list(self.feature_names))


@dataclass(frozen=True)
class TransitionModels:
    """The three horizons together, with their shared provenance."""

    models: dict[int, TransitionModel] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {str(h): m.as_dict() for h, m in sorted(self.models.items())}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> TransitionModels:
        body = raw or {}
        return cls(models={int(h): TransitionModel.from_dict(m) for h, m in body.items()})

    def __bool__(self) -> bool:
        return bool(self.models)


def adverse_entries(
    states: pd.Series,
    adverse: frozenset[str] = ADVERSE_REGIMES,
) -> pd.Series:
    """1 on each date the regime *enters* bear or crisis from somewhere else.

    The distinction that makes the published probability mean anything. Labelling
    the mere *presence* of an adverse regime in the window gives base rates of
    60%, 73% and 90% at 3, 6 and 12 months on the calibration series — because
    with a regime change every eleven weeks, almost every year-long window
    touches a bad state at some point. "P(transition to bear/crisis) = 81%" would
    then be a *reassuring* number printed in a way that reads as alarming, which
    is worse than publishing nothing.

    An entry is an event. Its probability is a statement a reader can act on.
    """
    is_adverse = states.isin(adverse)
    return (is_adverse & ~is_adverse.shift(1, fill_value=False)).astype(int)


def transition_labels(
    states: pd.Series,
    horizon_months: int,
    periods_per_year: float,
    adverse: frozenset[str] = ADVERSE_REGIMES,
) -> pd.Series:
    """1 where the regime *enters* bear or crisis within the horizon, else 0.

    Forward-looking **by construction** — that is what a label is. Dates near the
    end of the sample whose horizon runs past the data are dropped rather than
    labelled 0, because "no transition was observed" and "the window has not
    happened yet" are different statements, and conflating them teaches the model
    that the present is always safe.

    The label says nothing about whether the market is *already* adverse; that is
    a feature (``p_adverse_now``), supplied by :func:`build_training_frame`. So
    the published number is "the probability of a fresh deterioration", and a
    market already in crisis can correctly show a low one — it has already
    transitioned, and the current regime is published beside this.
    """
    horizon = max(int(round(horizon_months * periods_per_year / 12.0)), 1)
    entries = adverse_entries(states, adverse).astype(float)

    # Reverse rolling max over the *next* `horizon` observations: shift(-1) first
    # so the window is strictly forward and today's own entry does not label
    # today as predicting itself.
    forward = entries.shift(-1)[::-1].rolling(window=horizon, min_periods=horizon).max()[::-1]
    return forward.dropna().astype(int).rename(f"transition_{horizon_months}m")


def purged_folds(
    index: pd.DatetimeIndex,
    horizon_months: int,
    *,
    n_folds: int = DEFAULT_FOLDS,
    embargo_months: int = DEFAULT_EMBARGO_MONTHS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous test folds with overlapping and embargoed training rows removed.

    Yields ``(train_positions, test_positions)``. Folds are contiguous blocks of
    time rather than random subsets, because the leakage being prevented is
    temporal and a shuffled fold has a boundary with every other fold.

    A training row is dropped when either is true:

    * its own label horizon reaches into the test block (**purging**), or
    * it sits within the embargo *after* the test block, where its trailing
      feature windows still contain test-block observations.
    """
    if n_folds < 2:
        raise CalibrationError(f"purged CV needs at least 2 folds, got {n_folds}")

    n = len(index)
    if n < n_folds * 50:
        raise CalibrationError(
            f"{n} rows cannot support {n_folds} purged folds; the embargo would "
            "consume the training set"
        )

    horizon = pd.DateOffset(months=horizon_months)
    embargo = pd.DateOffset(months=embargo_months)
    positions = np.arange(n)
    bounds = np.array_split(positions, n_folds)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for block in bounds:
        if block.size == 0:
            continue
        test_start, test_end = index[block[0]], index[block[-1]]

        # Purge: a training row's label covers [t, t + horizon].
        label_end = index + horizon
        overlaps = (label_end >= test_start) & (index <= test_end)
        # Embargo: the window after the fold whose features still see it.
        embargoed = (index > test_end) & (index <= test_end + embargo)

        train = positions[~(overlaps | embargoed)]
        folds.append((train, block))
    return folds


def build_training_frame(
    design: RegimeDesign,
    posteriors: pd.DataFrame,
    *,
    factor_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The L3 feature matrix: K(t), the HMM posterior, and F(t) where available.

    §9 puts the classifier on ``[K(t), F(t), RII]``. K(t) is the design matrix,
    F(t) is the shared factor scores, and the posterior stands in for RII until
    sub-milestone C builds it — the RII's largest component is posterior entropy,
    which is included here directly.

    Factors are optional because the calibration series runs from 1987 and most
    of the shared factors do not. Where they are absent the classifier is a
    price-only model and says so through its feature list rather than through a
    silently shorter training window.
    """
    entropy = -(posteriors.clip(lower=1e-12) * np.log(posteriors.clip(lower=1e-12))).sum(axis=1)

    frame = design.frame.join(
        pd.DataFrame(
            {
                "posterior_entropy": entropy,
                "posterior_max": posteriors.max(axis=1),
                "p_adverse_now": posteriors[
                    [c for c in posteriors.columns if c in ADVERSE_REGIMES]
                ].sum(axis=1),
            }
        ),
        how="inner",
    )

    if factor_scores is not None and not factor_scores.empty:
        frame = frame.join(factor_scores, how="left")
        # Forward-fill only: a monthly factor is knowable until it is revised, and
        # a backward fill would carry a later release into an earlier date.
        frame[factor_scores.columns] = frame[factor_scores.columns].ffill()
        frame = frame.dropna(subset=list(factor_scores.columns))

    return frame.dropna()


def _fit_booster(x: np.ndarray, y: np.ndarray, *, seed: int, params: dict[str, Any]):
    import xgboost as xgb

    settings = {**DEFAULT_PARAMS, **params, "seed": seed}
    n_estimators = int(settings.pop("n_estimators", 300))
    # Class imbalance: adverse windows are a minority at 3m and a majority at 12m,
    # so the weight is computed rather than assumed.
    positives = float(y.sum())
    if 0 < positives < len(y):
        settings["scale_pos_weight"] = (len(y) - positives) / positives
    return xgb.train(settings, xgb.DMatrix(x, label=y), num_boost_round=n_estimators)


def fit_transition_model(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    horizon_months: int,
    fitted_on: str,
    is_proxy: bool = False,
    n_folds: int = DEFAULT_FOLDS,
    embargo_months: int = DEFAULT_EMBARGO_MONTHS,
    seed: int = DEFAULT_SEED,
    params: dict[str, Any] | None = None,
) -> TransitionModel:
    """Purged-CV fit plus isotonic calibration for one horizon.

    The isotonic map is fitted on **out-of-fold** predictions only. Calibrating on
    in-sample scores is the classic way to produce a beautiful reliability
    diagram for a model that is not calibrated at all.
    """
    import xgboost as xgb  # noqa: F401  (imported for the error if it is missing)

    aligned = features.join(labels.rename("label"), how="inner").dropna()
    if aligned.empty:
        raise CalibrationError(f"{horizon_months}m: no rows survive the label join")

    y = aligned["label"].to_numpy(dtype=float)
    x_frame = aligned.drop(columns=["label"])
    x = x_frame.to_numpy(dtype=float)
    index = pd.DatetimeIndex(aligned.index)

    positives = int(y.sum())
    if positives < 30 or positives > len(y) - 30:
        raise CalibrationError(
            f"{horizon_months}m: {positives} positives in {len(y)} rows is too "
            "unbalanced to calibrate a probability against"
        )

    folds = purged_folds(index, horizon_months, n_folds=n_folds, embargo_months=embargo_months)

    out_of_fold = np.full(len(y), np.nan)
    reports: list[FoldReport] = []
    for number, (train, test) in enumerate(folds, start=1):
        if train.size < 200 or np.unique(y[train]).size < 2:
            log.warning(
                "%dm fold %d: %d training rows after purging — skipped",
                horizon_months,
                number,
                train.size,
            )
            continue
        booster = _fit_booster(x[train], y[train], seed=seed, params=params or {})
        scores = booster.predict(xgb.DMatrix(x[test]))
        out_of_fold[test] = scores

        fold_positives = int(y[test].sum())
        reports.append(
            FoldReport(
                fold=number,
                train_rows=int(train.size),
                test_rows=int(test.size),
                positives=fold_positives,
                brier=float(brier_score_loss(y[test], scores)),
                auc=(
                    float(roc_auc_score(y[test], scores))
                    if 0 < fold_positives < test.size
                    else None
                ),
            )
        )

    scored = np.isfinite(out_of_fold)
    if scored.sum() < 200 or np.unique(y[scored]).size < 2:
        raise CalibrationError(
            f"{horizon_months}m: only {int(scored.sum())} out-of-fold predictions "
            "survived purging; there is nothing to calibrate against"
        )

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(out_of_fold[scored], y[scored])
    calibrated = isotonic.predict(out_of_fold[scored])

    # The production booster sees everything; only the calibration map is
    # restricted to out-of-fold data, which is where the leakage risk lives.
    final = _fit_booster(x, y, seed=seed, params=params or {})

    model = TransitionModel(
        horizon_months=horizon_months,
        feature_names=tuple(x_frame.columns),
        booster_json=final.save_raw(raw_format="json").decode("utf-8"),
        calibration_x=tuple(float(v) for v in isotonic.X_thresholds_),
        calibration_y=tuple(float(v) for v in isotonic.y_thresholds_),
        fitted_on=fitted_on,
        fitted_on_proxy=is_proxy,
        observations=len(y),
        positives=positives,
        base_rate=float(y.mean()),
        folds=tuple(reports),
        brier=float(brier_score_loss(y[scored], calibrated)),
        auc=float(roc_auc_score(y[scored], calibrated)),
        seed=seed,
    )

    log.info(
        "L3 %dm on %s: %d rows, base rate %.1f%%, out-of-fold Brier %.4f "
        "(vs %.4f for always predicting the base rate), AUC %.3f",
        horizon_months,
        fitted_on,
        model.observations,
        model.base_rate * 100,
        model.brier,
        model.base_rate * (1 - model.base_rate),
        model.auc or float("nan"),
    )
    return model


def fit_all_horizons(
    design: RegimeDesign,
    posteriors: pd.DataFrame,
    states: pd.Series,
    *,
    fitted_on: str,
    is_proxy: bool = False,
    factor_scores: pd.DataFrame | None = None,
    horizons: tuple[int, ...] = HORIZON_MONTHS,
    **kwargs: Any,
) -> TransitionModels:
    """Fit one calibrated classifier per horizon. A horizon that cannot be fitted
    is skipped with a warning rather than failing the run — 3m and 12m have very
    different positive rates and one may be trainable when the other is not."""
    features = build_training_frame(design, posteriors, factor_scores=factor_scores)

    models: dict[int, TransitionModel] = {}
    for months in horizons:
        labels = transition_labels(states, months, design.periods_per_year)
        try:
            models[months] = fit_transition_model(
                features,
                labels,
                horizon_months=months,
                fitted_on=fitted_on,
                is_proxy=is_proxy,
                **kwargs,
            )
        except CalibrationError as err:
            log.warning("L3 %dm could not be fitted: %s", months, err)
    return TransitionModels(models=models)


__all__ = [
    "ADVERSE_REGIMES",
    "DEFAULT_EMBARGO_MONTHS",
    "DEFAULT_FOLDS",
    "DEFAULT_PARAMS",
    "DEFAULT_SEED",
    "HORIZON_MONTHS",
    "CalibrationError",
    "FoldReport",
    "TransitionModel",
    "TransitionModels",
    "build_training_frame",
    "fit_all_horizons",
    "fit_transition_model",
    "purged_folds",
    "transition_labels",
]
