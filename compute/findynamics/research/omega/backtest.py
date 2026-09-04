"""The walk-forward harness — where leakage would actually enter. **Scaffold (KK0).**

Everything reported by this track must come from a model that had never seen
the observation it is being graded on. The unit tests in KK1 and KK2 cover the
transforms; they do not cover the harness, and the harness is where a leak is
both easiest to introduce and hardest to see.

**One PIT gateway, reused.** ``findynamics/backtest/replay.py::world_at`` binds
a ``PandasPITAccessor`` to the cutoff, so nothing downstream can widen the
information set. A second gateway written here would be a second place for the
release-date filter to be subtly wrong, and the first one is already tested.

Per rebalance date ``t``, on an **expanding** window (cadence from config,
default monthly):

1. build ``Z`` from data with ``release_date <= t``;
2. fit the scaler and the Ω estimator on that window — **fit, then freeze**;
3. transform forward only: Ω on ``(t, t + cadence]`` uses the spec fitted at
   ``t``;
4. compute Ω̇, Ω̈, ``C`` and ``K`` from the stitched causal path;
5. fit the arm's predictive model on the window and predict ``(t, t+cadence]``;
6. record the ``OmegaSpec``, the coefficients, the config hash and the model
   version for that window.

The stitched out-of-sample path is **the only thing any metric may be computed
on**. It is written to CSV under ``compute/backtests/omega/`` so the report is
regenerable from a clone with no API keys.

**Warm-up.** The first ``min_train_years`` of rebalance dates (config, default
5) are discarded entirely. An Ω fitted on 200 observations is a statement about
its own start-up, exactly as ``kalman_burn_in_years`` is for velocity.

The six arms
------------

Identical in everything but the feature set: **A** (``P, v, a, j``), **A′**
(``A`` + the published RII and its components), **B** (``A + Ω``), **C**
(``A + Ω, Ω̇, Ω̈``), **D** (``A + C``), **E** (full).

**A′ is mandatory.** Without it, "Ω adds information over price alone" is a
claim about a strawman — ``rii.py`` already publishes a composite built from
nearly the same observables. Any incremental result that survives A but not A′
is reported as *no incremental information over what the engine already
publishes*.

Ridge for continuous targets, logistic for binary, regularization chosen on an
**inner expanding split of the training window only**. Not XGBoost: the arms
differ by one to five columns, and a high-variance learner would make the
comparison a study of its own tuning rather than of the columns.

The four tests that decide whether any of this counts
-----------------------------------------------------

* **Harness truncation.** Run the walk-forward on data truncated at ``T`` and
  again on the full data; every prediction, Ω value and coefficient dated
  ``<= T`` must be bit-identical.
* **Shuffled-target control.** Re-run with the target permuted under a fixed
  seed; every arm's OOS R² must be within noise of zero and no AUC may exceed
  the configured ceiling (default 0.55). **If the shuffled control shows skill,
  the harness is leaking and every other number in the phase is void.** This is
  the acceptance gate.
* **Future spike.** Insert an extreme value after ``T``; nothing dated ``<= T``
  may change.
* **Window boundary.** No refit boundary may produce a jump in Ω larger than a
  configured multiple of its own trailing standard deviation — the harness-level
  catch for a sign flip or a scaler reset that KK1's two-window unit test missed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - scaffold
    from pathlib import Path

    import pandas as pd

    from findynamics.research.omega.config import OmegaConfig


def walk_forward(
    observations: pd.DataFrame,
    *,
    config: OmegaConfig,
    out_dir: Path | None = None,
) -> object:
    """The stitched out-of-sample path, one row per date, one spec per window."""
    raise NotImplementedError("KK3: implement the walk-forward on replay.world_at")


def shuffled_target_control(
    observations: pd.DataFrame,
    *,
    config: OmegaConfig,
    seed: int,
) -> object:
    """The acceptance gate: the same pipeline against a permuted target."""
    raise NotImplementedError("KK3: implement the shuffled-target control")


__all__ = ["shuffled_target_control", "walk_forward"]
