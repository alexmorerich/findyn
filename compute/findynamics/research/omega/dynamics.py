"""Ω̇, Ω̈, and the banding of Ω. **Scaffold (KK0).**

The temptation here is ``omega.diff()`` twice. ``features/kinematics.py`` opens
by explaining why that is wrong for price and the argument transfers exactly:
differencing a noisy series amplifies its noise by roughly the differencing
order, so a second difference of a PC1 is mostly microstructure wearing an
acceleration's name.

So Ω goes through the **same local-linear-trend Kalman filter price goes
through** (``engines/equity/features/kalman.py::filter_state`` — ``filter()``,
never ``smooth()``, because the RTS smoother conditions on the whole sample).
Ω̇ is the filtered slope, annualized by ``periods_per_year``; Ω̈ is its first
difference. Reusing that module rather than copying it is the point of placing
this package above ``engines`` in the layer stack: two implementations of the
same filter drift, and the drift looks like a market signal.

The same **burn-in** applies, for the same reason. The filter is initialized
diffusely and its slope standard error starts at 1e3; ``kinematics.py`` measured
the consequence on the century price path — the first published velocity was
**+142% annualized** — and discards ``kalman_burn_in_years`` of leading slope
because of it. An Ω path start-up is the same artifact with a new name, and
``OmegaPath.diagnostics`` reports ``burn_in_periods`` so the discarded span is
visible rather than assumed.

Everything else here is expanding and never rolling:

* ``omega_volatility`` — trailing standard deviation of Ω̇, window from config.
* ``omega_zscore`` — **expanding** z-score of Ω with a minimum baseline in
  years, exactly as ``jerk_z`` is scored. Never a rolling z, never a full-sample
  one: a full-sample mean puts every future observation into every historical
  score.

``omega_regime``
----------------

A three-state reading over Ω's **expanding percentile**, with thresholds in
config and the vocabulary in :mod:`findynamics.research.omega.domain`:
``latent_calm | latent_transition | latent_stress``.

This is a display and diagnostic banding of one continuous variable. It carries
no dynamics of its own — there is no transition matrix, no posterior, no
persistence prior, and it is **not** a second regime model. Whether it says
anything the existing five-state HMM does not is measured in KK3 §5, against
the episode list already in ``engines/equity/backtest.py::EPISODES``, and the
answer is allowed to be "it fires later and more often", which would be a
complete result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - scaffold
    import pandas as pd

    from findynamics.research.omega.config import OmegaConfig
    from findynamics.research.omega.contracts import OmegaSpec


def omega_dynamics(
    omega: pd.Series,
    spec: OmegaSpec,
    *,
    periods_per_year: float,
    config: OmegaConfig,
) -> object:
    """The full :class:`OmegaPath` for one stitched causal Ω path.

    ``omega`` is the *stitched* out-of-sample path from the walk-forward, not a
    single window's transform: derivatives taken per window would restart at
    every refit boundary and put a discontinuity where the model has none.
    """
    raise NotImplementedError("KK1: implement Ω̇/Ω̈ via features/kalman.py::filter_state")


def omega_regime(percentile: pd.Series, *, config: OmegaConfig) -> pd.Series:
    """Band an expanding percentile of Ω into the three latent states."""
    raise NotImplementedError("KK1: implement the banding (docs/research/kk-omega-design.md §2)")


__all__ = ["omega_dynamics", "omega_regime"]
