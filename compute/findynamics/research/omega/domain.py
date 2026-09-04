"""Vocabulary owned by the KK-Ω module — names, not numbers.

Kept apart from :mod:`findynamics.research.omega.config` for the reason
``engines/equity/domain.py`` is kept apart from ``config/engines/equity.yaml``:
a *threshold* is a tunable and belongs in yaml, but a *name* is a contract. The
three latent states below are printed in diagnostics, compared against the
equity HMM's regimes in KK3 and quoted in the KK5 report; renaming one is a
change to every artifact that has ever carried it, not a recalibration.

Nothing here is registered anywhere. ``core/contracts/vocab.py::ASSETS`` remains
a closed five-tuple and these names never travel through it.
"""

from __future__ import annotations

from typing import Final

#: The banding of Ω's expanding percentile (``docs/research/kk-omega-design.md``
#: §2, last row). Ordered from calm to stressed, and that order is load-bearing:
#: :data:`OMEGA_REGIME_CODES` reads it into integers so a chart of the code runs
#: upwards as latent stress rises.
#:
#: **These are not regimes in the sense ``engines/equity/domain.py::REGIMES``
#: means.** Those five come out of a fitted HMM with a transition matrix and a
#: posterior. These three are a banding of one continuous variable, carry no
#: dynamics, and exist so that a diagnostic panel can say something discrete
#: about a series that is otherwise a float. Whether they say anything the HMM
#: does not is measured in KK3 §5, and the answer is allowed to be "no".
OMEGA_REGIMES: Final[tuple[str, ...]] = (
    "latent_calm",
    "latent_transition",
    "latent_stress",
)

#: Label -> the integer a CSV artifact carries. ``-1`` for "not yet scored",
#: matching ``kinematics.JERK_LAMP_CODES``' treatment of ``unknown``: a warm-up
#: row has no reading, and calling that ``latent_calm`` would publish a claim
#: where there is an absence.
OMEGA_REGIME_CODES: Final[dict[str, int]] = {
    "unknown": -1,
    "latent_calm": 0,
    "latent_transition": 1,
    "latent_stress": 2,
}

#: Every candidate column of the observable block ``Z_t``, in the order the
#: design note's §3 table lists them. The *order here is documentation only* —
#: :func:`findynamics.research.omega.features.available_columns` sorts the
#: survivors, because the surviving set is dotted with a loading vector and a
#: projection that depended on assembly order would be a different projection.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "realized_vol",
    "vol_of_vol",
    "abs_return",
    "drawdown",
    "dispersion",
    "rate_level_chg",
    "curve_slope",
    "liquidity_stress",
    "credit_velocity",
)

#: The subset derivable from a price series alone. This is what makes the
#: 1927-2026 specification possible: ``YAHOO:^GSPC`` reaches back to 1927-12-30
#: in the committed fixture, and none of the rates, liquidity or credit columns
#: exist anywhere near that far back.
#:
#: It is a **different specification**, not a longer run of the same one, and
#: every table reports it as its own row.
PRICE_DERIVED_COLUMNS: Final[tuple[str, ...]] = (
    "realized_vol",
    "vol_of_vol",
    "abs_return",
    "drawdown",
)

__all__ = [
    "FEATURE_COLUMNS",
    "OMEGA_REGIMES",
    "OMEGA_REGIME_CODES",
    "PRICE_DERIVED_COLUMNS",
]
