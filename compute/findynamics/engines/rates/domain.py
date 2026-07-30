"""FinRates vocabulary — names this engine owns.

Every engine defines its own regime vocabulary (``03-contracts.md`` §1); these
names never move into ``core``. Mirrored in ``serving/src/domain.ts`` as
``RATE_REGIMES``, and ``tests/test_domain.py`` fails on drift.

The five regimes describe the *shape and direction* of the curve, not a forecast
of rates. They are mutually exclusive and evaluated in the order listed in
:mod:`findynamics.engines.rates.regime`:

``inverted``
    Long yields below short. The curve is pricing cuts; historically the most
    reliable late-cycle marker in the set.
``re_steepening``
    Out of inversion recently, and the spread is actually rising. Distinguished
    from a merely positive curve because the exit from inversion — not the
    inversion itself — is what has tended to coincide with recessions.
``flat``
    Spread inside a narrow band around zero. No directional information; the
    honest label for a curve that is not saying anything.
``steep_easing``
    Positively sloped with the level falling — policy accommodating.
``steep_tightening``
    Positively sloped with the level rising — policy restrictive, or term
    premium rebuilding.
"""

from __future__ import annotations

from typing import Final

#: Order is the wire order: ``engine_output`` publishes the regime as its index
#: here (``regime_code``), because that table stores REAL values.
RATE_REGIMES: Final[tuple[str, ...]] = (
    "steep_easing",
    "steep_tightening",
    "flat",
    "inverted",
    "re_steepening",
)

#: The wide metrics FinRates publishes per date (``engine_output.metric``).
#:
#: ``ns_lambda`` is published even though it is frozen and identical on every
#: date, because the three betas are meaningless without it: a consumer holding
#: level, slope and curvature cannot evaluate the curve at a maturity it was not
#: given. Publishing it per date rather than once is what makes the history
#: self-describing — a date's factors travel with the lambda they were fitted at,
#: so a refit that moves lambda cannot silently reinterpret older rows.
RATE_METRICS: Final[tuple[str, ...]] = (
    "ns_level",
    "ns_slope",
    "ns_curvature",
    "ns_rmse",
    "ns_lambda",
    "regime_code",
)


def regime_code(regime: str) -> int:
    """Index of ``regime`` in :data:`RATE_REGIMES`."""
    try:
        return RATE_REGIMES.index(regime)
    except ValueError:
        raise ValueError(
            f"unknown rate regime {regime!r}; expected one of {RATE_REGIMES}"
        ) from None
