"""KK-Ω — a latent market-state coordinate, as a falsifiable research question.

The equity engine represents an asset as ``(P, v, a, j)``: price, and three
derivatives of the Kalman-filtered log price (``FINDYN_V1_SPEC.md`` §8.1, §8.3).
This package asks whether extending that to
``(P, v, a, j, Ω, Ω̇, Ω̈, C, K)`` carries statistically significant
out-of-sample information about future returns, volatility, regime transitions
and drawdowns.

A fifth-dimensional latent-state representation inspired by Kaluza–Klein
geometry is being tested as a quantitative modelling hypothesis. The line
element ``ds² = g_ij dx^i dx^j + φ²(dΩ + A_i dx^i)²`` is guidance for the
*shape* of the model — an observable block, one latent coordinate, and an
explicit coupling between them — and nothing else is borrowed from it. No claim
is made about physics.

**The question is not "does Ω beat price alone".** ``engines/equity/rii.py``
already publishes a composite built from posterior entropy, confidence deficit,
``|jerk_z|``, vol-of-vol, stock-bond correlation breakdown, credit velocity and
liquidity stress — nearly the same observables Ω would be built from. Beating
four derivatives of one series is close to guaranteed and close to worthless.
The question is whether Ω carries information beyond ``(P, v, a, j)`` **and
beyond the RII the engine already publishes**, which is why every model
comparison in this track carries an RII control arm.

The module is allowed to fail. A well-evidenced ``REJECTED`` closes the track
successfully; ``docs/research/kk-omega-design.md`` §9 pre-registers the verdict
rules and §10 says what would make us delete this package.

Two independent gates keep it out of production:

1. the ``Research is quarantined`` import-linter contract, which makes a
   production import of this package a CI failure rather than a code review
   comment; and
2. ``config/research/omega.yaml``, which ships ``enabled: false``.

The first is the real one. The flag is the second line of defence, and reading
it as the only one is how a research module ends up in a daily job.
"""

from __future__ import annotations

from findynamics.research.omega.config import (
    OMEGA_CONFIG_PATH,
    OmegaConfig,
    OmegaConfigError,
    get_omega_config,
    is_enabled,
    load_omega_config,
)

__all__ = [
    "OMEGA_CONFIG_PATH",
    "OmegaConfig",
    "OmegaConfigError",
    "get_omega_config",
    "is_enabled",
    "load_omega_config",
]
