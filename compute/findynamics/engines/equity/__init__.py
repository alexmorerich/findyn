"""FinEquity — growth and risk premium. Phase 3.

FINDYN_V1_SPEC.md is the specification of this engine: Kalman kinematics, FFD,
5-state HMM, XGBoost calibration, RII, crash decomposition, Monte Carlo.

Landed so far (sub-milestone A / spec M2): role resolution and the causal
feature path. ``predict`` raises ``StateUnavailable`` until the regime model
lands in sub-milestone B — the engine publishes features it can stand behind and
declines to publish a state it cannot.

Importing this package registers the engine; ``findynamics/engines/__init__.py``
does that only when ``config/engines/equity.yaml`` says ``enabled: true``.
"""

from findynamics.engines.equity.engine import EquityEngine

__all__ = ["EquityEngine"]
