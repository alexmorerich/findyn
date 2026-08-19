"""Regime-conditional strategic weights, as a distribution (Phase 6).

The method, in full, because there is no black box and the dashboard promises
there is not:

1. **Score.** Each usable asset gets a risk-adjusted excess return —
   ``sharpe = (expected_return − risk_free) / sigma``, where ``sigma`` is the
   0-100 risk score mapped to a variance proxy (``config/portfolio.yaml`` risk
   block). Cash's excess is zero by definition, so its score is zero.

2. **Tilt.** Weights are an *exponential tilt* of the profile's neutral mix:
   ``w_a ∝ neutral_a · exp(tilt_strength · confidence_a · (sharpe_a − sharpē))``,
   where ``sharpē`` is the neutral-weighted average score across the usable
   assets. An asset with above-average, confidently-held risk-adjusted return is
   overweighted relative to neutral; a below-average one is underweighted;
   ``confidence`` scales how far each engine is allowed to move its own weight.
   This is entropy-tilting of a prior allocation — a documented method, not a
   fitted optimiser — and it degenerates to exactly the neutral mix when every
   score equals the average.

3. **Distribution.** The point estimate is not the whole answer; the width is.
   Each engine's expected return is resampled ``samples`` times with a Gaussian
   spread of ``return_sd · (1 − confidence)`` — a confident engine barely moves
   and its weight band is tight, an unconfident one swings and its band is wide.
   The tilt is recomputed per draw, giving a *distribution* over each asset's
   weight, reported as quantiles. The fan on the dashboard is the engines'
   confidence made visible.

4. **Guardrails.** Every draw is passed through ``guardrails``: capped per asset,
   summing to 1, with degraded assets pinned at neutral and crypto excluded.

The regime enters through the engines: an ``AssetState``'s ``expected_return``,
``risk_score`` and ``confidence`` are already regime-conditional (the equity
engine's return is conditioned on its HMM posterior, gold's on its switching
model). The portfolio layer does not re-model regimes; it consumes the states
that already did. What is "regime-conditional" here is the whole panel.

Non-goals held (FINDYN_V1_SPEC.md §0/§12): no trade command, no deterministic
target. The output is a weight *distribution* per asset and a templated
conditional implication.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from findynamics.portfolio import guardrails
from findynamics.portfolio.config import PortfolioConfig, ProfileConfig
from findynamics.portfolio.implications import render_implication
from findynamics.portfolio.inputs import DecisionPanel

METHOD = (
    "exponential tilt of the profile's neutral mix by confidence-weighted "
    "risk-adjusted excess return; distribution by resampling each engine's "
    "expected return at a spread set by its confidence"
)


@dataclass(frozen=True)
class WeightBand:
    """One asset's weight as a distribution, not a number."""

    asset: str
    #: Mean weight across the sampled allocations. The central read; sums to 1
    #: across assets by construction (every sample does).
    mean: float
    #: The profile's strategic neutral weight, for the delta the implication and
    #: the chart are drawn against.
    neutral: float
    #: Quantile -> weight, on ``config.distribution.quantiles``. A degraded asset
    #: is a point mass at ``neutral`` — every quantile equal.
    quantiles: dict[str, float]
    #: True when this asset fell back to neutral (its engine was missing/stale or
    #: published no return).
    degraded: bool
    reason: str = ""

    @property
    def delta(self) -> float:
        """Mean weight minus neutral — the tilt, in weight points."""
        return self.mean - self.neutral


@dataclass(frozen=True)
class InputRef:
    """A reference to one input ``AssetState``, for the 'why' trace."""

    asset: str
    as_of: date | None
    regime: str
    expected_return: float | None
    risk_score: float
    confidence: float
    stale: bool
    reason: str


@dataclass(frozen=True)
class Allocation:
    """The portfolio layer's output for one profile and one run."""

    profile: str
    as_of: date
    model_version: str
    method: str
    risk_free: float | None
    quantile_grid: tuple[float, ...]
    weights: dict[str, WeightBand]
    inputs: tuple[InputRef, ...]
    factors: dict[str, float]
    degraded: bool
    degraded_reason: tuple[str, ...]
    implication: str

    def caps(self, profile: ProfileConfig) -> dict[str, float]:
        return {a: profile.cap(a) for a in self.weights}


def _sigma(risk_score: float, config: PortfolioConfig) -> float:
    """Map a 0-100 risk score to a variance proxy (the risk block's agreed map)."""
    return config.sigma_floor + (max(risk_score, 0.0) / 100.0) * config.sigma_scale


def _profile_seed(config: PortfolioConfig, profile: str) -> list[int]:
    """A reproducible per-profile RNG seed, so profiles draw independently.

    A stable function of the configured seed and the profile name — the replay
    test depends on the same states producing the same bands, and two profiles
    sharing one stream would correlate their fans for no reason. The offset is
    position-weighted so permuted names (an anagram) get distinct streams rather
    than colliding, which a plain character sum would allow.
    """
    offset = sum((index + 1) * ord(ch) for index, ch in enumerate(profile))
    return [config.distribution.seed, offset]


def _tilt_matrix(
    excess: np.ndarray,
    sigma: np.ndarray,
    confidence: np.ndarray,
    neutral: np.ndarray,
    tilt_strength: float,
    budget: float,
) -> np.ndarray:
    """Exponential-tilt weights for the usable assets, one row per draw.

    ``excess`` is ``(samples, k)``; the rest are ``(k,)``. Returns ``(samples,
    k)`` weights that sum to ``budget`` along each row — the budget being the
    neutral mass of the usable assets, i.e. what is left after the degraded
    assets take their neutral weight.
    """
    sharpe = excess / sigma[None, :]
    neutral_sum = neutral.sum()
    # Neutral-weighted average score, per draw: the tilt is relative to the mix,
    # so an all-equal-score panel produces exactly the neutral weights.
    sharpe_bar = (sharpe * neutral[None, :]).sum(axis=1, keepdims=True) / neutral_sum
    signal = confidence[None, :] * (sharpe - sharpe_bar)
    raw = neutral[None, :] * np.exp(tilt_strength * signal)
    return budget * raw / raw.sum(axis=1, keepdims=True)


def _cap_to_budget(
    weights: dict[str, float],
    caps: dict[str, float],
    budget: float,
) -> dict[str, float]:
    """Enforce per-asset caps on a set of weights that should sum to ``budget``.

    Reuses the water-filling in :mod:`guardrails` by working in budget-relative
    units: normalise to 1, cap at ``cap/budget``, and scale back. Keeps the
    capping in one place rather than reimplementing it for the sub-budget case.
    """
    if budget <= 1e-12 or not weights:
        return {a: 0.0 for a in weights}
    normalized = {a: v / budget for a, v in weights.items()}
    relative_caps = {a: caps.get(a, float("inf")) / budget for a in weights}
    capped = guardrails.enforce_caps(normalized, relative_caps)
    return {a: v * budget for a, v in capped.items()}


def allocate(panel: DecisionPanel, profile: ProfileConfig, config: PortfolioConfig) -> Allocation:
    """Weight distribution + conditional implication for one profile.

    Consumes only the panel (states + cash's price) and the config. Deterministic
    given the panel and ``config.distribution.seed`` — the PIT replay test asserts
    it reproduces from a past information set.
    """
    universe = profile.universe
    dist = config.distribution
    quantile_labels = tuple(_qlabel(q) for q in dist.quantiles)

    # Partition the universe: which assets carry a real tilt, which fall back to
    # neutral. `usable` needs a fresh state with an excess return to respond to.
    usable = [a for a in universe if (inp := panel.get(a)) is not None and inp.usable]
    degraded = [a for a in universe if a not in usable]

    neutral = {a: profile.neutral.get(a, 0.0) for a in universe}
    caps = {a: profile.cap(a) for a in universe}
    budget = sum(neutral[a] for a in usable)

    samples = dist.samples
    full = np.zeros((samples, len(universe)))
    col = {a: i for i, a in enumerate(universe)}

    # Degraded assets are a point mass at their neutral weight across every draw.
    for a in degraded:
        full[:, col[a]] = neutral[a]

    if usable and budget > 1e-12:
        rng = np.random.default_rng(_profile_seed(config, profile.name))
        base_excess = np.array([panel.get(a).excess_return for a in usable], dtype=float)
        conf = np.clip(np.array([panel.get(a).confidence for a in usable], dtype=float), 0.0, 1.0)
        sigma = np.array([_sigma(panel.get(a).risk_score, config) for a in usable], dtype=float)
        neutral_arr = np.array([neutral[a] for a in usable], dtype=float)

        noise_sd = dist.return_sd * (1.0 - conf)
        draws = rng.standard_normal((samples, len(usable))) * noise_sd[None, :]
        excess_samples = base_excess[None, :] + draws

        w_usable = _tilt_matrix(
            excess_samples, sigma, conf, neutral_arr, profile.tilt_strength, budget
        )

        # Cap each draw. The cap step is nonlinear (water-filling), so it is done
        # per draw rather than vectorised — a thousand small dict operations,
        # cheap enough for the nightly run.
        for s in range(samples):
            draw = dict(zip(usable, w_usable[s].tolist(), strict=True))
            for a, value in _cap_to_budget(draw, caps, budget).items():
                full[s, col[a]] = value

    bands: dict[str, WeightBand] = {}
    for a in universe:
        column = full[:, col[a]]
        inp = panel.get(a)
        is_degraded = a in degraded
        bands[a] = WeightBand(
            asset=a,
            mean=float(column.mean()),
            neutral=neutral[a],
            quantiles={
                label: float(np.quantile(column, q))
                for label, q in zip(quantile_labels, dist.quantiles, strict=True)
            },
            degraded=is_degraded,
            reason=(inp.reason if (inp is not None and is_degraded) else ""),
        )

    means = {a: bands[a].mean for a in universe}
    # The point allocation is a valid allocation, not just an average of them:
    # every draw was capped and summed to 1, so their mean does too.
    guardrails.validate(means, caps)

    degraded_reason = tuple(f"{a}: {bands[a].reason}" for a in degraded if bands[a].reason)
    inputs = tuple(
        InputRef(
            asset=a,
            as_of=(panel.get(a).as_of if panel.get(a) else None),
            regime=(panel.get(a).regime if panel.get(a) else "unavailable"),
            expected_return=(panel.get(a).expected_return if panel.get(a) else None),
            risk_score=(panel.get(a).risk_score if panel.get(a) else 0.0),
            confidence=(panel.get(a).confidence if panel.get(a) else 0.0),
            stale=(panel.get(a).stale if panel.get(a) else True),
            reason=(panel.get(a).reason if panel.get(a) else "no state published"),
        )
        for a in universe
    )

    implication = render_implication(
        profile=profile,
        bands={a: (bands[a].mean, bands[a].neutral) for a in universe},
        degraded=bool(degraded),
        config=config,
    )

    return Allocation(
        profile=profile.name,
        as_of=panel.as_of,
        model_version=config.model_version,
        method=METHOD,
        risk_free=panel.risk_free,
        quantile_grid=dist.quantiles,
        weights=bands,
        inputs=inputs,
        factors=dict(panel.factors),
        degraded=bool(degraded),
        degraded_reason=degraded_reason,
        implication=implication,
    )


def _qlabel(q: float) -> str:
    """Stable string key for a quantile, e.g. 0.05 -> 'q05', 0.5 -> 'q50'.

    A string because it becomes a JSON object key in D1 and a chart series name;
    keyed off the integer percentile so 0.05 and 0.5 never collide as '0.5'.
    """
    return f"q{round(q * 100):02d}"


def weights_blob(allocation: Allocation) -> dict:
    """The JSON object stored in ``portfolio_state.weights`` and served by the API.

    Carries the whole distribution, the neutral mix it is tilted from, the input
    ``AssetState`` references for the 'why' expander, and the risk-free rate — so
    one row answers ``GET /api/v1/portfolio`` completely.
    """
    return {
        "method": allocation.method,
        "risk_free": allocation.risk_free,
        "quantile_grid": list(allocation.quantile_grid),
        "quantile_labels": [_qlabel(q) for q in allocation.quantile_grid],
        "factors": allocation.factors,
        "degraded_reason": list(allocation.degraded_reason),
        "assets": [
            {
                "asset": band.asset,
                "mean": round(band.mean, 6),
                "neutral": round(band.neutral, 6),
                "delta": round(band.delta, 6),
                "degraded": band.degraded,
                "reason": band.reason,
                "quantiles": {k: round(v, 6) for k, v in band.quantiles.items()},
            }
            for band in allocation.weights.values()
        ],
        "inputs": [
            {
                "asset": ref.asset,
                "as_of": ref.as_of.isoformat() if ref.as_of else None,
                "regime": ref.regime,
                "expected_return": ref.expected_return,
                "risk_score": round(ref.risk_score, 4),
                "confidence": round(ref.confidence, 4),
                "stale": ref.stale,
                "reason": ref.reason,
            }
            for ref in allocation.inputs
        ],
    }
