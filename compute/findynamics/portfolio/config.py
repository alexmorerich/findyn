"""Load and validate ``config/portfolio.yaml`` (Phase 6).

Strict, like :mod:`findynamics.core.config`: a malformed profile fails at load
time rather than producing a silently wrong allocation. The portfolio config is
its own file and its own loader because it is owned by no engine and belongs to
no series — it is the strategic layer's parameters, and mixing it into
``series.yaml`` would put a risk profile next to a FRED publication lag.

The one cross-check with the rest of the config that matters is enforced here:
every weight cap must be at least the asset's neutral weight, because the
stale-engine fallback holds an asset *at* its neutral weight, and a cap below
that would make the fallback itself violate the guardrails it exists to satisfy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from findynamics.core.contracts.vocab import ASSETS

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PORTFOLIO_CONFIG_PATH = CONFIG_DIR / "portfolio.yaml"

#: Tolerance on the neutral-weights-sum-to-one check. Floats read from yaml do
#: not land on 1.0 exactly, but a mix that sums to 0.97 is a typo, not rounding.
_SUM_TOLERANCE = 1e-6


class PortfolioConfigError(ValueError):
    """Raised when portfolio.yaml does not satisfy its contract."""


@dataclass(frozen=True)
class ProfileConfig:
    """One risk profile: a neutral strategic mix and its per-asset caps."""

    name: str
    #: Asset -> strategic neutral weight. Sums to 1; keys are the profile's
    #: allocatable universe.
    neutral: dict[str, float]
    #: Asset -> hard upper bound on that asset's weight. Every allocatable asset
    #: has one; crypto's is 0 in every shipped profile.
    caps: dict[str, float]
    #: How far the allocator may lean away from ``neutral`` toward attractive
    #: assets. Higher is more aggressive.
    tilt_strength: float

    @property
    def universe(self) -> tuple[str, ...]:
        """The assets this profile allocates over, in ASSETS order."""
        return tuple(a for a in ASSETS if a in self.neutral)

    def cap(self, asset: str) -> float:
        """Upper bound for ``asset``; 0.0 for anything the profile does not name."""
        return float(self.caps.get(asset, 0.0))


@dataclass(frozen=True)
class DistributionConfig:
    samples: int
    seed: int
    return_sd: float
    quantiles: tuple[float, ...]


@dataclass(frozen=True)
class ImplicationConfig:
    """Templated conditional-implication copy (FINDYN_V1_SPEC.md §12)."""

    modest_band: float
    pronounced_band: float
    templates: dict[str, str]
    degraded_suffix: str
    disclaimer: str


@dataclass(frozen=True)
class PortfolioConfig:
    model_version: str
    distribution: DistributionConfig
    sigma_floor: float
    sigma_scale: float
    max_state_age_days: int
    profiles: dict[str, ProfileConfig]
    implications: ImplicationConfig
    labels: dict[str, str] = field(default_factory=dict)

    def profile(self, name: str) -> ProfileConfig:
        try:
            return self.profiles[name]
        except KeyError:
            raise PortfolioConfigError(
                f"unknown profile {name!r}; configured: {sorted(self.profiles)}"
            ) from None

    def label(self, asset: str) -> str:
        """Plain-language name for an asset, falling back to the engine name."""
        return self.labels.get(asset, asset)


def _require_number(value: object, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PortfolioConfigError(f"{where}: expected a number, got {value!r}")
    return float(value)


def _parse_weight_map(raw: object, where: str) -> dict[str, float]:
    if not isinstance(raw, dict) or not raw:
        raise PortfolioConfigError(f"{where}: expected a non-empty mapping")
    out: dict[str, float] = {}
    for asset, value in raw.items():
        if asset not in ASSETS:
            raise PortfolioConfigError(f"{where}: {asset!r} is not one of {list(ASSETS)}")
        weight = _require_number(value, f"{where}.{asset}")
        if weight < 0.0:
            raise PortfolioConfigError(f"{where}.{asset}: weight must be >= 0, got {weight}")
        out[asset] = weight
    return out


def _parse_profile(name: str, raw: object) -> ProfileConfig:
    if not isinstance(raw, dict):
        raise PortfolioConfigError(f"profiles.{name}: expected a mapping")

    neutral = _parse_weight_map(raw.get("neutral"), f"profiles.{name}.neutral")
    # `math.fsum`, not `sum`, and the difference is a CI failure rather than a
    # rounding nicety. CPython 3.12 gave `sum` Neumaier compensation for floats;
    # 3.11 sums naively. So `sum([0.60, 0.30, 0.10])` is exactly 1.0 on a
    # developer's 3.13 and 0.9999999999999999 on the 3.11 the workflow pins —
    # which made the division below a no-op locally and a perturbation in CI,
    # turning the shipped 60/30/10 into 0.6000000000000001 and failing
    # `test_balanced_neutral_is_the_backtest_benchmark` on that runner alone.
    # `math.fsum` is exactly rounded on every version, so the two agree.
    total = math.fsum(neutral.values())
    if abs(total - 1.0) > _SUM_TOLERANCE:
        raise PortfolioConfigError(f"profiles.{name}.neutral must sum to 1.0, got {total:.6f}")
    # A zero-weight asset in the neutral mix can never be allocated — the tilt is
    # multiplicative (``neutral · exp(...)``), so a zero prior stays zero however
    # attractive the asset looks. Naming one is a mistake; omit it instead.
    for asset, weight in neutral.items():
        if weight <= 0.0:
            raise PortfolioConfigError(
                f"profiles.{name}.neutral.{asset} must be > 0; an asset with zero strategic "
                "weight can never be allocated (the tilt is multiplicative) — omit it instead"
            )
    # Absorb the sub-1e-6 rounding the check above tolerates, so every downstream
    # allocation sums to *exactly* 1.0. Without this a neutral mix off by 1e-7
    # loads fine and then fails ``guardrails.assert_sums_to_one`` (which checks at
    # 1e-9) at allocation time — and ``jobs.daily.build_portfolio`` swallows that,
    # silently dropping the whole night's portfolio_state for every profile.
    neutral = {asset: weight / total for asset, weight in neutral.items()}

    caps = _parse_weight_map(raw.get("caps"), f"profiles.{name}.caps")
    # Every allocatable asset needs a cap, and the cap can never sit below the
    # neutral weight — the fallback holds the asset there.
    for asset, weight in neutral.items():
        if asset not in caps:
            raise PortfolioConfigError(
                f"profiles.{name}.caps is missing a cap for {asset!r}, which is in the neutral mix"
            )
        if caps[asset] + _SUM_TOLERANCE < weight:
            raise PortfolioConfigError(
                f"profiles.{name}.caps.{asset}={caps[asset]} is below its neutral weight "
                f"{weight}; the stale-engine fallback would violate its own cap"
            )
    if caps.get("crypto", 0.0) != 0.0:
        raise PortfolioConfigError(
            f"profiles.{name}.caps.crypto must be 0.0 — the experimental engine is excluded "
            "from allocations (01-target-architecture.md §3 rule 5)"
        )

    tilt = _require_number(raw.get("tilt_strength", 1.0), f"profiles.{name}.tilt_strength")
    if tilt < 0.0:
        raise PortfolioConfigError(f"profiles.{name}.tilt_strength must be >= 0, got {tilt}")

    return ProfileConfig(name=name, neutral=neutral, caps=caps, tilt_strength=tilt)


def _parse_distribution(raw: object) -> DistributionConfig:
    block = raw if isinstance(raw, dict) else {}
    samples = int(_require_number(block.get("samples", 1000), "distribution.samples"))
    if samples < 1:
        raise PortfolioConfigError(f"distribution.samples must be >= 1, got {samples}")
    seed = int(_require_number(block.get("seed", 0), "distribution.seed"))
    return_sd = _require_number(block.get("return_sd", 0.04), "distribution.return_sd")
    if return_sd < 0.0:
        raise PortfolioConfigError(f"distribution.return_sd must be >= 0, got {return_sd}")

    quantiles_raw = block.get("quantiles") or [0.05, 0.25, 0.5, 0.75, 0.95]
    if not isinstance(quantiles_raw, list) or not quantiles_raw:
        raise PortfolioConfigError("distribution.quantiles must be a non-empty list")
    quantiles = tuple(_require_number(q, "distribution.quantiles[]") for q in quantiles_raw)
    for q in quantiles:
        if not 0.0 <= q <= 1.0:
            raise PortfolioConfigError(f"distribution.quantiles entries must be in [0,1], got {q}")
    if list(quantiles) != sorted(quantiles):
        raise PortfolioConfigError("distribution.quantiles must be sorted ascending")
    return DistributionConfig(samples=samples, seed=seed, return_sd=return_sd, quantiles=quantiles)


def _parse_implications(raw: object) -> ImplicationConfig:
    block = raw if isinstance(raw, dict) else {}
    bands = block.get("bands") if isinstance(block.get("bands"), dict) else {}
    modest = _require_number(bands.get("modest", 0.03), "implications.bands.modest")
    pronounced = _require_number(bands.get("pronounced", 0.10), "implications.bands.pronounced")
    if not 0.0 < modest < pronounced:
        raise PortfolioConfigError(
            "implications.bands must satisfy 0 < modest < pronounced, "
            f"got modest={modest} pronounced={pronounced}"
        )

    templates_raw = block.get("templates")
    if not isinstance(templates_raw, dict):
        raise PortfolioConfigError("implications.templates must be a mapping")
    required = {"neutral", "modest", "pronounced"}
    missing = required - set(templates_raw)
    if missing:
        raise PortfolioConfigError(f"implications.templates missing {sorted(missing)}")
    templates = {k: str(v).strip() for k, v in templates_raw.items()}

    return ImplicationConfig(
        modest_band=modest,
        pronounced_band=pronounced,
        templates=templates,
        degraded_suffix=str(block.get("degraded_suffix", "")).strip(),
        disclaimer=str(block.get("disclaimer", "")).strip(),
    )


def load_portfolio_config(path: Path | None = None) -> PortfolioConfig:
    """Parse and validate portfolio.yaml. Raises on any violation."""
    config_path = path or PORTFOLIO_CONFIG_PATH
    if not config_path.exists():
        raise PortfolioConfigError(f"portfolio config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise PortfolioConfigError("portfolio.yaml must contain a mapping at the top level")

    profiles_raw = raw.get("profiles")
    if not isinstance(profiles_raw, dict) or not profiles_raw:
        raise PortfolioConfigError("portfolio.yaml must define at least one profile")
    profiles = {name: _parse_profile(name, entry) for name, entry in profiles_raw.items()}

    risk = raw.get("risk") if isinstance(raw.get("risk"), dict) else {}
    sigma_floor = _require_number(risk.get("sigma_floor", 0.02), "risk.sigma_floor")
    sigma_scale = _require_number(risk.get("sigma_scale", 0.30), "risk.sigma_scale")
    if sigma_floor <= 0.0 or sigma_scale <= 0.0:
        raise PortfolioConfigError("risk.sigma_floor and risk.sigma_scale must be positive")

    max_age = int(_require_number(raw.get("max_state_age_days", 5), "max_state_age_days"))
    if max_age < 0:
        raise PortfolioConfigError(f"max_state_age_days must be >= 0, got {max_age}")

    labels_raw = raw.get("labels") if isinstance(raw.get("labels"), dict) else {}
    labels = {str(k): str(v) for k, v in labels_raw.items()}

    return PortfolioConfig(
        model_version=str(raw.get("model_version") or "portfolio-unversioned"),
        distribution=_parse_distribution(raw.get("distribution")),
        sigma_floor=sigma_floor,
        sigma_scale=sigma_scale,
        max_state_age_days=max_age,
        profiles=profiles,
        implications=_parse_implications(raw.get("implications")),
        labels=labels,
    )


@lru_cache(maxsize=1)
def get_portfolio_config() -> PortfolioConfig:
    """Cached accessor for the shipped portfolio config."""
    return load_portfolio_config()
