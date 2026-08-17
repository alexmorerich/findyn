"""Templated conditional-implication text (Phase 6, FINDYN_V1_SPEC.md §12).

Reviewed copy, filled from the allocation — never model-generated. §12 fixes the
mechanism for the equity engine (strings templated per regime × crash band); the
portfolio layer templates on the *tilt*: which asset the distribution leans
toward, which it leans away from, and how far.

The wording is conditional throughout — "may consider", never "should" — because
the output is a distribution and an implication, not an instruction. The §12
disclaimer travels on every string this module produces.
"""

from __future__ import annotations

from collections.abc import Mapping

from findynamics.portfolio.config import ImplicationConfig, PortfolioConfig, ProfileConfig


def _band(magnitude: float, config: ImplicationConfig) -> str:
    """Which template a tilt of this magnitude selects."""
    if magnitude >= config.pronounced_band:
        return "pronounced"
    if magnitude >= config.modest_band:
        return "modest"
    return "neutral"


def render_implication(
    *,
    profile: ProfileConfig,
    bands: Mapping[str, tuple[float, float]],
    degraded: bool,
    config: PortfolioConfig,
) -> str:
    """Build the conditional-implication string for one allocation.

    ``bands`` maps each asset to ``(mean_weight, neutral_weight)``. The template
    is chosen by the largest absolute deviation from neutral across the universe;
    ``{over}`` and ``{under}`` are filled with the plain-language names of the
    most over- and under-weighted assets. A degraded run appends the standing
    caveat and always carries the disclaimer.
    """
    impl = config.implications

    deltas = {asset: mean - neutral for asset, (mean, neutral) in bands.items()}
    if not deltas:
        text = impl.templates["neutral"]
        return _finalize(text, impl, degraded)

    over = max(deltas, key=lambda a: deltas[a])
    under = min(deltas, key=lambda a: deltas[a])
    magnitude = max(abs(deltas[over]), abs(deltas[under]))

    band = _band(magnitude, config.implications)
    template = impl.templates[band]
    text = template.format(over=config.label(over), under=config.label(under))
    return _finalize(text, impl, degraded)


def _finalize(text: str, impl: ImplicationConfig, degraded: bool) -> str:
    """Attach the degraded caveat (when relevant) and the §12 disclaimer.

    Assembled here rather than in the templates so the disclaimer cannot be
    forgotten on one sentence and present on another — a single seam, the way
    ``serving/src/lib/responses.ts`` appends the experimental disclaimer.
    """
    parts = [text.strip()]
    if degraded and impl.degraded_suffix:
        parts.append(impl.degraded_suffix)
    if impl.disclaimer:
        parts.append(impl.disclaimer)
    return " ".join(part for part in parts if part)
