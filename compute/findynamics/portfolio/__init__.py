"""Portfolio layer — Phase 6.

Consumes the latest ``AssetState`` per registered engine plus ``WorldState`` and
produces target-weight *distributions*. It reaches engines only through
``core.registry``, never through engine internals
(``01-target-architecture.md`` §3 rule 3), and excludes experimental engines
unless explicitly configured in.

That last exclusion is not left to P6's good intentions. P5 shipped an
experimental engine, so the rule became real before the layer that has to obey
it existed: :func:`findynamics.core.registry.portfolio_engines` is the accessor
this layer will use, and it filters ``experimental`` engines out by default.
``lint-imports`` separately forbids this package from importing
``findynamics.engines.crypto`` at all, so the filter cannot be bypassed by
reaching for the class directly.
"""

from findynamics.core.registry import portfolio_engines

__all__ = ["portfolio_engines"]
