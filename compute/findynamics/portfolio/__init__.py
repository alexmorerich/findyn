"""Portfolio layer — Phase 6.

Consumes the latest ``AssetState`` per registered engine plus ``WorldState`` and
produces target-weight *distributions*. It reaches engines only through
``core.registry``, never through engine internals
(``01-target-architecture.md`` §3 rule 3), and excludes experimental engines
unless explicitly configured in.
"""
