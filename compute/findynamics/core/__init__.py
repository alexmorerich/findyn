"""Core layer — contracts, registries and configuration.

The bottom of the dependency stack: ``core`` imports nothing from ``data``,
``factors``, ``engines`` or ``portfolio`` (``01-target-architecture.md`` §3
rule 1). Everything above it depends on the vocabulary and interfaces defined
here, which is what keeps the engines mutually ignorant.
"""
