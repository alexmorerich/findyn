"""Data layer — providers, point-in-time joins and quality control.

Everything an engine is ever allowed to read passes through here, and everything
that leaves does so as of an explicit information-set cutoff
(:class:`findynamics.core.contracts.pit.PITAccessor`).
"""
