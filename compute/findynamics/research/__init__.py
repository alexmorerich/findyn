"""Quarantined research extensions — the topmost layer, imported by nothing.

``01-target-architecture.md`` §3 stacks the compute plane
``core → data·factors → engines → portfolio``. This package sits above all of
it, and the direction of the arrow is the whole point: research code may read
the engines, and no engine, factor, portfolio or core module may read research
code. The rule is CI-enforced by the ``Research is quarantined`` contract in
``pyproject.toml``, exactly as ``Crypto is quarantined`` is, because a rule that
lives only in a docstring is a rule the next phase deletes by accident.

Nothing here is a sixth engine. ``core/contracts/vocab.py::ASSETS`` is a closed
five-tuple mirrored in ``serving/src/domain.ts``; a research module that wanted a
registry entry would have to widen a published vocabulary, a D1 schema and a
TypeScript union to do it. Research modules therefore register nothing, write no
``asset_state`` / ``engine_output`` / ``derived_features`` row, and ship no
migration. Their output is files: CSV under ``compute/backtests/`` and markdown
under ``docs/research/``.

Current occupants:

* :mod:`findynamics.research.omega` — KK-Ω, a latent market-state coordinate and
  its dynamics, tested against the representation the equity engine already
  publishes. See ``docs/research/kk-omega-design.md``.
"""

from __future__ import annotations
