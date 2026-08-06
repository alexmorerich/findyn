"""FinCrypto — network scarcity. Phase 5, **experimental and research-only**.

Bitcoin has no cash flow, no issuer and about fifteen years of tradeable
history, so this engine publishes **no expected return at all** — the field is
``None`` by design, not by omission. What it does publish is a vol/drawdown
regime, a 0-100 speculation index, an expanding-window liquidity beta, a jump
intensity and the deterministic supply schedule. ``confidence`` is capped at 0.5
by construction. ``engines/crypto/engine.py`` opens with the reasoning behind
each of those; read it before consuming anything from here.

Quarantined three ways, deliberately redundantly:

* nothing outside this package may import it (``01-target-architecture.md`` §3
  rule 5, enforced by the ``Crypto is quarantined`` contract in
  ``compute/pyproject.toml``);
* ``experimental = True``, which is what ``core.registry.portfolio_engines``
  filters on, so the portfolio layer excludes it unless explicitly configured in;
* ``config/engines/crypto.yaml`` ships ``enabled: false``.

Importing this package registers :class:`CryptoEngine` under the name ``crypto``.
"""

from findynamics.engines.crypto.engine import CryptoEngine

__all__ = ["CryptoEngine"]
