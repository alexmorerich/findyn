"""FinDynamics compute plane.

A financial physics framework: world state → risk factors → asset engines →
portfolio. The layer rules of ``docs/redesign/01-target-architecture.md`` §3 are
enforced in CI by import-linter; the no-lookahead law of FINDYN_V1_SPEC.md §14.1
is enforced structurally — engines read data only through ``WorldState.series``.

This package runs outside Cloudflare (Workers cannot host the scientific Python
stack, §6). Results reach D1 through the HMAC-signed admin write-back endpoint.
"""

__version__ = "1.0.0"
