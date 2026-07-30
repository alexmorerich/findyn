"""FinDyn v1.0 compute plane.

Feature engineering, state estimation, regime detection and simulation for the
S&P500 Dynamic State Engine. Specification: FINDYN_V1_SPEC.md.

This package runs outside Cloudflare (Workers cannot host the scientific Python
stack, §6). Results reach D1 through the HMAC-signed admin write-back endpoint.
"""

__version__ = "1.0.0"
