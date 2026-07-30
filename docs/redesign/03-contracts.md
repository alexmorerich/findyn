# Core Contracts — specification

These are the interfaces P0 creates in `findynamics/core/`. They are the only
framework code allowed before an engine exists ("do not build the framework
first — extract it from real usage"). Field lists here are normative; method
bodies are the coder's job.

## 1. `core/contracts/state.py`

```python
@dataclass(frozen=True)
class FactorState:
    name: str                 # 'rates', 'liquidity', 'real_rate', ...
    as_of: date
    score: float              # 0-100 expanding-window percentile
    components: dict[str, float]   # explanation trace, series_id → contribution

@dataclass(frozen=True)
class WorldState:
    as_of: date               # information-set cutoff (INFO_SET = t-1 convention)
    factors: dict[str, FactorState]
    # PIT accessor injected by the job layer; engines never query D1 directly.
    series: PITAccessor       # thin wrapper over data.pit.pit_join

@dataclass(frozen=True)
class Signal:
    name: str                 # 'curve_inversion', 'carry_positive', ...
    value: float
    direction: Literal[-1, 0, 1]
    note: str | None = None

@dataclass(frozen=True)
class AssetState:             # the universal engine output (FinDynamics doc §8)
    asset: str                # engine name: 'rates' | 'money' | ...
    as_of: date
    regime: str               # engine-defined vocabulary, documented per engine
    expected_return: float | None   # annualized, None where meaningless (crypto)
    risk_score: float         # 0-100
    confidence: float         # 0-1
    signals: tuple[Signal, ...]
    model_version: str
    components: dict[str, float] | None = None  # explainability trace
```

Rules:
- All dataclasses frozen; all dates are `datetime.date`; no pandas objects
  cross a contract boundary.
- `WorldState.series` is the **only** data access an engine gets — this is how
  the no-lookahead law is enforced structurally, not by convention.

## 2. `core/engine.py`

```python
class AssetEngine(ABC):
    name: ClassVar[str]           # registry key, e.g. 'rates'
    version: ClassVar[str]        # model_version stamped on outputs
    experimental: ClassVar[bool] = False   # True quarantines from portfolio

    @abstractmethod
    def required_series(self) -> tuple[str, ...]:
        """Series ids this engine needs, resolved against series.yaml."""

    @abstractmethod
    def fit(self, world: WorldState) -> None:
        """Expanding-window (re)fit. Persists parameters via its own artifact
        store handle; monthly_refit cadence."""

    @abstractmethod
    def predict(self, world: WorldState) -> AssetState:
        """Pure function of (fitted params, world). Daily cadence."""

    def outputs(self, world: WorldState) -> tuple[EngineOutput, ...]:
        """Optional wide metrics for engine_output table (NS level/slope, ...)."""
        return ()
```

`fit` and `predict` receive only `WorldState`. An engine that needs raw prices
declares them in `required_series()` and reads them through
`world.series` — never through a provider directly.

## 3. `core/registry.py`

```python
ENGINES: dict[str, type[AssetEngine]] = {}

def register_engine(cls: type[AssetEngine]) -> type[AssetEngine]: ...
def get_engine(name: str) -> AssetEngine: ...
def enabled_engines(config: Config) -> list[AssetEngine]: ...
```

- Engines self-register with `@register_engine` at import time.
- `findynamics/engines/__init__.py` imports each subpackage guarded by its
  config enable-flag; this file is the only place that names engines.
- The provider registry (moved from `providers/registry.py`) lives beside it.

## 4. Factors (`findynamics/factors/`)

- `definitions.py`: loads factor specs from `series.yaml` `factors:` block
  (id, constituent series, direction, weight). The nine v1 forces are the
  initial set; P1 adds `real_rate`, `usd_strength`, `curve` inputs.
- `compute.py`: the v1 scoring pipeline as a pure function —
  winsorized z-score → expanding-window percentile → 0–100 — consuming
  PIT frames, emitting `FactorState`. This is Layer 0; engines never
  recompute a shared factor privately.

## 5. Import-linter contracts (in `compute/pyproject.toml`)

```toml
[tool.importlinter]
root_package = "findynamics"

[[tool.importlinter.contracts]]
name = "Layers"
type = "layers"
layers = [
    "findynamics.portfolio",
    "findynamics.engines",
    "findynamics.factors | findynamics.data",
    "findynamics.core",
]

[[tool.importlinter.contracts]]
name = "Engines are independent"
type = "independence"
modules = [
    "findynamics.engines.money",
    "findynamics.engines.rates",
    "findynamics.engines.equity",
    "findynamics.engines.gold",
    "findynamics.engines.crypto",
]

[[tool.importlinter.contracts]]
name = "Crypto is quarantined"
type = "forbidden"
source_modules = ["findynamics.portfolio", "findynamics.factors", "findynamics.core", "findynamics.data"]
forbidden_modules = ["findynamics.engines.crypto"]
```

`lint-imports` joins ruff/pytest in CI (`ci.yml` compute job).

## 6. Write-back payloads (serving `admin/writeback.ts`)

Two new batch types under the existing HMAC envelope, mirroring the D1 tables
in `01-target-architecture.md` §6:

```jsonc
{ "kind": "asset_state",  "rows": [ { "asset": "rates", "as_of": "2026-07-29", ... } ] }
{ "kind": "engine_output", "rows": [ { "asset": "rates", "metric": "ns_level", "as_of": "...", "value": 4.31 } ] }
```

Validation mirrors the existing write-back style: reject unknown assets
(vocabulary in `domain.ts`), reject non-finite numbers, idempotent upserts on
the primary key.
