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

## 4b. Cross-engine outputs — `ENGINE:` series (P2)

P2 is the first phase where one engine needs another's work: FinMoney cannot
discount past a year without FinRates' fitted curve. The independence rule
(§3 rule 2) forbids the import, and handing over the in-memory result would be
worse — engines run in registry order, so the producer may not have run yet, and
a hidden ordering dependency between two supposedly independent engines is
exactly what the rule exists to prevent.

So the coupling is made explicit and turned into **data**. An engine's
`engine_output` rows are readable back as an ordinary series under a reserved id:

```
ENGINE:<asset>.<metric>        e.g. ENGINE:rates.ns_level
```

* The id vocabulary lives in `core/contracts/vocab.py`
  (`ENGINE_SERIES_PREFIX`, `engine_series_id`, `parse_engine_series_id`).
* The `engine_output` **provider** (`data/providers/published.py`) fetches them
  from the serving plane's own `/assets/:asset/history`, so they arrive through
  `pit_join` like any observation and are subject to the same release-date
  filter. A consumer physically cannot see an output published after its cutoff.
* `written_at` is the row's release date — a real vintage, not a synthesized
  lag. `data/vintages.py` skips its release-date repair for this provider
  (`AUTHORITATIVE_RELEASE_PROVIDERS`), because a daily run republishing a
  five-year window *is* the "bulk seeding" pattern that heuristic looks for, and
  here that pattern is the truth.
* Consumers declare the ids as ordinary roles in `series.yaml`, and **must
  degrade** when they are absent: the first run of a system has no published
  outputs at all. FinMoney falls back to a flat short rate and says so through a
  `curve_source_degraded` signal and a confidence penalty.

The producer owes consumers everything needed to interpret the metric. FinRates
therefore publishes `ns_lambda` per date alongside the betas — three of the four
numbers cannot be turned back into a curve.

CI enforces the direction with a named contract (§5), so a violation reports
"money imported rates" rather than a generic independence failure.

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
name = "Money reads the rates curve as data, never as code"
type = "forbidden"
source_modules = ["findynamics.engines.money"]
forbidden_modules = ["findynamics.engines.rates"]

[[tool.importlinter.contracts]]
name = "Gold reads the equity instability index as data, never as code"
type = "forbidden"
source_modules = ["findynamics.engines.gold"]
forbidden_modules = ["findynamics.engines.equity"]

[[tool.importlinter.contracts]]
name = "Crypto is quarantined"
type = "forbidden"
source_modules = ["findynamics.portfolio", "findynamics.factors", "findynamics.core", "findynamics.data"]
forbidden_modules = ["findynamics.engines.crypto"]
```

`lint-imports` joins ruff/pytest in CI (`ci.yml` compute job).

## 6. Write-back payloads (serving `admin/writeback.ts`)

New batch types under the existing HMAC envelope, mirroring the D1 tables in
`01-target-architecture.md` §6. They are **named arrays on the one envelope**,
alongside the M1-A batches that were already there:

```jsonc
{
  "model_version": "rates-1.0.0",
  "generated_at": "2026-07-30T03:00:00Z",
  "as_of": "2026-07-29",

  "factors":      [ { "force": "real_rate", "as_of": "2026-07-29", "score": 41.2,
                      "components": { "FRED:DGS10": 38.0 } } ],
  "asset_state":  [ { "asset": "rates", "as_of": "2026-07-29",
                      "model_version": "rates-1.0.0", "regime": "re_steepening",
                      "expected_return": 0.0504, "risk_score": 38.16, "confidence": 0.65,
                      "signals": [ { "name": "curve_inversion", "value": 0.84,
                                     "direction": 1, "note": "..." } ],
                      "components": { "ns_level": 5.13 } } ],
  "engine_output":[ { "asset": "rates", "metric": "ns_level",
                      "as_of": "2026-07-29", "value": 5.13, "meta": null } ]
}
```

> **Superseded (P1).** This section originally sketched one envelope per batch
> — `{ "kind": "asset_state", "rows": [...] }`. The implementation carries them
> as named arrays instead, because the envelope already had `metadata` /
> `observations` / `quality` / `ingestion` in exactly that shape: a `kind`
> discriminator would have meant two payload formats behind one signature, and
> a daily run that publishes factors, a state and its outputs would have needed
> three signed requests where one does. The named-array form above is canonical.

Validation mirrors the existing write-back style: reject unknown assets and
unknown factors (vocabulary in `domain.ts`), reject non-finite numbers and
out-of-range scores, reject a `direction` outside `-1|0|1`, and upsert
idempotently on the primary key. `model_version` is part of the `asset_state`
key, so a refit publishes alongside the model it replaces rather than
overwriting it.

**Chunking.** A historical backfill is hundreds of thousands of observations;
sent as one request it exceeds the Worker's CPU budget partway through and
leaves a partial write with no record of where it stopped. `jobs/backfill.py`
splits on `observations` (`--batch-size`, default 5000) and every chunk is
independently idempotent. Per-series rows ride with the first chunk only.

The daily run hit the same wall from the other direction once a second engine
shipped. Every engine republishes a multi-year window of every metric it charts,
so `engine_output` grows with the number of **enabled engines**, not with the
day's news: P1 alone sent ~7.5k rows a night, P2 sends ~20k, and P3–P6 keep
multiplying it. `jobs/daily.py` splits on `engine_output` the same way. The
mechanics are shared in `jobs/_common.py::chunk_on`; states and factor scores are
per-run rather than per-row, so they ride with the first chunk.

**Staleness of an `AssetState` is measured in market days, not ingestion hours.**
`isStale` (36 hours, `domain.ts`) answers "when did data last arrive", which is
the right question for `/health` and the wrong one for an engine: an `as_of` is a
market date, so it is a day old the moment it is published and three days old
every Monday. The asset endpoints use `isAssetStale` (`ASSET_STALE_DAYS`, the
same rule `/assets` already applied), so the two endpoints cannot give opposite
answers about the same row.
