-- FinDynamics P1 — multi-asset output tables
-- (docs/redesign/01-target-architecture.md §6)
--
-- Additive only. The v1 S&P500 tables (derived_features, force_scores,
-- regime_state, instability_index, forecast_distribution) keep their shape and
-- become the equity engine's private outputs; nothing here rewrites them, so no
-- existing query can break.
--
-- Two new tables, because engines produce two different kinds of thing:
--
--   asset_state    one row per engine per run — the AssetState the portfolio
--                  layer consumes. Narrow, one row is a complete answer.
--   engine_output  per-date wide metrics an engine publishes for charting
--                  (NS level/slope/curvature, carry, ...). Tall and skinny, keyed
--                  by metric name so a new metric never needs a migration.

CREATE TABLE asset_state (
  asset           TEXT NOT NULL,      -- 'money' | 'rates' | 'equity' | 'gold' | 'crypto'
  as_of           TEXT NOT NULL,      -- information-set date the state describes
  model_version   TEXT NOT NULL,
  regime          TEXT NOT NULL,      -- engine-defined vocabulary
  expected_return REAL,               -- annualized decimal; NULL where meaningless
  risk_score      REAL,               -- 0-100
  confidence      REAL,               -- 0-1
  signals         TEXT NOT NULL,      -- JSON array of Signal
  components      TEXT,               -- JSON explanation trace
  written_at      TEXT NOT NULL,
  -- model_version is in the key on purpose: a refit that changes the model must
  -- not overwrite the state the previous model published for the same date, or
  -- every backtest of the old model silently becomes a backtest of the new one.
  PRIMARY KEY (asset, as_of, model_version)
);

-- The dashboard's usual read: newest state per asset.
CREATE INDEX idx_asset_state_latest ON asset_state (asset, as_of DESC);

CREATE TABLE engine_output (
  asset      TEXT NOT NULL,
  metric     TEXT NOT NULL,           -- 'ns_level' | 'ns_slope' | 'regime_code' | ...
  as_of      TEXT NOT NULL,
  value      REAL NOT NULL,
  meta       TEXT,                    -- JSON; carries the label for coded metrics
  written_at TEXT NOT NULL,
  PRIMARY KEY (asset, metric, as_of)
);

-- History reads are always (asset, metric) over a date range.
CREATE INDEX idx_engine_output_series ON engine_output (asset, metric, as_of);

-- The v1 output tables predate multi-asset and are implicitly about the S&P500.
-- Naming that explicitly costs one column each and means a later engine can
-- write into them without ambiguity. Existing rows are SPX by definition.
ALTER TABLE derived_features      ADD COLUMN asset TEXT NOT NULL DEFAULT 'SPX';
ALTER TABLE force_scores          ADD COLUMN asset TEXT NOT NULL DEFAULT 'SPX';
ALTER TABLE regime_state          ADD COLUMN asset TEXT NOT NULL DEFAULT 'SPX';
ALTER TABLE instability_index     ADD COLUMN asset TEXT NOT NULL DEFAULT 'SPX';
ALTER TABLE forecast_distribution ADD COLUMN asset TEXT NOT NULL DEFAULT 'SPX';
