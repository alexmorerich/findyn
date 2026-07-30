-- FinDyn v1.0 — core schema
-- Source of truth: FINDYN_V1_SPEC.md §7. Do not alter table/column names.
--
-- Point-in-time contract (§14.1): every macro observation carries `release_date`,
-- the first date the value was publicly known. All reads for feature computation
-- must go through compute/findyn/pit.py::pit_join — never query macro_series directly.

CREATE TABLE market_price (
  date        TEXT NOT NULL,            -- observation date YYYY-MM-DD
  symbol      TEXT NOT NULL,            -- 'SP500', 'SPY', ...
  close       REAL NOT NULL,
  volume      REAL,
  source      TEXT NOT NULL,            -- provider id
  ingested_at TEXT NOT NULL,
  PRIMARY KEY (date, symbol, source)
);
CREATE INDEX idx_market_price_symbol_date ON market_price (symbol, date);

CREATE TABLE macro_series (
  series_id    TEXT NOT NULL,           -- e.g. 'FRED:CPIAUCSL'
  obs_date     TEXT NOT NULL,           -- period the value describes
  release_date TEXT NOT NULL,           -- POINT-IN-TIME KEY: first public availability
  value        REAL NOT NULL,
  vintage      TEXT,                    -- ALFRED vintage when available
  source       TEXT NOT NULL,
  ingested_at  TEXT NOT NULL,
  PRIMARY KEY (series_id, obs_date, release_date)
);
CREATE INDEX idx_macro_pit ON macro_series (series_id, release_date);

CREATE TABLE derived_features (
  date          TEXT NOT NULL,
  feature       TEXT NOT NULL,          -- 'price_filtered','velocity','acceleration','jerk_z','ffd_price', ...
  value         REAL NOT NULL,
  model_version TEXT NOT NULL,
  computed_at   TEXT NOT NULL,
  PRIMARY KEY (date, feature, model_version)
);
CREATE INDEX idx_derived_features_feature_date ON derived_features (feature, date);

CREATE TABLE force_scores (
  date          TEXT NOT NULL,
  force         TEXT NOT NULL,          -- valuation|earnings|liquidity|rates|credit|inflation|labor|risk_appetite|sentiment
  score         REAL NOT NULL,          -- 0-100 percentile, point-in-time
  components    TEXT,                   -- JSON breakdown for explainability (§14.3)
  model_version TEXT NOT NULL,
  PRIMARY KEY (date, force, model_version)
);
CREATE INDEX idx_force_scores_date ON force_scores (date);

CREATE TABLE regime_state (
  date          TEXT NOT NULL,
  regime        TEXT NOT NULL,          -- bull_expansion|normal_expansion|late_cycle|bear|crisis
  probability   REAL NOT NULL,
  model_version TEXT NOT NULL,
  PRIMARY KEY (date, regime, model_version)
);
CREATE INDEX idx_regime_state_date ON regime_state (date);

CREATE TABLE instability_index (
  date           TEXT NOT NULL,
  rii            REAL NOT NULL,         -- 0-100 Regime Instability Index (§3.2)
  p_transition   REAL,                  -- crash decomposition factors (§4)
  p_shock        REAL,
  p_transmission REAL,
  crash_risk     REAL,                  -- 0-100 composite
  components     TEXT,                  -- JSON
  model_version  TEXT NOT NULL,
  PRIMARY KEY (date, model_version)
);

CREATE TABLE forecast_distribution (
  as_of            TEXT NOT NULL,
  horizon          TEXT NOT NULL,       -- tactical|strategic|generational|educational_30y|educational_50y
  quantile         REAL NOT NULL,       -- 0.05,0.10,0.25,0.50,0.75,0.90,0.95
  value            REAL NOT NULL,       -- projected real log index level
  educational_only INTEGER NOT NULL DEFAULT 0,
  model_version    TEXT NOT NULL,
  PRIMARY KEY (as_of, horizon, quantile, model_version)
);

CREATE TABLE tradable_proxy_mapping (
  analysis_asset  TEXT NOT NULL,        -- 'SP500'
  jurisdiction    TEXT NOT NULL,        -- 'EU_UCITS','US','LATAM_DEFAULT'
  tradable_ticker TEXT NOT NULL,        -- 'SPYL','CSPX','VUAA','SPY','VOO'
  name            TEXT NOT NULL,
  notes           TEXT,
  PRIMARY KEY (analysis_asset, jurisdiction, tradable_ticker)
);

CREATE TABLE ingestion_log (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  run_at       TEXT NOT NULL,
  source       TEXT NOT NULL,
  series_id    TEXT,
  status       TEXT NOT NULL,           -- ok|degraded|failed
  rows_written INTEGER DEFAULT 0,
  error        TEXT
);
CREATE INDEX idx_ingestion_log_run_at ON ingestion_log (run_at DESC);
CREATE INDEX idx_ingestion_log_source ON ingestion_log (source, run_at DESC);
