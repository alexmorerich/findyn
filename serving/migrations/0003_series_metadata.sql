-- FinDyn M1-A — canonical series model
--
-- The normalized record is (series_id, provider, frequency, unit,
-- observation_date, release_date, revision_date, value). Per-observation
-- columns live on macro_series; the descriptive half is factored into
-- series_metadata so a unit or frequency is stated once rather than repeated
-- on every row and allowed to disagree with itself.

CREATE TABLE series_metadata (
  series_id           TEXT PRIMARY KEY,   -- 'FRED:CPIAUCSL', 'SHILLER:CAPE'
  provider            TEXT NOT NULL,      -- fred | shiller | stooq | bls | bea | yahoo | mock
  title               TEXT NOT NULL,
  frequency           TEXT NOT NULL,      -- daily | weekly | monthly | quarterly
  unit                TEXT NOT NULL,      -- 'index', 'percent', 'usd', 'ratio', ...
  seasonal_adjustment TEXT,
  first_observation   TEXT,
  last_observation    TEXT,
  notes               TEXT,
  updated_at          TEXT NOT NULL
);

-- macro_series is rebuilt rather than altered, because adding revision_date
-- changes the primary key.
--
-- release_date is when a period FIRST became public; revision_date is when this
-- particular figure was issued. FRED reissues a period under the same
-- release_date, so a key of (series_id, obs_date, release_date) silently
-- collapses every revision onto the original print — the second write upserts
-- over the first and the revision history disappears. Keying on revision_date
-- keeps each vintage, which is what makes an as-of query answerable.
--
-- Sources that publish no vintage set revision_date = release_date, so the key
-- degenerates to the previous behaviour for them.
CREATE TABLE macro_series_v2 (
  series_id     TEXT NOT NULL,
  obs_date      TEXT NOT NULL,           -- period the value describes
  release_date  TEXT NOT NULL,           -- first public availability of this period
  revision_date TEXT NOT NULL,           -- when THIS figure was issued
  value         REAL NOT NULL,
  vintage       TEXT,
  source        TEXT NOT NULL,
  ingested_at   TEXT NOT NULL,
  PRIMARY KEY (series_id, obs_date, revision_date)
);

INSERT INTO macro_series_v2
  (series_id, obs_date, release_date, revision_date, value, vintage, source, ingested_at)
SELECT series_id, obs_date, release_date, release_date, value, vintage, source, ingested_at
  FROM macro_series;

DROP TABLE macro_series;
ALTER TABLE macro_series_v2 RENAME TO macro_series;

-- Point-in-time reads filter on revision_date and scan by series.
CREATE INDEX idx_macro_pit ON macro_series (series_id, revision_date);
CREATE INDEX idx_macro_release ON macro_series (series_id, release_date);

-- Data quality outcomes are persisted, not just logged: a silent downgrade in
-- source quality is the kind of thing that only becomes visible in hindsight.
CREATE TABLE data_quality_report (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  run_at          TEXT NOT NULL,
  series_id       TEXT NOT NULL,
  provider        TEXT NOT NULL,
  status          TEXT NOT NULL,          -- ok | warning | error
  observations    INTEGER NOT NULL DEFAULT 0,
  warnings        TEXT,                   -- JSON array of {code, message, context}
  errors          TEXT,                   -- JSON array of {code, message, context}
  checked_range   TEXT                    -- 'YYYY-MM-DD..YYYY-MM-DD'
);

CREATE INDEX idx_dq_series ON data_quality_report (series_id, run_at DESC);
CREATE INDEX idx_dq_run ON data_quality_report (run_at DESC);
