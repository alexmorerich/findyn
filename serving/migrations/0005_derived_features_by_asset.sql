-- FinDynamics P3-A — derived_features becomes keyed by asset
--
-- 0004 added an `asset` column to the v1 output tables but could not add it to
-- their primary keys, because SQLite has no ALTER TABLE ... ADD CONSTRAINT. For
-- four of the five that is harmless. For derived_features it is not: the key is
-- (date, feature, model_version), and feature names like `velocity` are exactly
-- the ones a second engine would also want. Two engines publishing `velocity`
-- for the same date would upsert over each other, and the loser would look like
-- missing data rather than a collision.
--
-- FinEquity is the first writer of this table, so the rebuild is done now, while
-- it is empty, rather than after there is history to migrate.
--
-- The `asset` value is the ENGINE name ('equity'), matching asset_state and
-- engine_output. 0004's 'SPX' default described a symbol, which was the v1
-- single-asset vocabulary; any row still carrying it is mapped across.

CREATE TABLE derived_features_v2 (
  asset         TEXT NOT NULL,          -- 'equity' | 'rates' | ... (engine name)
  date          TEXT NOT NULL,
  feature       TEXT NOT NULL,          -- 'price_filtered','velocity','jerk_z','ffd_price', ...
  value         REAL NOT NULL,
  -- In the key on purpose, as in asset_state: a refit that changes the
  -- transform must land beside the features the previous model was fitted on,
  -- or every backtest of that model silently re-runs on inputs it never saw.
  model_version TEXT NOT NULL,
  computed_at   TEXT NOT NULL,
  PRIMARY KEY (asset, date, feature, model_version)
);

INSERT INTO derived_features_v2 (asset, date, feature, value, model_version, computed_at)
SELECT CASE WHEN asset = 'SPX' THEN 'equity' ELSE asset END,
       date, feature, value, model_version, computed_at
  FROM derived_features;

DROP TABLE derived_features;
ALTER TABLE derived_features_v2 RENAME TO derived_features;

-- The dashboard's read is always (asset, feature) over a date range.
CREATE INDEX idx_derived_features_series ON derived_features (asset, feature, date);
-- The /state snapshot wants the newest date per asset.
CREATE INDEX idx_derived_features_latest ON derived_features (asset, date DESC);
