-- FinDynamics P3-C — forecast_distribution becomes keyed by asset
--
-- The third and last of the v1 equity-private tables to be brought in line with
-- 0004's multi-asset model, for the same reason as 0005 and 0006: `tactical` and
-- `0.5` are not equity's words. FinGold will publish a tactical p50 too, and
-- under the old key (as_of, horizon, quantile, model_version) it would land on
-- the same row and win by whichever engine ran second.
--
-- Two engines silently overwriting each other's forecasts is not a failure any
-- test catches — both rows are valid, the count is right, and the number is
-- simply the wrong asset's. Done now, while the table is still empty.
--
-- Note what is *not* added: a column for a point estimate. §0's first non-goal
-- is "no deterministic price target", and a schema with no place to put one
-- cannot acquire one by someone deciding it would be convenient.

CREATE TABLE forecast_distribution_v2 (
  asset            TEXT NOT NULL,       -- 'equity' | 'gold' | ... (engine name)
  as_of            TEXT NOT NULL,
  horizon          TEXT NOT NULL,       -- tactical|strategic|generational|educational_30y|educational_50y
  quantile         REAL NOT NULL,       -- 0.05,0.10,0.25,0.50,0.75,0.90,0.95
  value            REAL NOT NULL,       -- projected log index level
  educational_only INTEGER NOT NULL DEFAULT 0,
  model_version    TEXT NOT NULL,
  PRIMARY KEY (asset, as_of, horizon, quantile, model_version)
);

INSERT INTO forecast_distribution_v2
       (asset, as_of, horizon, quantile, value, educational_only, model_version)
SELECT 'equity', as_of, horizon, quantile, value, educational_only, model_version
  FROM forecast_distribution;

DROP TABLE forecast_distribution;
ALTER TABLE forecast_distribution_v2 RENAME TO forecast_distribution;

-- /forecast resolves the newest (as_of, model_version) for one asset, then reads
-- every band under it. Both halves of that are this index.
CREATE INDEX idx_forecast_asset_date ON forecast_distribution (asset, as_of DESC, model_version);
