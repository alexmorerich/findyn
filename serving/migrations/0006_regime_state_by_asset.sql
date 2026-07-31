-- FinDynamics P3-B — regime_state becomes keyed by asset
--
-- Same reasoning as 0005 did for derived_features, and the same reason it could
-- not be done in 0004: SQLite cannot extend a primary key in place. The key is
-- (date, regime, model_version), and `bear` is exactly the kind of name a second
-- engine would also publish — FinGold's regime vocabulary is not written yet,
-- and when it is, a shared `crisis` row would silently overwrite equity's.
--
-- Done now, while the table is empty, rather than after there is history.

CREATE TABLE regime_state_v2 (
  asset         TEXT NOT NULL,          -- 'equity' | 'gold' | ... (engine name)
  date          TEXT NOT NULL,
  regime        TEXT NOT NULL,          -- engine-defined vocabulary
  probability   REAL NOT NULL,          -- 0-1; the row is one leg of a posterior
  model_version TEXT NOT NULL,
  PRIMARY KEY (asset, date, regime, model_version)
);

INSERT INTO regime_state_v2 (asset, date, regime, probability, model_version)
SELECT CASE WHEN asset = 'SPX' THEN 'equity' ELSE asset END,
       date, regime, probability, model_version
  FROM regime_state;

DROP TABLE regime_state;
ALTER TABLE regime_state_v2 RENAME TO regime_state;

-- /regime reads a window for one asset, oldest first.
CREATE INDEX idx_regime_state_series ON regime_state (asset, date);
