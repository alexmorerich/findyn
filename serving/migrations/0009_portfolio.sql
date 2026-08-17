-- FinDynamics P6 — the portfolio layer's output table
-- (docs/redesign/01-target-architecture.md §Portfolio layer, 04-ui-plan.md §P6)
--
-- One row per (profile, run, model_version): the strategic weight *distribution*
-- for that profile, its templated conditional implication, and whether the run
-- was degraded. Additive only — nothing here touches an engine's tables.
--
-- The weights column is a JSON blob, not a set of columns, on purpose: it holds
-- a distribution (quantiles per asset) plus the neutral mix, the input
-- AssetState references and the risk-free rate, and every one of those grows or
-- shrinks with the profile's universe. A wide schema would need a migration each
-- time a profile gains an asset; the engines' `engine_output` avoids exactly this
-- by keying on a metric name, and this avoids it by carrying one document.

CREATE TABLE portfolio_state (
  profile        TEXT NOT NULL,        -- 'conservative' | 'balanced' | 'growth'
  as_of          TEXT NOT NULL,        -- information-set date the allocation describes
  model_version  TEXT NOT NULL,        -- 'portfolio-1.0.0'
  weights        TEXT NOT NULL,        -- JSON: quantiles per asset, neutral mix, inputs, risk-free
  implication    TEXT NOT NULL,        -- templated conditional-implication string (§12)
  degraded       INTEGER NOT NULL,     -- 1 when any input engine was missing/stale
  written_at     TEXT NOT NULL,
  -- model_version is in the key for the same reason it is on asset_state: a
  -- refit publishes alongside the model it replaces rather than overwriting the
  -- allocation the previous model produced for the same date.
  PRIMARY KEY (profile, as_of, model_version)
);

-- The dashboard's read: newest allocation per profile.
CREATE INDEX idx_portfolio_state_latest ON portfolio_state (profile, as_of DESC);
