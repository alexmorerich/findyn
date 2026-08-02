-- FinDynamics — observation-level quality flags on macro_series
--
-- The quality engine used to make one decision per series: ingest it, or refuse
-- it. That works for defects that are properties of the series — a unit that
-- disagrees with the metadata, a frequency that does not match the spacing, a
-- coverage hole — because none of those leave a trustworthy subset behind.
--
-- It is wrong for `abnormal_jump`, which is a property of one observation. The
-- first full FRED backfill refused fifteen series on it, and after the jump test
-- itself was corrected three remained: initial claims in March 2020, the
-- unemployment rate in April 2020, and the St. Louis financial stress index in
-- September 2008. All three were correct. All three are precisely the history a
-- system that models crises exists to hold.
--
-- So the flag moves onto the row. The observation is stored, served, and carries
-- what the quality engine thought of it; a consumer that wants to exclude or
-- down-weight it has what it needs, and one that wants September 2008 gets
-- September 2008. Nothing is silently dropped, and nothing is silently trusted.
--
-- Nullable with no default: NULL means "the quality engine had nothing to say",
-- which is what every row written before this migration means too. There is no
-- backfill to run — the flag is re-derived on every ingestion, so it appears on
-- the next run of whatever produced the row.

ALTER TABLE macro_series ADD COLUMN quality_flag TEXT;

-- Anomalous rows are rare and asked for by exception ("show me what was flagged"),
-- never scanned in bulk, so this is a partial index over just the flagged rows.
CREATE INDEX idx_macro_quality_flag
  ON macro_series (series_id, obs_date)
  WHERE quality_flag IS NOT NULL;
