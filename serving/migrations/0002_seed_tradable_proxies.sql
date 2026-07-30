-- FinDyn v1.0 — tradable proxy seed (FINDYN_V1_SPEC.md §7, §12)
-- The engine ANALYZES SP500. The output layer maps that analysis to
-- jurisdiction-appropriate tradable instruments. This is a mapping table only:
-- it never implies a recommendation to trade any of these.

INSERT INTO tradable_proxy_mapping (analysis_asset, jurisdiction, tradable_ticker, name, notes) VALUES
  ('SP500', 'EU_UCITS', 'SPYL', 'SPDR S&P 500 UCITS ETF (Acc)',        'Accumulating, IE domicile'),
  ('SP500', 'EU_UCITS', 'CSPX', 'iShares Core S&P 500 UCITS ETF (Acc)', 'Accumulating, IE domicile'),
  ('SP500', 'EU_UCITS', 'VUAA', 'Vanguard S&P 500 UCITS ETF (Acc)',     'Accumulating, IE domicile'),
  ('SP500', 'US',       'SPY',  'SPDR S&P 500 ETF Trust',               'US-domiciled, distributing'),
  ('SP500', 'US',       'VOO',  'Vanguard S&P 500 ETF',                 'US-domiciled, distributing'),
  ('SP500', 'LATAM_DEFAULT', 'CSPX', 'iShares Core S&P 500 UCITS ETF (Acc)', 'UCITS default for LatAm investors via international brokers; verify local tax treatment'),
  ('SP500', 'LATAM_DEFAULT', 'SPYL', 'SPDR S&P 500 UCITS ETF (Acc)',         'UCITS default for LatAm investors via international brokers; verify local tax treatment');
