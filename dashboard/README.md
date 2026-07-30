# FinDyn Dashboard

Astro front-end for the S&P500 Dynamic State Engine. **Delivered in milestone M5** — see
[`FINDYN_V1_SPEC.md`](../FINDYN_V1_SPEC.md) §16.

Panels planned:

1. Regime probability gauge + history ribbon
2. Kinematics chart — filtered price, velocity, acceleration bands
3. Force scores (radar) with component drill-down
4. Regime Instability Index with alert thresholds
5. Crash-risk decomposition — the three factors of §4, never just the composite
6. Forecast fan charts per horizon, with educational horizons visually separated
7. Data health / staleness panel

Two constraints carry over from the spec:

- **Savitzky–Golay smoothing is permitted here and nowhere else.** It uses a centered
  window, so it looks at data after time *t*. That is fine for drawing a line and fatal
  for a feature (§8, §14.1).
- **No trading commands.** The UI renders the conditional-implication text supplied by
  the API; it never composes its own (§12).

Once built, `dashboard/dist` is served by the Worker via the `assets` block in
`serving/wrangler.jsonc`.
