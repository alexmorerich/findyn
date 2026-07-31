# P3 follow-on — deep history and a zoom control on /equity

Copy everything below this line to the coder agent. Requires P3-A merged.

---

You are working in the `findyn` repo. `/equity` charts five years and cannot show
more. D1 now holds daily S&P 500 closes back to **1927-12-30** (24,761 rows,
`YAHOO:^GSPC`) and monthly Shiller history to **1871**, so the data exists and
only the pipeline's ceilings hide it.

## The bug is not in the UI, or not only

Three independent limits sit in series. Lifting any one alone changes nothing —
diagnose before you edit, and expect to touch all three.

| # | limit | where | effect |
|---|---|---|---|
| 1 | `history_days: 1825` | `compute/config/engines/equity.yaml` (`outputs`) | **the binding one.** Only 5y of per-date features are ever written back, so `derived_features` holds 2021-07-30 → 2026-07-29 and nothing downstream can show more |
| 2 | `limit ?? 5000`, capped `20000` | `serving/src/api/*.ts` | the hard cap is **below** the 24,761 rows a full daily history needs, so the API would silently truncate even after #1 |
| 3 | `HISTORY_LIMIT = 1400` | `dashboard/src/scripts/equity.ts` | ~5.5y ceiling on what the page requests |

Verify the diagnosis before changing anything:

```bash
cd serving
npx wrangler d1 execute findyn --remote \
  --command "SELECT feature, COUNT(*) n, MIN(date) first, MAX(date) last FROM derived_features GROUP BY feature;"
```

## What to build

### 1. Republish deep history without paying for it nightly

`history_days` bounds the write-back only — the comment in `equity.yaml` is
correct that a longer window costs write time, never accuracy. But republishing
~25,000 dates × every feature on every daily run is a large nightly write for no
new information, since all but the newest rows are unchanged upserts.

Split the two cases rather than picking one number: a **backfill** window that
reaches the start of the data and runs deliberately, and the existing short
window for the nightly run. A CLI flag or a job argument is fine; what matters is
that the daily cron does not silently start writing 200k rows a night. Say in the
config comment which knob governs which run, because the next person will assume
one number does both.

### 2. Let the API serve a range it can actually cover

Raise the caps so a full-history request is expressible, and — more importantly —
**make truncation impossible to miss**. Today a request for more than the cap
returns fewer rows and looks like a short history rather than a clipped one.
Either return an explicit indicator that the window was truncated, or reject the
request and say what the maximum is. Silently returning a prefix of the answer is
the one behaviour to avoid, because it renders as a chart that is simply wrong
about when the market began.

`from`/`to` are already supported on these endpoints. Prefer them over ever
larger `limit` values: a zoom control asks for a window, not for everything.

### 3. Decimate on the server, not in the browser

25,000 points per series × five series is not a chart, it is a denial of service
against the main thread — and at 1920px wide it is roughly thirteen points per
pixel, so the detail is not visible anyway.

Downsample server-side to a target point count. Use **LTTB** (largest-triangle-
three-buckets) or an equivalent shape-preserving method, not naive stride
sampling: a crash is a single-day spike, and every-Nth-point sampling drops
1987-10-19 the moment N > 1. That would produce a chart of the last century of
markets with Black Monday missing, which is worse than showing five years
honestly. Assert in a test that the 1987, 2008 and 2020 extremes survive
decimation at every zoom level.

Keep the raw window available when the range is small enough to render whole, so
zooming in converges on real data rather than a smoothed approximation.

### 4. The zoom control

Ranges: **1Y · 5Y · 20Y · Max (1927+)**, plus a monthly tier backed by
`SHILLER:NOMINAL_PRICE` for **1871+** if you can do it without special-casing the
chart code — that series is monthly, so it is a different resolution rather than
a wider window, and mixing the two on one axis without saying so would imply a
daily record that does not exist.

Requirements:

- The selected range is in the URL (`?range=20y`), so a view can be linked and a
  reload does not silently reset to the default.
- Switching range refetches for that window rather than filtering a fixed
  payload client-side — that is the whole point of `from`/`to`.
- Every chart on the page moves together. Kinematics and jerk against a different
  window than price is a misreading waiting to happen.
- Log scale on the price chart for ranges past ~20 years. The S&P goes from 17 to
  7400; on a linear axis the first seventy years are a flat line on the floor,
  which is the same as not shipping them.
- Preserve the loading and empty states already in `equity.ts`. Deep ranges are
  slower and the page must not look broken while one is in flight.

### 5. Say what the data is

At Max, the chart spans two sources — `YAHOO:^GSPC` daily and, if you add it,
Shiller monthly. The page already has a provenance block; extend it to name the
series and resolution actually on screen for the selected range. A reader must
never have to guess whether they are looking at the S&P or a proxy, or at daily
closes or month-ends.

## Acceptance

- `/equity` at Max renders 1927→today, and 1987-10-19, 2008-09-29 and 2020-03-16
  are all visibly present.
- A truncated API response is impossible to mistake for a short history.
- The nightly run's write volume is materially unchanged from today.
- Range survives reload and is linkable.
- Decimation tests assert the crash extremes survive at every zoom level.
- `lint-imports`, full pytest, ruff, serving vitest, dashboard builds.

## Do not

- Do not fetch the full history and filter client-side.
- Do not use stride sampling.
- Do not raise `history_days` for the nightly run and leave it there.
- Do not silently blend daily and monthly series on one axis.
