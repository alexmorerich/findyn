# Equity Dynamics Lab — backend follow-ups

Recorded while building the Lab UI (`feature/equity-dynamics-lab-ui`). **None of
these were fixed there, and none of them should be fixed to unblock the UI**:
the Lab ships a frontend fallback for each, and every one of those fallbacks is
a thing the page should arguably do anyway. They are written down here so the
gap is visible to whoever owns the contract, not as a request.

Each item states what was observed, where, and what the UI does instead.

---

## 1. Raw jerk is not published — only `jerk_z`

**Observed.** `EquityEngine.outputs` publishes `price_close`, `price_filtered`,
`velocity`, `acceleration`, `jerk_z` and `jerk_lamp`
(`compute/findynamics/engines/equity/engine.py`). The third derivative exists
inside the feature builder as `jerk_raw = acceleration.diff()`
(`features/kinematics.py`) but is never emitted — only its expanding z-score is.

Confirmed against production:

```
metric          available   first
price_close        24,767   1927-12-30
velocity           24,515   1929-01-03
acceleration       24,514   1929-01-04
jerk_z             21,994   1939-02-08
```

**Why it matters to a UI.** A "four derivatives" panel set is really three
derivatives plus a standardized one. The Jerk panel is in different units from
the three above it, so its y-axis cannot be read as `d³price/dt³`, and the
overlay mode's z-scoring standardizes an already-standardized series.

**What the Lab does instead.** Draws `jerk_z`, labels the panel *"third
derivative — z-score against its own expanding baseline"*, formats the values
with a `σ` suffix, and names the double standardization in the overlay panel's
own subtitle (*"jerk (z of z)"*). No raw jerk is reconstructed in the browser:
differencing the acceleration series client-side would produce a number that
looks like a model output and is not one.

**Note.** §3.1 demotes jerk deliberately — a raw third derivative of a daily
index is mostly microstructure. So "publish raw jerk" may be the wrong fix and
"nothing to do, the UI wording is sufficient" may be the right one. Recorded as
an observation, not a request.

---

## 2. Burn-in, splice boundary and smoothing parameters are not exposed

**Observed.** No `/api/v1` endpoint returns them. The Lab can see their
*consequences* — `velocity` starts a year after `price_close`, `jerk_z` twelve
years after — but not the parameters that produced those dates. The splice
between `FRED:SP500` and `YAHOO:^GSPC` is described in prose in the provenance
panel and encoded in `model_version`
(`equity-1.2.0+pub.fred_sp500_yahoo_gspc+cal.yahoo_gspc`), but the boundary
*date* is not a field anywhere.

**Why it matters to a UI.** Three lines that begin on three different dates read
as missing data. A debug surface that infers "burn-in = 252 observations" from
the gap between two first-dates would be guessing, and would be wrong the moment
the filter's start-up rule changed.

**What the Lab does instead.** Reports the observed first date per series as a
fact (it is in the response), and reports the parameters as
`"Not available from API"` in the debug panel, with the distinction stated
explicitly: what is *observed* is separated from what is *exposed*, and neither
is inferred from the other. The panel copy under the chart explains the gaps in
words instead.

**If it were exposed**, the natural shape would be metadata on the history
envelope rather than a new endpoint — the consumer needs it beside the series it
describes, not as a second request.

---

## 3. `access-control-allow-origin` is intermittent under load

**Observed.** `grep -ri "access-control" serving/src/` returns nothing — the
Worker source has no CORS handling at all — yet the deployed Worker returns
`access-control-allow-origin: *` on a quiet request. Under the ~13 parallel
requests `/equity` issues on load, a browser on a different origin got:

```
Access to fetch at 'https://findyn.<acct>.workers.dev/api/v1/assets/equity/history?...'
from origin 'http://localhost:4321' has been blocked by CORS policy:
No 'Access-Control-Allow-Origin' header is present on the requested resource.
```

Direct `curl` of the same URLs succeeded with the header present. A Python
client with a default user-agent got Cloudflare **error 1010** ("browser
signature banned"), which strongly suggests the header comes from a Cloudflare
edge feature and is absent on challenge/block responses rather than being set by
the Worker.

**Why it matters.** The deployed topology is same-origin — the Worker serves the
dashboard from its own assets binding — so **production is unaffected**. It bites
exactly two cases: `npm run dev` against a deployed Worker, and any future
Pages-hosted or custom-domain dashboard, which `lib/api.ts` explicitly supports
via `PUBLIC_FINDYN_API`.

**What the Lab does instead.** Nothing — this is not a UI concern. It was worked
around *for testing only* with a throwaway local proxy that served
`dashboard/dist` and forwarded `/api/v1` to the Worker, reproducing the
production same-origin topology. That proxy lived in `/tmp` and is not in the
repo.

---

## 4. Minor: server-side LTTB can return `points + 1`

**Observed.** Requesting `points=3000` returns 3,001 rows for some metrics and
exactly 3,000 for others, varying by metric and by window:

```
acceleration | drawn 3,001 | lttb 24514→3001   (requested 3000)
jerk_z       | drawn 3,001 | lttb 21994→3001   (requested 3000, custom window)
```

**Impact: none.** The count is reported honestly in the `decimated` block, and
one extra vertex is invisible. Recorded only because a decimator that overshoots
its target by one is usually an off-by-one in a bucket-boundary calculation, and
it is cheaper to notice now than to rediscover it later.

---

## Not a gap: `risk_score` is the RII

Worth writing down because it is easy to get wrong from the API surface alone.
`AssetState.risk_score` is not an independent quantity — `engine.py::_risk_score`
returns the §3.2 Regime Instability Index directly, falling back to
posterior-weighted regime severity only when the composite cannot be built.

The Lab therefore reads it against the RII's own thresholds (elevated 60,
high 80), the same two numbers the instability panel further down the page uses.
An earlier draft used 33/66 and produced a page that coloured one quantity by
two different rules — a contradiction a reader has no way to resolve. Any future
panel showing `risk_score` should inherit the same thresholds.

See `docs/backtests/equity-open-issues.md` §12 before treating a move in this
number as information.
