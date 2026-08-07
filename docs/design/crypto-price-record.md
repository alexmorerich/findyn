# FinCrypto — the stitched price record

Design note for `compute/findynamics/engines/crypto/prices.py`, recorded at the
end of Phase 5.

It answers three questions a future contributor will have, in the order they
tend to be asked:

1. Why is a **daily average** allowed into a price record at all?
2. Why do **closes always win** once the two records overlap?
3. What does the **confidence penalty** actually do, and where does it surface?

The splice thresholds and their measured values are in the table at the end.
They are empirical guardrails, not tuned parameters, and the rule for changing
one is stated there.

---

## 0. The problem

Bitcoin has no single authoritative price. It trades continuously on dozens of
venues with real spreads between them, so every "the" bitcoin price is some
vendor's composite and the composites differ.

The configured primary, `STOOQ:BTCUSD`, is unreachable: stooq.com fronts its CSV
with the same JavaScript proof-of-work interstitial that blocks `^SPX`, from a
developer network and from GitHub's runners alike. The reachable keyless close,
`YAHOO:BTC-USD`, begins **2014-09-17**.

That start date is the whole problem. Bitcoin has had roughly six cycles and
2014 cuts off two of them — 2011 (a run to ~$31 and an ~93% drawdown) and 2013
(two separate peaks and the Mt. Gox collapse). A regime model whose confidence
ceiling is already justified by *how few cycles there are* was working from four
when six exist.

`BLOCKCHAIN:MARKET_PRICE` is keyless, daily, and starts **2010-08-18** — the
first date bitcoin had a market price at all. It is also a **volume-weighted
daily average across exchanges**, not a close.

---

## 1. Why a daily average is accepted before the closes begin

Because on the evidence it is the same market measured slightly differently,
and because the alternative is worse.

The two legs were compared over their 4,340 shared dates before anything was
built. They differ by a **median 1.5%**, 63% of days differ by more than 1%, and
2020-03-12 differs by **60%** — a $4,971 close against a $7,937 daily average.

That looks disqualifying and is not, because of *which* days disagree. Every one
of the worst is a large-intraday-range day:

| date | close | daily average | gap | what happened |
|---|---|---|---|---|
| 2020-03-12 | 4,971 | 7,937 | 59.7% | covid crash, ~-37% intraday |
| 2015-01-14 | 178 | 222 | 24.6% | capitulation low |
| 2017-12-07 | 17,900 | 13,844 | 22.7% | parabolic top |

A close is one instant; an average is the whole session. On a day with a 40%
range those two numbers are *supposed* to be far apart. The disagreement is the
definition of the two statistics, not an error in either.

What matters is whether the two describe the same **process**, and three
measurements say they do: no step at the seam (−2.9%, an ordinary day against a
1.4% median move), no systematic level bias (−0.18% mean signed gap), and
comparable volatility (ratio 1.016). A series that was smoothed, weekly, or
interpolated would fail the third badly — a 30-day rolling mean of the closes
scores about 0.2.

**The alternative was worse.** The options were: (a) ship four cycles and
silently call it the sample, (b) commit a static file of uncertain provenance,
or (c) splice a documented keyless series under a validated guard and label every
date it supplied. Only (c) both lengthens the sample and leaves the reader able
to tell which statistic they are looking at.

Option (b) was specifically rejected on architecture grounds, not taste: an
engine-level file loader reads data that never passed through `pit_join`, and
`WorldState.series` being the *only* data access an engine gets is how the
no-lookahead law is enforced structurally rather than by convention
(`01-target-architecture.md` §3). A parquet read inside `prices.py` would make
the replay test stop proving anything about the spliced years. If a static file
is ever genuinely needed, the right shape is the existing `--from-file` provider
path (as `StooqFileProvider` uses), which still lands rows in `macro_series`.

### What is still not equivalent, and is published rather than hidden

A daily average and a daily close remain different statistics **on any single
day**, even though their return processes match in aggregate. So the dates the
extension supplied are flagged per date rather than blended away — the same
choice `engines/gold/drivers.py` makes for its ex-post real rate: *"a different
quantity, so the splice is recorded per date rather than blended silently."*

The consumer most affected is the jump detector. It asks whether one day stood
out against its neighbours, and an average suppresses exactly the intraday range
that makes a day stand out. It self-scales — the threshold is set from each
date's own trailing bipower volatility, so a quieter window raises *fewer* flags
rather than wrong ones — but **a jump date before 2014-09-17 and one after are
not strictly the same measurement**, and nothing in the engine pretends
otherwise.

---

## 2. Why closes always take precedence once the overlap begins

Two reasons, one architectural and one arithmetic.

**A spliced record must never restate a published figure.** The extension
supplies dates *strictly before* the closes begin and nothing else. So the splice
can only ever lengthen the record. This is the same rule the equity engine's
`publication_path` follows, and it is what makes the operation safe to re-run: a
later run that gains access to more history extends the past, it does not rewrite
it. `test_prices.py::test_the_closes_are_never_restated_by_the_extension`
asserts the closes come through untouched across all 4,340 shared dates.

**Where they disagree, the close is the better number for this engine.** The two
legs differ by a median 1.5% on shared dates, so "whichever we saw last" would
introduce a visible, arbitrary 1.5% wobble. The close wins because it is what the
configured primary role (`STOOQ:BTCUSD`) also is — the day Stooq becomes
reachable it takes over from Yahoo without a config edit and without changing the
*kind* of number the record holds.

The precedence is expressed as an ordered tuple rather than a branch, so adding a
vendor is a list edit:

```python
CLOSE_ROLES: tuple[str, ...] = ("price", "price_fallback")
HISTORY_ROLE = "price_history"
```

### When the splice is refused

If any guardrail fires, the extension is dropped, the record falls back to the
closes alone, and the reason is carried on `PriceRecord.declined_reason` and
logged at ERROR with the measured value that tripped it. The failure is loud on
purpose: **silently publishing 2014-onwards where 2010 was expected looks
exactly like a run that worked.**

---

## 3. How the price provenance reaches the published state

`prices.py` decides nothing about confidence. It returns a `PriceRecord`
carrying facts — `from_fallback`, `average_share`, `declined_reason` — and
`engine.py::_confidence` prices them. Keeping the measurement and the judgement
apart is what lets the thresholds be re-tuned without touching the splice, and
vice versa.

Three of the seven confidence terms come from the price record:

| condition | key in `crypto.yaml` | charge |
|---|---|---|
| primary absent, fallback carried the run | `fallback_price_penalty` | 0.02 flat |
| part of the record is a daily average | `daily_average_price_penalty` | 0.05 **× average share** |
| extension configured but refused | `declined_splice_penalty` | 0.03 flat |

The middle one is **proportional, not flat**, because a record that is 5% average
and one that is 100% average are different claims. On the shipped snapshot the
share is 0.2557, so the charge is `0.05 × 0.2557 = 0.0128`.

Worked, for the current state:

```
0.50    ceiling (config: confidence.ceiling)
-0.02   Stooq unreachable, Yahoo carried the run
-0.0128 25.6% of the record is a daily average
------
0.4672  published confidence
```

Every term **subtracts**. Nothing in `_confidence` adds, and the function clamps
to `ceiling` rather than to 1.0 — so a future edit that introduces a bonus term
still cannot publish 0.6. `test_engine.py::test_no_configuration_can_raise_it_
above_the_ceiling` proves that by zeroing every penalty to −5.0 and asserting the
result is still exactly 0.5.

### Where it surfaces

* **`AssetState.confidence`** — the number above, and the field the portfolio
  layer would weight by if it could see this engine (it cannot; `experimental`).
* **`AssetState.components`** — `price_average_share`, `price_is_spliced`,
  `price_from_fallback_source`, `price_history_declined`.
* **`AssetState.signals`** — a `price_record_spliced` signal naming both vendors
  and the share, or `price_history_declined` carrying the refusal reason.
  Direction is `0`: it says something about the *measurement*, not the asset.
* **`engine_output`** — `price_is_daily_average`, 1.0/0.0 per date, so the seam
  is chartable rather than only summarisable.

---

## 4. The thresholds

**These are empirical guardrails, not tuned parameters.** Each was fixed by
asking "what would a *broken* splice look like?" and drawing the line there,
before checking whether the shipped pair cleared it. The measured values are
evidence that the guardrail is not binding — never the reason for its value,
which is why each sits an order of magnitude from what the data does rather than
comfortably just past it.

| guardrail | limit | measured | headroom | catches |
|---|---|---|---|---|
| `MIN_SPLICE_OVERLAP` | 250 obs | 4,340 | 17× | a comparison too short to mean anything |
| `MAX_SEAM_RETURN` | 0.25 | −2.9% | 8.6× | a manufactured price move — rebasing, wrong currency, units error |
| `MAX_LEVEL_BIAS` | 0.05 | −0.18% | 28× | a persistent offset, i.e. not the same quantity |
| `VOLATILITY_RATIO_BOUNDS` | 0.75–1.35 | 1.016 | centred | an extension whose return process is a different market |

Notes on the two least obvious:

* **`MAX_SEAM_RETURN` is deliberately permissive.** Bitcoin has had genuine
  sessions past 25% (2013-04-10 fell about half), so a tighter limit would refuse
  a sound splice whenever the join landed on a real crash. It cannot catch a
  small scale error and is not meant to; a scale error that matters is an order
  of magnitude bigger. Known failure direction: if a join *does* land on a
  genuine >25% day the splice is refused and history is lost — the conservative
  outcome, and a reported one.
* **`VOLATILITY_RATIO_BOUNDS` is roughly symmetric in ratio terms**
  (1/0.75 = 1.33 against the 1.35 ceiling). An extension that is *more* volatile
  than the closes is equally not the same series.

### The rule for changing one

If a guardrail fires on a new vendor or a regenerated snapshot, **investigate
the data — do not widen the limit.** A threshold moved to make a red check go
green has stopped being a check.

If after investigating you conclude the limit itself was wrong, change it in a
commit that does nothing else, states which failure mode the new value still
catches, and updates the measured figures in this table.

`test_prices.py` builds a synthetic pair per branch that trips exactly one
check, because the shipped pair passes all four — a suite that only ran against
real data would be testing that a splice happens, never that a bad one is
refused.

---

## 5. Consequences for the record as it stands

* 4,341 → **5,831 observations**, from 2010-08-18.
* The liquidity regression gained 49 months (142 → 191). The coefficient moved
  from 0.656 to 2.266 and R² from 0.0018 to **0.0072** — still a null result two
  orders of magnitude below the 0.2 ceiling the anti-p-hacking test allows. The
  change is data, not tuning: no model change, no smoothing.
* 2011 and 2013 are now in-sample, so the regime labels cover six cycles.
  `normal` remains the modal state at 45% (down from 57%), which is the expected
  direction — the added years contain a 93% drawdown and two peaks.
* 25.6% of the record is average-based, priced at 0.013 of confidence.

## 6. Open

* The `/crypto` page renders the new `price_record_spliced` signal automatically
  (the signals table is generic) but does **not** chart `price_is_daily_average`,
  so the seam is not visible on the price chart yet.
* Nothing here is in production. The compute plane runs from the default branch
  and this work is unmerged; `config/engines/crypto.yaml` also still ships
  `enabled: false`. Enabling, backfilling and the full-history run all come after
  merge.
