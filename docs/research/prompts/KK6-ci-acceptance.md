# KK6 — CI, determinism, and proof that production is unchanged

Copy everything below this line to the coder agent. Final phase. Requires KK5
merged.

---

You are working in the `findyn` repo. Read `docs/research/prompts/MASTER-kk-omega.md`.

Branch: `research/kk6-acceptance`.

## Task

Close the track. This phase adds almost no features — it produces *evidence* that
the previous five phases did what they claimed, and wires the minimum CI needed
to keep it true.

## 1. Full-suite verification

Run and paste real output (not summaries you wrote yourself):

```
cd compute
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/lint-imports
.venv/bin/pytest -q
.venv/bin/pytest -q --cov=findynamics --cov-report=term-missing
```
```
cd serving && npm run typecheck && npm run check:crons && npm test && npm run deploy:check
cd dashboard && npm run typecheck && npm run build
```

Report the compute test count **before and after** the whole track (get "before"
from the merge-base commit). Every pre-existing test must still pass; none may
have been modified to accommodate the new module. Prove it:

```
git diff --stat <merge-base> -- compute/tests/ | grep -v 'tests/research/'
```

must be empty. If it is not, justify every line.

## 2. The production-unchanged proof

A feature flag is a promise; this is the evidence.

- **Structural:** `lint-imports` green with the `Research is quarantined`
  contract. Plant an import of `findynamics.research.omega` in
  `findynamics/engines/equity/engine.py`, show `lint-imports` failing, revert.
- **Behavioural:** run the daily job against the committed fixture on the
  merge-base commit and on HEAD, and diff the write-back payloads.
  They must be **byte-identical**. Use the pattern the crypto phase used
  ("prove with a test or a recorded run diff"). Commit the comparison as a test
  if it can be made fast; otherwise commit the recorded payloads and the diff
  command in the phase report.
- **Config:** `config/research/omega.yaml` still ships `enabled: false`;
  `core.config.load_series_config()` is unaffected by the presence of
  `config/research/`; no engine's `enabled` flag changed;
  `git diff <merge-base> -- compute/config/engines/ compute/config/series.yaml`
  is empty.
- **Registry:** `registered_engines()` returns the same five names;
  `portfolio_engines(config)` returns the same list as on the merge-base.
- **Serving/D1:** `git diff --stat <merge-base> -- serving/` is empty. No
  migration was added. `npm run check:crons` green.

## 3. Determinism gate

A dedicated test module `tests/research/omega/test_determinism.py`:

- Ω spec, Ω path, `C`, `K` and the walk-forward artifact all serialize to
  identical bytes across two runs **in separate processes**.
- The same, with `PYTHONHASHSEED` set to two different values — catches any
  dict/set iteration-order dependence.
- The same, with `OMP_NUM_THREADS=1` and `OMP_NUM_THREADS=4` — catches an
  unpinned thread pool. If this one fails, the fix is `threadpool_limits(1)`
  around the fit, exactly as `engines/equity/regime/hmm.py` does; it is not to
  loosen the assertion to a tolerance.
- Mark slow cases with the existing `@pytest.mark.slow` marker (registered in
  `pyproject.toml`) rather than adding a new marker.

## 4. CI wiring — minimal and honest

In `.github/workflows/ci.yml`, the compute job already runs the whole suite, so
the research tests run automatically. Add **only** what is missing:

- Install the research extra: change the install step to
  `pip install -e ".[dev,research]"` (the extra is empty today, which is the
  point — it must be exercised so the first added dependency does not break CI
  silently).
- Nothing else. **Do not** add a scheduled workflow, a cron entry, or a
  `[project.scripts]` entry for the research job. The report is regenerated
  deliberately, for the same reason `jobs/backtest.py` is: a file that changes
  under a cron makes every claim in the repo un-anchored.

## 5. Documentation closeout

- `docs/research/README.md`: an index in the style of `docs/redesign/README.md` —
  the design note, the report, the prompts, and a one-paragraph statement of what
  the track concluded.
- Add one line to the top-level `README.md` research/experimental section
  pointing at it, with the verdict named. If the verdict is negative, the line
  says so.
- If the verdict is `REJECTED` or `NO INCREMENTAL INFORMATION`, open
  `docs/follow-ups/kk-omega.md` recording what was learned, what should be
  deleted, and under what new evidence it would be worth revisiting.

## Acceptance criteria — the full track

Tick these only with evidence pasted beside each:

- [ ] Ω implemented as a latent financial state
- [ ] Ω estimated only from information available at time *t*
- [ ] Ω velocity implemented
- [ ] Ω acceleration implemented
- [ ] KK-inspired coupling implemented (three estimators)
- [ ] Financial curvature implemented, weights not fitted on the full sample
- [ ] Baseline vs Ω models compared — **including the A′ / RII control arm**
- [ ] Walk-forward testing implemented on the existing PIT replay gateway
- [ ] No lookahead leakage: truncation tests bit-identical; shuffled-target
      control shows no skill
- [ ] Determinism verified across processes, hash seeds and thread counts
- [ ] Existing production behaviour unchanged — structural, behavioural, config,
      registry and serving evidence all pasted
- [ ] Feature flag defaults to disabled; quarantine contract green
- [ ] Unit tests added
- [ ] Integration tests added (feature pipeline, HMM regime comparison, replay
      harness, artifact round-trip)
- [ ] Leakage tests added
- [ ] Existing test suite green, unmodified
- [ ] Research metrics reported with multiplicity-adjusted significance
- [ ] Failure cases documented, with a sensitivity table
- [ ] Conclusions empirical, not metaphysical — language test green

## Final report

One page: what was built, what the data said, the verdict, the three biggest
threats to that verdict's validity, and the single cheapest experiment that would
most change it.
