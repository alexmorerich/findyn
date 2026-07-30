# MASTER prompt — FinDynamics implementation driver

Copy everything below this line to the coder agent. It drives the whole
redesign; for single-phase execution use the individual `P*.md` prompts
instead.

---

You are implementing the FinDynamics redesign in the `findyn` repo
(`/Users/alexkou/Documents/github/findyn`). This is a multi-phase build with
a complete, authoritative design pack already written. Your job is execution,
not redesign.

## Step 0 — Orient (before any code)

Read, in this order:

1. `docs/redesign/README.md` — index and global rules for you
2. `docs/redesign/01-target-architecture.md` — target architecture
3. `docs/redesign/02-migration-map.md` — file-by-file migration mapping
4. `docs/redesign/03-contracts.md` — normative interface specs
5. `FINDYN_V1_SPEC.md` §14.1 — the no-lookahead rules (global law)

Then check `git status`. If there are uncommitted changes (there is
in-progress M1a work: providers, quality, resilience, dashboard scaffold,
migration 0003), commit them first as their own commit(s) with a clear
message — never mix them into refactor commits, and never discard them.

## Step 1 — Execute phases strictly in order

Each phase has a self-contained prompt. Open it, follow it exactly, and treat
its Acceptance section as the definition of done:

| Order | Prompt file | Summary |
|---|---|---|
| 1 | `docs/redesign/prompts/P0-core-refactor.md` | Package restructure `findyn`→`findynamics`, core contracts, registry, import-linter. Zero behavior change |
| 2 | `docs/redesign/prompts/P1-finrates-mvp.md` | FinRates: treasury curve → Nelson-Siegel → rate regime → D1/API/dashboard |
| 3 | `docs/redesign/prompts/P2-finmoney.md` | FinMoney: money-market account, carry, discount factors, liquidity state |
| 4 | `docs/redesign/prompts/P3-finequity.md` | FinEquity: FINDYN_V1_SPEC M2–M4 inside `engines/equity` (3 sub-milestones, 3 PRs) |
| 5 | `docs/redesign/prompts/P4-fingold.md` | FinGold: regime switching + jump detection + hedge score |
| 6 | `docs/redesign/prompts/P5-fincrypto.md` | FinCrypto: experimental, quarantined, default-disabled |
| 7 | `docs/redesign/prompts/P6-portfolio.md` | Portfolio engine + multi-asset dashboard |

Phase discipline:

- **One phase = one branch + one PR** (P3 = three PRs, one per sub-milestone).
  Branch names: `redesign/p0-core`, `redesign/p1-finrates`, etc.
- A phase is complete only when every item in its Acceptance section passes
  locally. Run the full verification suite before declaring done:
  `cd compute && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/pytest && lint-imports`
  and
  `cd serving && npm run typecheck && npm run check:crons && npm test && npm run deploy:check`.
- **Stop after each phase.** Report what was delivered, the acceptance
  results (paste actual test output summaries, not claims), and any deviation
  from the prompt — then wait for approval before starting the next phase.
- Do not start a later phase's work early, even if it seems convenient.

## Global rules (repeated because they are the ones agents break)

1. **No-lookahead is law.** Engines access data only through
   `WorldState.series` / `pit_join`. No centered filters. Expanding-window
   fits, frozen between refits. Every engine ships a PIT replay test.
2. **No cross-engine imports, ever.** Engines share data through
   `engine_output` / factor tables read via `WorldState.series`. If you feel
   the need to import another engine, you are wrong — check the design doc
   for the intended data path. import-linter must stay green.
3. **Config over code.** New series, thresholds, and enable-flags go in
   `series.yaml` / `config/engines/*.yaml`. Never hard-code a series id in
   Python.
4. **Never delete a test.** Move tests with their code; add tests per phase.
5. **Outputs are states, scores, quantiles, and signals** — never
   deterministic price targets, never trade commands.
6. Match the existing codebase's style: strict validation, frozen
   dataclasses, docstrings that cite the spec section, tests that assert on
   named historical windows.
7. Where a prompt and reality conflict (a file moved, an API changed), follow
   the design's *intent*, note the deviation in your phase report, and keep
   going — do not silently improvise a different architecture.

## Deliverable per phase report

- Branch name and commit list
- Acceptance checklist from the phase prompt, each item ✅/❌ with evidence
- Test totals (compute pytest count, serving vitest count) before → after
- Deviations from the prompt, with one-line justification each

Begin with Step 0 now.
