# FinDynamics Redesign — Design Pack

This directory is the **design-only** deliverable for evolving the current repo
(FinDyn v1.0 — S&P500 Dynamic State Engine) into **FinDynamics**, a multi-asset
financial physics framework with five independent asset engines
(FinMoney / FinRates / FinEquity / FinGold / FinCrypto).

Nothing in here is code. Each phase has a self-contained prompt that can be
copy-pasted to a coder agent verbatim.

## Documents

| File | Content |
|---|---|
| [`01-target-architecture.md`](./01-target-architecture.md) | Target architecture, layers, dependency rules, repo layout, data model, API |
| [`02-migration-map.md`](./02-migration-map.md) | Exact mapping from current code to the target structure; what is kept, moved, split, deferred |
| [`03-contracts.md`](./03-contracts.md) | Core interface specifications: `AssetEngine`, `AssetState`, `WorldState`, factors, registry |
| [`04-ui-plan.md`](./04-ui-plan.md) | UI evolution per phase — every phase ends with a visible, deployed upgrade to the live site |

## Prompts (execution order)

| Prompt | Phase | Deliverable |
|---|---|---|
| [`prompts/P0-core-refactor.md`](./prompts/P0-core-refactor.md) | Phase 0 | Restructure `compute/` into the `findynamics` package; core contracts + registry; zero behavior change |
| [`prompts/P1-finrates-mvp.md`](./prompts/P1-finrates-mvp.md) | Phase 1 | FinRates MVP: treasury curve → Nelson-Siegel → rate regime → API + dashboard |
| [`prompts/P2-finmoney.md`](./prompts/P2-finmoney.md) | Phase 2 | FinMoney: cash carry, discount factors, risk-free benchmark |
| [`prompts/P3-finequity.md`](./prompts/P3-finequity.md) | Phase 3 | FinEquity: the original FinDyn v1 spec (Kalman/HMM/RII) rebuilt as an engine |
| [`prompts/P4-fingold.md`](./prompts/P4-fingold.md) | Phase 4 | FinGold: real-rate/USD/stress driven regime + hedge score |
| [`prompts/P5-fincrypto.md`](./prompts/P5-fincrypto.md) | Phase 5 | FinCrypto: isolated experimental engine |
| [`prompts/P6-portfolio.md`](./prompts/P6-portfolio.md) | Phase 6 | Portfolio decision engine + multi-asset dashboard |

## Rules for the coder agent (apply to every prompt)

1. Read `docs/redesign/01-target-architecture.md`, `02-migration-map.md`,
   `03-contracts.md` before writing any code.
2. The no-lookahead rules of `FINDYN_V1_SPEC.md` §14.1 are **global law**:
   every engine reads macro data only through `pit_join`; no centered filters;
   expanding-window fits only.
3. Never let one engine import another engine. Never let `core` import an
   engine. CI enforces this (import-linter) after P0.
4. Do not break the existing test suite; move tests together with the code
   they cover.
5. One phase = one PR-sized change set. Do not start the next phase's work.
6. Every phase ships and **deploys** its UI per `04-ui-plan.md` — a phase is
   not complete until the live workers.dev site shows it.
