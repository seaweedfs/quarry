# Development Plan

Implements [`../core-design.md`](../core-design.md). Each phase ships something
testable. Phases 1–5 have no external dependencies, which is deliberate: the
decision logic that must be *correct* does not need a query engine to be
exercised, so it gets built and tested first.

Legend: `[x]` done · `[~]` in progress · `[ ]` not started

---

## Phase 0 — Scaffolding `[x]`

- [x] Crate, lints (`forbid(unsafe_code)`, `missing_docs`), `.gitignore`
- [x] This plan

**Done when** `cargo test && cargo clippy --all-targets && cargo fmt --check`
passes and keeps passing.

---

## Phase 1 — Place and Distance `[x]`

The topology primitive. Every placement question in the system — which worker
runs a task, which replica it reads, whether a shuffle crosses a boundary — is
"how far, and what does that distance cost?"

- [x] `Place`: a hierarchical path (`/onprem/dc1/rack2/node7`)
- [x] `Distance`: `Local | Near | Far`, from longest common prefix
- [x] `Place::unknown()` → every distance is `Far`

**Design constraint.** Three distances now; the path representation means
adding rack-awareness or a cross-cloud boundary later changes a price table,
not a type.

**Done when** distance is symmetric, reflexive on identical paths, and
`unknown` is `Far` from everything including itself.

---

## Phase 2 — Cost and prices `[x]`

One currency for every decision.

- [x] `Tier`: `Hot | Cold`
- [x] `Cost`: bytes, cpu-seconds, and a money total
- [x] `PriceTable`: byte price per `(Tier, Distance)`, cpu-second price
- [x] `Cost` addition, so plan costs compose

**Done when** a cold/far byte prices strictly above a hot/local byte, and
summing sub-costs equals costing the sum.

---

## Phase 3 — Snapshots and lineage `[ ]`

The `LINEAGE` condition of the rule needs an ancestry test, and `RESIDUAL`
needs to know which files changed between two snapshots.

- [ ] `TableId`, `SnapshotId`, `FileId`
- [ ] `Snapshot`: id, parent, the set of live data files, delete-state per file
- [ ] `SnapshotGraph`: `is_descendant(a, b)`, `changed_files(from, to)`
- [ ] Rollback produces a sibling, not a descendant

**Design constraint.** `changed_files` must report a file whose *deletes*
changed even though the file itself was neither added nor removed. A commit
that only adds delete files changes which rows are live; missing this is a
correctness bug, not an optimization miss.

**Done when** a delete-only commit shows up in `changed_files`, and a
rolled-back snapshot is not a descendant of the abandoned branch.

---

## Phase 4 — Derived state and the one rule `[ ]`

The heart of the design.

- [ ] `Derived`: id, kind, source `(table, snapshot)`, definition, policy
      fingerprint, bytes, use counter
- [ ] `Kind` trait: `matches`, `cost`, `refresh`
- [ ] `Rewrite`: `Prune` (safe when stale) vs `Substitute` (is not)
- [ ] `may_serve(&Derived, &Query, snapshot) -> Decision` implementing
      MATCH / LINEAGE / POLICY / RESIDUAL
- [ ] `Decision::Union` carrying the residual for substituting kinds

**Design constraint.** One function, every kind. A stale *pruning* rewrite is
sound because over-selection is conservative; a stale *substituting* rewrite
returns wrong answers. The rule encodes that asymmetry so no kind has to
remember it.

**Done when** a stale index is still usable, a stale projection is not, a
policy mismatch is refused, and a substituting kind at an advanced snapshot
returns a union with the correct residual.

---

## Phase 5 — Registry `[ ]`

- [ ] `Registry`: register, lookup candidates by table, drop
- [ ] Candidate selection: cheapest admissible derived state wins
- [ ] Eviction by bytes under a budget, least-valuable first

**Done when** a registry with several candidates picks the cheapest admissible
one and never an inadmissible one.

---

## Phase 6 — First kinds `[ ]`

Two implementations of `Kind`, easiest first, to prove the trait before the
hard cases.

- [ ] `ResultCache`: matches an identical plan hash. Substituting.
- [ ] `Index`: matches a predicate on indexed columns, prunes files. Pruning.

**Done when** adding a kind touches no core file.

---

## Phase 7 — Budgets `[ ]`

- [ ] `Budget`: byte and money ceilings
- [ ] A metered reader that aborts at the ceiling
- [ ] `Outcome`: `Complete | Aborted { at }`

**Design constraint.** Estimation informs plan choice and may be wrong.
Enforcement stops execution and must not be. A budget that only feeds the
planner is decoration.

**Done when** a budget provably aborts mid-read rather than after.

---

## Phase 8 — Explain `[ ]`

- [ ] A structured `Explain` value: plan, derived state used, residual, cost,
      coverage
- [ ] Text rendering

**Design constraint.** This is the entire introspection surface, and also the
whole agent-facing API: `EXPLAIN` is `plan()` — free, and it returns cost and
coverage before anything is spent.

---

## Phase 9 — Engine integration `[ ]`

Where external dependencies arrive. Everything above stays dependency-free.

- [ ] DataFusion: `TableProvider` over a toy in-memory table, end-to-end SQL
- [ ] An optimizer rule that consults the registry and applies a `Rewrite`
- [ ] `object_store` for reads; metadata and block caches
- [ ] `iceberg-rust` for real tables; Iceberg REST catalog client

**Design constraint.** Use DataFusion's own extension points —
`ObjectStore`, `TableProvider`, `Catalog` — not parallel ones. The only new
trait is the one that has no DataFusion equivalent: what a backend can tell us
beyond reading bytes.

---

## Phase 10 — Facts, telemetry, and the loop `[ ]`

- [ ] `StorageFacts`: `tier`, `distance`, `commit_feed`, all `Option`
- [ ] Two implementations: plain S3 (all `None`) and a locality-aware backend
- [ ] Iceberg `ScanReport` ingestion; query telemetry
- [ ] `auto_optimize`: build, measure realized benefit, retire

**Done when** the optimizer contains no backend name, and an unused derived
state is retired on its own.

---

## Deliberately out of scope

Carried over from the design, with the condition for revisiting:

```text
aggregate/cube kind          after Phase 10 has real GROUP BY telemetry
subsumption matching         after the result cache shows a hit rate worth
                             extending; time-range narrowing first
future-reuse optimization    after Phase 10 can measure predicted vs realized
distributed execution        when one large machine is provably exhausted
execution domains            on demand; it is a cut in the Place path
cross-principal sharing      after a policy-fingerprint audit
writes (INSERT/MERGE/DELETE) read-only; iceberg-rust lacks row-level writes
natural-language queries     belongs in the client, not the engine
```
