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

## Phase 3 — Snapshots and lineage `[x]`

The `LINEAGE` condition of the rule needs an ancestry test, and `RESIDUAL`
needs to know which files changed between two snapshots.

- [x] `TableId`, `SnapshotId`, `FileId`, `DeleteState`
- [x] `Snapshot`: id, parent, the set of live data files, delete-state per file
- [x] `SnapshotGraph`: `is_descendant_or_self`, `diff`
- [x] Rollback produces a sibling, not a descendant
- [x] Cycles and unknown snapshots return `false` rather than hanging

**Design constraint.** `diff` must report a file whose *deletes* changed even
though the file itself was neither added nor removed. A commit that only adds
delete files changes which rows are live; missing this is a correctness bug,
not an optimization miss.

**Found while implementing — the design doc's rule was incomplete.**
`core-design.md` states condition 4 as "residual is empty, or the rewrite
becomes `Union(use(D), scan(residual))`". That holds only when every change
since the derived state was built was *additive*:

```text
additive     files added
             → derived state is still correct as far as it goes
             → Union(use(D), scan(added)) is right

subtractive  files removed, or their deletes changed
             → derived state contains rows that are no longer live
             → NO amount of extra reading removes them
             → the derived state is unusable, not repairable
```

Hence `Diff` classifies rather than returning a flat file list, and
`Diff::is_purely_additive()` is what condition 4 actually tests. Compaction
and any row-level delete both land in the subtractive case.

**Done when** a delete-only commit shows up in `diff`, a compaction is not
purely additive, and a rolled-back snapshot is not a descendant of the
abandoned branch.

---

## Phase 4 — Derived state and the one rule `[x]`

The heart of the design.

- [x] `Derived`: id, kind, source `(table, snapshot)`, policy fingerprint,
      bytes, use counter
- [x] `Kind` trait: `name`, `matches`, `cost`, `refresh`
- [x] `Rewrite`: `Prune` vs `Substitute`
- [x] `Derived::may_serve(&Query, &SnapshotGraph) -> Decision` implementing
      MATCH / LINEAGE / POLICY / RESIDUAL
- [x] `Decision::UseWith` carrying the files to scan alongside
- [x] `Reason` names the first condition that failed, for `EXPLAIN`

**Design constraint.** One function, every kind. Kinds answer only about
*shape* — which columns or plan they can serve. Lineage, policy and staleness
are the rule's job, so no kind can forget them.

**Found while implementing — "pruning is safe when stale" needs a caveat.**
`core-design.md` says a pruning rewrite is safe even when stale, because
over-selection is conservative. True, but the dangerous direction is
*under*-selection: an index built at an older snapshot has never seen the
files added since, so pruning to only the files it knows about silently drops
their rows.

```text
Prune      tolerates ANY change, but files added since MUST be
           scanned alongside it. Removed and re-deleted files are
           harmless: the engine reads only live files and applies
           deletes as it goes.

Substitute tolerates ADDITIVE change only, also unioned with the
           added files. Any subtractive change disqualifies it.
```

So condition 4 is rewrite-dependent, and `may_serve` encodes both halves. A
pruning kind is strictly more permissive than a substituting one — which is
the concrete reason indexes ship before projections and aggregates.

Lineage is required for both, though a pruning rewrite could in principle
tolerate a cross-branch source. Deliberately not exploited: cross-branch reuse
only matters after a rollback, and one rule that is easy to verify is worth
more than the rare hit.

**Found in phase 9 — the union half of condition 4 needs a shape check.**
`core-design.md` repairs additive staleness with `Union(use(D), scan(added))`
for every substituting kind. That is only sound when `D` holds *table-shaped*
rows. Reading a pre-computed `count(*)` alongside newly appended raw rows is
nonsense — combining those needs a merge step, not a concatenation.

```text
Substitute { unionable: true }   a filter or projection result
                                 → residual may be read alongside it

Substitute { unionable: false }  an aggregate
                                 → admitted ONLY when nothing was added
                                 → Reason::ResidualNotUnionable otherwise
```

So `ResultCache` has two constructors, `rows_of` and `aggregate_of`, and the
caller has to say which it built. The conservative case is the one that needs
declaring, which is the right way round for an aggregate kind arriving later.

**Done when** a stale index is used with the added files, a delete-only commit
disqualifies a cached result but not an index, an abandoned branch is refused,
and a policy mismatch is refused.

---

## Phase 5 — Registry `[x]`

- [x] `Registry`: register, remove, get, byte total
- [x] `candidates` / `best`: cheapest admissible derived state wins
- [x] Total, deterministic ordering — cost, then fewest extra files, then id
- [x] `evict_to(max_bytes)`, least valuable first
- [x] Use counting, to drive retirement later

**Design constraint.** The registry holds no per-kind logic; everything it
needs arrives through `Kind`. It also never second-guesses the rule — an
inadmissible piece is omitted rather than ranked.

Eviction currently ranks by uses per byte. The design calls for expected
*remaining* value, which cannot be computed before realized benefit is
measured in phase 10; the signature will not change when it is. Eviction is
unconditionally safe because derived state is disposable, so the worst outcome
is a slower query.

**Done when** the cheapest admissible candidate wins, an inadmissible one is
never returned, ordering is stable across ties, and eviction keeps the
high-value entry.

---

## Phase 6 — First kinds `[x]`

Two implementations of `Kind`, easiest first, to prove the trait before the
hard cases. Between them they cover both halves of the rule's staleness
asymmetry.

- [x] `Predicate` in `Query`, so an index has something to probe
- [x] `ResultCache`: matches an identical plan hash. Substituting.
- [x] `Index`: equality on one field, prunes to matching files. Pruning.

**Confirmed: adding a kind touched no core file.** `Index` and `ResultCache`
live entirely under `src/kinds/`. The one core change was adding `Predicate`
to `Query`, which is the *query* model rather than the derived-state model —
and it was needed because an index cannot prune without knowing what a
predicate compares against.

Deliberate limits, each with the cheaper thing done first:

```text
result cache   exact plan match only. Subsumption (a 30-day filter
               serving a 7-day query) is most of the reuse value and
               is deferred until the exact-match hit rate justifies
               building a matcher.

index          equality on a single field. Ranges are a Predicate
               variant away; composite indexes are a different
               matching problem and belong in their own kind.

refresh        both report NeedsRebuild for any change, since neither
               can read data files. For the index this is an
               efficiency concern only: a stale index stays usable
               under the rule.
```

**Done when** an absent value prunes to the empty set rather than failing to
match, an unmodelled predicate does not prune, and a delete-only commit
disqualifies the result cache while leaving the index usable.

---

## Phase 7 — Budgets `[x]`

- [x] `Budget`: byte and money ceilings, either or both optional
- [x] `Meter`: charges per read, refuses once a ceiling is breached
- [x] `Permit`: `Continue | Stop(Exceeded)`
- [x] `Outcome`: `Complete | Aborted(Exceeded)`

**Design constraint.** Estimation informs plan choice and may be wrong.
Enforcement stops execution and must not be. A budget that only feeds the
planner is decoration.

Three decisions worth recording, all about being honest after the fact:

```text
charge then check   the breaching read is charged before the ceiling is
                    tested, so the report says 5 MB were read against a
                    4 MB limit rather than pretending only 4 were

stay stopped        a stopped meter charges nothing further and keeps
                    returning the original breach, so the first cause
                    is never overwritten by a later one

bytes before money  when both ceilings breach at once, bytes are
                    reported: it is the limit a caller set deliberately
                    and the easier one to act on
```

**Done when** ten 1 MB reads against a 4 MB ceiling permit exactly four, the
fifth stops, and the reported spend is 5 MB.

---

## Phase 8 — Explain `[x]`

- [x] `Explain::plan`: derived state chosen, files to also scan, cost, coverage
- [x] `Refused`: what was considered and which rule condition it failed
- [x] `Coverage`: `Exact | Incomplete(Exceeded)`
- [x] Text rendering
- [x] `Registry::assess`, so refusals are visible

**Design constraint.** This is the entire introspection surface, and also the
whole agent-facing API: `EXPLAIN` is `plan()` — free, and it returns cost and
coverage before anything is spent. Tests pin both halves of that: planning
twice gives an identical answer, and planning does not count as a use.

Two things the output does on purpose:

```text
names refusals    a query that is slower than expected is usually one
                  whose derived state was refused, and the reason is the
                  only thing that tells a user what to fix. A candidate
                  that merely lost on cost is NOT listed as refused.

separates known   byte counts come from immutable metadata and are
from estimated    reported as known. Latency is not modelled, so it is
                  absent rather than guessed — it is the least reliable
                  dimension and the one an agent would trust most.
```

`Coverage::Exact` is currently an invariant rather than a hope: the rule
refuses anything that would be wrong, so the only way to lose exactness is a
budget abort. Approximate kinds would add a `Bounded` variant carrying a
confidence interval, which is why the type exists now with two variants
instead of being a boolean.

---

## Phase 9 — Engine integration `[~]`

Where external dependencies arrive. Everything above stays dependency-free,
behind `--features engine`.

- [x] DataFusion `TableProvider` whose `scan` is planned by the rule
- [x] DataFusion `Expr` → `Predicate` translation, operand order normalised
- [x] End-to-end SQL tests: pruning, staleness, rollback, compaction, policy
- [x] `MaterializedResult`: a substituting kind holding real Arrow batches
- [x] Parquet objects read through `object_store`, selected by the rule
- [x] `MeteredStore`: counts bytes fetched, enforces a budget against real I/O
- [ ] Metadata and block caches
- [ ] `iceberg-rust` for real tables; Iceberg REST catalog client

**Design constraint.** Use DataFusion's own extension points —
`ObjectStore`, `TableProvider`, `Catalog` — not parallel ones. Held: the only
new trait so far is `Kind`, which has no DataFusion equivalent. No custom
optimizer rule was needed either, because `TableProvider::scan` already
receives the projection and filters, which is exactly what the rule wants.

Two things worth recording:

```text
Inexact pushdown    filters are used to PRUNE, never to filter rows, so
                    DataFusion re-applies them above the scan. Claiming
                    Exact would be wrong: pruning is conservative, so an
                    admitted file still holds rows the predicate rejects.

Kind: Send + Sync   a registry is shared across planning and execution
                    threads, so kinds must be too. Requiring it on the
                    trait keeps a lock off the planning path.
```

**The extension point holds.** `MaterializedResult` is a `Kind` defined in
`src/engine/`, *outside* the core, because it holds Arrow batches. Nothing in
`derived`, `registry`, or `explain` knows it exists and none of them changed to
accommodate it. That was the design's central claim about adding a kind, and it
survived contact with a real dependency.

Stored rows are held by the table keyed on `DerivedId`, not inside the kind, so
no downcasting is needed: the registry decides *whether* derived state may be
used and the engine knows *how* to read it. Neither has to know the other's
types.

A substituting candidate whose rows were never supplied is skipped rather than
failing — the safe direction, since the answer is right and merely slower.
`ResultCache` remains useful for exactly that: recording that a result exists
and what it would cost, which is enough to plan and explain with.

**The compaction test earns its place.** Without the prune-set filtering added
in the phase-4 follow-up, `SELECT * WHERE tenant_id = 1` returns two rows
instead of one, because the index still points at a file compaction removed.
That is a wrong answer reachable through plain SQL, which is what makes the
"prevented by construction" rule worth having.

**Parquet reads go through DataFusion's own file source.** `FileScanConfig` +
`ParquetSource` + `DataSourceExec`, with one file group per selected file. No
Parquet reading was hand-rolled, which keeps row-group pruning, predicate
pushdown and the reader's own optimisations for free.

`FileId` holds the object path, so the identity the rule reasons about and the
identity the store reads are the same string — as an Iceberg data file path
will be.

The test worth keeping is `a_pruned_object_is_never_opened`: it **deletes** the
file the index excludes and asserts the query still succeeds. Counting bytes
would show pruning saving I/O; deleting the object proves the plan never
reaches for it at all.

Substitution over a file-backed table unions a memory source of stored rows
with the file source, which is sound only because the rule already refused any
derived state whose rows are not table-shaped.

**Two claims stopped being claims.** `MeteredStore` wraps any `ObjectStore`,
counts what it fetches, and refuses reads once a budget is spent:

```text
pruning saves I/O    the same query, with and without an index, against a
                     counting store: fewer bytes AND fewer requests, same
                     answer. Previously an argument; now a number.

budgets are real     a 64-byte ceiling aborts a Parquet scan with an error
                     naming the breach. It FAILS rather than returning
                     fewer rows, because a silently truncated answer looks
                     complete, which is the worse failure.
```

Counting happens in `get_opts` only. Every other read path (`get`,
`get_range`, `get_ranges`, `head`) reaches the store through it by default, so
one counted method covers all of them, and `GetResult::range` gives the byte
count without consuming the payload stream.

Reads are priced hot and `Far` by default, which is what remote object storage
is. A colocated store says otherwise via `with_locality`, and then the same
money budget buys far more bytes — tested, and the first place `Distance` pays
for itself outside a unit test.

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
