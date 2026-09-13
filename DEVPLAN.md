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
- [x] `RangeCache`: read-through cache for object ranges
- [x] `Quarry` / `Session`: one place that assembles the stack
- [x] `from_iceberg`: a `SnapshotGraph` derived from real Iceberg metadata
- [x] `table_from_iceberg`: SQL over a real Iceberg table, planned by the rule
- [x] `table_from_catalog`: SQL over a table loaded from a REST catalog

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

**The cache, and why it is allowed to exist.** `RangeCache` caches bounded byte
ranges keyed on `(path, range)`. That would be *unsound* in general: an object
overwritten in place under the same path would serve stale bytes, and the
answer would look entirely normal.

What makes it sound is a property of the table format, not of the code:

```text
Iceberg data files are immutable.
New data means new files; a committed data file is never rewritten.
```

So the constructor is `RangeCache::for_immutable_objects`, not `new` — a caller
pointing it at mutable objects has to type out the assumption being broken.
Writes and deletes still invalidate, as defence in depth rather than the
argument.

Composition order is the useful kind of ordering:

```text
RangeCache(MeteredStore(store))   the meter counts only what MISSED, so
                                  its stats measure real origin I/O

MeteredStore(RangeCache(store))   the meter counts every read including
                                  hits, which measures demand, not cost
```

The first arrangement gives the headline test: the same query, run twice in
*separate sessions* so DataFusion's own metadata caching cannot be mistaken for
ours, fetches **zero** extra bytes from the origin the second time.

Deliberate limits: only bounded ranges are cached, because resolving an offset
or suffix needs the object size and that is one round trip more than a cache
should cost. Conditional and versioned requests bypass entirely, since their
purpose is to ask the store something the cache cannot answer. Eviction is
insertion-ordered rather than frequency-based; the design wants the latter, and
eviction is unconditionally safe either way.

**One entry point, and it needed two meters.** Assembling a working engine took
about thirty lines of setup, which is fine in a test and wrong as an interface
for a design whose user-facing claim is that ordinary SQL gets faster.
`Quarry` owns the stack and hands out a `Session` per query:

```text
MeteredStore   per session: enforces this query's budget
     └─ RangeCache   shared: immutable object ranges
           └─ MeteredStore   shared: lifetime origin I/O
                 └─ origin
```

Writing this down forced a distinction that had been left implicit. Both
meters are wanted, because they answer different questions:

```text
outer   sees every read INCLUDING cache hits
        → the right basis for a budget: a query reading 10 GB out of
          cache still consumed 10 GB of work

inner   sees only what MISSED
        → real origin I/O: what the cache saved, and what a remote
          store would have billed
```

The cache has to sit between them, because it must outlive any one session
while a budget must not. The caching test now asserts all three numbers at
once: origin bytes unchanged, cache hits above zero, session bytes above zero.

**Real Iceberg metadata, behind its own feature.** `from_iceberg` walks a
table's snapshots, manifest lists and manifests and produces a `SnapshotGraph`.
It is gated on `iceberg` rather than `engine`, because the bridge is pure
metadata: it needs no query engine, and the engine needs no Iceberg.

The test table is genuine — real metadata JSON, real Avro manifest lists and
manifests, written with `iceberg-rust`'s own writers and read back through its
own readers. Only the Parquet data files are absent, because the bridge never
opens them. That is the point: everything the rule needs is metadata.

Delete attribution is deliberately coarse:

```text
no delete files in a snapshot
    → every data file is DeleteState::NONE

any delete files
    → every data file gets the SAME fingerprint, from the whole
      delete-file set
```

Position deletes can name a `referenced_data_file`, so partial attribution is
possible; equality deletes apply by value across a partition, so exact
attribution is not generally available. Over-invalidation is the safe
direction: a delete anywhere makes every substituting piece look stale, while
pruning keeps working because pruning tolerates any change.

**The two halves joined without either side changing.** `table_from_iceberg`
is about twenty lines: take the Arrow schema and field ids from Iceberg's
schema, the graph from `snapshot_graph`, the file list from `live_data_files`,
and hand them to `QuarryTable`. That it needed no change on either side is the
useful signal — if joining them had forced one, the boundary would have been in
the wrong place.

`SELECT * FROM events WHERE tenant_id = 1` over a genuine Iceberg table now
reads only the Parquet objects an index admits. The test deletes the excluded
object first, so the query would fail rather than merely slow down if the plan
reached for it.

**One identity for a file path.** Iceberg records data file locations as URIs;
an object store addresses them as paths *within* a store named separately. So
`object_path` strips the scheme and authority — `s3://bucket/data/a.parquet`
becomes `data/a.parquet`, and the bucket travels with the store's URL. Doing it
once, in the bridge, means `FileId` means the same thing to the rule and to the
scan, and neither converts. This was a latent mismatch: the engine tests had
been using absolute local paths, which happened to work.

**The catalog needed no dependency.** `table_from_catalog` takes
`iceberg::table::Table`, which every catalog implementation hands back carrying
both the metadata and a `FileIO` configured for wherever the table lives. So
the library depends on no catalog at all — REST, in-memory, Glue and anything
else work through the same three-line function. `iceberg-catalog-rest` and
`mockito` are pulled in only by the `rest-catalog` feature, for the test.

The test is worth having anyway, because it exercises the deployment shape that
matters: a catalog URI, a table name, a query. `iceberg-catalog-rest` talks
real HTTP over a real socket to a mock server answering `/v1/config` and
`loadTable`, and the table below it is on disk. Pruning through the catalog path
is asserted to match pruning through the local path exactly.

One configuration detail that would have cost an afternoon: the mock's
`/v1/config` returns **no** `warehouse` override. The client derives its
`FileIO` from `warehouse.or(metadata_location)`, so an `s3://` warehouse would
send it looking for S3 credentials for a table sitting on local disk.

**One API trap worth recording.** `ManifestWriter::add_delete_file` reads as
though it adds a delete file. It does not — it sets the entry's status to
`Deleted`, which per the spec means *this file was removed from the table*. A
delete file that is live in a snapshot is an `Added` entry in a
deletes-content manifest, written with `add_file`. Getting this backwards makes
deletes vanish rather than fail, and it was the bridge's
`ManifestStatus::Deleted` skip — correct per spec — that surfaced it.

---

## Phase 10 — Facts, telemetry, and the loop `[~]`

- [x] `StorageFacts`: `tier` and `distance`, both `Option`
- [x] Two implementations: `OpaqueStorage` (all `None`) and `PlacedStorage`
- [x] `resolve`: the one place defaults are applied
- [x] Wired through `MeteredStore` and `Quarry`
- [x] `Workload`: observation by query shape, with literals stripped
- [x] `proposals`: which index to build, ranked by what is at stake
- [x] `retirements`: what has not paid for keeping it
- [x] `PriceTable::byte_day_usd`, so retention and reads compare
- [x] The loop, tested end to end on real queries over real Parquet
- [x] `build_index`: building a proposed index by reading the data
- [x] `Policy` and `Optimizer`: a driver that runs the loop on its own
- [x] Refresh: rebuilding derived state the table has grown past
- [x] `StableHasher` and `HASH_VERSION`, gating anything written down
- [x] Index bytes persisted as Puffin blobs at self-describing paths
- [x] Registry rebuilt by listing after a restart
- [ ] Publishing to Iceberg `TableMetadata.statistics`
- [x] Credits and the shape aggregate persisted, with a retirement grace window
- [ ] `commit_notifications`, so a round knows the table moved without asking
- [x] Telemetry from other engines, so the workload is the table's, not ours

**Done when** the optimizer contains no backend name, and an unused derived
state is retired on its own. **Both halves now hold.** `MeteredStore` used to
hardcode hot-and-far and now asks the backend; nothing outside `facts.rs`
mentions one. And `tests/loop_closes.rs` walks the whole cycle: ten unaided
queries, a proposal, a build, ten aided queries, a measured saving, and the
proposal stopping — plus a gigabyte of unprobed index being retired.

**The counterfactual problem mostly dissolves for pruning.** Measuring an
optimizer is usually circular: you cannot know what a candidate would have
saved without building it, and once built the baseline is gone. The usual
answers — holdout sampling, shadow execution — both cost something.

For pruning it is not needed, because the baseline is *computable*:

```text
a full scan's cost = the sum of the live data files' sizes
                     known exactly, at any snapshot, reading nothing
```

So the saving is `full_scan_bytes - bytes_read`: measured, per query, with no
holdout. `ScanReport::bytes_if_full_scan` carries it, and the test asserts the
credited saving equals exactly the three of four files the index ruled out.

Where it genuinely does not dissolve, stated rather than papered over:

```text
latency            not modelled; bytes are a poor proxy for a cpu-bound query
substituting kinds a cached aggregate's alternative is "read N bytes AND
                   compute", and the compute is not priced
never built        a proposal's saving can only be BOUNDED, which is why
                   Proposal::ceiling_usd is named a ceiling and documented
                   as assuming perfect selectivity. Inventing a selectivity
                   constant would have looked more precise and been less
                   honest; the ceiling still ranks proposals correctly.
```

Two deliberate choices in the proposer: it only proposes for shapes that went
**unaided**, so a served shape is left alone; and it merges by `(table, field)`
rather than by shape, so ten shapes filtering the same column ask for one index
rather than ten.

Retirement treats *never used* as a reason to retire. Something built and never
touched is indistinguishable from a leak, and rebuilding is cheap because
derived state is disposable.

**Building costs a fraction of a scan, for free.** `build_index` projects to
the single indexed column, so Parquet reads one column chunk per row group and
skips the rest — the same projection pushdown a query gets, through the same
DataFusion file source rather than a separate reader. The test asserts that
building read fewer bytes than the table holds.

Values are hashed through `ScalarValue`, the same path a query literal takes,
so build and probe cannot disagree about what a value hashes to. That is slower
than reading the array's native type and is the obvious thing to optimise once
correctness is pinned; getting it wrong would produce an index that silently
never matches.

Nulls are skipped. `x = NULL` matches nothing in SQL, and `IS NULL` is an
opaque predicate the index would not be probed for.

**A test caught itself being circular.** The first version of
`the_loop_builds_its_own_index_from_the_data` built an index and then measured
a *hand-written* one, so it would have passed whether or not the builder
worked. It now queries through the index it built and asserts by id that the
built one served the query.

**The driver, and the order within a round.** `Optimizer` owns the workload and
a shared registry, and `round()` does the whole sequence itself:

```text
1. retire first    frees budget, so a useful index is not refused because
                   a useless one is occupying the space
2. then build      most-at-stake first, which is the order proposals
                   already arrive in
3. budget after    an index's size is not knowable until it is built, so
                   the ceiling is enforced on what was produced
```

Step 3 is the same estimation-versus-enforcement line drawn everywhere else in
this crate. A build that turns out too large for the remaining budget is
discarded, not kept and apologised for.

**Sharing the registry was the blocker.** `QuarryTable` owned its `Registry` by
value, which is why every test rebuilt the table per query and why an optimizer
could not have added anything. It now holds `Arc<RwLock<Registry>>` — a plain
lock, not an async one, because planning takes a read guard and never awaits
while holding it. The test asserts the built index serves the *same* table
instance that was registered before it existed.

**Advisory and automatic share one code path.** `Policy::ADVISORY` runs the
same decisions and reports them as `Declined::Advisory` rather than acting. A
separate advisory path could disagree with what automatic mode would do, which
would make the advice worthless.

Rebuilding after a freshly-built index would otherwise happen every round: the
index has not served a query yet, so the workload still sees the shape as
unaided. The round checks by id, which is why `index_id` is derived from
`(table, field)` rather than being random.

**Decay: the failure the driver made visible.** An index is built against one
snapshot. When the table commits, the rule keeps using it — correctly, reading
the files added since alongside it — so the answer stays right. But that
residual is scanned by *every* query, growing with the table, until the index is
doing almost nothing.

Neither existing mechanism catches this:

```text
retirement   won't: the index is still credited with savings, just
             less and less of them

proposals    won't: the shape is being served, so it counts as helped
             and nothing asks for an index again
```

So `round()` gained a **refresh** step between retiring and building. Staleness
is measured as the fraction of the table the piece cannot help with — the added
files' bytes over the live bytes — because ten tiny appends matter less than one
large one and both sizes are known exactly. `Policy::max_residual_pct` is the
threshold.

The replacement is registered only once it exists, so a failed rebuild leaves
the stale-but-correct piece in place rather than nothing.

**Rebuilding needs to know what to rebuild, and the `Kind` will not say.** A
kind describes what it can *answer*, not how to remake itself, and growing that
trait to carry build instructions would make every kind pay for this one's
convenience. So the optimizer remembers what it built. The consequence, stated
plainly: it maintains only its own work — derived state registered by hand is
left alone — and it forgets on restart, along with the in-memory registry.

**The rollback case turns out to self-heal, because retirement runs first.** A
piece built on an abandoned branch is refused by the rule, so it saves nothing,
so retirement drops it; the eligible-to-build set is computed *after* that, so
the same round rebuilds it. That ordering was chosen to free budget and happens
to fix this too.

**A unit test was deleted for asserting on constants.** `Policy::ADVISORY` is a
`const`, so `assert!(!Policy::ADVISORY.auto_optimize)` is constant-folded and
proves nothing — clippy caught it. The behaviour that matters is tested against
a real table instead. Silencing the lint with a `const` block would have kept a
test that tests nothing.

**Found while implementing — "assume the worst" is wrong for one dimension.**
`core-design.md` says an unanswered capability should default to "unknown,
assume worst". That is right for distance and actively harmful for tier:

```text
unknown distance → Far
    correct, and usually true. If a backend cannot say where an object
    is, claiming locality would be a lie.

unknown tier → Hot, NOT Cold
    a backend that cannot report tiers almost certainly does not HAVE
    them; plain S3 is uniformly hot. Defaulting to Cold multiplies every
    price by the cold factor, overprices every read, and would abort
    budgeted queries that should have succeeded.
```

Pessimism about a dimension a backend does not have is not caution, it is a
made-up cost. `Resolved` carries an `assumed` flag so a caller can tell a
reported answer from a defaulted one.

**`commit_notifications` deliberately deferred.** The design names it as the
third method and calls it the capability that decides whether derived state is
maintainable. It is omitted until phase 10's loop can consume it: a method
nothing reads is the speculative generality this project has already deleted
once (`ByteRange`, phase 9).

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


---

## Phase 11 — Persistence `[~]`

Everything the optimizer learned and built lived in memory. A restart lost it,
and then — with credits gone — retirement would have deleted the indexes whose
bytes were still in storage. Disposable state is fine; destroying it on every
restart is not.

**It is three problems, not one**, and only one of them is large:

```text
index bytes         MB–GB   read every query      slower queries
registry metadata   ~200 B  startup + planning    orphaned bytes, rebuild
per-derived credits  ~40 B  per round             DESTRUCTIVE, see below
per-shape workload   ~40 B  per round             re-learns in min_queries
raw observations    GB/day  never read            nothing
```

Losing **credits** is the one that does harm rather than costing time:
`retirements` treats never-used as a reason to retire, so a restart with an
intact registry and empty credits deletes every index it has. Forty bytes per
piece.

The only large thing, the raw observation stream, is never read — the aggregate
is kilobytes.

### The hash had to be fixed first

`DefaultHasher`'s algorithm is not stable across Rust releases. An index keyed
on it survives a compiler upgrade as an index matching **nothing**: a miss, not
an error, so the system quietly slows and the loop rebuilds everything with no
signal. `StableHasher` is FNV-1a plus a SplitMix64 finalizer, both published
constants, with the pinned test vectors derived from an independent
implementation rather than from this code's output.

Writing that test found a portability bug that would have shipped:
`Hasher::write_u64` defaults to `to_ne_bytes` and `write_usize` to the
platform's pointer width, so hashes would have differed across endianness and
across 32- versus 64-bit builds.

`HASH_VERSION` exists because a fixed algorithm is not the whole story: the
bytes fed to it come from `std`'s and `arrow`'s `Hash` implementations, which
are theirs to change. Persisted state carries the version and is refused
loudly rather than matching silently.

### The format is Iceberg's

Puffin, and the shapes line up closely enough to be worth writing down:

```text
Puffin BlobMetadata          Quarry
  snapshot_id            ≡     Derived.source.snapshot   ← the lineage anchor
  fields: Vec<i32>       ≡     Index.field               ← a Vec, so composite is free
  type: String           ≡     Kind::name()
  properties: Map              policy fingerprint, hash version
```

An unfamiliar engine skips an unknown blob type safely. It cannot *use* the
index: this is a standard location, not a standard index format, and claiming
otherwise would oversell it.

### The path is the metadata

```text
<table location>/_quarry/idx/<table-uuid>/<field>/<source-snapshot>.puffin
```

so the registry is recoverable by listing, with nothing read and no side store
to keep in step. Write order follows: **blob first, register second** — an
orphan costs storage, whereas a registry entry with no blob behind it costs
confidence in every decision.

`Layout::parse` reads identity from the **tail** of a path rather than by
stripping a prefix, because `FileIO` and `ObjectStore` name the same object
differently and recovery has to recognise both.

### Two IO stacks, because FileIO cannot list

`iceberg::io::FileIO` has no listing operation at all — read, write, delete,
exists. So recovery uses `FileIO` for Puffin blobs and `ObjectStore` for
listing. Named in the docs rather than hidden, because it is a real seam.

A correction to something claimed earlier: the `RangeCache` does *not* serve
index reads, since Puffin goes through FileIO's own IO stack. It does not
matter — an index is read once into the registry and then held, so the registry
is the cache. Indexes are not re-read per query.

### Two bugs, one of them upstream

**`iceberg-rust` 0.6 panics on a truncated Puffin file.** Its footer reader
computes `input_file_length - footer_length` where `footer_length` comes from
four bytes read out of the file; on a truncated object those bytes are garbage,
the subtraction underflows, and the process aborts. In release builds it wraps
and requests an absurd range instead. A half-written object taking down an
engine is not an acceptable failure mode for state that is meant to be
disposable, so the footer is validated first — magic at both ends, and a
declared length that fits inside the file.

**`discard` reported deletes that deleted nothing.** `recover` put
object-store paths in `discarded` while `FileIO::delete` wanted its own form,
and deleting an absent key succeeds on every object store — so the count was
confident and wrong. It now records the readable form, and counts only objects
that existed and are now gone.

### Refusing is the only safe response to bad bytes

A truncated index is not a smaller index: it would prune away files holding
matching rows, which is the one direction that returns wrong answers. So
`decode` refuses on any inconsistency, including trailing bytes, and the test
truncates at **every** byte offset rather than at one convenient point.

### Credits, and the grace window they still need

Credits and the shape aggregate are now written as a second Puffin blob type.
The restart test asserts the whole cycle: learn, build, save, discard every
in-memory structure, recover, and run a round that retires **nothing** and
rebuilds **nothing**.

Persisting credits was necessary and is not sufficient, so
`Policy::retire_after_queries` holds retirement off until enough queries have
been observed to judge by. Two reasons:

```text
cold start     credits can be lost or absent, and a recovered index that
               has served nothing looks exactly like one that is useless

concurrency    two optimizers over the same storage overwrite each other's
               telemetry, so a busy index can legitimately read as unused
```

The asymmetry settles it rather than any cleverness: rebuilding a deleted index
costs a full scan, while keeping a useless one for another round costs almost
nothing. There is a test for the destructive case specifically — recovered
index, no telemetry at all, round retires nothing.

`Optimizer::adopt` exists because a recovered index is otherwise unmaintainable:
the optimizer would not know what field it covers, so it could neither refresh
it nor recognise it as already built, and would build a second one beside it.

### Still not persisted

Nothing that causes harm. The raw observation stream is dropped, which only
affects how quickly proposals re-converge, and it is the one thing here large
enough that writing it down would need compaction.


---

## Phase 12 — Plan identity `[x]`

A stored result was served to a query on the strength of a **hash match**. That
is a wrong answer, silently, and it was reachable through plain SQL.

### The bug was not the hash

The obvious reading is "64-bit hash, birthday bound, unlucky". That was the
smaller half. The hash was computed over the *translated predicates*, and
`Predicate` is deliberately lossy — it exists to decide which files might match,
where over-approximating is safe:

```text
Opaque { field }      tenant_id > 1 and tenant_id < 3 are the SAME predicate
Eq { value: u64 }     the literal is a hash, so colliding literals are one
columns.first()       a filter over two columns records only the first
filter_map on ids     a projected column with no field id vanishes
no known column       the filter vanishes entirely
```

Five losses, every one harmless for pruning and fatal for deciding two queries
are the same. So `tenant_id > 1` and `tenant_id < 3` hashed **equal by
construction**, not by coincidence: a result stored for one was returned for the
other, every time, with no collision required.

Fixing only the hash would have left that untouched. Comparing `Predicate`
structurally instead of hashing it would also have left it untouched.

### Identity and shape are now separate

`Query` carries both, and they answer different questions:

```text
predicates, projected   SHAPE. May over-approximate. Decides which files
                        might contain matching rows.

plan: Option<Plan>      IDENTITY. May not approximate at all. Decides that
                        two queries compute the same thing.
```

`Plan` holds the projected field ids and one faithful rendering per filter,
sorted so a conjunction has no order. `ResultCache` and `MaterializedResult`
store a `Plan` and compare it. `plan_hash` survives as a *name* for keying
stored bytes, documented as naming rather than identifying, and now hashes the
exact plan so the name corresponds to what it labels.

`Option` is the load-bearing part: when the engine cannot render a plan
faithfully — a projected column with no field id, so the mapping is not
injective — the plan is **absent**, and every substituting kind refuses.
Refusing costs a scan; approximating costs the wrong rows.

Renderings use `Debug`, not `Display`, because `Display` erases types:
`Int64(1)` and `Utf8("1")` both print as `1`. Operand order is normalised only
for the symmetric operators, since normalising `1 < a` wrongly would cost a
wrong answer while failing to normalise it only costs a missed match.

### The test asserts why, not just what

`two_different_filters_do_not_share_a_cached_answer` does the thing the fix is
for, and also asserts the *reason*: that the two queries have **identical
fingerprints**, because the lossy shape genuinely cannot tell them apart. A
test that only checked the outcome would not record why the outcome needs
defending.

### What this unblocks

Substituting derived state can now be persisted and shared between principals,
which the earlier note on `hash_plan` explicitly ruled out. Not done here, but
no longer unsound.

One precondition remains a caller's to keep and is documented on
`MaterializedResult::rows_of`: stored rows must be the **complete** answer to
the plan. DataFusion passes `scan` a row limit as a hint and applies `LIMIT`
above the scan, so returning more rows than asked is safe and returning fewer
is not.


---

## Phase 13 — Measurement `[~]`

234 tests, all on three or four files and at most a couple of thousand rows,
deciding with five constants chosen by reasoning. `examples/measure.rs` runs the
same machinery on a million rows and prints what actually happens.

Three regimes, and they behave completely differently:

```text
                     index size   prunes   bytes saved   proposer ceiling
CLUSTERED             0.4% of tbl  19/20     1.1%         1775x optimistic
SCATTERED             6.3% of tbl   0/20     0.0%         unboundedly so
SELECTIVE            67.8% of tbl  19/20    95.0%         1x  (accurate)
```

### Four findings, in order of how much they matter

**1. The proposer could not tell the useful case from the useless ones — fixed.**
`ceiling_usd` assumes perfect selectivity, and across these three it is 1775x
optimistic, unboundedly optimistic, and exactly right — in that order. The loop
would build the two worthless indexes as eagerly as the valuable one. This is
the most damaging result for the self-optimizing claim, because the mechanism
works and the *judgement* does not.

Selectivity alone cannot fix it: CLUSTERED has 1.0 files per value and saves
nothing, SELECTIVE has 1.1 and saves 95%. What separates them is whether
**Parquet's own row-group statistics already prune**, which is a question about
per-file min/max ranges overlapping — disjoint in CLUSTERED, total in
SELECTIVE. Iceberg manifests carry `lower_bounds` and `upper_bounds` per file
per field, so that is answerable from metadata alone, before building anything.

**2. A file-level equality index is largely redundant with the format.** On
clustered data it pruned 19 of 20 files and saved 1.1% of the bytes, because
the files it skipped were ones Parquet was already reading almost nothing from.
The index's real value is confined to the case where a value is rare *and*
ranges overlap, which is exactly the SELECTIVE regime.

**3. `bytes_estimate` was wrong by 5.7x — fixed.** `BYTES_PER_POSTING = 16`
against 77–92 bytes measured. The budget — `optimize_budget_pct` — is enforced
against that number, so a "5% of table" ceiling really admitted about 30%. And
a *recovered* index reported its true encoded length while a freshly built one
reported the estimate, so the same index was sized differently either side of a
restart.

`Index::encoded_len` now computes the exact figure instead, and a test asserts
the encoder produces precisely that many bytes, since the two must agree or the
budget changes its mind across a restart. Re-running the measurement confirms
it: reported size now equals encoded size in all three regimes.

**4. Most of a posting was a file path — fixed.** 91 bytes per posting, of
which 8 was the hashed value and the rest a path repeated once per posting. The
encoding now writes a file table once and has postings refer to it by number:

```text
                    before        after
CLUSTERED           0.46 MB       0.08 MB
SCATTERED           7.68 MB       0.46 MB      (16.7x, many postings, few paths)
SELECTIVE          90.12 MB      14.86 MB      (6.1x)
SELECTIVE, as %      67.8%         11.2%       of the table
```

68% of a table is not a size any storage budget admits; 11% is arguable. The
measurement recomputes the encoded length independently of
`Index::encoded_len`, so the two agreeing is a check rather than a tautology,
and it prints a warning when they diverge.

An uncomfortable side effect: the *useless* SCATTERED index is now the cheapest
of the three at 5 bytes per posting, because it has many postings over few
paths. Making indexes cheap does nothing to stop the loop building worthless
ones — which is finding 1 again, and the reason it is finding 1.

### One worry that did not materialise

Building allocates a `ScalarValue` per row, which looked like a scaling
problem. A million rows indexes in 0.05–0.64s. Not worth optimising.


---

## Phase 14 — Judgement `[~]`

The measurement's worst finding was not a bug. The loop would build an index
that saves 95% of a scan and two that save nothing, with equal enthusiasm:
`ceiling_usd` assumes perfect selectivity, so it was 1775x optimistic,
unboundedly optimistic, and exactly right, in that order.

### Selectivity cannot decide it

```text
             files per value   bytes saved
CLUSTERED         1.0            1.1%
SELECTIVE         1.1           95.0%
```

Nearly identical selectivity, opposite conclusions. What separates them is that
on clustered data **Parquet's own row-group statistics already prune** — the
index skips files the reader was barely touching.

So the question is not "how selective is this field" but "how much better than
per-file min/max ranges can an index do". `layout::Spread` answers it:

```text
files_by_bounds = sum of range widths / width of their union
                  1 when files partition the domain, N when all overlap

files_by_index  = min(files_by_bounds, rows per value)

advantage       = (files_by_bounds - files_by_index) / files
```

Against the three measured regimes that gives ~0%, 0% and ~94.5%, versus
realized savings of 1.1%, 0.0% and 95.0%. A test asserts the agreement, so if
the estimator drifts from what was measured it fails.

`Policy::min_index_advantage_pct` gates on it, and a round now declines with
`Declined::NoAdvantage`.

### Refusing without evidence, and what that costs

Rows per value needs a count of distinct values, which Iceberg does not record.
`Spread` takes it as an input rather than inventing it, so the requirement is
visible rather than buried in a constant.

The consequence is deliberate and worth stating plainly: with the default
policy, an optimizer given **no** layout evidence declines everything with
`Declined::NoEvidence`. Building on hope is what measurement showed to be
wrong, and a useless index costs a build and storage forever. But it also means
**the loop does nothing until per-field bounds are plumbed in** — which is the
next piece of work, not a property to be happy about.

### The fixture tests had to opt out, which is itself a finding

Four existing tests started declining, correctly: the fixture puts one tenant
per file, so its ranges are perfectly disjoint — the exact case where an index
is redundant. They now pass `mechanism_policy`, which sets the threshold to
zero, with a comment saying they test propose/build/refresh/retire rather than
the judgement.

That those tests were demonstrating the loop building a worthless index, and
had been read as evidence it worked, is the clearest illustration of why the
measurement was worth doing before more features.

### The input the gate wanted could not be obtained

`Spread::from_bounds` asks for rows per value, which needs a count of distinct
values. Iceberg does not record one, so it has to be estimated — and every
cheap estimator fails on exactly the cases the gate must separate. Sampling one
file of twenty and counting distinct values `d`:

```text
                d      NDV as d    NDV as d x files
SCATTERED    5,000      5,000       100,000  (true 5,000)
SELECTIVE   49,900     49,900       998,000  (true 906,341)
```

`NDV = d` makes SELECTIVE look worthless; `NDV = d x files` makes SCATTERED
look valuable. Wrong in opposite directions, and no constant factor fixes both.

So `Spread::from_overlap` asks a question that *is* measurable: do two files
hold the same values? Scattered files share nearly all of them, selective files
almost none.

```text
files_by_index = 1 + (files - 1) x shared
```

`estimate_overlap` measures `shared` from **two** files rather than the table,
which is what keeps the gate cheaper than the build it is gating — building an
index to find out whether the index is worth building would defeat the point.
It picks the first and last file rather than adjacent ones, since neighbours in
an ingestion-ordered table resemble each other and would make almost any column
look clustered.

`from_iceberg::field_bounds` supplies the ranges, free, from manifest
`lower_bounds`/`upper_bounds`. Only numeric and temporal types are mapped:
strings have bounds but projecting them onto a number to compare range *widths*
would invent a distance that does not exist.

Tests now measure both regimes off real data — one tenant per file gives an
overlap of exactly 0.0, four tenants in every file exactly 1.0 — and a
single-file table reports `None` rather than a guess.

### The gate now gathers its own evidence

`Optimizer::round` asks for it rather than being handed it. The two inputs come
from two places, and both are cheap:

```text
per-file ranges   from the table, which took them from the catalog's
                  manifest bounds during load, at no extra cost
value overlap     measured from two files, not the whole table
```

Bounds belong on `QuarryTable` because they describe the table's files, so
`table_from_iceberg` fills them in with one metadata pass over all fields —
walking the manifests per field would have multiplied the reads by the width of
the table.

**Absent bounds means declining, not assuming.** `Spread::from_bounds` treats
missing ranges as "the format prunes nothing", which is right as a statement
about the spread and wrong as a basis for building: it makes an index look
maximally valuable, optimistic in exactly the direction that produced the
useless indexes. So `evidence()` returns `None` without ranges and the round
reports `NoEvidence`. A caller who knows better can still override with
`with_spreads`.

The consequence is worth stating: a table built by hand, or written by
something that omits statistics, gets no automatic indexes at all.

**A test fixture was unfaithful and it mattered.** `tests/iceberg_sql.rs` built
its `DataFile`s without `lower_bounds`/`upper_bounds` — something real Iceberg
writers always populate — so the gate correctly found nothing to judge. The
fixture now records them, which makes it a more honest Iceberg table in
general. Three tests across two files also had to opt out of the gate with
`min_index_advantage_pct: 0.0`, since they exercise mechanism on hand-built
tables.

`the_gate_judges_a_real_iceberg_table_for_itself` is the one that matters: a
genuine Iceberg table, default policy, nothing supplied, and it reaches
`NoAdvantage` rather than `NoEvidence` — a judgement rather than a shrug.

### Still open






---

## Phase 15 — Calibrating the gate `[x]`

The sample size was the last unmeasured guess in a mechanism built to stop
guessing, so `examples/measure.rs` grew a ground truth: the built index's
postings over distinct values *is* the average files a value occupies. The gate
has to reach that number from a handful of column reads.

A fourth regime was added for it — PARTIAL, each value in three files of twenty
— because the existing three all sit at the extremes and would have flattered
any estimator.

### Wrong twice, in opposite directions

```text
sample                       PARTIAL estimate   truth
first and last only               13.7           3.0
four files, evenly spread          1.0           3.0
three adjacent pairs               3.5           3.0
```

The first sampled two files, chosen on the reasoning that neighbouring files in
an ingestion-ordered table resemble each other and would flatter any column.
Wrong twice over: "first and last" is distant only in *path order*, which need
not relate to content, and in that table they were neighbours.

Spreading the sample out failed the opposite way. A value spanning three
consecutive files shows **zero** overlap between files five apart, so widely
separated samples cannot see local clustering and report every value as living
in one file.

Both errors are invisible without ground truth. The first would have predicted
a 31% advantage where 85% was realized — a wrong *decision* at any threshold
above a third.

### The sample that works

A few positions across the list, each contributing a file **and its
neighbour**: adjacent pairs reveal local clustering, distant pairs reveal
global spread. Six files, fifteen pairs.

```text
              estimate   truth   predicted   realized   decision
CLUSTERED       1.00      1.00      0.0%       1.1%      refuse
SCATTERED      20.00     20.00      0.0%       0.0%      refuse
SELECTIVE       1.19      1.10     94.0%      95.0%      BUILD
PARTIAL         3.53      3.00     82.0%      85.0%      BUILD
```

Every decision correct, and every prediction within three points of what was
realized — against a `ceiling_usd` that was 1775x out on the first row.

Adjacent pairs are slightly over-represented compared with a uniform sample of
pairs, which biases toward *more* files per value, so less advantage and fewer
indexes built. The conservative direction, and deliberate.

`overlap_is_estimated_correctly_under_partial_clustering` pins the regime that
broke it twice, comparing the sampled estimate against the index's own
postings.


---

## Phase 16 — Ranking `[x]`

`Proposal::expected_usd` scales the ceiling by the advantage, so a proposal
reports what it is expected to save rather than the most it conceivably could.
Against measured data the ceiling was 1775x out; the expected figure lands
within three points.

### A bug that fell out of writing the test

Nothing had measured whether proposals were *ordered* well, only whether
individual decisions were right. They were not:

```text
max_builds_per_round was applied while walking proposals in ceiling order
```

and the ceiling is the number measurement found to be 1775x wrong on one
regime and unbounded on another. With the default of **one build per round**, a
round could build the worst candidate and decline the best as `RoundFull`.

The build phase is now: cheap refusals, then evidence, then rank by expected
saving, then cap. The cap applies to the ranked order, which is the whole
point of having one.

`the_best_candidate_is_built_not_the_one_with_the_biggest_ceiling` pins it with
two fields whose orders disagree — 1000 MB at 12% against 500 MB at 95%, so
the ceiling prefers the first and the expected saving the second. The round
must spend its single build on the second and defer the first rather than skip
it.

### On getting the test wrong first

The first version used 1000 MB at 15% against 30 MB at 95%, which the ceiling
and the expected saving *agree* on — 150 against 28.5. It would have passed
whether or not the ranking worked. Caught only because the assertion comparing
the two expected values failed before reaching the interesting part.


---

## Phase 17 — Bounds without a catalog `[x]`

The gate declined every proposal on a plain Parquet table, because per-file
ranges only arrived with Iceberg manifests. That confined self-optimization to
tables behind a catalog.

Parquet keeps the same statistics in every file's footer. So there was never
really nothing to go on — only nothing already in hand.

`parquet_bounds` reads footers, a few kilobytes per file and no column data,
and takes the union of each file's row-group ranges. `evidence` uses the
catalog's copy when the table came with one and falls back to footers
otherwise; both describe the same thing.

### A test asserted the old limitation and had to be rewritten

`the_gate_refuses_when_it_knows_nothing` began failing with `NoAdvantage` where
it expected `NoEvidence` — the gate now *had* evidence and reached a judgement.
Strictly better, and the test was encoding a limitation rather than a
requirement. It is now
`the_gate_reads_bounds_from_parquet_footers_without_a_catalog` and asserts the
judgement.

`NoEvidence` is still reachable and still tested, on a **string** column:
Parquet records bounds for it, but comparing the *width* of two string ranges
would invent a distance that does not exist, so the estimate refuses them.
Types whose values are ordered but not measurable get no automatic index,
which is a real limitation and an honest one.


---

## Phase 18 — Cross-engine telemetry `[x]`

The optimizer only ever saw queries that came through this engine, which in a
real lakehouse is a minority of them. Spark and Trino read the same tables, and
an index built for what *we* happen to serve optimizes for a sample rather than
for the workload.

### Not Iceberg's `ScanReport`, because there isn't one

`iceberg-rust` 0.6 has no metrics reporting at all — no `ScanReport`, no
`MetricsReporter`. There was nothing to adapt from, and coupling to an absent
API would have been the wrong shape anyway: the point is to accept telemetry
from *any* engine. `ForeignScan` mirrors the fields Iceberg's REST
`report-metrics` payload carries, so a handler for it can populate this
directly, without the type being Iceberg's.

### Two things a foreign engine does not have to be trusted with

```text
literals        a Fingerprint keeps only WHICH fields were restricted and
                whether an equality could probe them. Nothing depends on
                another engine hashing values the way this one does, and no
                value crosses the boundary.

the baseline    bytes_if_full_scan comes from the table's own file sizes,
                not from the report. It is what every saving is measured
                against, so a reporter that under-states what it read
                cannot inflate a saving.
```

And `used` is always `None`: another engine cannot have been served by derived
state it does not know about. If foreign scans could credit an index, a useless
one would look busy and survive retirement forever.

### The property it all rests on

A scan reported by another engine must land in the *same* shape as the identical
query run here. If it did not, cross-engine telemetry would aggregate nothing
and the optimizer would still see only its own traffic.
`a_foreign_scan_shares_a_shape_with_an_identical_local_query` asserts the
fingerprints are equal and that two observations collapse to one shape.

`another_engines_traffic_alone_justifies_an_index` is the end-to-end case: not a
single query runs through this engine, and a round still builds a real index
that a later local query uses.


---

## Phase 19 — Calibration `[x]`

The last three invented constants, against rates published in January 2026
(AWS, US East N. Virginia):

```text
                  guessed    derived    verdict
hot_byte_usd       1e-11     4.0e-13    25x too high
cold_multiplier    1000      26         38x too high
near_multiplier    2.0       1.0        in-region transfer is free
far_multiplier     100       51         about right (cross-region)
byte_day_usd       7e-13     7.67e-13   about right
```

`hot_byte_usd` is a GET at $0.0004 per 1,000 spread over a 1 MB read, because
same-region transfer from S3 to compute is not billed at all. `cold_multiplier`
adds $0.01 per GB of retrieval. `far_multiplier` adds egress: $0.02 per GB
cross-region, $0.09 to the internet.

`PriceTable::from_rates` takes the five numbers a user can read off an invoice —
storage, GET rate, average read size, egress, cold retrieval — so nobody has to
understand the multipliers to replace them. `aws_s3_same_region`,
`aws_s3_cross_region` and `aws_s3_internet` are that function with published
rates filled in, and `Default` is now the first of them rather than a guess.

**The `cold_multiplier` error was the one worth catching.** At 1000x, anything
priced in money would refuse to read cold data under practically any
circumstances. The real ratio is about 26: a reason to prefer hot data, not a
reason never to touch cold.

### A design assumption turned out to be false

Three tests asserted that distance costs money, and two more that reporting
locality makes reads cheaper. All five failed, because **AWS bills nothing for
transfer from S3 to compute in the same region**, whatever availability zone
either is in.

So by default `Local`, `Near` and `Far` are the same price, and the reason to
prefer local data there is **latency**, which `Cost` does not model. That is a
real gap in the cost model, and pricing distance as though money were the reason
would have hidden it behind a number nobody could defend.

The tests now assert what is true — distance is free within a region, costs
money once bytes leave it — and the two that need distance to be priced say so
and use `aws_s3_internet`. `a_deployment_can_price_distance_however_it_likes`
covers the on-premises case, where cross-rack traffic contends for a shared
uplink even though nobody sends an invoice for it.

### Absolute figures are now plausible

Reading a 130 MB table ten times costs about half a thousandth of a dollar,
dominated by request charges. Under the guessed table it was $1.20. The
*decisions* are unchanged — all four measured regimes still get the same
verdict — because those turn on ratios between candidates, which is why the
constants mattered less than the structural errors found earlier.
