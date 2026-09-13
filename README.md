# Quarry

A self-optimizing query engine over Iceberg tables on object storage.

Ordinary SQL gets faster as the tables are used, because the engine keeps a
registry of **derived state** — cached results, indexes, projections,
pre-aggregates — and rewrites queries to use it when one correctness rule
permits.

```text
Iceberg tables on object storage are the only authoritative data.
Everything else is derived, priced, and disposable.
```

## Status

Phases 0–8 of [DEVPLAN.md](DEVPLAN.md) are done; 9 and 10 are under way. The
decision core is 112 tests with **no dependencies**. Optional features add 87
more: `engine` brings DataFusion and SQL over Parquet, `iceberg` brings real
table metadata, and `rest-catalog` queries a table loaded from a live Iceberg
REST catalog over HTTP.

Real SQL is planned by the rule today:

```sh
cargo test --features rest-catalog     # 199 tests, incl. a live REST catalog
cargo run --example explain            # no dependencies; a table over 4 commits
```

`SELECT * FROM events WHERE tenant_id = 1`, over a real Iceberg table, reads
only the Parquet objects an index says can match — measurably fewer bytes off the store — and still returns
every row when that index is stale, is refused outright when it was built on a
rolled-back branch, and never opens a file that compaction removed. A byte
budget aborts the scan rather than quietly returning fewer rows. Run the same
query twice and the second fetches zero bytes from storage.

And the loop runs itself. Set a policy and the optimizer watches what queries
ask for, proposes the index that would help, builds it by reading only that
column, registers it against the live table, credits it with what it measurably
saved, rebuilds it once the table has grown past it, and retires derived state
that has not paid for keeping it:

```rust
let mut optimizer = Optimizer::new(registry, Policy::automatic_pct(table_bytes, 5.0));
optimizer.observe(report.observation(bytes_read));   // per query
let round = optimizer.round(&session, &table).await; // periodically
```

The design is in [`../core-design.md`](../core-design.md); the longer documents
beside it are rationale and detail. Two corrections to the design were found by
implementing it, both recorded in DEVPLAN phases 3 and 4.

## Layout

```text
src/
  lib.rs        crate docs and module wiring
  place.rs      where things are; distance from a path prefix
  cost.rs       one currency: bytes, cpu-seconds, money
  snapshot.rs   table snapshots, lineage, classified diffs
  derived.rs    derived state, the Kind trait, and THE ONE RULE
  registry.rs   what exists, and which piece to use
  budget.rs     ceilings that stop execution, not estimates
  explain.rs    what a query will cost and why
  workload.rs   what queries asked for, what to build, what to retire; Policy
  facts.rs      what a backend can say about tiers and placement
  from_iceberg.rs   a SnapshotGraph from real Iceberg metadata (--features iceberg)
  kinds/
    result_cache.rs   a stored answer to one exact query (substituting)
    index.rs          equality on one field, prunes files (pruning)
  engine/             behind --features engine
    table.rs          a DataFusion TableProvider planned by the rule
    iceberg_table.rs  a QuarryTable over a real Iceberg table
    build.rs          builds the index the loop proposed, by reading the data
    optimizer.rs      runs the loop: retire, refresh, build, enforce the budget
    materialized.rs   a Kind holding Arrow batches, defined outside the core
    store.rs          an object store that counts bytes and enforces budgets
    cache.rs          a read-through cache for ranges of immutable objects
    quarry.rs         Quarry and Session: the assembled stack
examples/
  explain.rs    end-to-end walkthrough
tests/
  engine_sql.rs     SQL through DataFusion, asserting which files were read
  engine_parquet.rs real Parquet on an object store, selected by the rule
  iceberg_bridge.rs a genuine Iceberg table: metadata, manifest lists, manifests
  iceberg_sql.rs    SQL over that table, with the rule choosing objects
  iceberg_rest.rs   the same, loaded from a REST catalog over a real socket
  loop_closes.rs    the optimizer driving itself against a live table
```

The file to read first is `derived.rs`. `Derived::may_serve` is the only place
that decides whether derived state may replace a scan, for every kind, so it
is where the correctness of the whole engine lives.

## Development

```sh
cargo test                                    # core, no dependencies, instant
cargo test --features engine                  # adds the DataFusion tests
cargo clippy --all-targets --features engine
cargo fmt --check
cargo doc --no-deps --features engine         # must be warning-free
```

## License

Apache-2.0
