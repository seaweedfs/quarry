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

Phases 0–8 of [DEVPLAN.md](DEVPLAN.md) are done, and phase 9 is under way. The
decision core is 89 tests with **no dependencies**; DataFusion sits behind
`--features engine` and adds 10 end-to-end SQL tests.

Real SQL is planned by the rule today:

```sh
cargo test --features engine        # 99 tests, including SQL through DataFusion
cargo run --example explain         # no dependencies; walks a table over 4 commits
```

`SELECT * FROM events WHERE tenant_id = 1` reads only the files an index says
can match — and still returns every row when that index is stale, is refused
outright when it was built on a rolled-back branch, and never reads a file that
compaction removed.

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
  kinds/
    result_cache.rs   a stored answer to one exact query (substituting)
    index.rs          equality on one field, prunes files (pruning)
  engine/             behind --features engine
    table.rs          a DataFusion TableProvider planned by the rule
examples/
  explain.rs    end-to-end walkthrough
tests/
  engine_sql.rs SQL through DataFusion, asserting which files were read
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
