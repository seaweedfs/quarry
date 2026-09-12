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

Phases 0–8 of [DEVPLAN.md](DEVPLAN.md) are done: the whole decision core, with
89 tests and no external dependencies. Nothing executes SQL yet — that is
phase 9, where DataFusion and Iceberg arrive.

What works today:

```sh
cargo run --example explain
```

walks one table through append, delete and compaction and explains the same
query at each snapshot, showing the rule admit and refuse derived state as the
table moves under it.

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
examples/
  explain.rs    end-to-end walkthrough
```

The file to read first is `derived.rs`. `Derived::may_serve` is the only place
that decides whether derived state may replace a scan, for every kind, so it
is where the correctness of the whole engine lives.

## Development

```sh
cargo test
cargo clippy --all-targets
cargo fmt --check
cargo doc --no-deps        # must be warning-free: broken links rot silently
```

## License

Apache-2.0
