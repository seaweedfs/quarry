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

Early. See [DEVPLAN.md](DEVPLAN.md) for the phase plan and what is done.

The design is in [`../core-design.md`](../core-design.md); the longer documents
beside it are rationale and detail.

## Layout

```text
src/
  lib.rs        crate docs and module wiring
```

## Development

```sh
cargo test
cargo clippy --all-targets
cargo fmt --check
cargo doc --no-deps        # must be warning-free: broken links rot silently
```

## License

Apache-2.0
