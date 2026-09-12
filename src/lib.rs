//! Quarry is a self-optimizing query engine over Iceberg tables on object
//! storage.
//!
//! The model has three concepts and one rule:
//!
//! - **Table** — an Iceberg table. Authoritative. Never modified by the
//!   optimizer.
//! - **Derived state** — bytes computed from a table at a known snapshot that
//!   can answer some queries more cheaply than the table can. Always
//!   disposable: losing all of it costs performance, never correctness.
//! - **Cost** — one currency for every decision.
//!
//! The rule decides when derived state may replace a scan. It is checked in
//! one place, for every kind of derived state, so that a wrong answer is
//! prevented by construction rather than by care.
//!
//! See `../core-design.md` for the design this implements, and `DEVPLAN.md`
//! for the phase this code is currently in.

#![forbid(unsafe_code)]

pub mod cost;
pub mod derived;
pub mod place;
pub mod snapshot;
