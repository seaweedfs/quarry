//! Walks a table through four commits and explains the same query at each,
//! showing how the rule admits and refuses derived state as the table moves.
//!
//! ```sh
//! cargo run --example explain
//! ```

use std::collections::BTreeSet;

use quarry::budget::{Budget, Meter, Permit};
use quarry::cost::{PriceTable, Tier};
use quarry::derived::{Derived, DerivedId, Plan, PolicyFingerprint, Predicate, Query, Source};
use quarry::explain::Explain;
use quarry::kinds::{Index, ResultCache};
use quarry::place::{Distance, Place};
use quarry::registry::Registry;
use quarry::snapshot::{DeleteState, FileId, Snapshot, SnapshotGraph, SnapshotId, TableId};

const TENANT: u32 = 4;
const TENANT_42: u64 = 0x2A; // hash of the literal 42
const POLICY: PolicyFingerprint = PolicyFingerprint(7);

fn file(name: &str) -> FileId {
    FileId(name.to_owned())
}

/// The one plan the cached answer in this example was built for.
fn cached_plan() -> Plan {
    Plan::new(BTreeSet::from([TENANT]), ["tenant_id = 42".to_owned()])
}

fn main() {
    let table = TableId("events".into());
    let prices = PriceTable::default();

    // 810  two files
    // 811  appends c
    // 812  deletes rows from a          <- subtractive
    // 813  compacts a,b,c into merged   <- subtractive
    let graph = SnapshotGraph::new()
        .with(
            Snapshot::root(SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b")),
        )
        .with(
            Snapshot::child_of(SnapshotId(811), SnapshotId(810))
                .with_clean_file(file("a"))
                .with_clean_file(file("b"))
                .with_clean_file(file("c")),
        )
        .with(
            Snapshot::child_of(SnapshotId(812), SnapshotId(811))
                .with_file(file("a"), DeleteState(1))
                .with_clean_file(file("b"))
                .with_clean_file(file("c")),
        )
        .with(Snapshot::child_of(SnapshotId(813), SnapshotId(812)).with_clean_file(file("merged")));

    // Both built from snapshot 810, and never rebuilt.
    let mut registry = Registry::new();
    registry.register(Derived::new(
        DerivedId("tenant_idx".into()),
        Source {
            table: table.clone(),
            snapshot: SnapshotId(810),
        },
        POLICY,
        4096,
        Box::new(
            Index::new(TENANT)
                .with(TENANT_42, file("a"))
                .with_bytes(4096),
        ),
    ));
    registry.register(Derived::new(
        DerivedId("cached_answer".into()),
        Source {
            table: table.clone(),
            snapshot: SnapshotId(810),
        },
        POLICY,
        64,
        Box::new(ResultCache::rows_of(cached_plan(), 3, 64)),
    ));

    // SELECT ... FROM events WHERE tenant_id = 42
    //
    // The plan, not its hash, is what a stored result is matched against: two
    // different plans can share a hash, and serving on that would hand one
    // query another's answer.
    let query_at = |snapshot: i64| Query {
        table: table.clone(),
        snapshot: SnapshotId(snapshot),
        policy: POLICY,
        plan_hash: 0xC0FFEE,
        plan: Some(cached_plan()),
        projected: BTreeSet::from([TENANT]),
        predicates: vec![Predicate::Eq {
            field: TENANT,
            value: TENANT_42,
        }],
        aggregate: None,
            nearest: None,
        approximate: false,
    };

    println!("Derived state built at snapshot 810, never rebuilt.\n");
    for snapshot in [810, 811, 812, 813] {
        let what_happened = match snapshot {
            810 => "as built",
            811 => "appended a file",
            812 => "deleted rows from a file",
            _ => "compacted",
        };
        println!("--- snapshot {snapshot} ({what_happened})");
        println!(
            "{}",
            Explain::plan(&query_at(snapshot), &registry, &graph, &prices)
        );
    }

    // Placement: the same bytes cost very different amounts depending on
    // where the worker is relative to them.
    let worker = Place::parse("/onprem/dc1/rack2/node7");
    println!("--- placement, from {worker}");
    for (label, replica) in [
        ("same node", "/onprem/dc1/rack2/node7"),
        ("same rack", "/onprem/dc1/rack2/node8"),
        ("other rack", "/onprem/dc1/rack9/node1"),
        ("other cloud", "/aws/us-west-2/usw2-az1/i-1"),
    ] {
        let distance = worker.distance(&Place::parse(replica));
        let cost = prices.price(1_000_000_000, Tier::Hot, distance, 0.0);
        println!("{label:<12} {distance:<6?} 1 GB = ${:.6}", cost.usd);
    }

    // Enforcement: a ceiling stops a scan part-way rather than after.
    println!("\n--- a 4 MB budget against ten 1 MB reads");
    let mut meter = Meter::new(Budget::bytes(4_000_000), prices);
    for read in 0..10 {
        if let Permit::Stop(exceeded) = meter.charge(1_000_000, Tier::Hot, Distance::Local) {
            println!("stopped on read {}: {exceeded:?}", read + 1);
            break;
        }
    }
    println!("spent: {} bytes", meter.spent().bytes);
}
