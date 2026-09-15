//! HyperLogLog sketches: mergeable distinct-count partials.
//!
//! `quarry_hll_state(x)` turns a column's values into register bytes — the
//! partial a sketch cube stores. `quarry_hll_merge(states)` combines stored
//! bytes into the estimate — what `approx_distinct` and, under the
//! `quarry.approximate` opt-in, `count(distinct)` rewrite to.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{Array, BinaryArray};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::ScalarValue;
use datafusion::error::Result as DfResult;
use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::{
    Accumulator, AggregateUDF, AggregateUDFImpl, Expr, Signature, Volatility,
};

/// 2^14 registers — the same shape `approx_distinct`'s sketch has.
const REGISTERS: usize = 1 << 14;
const INDEX_BITS: u32 = 14;

/// Ertl's HyperLogLog: hashed values split into a register index and a
/// leading-zeros rank, registers kept as element-wise maxima.
#[derive(Debug)]
struct Hll {
    registers: [u8; REGISTERS],
}

impl Default for Hll {
    fn default() -> Self {
        Hll {
            registers: [0; REGISTERS],
        }
    }
}

impl Hll {
    fn add(&mut self, value: &ScalarValue) {
        let hash = crate::stable_hash::StableHasher::of(value);
        let index = (hash & (REGISTERS as u64 - 1)) as usize;
        // The rank is the run of zeros after the index bits, plus one.
        let rank = ((hash >> INDEX_BITS) | (1_u64 << (64 - INDEX_BITS))).trailing_zeros() + 1;
        self.registers[index] = self.registers[index].max(rank as u8);
    }

    fn merge(&mut self, other: &[u8]) -> DfResult<()> {
        let other: &[u8; REGISTERS] = other.try_into().map_err(|_| {
            datafusion::error::DataFusionError::Internal(
                "a sketch register array is always 16384 bytes".into(),
            )
        })?;
        for (register, theirs) in self.registers.iter_mut().zip(other) {
            *register = (*register).max(*theirs);
        }
        Ok(())
    }

    /// The estimate, per Ertl's maximum-likelihood method ("New cardinality
    /// estimation algorithms for HyperLogLog sketches", arXiv:1702.01284).
    fn count(&self) -> u64 {
        let mut histogram = [0u32; 64 - INDEX_BITS as usize + 2];
        for register in self.registers {
            histogram[register as usize] += 1;
        }
        let m = REGISTERS as f64;
        let mut z = m * tau((m - histogram[histogram.len() - 1] as f64) / m);
        for h in histogram[1..histogram.len() - 1].iter().rev() {
            z += *h as f64;
            z *= 0.5;
        }
        z += m * sigma(histogram[0] as f64 / m);
        (0.5 / 2_f64.ln() * m * m / z).round() as u64
    }
}

fn sigma(x: f64) -> f64 {
    if x == 1. {
        return f64::INFINITY;
    }
    let (mut y, mut z, mut x) = (1.0, x, x);
    loop {
        x *= x;
        let before = z;
        z += x * y;
        y += y;
        if before == z {
            return z;
        }
    }
}

fn tau(x: f64) -> f64 {
    if x == 0.0 || x == 1.0 {
        return 0.0;
    }
    let (mut y, mut z, mut x) = (1.0, 1.0 - x, 1.0 - x);
    loop {
        x = x.sqrt();
        let before = z;
        y *= 0.5;
        z -= (1.0 - x).powi(2) * y;
        if before == z {
            return z / 3.0;
        }
    }
}

/// One accumulator serving both functions: `update` feeds values to a state
/// builder and register bytes to a merger; `merge_batch` and `state` are the
/// same bytes either way.
#[derive(Debug)]
struct Sketch {
    hll: Hll,
    /// `evaluate` yields the count (`merge`) or the registers (`state`).
    count: bool,
}

impl Accumulator for Sketch {
    fn update_batch(&mut self, values: &[datafusion::arrow::array::ArrayRef]) -> DfResult<()> {
        if self.count {
            return self.merge_batch(values);
        }
        for i in 0..values[0].len() {
            let value = ScalarValue::try_from_array(values[0].as_ref(), i)?;
            if !value.is_null() {
                self.hll.add(&value);
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[datafusion::arrow::array::ArrayRef]) -> DfResult<()> {
        let Some(states) = states[0].as_any().downcast_ref::<BinaryArray>() else {
            return datafusion::common::exec_err!(
                "quarry_hll_merge takes sketch registers, not values"
            );
        };
        for state in states.iter().flatten() {
            self.hll.merge(state)?;
        }
        Ok(())
    }

    fn state(&mut self) -> DfResult<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Binary(Some(self.hll.registers.to_vec()))])
    }

    fn evaluate(&mut self) -> DfResult<ScalarValue> {
        if self.count {
            Ok(ScalarValue::UInt64(Some(self.hll.count())))
        } else {
            Ok(ScalarValue::Binary(Some(self.hll.registers.to_vec())))
        }
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.hll.registers.len()
    }
}

/// `x` → sketch registers, for building a cube's stored partials.
#[derive(Debug)]
struct HllState {
    signature: Signature,
}

/// Sketch registers → the merged estimate, for serving from stored partials.
#[derive(Debug)]
struct HllMerge {
    signature: Signature,
}

impl AggregateUDFImpl for HllState {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "quarry_hll_state"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> DfResult<DataType> {
        Ok(DataType::Binary)
    }

    fn accumulator(&self, _args: AccumulatorArgs) -> DfResult<Box<dyn Accumulator>> {
        Ok(Box::new(Sketch {
            hll: Hll::default(),
            count: false,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> DfResult<Vec<FieldRef>> {
        Ok(vec![Arc::new(Field::new(
            format!("{}_state", args.name),
            DataType::Binary,
            true,
        ))])
    }
}

impl AggregateUDFImpl for HllMerge {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "quarry_hll_merge"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> DfResult<DataType> {
        Ok(DataType::UInt64)
    }

    fn accumulator(&self, _args: AccumulatorArgs) -> DfResult<Box<dyn Accumulator>> {
        Ok(Box::new(Sketch {
            hll: Hll::default(),
            count: true,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs) -> DfResult<Vec<FieldRef>> {
        Ok(vec![Arc::new(Field::new(
            format!("{}_state", args.name),
            DataType::Binary,
            true,
        ))])
    }
}

/// `quarry_hll_merge(stored)` as an expression — what a sketch's partials
/// re-aggregate through.
pub(crate) fn merge_expr(arg: Expr) -> Expr {
    Expr::AggregateFunction(AggregateFunction::new_udf(
        Arc::new(AggregateUDF::new_from_impl(HllMerge {
            signature: Signature::exact(vec![DataType::Binary], Volatility::Immutable),
        })),
        vec![arg],
        false,
        None,
        None,
        None,
    ))
}

/// Register both functions on a session context, so `quarry_hll_state` is
/// nameable in build SQL.
pub(crate) fn register(ctx: &datafusion::prelude::SessionContext) {
    ctx.register_udaf(AggregateUDF::new_from_impl(HllState {
        signature: Signature::any(1, Volatility::Immutable),
    }));
    ctx.register_udaf(AggregateUDF::new_from_impl(HllMerge {
        signature: Signature::exact(vec![DataType::Binary], Volatility::Immutable),
    }));
}
