//! `quarry_l2_distance` and `quarry_cosine_distance`: distance between a
//! vector column and a query vector, as scalar functions DataFusion plans and
//! executes.
//!
//! The functions the ANN plan shape is recognised by. A query writes
//! `ORDER BY quarry_l2_distance(embedding, [1.0, 2.0, 3.0]) LIMIT 10`, the
//! recogniser reads the column and the metric from the expression and the
//! dimension from the table's schema, and the `Sort`+`Limit` above a
//! substituted scan re-ranks rows by exactly this function — so the UDF is
//! real arithmetic, not a marker the rewrite pattern-matches and discards.
//!
//! # Why the signature is permissive
//!
//! `Signature::any(2)` rather than an exact one. A vector column is a
//! `FixedSizeList<Float32, d>`, but a SQL literal `[1.0, 2.0, 3.0]` arrives
//! as a `List<Float64>`, and `d` differs per table — so no single exact
//! signature admits the shapes that must work. The types are checked when
//! the function runs, where the real dimensions are in hand, and a mismatch
//! is an error rather than a silently wrong distance.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float32Array, Float64Array};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{DataFusionError, ScalarValue};
use datafusion::error::Result as DfResult;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

use crate::derived::Metric;

/// `quarry_l2_distance`, the squared Euclidean distance.
pub(crate) const L2_NAME: &str = "quarry_l2_distance";
/// `quarry_cosine_distance`, one minus cosine similarity.
pub(crate) const COSINE_NAME: &str = "quarry_cosine_distance";

/// The metric a function name denotes, or `None` if it is not one of ours.
///
/// The recogniser's only way in: a `Sort` on any other function is a sort
/// this engine does not rewrite.
pub(crate) fn metric_of(name: &str) -> Option<Metric> {
    match name {
        L2_NAME => Some(Metric::L2),
        COSINE_NAME => Some(Metric::Cosine),
        _ => None,
    }
}

/// A distance function over two vectors.
#[derive(Debug)]
struct Distance {
    metric: Metric,
    signature: Signature,
}

impl Distance {
    fn of(metric: Metric) -> Self {
        Distance {
            metric,
            // Permissive by necessity — see the module comment.
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for Distance {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        match self.metric {
            Metric::L2 => L2_NAME,
            Metric::Cosine => COSINE_NAME,
        }
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> DfResult<DataType> {
        Ok(DataType::Float64)
    }

    /// One query vector against every row's.
    ///
    /// A null row scores infinity rather than null: the ask is "the nearest
    /// k", and a row with no vector is nowhere near anything. Sorting nulls
    /// would depend on `NULLS FIRST`, which is not what the ask means.
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DfResult<ColumnarValue> {
        let [left, right] = args.args.as_slice() else {
            return Err(DataFusionError::Internal(format!(
                "{} takes exactly two arguments",
                self.name()
            )));
        };

        // The query vector is whichever side is a single value; when both are
        // arrays the second is taken as the query and read once, which is
        // what a literal in a sort expression reduces to.
        let query = match right {
            ColumnarValue::Scalar(scalar) => vector_of_scalar(scalar)?,
            ColumnarValue::Array(array) => vector_at(array, 0)?,
        };
        let column = match left {
            ColumnarValue::Array(array) => Arc::clone(array),
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(args.number_rows)?,
        };

        let metric = self.metric;
        let mut distances = Vec::with_capacity(column.len());
        for row in 0..column.len() {
            if column.is_null(row) {
                distances.push(f64::INFINITY);
                continue;
            }
            let vector = vector_at(&column, row)?;
            if vector.len() != query.len() {
                return Err(DataFusionError::Internal(format!(
                    "{}: dimension mismatch ({} vs {})",
                    self.name(),
                    vector.len(),
                    query.len()
                )));
            }
            distances.push(distance(metric, &vector, &query));
        }
        Ok(ColumnarValue::Array(Arc::new(Float64Array::from(
            distances,
        ))))
    }
}

/// The distance between two vectors of equal length.
///
/// L2 is left **squared**: the square root is monotone, so it does not change
/// a top-k order, and skipping it keeps the arithmetic exact for integers.
pub(crate) fn distance(metric: Metric, one: &[f64], other: &[f64]) -> f64 {
    match metric {
        Metric::L2 => one
            .iter()
            .zip(other)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>(),
        Metric::Cosine => {
            let dot: f64 = one.iter().zip(other).map(|(a, b)| a * b).sum();
            let left: f64 = one.iter().map(|a| a * a).sum::<f64>().sqrt();
            let right: f64 = other.iter().map(|b| b * b).sum::<f64>().sqrt();
            if left == 0.0 || right == 0.0 {
                // An undefined angle: no direction to be near. Farthest,
                // for the same reason a null row is.
                return f64::INFINITY;
            }
            1.0 - dot / (left * right)
        }
    }
}

/// One row of a list-shaped array, as `f64`s.
///
/// `FixedSizeList` is what a vector column is; `List` is what a SQL literal
/// becomes. Elements may be `Float32` (the stored form) or `Float64` (a
/// literal's). Anything else is an error rather than a coercion, because a
/// silently wrong distance returns silently wrong neighbours.
pub(crate) fn vector_at(array: &ArrayRef, row: usize) -> DfResult<Vec<f64>> {
    use datafusion::arrow::array::{FixedSizeListArray, ListArray};

    let values = if let Some(fixed) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        fixed.value(row)
    } else if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        list.value(row)
    } else {
        return Err(DataFusionError::Internal(format!(
            "a vector must be a list, not {}",
            array.data_type()
        )));
    };
    floats_of(&values)
}

/// A scalar list literal, as `f64`s.
fn vector_of_scalar(scalar: &ScalarValue) -> DfResult<Vec<f64>> {
    let array = scalar.to_array()?;
    vector_at(&array, 0)
}

fn floats_of(values: &ArrayRef) -> DfResult<Vec<f64>> {
    if let Some(f32s) = values.as_any().downcast_ref::<Float32Array>() {
        return Ok(f32s.values().iter().map(|v| *v as f64).collect());
    }
    if let Some(f64s) = values.as_any().downcast_ref::<Float64Array>() {
        return Ok(f64s.values().to_vec());
    }
    Err(DataFusionError::Internal(format!(
        "vector elements must be Float32 or Float64, not {}",
        values.data_type()
    )))
}

/// The dimension a `FixedSizeList` column holds, or `None` for any other type.
///
/// The recogniser reads the dimension from the table's schema rather than
/// from the query's literal, so a query whose vector is the wrong length is
/// refused by the index's own dimension check instead of building an ask that
/// cannot be served.
pub(crate) fn dimension_of(data_type: &DataType) -> Option<u32> {
    match data_type {
        DataType::FixedSizeList(_, size) => u32::try_from(*size).ok(),
        _ => None,
    }
}

/// Register both distance functions on a session context.
pub(crate) fn register(ctx: &datafusion::prelude::SessionContext) {
    ctx.register_udf(ScalarUDF::new_from_impl(Distance::of(Metric::L2)));
    ctx.register_udf(ScalarUDF::new_from_impl(Distance::of(Metric::Cosine)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_is_squared_so_the_order_is_unchanged() {
        assert_eq!(distance(Metric::L2, &[0.0, 0.0], &[3.0, 4.0]), 25.0);
        assert_eq!(distance(Metric::L2, &[1.0], &[1.0]), 0.0);
    }

    #[test]
    fn cosine_is_zero_for_parallel_and_one_for_orthogonal() {
        assert!(distance(Metric::Cosine, &[1.0, 0.0], &[2.0, 0.0]).abs() < 1e-12);
        assert!((distance(Metric::Cosine, &[1.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_zero_vector_has_no_direction_so_it_is_farthest() {
        assert_eq!(
            distance(Metric::Cosine, &[0.0, 0.0], &[1.0, 1.0]),
            f64::INFINITY
        );
    }

    #[test]
    fn only_our_own_function_names_denote_a_metric() {
        assert_eq!(metric_of(L2_NAME), Some(Metric::L2));
        assert_eq!(metric_of(COSINE_NAME), Some(Metric::Cosine));
        assert_eq!(metric_of("abs"), None);
        assert_eq!(metric_of("l2_distance"), None);
    }

    #[test]
    fn a_dimension_comes_only_from_a_fixed_size_list() {
        use datafusion::arrow::datatypes::Field;

        let vector =
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), 3);
        assert_eq!(dimension_of(&vector), Some(3));
        // A variable-length list has no dimension an index could be built for.
        let list = DataType::List(Arc::new(Field::new("item", DataType::Float32, true)));
        assert_eq!(dimension_of(&list), None);
        assert_eq!(dimension_of(&DataType::Int64), None);
    }
}
