//! `quarry_matches(column, 'terms')`: the full-text predicate, and the one
//! tokenizer that defines what a term is.
//!
//! A text index answers "which files hold this word". That makes the
//! tokenizer load-bearing rather than incidental: a build that splits text
//! one way and a probe that splits the query another way agree on nothing,
//! and the index silently misses documents it holds. So there is exactly one
//! [`terms`] function, and both sides call it — the same discipline
//! [`hash_scalar`](super::hash_scalar) keeps for equality.
//!
//! # What the predicate means
//!
//! `quarry_matches(body, 'error timeout')` is true of a row whose `body`
//! contains **every** token of the argument, each as a whole token. Not a
//! substring: `'err'` does not match `"error"`, because an inverted index
//! cannot answer that without scanning, and claiming otherwise would return
//! wrong rows rather than merely fewer.
//!
//! Conjunctive because that is what a search box means, and because it makes
//! the index probe an intersection — see
//! [`Predicate::Matches`](crate::derived::Predicate::Matches).

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, BooleanBuilder, StringArray};
use datafusion::arrow::datatypes::DataType;
use datafusion::common::{DataFusionError, ScalarValue};
use datafusion::error::Result as DfResult;
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature, Volatility,
};

/// The name a query writes, and the recogniser matches on.
pub(crate) const MATCHES_NAME: &str = "quarry_matches";

/// The tokens of `text`, hashed — the only definition of a term.
///
/// Lowercased and split on anything not alphanumeric, so `"Error: timeout!"`
/// and `"error timeout"` yield the same terms. Deliberately unsophisticated:
/// no stemming, no stop words, no language rules. Each of those changes which
/// documents match, so each belongs in a versioned decision rather than
/// arriving quietly with a dependency upgrade.
///
/// Hashed with [`StableHasher`](crate::stable_hash::StableHasher), because
/// postings are written to storage and a hash that shifts across releases
/// makes a recovered index match nothing.
///
/// Duplicates are kept: the caller decides whether repetition matters. An
/// index inserts each term once per file; a probe intersects, where a repeat
/// is harmless.
pub fn terms(text: &str) -> Vec<u64> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| crate::stable_hash::StableHasher::of(&token.to_lowercase()))
        .collect()
}

/// `quarry_matches(column, 'terms')` — true where every term is present.
#[derive(Debug)]
struct Matches {
    signature: Signature,
}

impl ScalarUDFImpl for Matches {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        MATCHES_NAME
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _args: &[DataType]) -> DfResult<DataType> {
        Ok(DataType::Boolean)
    }

    /// Evaluated over real rows, because pruning is inexact: a file the index
    /// admits still holds rows the predicate rejects, and DataFusion
    /// re-applies this above the scan to drop them.
    ///
    /// A null row is null, not false — three-valued logic, as SQL has it, and
    /// unlike a distance where the ask itself made infinity the right answer.
    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DfResult<ColumnarValue> {
        let [column, query] = args.args.as_slice() else {
            return Err(DataFusionError::Internal(format!(
                "{MATCHES_NAME} takes exactly two arguments"
            )));
        };
        let ColumnarValue::Scalar(
            ScalarValue::Utf8(Some(query))
            | ScalarValue::LargeUtf8(Some(query))
            | ScalarValue::Utf8View(Some(query)),
        ) = query
        else {
            return Err(DataFusionError::Internal(format!(
                "{MATCHES_NAME}'s second argument must be a string literal"
            )));
        };
        let wanted = terms(query);

        let column = match column {
            ColumnarValue::Array(array) => Arc::clone(array),
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(args.number_rows)?,
        };
        let text = strings_of(&column)?;

        let mut out = BooleanBuilder::with_capacity(text.len());
        for row in 0..text.len() {
            if text.is_null(row) {
                out.append_null();
                continue;
            }
            let held = terms(text.value(row));
            out.append_value(wanted.iter().all(|term| held.contains(term)));
        }
        Ok(ColumnarValue::Array(Arc::new(out.finish()) as ArrayRef))
    }
}

/// A string column, whichever of Arrow's string types it is.
fn strings_of(array: &ArrayRef) -> DfResult<StringArray> {
    use datafusion::arrow::compute::cast;

    if let Some(strings) = array.as_any().downcast_ref::<StringArray>() {
        return Ok(strings.clone());
    }
    match array.data_type() {
        DataType::LargeUtf8 | DataType::Utf8View => {
            let cast = cast(array.as_ref(), &DataType::Utf8)?;
            Ok(cast
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("cast to Utf8 yields a StringArray")
                .clone())
        }
        other => Err(DataFusionError::Internal(format!(
            "{MATCHES_NAME} needs a string column, not {other}"
        ))),
    }
}

/// Register the predicate on a session context.
pub(crate) fn register(ctx: &datafusion::prelude::SessionContext) {
    ctx.register_udf(ScalarUDF::new_from_impl(Matches {
        signature: Signature::any(2, Volatility::Immutable),
    }));
}

/// Whether `name` is the full-text predicate.
pub(crate) fn is_matches(name: &str) -> bool {
    name == MATCHES_NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The property the index rests on: what the build stores and what the
    /// probe looks for come from one function, so punctuation and case
    /// cannot make them disagree.
    #[test]
    fn tokenizing_is_case_and_punctuation_insensitive() {
        assert_eq!(terms("Error: timeout!"), terms("error timeout"));
        assert_eq!(terms("a-b_c"), terms("A B C"));
    }

    #[test]
    fn empty_text_has_no_terms() {
        assert!(terms("").is_empty());
        assert!(terms("   ...  ").is_empty());
    }

    #[test]
    fn tokens_are_whole_words_not_substrings() {
        // The whole reason a match is not a LIKE: an inverted index cannot
        // answer a substring without scanning.
        let held = terms("error");
        assert!(!held.contains(&terms("err")[0]));
    }

    #[test]
    fn distinct_words_hash_distinctly() {
        assert_ne!(terms("error"), terms("timeout"));
    }

    #[test]
    fn repetition_is_preserved_for_the_caller_to_decide_about() {
        assert_eq!(terms("a a").len(), 2);
    }

    #[test]
    fn only_our_own_name_is_the_predicate() {
        assert!(is_matches(MATCHES_NAME));
        assert!(!is_matches("matches"));
        assert!(!is_matches("like"));
    }
}
