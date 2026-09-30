//! Confidence guardrails for prediction results (port of `laya/confidence.py`).
//!
//! Two helpers used by [`crate::router::Router`]: [`check_min_confidence`] validates
//! the opt-in abstention threshold a caller passes, and [`flag_low_confidence`] walks
//! `system_one` payloads marking answers whose confidence falls below it. Flagging is
//! best-effort: payloads without an `answers` object are skipped, as in laya.

use serde_json::Value;

use crate::{LuminaError, Result};

/// Validate an opt-in `min_confidence` threshold.
///
/// Must be a real number in `[0.0, 1.0]`; the error text is laya's.
///
/// # Errors
/// [`LuminaError::Invalid`] naming the rejected value.
pub fn check_min_confidence(value: f64) -> Result<f64> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(LuminaError::Invalid(format!(
            "min_confidence must be a float in [0.0, 1.0], got {}",
            py_float_repr(value)
        )))
    }
}

/// Python's `repr` for an f64: `nan` (Rust's `{:?}` says `NaN`); every other value
/// agrees (`1.1`, `0.0`, `inf`, `-inf`).
fn py_float_repr(value: f64) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else {
        format!("{value:?}")
    }
}

/// Flag answers whose confidence falls below `min_confidence`.
///
/// Reads `answer_confidence` — the calibrated confidence, invariant to label count —
/// falling back to `confidence` when it is absent or null. The raw answer and its
/// confidence stay intact; `low_confidence: true` is added. A threshold of 0.0 is a
/// no-op.
pub fn flag_low_confidence(results: &mut [Value], min_confidence: f64) {
    if min_confidence == 0.0 {
        return;
    }
    for result in results {
        flag_result(result, min_confidence);
    }
}

/// Flag one payload's answers.
fn flag_result(result: &mut Value, min_confidence: f64) {
    let Some(answers) = result.get_mut("answers") else {
        return;
    };
    let Some(answers) = answers.as_object_mut() else {
        return;
    };
    for answer in answers.values_mut() {
        let Some(answer) = answer.as_object_mut() else {
            continue;
        };
        // `answer_confidence` wins; an absent or null one falls back to `confidence`.
        let confidence = match answer.get("answer_confidence") {
            None | Some(Value::Null) => answer.get("confidence"),
            Some(v) => Some(v),
        };
        // `as_f64` rejects bools, strings and null, matching laya's isinstance check.
        let Some(conf) = confidence.and_then(Value::as_f64) else {
            continue;
        };
        if conf.is_finite() && conf < min_confidence {
            answer.insert("low_confidence".to_owned(), Value::Bool(true));
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use serde_json::json;

    use super::*;

    #[test]
    fn accepts_bounds_and_rejects_out_of_range() {
        assert_eq!(check_min_confidence(0.0).unwrap(), 0.0);
        assert_eq!(check_min_confidence(0.5).unwrap(), 0.5);
        assert_eq!(check_min_confidence(1.0).unwrap(), 1.0);
        let err = check_min_confidence(1.1).unwrap_err().to_string();
        assert_eq!(err, "min_confidence must be a float in [0.0, 1.0], got 1.1");
        assert_eq!(
            check_min_confidence(-0.1).unwrap_err().to_string(),
            "min_confidence must be a float in [0.0, 1.0], got -0.1"
        );
        assert_eq!(
            check_min_confidence(f64::INFINITY).unwrap_err().to_string(),
            "min_confidence must be a float in [0.0, 1.0], got inf"
        );
        assert_eq!(
            check_min_confidence(f64::NAN).unwrap_err().to_string(),
            "min_confidence must be a float in [0.0, 1.0], got nan"
        );
    }

    #[test]
    fn flags_below_threshold_with_fallback() {
        let mut results = vec![json!({
            "answers": {
                "a": {"answer": "x", "confidence": 0.10},
                "b": {"answer": "y", "answer_confidence": 0.99, "confidence": 0.10},
                "c": {"answer": "z", "answer_confidence": null, "confidence": 0.2},
                "d": {"answer": "w"},
                "e": {"answer": "v", "confidence": 0.9}
            }
        })];
        flag_low_confidence(&mut results, 0.5);
        let answers = results[0]["answers"].as_object().unwrap();
        assert_eq!(answers["a"]["low_confidence"], json!(true));
        // `answer_confidence` wins when present and numeric
        assert!(answers["b"].get("low_confidence").is_none());
        // a null `answer_confidence` falls back to `confidence`
        assert_eq!(answers["c"]["low_confidence"], json!(true));
        // no confidence at all: no flag, no panic
        assert!(answers["d"].get("low_confidence").is_none());
        assert!(answers["e"].get("low_confidence").is_none());
    }

    #[test]
    fn skips_malformed_payloads_and_zero_threshold() {
        let mut results = vec![
            json!("not a dict"),
            json!({"answers": null}),
            json!({"answers": [1, 2]}),
            json!({"answers": {"a": "not a dict"}, "other": {"confidence": 0.1}}),
            json!({"answers": {"a": {"confidence": 0.1}}}),
        ];
        flag_low_confidence(&mut results, 0.9);
        // the well-formed entry was flagged; the others were skipped
        assert_eq!(results[4]["answers"]["a"]["low_confidence"], json!(true));
        assert!(results[3]["answers"]["a"].is_string());
        assert_eq!(results[3]["other"]["confidence"], json!(0.1));

        // a threshold of 0.0 is a no-op even for zero confidence
        let mut zero = vec![json!({"answers": {"a": {"confidence": 0.0}}})];
        flag_low_confidence(&mut zero, 0.0);
        assert!(zero[0]["answers"]["a"].get("low_confidence").is_none());
    }

    #[test]
    fn bools_strings_and_null_are_not_confidence() {
        let mut results = vec![json!({"answers": {
            "a": {"confidence": true},
            "b": {"confidence": "0.1"},
            "c": {"confidence": null},
            "d": {"answer_confidence": false}
        }})];
        flag_low_confidence(&mut results, 0.5);
        let answers = results[0]["answers"].as_object().unwrap();
        for answer in answers.values() {
            assert!(answer.get("low_confidence").is_none());
        }
    }
}
