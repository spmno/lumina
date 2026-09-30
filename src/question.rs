//! Question definitions: validation, normalization, and option rendering.

use serde_json::{Map, Value};

use crate::pyjson::{py_repr, py_str, py_type_name, to_python_json};
use crate::{LuminaError, Result};

/// The three question types, in `type_emb` and temperature-index order
/// (choice 0, score 1, noul 2 — laya's `QTYPES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuestionType {
    /// Pick one label from a dict or list of criteria.
    Choice,
    /// Pick a level on an ordered scale.
    Score,
    /// Boolean: the answer reports P(true).
    Noul,
}

impl QuestionType {
    /// The `type_emb` id and temperature index.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }

    /// Wire name, also the head text prefix (`"choice question: …"`).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    fn from_name(s: &str) -> Option<Self> {
        match s {
            "choice" => Some(Self::Choice),
            "score" => Some(Self::Score),
            "noul" => Some(Self::Noul),
            _ => None,
        }
    }
}

/// A choice label under Python dict-key equality: `1`, `1.0` and `true` are one key,
/// `"1"` is another. Labels too large for f64 never compare equal (documented edge).
#[derive(Debug, Clone, PartialEq)]
enum LabelKey {
    Str(String),
    Num(f64),
}

impl LabelKey {
    fn of(v: &Value) -> Self {
        match v {
            Value::String(s) => Self::Str(s.clone()),
            // Python bools hash and compare equal to 0.0 / 1.0.
            Value::Bool(b) => Self::Num(if *b { 1.0 } else { 0.0 }),
            Value::Number(n) => Self::Num(n.as_f64().unwrap_or(f64::NAN)),
            _ => Self::Num(f64::NAN),
        }
    }
}

/// Criteria in the shape each question type expects, after `_to_internal` normalization.
#[derive(Debug, Clone)]
pub enum Criteria {
    /// `(label, criterion)` pairs in definition order; a null or empty-string criterion
    /// renders the label alone. The list form normalizes to null criteria.
    Choice(Vec<(Value, Value)>),
    /// Level descriptions, index 0 first.
    Score(Vec<Value>),
    /// Optional `false` / `true` descriptions (`Value::Null` when absent).
    Noul {
        /// Description shown for the `false` option.
        false_crit: Value,
        /// Description shown for the `true` option.
        true_crit: Value,
    },
}

/// One validated and normalized question — laya's `_check_question` + `_to_internal`.
#[derive(Debug, Clone)]
pub struct Question {
    /// Question type.
    pub qtype: QuestionType,
    /// Instructions text; a non-string input was JSON-dumped, as laya does.
    pub instructions: String,
    /// Criteria in the shape the type expects.
    pub criteria: Criteria,
    /// Noul display labels `(false, true)`; `None` keeps laya's `false` / `true`.
    pub noul_labels: Option<(String, String)>,
}

impl Question {
    /// Validate and normalize a question definition.
    ///
    /// # Errors
    /// Returns [`LuminaError::Question`] naming `qid` and what to fix, mirroring laya's
    /// `_check_question` messages.
    pub fn from_def(qid: &str, qdef: &Value) -> Result<Self> {
        let err = |msg: String| LuminaError::Question {
            qid: qid.to_owned(),
            msg,
        };
        let obj = qdef.as_object().ok_or_else(|| {
            err(format!(
                "definition must be a dict, got {}",
                py_type_name(qdef)
            ))
        })?;
        let type_val = obj.get("type").unwrap_or(&Value::Null);
        let qtype = type_val
            .as_str()
            .and_then(QuestionType::from_name)
            .ok_or_else(|| {
                err(format!(
                    "unknown type {}; use one of ['choice', 'noul', 'score']",
                    py_repr(type_val)
                ))
            })?;
        if !obj.contains_key("instructions") {
            return Err(err(
                "no 'instructions'; add the text the model should answer".into(),
            ));
        }
        let instructions = match obj.get("instructions") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => to_python_json(other)?,
            None => Value::Null.to_string(),
        };
        let criteria = match qtype {
            QuestionType::Choice => choice_criteria(qid, obj)?,
            QuestionType::Score => score_criteria(qid, obj)?,
            QuestionType::Noul => noul_criteria(qid, obj)?,
        };
        let noul_labels = match obj.get("labels") {
            Some(labels) => {
                if qtype != QuestionType::Noul {
                    return Err(err("'labels' is only supported for noul questions".into()));
                }
                Some(resolve_noul_labels(qid, labels)?)
            }
            None => None,
        };
        Ok(Self {
            qtype,
            instructions,
            criteria,
            noul_labels,
        })
    }

    /// Rendered option texts in label order — laya's `render_options`.
    #[must_use]
    pub fn options(&self) -> Vec<String> {
        match &self.criteria {
            Criteria::Choice(entries) => entries
                .iter()
                .map(|(label, crit)| {
                    if crit.is_null() || crit.as_str() == Some("") {
                        py_str(label)
                    } else {
                        format!("{}: {}", py_str(label), render_criterion(crit))
                    }
                })
                .collect(),
            Criteria::Score(levels) => levels
                .iter()
                .enumerate()
                .map(|(i, c)| format!("level {i}: {}", render_criterion(c)))
                .collect(),
            Criteria::Noul {
                false_crit,
                true_crit,
            } => {
                let (f, t) = self.noul_labels();
                vec![
                    format!("{f}: {}", noul_crit_text(false_crit, false)),
                    format!("{t}: {}", noul_crit_text(true_crit, true)),
                ]
            }
        }
    }

    /// Effective noul labels: the caller's, or laya's `false` / `true` defaults.
    #[must_use]
    pub fn noul_labels(&self) -> (String, String) {
        self.noul_labels
            .clone()
            .unwrap_or_else(|| ("false".to_owned(), "true".to_owned()))
    }
}

fn choice_criteria(qid: &str, obj: &Map<String, Value>) -> Result<Criteria> {
    let err = |msg: String| LuminaError::Question {
        qid: qid.to_owned(),
        msg,
    };
    match obj.get("criteria") {
        Some(Value::Object(map)) => {
            if map.is_empty() {
                return Err(err("a choice question needs at least one criterion".into()));
            }
            Ok(Criteria::Choice(
                map.iter()
                    .map(|(k, v)| (Value::String(k.clone()), v.clone()))
                    .collect(),
            ))
        }
        Some(Value::Array(list)) => {
            if list.is_empty() {
                return Err(err("a choice question needs at least one criterion".into()));
            }
            let mut seen: Vec<LabelKey> = Vec::with_capacity(list.len());
            let mut entries = Vec::with_capacity(list.len());
            for (i, label) in list.iter().enumerate() {
                if label.is_array() || label.is_object() {
                    return Err(err(format!(
                        "choice label {i} is a {}; a label is rendered as option text and used \
                         as the answer key, so it must be a scalar (a string, number or bool), \
                         got {}",
                        py_type_name(label),
                        py_repr(label)
                    )));
                }
                if label.is_null() {
                    return Err(err(format!(
                        "choice label {i} is null; a label is rendered as option text and used \
                         as the answer key, so it must be a string, number or bool -- a null \
                         label renders as the text \"None\" while its answer key is \"null\""
                    )));
                }
                let key = LabelKey::of(label);
                if let Some(first) = seen.iter().position(|k| *k == key) {
                    return Err(err(format!(
                        "choice label {i} ({}) repeats label {first}; the labels are the answer \
                         keys, so every option needs its own (1, 1.0 and True are one key)",
                        py_repr(label)
                    )));
                }
                seen.push(key);
                entries.push((label.clone(), Value::Null));
            }
            Ok(Criteria::Choice(entries))
        }
        _ => Err(err(
            "a choice question takes 'criteria' as a dict of label -> description, or a list of \
             labels"
                .into(),
        )),
    }
}

fn score_criteria(qid: &str, obj: &Map<String, Value>) -> Result<Criteria> {
    let err = |msg: String| LuminaError::Question {
        qid: qid.to_owned(),
        msg,
    };
    let Some(Value::Array(list)) = obj.get("criteria") else {
        return Err(err(
            "a score question takes 'criteria' as a list of level descriptions, index 0 \
             first"
                .into(),
        ));
    };
    if list.is_empty() {
        return Err(err("a score question needs at least one level".into()));
    }
    if let Some(i) = list.iter().position(Value::is_null) {
        return Err(err(format!(
            "score level {i} is null; give every level a description, index 0 first"
        )));
    }
    Ok(Criteria::Score(list.clone()))
}

fn noul_criteria(qid: &str, obj: &Map<String, Value>) -> Result<Criteria> {
    let err = |msg: String| LuminaError::Question {
        qid: qid.to_owned(),
        msg,
    };
    let (false_crit, true_crit) = match obj.get("criteria") {
        None | Some(Value::Null) => (Value::Null, Value::Null),
        Some(Value::Object(map)) => {
            let lowered: Vec<String> = map.keys().map(|k| k.to_lowercase()).collect();
            let mut keys = lowered;
            keys.sort();
            keys.dedup();
            if keys.iter().any(|k| k != "true" && k != "false") {
                let listed = keys
                    .iter()
                    .map(|k| format!("'{k}'"))
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(err(format!(
                    "a noul question takes 'criteria' keyed only 'true'/'false' (either or \
                     both, and omitted is fine), got [{listed}]. Those keys are the option texts \
                     the model reads; any other key was silently dropped and replaced with the \
                     defaults. If you want the answer worded differently, keep 'criteria' keyed \
                     'true'/'false' and set 'labels' instead."
                )));
            }
            (
                map.iter()
                    .find(|(k, _)| k.to_lowercase() == "false")
                    .map_or(Value::Null, |(_, v)| v.clone()),
                map.iter()
                    .find(|(k, _)| k.to_lowercase() == "true")
                    .map_or(Value::Null, |(_, v)| v.clone()),
            )
        }
        Some(_) => {
            return Err(err(
                "a noul question takes 'criteria' as a dict with optional 'true'/'false' \
                 descriptions, or omits it"
                    .into(),
            ));
        }
    };
    Ok(Criteria::Noul {
        false_crit,
        true_crit,
    })
}

fn resolve_noul_labels(qid: &str, v: &Value) -> Result<(String, String)> {
    let err = || LuminaError::Question {
        qid: qid.to_owned(),
        msg: "noul labels must map exactly 'false' and 'true' to distinct non-empty strings"
            .to_owned(),
    };
    let map = v.as_object().ok_or_else(err)?;
    if map.len() != 2 || !map.contains_key("false") || !map.contains_key("true") {
        return Err(err());
    }
    let (Some(Value::String(f)), Some(Value::String(t))) = (map.get("false"), map.get("true"))
    else {
        return Err(err());
    };
    let (f, t) = (f.trim().to_owned(), t.trim().to_owned());
    if f.is_empty() || t.is_empty() || f == t {
        return Err(err());
    }
    Ok((f, t))
}

/// laya's `render_criterion`: strings pass through, structured values become JSON.
fn render_criterion(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        // serde_json::Value is always serializable, so the fallback is unreachable.
        other => to_python_json(other).unwrap_or_default(),
    }
}

fn noul_crit_text(v: &Value, is_true: bool) -> String {
    if v.is_null() || v.as_str() == Some("") {
        if is_true {
            "yes, the statement holds".to_owned()
        } else {
            "no, the statement does not hold".to_owned()
        }
    } else {
        render_criterion(v)
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
    use super::*;
    use serde_json::json;

    fn q(qdef: Value) -> Result<Question> {
        Question::from_def("q1", &qdef)
    }

    #[test]
    fn validates_choice_dict() {
        let question = q(json!({
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payment issues", "tech": null}
        }))
        .unwrap();
        assert_eq!(
            question.options(),
            vec!["billing: Payment issues".to_owned(), "tech".to_owned()]
        );
    }

    #[test]
    fn validates_choice_list_with_python_equality() {
        let question = q(json!({
            "type": "choice",
            "instructions": "Pick",
            "criteria": [1, 1.5, "1", "one"]
        }))
        .unwrap();
        assert_eq!(question.options().len(), 4);
        // 1 and true are one Python key, so the list [1, true] must be rejected
        assert!(q(json!({"type": "choice", "instructions": "x", "criteria": [1, true]})).is_err());
        assert!(q(json!({"type": "choice", "instructions": "x", "criteria": [1, 1]})).is_err());
        assert!(q(json!({"type": "choice", "instructions": "x", "criteria": [1, 1.0]})).is_err());
        assert!(q(json!({"type": "choice", "instructions": "x", "criteria": [null]})).is_err());
        assert!(q(json!({"type": "choice", "instructions": "x", "criteria": [[1]]})).is_err());
        assert!(q(json!({"type": "choice", "instructions": "x", "criteria": []})).is_err());
        assert!(q(json!({"type": "choice", "instructions": "x"})).is_err());
    }

    #[test]
    fn renders_score_and_noul() {
        let score = q(json!({
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["low", "high"]
        }))
        .unwrap();
        assert_eq!(
            score.options(),
            vec!["level 0: low".to_owned(), "level 1: high".to_owned()]
        );
        assert!(q(json!({"type": "score", "instructions": "x", "criteria": [null, "a"]})).is_err());
        assert!(q(json!({"type": "score", "instructions": "x", "criteria": {"a": "b"}})).is_err());

        let noul = q(json!({
            "type": "noul",
            "instructions": "Does it hold?",
            "criteria": {"true": "the statement is satisfied"}
        }))
        .unwrap();
        assert_eq!(
            noul.options(),
            vec![
                "false: no, the statement does not hold".to_owned(),
                "true: the statement is satisfied".to_owned()
            ]
        );
        assert!(
            q(json!({"type": "noul", "instructions": "x", "criteria": {"other": "y"}})).is_err()
        );
    }

    #[test]
    fn noul_labels_validate_and_render() {
        let noul = q(json!({
            "type": "noul",
            "instructions": "Holds?",
            "labels": {"false": "B", "true": "A"}
        }))
        .unwrap();
        assert_eq!(
            noul.options(),
            vec![
                "B: no, the statement does not hold".to_owned(),
                "A: yes, the statement holds".to_owned()
            ]
        );
        assert!(
            q(json!({"type": "choice", "instructions": "x", "criteria": {"a": "b"}, "labels": {"false": "B", "true": "A"}})).is_err()
        );
        assert!(
            q(json!({"type": "noul", "instructions": "x", "labels": {"false": "A", "true": "A"}}))
                .is_err()
        );
    }

    #[test]
    fn rejects_bad_shapes_and_dumps_structured_instructions() {
        assert!(q(json!("not a dict")).is_err());
        assert!(q(json!({"instructions": "x"})).is_err());
        assert!(q(json!({"type": "nope", "instructions": "x"})).is_err());
        assert!(q(json!({"type": "choice"})).is_err());
        let dumped = q(json!({"type": "noul", "instructions": {"complex": true}})).unwrap();
        assert_eq!(dumped.instructions, r#"{"complex": true}"#);
    }

    #[test]
    fn non_string_criteria_render_as_json() {
        let question = q(json!({
            "type": "choice",
            "instructions": "Pick",
            "criteria": {"a": {"desc": "x"}, "b": 3, "c": false}
        }))
        .unwrap();
        assert_eq!(
            question.options(),
            vec![
                "a: {\"desc\": \"x\"}".to_owned(),
                "b: 3".to_owned(),
                "c: false".to_owned()
            ]
        );
    }
}
