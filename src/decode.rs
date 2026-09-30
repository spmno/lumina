//! Decoding: temperature-scaled softmax, confidence, and typed answer objects.

use serde_json::{Map, Value, json};

use crate::config::AgentConfig;
use crate::question::{Criteria, Question, QuestionType};
use crate::sequence::EncodedItem;
use crate::{LuminaError, Result};

/// Python's `round(x, 4)`: round-half-even applied to the binary value.
#[must_use]
pub fn round4(x: f64) -> f64 {
    // A "{:.4}" string always parses back, so the fallback is unreachable.
    format!("{x:.4}").parse().unwrap_or(x)
}

/// laya's `temp_bucket`: the temperature-override key for a question type and option count.
fn temp_bucket(qt: QuestionType, k: usize) -> String {
    let size = match k {
        0..=2 => "2",
        3..=5 => "3-5",
        6..=10 => "6-10",
        _ => "11+",
    };
    format!("{}:{size}", qt.name())
}

/// Stable f32 softmax, matching `np.exp(z - z.max()) / sum` on the first `k` logits.
fn softmax(z: &[f32]) -> Vec<f32> {
    let max = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = z.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

/// laya's `answer_confidence`: max(p), the calibrated quantity temperature scaling fits.
fn answer_confidence(p: &[f32]) -> f64 {
    f64::from(
        p.iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            .clamp(0.0, 1.0),
    )
}

/// laya's `confidence_from_probs`: 1 − H(p) / log(k), clamped to [0, 1].
fn confidence_from_probs(p: &[f32]) -> f64 {
    if p.len() < 2 {
        return 1.0;
    }
    let ent = -p.iter().map(|&v| v * v.max(1e-12).ln()).sum::<f32>();
    f64::from((1.0f32 - ent / (p.len() as f32).ln()).clamp(0.0, 1.0))
}

/// A choice label as a JSON object key, the way Python stringifies dict keys when
/// dumping: `"billing"`, `1` → `"1"`, `true` → `"true"`.
fn py_key(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Bool(b) => {
            if *b {
                "true".to_owned()
            } else {
                "false".to_owned()
            }
        }
        other => other.to_string(),
    }
}

/// laya's `_decode_answers` for one state's rows. `logit_rows` and `act_rows` are CPU f32
/// copies; `act_rows` must already be softmaxed, as laya's `_forward` does.
///
/// # Errors
/// Returns [`LuminaError::Invalid`] if a row is missing (a caller bug).
pub fn decode_state(
    logit_rows: &[Vec<f32>],
    act_rows: &[Vec<f32>],
    items: &[EncodedItem],
    questions: &[(String, Question)],
    cfg: &AgentConfig,
) -> Result<Map<String, Value>> {
    let mut answers = Map::new();
    for (j, (qid, q)) in questions.iter().enumerate() {
        let k = items
            .get(j)
            .map(|it| it.markers.len())
            .ok_or_else(|| LuminaError::Invalid(format!("missing encoded row {j}")))?;
        let row = logit_rows
            .get(j)
            .ok_or_else(|| LuminaError::Invalid(format!("missing logits row {j}")))?;
        let act_row = act_rows
            .get(j)
            .ok_or_else(|| LuminaError::Invalid(format!("missing act row {j}")))?;
        let base_t = cfg.temperature.get(q.qtype.index()).copied().unwrap_or(1.0);
        let t_scale = cfg
            .temperature_by_options
            .get(&temp_bucket(q.qtype, k))
            .copied()
            .unwrap_or(base_t);
        let z: Vec<f32> = row.iter().take(k).map(|&v| v / t_scale).collect();
        let p = softmax(&z);
        let ans_conf = round4(answer_confidence(&p));
        let action = json!({
            "act_probability": round4(f64::from(act_row.first().copied().unwrap_or(0.0)))
        });
        let answer = match &q.criteria {
            Criteria::Choice(entries) => {
                // np.argmax keeps the first maximum; so does a strict `>`.
                let mut best = 0;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &v) in p.iter().enumerate() {
                    if v > best_v {
                        best_v = v;
                        best = i;
                    }
                }
                let mut probabilities = Map::new();
                for ((label, _), &pv) in entries.iter().zip(p.iter()) {
                    probabilities.insert(py_key(label), json!(round4(f64::from(pv))));
                }
                json!({
                    "type": "choice",
                    "choice": entries.get(best).map_or(Value::Null, |(l, _)| l.clone()),
                    "probabilities": probabilities,
                    "confidence": round4(confidence_from_probs(&p)),
                    "answer_confidence": ans_conf,
                    "action": action,
                })
            }
            Criteria::Score(levels) => {
                let exp_score: f32 = p.iter().enumerate().map(|(i, &v)| i as f32 * v).sum();
                let mut legend = Map::new();
                for (i, c) in levels.iter().enumerate() {
                    legend.insert(i.to_string(), c.clone());
                }
                let mut probabilities = Map::new();
                for (i, &v) in p.iter().enumerate() {
                    probabilities.insert(i.to_string(), json!(round4(f64::from(v))));
                }
                json!({
                    "type": "score",
                    "score": round4(f64::from(exp_score)),
                    "legend": legend,
                    "probabilities": probabilities,
                    "confidence": round4(confidence_from_probs(&p)),
                    "answer_confidence": ans_conf,
                    "action": action,
                })
            }
            Criteria::Noul { .. } => {
                let p_true = p.get(1).copied().unwrap_or(0.0);
                json!({
                    "type": "noul",
                    "noul": round4(f64::from(p_true)),
                    "confidence": round4(f64::from(p_true.max(1.0 - p_true))),
                    "answer_confidence": ans_conf,
                    "action": action,
                })
            }
        };
        answers.insert(qid.clone(), answer);
    }
    Ok(answers)
}

/// laya's `collapsed_options`: the `usage["options"]` report for questions whose options
/// lost their distinct token spans to the head budget.
#[must_use]
pub fn collapsed_options(
    questions: &[(String, Question)],
    items: &[EncodedItem],
) -> Map<String, Value> {
    let mut out = Map::new();
    for ((qid, _), item) in questions.iter().zip(items.iter()) {
        if item.stats.options_distinct < item.stats.options {
            out.insert(
                qid.clone(),
                json!({
                    "total": item.stats.options,
                    "distinct": item.stats.options_distinct,
                    "tokens_per_option": item.stats.tokens_per_option,
                }),
            );
        }
    }
    out
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
    use crate::question::QuestionType;
    use serde_json::json;

    #[test]
    fn round4_matches_python_round() {
        // Python: round(2.675, 4) == 2.675 (binary value is just under).
        assert_eq!(round4(2.675), 2.675);
        assert_eq!(round4(0.123_456), 0.1235);
        assert_eq!(round4(0.94), 0.94);
        assert_eq!(round4(1.0), 1.0);
        assert_eq!(round4(-0.00001), -0.0);
        assert_eq!(round4(0.99995), 1.0);
    }

    #[test]
    fn buckets_follow_laya_names() {
        assert_eq!(temp_bucket(QuestionType::Choice, 2), "choice:2");
        assert_eq!(temp_bucket(QuestionType::Score, 5), "score:3-5");
        assert_eq!(temp_bucket(QuestionType::Noul, 2), "noul:2");
        assert_eq!(temp_bucket(QuestionType::Choice, 11), "choice:11+");
    }

    #[test]
    fn softmax_and_confidences() {
        let p = softmax(&[1.0_f32, 2.0, 2.0]);
        assert!((p[1] - p[2]).abs() < 1e-6);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // answer_confidence = max(p)
        let top = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((answer_confidence(&p) - f64::from(top)).abs() < 1e-6);
        assert_eq!(confidence_from_probs(&[0.5, 0.5]), 0.0);
        assert_eq!(confidence_from_probs(&[1.0]), 1.0);
    }

    fn agent_cfg() -> AgentConfig {
        AgentConfig::from_value(json!({
            "encoder": "answerdotai/ModernBERT-large",
            "temperature": [1.0, 1.0, 1.0],
            "temperature_by_options": {"choice:2": 2.0}
        }))
        .unwrap()
    }

    fn question(qid: &str, qdef: Value) -> (String, Question) {
        (qid.to_owned(), Question::from_def(qid, &qdef).unwrap())
    }

    fn item(markers: usize) -> EncodedItem {
        EncodedItem {
            qtype: QuestionType::Choice,
            ids: vec![1; 4],
            markers: vec![2; markers],
            stats: crate::sequence::SequenceStats {
                options: markers,
                options_distinct: markers,
                tokens_per_option: None,
            },
        }
    }

    #[test]
    fn decodes_choice_with_bucket_temperature() {
        let questions = vec![question(
            "q",
            json!({"type": "choice", "instructions": "pick",
                   "criteria": {"a": "A", "b": "B"}}),
        )];
        let items = vec![item(2)];
        // bucket choice:2 → t=2.0; logits [0, 2] → z [0, 1] → p ≈ [0.2689, 0.7311]
        let logits = vec![vec![0.0_f32, 2.0, 0.0]];
        let act = vec![vec![0.9_f32, 0.1]];
        let answers = decode_state(&logits, &act, &items, &questions, &agent_cfg()).unwrap();
        let a = answers.get("q").unwrap();
        assert_eq!(a["choice"], json!("b"));
        assert!((a["probabilities"]["b"].as_f64().unwrap() - 0.7311).abs() < 0.0002);
        assert!((a["action"]["act_probability"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn decodes_score_and_noul() {
        let questions = vec![
            question(
                "s",
                json!({"type": "score", "instructions": "rate", "criteria": ["low", "mid", "high"]}),
            ),
            question("n", json!({"type": "noul", "instructions": "holds?"})),
        ];
        let items = vec![item(3), item(2)];
        let logits = vec![vec![1.0_f32, 0.0, 0.0, 0.0], vec![0.5, 1.5, 0.0]];
        let act = vec![vec![0.2_f32, 0.8], vec![0.5, 0.5]];
        let answers = decode_state(&logits, &act, &items, &questions, &agent_cfg()).unwrap();
        let s = answers.get("s").unwrap();
        assert_eq!(s["legend"]["1"], json!("mid"));
        assert!(s.get("probabilities").unwrap().get("0").is_some());
        let n = answers.get("n").unwrap();
        assert!((n["noul"].as_f64().unwrap() - 0.7311).abs() < 0.0002);
        assert!(n.get("probabilities").is_none());
    }
}
