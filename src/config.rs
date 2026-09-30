//! Checkpoint configuration: `rl_agent_config.json` and `encoder/config.json`.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::{LuminaError, Result};

/// Lower bound of the temperature clamp laya applies to every fitted temperature.
pub const TEMP_MIN: f64 = 0.5;
/// Upper bound of the temperature clamp laya applies to every fitted temperature.
pub const TEMP_MAX: f64 = 5.0;

/// laya's `clamp_temperature`: non-finite values fall back to `1.0`, everything else is
/// confined to `[TEMP_MIN, TEMP_MAX]`.
#[allow(clippy::missing_const_for_fn)]
pub fn clamp_temperature(t: f64) -> f32 {
    if t.is_nan() || t.is_infinite() {
        return 1.0;
    }
    t.clamp(TEMP_MIN, TEMP_MAX) as f32
}

/// Parsed `rl_agent_config.json` with laya's defaults applied and temperatures clamped.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Base encoder id (informational; weights come from the checkpoint).
    pub encoder: String,
    /// Number of head transformer layers.
    pub head_layers: usize,
    /// Encoder context budget.
    pub max_len: usize,
    /// Head budget for question text plus options.
    pub head_max_len: usize,
    /// Action costs; the act head has `act_costs.len() + 1` outputs.
    pub act_costs: HashMap<String, f64>,
    /// Per-question-type temperatures `[choice, score, noul]`, clamped.
    pub temperature: [f32; 3],
    /// Per-bucket temperature overrides (`"choice:11+"`, …), clamped.
    pub temperature_by_options: HashMap<String, f32>,
}

/// Serde view of `rl_agent_config.json`; unknown fields are ignored.
#[derive(Deserialize)]
struct RawAgentConfig {
    encoder: String,
    head_layers: Option<usize>,
    max_len: Option<usize>,
    head_max_len: Option<usize>,
    act_costs: Option<HashMap<String, f64>>,
    temperature: Option<Vec<f64>>,
    temperature_by_options: Option<HashMap<String, f64>>,
}

impl AgentConfig {
    /// Parse and validate a `rl_agent_config.json` document.
    ///
    /// # Errors
    /// Returns [`LuminaError::Incompatible`] when the document does not describe a laya
    /// checkpoint (`temperature` not a list of 3 floats, malformed JSON).
    pub fn from_value(v: Value) -> Result<Self> {
        let raw: RawAgentConfig = serde_json::from_value(v)
            .map_err(|e| LuminaError::Incompatible(format!("rl_agent_config.json: {e}")))?;
        let temperature = match raw.temperature {
            None => [1.0, 1.0, 1.0],
            Some(list) => match list.as_slice() {
                [a, b, c] => [
                    clamp_temperature(*a),
                    clamp_temperature(*b),
                    clamp_temperature(*c),
                ],
                _ => {
                    return Err(LuminaError::Incompatible(format!(
                        "temperature must be a list of 3 floats, got {} entries",
                        list.len()
                    )));
                }
            },
        };
        let temperature_by_options = raw
            .temperature_by_options
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, clamp_temperature(v)))
            .collect();
        Ok(Self {
            encoder: raw.encoder,
            head_layers: raw.head_layers.unwrap_or(2),
            max_len: raw.max_len.unwrap_or(512),
            head_max_len: raw.head_max_len.unwrap_or(192),
            act_costs: raw.act_costs.unwrap_or_default(),
            temperature,
            temperature_by_options,
        })
    }

    /// Number of act outputs the checkpoint's `act_head` produces.
    pub fn n_act(&self) -> usize {
        self.act_costs.len() + 1
    }
}

/// Attention pattern of one encoder layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    /// Full (global) self-attention.
    Global,
    /// Sliding-window attention: each side sees `local_attention / 2` tokens.
    Sliding,
}

/// Compiled ModernBERT encoder configuration (transformers v5 flat layout).
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Hidden size.
    pub hidden: usize,
    /// Number of encoder layers.
    pub num_layers: usize,
    /// Attention heads.
    pub heads: usize,
    /// Per-head dimension (`hidden / heads`).
    pub head_dim: usize,
    /// MLP intermediate size (the GeGLU input projection is `2 * intermediate`).
    pub intermediate: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// LayerNorm epsilon.
    pub eps: f64,
    /// Per-layer attention pattern.
    pub layers: Vec<AttentionKind>,
    /// Sliding-window diameter.
    pub local_attention: usize,
    /// RoPE base for global layers.
    pub global_theta: f64,
    /// RoPE base for sliding layers.
    pub local_theta: f64,
    /// Maximum positions to precompute RoPE tables for.
    pub max_positions: usize,
}

/// Serde view of `encoder/config.json`; unknown fields are ignored.
#[derive(Deserialize)]
struct RawEncoderConfig {
    hidden_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    intermediate_size: usize,
    vocab_size: usize,
    #[serde(default = "default_eps")]
    norm_eps: f64,
    #[serde(default)]
    norm_bias: bool,
    #[serde(default)]
    attention_bias: Option<bool>,
    #[serde(default)]
    mlp_bias: Option<bool>,
    #[serde(default)]
    hidden_activation: Option<String>,
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(default = "default_every")]
    global_attn_every_n_layers: usize,
    #[serde(default = "default_local")]
    local_attention: usize,
    #[serde(default = "default_max_pos")]
    max_position_embeddings: usize,
}

const fn default_eps() -> f64 {
    1e-5
}
const fn default_every() -> usize {
    3
}
const fn default_local() -> usize {
    128
}
const fn default_max_pos() -> usize {
    8192
}

impl EncoderConfig {
    /// Parse and validate an `encoder/config.json` document.
    ///
    /// # Errors
    /// Returns [`LuminaError::Incompatible`] when the config uses settings this port does
    /// not implement (biased norms or projections, non-GELU activation, unknown layer
    /// types, a head size that does not divide the hidden size evenly).
    pub fn from_value(v: &Value) -> Result<Self> {
        let raw: RawEncoderConfig = serde_json::from_value(v.clone())
            .map_err(|e| LuminaError::Incompatible(format!("encoder/config.json: {e}")))?;
        if raw.norm_bias {
            return Err(LuminaError::Incompatible(
                "norm_bias=true is not supported (laya checkpoints ship false)".into(),
            ));
        }
        for (name, bias) in [
            ("attention_bias", raw.attention_bias),
            ("mlp_bias", raw.mlp_bias),
        ] {
            if bias == Some(true) {
                return Err(LuminaError::Incompatible(format!(
                    "{name}=true is not supported (laya checkpoints ship false)"
                )));
            }
        }
        if let Some(act) = &raw.hidden_activation {
            if act != "gelu" {
                return Err(LuminaError::Incompatible(format!(
                    "hidden_activation {act:?} is not supported (only \"gelu\")"
                )));
            }
        }
        if raw.hidden_size == 0 || raw.num_attention_heads == 0 {
            return Err(LuminaError::Incompatible(
                "hidden_size and num_attention_heads must be positive".into(),
            ));
        }
        if raw.hidden_size % raw.num_attention_heads != 0 {
            return Err(LuminaError::Incompatible(format!(
                "hidden_size {} is not divisible by num_attention_heads {}",
                raw.hidden_size, raw.num_attention_heads
            )));
        }
        let layers = match raw.layer_types {
            Some(names) => {
                if names.len() != raw.num_hidden_layers {
                    return Err(LuminaError::Incompatible(format!(
                        "layer_types has {} entries but num_hidden_layers is {}",
                        names.len(),
                        raw.num_hidden_layers
                    )));
                }
                names
                    .iter()
                    .map(|s| match s.as_str() {
                        "full_attention" => Ok(AttentionKind::Global),
                        "sliding_attention" => Ok(AttentionKind::Sliding),
                        other => Err(LuminaError::Incompatible(format!(
                            "unknown layer type {other:?}"
                        ))),
                    })
                    .collect::<Result<Vec<_>>>()?
            }
            None => (0..raw.num_hidden_layers)
                .map(|i| {
                    if i % raw.global_attn_every_n_layers.max(1) == 0 {
                        AttentionKind::Global
                    } else {
                        AttentionKind::Sliding
                    }
                })
                .collect(),
        };
        let theta_of = |key: &str| -> Option<f64> {
            v.get("rope_parameters")
                .and_then(|r| r.get(key))
                .and_then(|e| e.get("rope_theta"))
                .and_then(Value::as_f64)
        };
        let flat_theta = v
            .get("rope_parameters")
            .and_then(|r| r.get("rope_theta"))
            .and_then(Value::as_f64);
        let global_theta = theta_of("full_attention")
            .or(flat_theta)
            .unwrap_or(160_000.0);
        let local_theta = theta_of("sliding_attention")
            .or(flat_theta)
            .unwrap_or(10_000.0);
        Ok(Self {
            hidden: raw.hidden_size,
            num_layers: raw.num_hidden_layers,
            heads: raw.num_attention_heads,
            head_dim: raw.hidden_size / raw.num_attention_heads,
            intermediate: raw.intermediate_size,
            vocab: raw.vocab_size,
            eps: raw.norm_eps,
            layers,
            local_attention: raw.local_attention,
            global_theta,
            local_theta,
            max_positions: raw.max_position_embeddings,
        })
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

    #[test]
    fn clamps_temperatures_like_laya() {
        assert_eq!(clamp_temperature(f64::NAN), 1.0);
        assert_eq!(clamp_temperature(f64::INFINITY), 1.0);
        assert_eq!(clamp_temperature(0.1), 0.5);
        assert_eq!(clamp_temperature(9.0), 5.0);
        assert_eq!(clamp_temperature(1.25), 1.25);
    }

    #[test]
    fn parses_english_encoder_config() {
        // laya's English checkpoint: 28 layers, global every 3rd (0, 3, …, 27).
        let layer_types: Vec<&str> = (0..28)
            .map(|i| {
                if i % 3 == 0 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            })
            .collect();
        let raw = serde_json::json!({
            "hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16,
            "intermediate_size": 2624, "vocab_size": 50368, "norm_eps": 1e-5, "norm_bias": false,
            "layer_types": layer_types,
            "local_attention": 128, "max_position_embeddings": 8192,
            "rope_parameters": {
                "full_attention": {"rope_theta": 160_000.0, "rope_type": "default"},
                "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"}
            }
        });
        let cfg = EncoderConfig::from_value(&raw).unwrap();
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.global_theta, 160_000.0);
        assert_eq!(cfg.local_theta, 10_000.0);
        assert_eq!(cfg.layers.len(), 28);
        assert_eq!(cfg.layers.first(), Some(&AttentionKind::Global));
        assert_eq!(
            cfg.layers
                .iter()
                .filter(|k| **k == AttentionKind::Global)
                .count(),
            10
        );
    }

    #[test]
    fn derives_layer_pattern_when_missing() {
        let raw = serde_json::json!({
            "hidden_size": 768, "num_hidden_layers": 6, "num_attention_heads": 12,
            "intermediate_size": 1152, "vocab_size": 256_000, "global_attn_every_n_layers": 2
        });
        let cfg = EncoderConfig::from_value(&raw).unwrap();
        assert_eq!(
            cfg.layers,
            vec![
                AttentionKind::Global,
                AttentionKind::Sliding,
                AttentionKind::Global,
                AttentionKind::Sliding,
                AttentionKind::Global,
                AttentionKind::Sliding
            ]
        );
    }

    #[test]
    fn rejects_rejected_settings() {
        let mut raw = serde_json::json!({
            "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
            "intermediate_size": 128, "vocab_size": 100
        });
        raw["norm_bias"] = serde_json::json!(true);
        assert!(EncoderConfig::from_value(&raw).is_err());
    }
}
