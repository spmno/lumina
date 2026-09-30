//! The [`Agent`]: a loaded laya checkpoint that answers questions over a state.

use std::path::Path;

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, ops};
use serde_json::{Map, Value, json};

use crate::config::{AgentConfig, EncoderConfig};
use crate::decode::{collapsed_options, decode_state};
use crate::hub;
use crate::model::DecisionModel;
use crate::question::Question;
use crate::sequence::{self, EncodedItem};
use crate::tokenizer::TokenizerBundle;
use crate::{LuminaError, Result};

/// A loaded laya agent: decision model, tokenizer, and clamped runtime configuration.
pub struct Agent {
    model: DecisionModel,
    tok: TokenizerBundle,
    cfg: AgentConfig,
    device: Device,
}

/// Per-call token-budget overrides for [`Agent::predict`]; `None` fields keep the
/// checkpoint's configured budget.
#[derive(Clone, Copy, Debug, Default)]
pub struct PredictOverrides {
    /// State token budget for this call.
    pub max_len: Option<usize>,
    /// Question-head token budget for this call.
    pub head_max_len: Option<usize>,
}

impl Agent {
    /// Load a laya checkpoint with every optional parameter defaulted — Python's
    /// `Agent("convaiinnovations/laya")`.
    ///
    /// # Errors
    /// See [`Agent::load_with`].
    pub fn load(source: &str) -> Result<Self> {
        Self::load_with(source, None, None, None, None)
    }

    /// Load a laya checkpoint from a hub id (`"convaiinnovations/laya"`) or a local
    /// directory a training run wrote, with every load knob explicit.
    ///
    /// * `device` — `"cpu"`, `"cuda"`, or `None` for CUDA-when-available.
    /// * `token` — Hugging Face access token for gated checkpoints; `None` falls
    ///   back to `$HF_TOKEN` and the CLI login file.
    /// * `subfolder` — a bundled checkpoint within the repo (e.g. `"multilingual"`).
    /// * `revision` — pins the hub snapshot to a commit.
    ///
    /// # Errors
    /// Hub or filesystem failures, a directory without laya's config files, and
    /// tensor-load or shape mismatches.
    pub fn load_with(
        source: &str,
        device: Option<&str>,
        token: Option<&str>,
        subfolder: Option<&str>,
        revision: Option<&str>,
    ) -> Result<Self> {
        let device = parse_device(device)?;
        let dir = hub::resolve(source, token, subfolder, revision)?;
        let cfg_path = dir.join("rl_agent_config.json");
        if !cfg_path.is_file() {
            return Err(LuminaError::Incompatible(format!(
                "{source:?} does not contain 'rl_agent_config.json'. That file ships with a \
                 checkpoint's weights, so point at a compatible checkpoint: a hub repo or a \
                 directory a training run wrote."
            )));
        }
        let cfg = AgentConfig::from_value(read_json(&cfg_path)?)?;
        let enc_path = dir.join("encoder").join("config.json");
        let enc_cfg = EncoderConfig::from_value(&read_json(&enc_path)?)?;
        let tok = TokenizerBundle::load(&dir.join("tokenizer"))?;
        let weights = std::fs::read(dir.join("model.safetensors")).map_err(|e| {
            LuminaError::Invalid(format!("{}/model.safetensors: {e}", dir.display()))
        })?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let model = DecisionModel::from_vb(vb, &cfg, &enc_cfg)?;
        Ok(Self {
            model,
            tok,
            cfg,
            device,
        })
    }

    /// Where this agent computes (`"cpu"` or `"cuda"`), for status reporting.
    ///
    /// This is the device the tensors were actually placed on, not whatever
    /// preference was passed to [`Agent::load_with`]. Like torch's `str(device)`,
    /// CUDA is reported without an index — lumina never pins one.
    pub fn device(&self) -> String {
        match &self.device {
            Device::Cpu => "cpu".to_owned(),
            Device::Cuda(_) => "cuda".to_owned(),
            Device::Metal(_) => "metal".to_owned(),
        }
    }

    /// Answer typed questions over one state in a single forward pass — laya's
    /// `system_one`. A list state is a conversation: it truncates from the left so
    /// only the newest turns fit the window.
    ///
    /// # Errors
    /// [`LuminaError::Question`] when a question's options exceed the head budget;
    /// tokenizer and model failures otherwise.
    pub fn predict(&self, state: &Value, questions: &[(String, Question)]) -> Result<Value> {
        self.predict_overridden(state, questions, &PredictOverrides::default())
    }

    /// [`Agent::predict`] with per-call `max_len` / `head_max_len` overrides, the form
    /// [`crate::router::Router`] uses so one request can widen its window without
    /// rebuilding the checkpoint.
    ///
    /// # Errors
    /// [`LuminaError::Question`] when a question's options exceed the head budget;
    /// tokenizer and model failures otherwise.
    pub fn predict_overridden(
        &self,
        state: &Value,
        questions: &[(String, Question)],
        overrides: &PredictOverrides,
    ) -> Result<Value> {
        if questions.is_empty() {
            return Ok(json!({
                "model": "laya-rl-agent",
                "answers": {},
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }));
        }
        let truncate_left = state.is_array();
        let state_text = sequence::serialize_state(state)?.replace(&self.tok.mask_str, " ");
        let state_ids = self.tok.encode(&state_text)?;
        let max_len = overrides.max_len.unwrap_or(self.cfg.max_len);
        let head_max_len = overrides.head_max_len.unwrap_or(self.cfg.head_max_len);
        let mut items: Vec<EncodedItem> = Vec::with_capacity(questions.len());
        for (qid, q) in questions {
            let item = sequence::build_sequence(
                &self.tok,
                &state_ids,
                q,
                max_len,
                head_max_len,
                truncate_left,
            )?;
            if item.markers.len() != q.options().len() {
                return Err(LuminaError::Question {
                    qid: qid.clone(),
                    msg: format!("options exceed head_max_len={head_max_len}"),
                });
            }
            items.push(item);
        }
        let batch = sequence::collate(&items, self.tok.specials.pad, &self.device)?;
        let (logits, act_logits) = self.model.forward(
            &batch.input_ids,
            &batch.attention_mask,
            &batch.type_ids,
            &batch.marker_pos,
            &batch.marker_mask,
        )?;
        // laya's `_forward` softmaxes the act head before `_decode_answers` runs.
        let act = ops::softmax_last_dim(&act_logits)?;
        let answers = decode_state(
            &logits.to_vec2::<f32>()?,
            &act.to_vec2::<f32>()?,
            &items,
            questions,
            &self.cfg,
        )?;
        let mut usage = Map::new();
        usage.insert(
            "input_tokens".to_owned(),
            json!(items.iter().map(|it| it.ids.len()).sum::<usize>()),
        );
        usage.insert("output_tokens".to_owned(), json!(0));
        let collapsed = collapsed_options(questions, &items);
        if !collapsed.is_empty() {
            usage.insert("options".to_owned(), Value::Object(collapsed));
        }
        Ok(json!({
            "model": "laya-rl-agent",
            "answers": Value::Object(answers),
            "usage": usage,
        }))
    }
}

fn parse_device(spec: Option<&str>) -> Result<Device> {
    match spec {
        None | Some("auto") => Device::cuda_if_available(0).map_err(LuminaError::from),
        Some("cpu") => Ok(Device::Cpu),
        Some("cuda" | "gpu" | "cuda:0") => Device::new_cuda(0).map_err(LuminaError::from),
        Some(other) => Err(LuminaError::Invalid(format!(
            "unsupported device {other:?}; use \"cpu\", \"cuda\", or None for auto"
        ))),
    }
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| LuminaError::Invalid(format!("{}: {e}", path.display())))?;
    Ok(serde_json::from_str(&text)?)
}
