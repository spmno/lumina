//! Sequence assembly and batch collation — laya's `build_sequence` and `collate_items`.

use candle_core::{Device, Tensor};
use serde_json::Value;

use crate::pyjson::to_python_json;
use crate::question::{Question, QuestionType};
use crate::tokenizer::TokenizerBundle;
use crate::{LuminaError, Result};

/// laya's per-question head-budget stats.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SequenceStats {
    /// Options the question defines.
    pub options: usize,
    /// Options that still have a token span of their own after the cap.
    pub options_distinct: usize,
    /// The cap applied to each option, when one was.
    pub tokens_per_option: Option<usize>,
}

/// One encoded question row.
#[derive(Debug, Clone)]
pub struct EncodedItem {
    /// Question type (drives the type embedding and the temperature index).
    pub qtype: QuestionType,
    /// Token ids: `[CLS] head [SEP] ⌜mask⌟opt… [SEP] state [SEP]`, truncated to `max_len`.
    pub ids: Vec<u32>,
    /// Position of each option's mask token within `ids`.
    pub markers: Vec<usize>,
    /// Head-budget stats.
    pub stats: SequenceStats,
}

/// laya's `serialize_state`: strings pass through, structures become JSON.
///
/// # Errors
/// Returns [`LuminaError::Json`] if the value cannot be serialized (impossible for
/// values that came from JSON).
pub fn serialize_state(state: &Value) -> Result<String> {
    match state {
        Value::String(s) => Ok(s.clone()),
        other => to_python_json(other),
    }
}

/// laya's `build_sequence`; see [`EncodedItem`] for the layout. `state_ids` is the state
/// pre-tokenized with the mask token stripped, so the shared state is tokenized once.
///
/// # Errors
/// Returns [`LuminaError::Tokenizer`] on tokenization failure.
#[allow(clippy::too_many_arguments)]
pub fn build_sequence(
    tok: &TokenizerBundle,
    state_ids: &[u32],
    q: &Question,
    max_len: usize,
    head_max_len: usize,
    truncate_left: bool,
) -> Result<EncodedItem> {
    let opts = q.options();
    let ins = q.instructions.replace(&tok.mask_str, " ");
    let mut head_ids = tok.encode(&format!("{} question: {}", q.qtype.name(), ins))?;
    let mut opt_ids: Vec<Vec<u32>> = Vec::with_capacity(opts.len());
    for opt in &opts {
        let text = format!(" {}", opt.replace(&tok.mask_str, " "));
        let body = tok.encode_truncated(&text, 48)?;
        let mut ids = Vec::with_capacity(body.len() + 1);
        ids.push(tok.specials.mask);
        ids.extend_from_slice(&body);
        opt_ids.push(ids);
    }
    let total = |lists: &[Vec<u32>]| -> usize { lists.iter().map(std::vec::Vec::len).sum() };
    let mut opt_budget = head_max_len as isize - total(&opt_ids) as isize;
    let mut tokens_per_option = None;
    if opt_budget < 16 {
        // Share the head budget out so every option keeps at least 4 tokens of span.
        let per = (((head_max_len as isize - 16) / opt_ids.len().max(1) as isize).max(4)) as usize;
        tokens_per_option = Some(per);
        for ids in &mut opt_ids {
            ids.truncate(per);
        }
        opt_budget = head_max_len as isize - total(&opt_ids) as isize;
    }
    head_ids.truncate(opt_budget.max(8) as usize);
    let mut ids = Vec::with_capacity(max_len);
    ids.push(tok.specials.cls);
    ids.extend_from_slice(&head_ids);
    ids.push(tok.specials.sep);
    let mut markers = Vec::with_capacity(opt_ids.len());
    for opt in &opt_ids {
        markers.push(ids.len());
        ids.extend_from_slice(opt);
    }
    ids.push(tok.specials.sep);
    let room = (max_len as isize - ids.len() as isize - 1).max(0) as usize;
    if truncate_left {
        ids.extend(state_ids.iter().skip(state_ids.len().saturating_sub(room)));
    } else {
        ids.extend(state_ids.iter().take(room));
    }
    ids.push(tok.specials.sep);
    ids.truncate(max_len);
    markers.retain(|&m| m < max_len);
    let options_distinct = opt_ids
        .iter()
        .fold(Vec::<&[u32]>::new(), |mut seen, opt| {
            if !seen.contains(&opt.as_slice()) {
                seen.push(opt.as_slice());
            }
            seen
        })
        .len();
    Ok(EncodedItem {
        qtype: q.qtype,
        ids,
        markers,
        stats: SequenceStats {
            options: opt_ids.len(),
            options_distinct,
            tokens_per_option,
        },
    })
}

/// A collated batch, ready for [`crate::model::DecisionModel::forward`].
pub struct Batch {
    /// `(n, L)` u32 input ids, padded with the pad token.
    pub input_ids: Tensor,
    /// `(n, L)` u8, 1 for real tokens.
    pub attention_mask: Tensor,
    /// `(n, 1)` u32 question-type ids.
    pub type_ids: Tensor,
    /// `(n, kmax)` u32 marker positions (0 for padding).
    pub marker_pos: Tensor,
    /// `(n, kmax)` u8, 1 where a slot holds a real option.
    pub marker_mask: Tensor,
}

/// laya's `collate_items`: pad to the longest row and pack flat buffers into tensors.
///
/// # Errors
/// Returns [`LuminaError::Candle`] on tensor construction failure and
/// [`LuminaError::Invalid`] for an empty batch.
pub fn collate(items: &[EncodedItem], pad: u32, device: &Device) -> Result<Batch> {
    if items.is_empty() {
        return Err(LuminaError::Invalid("collate: empty batch".into()));
    }
    let n = items.len();
    let l = items.iter().map(|it| it.ids.len()).max().unwrap_or(0);
    let kmax = items.iter().map(|it| it.markers.len()).max().unwrap_or(0);
    if l == 0 || kmax == 0 {
        return Err(LuminaError::Invalid(
            "collate: every item needs ids and at least one marker".into(),
        ));
    }
    let mut ids = Vec::with_capacity(n * l);
    let mut att = Vec::with_capacity(n * l);
    let mut mpos = Vec::with_capacity(n * kmax);
    let mut mmask = Vec::with_capacity(n * kmax);
    let mut qtypes = Vec::with_capacity(n);
    for it in items {
        ids.extend_from_slice(&it.ids);
        ids.extend(std::iter::repeat_n(pad, l - it.ids.len()));
        att.extend(std::iter::repeat_n(1u8, it.ids.len()));
        att.extend(std::iter::repeat_n(0u8, l - it.ids.len()));
        mpos.extend(it.markers.iter().map(|&m| m as u32));
        mpos.extend(std::iter::repeat_n(0u32, kmax - it.markers.len()));
        mmask.extend(std::iter::repeat_n(1u8, it.markers.len()));
        mmask.extend(std::iter::repeat_n(0u8, kmax - it.markers.len()));
        qtypes.push(it.qtype.index() as u32);
    }
    Ok(Batch {
        input_ids: Tensor::from_vec(ids, (n, l), device)?,
        attention_mask: Tensor::from_vec(att, (n, l), device)?,
        type_ids: Tensor::from_vec(qtypes, (n, 1), device)?,
        marker_pos: Tensor::from_vec(mpos, (n, kmax), device)?,
        marker_mask: Tensor::from_vec(mmask, (n, kmax), device)?,
    })
}
