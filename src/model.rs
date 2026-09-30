//! laya's decision head: stacked pre-LN transformer layers, the option scorer, and the
//! act head that maps pooled features to escalate probabilities.

use candle_core::{D, DType, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, ops};

use crate::Result;
use crate::config::{AgentConfig, EncoderConfig};
use crate::encoder::ModernBertEncoder;

/// PyTorch `nn.TransformerEncoderLayer` (pre-LN, ReLU feed-forward, batch-first) with a
/// fused `nn.MultiheadAttention` projection. The attention scale is `1/sqrt(head_dim)`,
/// PyTorch's MHA convention (not `1/sqrt(hidden)`).
struct HeadLayer {
    norm1: LayerNorm,
    in_proj_weight: Tensor,
    in_proj_bias: Option<Tensor>,
    out_proj: Linear,
    norm2: LayerNorm,
    linear1: Linear,
    linear2: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl HeadLayer {
    fn from_vb(vb: VarBuilder<'_>, hidden: usize, heads: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::new(
                vb.get((hidden,), "norm1.weight")?,
                vb.get((hidden,), "norm1.bias")?,
                eps,
            ),
            in_proj_weight: vb.get((3 * hidden, hidden), "self_attn.in_proj_weight")?,
            in_proj_bias: Some(vb.get((3 * hidden,), "self_attn.in_proj_bias")?),
            out_proj: Linear::new(
                vb.get((hidden, hidden), "self_attn.out_proj.weight")?,
                Some(vb.get((hidden,), "self_attn.out_proj.bias")?),
            ),
            norm2: LayerNorm::new(
                vb.get((hidden,), "norm2.weight")?,
                vb.get((hidden,), "norm2.bias")?,
                eps,
            ),
            linear1: Linear::new(
                vb.get((4 * hidden, hidden), "linear1.weight")?,
                Some(vb.get((4 * hidden,), "linear1.bias")?),
            ),
            linear2: Linear::new(
                vb.get((hidden, 4 * hidden), "linear2.weight")?,
                Some(vb.get((hidden,), "linear2.bias")?),
            ),
            heads,
            head_dim: hidden / heads,
            scale: 1.0 / ((hidden / heads) as f64).sqrt(),
        })
    }

    /// `x + MHA(norm1(x))` then `x + FFN(norm2(x))` with the key-padding mask added to
    /// attention scores as `-inf` (torch's `_canonical_mask`).
    #[allow(clippy::many_single_char_names)]
    fn forward(&self, x: &Tensor, pad_add: &Tensor) -> Result<Tensor> {
        let n = self.norm1.forward(x)?;
        let (b, seq, d) = n.dims3()?;
        let qkv = n.reshape((b * seq, d))?.matmul(&self.in_proj_weight.t()?)?;
        let qkv = match &self.in_proj_bias {
            Some(bias) => qkv.broadcast_add(bias)?,
            None => qkv,
        };
        let qkv = qkv.reshape((b, seq, 3 * d))?;
        let split = |i: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(2, i * d, d)?
                .reshape((b, seq, self.heads, self.head_dim))?
                .transpose(1, 2)?)
        };
        let q = split(0)?.affine(self.scale, 0.0)?;
        let k = split(1)?.contiguous()?;
        let v = split(2)?.contiguous()?;
        let scores = q.matmul(&k.t()?)?.broadcast_add(pad_add)?;
        let probs = ops::softmax(&scores, D::Minus1)?;
        let attn = probs.matmul(&v)?;
        let attn = attn.transpose(1, 2)?.flatten_from(2)?;
        let x = (x + self.out_proj.forward(&attn)?)?;
        let ff = self
            .linear2
            .forward(&self.linear1.forward(&self.norm2.forward(&x)?)?.relu()?)?;
        Ok((x + ff)?)
    }
}

/// laya's option scorer: `LayerNorm → Linear(d, d) → GELU → Linear(d, 1)`.
struct Scorer {
    norm: LayerNorm,
    l1: Linear,
    l2: Linear,
}

impl Scorer {
    fn from_vb(vb: VarBuilder<'_>, hidden: usize, eps: f64) -> Result<Self> {
        Ok(Self {
            norm: LayerNorm::new(
                vb.get((hidden,), "0.weight")?,
                vb.get((hidden,), "0.bias")?,
                eps,
            ),
            l1: Linear::new(
                vb.get((hidden, hidden), "1.weight")?,
                Some(vb.get((hidden,), "1.bias")?),
            ),
            l2: Linear::new(
                vb.get((1, hidden), "3.weight")?,
                Some(vb.get((1,), "3.bias")?),
            ),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.norm.forward(x)?;
        let h = self.l1.forward(&h)?.gelu_erf()?;
        Ok(self.l2.forward(&h)?)
    }
}

/// The full laya model: encoder, head, type embedding, scorer, act head. All tensors are
/// F32 — laya runs float32 on CPU and the port keeps that everywhere.
pub struct DecisionModel {
    encoder: ModernBertEncoder,
    head: Vec<HeadLayer>,
    type_emb: Embedding,
    scorer: Scorer,
    act_l1: Linear,
    act_l2: Linear,
    hidden: usize,
}

impl DecisionModel {
    /// Build from the root of a laya safetensors checkpoint. Requested shapes encode the
    /// architecture, so shape mismatches (wrong head layer count, act head width, …)
    /// surface as load errors.
    ///
    /// # Errors
    /// Returns [`crate::LuminaError::Candle`] when a tensor is missing or mis-shaped.
    pub fn from_vb(vb: VarBuilder<'_>, agent: &AgentConfig, enc: &EncoderConfig) -> Result<Self> {
        let hidden = enc.hidden;
        let mut head = Vec::with_capacity(agent.head_layers);
        for i in 0..agent.head_layers {
            head.push(HeadLayer::from_vb(
                vb.pp(format!("head.layers.{i}")),
                hidden,
                enc.heads,
                enc.eps,
            )?);
        }
        Ok(Self {
            encoder: ModernBertEncoder::from_vb(vb.pp("encoder"), enc)?,
            head,
            type_emb: Embedding::new(vb.get((3, hidden), "type_emb.weight")?, hidden),
            scorer: Scorer::from_vb(vb.pp("scorer"), hidden, enc.eps)?,
            act_l1: Linear::new(
                vb.get((256, hidden + 4), "act_head.0.weight")?,
                Some(vb.get((256,), "act_head.0.bias")?),
            ),
            act_l2: Linear::new(
                vb.get((agent.n_act(), 256), "act_head.2.weight")?,
                Some(vb.get((agent.n_act(),), "act_head.2.bias")?),
            ),
            hidden,
        })
    }

    /// One forward pass over a padded batch.
    ///
    /// * `input_ids` — `(b, seq)` u32.
    /// * `attention_mask` — `(b, seq)` u8, 1 for real tokens.
    /// * `type_ids` — `(b, 1)` u32, the question-type id of each row.
    /// * `marker_pos` — `(b, kmax)` u32, position of every option slot (0 for padding).
    /// * `marker_mask` — `(b, kmax)` u8, 1 where a slot holds a real option.
    ///
    /// Returns `(logits (b, kmax) f32, act_logits (b, n_act) f32)`; padded slots carry
    /// `-1e4` logits, exactly laya's `masked_fill`.
    ///
    /// # Errors
    /// Returns [`crate::LuminaError::Candle`] on shape or compute failures.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        type_ids: &Tensor,
        marker_pos: &Tensor,
        marker_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (b, seq) = input_ids.dims2()?;
        let device = input_ids.device();
        let mut h = self.encoder.forward(input_ids, attention_mask)?;
        // (b, seq, d) + (b, 1, d): the type embedding broadcasts over the sequence.
        h = h.broadcast_add(&self.type_emb.forward(type_ids)?)?;
        // Head key-padding uses -inf (torch's `_canonical_mask`), unlike the encoder's
        // f32::MIN additive mask.
        let head_pad = {
            let rows = attention_mask.to_vec2::<u8>()?;
            let mut data = Vec::with_capacity(b * seq);
            for row in &rows {
                for &m in row {
                    data.push(if m == 0 { f32::NEG_INFINITY } else { 0.0 });
                }
            }
            Tensor::from_vec(data, (b, 1, 1, seq), device)?
        };
        for layer in &self.head {
            h = layer.forward(&h, &head_pad)?;
        }
        let (_, kmax) = marker_pos.dims2()?;
        let idx = marker_pos
            .unsqueeze(2)?
            .expand((b, kmax, self.hidden))?
            .contiguous()?;
        let markers = h.gather(&idx, 1)?;
        let logits = self.scorer.forward(&markers)?.squeeze(D::Minus1)?;
        let mask_f = marker_mask.to_dtype(DType::F32)?;
        // masked_fill(mask, -1e4) as pure arithmetic: keep where m == 1, -1e4 where m == 0.
        let logits = ((logits * &mask_f)? + mask_f.affine(1e4, -1e4)?)?;
        let act_logits = self.act_features(&h, &logits, marker_mask)?;
        Ok((logits, act_logits))
    }

    /// act-head path: pooled `[CLS]` concatenated with `[top1, top1 − top2, entropy,
    /// k / 255]`. As in laya, the features come from the untempered softmax over every
    /// slot of the row (masked slots already carry −1e4 and vanish), the option count
    /// `k` is clamped to two, and a width-one marker dimension gives the second top
    /// value 0.0; candle has no topk, so this runs on CPU in f32 — value-identical to
    /// torch's on-device result up to reduction order.
    #[allow(clippy::many_single_char_names)]
    fn act_features(&self, h: &Tensor, logits: &Tensor, marker_mask: &Tensor) -> Result<Tensor> {
        let (b, kmax) = logits.dims2()?;
        let logit_rows = logits.to_vec2::<f32>()?;
        let mask_rows = marker_mask.to_vec2::<u8>()?;
        let mut feats = Vec::with_capacity(b * 4);
        for (row, mask) in logit_rows.iter().zip(mask_rows.iter()) {
            let k = mask.iter().filter(|&&m| m != 0).count().max(2);
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let p: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
            let mut top1 = f32::NEG_INFINITY;
            let mut top2 = f32::NEG_INFINITY;
            for &pv in &p {
                if pv >= top1 {
                    top2 = top1;
                    top1 = pv;
                } else if pv > top2 {
                    top2 = pv;
                }
            }
            if kmax < 2 {
                top2 = 0.0;
            }
            let ent = -p.iter().map(|&pv| pv * pv.max(1e-9).ln()).sum::<f32>() / (k as f32).ln();
            feats.extend_from_slice(&[top1, top1 - top2, ent, k as f32 / 255.0]);
        }
        let feats_t = Tensor::from_vec(feats, (b, 4), h.device())?;
        let pooled = h.narrow(1, 0, 1)?.squeeze(1)?;
        let act_in = Tensor::cat(&[&pooled, &feats_t], D::Minus1)?;
        let act_h = self.act_l1.forward(&act_in)?.gelu_erf()?;
        Ok(self.act_l2.forward(&act_h)?)
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
    use candle_core::Device;

    #[test]
    fn masked_fill_arithmetic_matches_where() {
        // logits * m + (m * 1e4 - 1e4) must keep values where m == 1 and set -1e4 where m == 0
        let logits = Tensor::from_vec(vec![0.25f32, -3.5, 12.0], (1, 3), &Device::Cpu).unwrap();
        let mask = Tensor::from_vec(vec![1u8, 0, 1], (1, 3), &Device::Cpu).unwrap();
        let mask_f = mask.to_dtype(DType::F32).unwrap();
        let out = ((logits * &mask_f).unwrap() + mask_f.affine(1e4, -1e4).unwrap()).unwrap();
        let v = out.to_vec2::<f32>().unwrap();
        assert_eq!(v[0], vec![0.25, -1e4, 12.0]);
    }

    #[test]
    fn head_and_scorer_request_checkpoint_shapes() {
        use std::collections::HashMap;
        let dev = Device::Cpu;
        let hidden = 8;
        let f = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &dev).unwrap();
        let mut head: HashMap<String, Tensor> = HashMap::new();
        for key in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"] {
            head.insert(key.to_string(), f(&[hidden]));
        }
        head.insert(
            "self_attn.in_proj_weight".to_string(),
            f(&[3 * hidden, hidden]),
        );
        head.insert("self_attn.in_proj_bias".to_string(), f(&[3 * hidden]));
        head.insert(
            "self_attn.out_proj.weight".to_string(),
            f(&[hidden, hidden]),
        );
        head.insert("self_attn.out_proj.bias".to_string(), f(&[hidden]));
        head.insert("linear1.weight".to_string(), f(&[4 * hidden, hidden]));
        head.insert("linear1.bias".to_string(), f(&[4 * hidden]));
        head.insert("linear2.weight".to_string(), f(&[hidden, 4 * hidden]));
        head.insert("linear2.bias".to_string(), f(&[hidden]));
        let vb = VarBuilder::from_tensors(head, DType::F32, &dev);
        HeadLayer::from_vb(vb, hidden, 2, 1e-5).unwrap();

        let mut scorer: HashMap<String, Tensor> = HashMap::new();
        scorer.insert("0.weight".to_string(), f(&[hidden]));
        scorer.insert("0.bias".to_string(), f(&[hidden]));
        scorer.insert("1.weight".to_string(), f(&[hidden, hidden]));
        scorer.insert("1.bias".to_string(), f(&[hidden]));
        scorer.insert("3.weight".to_string(), f(&[1, hidden]));
        scorer.insert("3.bias".to_string(), f(&[1]));
        let vb = VarBuilder::from_tensors(scorer, DType::F32, &dev);
        Scorer::from_vb(vb, hidden, 1e-5).unwrap();
    }
}
