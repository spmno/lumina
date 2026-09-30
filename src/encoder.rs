//! ModernBERT encoder ported to candle: bidirectional, GeGLU MLPs, alternating global /
//! sliding attention with per-pattern RoPE bases, and no biases anywhere.

use candle_core::{D, Device, Module, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, ops};

use crate::Result;
use crate::config::{AttentionKind, EncoderConfig};

/// cos/sin tables `(1, 1, seq, head_dim / 2)` for one RoPE base, computed in f32 to
/// match transformers' `RoPEInit` numerics.
fn rope_tables(
    theta: f64,
    seq: usize,
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let theta_f = theta as f32;
    let mut inv_freq = Vec::with_capacity(half);
    for j in 0..half {
        inv_freq.push(1.0 / theta_f.powf(2.0 * j as f32 / head_dim as f32));
    }
    let mut cos = Vec::with_capacity(seq * half);
    let mut sin = Vec::with_capacity(seq * half);
    for pos in 0..seq {
        for &freq in &inv_freq {
            let angle = pos as f32 * freq;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (1, 1, seq, half), device)?,
        Tensor::from_vec(sin, (1, 1, seq, half), device)?,
    ))
}

/// Rotate-half RoPE on `(b, heads, seq, head_dim)`: `y1 = x1·cos − x2·sin`,
/// `y2 = x2·cos + x1·sin` over the two halves of each head dimension.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;
    let y1 = (x1.broadcast_mul(cos)? - &x2.broadcast_mul(sin)?)?;
    let y2 = (x2.broadcast_mul(cos)? + &x1.broadcast_mul(sin)?)?;
    Ok(Tensor::cat(&[&y1, &y2], 3)?)
}

/// Additive key-padding mask `(b, 1, 1, seq)`: 0 for real tokens, `f32::MIN` for padding —
/// transformers' `_prepare_4d_attention_mask` convention.
fn additive_pad_mask(mask: &Tensor) -> Result<Tensor> {
    let (b, seq) = mask.dims2()?;
    let rows = mask.to_vec2::<u8>()?;
    let mut data = Vec::with_capacity(b * seq);
    for row in &rows {
        for &m in row {
            data.push(if m == 0 { f32::MIN } else { 0.0 });
        }
    }
    Ok(Tensor::from_vec(data, (b, 1, 1, seq), mask.device())?)
}

/// Additive sliding-window mask `(1, 1, seq, seq)`: 0 within radius
/// `local_attention / 2` of each position, `f32::MIN` outside.
fn additive_window_mask(seq: usize, local_attention: usize, device: &Device) -> Result<Tensor> {
    let radius = local_attention / 2;
    let mut data = Vec::with_capacity(seq * seq);
    for i in 0..seq {
        for j in 0..seq {
            let dist = i.abs_diff(j);
            data.push(if dist <= radius { 0.0 } else { f32::MIN });
        }
    }
    Ok(Tensor::from_vec(data, (1, 1, seq, seq), device)?)
}

/// Per-forward attention context: additive masks plus RoPE tables for both patterns.
struct LayerContext<'a> {
    pad_add: &'a Tensor,
    window_add: Option<&'a Tensor>,
    global: (&'a Tensor, &'a Tensor),
    local: (&'a Tensor, &'a Tensor),
}

/// One ModernBERT layer. Layer 0 skips `attn_norm` (residual starts from the embedding
/// norm), matching the reference implementation.
struct EncLayer {
    kind: AttentionKind,
    attn_norm: Option<LayerNorm>,
    wqkv: Linear,
    wo: Linear,
    mlp_norm: LayerNorm,
    wi: Linear,
    wo2: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl EncLayer {
    fn from_vb(
        vb: VarBuilder<'_>,
        cfg: &EncoderConfig,
        kind: AttentionKind,
        first: bool,
    ) -> Result<Self> {
        let hidden = cfg.hidden;
        Ok(Self {
            kind,
            attn_norm: if first {
                None
            } else {
                Some(LayerNorm::new_no_bias(
                    vb.get((hidden,), "attn_norm.weight")?,
                    cfg.eps,
                ))
            },
            wqkv: Linear::new(vb.get((3 * hidden, hidden), "attn.Wqkv.weight")?, None),
            wo: Linear::new(vb.get((hidden, hidden), "attn.Wo.weight")?, None),
            mlp_norm: LayerNorm::new_no_bias(vb.get((hidden,), "mlp_norm.weight")?, cfg.eps),
            wi: Linear::new(
                vb.get((2 * cfg.intermediate, hidden), "mlp.Wi.weight")?,
                None,
            ),
            wo2: Linear::new(vb.get((hidden, cfg.intermediate), "mlp.Wo.weight")?, None),
            heads: cfg.heads,
            head_dim: cfg.head_dim,
            scale: 1.0 / (cfg.head_dim as f64).sqrt(),
        })
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, h: &Tensor, ctx: &LayerContext<'_>) -> Result<Tensor> {
        let attn_in = match &self.attn_norm {
            Some(norm) => norm.forward(h)?,
            None => h.clone(),
        };
        let qkv = self.wqkv.forward(&attn_in)?;
        let (b, seq, _) = qkv.dims3()?;
        let qkv = qkv.reshape((b, seq, 3, self.heads, self.head_dim))?;
        let split =
            |i: usize| -> Result<Tensor> { Ok(qkv.narrow(2, i, 1)?.squeeze(2)?.transpose(1, 2)?) };
        let (cos, sin) = match self.kind {
            AttentionKind::Global => ctx.global,
            AttentionKind::Sliding => ctx.local,
        };
        let q = apply_rope(&split(0)?, cos, sin)?.affine(self.scale, 0.0)?;
        let k = apply_rope(&split(1)?, cos, sin)?;
        let v = split(2)?.contiguous()?;
        let mut scores = q.matmul(&k.t()?)?.broadcast_add(ctx.pad_add)?;
        if self.kind == AttentionKind::Sliding {
            if let Some(window) = ctx.window_add {
                scores = scores.broadcast_add(window)?;
            }
        }
        let probs = ops::softmax(&scores, D::Minus1)?;
        let attn = probs.matmul(&v)?;
        let attn = attn.transpose(1, 2)?.flatten_from(2)?;
        let h = (h + self.wo.forward(&attn)?)?;
        let mlp_in = self.mlp_norm.forward(&h)?;
        let fused = self.wi.forward(&mlp_in)?;
        let (_, _, two_inter) = fused.dims3()?;
        let inter = two_inter / 2;
        let act = fused.narrow(2, 0, inter)?;
        let gate = fused.narrow(2, inter, inter)?;
        let mlp_out = self.wo2.forward(&(act.gelu_erf()? * gate)?)?;
        Ok((h + mlp_out)?)
    }
}

/// The ModernBERT backbone of a laya checkpoint (`encoder.*` tensors).
pub struct ModernBertEncoder {
    tok_embeddings: Embedding,
    emb_norm: LayerNorm,
    layers: Vec<EncLayer>,
    final_norm: LayerNorm,
    cfg: EncoderConfig,
}

impl ModernBertEncoder {
    /// Build from the `encoder.` sub-tree of a laya checkpoint.
    ///
    /// # Errors
    /// Returns [`crate::LuminaError::Candle`] when a tensor is missing or mis-shaped; the
    /// requested shapes encode the architecture, so mismatches surface here.
    pub fn from_vb(vb: VarBuilder<'_>, cfg: &EncoderConfig) -> Result<Self> {
        let hidden = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for (i, kind) in cfg.layers.iter().copied().enumerate() {
            layers.push(EncLayer::from_vb(
                vb.pp(format!("layers.{i}")),
                cfg,
                kind,
                i == 0,
            )?);
        }
        Ok(Self {
            tok_embeddings: Embedding::new(
                vb.get((cfg.vocab, hidden), "embeddings.tok_embeddings.weight")?,
                hidden,
            ),
            emb_norm: LayerNorm::new_no_bias(vb.get((hidden,), "embeddings.norm.weight")?, cfg.eps),
            layers,
            final_norm: LayerNorm::new_no_bias(vb.get((hidden,), "final_norm.weight")?, cfg.eps),
            cfg: cfg.clone(),
        })
    }

    /// Run the encoder over a padded batch.
    ///
    /// # Errors
    /// Returns [`crate::LuminaError::Candle`] on shape or compute failures, and for an
    /// empty sequence.
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let (_, seq) = input_ids.dims2()?;
        if seq == 0 {
            return Err(candle_core::Error::msg("empty input sequence").into());
        }
        let device = input_ids.device();
        let mut h = self
            .emb_norm
            .forward(&self.tok_embeddings.forward(input_ids)?)?;
        let pad_add = additive_pad_mask(attention_mask)?;
        let need_window = self.cfg.layers.contains(&AttentionKind::Sliding);
        let window_add = need_window
            .then(|| additive_window_mask(seq, self.cfg.local_attention, device))
            .transpose()?;
        let global = rope_tables(self.cfg.global_theta, seq, self.cfg.head_dim, device)?;
        let local = rope_tables(self.cfg.local_theta, seq, self.cfg.head_dim, device)?;
        let ctx = LayerContext {
            pad_add: &pad_add,
            window_add: window_add.as_ref(),
            global: (&global.0, &global.1),
            local: (&local.0, &local.1),
        };
        for layer in &self.layers {
            h = layer.forward(&h, &ctx)?;
        }
        Ok(self.final_norm.forward(&h)?)
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
    fn rope_tables_cover_unit_circle() {
        let (cos, sin) = rope_tables(10_000.0, 4, 8, &Device::Cpu).unwrap();
        assert_eq!(cos.dims(), [1, 1, 4, 4]);
        let c = cos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let s = sin.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // position 0: cos = 1, sin = 0 on every frequency
        for j in 0..4 {
            assert!((c[j] - 1.0).abs() < 1e-6);
            assert!(s[j].abs() < 1e-6);
        }
        // position 1, frequency j=0 (inv_freq 1.0, one rad per step): angle = 1.0 rad
        assert!((c[4] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((s[4] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn window_mask_blocks_beyond_radius() {
        let m = additive_window_mask(6, 4, &Device::Cpu).unwrap();
        let v = m.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // dist(i, j) > 2 must be f32::MIN; (0, 3) has dist 3 → blocked, (0, 2) is open
        assert_eq!(v[3], f32::MIN);
        assert_eq!(v[2], 0.0);
    }

    #[test]
    fn apply_rope_broadcasts_tables_across_batch_and_heads() {
        let dev = Device::Cpu;
        let (cos, sin) = rope_tables(10_000.0, 3, 4, &dev).unwrap();
        let base = Tensor::arange(0f32, 12f32, &dev)
            .unwrap()
            .reshape((1, 1, 3, 4))
            .unwrap();
        let x = base.broadcast_as((2, 2, 3, 4)).unwrap();
        let y = apply_rope(&x, &cos, &sin).unwrap();
        assert_eq!(y.dims(), [2, 2, 3, 4]);
        let got = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for block in got.chunks(12) {
            assert_eq!(block, &got[0..12]);
        }
        let rot_c = 4.0 * 1.0_f32.cos();
        let rot_s = 6.0 * 1.0_f32.sin();
        let expected = rot_c - rot_s;
        assert!((got[4] - expected).abs() < 1e-5);
    }
}
