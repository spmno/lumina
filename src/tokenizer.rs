//! Tokenizer loading and encoding for laya checkpoints.

use std::path::Path;

use serde_json::Value;
use tokenizers::Tokenizer;

use crate::{LuminaError, Result};

/// Special-token ids resolved from `tokenizer_config.json` through the vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokens {
    /// Token that opens every sequence (`[CLS]` / `<bos>`).
    pub cls: u32,
    /// Token separating the head, option block and state (`[SEP]` / `<eos>`).
    pub sep: u32,
    /// Padding token (`[PAD]` / `<pad>`).
    pub pad: u32,
    /// Unknown-token id.
    pub unk: u32,
    /// The mask token that marks each option slot (`⌜mask⌟` / `<mask>`).
    pub mask: u32,
}

/// Resolve one special token: its string from `tokenizer_config.json`, its id from the
/// vocabulary. `keys` lists acceptable config keys (primary first).
fn resolve_token(cfg: &Value, tk: &Tokenizer, keys: &[&str]) -> Result<(String, u32)> {
    for key in keys {
        if let Some(s) = cfg.get(*key).and_then(Value::as_str) {
            let id = tk.token_to_id(s).ok_or_else(|| {
                LuminaError::Incompatible(format!("token {s:?} ({key}) is not in the vocabulary"))
            })?;
            return Ok((s.to_owned(), id));
        }
    }
    Err(LuminaError::Incompatible(format!(
        "tokenizer_config.json is missing {}",
        keys.first().unwrap_or(&"special token")
    )))
}

/// A checkpoint's tokenizer plus its resolved special tokens.
pub struct TokenizerBundle {
    /// The parsed tokenizer (read-only after load, shareable across threads).
    pub tokenizer: Tokenizer,
    /// Special token ids.
    pub specials: SpecialTokens,
    /// The mask token string; option texts have it stripped before encoding.
    pub mask_str: String,
}

impl TokenizerBundle {
    /// Load `tokenizer.json` and `tokenizer_config.json` from a checkpoint's tokenizer
    /// directory.
    ///
    /// # Errors
    /// Returns [`LuminaError::Tokenizer`] when `tokenizer.json` cannot be parsed,
    /// [`LuminaError::Io`]/[`LuminaError::Json`] for the config, and
    /// [`LuminaError::Incompatible`] when special tokens are missing or unknown.
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path)
            .map_err(|e| LuminaError::Tokenizer(format!("{}: {e}", path.display())))?;
        let cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer_config.json"))?)?;
        let (cls_str, cls) = resolve_token(&cfg, &tokenizer, &["cls_token", "bos_token"])?;
        let (sep_str, sep) = resolve_token(&cfg, &tokenizer, &["sep_token", "eos_token"])?;
        let (_, pad) = resolve_token(&cfg, &tokenizer, &["pad_token"])?;
        let (_, unk) = resolve_token(&cfg, &tokenizer, &["unk_token"])?;
        let (mask_str, mask) = resolve_token(&cfg, &tokenizer, &["mask_token"])?;
        let _ = (cls_str, sep_str);
        Ok(Self {
            tokenizer,
            specials: SpecialTokens {
                cls,
                sep,
                pad,
                unk,
                mask,
            },
            mask_str,
        })
    }

    /// Encode `text` without adding special tokens — laya assembles specials itself in
    /// [`crate::sequence`].
    ///
    /// # Errors
    /// Returns [`LuminaError::Tokenizer`] when tokenization fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|enc| enc.get_ids().to_vec())
            .map_err(|e| LuminaError::Tokenizer(format!("{text:?}: {e}")))
    }

    /// Encode and truncate to at most `max` ids (used for the question head, whose budget
    /// is dynamic).
    ///
    /// # Errors
    /// Returns [`LuminaError::Tokenizer`] when tokenization fails.
    pub fn encode_truncated(&self, text: &str, max: usize) -> Result<Vec<u32>> {
        let mut ids = self.encode(text)?;
        ids.truncate(max);
        Ok(ids)
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

    fn bundle() -> Option<TokenizerBundle> {
        TokenizerBundle::load(Path::new("/tmp/opencode/laya-hf/tokenizer")).ok()
    }

    #[test]
    fn resolves_english_special_tokens() {
        if let Some(b) = bundle() {
            assert_eq!(
                b.specials,
                SpecialTokens {
                    cls: 50281,
                    sep: 50282,
                    pad: 50283,
                    unk: 50280,
                    mask: 50284
                }
            );
            assert_eq!(b.mask_str, "[MASK]");
        }
    }

    #[test]
    fn encodes_without_specials() {
        if let Some(b) = bundle() {
            let ids = b.encode("hello").unwrap();
            assert!(!ids.is_empty());
            assert!(!ids.contains(&b.specials.cls));
            assert!(!ids.contains(&b.specials.sep));
            let trunc = b.encode_truncated("hello world", 2).unwrap();
            assert_eq!(trunc.len(), 2);
        }
    }
}
