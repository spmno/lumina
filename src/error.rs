//! Error type for the lumina crate.

/// Errors returned by the lumina library.
#[derive(Debug, thiserror::Error)]
pub enum LuminaError {
    /// Filesystem / I/O failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// JSON (de)serialization failure.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),

    /// Candle tensor operation failure.
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),

    /// Hugging Face Hub download failure.
    #[error("hub: {0}")]
    Hub(#[from] hf_hub::HFError),

    /// Tokenizer load or encode failure.
    #[error("tokenizer: {0}")]
    Tokenizer(String),

    /// A question failed validation; `qid` names the question.
    #[error("question {qid:?}: {msg}")]
    Question {
        /// Identifier of the offending question.
        qid: String,
        /// What to fix.
        msg: String,
    },

    /// The checkpoint does not match the architecture lumina implements.
    #[error("incompatible model: {0}")]
    Incompatible(String),

    /// Invalid argument, request payload, or state.
    #[error("{0}")]
    Invalid(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, LuminaError>;

impl LuminaError {
    /// Whether this error is a caller-input problem — laya's `ValueError` class —
    /// worth reporting verbatim (the CLI prints it as `lumina: …`, an HTTP server
    /// answers 422 with it) rather than through the generic runtime-failure path.
    pub const fn is_validation(&self) -> bool {
        matches!(
            self,
            Self::Invalid(_) | Self::Question { .. } | Self::Incompatible(_)
        )
    }
}
