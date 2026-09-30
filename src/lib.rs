//! Typed-decision System 1 engine: a Rust port of [laya](https://github.com/NandhaKishorM/laya)
//! (Jev) built on [candle](https://github.com/huggingface/candle).
//!
//! lumina loads compatible checkpoints from a hub repo or local directory (e.g.
//! laya's `convaiinnovations/laya`) and answers typed
//! questions — `choice`, `score`, `noul` — over a state in a single non-autoregressive forward
//! pass, reproducing laya's numerics (sequence format, head architecture, temperature-scaled
//! decoding, confidence reporting).
//!
//! Quick start:
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use lumina::{Agent, Question};
//!
//! let agent = Agent::load("convaiinnovations/laya")?;
//! let questions = vec![
//!     ("department".into(), Question::from_def("department", &serde_json::json!({
//!         "type": "choice",
//!         "instructions": "Which department should handle this ticket?",
//!         "criteria": {"billing": "Payment, invoices, refunds", "technical": "Bugs and outages"}
//!     }))?),
//! ];
//! let state = serde_json::json!("Customer asks for a refund of last month's invoice.");
//! let result = agent.predict(&state, &questions)?;
//! println!("{}", serde_json::to_string_pretty(&result)?);
//! # Ok(())
//! # }
//! ```

pub mod agent;
pub mod confidence;
pub mod config;
pub mod decode;
pub mod encoder;
pub mod error;
pub mod hub;
pub mod lang;
mod lang_data;
pub mod model;
pub mod presets;
pub mod pyjson;
pub mod question;
pub mod router;
pub mod routing;
pub mod sequence;
pub mod tokenizer;

pub use agent::{Agent, PredictOverrides};
pub use confidence::{check_min_confidence, flag_low_confidence};
pub use config::{AgentConfig, EncoderConfig};
pub use error::{LuminaError, Result};
pub use lang::{Analysis, LatinProfile, analyse, detect_script, is_english, script_profile};
pub use presets::{
    email_questions, guard_questions, moderation_questions, router_questions, triage_questions,
};
pub use question::{Question, QuestionType};
pub use router::{PredictOptions, Router, RouterOptions, parse_model_sources, parse_questions};
pub use routing::{RouteDecision, RouteOptions, normalise_name};
pub use sequence::{EncodedItem, SequenceStats};
