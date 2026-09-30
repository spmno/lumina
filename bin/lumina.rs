//! Command-line interface for testing Lumina locally.
//!
//!     lumina "I was charged twice, please refund"           # routing decision only (no model download)
//!     lumina "Refactor this service" --predict              # full answers, loads the checkpoint
//!     lumina                                                # interactive mode
//!     lumina "Mein Konto wurde zweimal belastet" --lang de  # explicit language
//!     lumina "My payment failed twice" --model ml           # pin a checkpoint, by name or alias
//!     lumina "My payment failed twice" --preset triage      # a ready-made question preset
//!     lumina --batch tickets.txt --predict                  # a file of requests, one per line
//!     cat tickets.txt | lumina --batch - --predict --json   # stdin, one JSON line of answers each
//!     lumina "Where is my card" --questions intents.json    # your own questions, from a JSON file
//!     lumina "Where is my card" --questions intents.json --head-max-len 384
//!
//! Routing (the default) never downloads a checkpoint, so it works offline and
//! returns in milliseconds. `--predict` loads the routed checkpoint on first use,
//! which needs network access to the Hugging Face hub. Checkpoint sources come
//! from `LAYA_MODELS` — `name=source` bindings (hub repo, optional
//! `:subfolder`, or a local directory; see `lumina::parse_model_sources`).
//! `--preset` answers one of
//! the ready-made question presets from [`lumina::presets`] and implies
//! `--predict`. `--batch` scores a whole file (or stdin, with `-`) through
//! [`Router::predict_batch`](lumina::Router::predict_batch) in one process, so
//! the states share checkpoint loads; `--json` prints JSONL. `--questions`
//! answers the question set in a JSON file instead; `--max-len` and
//! `--head-max-len` raise its token budget for the request, which is what a
//! many-label question needs.
//!
//! Deviation from laya: `--batch-size` is not ported — batch requests are grouped
//! by routed checkpoint, but each state runs its own forward pass.

use std::io::{Read, Write};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use serde_json::{Value, json};

use lumina::pyjson::{py_repr, py_type_name, to_python_json};
use lumina::{
    LuminaError, PredictOptions, RouteOptions, Router, RouterOptions, normalise_name,
    parse_model_sources, parse_questions,
};

/// A failure the CLI reports as either a usage problem (laya's `ValueError`
/// class) or a runtime problem (laya's `ImportError`/`OSError`/`RuntimeError`
/// class, which gets the check-the-hub hint).
enum CliError {
    /// A user-input problem, printed as `lumina: <message>`.
    Usage(String),
    /// A runtime failure, printed with the hub hint.
    Runtime(LuminaError),
}

impl From<LuminaError> for CliError {
    fn from(error: LuminaError) -> Self {
        if error.is_validation() {
            Self::Usage(error.to_string())
        } else {
            Self::Runtime(error)
        }
    }
}

impl From<String> for CliError {
    fn from(message: String) -> Self {
        Self::Usage(message)
    }
}

/// Ready-made question presets `--preset` can answer.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Preset {
    /// Inbound email triage and threat filtering (reads `body`).
    Email,
    /// Real-time LLM input guardrails (reads `prompt`).
    Guard,
    /// Content safety and moderation (reads `post`).
    Moderation,
    /// Intelligent model routing (reads `request`).
    Router,
    /// Customer support ticket triage (reads `message`).
    Triage,
}

impl Preset {
    fn questions(self) -> Value {
        match self {
            Self::Email => lumina::presets::email_questions(),
            Self::Guard => lumina::presets::guard_questions(),
            Self::Moderation => lumina::presets::moderation_questions(),
            Self::Router => lumina::presets::router_questions(),
            Self::Triage => lumina::presets::triage_questions(),
        }
    }

    const fn state_key(self) -> &'static str {
        match self {
            Self::Email => "body",
            Self::Guard => "prompt",
            Self::Moderation => "post",
            Self::Router => "request",
            Self::Triage => "message",
        }
    }
}

/// Resolve a `--model` argument the way the library resolves it: same names,
/// same aliases, same casing. "auto" means "do not pin one", which is what
/// omitting the flag means.
fn parse_model(value: &str) -> Result<String, String> {
    if value.trim().eq_ignore_ascii_case("auto") {
        return Ok("auto".to_owned());
    }
    normalise_name(value).map_err(|error| format!("{error}, or 'auto'"))
}

#[derive(Parser)]
#[command(
    name = "lumina",
    about = "Test Lumina locally: route or answer a request from the command line."
)]
struct Cli {
    /// the request text (omit for interactive mode)
    text: Vec<String>,

    /// run the full prediction, not just the routing decision (downloads the checkpoint on first use)
    #[arg(long)]
    predict: bool,

    /// force a checkpoint instead of auto-routing: a checkpoint name or any of the library's aliases, in any casing, or 'auto' to route it (the default)
    #[arg(long, value_name = "NAME", value_parser = parse_model)]
    model: Option<String>,

    /// force a language, e.g. en or de, instead of detecting it
    #[arg(long)]
    lang: Option<String>,

    /// force a typed-decisions workflow instead of detecting it
    #[arg(long)]
    task: Option<String>,

    /// answer a ready-made question preset instead of the router questions; implies --predict
    #[arg(long, value_enum, value_name = "NAME")]
    preset: Option<Preset>,

    /// a JSON file of your own typed questions to answer instead of the router questions or a preset; implies --predict
    #[arg(long, value_name = "FILE")]
    questions: Option<String>,

    /// token budget for the whole request (state plus options); defaults to the checkpoint's own budget
    #[arg(long, value_name = "N")]
    max_len: Option<usize>,

    /// token budget the choice options share; raise it when a question has many labels, so each keeps enough tokens to stay distinct
    #[arg(long, value_name = "N")]
    head_max_len: Option<usize>,

    /// device, e.g. cpu or cuda
    #[arg(long)]
    device: Option<String>,

    /// print the raw result as JSON
    #[arg(long)]
    json: bool,

    /// score a file of requests, one per line (use '-' for stdin), instead of a single text; implies neither --predict nor --preset, but the same modes apply: routing by default, answers with --predict/--preset
    #[arg(long, value_name = "FILE")]
    batch: Option<String>,
}

impl Cli {
    /// The pin from `--model` if one was given; both no flag and "auto" mean
    /// auto-routing.
    fn pinned_model(&self) -> Option<&str> {
        match self.model.as_deref() {
            None | Some("auto") => None,
            Some(name) => Some(name),
        }
    }
}

fn main() -> ExitCode {
    let args = Cli::parse();
    let text = args.text.join(" ").trim().to_owned();
    if let Some(batch) = &args.batch {
        if !text.is_empty() {
            eprintln!("lumina: pass either a text or --batch FILE, not both.");
            return ExitCode::from(2);
        }
        let lines = match read_batch_lines(batch) {
            Ok(lines) => lines,
            Err(message) => {
                eprintln!("lumina: {message}");
                return ExitCode::from(2);
            }
        };
        if lines.is_empty() {
            eprintln!("lumina: no requests found in {batch}.");
            return ExitCode::from(2);
        }
        return report(run_batch(&lines, &args));
    }
    if text.is_empty() {
        return interactive(&args);
    }
    report(run(&text, &args))
}

fn report(outcome: Result<(), CliError>) -> ExitCode {
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            print_error(&error);
            ExitCode::from(2)
        }
    }
}

fn print_error(error: &CliError) {
    match error {
        CliError::Usage(message) => eprintln!("lumina: {message}"),
        CliError::Runtime(error) => {
            eprintln!("lumina: could not run Lumina ({error}).");
            eprintln!(
                "Check that the checkpoints can be downloaded from the Hugging Face hub (network access is needed on first use)."
            );
        }
    }
}

fn make_router(args: &Cli) -> Result<Router, CliError> {
    let models = parse_model_sources(&std::env::var("LAYA_MODELS").unwrap_or_default())?;
    let registry: Vec<(&str, &str, Option<&str>)> = models
        .iter()
        .map(|(name, repo, subfolder)| (name.as_str(), repo.as_str(), subfolder.as_deref()))
        .collect();
    Ok(Router::new(&RouterOptions {
        device: args.device.as_deref(),
        models: &registry,
        ..RouterOptions::default()
    })?)
}

/// Route or predict one request.
fn run(text: &str, args: &Cli) -> Result<(), CliError> {
    let router = make_router(args)?;
    run_with_router(text, args, &router)
}

fn run_with_router(text: &str, args: &Cli, router: &Router) -> Result<(), CliError> {
    if args.predict || args.preset.is_some() || args.questions.is_some() {
        let (questions_value, state_key) = resolve_questions(args)?;
        let questions = parse_questions(&questions_value)?;
        let state = text_state(&state_key, text);
        let result = router.predict(
            &state,
            &questions,
            &PredictOptions {
                model: args.pinned_model(),
                task: args.task.as_deref(),
                lang: args.lang.as_deref(),
                lang_guess: None,
                max_len: args.max_len,
                head_max_len: args.head_max_len,
                min_confidence: None,
            },
        )?;
        if args.json {
            println!("{}", pretty_json(&result));
        } else {
            show_answers(&result);
        }
    } else {
        let state = text_state("text", text);
        let decision = router.route(
            &state,
            &[],
            &RouteOptions {
                model: args.pinned_model(),
                task: args.task.as_deref(),
                lang: args.lang.as_deref(),
                lang_guess: None,
            },
        )?;
        let decision_value = serde_json::to_value(&decision).unwrap_or_default();
        if args.json {
            println!("{}", pretty_json(&decision_value));
        } else {
            show_decision(&decision_value);
        }
    }
    Ok(())
}

/// `{key: text}` — the state each CLI path sends its text under; routing reads
/// the state only for language detection, which is key-invariant, so the default
/// path uses `"text"` while prediction paths use the key their question set names.
fn text_state(key: &str, text: &str) -> Value {
    let mut state = serde_json::Map::new();
    state.insert(key.to_owned(), json!(text));
    Value::Object(state)
}

/// The question set to answer and the state field it reads, from the flags given.
fn resolve_questions(args: &Cli) -> Result<(Value, String), CliError> {
    if args.questions.is_some() && args.preset.is_some() {
        return Err(CliError::Usage(
            "--questions and --preset choose different question sets; pass one".to_owned(),
        ));
    }
    if let Some(path) = &args.questions {
        return load_questions(path).map_err(CliError::Usage);
    }
    if let Some(preset) = args.preset {
        return Ok((preset.questions(), preset.state_key().to_owned()));
    }
    Ok((lumina::presets::router_questions(), "request".to_owned()))
}

/// Read a `--questions` file: returns `(questions, state_key)`.
///
/// The file is either a bare `question id -> definition` mapping, or an object
/// holding one under `"questions"` plus an optional `"state_key"` naming the field
/// the instructions refer to (default `"request"`, matching bare `--predict`).
fn load_questions(path: &str) -> Result<(Value, String), String> {
    let raw =
        std::fs::read_to_string(path).map_err(|_| format!("no such --questions file: {path}"))?;
    let raw: Value = serde_json::from_str(&raw).map_err(|error| format!("{path}: {error}"))?;
    let object = raw.as_object().ok_or_else(|| {
        "--questions file must be a JSON object of question id -> definition".to_owned()
    })?;
    let mut state_key = "request".to_owned();
    let questions = match object.get("questions") {
        None => raw.clone(),
        Some(questions) => {
            if questions.as_object().is_none() {
                return Err(
                    "the 'questions' field of the --questions file must be an object".to_owned(),
                );
            }
            state_key = match object.get("state_key") {
                None => "request".to_owned(),
                Some(value) => value
                    .as_str()
                    .map(str::to_owned)
                    .filter(|key| !key.trim().is_empty())
                    .ok_or_else(|| {
                        format!(
                            "'state_key' must be a non-empty string, got {}",
                            py_repr(value)
                        )
                    })?,
            };
            questions.clone()
        }
    };
    let question_map = questions.as_object().ok_or_else(|| {
        "--questions file must be a JSON object of question id -> definition".to_owned()
    })?;
    if question_map.is_empty() {
        return Err("--questions file holds no questions".to_owned());
    }
    for (qid, question) in question_map {
        if question.as_object().is_none() {
            return Err(format!(
                "question {} must map to an object, got {}",
                py_repr(&json!(qid)),
                py_type_name(question)
            ));
        }
    }
    Ok((questions, state_key))
}

fn show_decision(decision: &Value) {
    println!(
        "Model     : {}",
        decision
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
    );
    println!(
        "Reason    : {}",
        decision
            .get("reason")
            .and_then(Value::as_str)
            .unwrap_or_default()
    );
    if let Some(detection) = decision
        .get("detection")
        .filter(|d| !d.is_null() && d.as_object().is_none_or(|map| !map.is_empty()))
    {
        println!("Detected  : {}", python_json(detection));
    }
}

fn show_answers(result: &Value) {
    if let Some(routing) = result.get("routing").filter(|r| !r.is_null()) {
        show_decision(routing);
        println!();
    }
    if let Some(answers) = result.get("answers").and_then(Value::as_object) {
        for (qid, answer) in answers {
            let detail = match (
                answer.get("choice").and_then(Value::as_str),
                answer.get("score").and_then(Value::as_f64),
                answer.get("noul").and_then(Value::as_f64),
            ) {
                (Some(choice), _, _) => {
                    let probability = answer
                        .get("probabilities")
                        .and_then(|p| p.get(choice))
                        .and_then(Value::as_f64)
                        .unwrap_or(0.0);
                    format!("{choice} (p={probability:.3})")
                }
                (None, Some(score), _) => format!("{score:.2}"),
                (None, None, Some(noul)) => format!("{noul:.3}"),
                (None, None, None) => python_json(answer),
            };
            println!("{qid:<12}: {detail}");
        }
    }
}

/// One request per non-blank line of FILE, or of stdin for '-'.
fn read_batch_lines(source: &str) -> Result<Vec<String>, String> {
    let content = if source == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|error| format!("could not read stdin ({error})."))?;
        buffer
    } else {
        std::fs::read_to_string(source)
            .map_err(|error| format!("could not read {source} ({error})."))?
    };
    Ok(content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect())
}

/// Route or predict a whole file in one call, so states share checkpoint loads.
fn run_batch(lines: &[String], args: &Cli) -> Result<(), CliError> {
    let router = make_router(args)?;
    let model = args.pinned_model();
    if args.predict || args.preset.is_some() || args.questions.is_some() {
        let (questions_value, state_key) = resolve_questions(args)?;
        let requests: Vec<Value> = lines
            .iter()
            .map(|line| {
                let mut request = json!({
                    "state": text_state(&state_key, line),
                    "questions": questions_value.clone(),
                });
                set_str_field(&mut request, "model", model);
                set_str_field(&mut request, "task", args.task.as_deref());
                set_str_field(&mut request, "lang", args.lang.as_deref());
                set_usize_field(&mut request, "max_len", args.max_len);
                set_usize_field(&mut request, "head_max_len", args.head_max_len);
                request
            })
            .collect();
        let results = router.predict_batch(&requests, None)?;
        for (line, result) in lines.iter().zip(&results) {
            if args.json {
                println!("{}", python_json(result));
            } else {
                println!("# {}", truncate(line, 72));
                show_answers(result);
                println!();
            }
        }
    } else {
        let requests: Vec<Value> = lines
            .iter()
            .map(|line| {
                let mut request = json!({
                    "state": text_state("text", line),
                    "questions": {},
                });
                set_str_field(&mut request, "model", model);
                set_str_field(&mut request, "task", args.task.as_deref());
                set_str_field(&mut request, "lang", args.lang.as_deref());
                request
            })
            .collect();
        let decisions = router.route_batch(&requests)?;
        for (line, decision) in lines.iter().zip(&decisions) {
            if args.json {
                println!(
                    "{}",
                    python_json(&serde_json::to_value(decision).unwrap_or_default())
                );
            } else {
                println!(
                    "{:<14} {}  <- {}",
                    decision.model,
                    decision.reason,
                    truncate(line, 48)
                );
            }
        }
    }
    Ok(())
}

fn set_str_field(request: &mut Value, key: &str, value: Option<&str>) {
    if let (Some(value), Some(object)) = (value, request.as_object_mut()) {
        object.insert(key.to_owned(), json!(value));
    }
}

fn set_usize_field(request: &mut Value, key: &str, value: Option<usize>) {
    if let (Some(value), Some(object)) = (value, request.as_object_mut()) {
        object.insert(key.to_owned(), json!(value));
    }
}

fn interactive(args: &Cli) -> ExitCode {
    println!("Lumina interactive mode. Type a request and press Enter; Ctrl-D or 'quit' to exit.");
    let router = match make_router(args) {
        Ok(router) => router,
        Err(error) => return report(Err(error)),
    };
    let stdin = std::io::stdin();
    loop {
        print!("lumina> ");
        if std::io::stdout().flush().is_err() {
            break;
        }
        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) | Err(_) => {
                println!();
                break;
            }
            Ok(_) => {}
        }
        let text = line.trim();
        if text.is_empty() || text.eq_ignore_ascii_case("quit") || text.eq_ignore_ascii_case("exit")
        {
            break;
        }
        if let Err(error) = run_with_router(text, args, &router) {
            print_error(&error);
        }
    }
    ExitCode::SUCCESS
}

fn truncate(line: &str, limit: usize) -> String {
    if line.chars().count() <= limit {
        line.to_owned()
    } else {
        let cut: String = line.chars().take(limit.saturating_sub(3)).collect();
        format!("{cut}...")
    }
}

fn pretty_json(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_default()
}

fn python_json(value: &Value) -> String {
    to_python_json(value).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use clap::Parser;

    use super::*;

    #[test]
    fn model_flag_resolves_like_the_library() {
        assert_eq!(parse_model("auto").unwrap(), "auto");
        assert_eq!(parse_model("AUTO").unwrap(), "auto");
        assert_eq!(parse_model("ml").unwrap(), "multilingual");
        assert_eq!(parse_model(" Typed-Decisions ").unwrap(), "typed-decisions");
        assert_eq!(
            parse_model("bogus").unwrap_err(),
            "unknown model 'bogus'; choose one of ['english', 'multilingual', 'typed-decisions'] (or an alias: ['decisions', 'default', 'en', 'ml', 'multi', 'typed', 'typed_decisions']), or 'auto'"
        );
        let args = Cli::try_parse_from(["lumina", "hi", "--model", "en"]).unwrap();
        assert_eq!(args.pinned_model(), Some("english"));
        let args = Cli::try_parse_from(["lumina", "hi", "--model", "auto"]).unwrap();
        assert_eq!(args.pinned_model(), None);
        // an invalid model name is a clap-level rejection, like argparse's choices
        assert!(Cli::try_parse_from(["lumina", "hi", "--model", "bogus"]).is_err());
    }

    #[test]
    fn batch_conflicts_with_text_and_keeps_flags() {
        let err = Cli::try_parse_from(["lumina", "hi", "--model", "bogus"]);
        assert!(err.is_err());
        let args = Cli::try_parse_from([
            "lumina",
            "--batch",
            "tickets.txt",
            "--predict",
            "--max-len",
            "2048",
            "--head-max-len",
            "384",
            "--preset",
            "triage",
        ])
        .unwrap();
        assert_eq!(args.batch.as_deref(), Some("tickets.txt"));
        assert!(args.predict);
        assert_eq!(args.max_len, Some(2048));
        assert_eq!(args.head_max_len, Some(384));
        assert_eq!(args.preset.map(Preset::state_key), Some("message"));
    }

    #[test]
    fn truncation_matches_python_slicing() {
        assert_eq!(truncate("short", 48), "short");
        let long = "x".repeat(49);
        assert_eq!(truncate(&long, 48), format!("{}...", "x".repeat(45)));
    }

    #[test]
    fn load_questions_reports_missing_files() {
        let error = load_questions("/nonexistent/intents.json").unwrap_err();
        assert_eq!(error, "no such --questions file: /nonexistent/intents.json");
    }

    #[test]
    fn load_questions_reads_bare_and_wrapped_forms() {
        let bare = std::env::temp_dir().join("lumina-cli-bare.json");
        std::fs::write(
            &bare,
            r#"{"tone": {"type": "noul", "instructions": "Rude in `request`?"}}"#,
        )
        .unwrap();
        let (questions, state_key) = load_questions(bare.to_str().unwrap()).unwrap();
        assert_eq!(state_key, "request");
        assert!(questions.get("tone").is_some());

        let wrapped = std::env::temp_dir().join("lumina-cli-wrapped.json");
        std::fs::write(
            &wrapped,
            r#"{"questions": {"tone": {"type": "noul", "instructions": "Rude in `body`?"}},
                "state_key": "body"}"#,
        )
        .unwrap();
        let (questions, state_key) = load_questions(wrapped.to_str().unwrap()).unwrap();
        assert_eq!(state_key, "body");
        assert!(questions.get("tone").is_some());

        let bad_key = std::env::temp_dir().join("lumina-cli-bad-key.json");
        std::fs::write(
            &bad_key,
            r#"{"questions": {"tone": {"type": "noul", "instructions": "?"}}, "state_key": 5}"#,
        )
        .unwrap();
        let error = load_questions(bad_key.to_str().unwrap()).unwrap_err();
        assert_eq!(error, "'state_key' must be a non-empty string, got 5");

        let not_object = std::env::temp_dir().join("lumina-cli-not-object.json");
        std::fs::write(&not_object, r#"["nope"]"#).unwrap();
        let error = load_questions(not_object.to_str().unwrap()).unwrap_err();
        assert_eq!(
            error,
            "--questions file must be a JSON object of question id -> definition"
        );
    }

    #[test]
    fn state_wraps_text_under_the_named_key() {
        assert_eq!(text_state("message", "hello"), json!({"message": "hello"}));
    }
}
