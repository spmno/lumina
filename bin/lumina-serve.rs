//! HTTP server exposing Lumina over TypeSafe Jev's `/v1/systemone` wire protocol.
//!
//! Lumina's `predict` output is already schema-compatible with the Jev decision
//! API — `choice`/`score`/`noul` answers and an `{input_tokens, output_tokens}`
//! usage block — so a client written against Jev can point its `baseUrl` at this
//! server and keep working unchanged. All this binary adds is the HTTP surface
//! the library does not ship: a `POST /v1/systemone` route, an optional bearer
//! check, and a health probe.
//!
//! Configuration is entirely via environment variables so the same entry point
//! serves a laptop dev run and a systemd unit:
//!
//! | env var | meaning | default |
//! |---|---|---|
//! | `LAYA_HOST` | bind address | `0.0.0.0` |
//! | `LAYA_PORT` | bind port | `8000` |
//! | `LAYA_DEVICE` | device for every checkpoint | (auto) |
//! | `LAYA_PRELOAD` | build the checkpoints at startup, not lazily | `1` |
//! | `LAYA_MODELS` | `name=source` checkpoints to serve (hub repo, optional `:subfolder`, or local dir); empty = routing only | (none) |
//! | `LAYA_AUTO_TASK` | auto-route to the typed-decisions checkpoint | `0` |
//! | `LAYA_MAX_LOADED` | checkpoints kept resident at once | `2` |
//! | `LAYA_API_KEY` | require `Authorization: Bearer <it>` | (none) |
//! | `LAYA_LOG_LEVEL` | log level | `info` |
//! | `LAYA_MAX_CONCURRENT` | cap on requests past auth at once; excess gets 503 | `16` |
//! | `LAYA_MAX_TOKEN_BUDGET` | cap on per-request `max_len`/`head_max_len` | `8192` |
//!
//! Deviations from laya: `LAYA_THREADS` is not ported — candle reads its thread
//! count from `CANDLE_NUM_THREADS` directly. `/health` reports no per-checkpoint
//! revisions or CPU-fallback counters, because candle has no silent device
//! fallback to count. The body-size cap is enforced while reading the body, so a
//! lying `Content-Length` header is caught the same way, just without an early
//! rejection.

use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use tokio::sync::{Semaphore, mpsc, oneshot};

use lumina::pyjson::py_repr;
use lumina::{
    LuminaError, PredictOptions, Question, Router, RouterOptions, normalise_name,
    parse_model_sources, parse_questions,
};

// Guardrails for unauthenticated remote input. The state is tokenized once per
// question and collated into one tensor, so an unbounded body can OOM the
// worker; the single inference lane means one large request would also starve
// /health.
const MAX_QUESTIONS: usize = 64;
const MAX_STATE_CHARS: usize = 50_000;
const MAX_BODY_BYTES: usize = 2 * 1024 * 1024;
// HTTP-only amplification guard; the library keeps its head_max_len-aware budget.
const MAX_CHOICE_OPTIONS: usize = 100;
const MAX_SCORE_LEVELS: usize = 32;
const MAX_TOTAL_OPTIONS: usize = 512;

// Cap on requests past auth at once. Each one can buffer up to MAX_BODY_BYTES
// before inference, so without a bound many concurrent near-cap requests OOM
// the worker even though every request is individually valid.
const DEFAULT_MAX_CONCURRENT: usize = 16;
// Server-side ceiling on per-request max_len/head_max_len token budget overrides.
const DEFAULT_MAX_TOKEN_BUDGET: usize = 8192;

/// Everything the handlers need: the router, auth, the admission bound, and the
/// sender end of the single inference lane.
struct AppInner {
    router: Arc<Router>,
    api_key: Option<String>,
    sender: mpsc::Sender<Job>,
    admission: Arc<Semaphore>,
    max_token_budget: usize,
}

/// One admitted request, queued to the single inference lane.
struct Job {
    state: Value,
    questions: Vec<(String, Question)>,
    model: Option<String>,
    max_len: Option<usize>,
    head_max_len: Option<usize>,
    reply: oneshot::Sender<Reply>,
}

/// The single-lane inference result: the payload plus the pure inference time.
struct Reply {
    result: Result<Value, Fail>,
    infer_ms: f64,
}

/// Why inference failed, split the way the wire protocol answers it.
enum Fail {
    /// A caller-input error (laya's `ValueError` class): safe to report verbatim
    /// as a 422.
    Validation(String),
    /// A runtime failure: logged server-side, answered with a bare 500 so
    /// nothing about paths, weights or memory state leaks.
    Internal(String),
}

impl Fail {
    fn from_error(error: LuminaError) -> Self {
        if error.is_validation() {
            Self::Validation(error.to_string())
        } else {
            Self::Internal(error.to_string())
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let level = std::env::var("LAYA_LOG_LEVEL").unwrap_or_else(|_| "info".to_owned());
    let filter = tracing_subscriber::EnvFilter::try_new(&level)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    let port = match resolve_port() {
        Ok(port) => port,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::FAILURE;
        }
    };
    let router = match build_router() {
        Ok(router) => Arc::new(router),
        Err(error) => {
            eprintln!("lumina-serve: {error}");
            return ExitCode::FAILURE;
        }
    };
    let max_concurrent = resolve_max_concurrent().max(1);
    let (sender, receiver) = mpsc::channel::<Job>(max_concurrent);
    let app_state = Arc::new(AppInner {
        router: Arc::clone(&router),
        api_key: std::env::var("LAYA_API_KEY")
            .ok()
            .filter(|key| !key.is_empty()),
        sender,
        admission: Arc::new(Semaphore::new(max_concurrent)),
        max_token_budget: resolve_max_token_budget(),
    });
    tokio::spawn(inference_loop(receiver, router));
    let host = std::env::var("LAYA_HOST").unwrap_or_else(|_| "0.0.0.0".to_owned());
    let listener = match tokio::net::TcpListener::bind((host.as_str(), port)).await {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("lumina-serve: could not bind {host}:{port} ({error})");
            return ExitCode::FAILURE;
        }
    };
    let app = axum::Router::new()
        .route("/health", get(health))
        .route("/v1/systemone", post(systemone))
        .with_state(app_state);
    match axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("lumina-serve: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn shutdown_signal() {
    // Ctrl-C triggers axum's graceful shutdown; other kill signals end the
    // process without it, as with any unix service.
    let _ = tokio::signal::ctrl_c().await;
}

/// The single inference lane, porting laya's one-worker executor: one forward
/// pass at a time, because that is what a single CPU or GPU checkpoint wants.
async fn inference_loop(mut receiver: mpsc::Receiver<Job>, router: Arc<Router>) {
    while let Some(job) = receiver.recv().await {
        let Job {
            state,
            questions,
            model,
            max_len,
            head_max_len,
            reply,
        } = job;
        let router = Arc::clone(&router);
        let outcome = tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let options = PredictOptions {
                model: model.as_deref(),
                task: None,
                lang: None,
                lang_guess: None,
                max_len,
                head_max_len,
                min_confidence: None,
            };
            let result = router
                .predict(&state, &questions, &options)
                .map_err(Fail::from_error);
            Reply {
                result,
                infer_ms: started.elapsed().as_secs_f64() * 1000.0,
            }
        })
        .await;
        let reply_payload = outcome.unwrap_or_else(|join_error| Reply {
            result: Err(Fail::Internal(format!("inference lane: {join_error}"))),
            infer_ms: 0.0,
        });
        let _ = reply.send(reply_payload);
    }
}

// Constant-time comparison: a byte-wise `==` short-circuits on the first
// differing byte, leaking the bearer token through timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |diff, (x, y)| diff | (x ^ y)) == 0
}

fn err_response(status: StatusCode, detail: &str) -> Response {
    (status, Json(json!({ "detail": detail }))).into_response()
}

fn busy_response() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, HeaderValue::from_static("1"))],
        Json(json!({ "detail": "server busy, try again later" })),
    )
        .into_response()
}

fn ok_response(value: Value, infer_ms: f64) -> Response {
    let timing = format!("inference;dur={infer_ms:.2}");
    let elapsed = format!("{infer_ms:.2}");
    (
        StatusCode::OK,
        [
            (
                HeaderName::from_static("server-timing"),
                HeaderValue::from_str(&timing)
                    .unwrap_or(HeaderValue::from_static("inference;dur=0")),
            ),
            (
                HeaderName::from_static("x-inference-time-ms"),
                HeaderValue::from_str(&elapsed).unwrap_or(HeaderValue::from_static("0")),
            ),
        ],
        Json(value),
    )
        .into_response()
}

fn check_auth(app: &AppInner, headers: &HeaderMap) -> Option<Response> {
    let api_key = app.api_key.as_deref()?;
    let supplied: &[u8] = headers
        .get(header::AUTHORIZATION)
        .map_or(&[], HeaderValue::as_bytes);
    let expected = format!("Bearer {api_key}");
    if !constant_time_eq(supplied, expected.as_bytes()) {
        return Some(err_response(
            StatusCode::UNAUTHORIZED,
            "invalid or missing bearer token",
        ));
    }
    None
}

/// Read the request body, refusing to buffer more than `MAX_BODY_BYTES`, then
/// parse it as JSON.
async fn parse_json_request(request: Request) -> std::result::Result<Value, Box<Response>> {
    let Ok(bytes) = axum::body::to_bytes(request.into_body(), MAX_BODY_BYTES).await else {
        return Err(Box::new(err_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body too large",
        )));
    };
    serde_json::from_slice(&bytes).map_err(|_| {
        Box::new(err_response(
            StatusCode::BAD_REQUEST,
            "request body must be valid JSON",
        ))
    })
}

/// Reject absent or oversized inference requests before tokenization.
fn check_request_limits(state: &Value, questions: &Value) -> Option<Response> {
    // A body with no `state` key, or `"state": null`, was answered as a decision
    // about the literal text "null"; nothing downstream can tell that apart
    // from a real string, so the check happens before serialization.
    let Some(question_map) = questions.as_object() else {
        return Some(err_response(
            StatusCode::BAD_REQUEST,
            "'questions' must be an object",
        ));
    };
    if question_map.len() > MAX_QUESTIONS {
        return Some(err_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            &format!(
                "too many questions ({} > {})",
                question_map.len(),
                MAX_QUESTIONS
            ),
        ));
    }
    let mut total_options = 0;
    for (qid, question) in question_map {
        let Some(question_object) = question.as_object() else {
            continue;
        };
        let question_type = question_object.get("type").and_then(Value::as_str);
        let criteria = question_object.get("criteria");
        if question_type == Some("choice") {
            if let Some(count) = option_count(criteria) {
                total_options += count;
                if count > MAX_CHOICE_OPTIONS {
                    return Some(err_response(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        &format!(
                            "too many choice options for {} ({count} > {MAX_CHOICE_OPTIONS})",
                            py_repr(&json!(qid))
                        ),
                    ));
                }
            }
        } else if question_type == Some("score") {
            if let Some(count) = criteria.and_then(Value::as_array).map(Vec::len) {
                total_options += count;
                if count > MAX_SCORE_LEVELS {
                    return Some(err_response(
                        StatusCode::PAYLOAD_TOO_LARGE,
                        &format!(
                            "too many score levels for {} ({count} > {MAX_SCORE_LEVELS})",
                            py_repr(&json!(qid))
                        ),
                    ));
                }
            }
        }
    }
    if total_options > MAX_TOTAL_OPTIONS {
        return Some(err_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            &format!(
                "too many answer options across questions ({total_options} > {MAX_TOTAL_OPTIONS})"
            ),
        ));
    }
    let state_len = state_char_len(state);
    if state_len > MAX_STATE_CHARS {
        return Some(err_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            &format!("state too large ({state_len} > {MAX_STATE_CHARS} chars)"),
        ));
    }
    None
}

fn option_count(criteria: Option<&Value>) -> Option<usize> {
    match criteria {
        Some(Value::Object(map)) => Some(map.len()),
        Some(Value::Array(items)) => Some(items.len()),
        _ => None,
    }
}

/// The state's length the way laya measures it: characters of a string state,
/// the string form of anything else.
fn state_char_len(state: &Value) -> usize {
    match state {
        Value::String(text) => text.chars().count(),
        other => other.to_string().chars().count(),
    }
}

/// Map a client's `model` field onto a Lumina checkpoint, or `None` to auto-route.
fn resolve_model(model: Option<&Value>) -> Option<String> {
    let model = model?.as_str()?;
    // normalise_name rejects anything that is not a known checkpoint or alias; a
    // Jev client's `model` field (e.g. "jev-1") is expected to miss, which means
    // "no explicit checkpoint" and lets the router auto-select.
    normalise_name(model).ok()
}

/// Validate an optional integer budget parameter (max_len / head_max_len).
fn validate_budget(
    value: Option<&Value>,
    key: &str,
    max_cap: usize,
) -> std::result::Result<Option<usize>, Box<Response>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(value) => {
            let Some(number) = value.as_i64() else {
                return Err(Box::new(err_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    &format!("{key} must be an integer"),
                )));
            };
            if number <= 0 {
                return Err(Box::new(err_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    &format!("{key} must be a positive integer"),
                )));
            }
            let number = usize::try_from(number).unwrap_or(usize::MAX);
            if number > max_cap {
                return Err(Box::new(err_response(
                    StatusCode::UNPROCESSABLE_ENTITY,
                    &format!("{key} exceeds server limit ({number} > {max_cap})"),
                )));
            }
            Ok(Some(number))
        }
    }
}

#[allow(clippy::unused_async)] // an axum handler must be async
async fn health(State(app): State<Arc<AppInner>>) -> Json<Value> {
    let devices = app.router.loaded_devices();
    let actual = devices.first().map(|(_, device)| device.clone());
    let checkpoint_devices: serde_json::Map<String, Value> = devices
        .into_iter()
        .map(|(name, device)| (name, Value::String(device)))
        .collect();
    Json(json!({
        "status": "ok",
        "loaded": app.router.loaded(),
        "device": actual.clone().unwrap_or_else(resolve_device),
        "device_is_preference": actual.is_none(),
        "checkpoint_devices": checkpoint_devices,
    }))
}

async fn systemone(
    State(app): State<Arc<AppInner>>,
    headers: HeaderMap,
    request: Request,
) -> Response {
    if let Some(response) = check_auth(&app, &headers) {
        return response;
    }
    // Non-blocking: excess load is refused rather than queued, so the buffered
    // bodies stay within the admission bound. Retry-After tells well-behaved
    // clients when a slot is likely free: admission turns over at inference
    // speed, so one second is the honest hint.
    let Ok(_permit) = app.admission.clone().try_acquire_owned() else {
        return busy_response();
    };
    let parsed = match parse_json_request(request).await {
        Ok(parsed) => parsed,
        Err(response) => return *response,
    };
    let Some(object) = parsed.as_object() else {
        return err_response(
            StatusCode::BAD_REQUEST,
            "request body must be an object with a 'questions' field",
        );
    };
    let Some(questions_value) = object.get("questions") else {
        return err_response(
            StatusCode::BAD_REQUEST,
            "request body must be an object with a 'questions' field",
        );
    };
    let Some(state) = object.get("state").filter(|state| !state.is_null()) else {
        return err_response(StatusCode::BAD_REQUEST, "'state' is required");
    };
    if let Some(response) = check_request_limits(state, questions_value) {
        return response;
    }
    let model = resolve_model(object.get("model"));
    let max_len = match validate_budget(object.get("max_len"), "max_len", app.max_token_budget) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let head_max_len = match validate_budget(
        object.get("head_max_len"),
        "head_max_len",
        app.max_token_budget,
    ) {
        Ok(value) => value,
        Err(response) => return *response,
    };
    let questions = match parse_questions(questions_value) {
        Ok(questions) => questions,
        Err(error) => {
            // Question validation errors name the question and what to fix: safe for clients.
            return err_response(StatusCode::UNPROCESSABLE_ENTITY, &error.to_string());
        }
    };
    let (reply_sender, reply_receiver) = oneshot::channel();
    let job = Job {
        state: state.clone(),
        questions,
        model,
        max_len,
        head_max_len,
        reply: reply_sender,
    };
    if app.sender.send(job).await.is_err() {
        return err_response(StatusCode::INTERNAL_SERVER_ERROR, "inference failed");
    }
    let Ok(Reply { result, infer_ms }) = reply_receiver.await else {
        return err_response(StatusCode::INTERNAL_SERVER_ERROR, "inference failed");
    };
    match result {
        Ok(value) => ok_response(value, infer_ms),
        Err(Fail::Validation(detail)) => err_response(StatusCode::UNPROCESSABLE_ENTITY, &detail),
        Err(Fail::Internal(detail)) => {
            // `model` is caller-supplied text; the line breaks are replaced so a
            // crafted value cannot forge log entries.
            let model = sanitized_model(object);
            tracing::error!("inference failed for model={model}: {detail}");
            err_response(StatusCode::INTERNAL_SERVER_ERROR, "inference failed")
        }
    }
}

fn sanitized_model(object: &serde_json::Map<String, Value>) -> String {
    let text = object
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("None");
    text.replace('\n', "\\n").replace('\r', "\\r")
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name).map_or(default, |value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

fn env_positive_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
}

/// Bound on requests past auth at once, from LAYA_MAX_CONCURRENT; a typo in a
/// deployment file should not stop the server from starting.
fn resolve_max_concurrent() -> usize {
    env_positive_usize("LAYA_MAX_CONCURRENT").unwrap_or(DEFAULT_MAX_CONCURRENT)
}

/// Server-side cap on per-request max_len from LAYA_MAX_TOKEN_BUDGET.
fn resolve_max_token_budget() -> usize {
    if let Ok(raw) = std::env::var("LAYA_MAX_TOKEN_BUDGET") {
        if !raw.trim().is_empty() {
            match raw.trim().parse::<usize>() {
                Ok(value) if value > 0 => return value,
                _ => tracing::warn!(
                    "invalid LAYA_MAX_TOKEN_BUDGET {raw:?}; falling back to {DEFAULT_MAX_TOKEN_BUDGET}"
                ),
            }
        }
    }
    DEFAULT_MAX_TOKEN_BUDGET
}

/// Resident-checkpoint cap from LAYA_MAX_LOADED; `None` leaves it to the Router,
/// so this default cannot drift from the library's own.
fn resolve_max_loaded() -> Option<usize> {
    env_positive_usize("LAYA_MAX_LOADED")
}

/// Port from LAYA_PORT, validated; exits with a message instead of a traceback.
fn resolve_port() -> std::result::Result<u16, String> {
    let raw = std::env::var("LAYA_PORT").unwrap_or_else(|_| "8000".to_owned());
    match raw.trim().parse::<u16>() {
        Ok(port) if port > 0 => Ok(port),
        _ => Err(format!(
            "invalid LAYA_PORT {raw:?}: must be an integer 1-65535"
        )),
    }
}

/// The configured device preference (`LAYA_DEVICE`), trimmed; `None` when unset.
fn env_device() -> Option<String> {
    std::env::var("LAYA_DEVICE")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Where a resident checkpoint computes, or the preference when nothing is
/// resident: LAYA_DEVICE if set, else CUDA-when-available, else CPU.
fn resolve_device() -> String {
    if let Some(preference) = env_device() {
        return preference;
    }
    if candle_core::utils::cuda_is_available() {
        "cuda".to_owned()
    } else {
        "cpu".to_owned()
    }
}

/// Build a Router from the environment: `LAYA_MODELS` names the checkpoints to
/// serve and where their weights live, and `LAYA_PRELOAD` (the default) builds
/// them at startup.
fn build_router() -> lumina::Result<Router> {
    let device = env_device();
    let models = parse_model_sources(&std::env::var("LAYA_MODELS").unwrap_or_default())?;
    let registry: Vec<(&str, &str, Option<&str>)> = models
        .iter()
        .map(|(name, repo, subfolder)| (name.as_str(), repo.as_str(), subfolder.as_deref()))
        .collect();
    let options = RouterOptions {
        device: device.as_deref(),
        auto_task_detection: env_bool("LAYA_AUTO_TASK", false),
        max_loaded: resolve_max_loaded(),
        models: &registry,
        ..RouterOptions::default()
    };
    let router = Router::new(&options)?;
    if env_bool("LAYA_PRELOAD", true) && !registry.is_empty() {
        // Router::preload raises the cap to hold whatever it builds, so a lower
        // LAYA_MAX_LOADED never evicts a preloaded checkpoint.
        router.preload(None)?;
    }
    Ok(router)
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
    fn resolve_model_pins_models_and_falls_back_to_auto() {
        assert_eq!(
            resolve_model(Some(&json!("  English "))).as_deref(),
            Some("english")
        );
        assert_eq!(
            resolve_model(Some(&json!("ml"))).as_deref(),
            Some("multilingual")
        );
        assert_eq!(resolve_model(None), None);
        assert_eq!(resolve_model(Some(&json!(""))), None);
        assert_eq!(resolve_model(Some(&json!(1))), None);
        assert_eq!(resolve_model(Some(&json!("jev-1"))), None);
        assert_eq!(resolve_model(Some(&json!("acme/models"))), None);
    }

    #[test]
    fn validate_budget_accepts_absent_null_and_in_cap() {
        assert_eq!(validate_budget(None, "max_len", 8192).unwrap(), None);
        assert_eq!(
            validate_budget(Some(&Value::Null), "max_len", 8192).unwrap(),
            None
        );
        assert_eq!(
            validate_budget(Some(&json!(512)), "max_len", 8192).unwrap(),
            Some(512)
        );
        assert_eq!(
            validate_budget(Some(&json!(8192)), "max_len", 8192).unwrap(),
            Some(8192)
        );
    }

    #[test]
    fn validate_budget_answers_422_for_bad_values() {
        for value in [
            json!("512"),
            json!(1.5),
            json!(true),
            json!(0),
            json!(-8),
            json!(8193),
        ] {
            let response = validate_budget(Some(&value), "max_len", 8192).unwrap_err();
            assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        }
    }

    fn question_map(count: usize, question: Value) -> Value {
        let mut questions = serde_json::Map::new();
        for index in 0..count {
            questions.insert(format!("q{index}"), question.clone());
        }
        Value::Object(questions)
    }

    fn choice_criteria(count: usize) -> Value {
        let mut criteria = serde_json::Map::new();
        for index in 0..count {
            criteria.insert(format!("o{index}"), json!("option"));
        }
        Value::Object(criteria)
    }

    #[test]
    fn request_limits_admit_typical_and_boundary_requests() {
        let state = json!({"request": "I was charged twice, please refund"});
        let questions = json!({
            "department": {
                "type": "choice",
                "criteria": {"billing": "billing work", "support": "customer support"}
            },
            "risk": {"type": "score", "criteria": [1, 2, 3, 4, 5]}
        });
        assert!(check_request_limits(&state, &questions).is_none());
        let at_question_cap = question_map(64, json!({"type": "noul"}));
        assert!(check_request_limits(&json!({"request": "hi"}), &at_question_cap).is_none());
        let choice_at_cap = json!({"type": "choice", "criteria": choice_criteria(100)});
        assert!(
            check_request_limits(&json!({"request": "hi"}), &json!({"q": choice_at_cap})).is_none()
        );
        let levels: Vec<Value> = (0..32).map(|level| json!(level)).collect();
        let score_at_cap = json!({"type": "score", "criteria": levels});
        assert!(
            check_request_limits(&json!({"request": "hi"}), &json!({"q": score_at_cap})).is_none()
        );
        let state_at_cap = Value::String("x".repeat(50_000));
        assert!(check_request_limits(&state_at_cap, &json!({})).is_none());
    }

    #[test]
    fn request_limits_reject_non_object_questions() {
        let response = check_request_limits(&json!({"request": "hi"}), &json!([]));
        assert_eq!(response.unwrap().status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn request_limits_reject_too_many_questions() {
        let questions = question_map(65, json!({"type": "noul"}));
        let response = check_request_limits(&json!({"request": "hi"}), &questions);
        assert_eq!(response.unwrap().status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn request_limits_reject_too_many_choice_options() {
        let question = json!({"type": "choice", "criteria": choice_criteria(101)});
        let response = check_request_limits(&json!({"request": "hi"}), &json!({"q": question}));
        assert_eq!(response.unwrap().status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn request_limits_reject_too_many_score_levels() {
        let levels: Vec<Value> = (0..33).map(|level| json!(level)).collect();
        let question = json!({"type": "score", "criteria": levels});
        let response = check_request_limits(&json!({"request": "hi"}), &json!({"q": question}));
        assert_eq!(response.unwrap().status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn request_limits_reject_too_many_total_options() {
        let question = json!({"type": "choice", "criteria": choice_criteria(90)});
        let questions = question_map(6, question);
        let response = check_request_limits(&json!({"request": "hi"}), &questions);
        assert_eq!(response.unwrap().status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn request_limits_reject_oversized_state() {
        let state = Value::String("x".repeat(50_001));
        let response = check_request_limits(&state, &json!({}));
        assert_eq!(response.unwrap().status(), StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[test]
    fn request_limits_ignore_non_container_criteria() {
        let question = json!({"type": "choice", "criteria": "not-a-mapping"});
        let response = check_request_limits(&json!({"request": "hi"}), &json!({"q": question}));
        assert!(response.is_none());
    }

    #[test]
    fn state_char_len_counts_characters_not_bytes() {
        assert_eq!(state_char_len(&json!("héllo")), 5);
        assert_eq!(state_char_len(&json!({"a": 1})), 7);
    }

    #[test]
    fn constant_time_eq_compares_content_and_length() {
        assert!(constant_time_eq(b"bearer secret", b"bearer secret"));
        assert!(!constant_time_eq(b"bearer secret", b"bearer secreT"));
        assert!(!constant_time_eq(b"short", b"longer value"));
        assert!(constant_time_eq(b"", b""));
    }
}
