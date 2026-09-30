//! Pure model-selection logic, ported from the decision half of `laya/router.py`.
//!
//! The three checkpoint names and the signal that picks between them. The
//! registry binding names to checkpoint sources is configuration, not code:
//! `RouterOptions::models` for library callers, and `LAYA_MODELS` (entries like
//! `english=org/repo` or `english=org/repo:subfolder`, or a local directory)
//! for the binaries. For example, laya's published bundle configures:
//!
//! english          convaiinnovations/laya (bundle root)    421M, 512-token window
//! multilingual     convaiinnovations/laya/multilingual     322M, 1024 tokens, 100+ langs
//! typed-decisions  convaiinnovations/laya/typed-decisions  fine-tuned on 4 workflows
//!
//! A model with no source bound still routes — its decision carries a null
//! `repo` — but cannot be loaded for prediction.
//! Routing is worth it because the English checkpoint does not degrade gently off
//! English, it collapses (0.100 on Hindi at 20 options, against 0.050 for random
//! guessing) while reporting high confidence — script detection is therefore the
//! primary signal. `typed-decisions` is never selected automatically unless the
//! router opts in with `auto_task_detection`.
//!
//! Decision precedence (laya's `_route`): explicit `model` > explicit `task` >
//! workflow match (opt-in) > explicit `lang` > `lang_guess` (per-call, then the
//! router's) > script/language detection > default. This module holds no I/O; the
//! lifecycle half lives in [`crate::router`].

use serde::Serialize;
use serde_json::Value;

use crate::error::{LuminaError, Result};
use crate::lang::analyse;
use crate::pyjson::py_repr;

/// Registry names, in registry order.
const MODEL_NAMES: [&str; 3] = ["english", "multilingual", "typed-decisions"];

/// Alias keys, in the order Python's `sorted()` lists them in the error text.
const ALIAS_KEYS: [&str; 7] = [
    "decisions",
    "default",
    "en",
    "ml",
    "multi",
    "typed",
    "typed_decisions",
];

/// Question-id signatures of the four typed-decisions workflows, matched exactly.
const TYPED_DECISION_WORKFLOWS: [(&str, [&str; 5]); 4] = [
    (
        "agent_trace_observability",
        ["action", "needs_review", "outcome", "risk", "urgency"],
    ),
    (
        "customer_service",
        ["action", "category", "churn_risk", "needs_human", "urgency"],
    ),
    (
        "invoice_processing",
        [
            "discrepancy_severity",
            "disposition",
            "duplicate",
            "matches_order",
            "urgency",
        ],
    ),
    (
        "security_incidents",
        [
            "credential_compromise",
            "disposition",
            "severity",
            "true_positive",
            "urgency",
        ],
    ),
];

/// Subtags that mean "the English checkpoint can read this".
const ENGLISH_SUBTAGS: [&str; 3] = ["en", "eng", "english"];

/// `$LANG` values that name no language, so they answer nothing about the state:
/// `C`, `POSIX` and `C.UTF-8` are what minimal images ship, and the ISO 639-2
/// special codes say the same thing (`und` undetermined, `zxx` no linguistic
/// content, `mul` multiple languages). They abstain rather than forcing the
/// multilingual checkpoint on English text.
const LANGUAGE_AGNOSTIC_CODES: [&str; 5] = ["c", "posix", "und", "zxx", "mul"];

/// One entry in the router registry: the repo to load and the subfolder it lives in.
#[derive(Clone, Debug)]
pub(crate) struct ModelSpec {
    /// Hugging Face repository id (or local directory path).
    pub(crate) repo: String,
    /// Subfolder of the repo holding this checkpoint, `None` for the root.
    pub(crate) subfolder: Option<String>,
}

impl ModelSpec {
    /// Human-readable id: `repo` or `repo/subfolder`.
    fn repo_str(&self) -> String {
        self.subfolder
            .as_ref()
            .map_or_else(|| self.repo.clone(), |sub| format!("{}/{}", self.repo, sub))
    }
}

/// Canonical registry key for `name`: trimmed, lowercased, alias-resolved.
///
/// # Errors
/// [`LuminaError::Invalid`] with laya's text when `name` is not a model or alias.
pub fn normalise_name(name: &str) -> Result<String> {
    let key = name.trim().to_ascii_lowercase();
    let key = match key.as_str() {
        "en" | "default" => "english",
        "multi" | "ml" => "multilingual",
        "typed" | "typed_decisions" | "decisions" => "typed-decisions",
        other => other,
    };
    if MODEL_NAMES.contains(&key) {
        Ok(key.to_owned())
    } else {
        Err(LuminaError::Invalid(format!(
            "unknown model {}; choose one of {} (or an alias: {})",
            repr(name),
            py_str_list(&MODEL_NAMES),
            py_str_list(&ALIAS_KEYS)
        )))
    }
}

/// Name of the typed-decisions workflow whose question ids these are, else `None`.
/// Exact id-set match, so an unrelated schema that happens to contain `urgency` is
/// never captured.
fn match_typed_decisions_workflow(question_ids: &[&str]) -> Option<&'static str> {
    TYPED_DECISION_WORKFLOWS
        .iter()
        .find(|(_, sig)| {
            question_ids.len() == sig.len() && sig.iter().all(|id| question_ids.contains(id))
        })
        .map(|(name, _)| *name)
}

/// True/False for a language code, or `None` when the code identifies nothing.
///
/// Accepts the forms a caller is likely to have to hand: `"en"`, `"EN"`, `"en-US"`,
/// the POSIX `"en_US"` (which `$LANG` holds), and `"en_US.UTF-8"`. A code that names
/// no language returns `None`, which is what lets `LANG=C` fall through to detection
/// instead of pinning every request to one checkpoint.
fn english_from_code(value: &str) -> Option<bool> {
    let code = value.trim().to_ascii_lowercase();
    if code.is_empty() {
        return None;
    }
    let normalised = code.split('.').next().unwrap_or_default().replace('_', "-");
    let primary = normalised.split('-').next().unwrap_or_default();
    if primary.is_empty() || LANGUAGE_AGNOSTIC_CODES.contains(&primary) {
        return None;
    }
    Some(ENGLISH_SUBTAGS.contains(&primary))
}

/// The routing outcome: which model, why, and what was detected.
///
/// Serializes with laya's key order (`model`, `repo`, `reason`, `detection`,
/// `workflow`) so a `routing` block in a prediction payload matches laya's.
#[derive(Clone, Debug, Serialize)]
pub struct RouteDecision {
    /// Registry name of the selected model.
    pub model: String,
    /// `repo` or `repo/subfolder` of the selected checkpoint, or null when no
    /// source is bound to that model (routing only, no prediction possible).
    pub repo: Option<String>,
    /// Human-readable explanation of the decision.
    pub reason: String,
    /// Language analysis behind the decision, or null for explicit routing.
    pub detection: Value,
    /// Typed-decisions workflow name when question ids matched one.
    pub workflow: Option<String>,
}

/// Routing overrides for one request.
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteOptions<'a> {
    /// Pin the model by registry name or alias.
    pub model: Option<&'a str>,
    /// Pin the model by task name; `"typed_decisions"` maps to the typed-decisions
    /// checkpoint.
    pub task: Option<&'a str>,
    /// ISO-style language code: English-like codes pin `english`, others pin
    /// `multilingual`.
    pub lang: Option<&'a str>,
    /// Caller's language guess, consulted after `lang` and before detection.
    pub lang_guess: Option<&'a str>,
}

/// Router state the pure decision reads (laya's `Router` attributes, borrowed).
pub(crate) struct RoutingCtx<'a> {
    /// Registry: model name to checkpoint spec.
    pub(crate) models: &'a [(String, ModelSpec)],
    /// Model for states detection cannot decide.
    pub(crate) default: &'a str,
    /// Whether a workflow match may select `typed-decisions`.
    pub(crate) auto_task_detection: bool,
    /// Router-level language-guess code, consulted after the per-call one.
    pub(crate) lang_guess: Option<&'a str>,
}

/// Decide which checkpoint to use, without loading or running anything (laya's
/// `Router._route`).
pub(crate) fn decide(
    ctx: &RoutingCtx<'_>,
    state: &Value,
    question_ids: &[&str],
    opts: &RouteOptions<'_>,
) -> Result<RouteDecision> {
    if let Some(model) = opts.model {
        let key = normalise_name(model)?;
        return Ok(decision(
            ctx,
            key,
            format!("explicit model={}", repr(model)),
            Value::Null,
            None,
        ));
    }
    if let Some(task) = opts.task {
        let effective = if task.to_ascii_lowercase().replace('-', "_") == "typed_decisions" {
            "typed-decisions"
        } else {
            task
        };
        let key = normalise_name(effective)?;
        return Ok(decision(
            ctx,
            key,
            format!("explicit task={}", repr(task)),
            Value::Null,
            None,
        ));
    }
    let workflow = match_typed_decisions_workflow(question_ids);
    if let Some(workflow) = workflow.filter(|_| ctx.auto_task_detection) {
        return Ok(decision(
            ctx,
            "typed-decisions".to_owned(),
            format!(
                "question ids match the {} typed-decisions workflow",
                repr(workflow)
            ),
            Value::Null,
            Some(workflow),
        ));
    }
    if let Some(lang) = opts.lang {
        // An explicit `lang` is decisive only when the code names a language; blank
        // or language-agnostic codes fall through to lang_guess/detection.
        if let Some(resolved) = english_from_code(lang) {
            let key = if resolved { "english" } else { "multilingual" };
            return Ok(decision(
                ctx,
                key.to_owned(),
                format!("explicit lang={}", repr(lang)),
                Value::Null,
                workflow,
            ));
        }
    }
    // Caller-supplied hint, per-call first then the one installed on the router.
    for (source, hint) in [
        ("lang_guess", opts.lang_guess),
        ("Router(lang_guess=...)", ctx.lang_guess),
    ] {
        if let Some(resolved) = hint.and_then(english_from_code) {
            let key = if resolved { "english" } else { "multilingual" };
            return Ok(decision(
                ctx,
                key.to_owned(),
                format!(
                    "{source}: the caller identified this as {} text",
                    if resolved { "English" } else { "non-English" }
                ),
                Value::Null,
                workflow,
            ));
        }
    }
    route_from_detection(ctx, state, workflow)
}

/// Detection branch of the decision tree (laya's `_route` tail).
fn route_from_detection(
    ctx: &RoutingCtx<'_>,
    state: &Value,
    workflow: Option<&str>,
) -> Result<RouteDecision> {
    let det = analyse(state);
    let (key, reason): (String, String) = if det.script == "unknown" {
        (
            ctx.default.to_owned(),
            format!(
                "no letters detected in state; using default ({})",
                ctx.default
            ),
        )
    } else if det.script != "latin" {
        (
            "multilingual".to_owned(),
            format!(
                "non-Latin script ({}, {:.0}% of letters); the English checkpoint cannot read it",
                det.script,
                100.0 * det.non_latin_fraction
            ),
        )
    } else if !det.is_english {
        (
            "multilingual".to_owned(),
            if let Some(segment) = &det.mixed_segment {
                // `mixed_segment` is only ever set alongside a named language
                let language = det.language.as_deref().unwrap_or_default();
                format!(
                    "Latin script, mostly English, but a line or field reads as {} ({}); \
                     the English checkpoint cannot read it",
                    repr(language),
                    repr(&segment.chars().take(60).collect::<String>())
                )
            } else if det.language.is_some() {
                format!(
                    "Latin script but language looks like {}, not English",
                    repr(det.language.as_deref().unwrap_or_default())
                )
            } else {
                format!(
                    "Latin script, language not identified but {:.0}% non-English letters; \
                     not safe for the English checkpoint",
                    100.0 * det.diacritic_rate
                )
            },
        )
    } else if det.language_undecided {
        (
            ctx.default.to_owned(),
            format!(
                "Latin script, language not identified and no non-English letters; \
                 using default ({})",
                ctx.default
            ),
        )
    } else {
        ("english".to_owned(), "English Latin text".to_owned())
    };
    Ok(decision(
        ctx,
        key,
        reason,
        serde_json::to_value(&det)?,
        workflow,
    ))
}

/// Build a decision for a registry key (the key is validated by construction);
/// `repo` is null when no source is bound to the key.
fn decision(
    ctx: &RoutingCtx<'_>,
    key: String,
    reason: String,
    detection: Value,
    workflow: Option<&str>,
) -> RouteDecision {
    let repo = ctx
        .models
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, spec)| spec.repo_str());
    RouteDecision {
        model: key,
        repo,
        reason,
        detection,
        workflow: workflow.map(str::to_owned),
    }
}

/// Python's `%r` of a string.
fn repr(s: &str) -> String {
    py_repr(&Value::String(s.to_owned()))
}

/// Python's `repr` of a list of strings (error text: `['english', …]`).
fn py_str_list(items: &[&str]) -> String {
    format!(
        "[{}]",
        items.iter().map(|s| repr(s)).collect::<Vec<_>>().join(", ")
    )
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic
    )]
    use serde_json::json;

    use super::*;

    fn test_models() -> Vec<(String, ModelSpec)> {
        vec![
            (
                "english".to_owned(),
                ModelSpec {
                    repo: "acme/models".to_owned(),
                    subfolder: None,
                },
            ),
            (
                "multilingual".to_owned(),
                ModelSpec {
                    repo: "acme/models".to_owned(),
                    subfolder: Some("multilingual".to_owned()),
                },
            ),
            (
                "typed-decisions".to_owned(),
                ModelSpec {
                    repo: "acme/typed".to_owned(),
                    subfolder: None,
                },
            ),
        ]
    }

    fn ctx_of<'a>(
        models: &'a [(String, ModelSpec)],
        default: &'a str,
        auto_task_detection: bool,
        lang_guess: Option<&'a str>,
    ) -> RoutingCtx<'a> {
        RoutingCtx {
            models,
            default,
            auto_task_detection,
            lang_guess,
        }
    }

    fn reason_for(models: &[(String, ModelSpec)], default: &str, state: &Value) -> String {
        let ctx = ctx_of(models, default, false, None);
        decide(&ctx, state, &[], &RouteOptions::default())
            .unwrap()
            .reason
    }

    #[test]
    fn normalises_names_and_aliases() {
        assert_eq!(normalise_name(" english ").unwrap(), "english");
        assert_eq!(normalise_name("EN").unwrap(), "english");
        assert_eq!(normalise_name("ml").unwrap(), "multilingual");
        assert_eq!(normalise_name("decisions").unwrap(), "typed-decisions");
        assert_eq!(
            normalise_name("typed_decisions").unwrap(),
            "typed-decisions"
        );
    }

    #[test]
    fn unknown_model_error_matches_laya() {
        let err = normalise_name("bogus").unwrap_err().to_string();
        assert_eq!(
            err,
            "unknown model 'bogus'; choose one of ['english', 'multilingual', 'typed-decisions'] \
             (or an alias: ['decisions', 'default', 'en', 'ml', 'multi', 'typed', \
             'typed_decisions'])"
        );
    }

    #[test]
    fn english_from_code_forms() {
        for code in [
            "en",
            "EN",
            "eng",
            "english",
            "en-US",
            "en_US",
            "en_US.UTF-8",
        ] {
            assert_eq!(english_from_code(code), Some(true), "{code}");
        }
        for code in ["de", "DE-latn", "pt-BR"] {
            assert_eq!(english_from_code(code), Some(false), "{code}");
        }
        for code in [
            "", "  ", "c", "C", "POSIX", "c.UTF-8", "und", "zxx", "mul", ".en", "-en",
        ] {
            assert_eq!(english_from_code(code), None, "{code}");
        }
    }

    #[test]
    fn workflow_needs_exact_id_set() {
        let ids = ["action", "category", "churn_risk", "needs_human", "urgency"];
        assert_eq!(
            match_typed_decisions_workflow(&ids),
            Some("customer_service")
        );
        let extra: Vec<&str> = ids.into_iter().chain(["tone"]).collect();
        assert_eq!(match_typed_decisions_workflow(&extra), None);
        let short: Vec<&str> = ids.into_iter().take(4).collect();
        assert_eq!(match_typed_decisions_workflow(&short), None);
        assert_eq!(match_typed_decisions_workflow(&[]), None);
    }

    #[test]
    fn explicit_model_pins_and_wins() {
        let models = test_models();
        let ctx = ctx_of(&models, "english", false, None);
        let d = decide(
            &ctx,
            &json!("मुझसे रिफंड चाहिए"),
            &[],
            &RouteOptions {
                model: Some("ml"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.model, "multilingual");
        assert_eq!(d.repo.as_deref(), Some("acme/models/multilingual"));
        assert_eq!(d.reason, "explicit model='ml'");
        assert!(d.detection.is_null());
        assert_eq!(d.workflow, None);
    }

    #[test]
    fn explicit_task_routes_and_normalises() {
        let models = test_models();
        let ctx = ctx_of(&models, "english", false, None);
        let d = decide(
            &ctx,
            &json!({}),
            &[],
            &RouteOptions {
                task: Some("typed_decisions"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.model, "typed-decisions");
        assert_eq!(d.reason, "explicit task='typed_decisions'");
        let d = decide(
            &ctx,
            &json!({}),
            &[],
            &RouteOptions {
                task: Some("Typed-Decisions"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.model, "typed-decisions");
        assert_eq!(d.reason, "explicit task='Typed-Decisions'");
        let d = decide(
            &ctx,
            &json!({}),
            &[],
            &RouteOptions {
                task: Some("ML"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.model, "multilingual");
        // an unknown task is an unknown model, in laya's words
        let err = decide(
            &ctx,
            &json!({}),
            &[],
            &RouteOptions {
                task: Some("billing"),
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
        assert!(err.starts_with("unknown model 'billing';"));
    }

    #[test]
    fn workflow_match_requires_opt_in() {
        let models = test_models();
        let ids = ["action", "category", "churn_risk", "needs_human", "urgency"];
        let opt_in = ctx_of(&models, "english", true, None);
        let d = decide(
            &opt_in,
            &json!("I was charged twice"),
            &ids,
            &RouteOptions::default(),
        )
        .unwrap();
        assert_eq!(d.model, "typed-decisions");
        assert_eq!(
            d.reason,
            "question ids match the 'customer_service' typed-decisions workflow"
        );
        assert_eq!(d.workflow.as_deref(), Some("customer_service"));
        // without opt-in the workflow is carried but detection routes
        let ctx = ctx_of(&models, "english", false, None);
        let d = decide(
            &ctx,
            &json!("I was charged twice"),
            &ids,
            &RouteOptions::default(),
        )
        .unwrap();
        assert_eq!(d.model, "english");
        assert_eq!(d.reason, "English Latin text");
        assert_eq!(d.workflow.as_deref(), Some("customer_service"));
    }

    #[test]
    fn explicit_lang_is_decisive() {
        let models = test_models();
        let ctx = ctx_of(&models, "english", false, None);
        // an English code beats Hindi detection
        let d = decide(
            &ctx,
            &json!("मुझसे रिफंड चाहिए"),
            &[],
            &RouteOptions {
                lang: Some("en"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.model, "english");
        assert_eq!(d.reason, "explicit lang='en'");
        let d = decide(
            &ctx,
            &json!("I was charged twice"),
            &[],
            &RouteOptions {
                lang: Some("de"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.model, "multilingual");
        assert_eq!(d.reason, "explicit lang='de'");
        // language-agnostic codes abstain and fall through to detection
        let d = decide(
            &ctx,
            &json!("I was charged twice"),
            &[],
            &RouteOptions {
                lang: Some("C.UTF-8"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.model, "english");
        assert_eq!(d.reason, "English Latin text");
    }

    #[test]
    fn lang_guess_per_call_then_router() {
        let models = test_models();
        let ctx = ctx_of(&models, "english", false, Some("en"));
        // the per-call hint wins over the router's
        let d = decide(
            &ctx,
            &json!("x"),
            &[],
            &RouteOptions {
                lang_guess: Some("de"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(d.model, "multilingual");
        assert_eq!(
            d.reason,
            "lang_guess: the caller identified this as non-English text"
        );
        // router-level hint used when the per-call one abstains
        let d = decide(&ctx, &json!("x"), &[], &RouteOptions::default()).unwrap();
        assert_eq!(d.model, "english");
        assert_eq!(
            d.reason,
            "Router(lang_guess=...): the caller identified this as English text"
        );
    }

    #[test]
    fn detection_reasons_match_laya() {
        let models = test_models();
        assert_eq!(
            reason_for(&models, "english", &json!("12345 !!!")),
            "no letters detected in state; using default (english)"
        );
        assert_eq!(
            reason_for(&models, "multilingual", &json!("12345 !!!")),
            "no letters detected in state; using default (multilingual)"
        );
        assert_eq!(
            reason_for(&models, "english", &json!("Refund please")),
            "Latin script, language not identified and no non-English letters; using default (english)"
        );
        assert_eq!(
            reason_for(&models, "english", &json!("I was charged twice")),
            "English Latin text"
        );
        assert_eq!(
            reason_for(
                &models,
                "english",
                &json!("El pedido llegó roto y quiero que lo cambien")
            ),
            "Latin script but language looks like 'es', not English"
        );
        assert_eq!(
            reason_for(
                &models,
                "english",
                &json!("La aplicación se cierra cada vez que abro la configuración.")
            ),
            "Latin script, language not identified but 3% non-English letters; not safe for \
             the English checkpoint"
        );
        let mixed = json!({"note": "Please check this.",
                           "message": "Mein Konto wurde zweimal belastet"});
        assert_eq!(
            reason_for(&models, "english", &mixed),
            "Latin script, mostly English, but a line or field reads as 'de' \
             ('Mein Konto wurde zweimal belastet'); the English checkpoint cannot read it"
        );
    }

    #[test]
    fn detection_details_reach_the_decision() {
        let models = test_models();
        let ctx = ctx_of(&models, "english", false, None);
        let hindi = json!("मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।");
        let d = decide(&ctx, &hindi, &[], &RouteOptions::default()).unwrap();
        assert_eq!(d.model, "multilingual");
        assert_eq!(
            d.reason,
            "non-Latin script (devanagari, 100% of letters); the English checkpoint cannot \
             read it"
        );
        assert!(d.detection.is_object());
        assert_eq!(d.detection["script"], json!("devanagari"));
        let mixed = json!({"note": "Please check this.",
                           "message": "Mein Konto wurde zweimal belastet"});
        let d = decide(&ctx, &mixed, &[], &RouteOptions::default()).unwrap();
        assert_eq!(d.model, "multilingual");
        assert_eq!(d.detection["language"], json!("de"));
        assert_eq!(
            d.detection["mixed_segment"],
            json!("Mein Konto wurde zweimal belastet")
        );
    }

    #[test]
    fn unbound_models_route_with_a_null_repo() {
        let english_only: Vec<(String, ModelSpec)> = test_models().into_iter().take(1).collect();
        let ctx = ctx_of(&english_only, "english", false, None);
        let d = decide(
            &ctx,
            &json!("Ich habe eine Frage"),
            &[],
            &RouteOptions::default(),
        )
        .unwrap();
        assert_eq!(d.model, "multilingual");
        assert_eq!(d.repo, None);
        let d = decide(
            &ctx,
            &json!("I was charged twice"),
            &[],
            &RouteOptions::default(),
        )
        .unwrap();
        assert_eq!(d.repo.as_deref(), Some("acme/models"));
    }

    #[test]
    fn decision_serializes_in_laya_key_order() {
        let models = test_models();
        let ctx = ctx_of(&models, "english", false, None);
        let d = decide(
            &ctx,
            &json!("I was charged twice"),
            &[],
            &RouteOptions::default(),
        )
        .unwrap();
        let value = serde_json::to_value(&d).unwrap();
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec!["model", "repo", "reason", "detection", "workflow"]
        );
    }
}
