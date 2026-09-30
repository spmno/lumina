//! Lazy multi-checkpoint router: port of `laya/router.py`'s `Router`.
//!
//! The registry holds three checkpoints — `english`, `multilingual`,
//! `typed-decisions` — downloaded and built on first use. `max_loaded` caps how
//! many stay resident (least-recently-used is evicted), because all three
//! together are ~1.16B parameters; the default of 2 matches automatic routing,
//! which only ever chooses between `english` and `multilingual`.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let router = lumina::Router::new(&lumina::RouterOptions::default())?;
//! let decision = router.route(
//!     &serde_json::json!("I was charged twice"),
//!     &[],
//!     &lumina::RouteOptions::default(),
//! )?;
//! assert_eq!(decision.model, "english");
//! # Ok(())
//! # }
//! ```
//!
//! ## Deviations from laya
//!
//! * No hooks layer (`laya.hooks`), no `predict_long`, no structured
//!   `decide`/`decide_batch`, no `agent_kwargs` passthrough and no SHA-256 digest
//!   pinning — [`Agent::load_with`](crate::Agent::load_with) is the whole checkpoint surface.
//! * `lang_guess` takes language codes (strings), not callables.
//! * `predict` forwards no `lang` to the agent: the shipped checkpoints carry no
//!   per-language temperature schedules to apply.
//! * `predict_batch` keeps laya's model grouping (at most one load per routed
//!   checkpoint, results restored to input order) but runs each request's forward
//!   pass individually instead of sharing passes across requests with identical
//!   question schemas; per-request `max_len`/`head_max_len` budgets therefore apply
//!   directly.
//! * `route_batch`/`predict_batch` accept `model`/`task`/`lang`/`lang_guess` as
//!   JSON strings only — laya stringifies whatever it finds — and reject other
//!   types at the boundary.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use serde_json::Value;

use crate::agent::{Agent, PredictOverrides};
use crate::confidence::{check_min_confidence, flag_low_confidence};
use crate::error::{LuminaError, Result};
use crate::pyjson::py_type_name;
use crate::question::Question;
use crate::routing::{ModelSpec, RouteDecision, RouteOptions, RoutingCtx, decide, normalise_name};

/// Options for constructing a [`Router`] (laya's `Router.__init__`).
#[derive(Clone, Copy, Debug, Default)]
pub struct RouterOptions<'a> {
    /// The whole registry: `(name, repo, subfolder)` entries binding model names
    /// to checkpoint sources (hub repos or local directories); names normalise
    /// to the three routing names, a later entry replaces an earlier one for the
    /// same name, and an empty list routes but cannot predict.
    pub models: &'a [(&'a str, &'a str, Option<&'a str>)],
    /// Device passed to every checkpoint load (`"cpu"`, `"cuda"`, or `None`).
    pub device: Option<&'a str>,
    /// Hub access token; `None` falls back to `$HF_TOKEN`.
    pub token: Option<&'a str>,
    /// Hub revision (commit/branch/tag) applied to every checkpoint load.
    pub revision: Option<&'a str>,
    /// Per-model revision overrides; a present `None` masks `revision`.
    pub revisions: &'a [(&'a str, Option<&'a str>)],
    /// Cap on resident checkpoints (least-recently-used eviction); default 2,
    /// minimum 1.
    pub max_loaded: Option<usize>,
    /// Model for states detection cannot decide; default `"english"`.
    pub default: Option<&'a str>,
    /// Opt in to selecting `typed-decisions` when question ids match a workflow.
    pub auto_task_detection: bool,
    /// Download and build every checkpoint on construction.
    pub preload: bool,
    /// Language code consulted as a hint for every request, after any per-call
    /// hint.
    pub lang_guess: Option<&'a str>,
}

/// Routing and token-budget options for one [`Router::predict`] call.
#[derive(Clone, Copy, Debug, Default)]
pub struct PredictOptions<'a> {
    /// Pin the model by registry name or alias.
    pub model: Option<&'a str>,
    /// Pin the model by task name.
    pub task: Option<&'a str>,
    /// ISO-style language code.
    pub lang: Option<&'a str>,
    /// Caller's language guess, consulted after `lang`.
    pub lang_guess: Option<&'a str>,
    /// State token budget for this call.
    pub max_len: Option<usize>,
    /// Question-head token budget for this call.
    pub head_max_len: Option<usize>,
    /// Flag answers whose confidence falls below this.
    pub min_confidence: Option<f64>,
}

impl PredictOptions<'_> {
    /// The routing subset of these options.
    const fn route_options(&self) -> RouteOptions<'_> {
        RouteOptions {
            model: self.model,
            task: self.task,
            lang: self.lang,
            lang_guess: self.lang_guess,
        }
    }
}

/// Parse a `LAYA_MODELS` list into registry entries.
///
/// Entries are comma-separated `name=source` pairs, where `source` is a hub repo
/// or a local directory and may carry a `:subfolder` suffix
/// (`org/repo:multilingual`).
///
/// Empty entries are skipped, so an empty or unset variable configures no
/// models: routing still works, prediction reports the missing source.
///
/// # Errors
/// Malformed entries, and unknown model names (see
/// [`normalise_name`](crate::normalise_name)).
pub fn parse_model_sources(value: &str) -> Result<Vec<(String, String, Option<String>)>> {
    let mut out: Vec<(String, String, Option<String>)> = Vec::new();
    for entry in value.split(',') {
        let entry = entry.trim();
        if entry.is_empty() {
            continue;
        }
        let Some((name, source)) = entry.split_once('=') else {
            return Err(LuminaError::Invalid(format!(
                "LAYA_MODELS entry {entry:?} must be 'name=source'"
            )));
        };
        let key = normalise_name(name)?;
        let source = source.trim();
        if source.is_empty() {
            return Err(LuminaError::Invalid(format!(
                "LAYA_MODELS entry {entry:?}: empty source"
            )));
        }
        let (repo, subfolder) = match source.split_once(':') {
            Some((repo, sub)) => {
                if sub.is_empty() {
                    return Err(LuminaError::Invalid(format!(
                        "LAYA_MODELS entry {entry:?}: empty subfolder"
                    )));
                }
                (repo, Some(sub))
            }
            None => (source, None),
        };
        let repo = repo.to_owned();
        let subfolder = subfolder.map(str::to_owned);
        match out.iter_mut().find(|(k, _, _)| *k == key) {
            Some(existing) => {
                existing.1 = repo;
                existing.2 = subfolder;
            }
            None => out.push((key, repo, subfolder)),
        }
    }
    Ok(out)
}

/// Lazily loads laya checkpoints and sends each request to the right one.
///
/// Models are downloaded and built on first use, and inference runs without the
/// lifecycle lock so concurrent predictions share a resident checkpoint.
pub struct Router {
    models: Vec<(String, ModelSpec)>,
    device: Option<String>,
    token: Option<String>,
    revision: Option<String>,
    revisions: Vec<(String, Option<String>)>,
    default: String,
    auto_task_detection: bool,
    lang_guess: Option<String>,
    state: Mutex<RouterState>,
}

/// Resident agents and the LRU order bookkeeping, guarded by the router's mutex.
struct RouterState {
    agents: HashMap<String, Arc<Agent>>,
    /// Least-recently-used first.
    order: Vec<String>,
    max_loaded: usize,
}

impl RouterState {
    /// Move `key` to most-recently-used, appending unknown keys (as `attach` does).
    fn touch(&mut self, key: &str) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push(key.to_owned());
    }

    /// Drop least-recently-used agents until `max_loaded` holds.
    fn evict(&mut self) {
        while self.order.len() > self.max_loaded {
            let victim = self.order.remove(0);
            self.agents.remove(&victim);
        }
    }
}

/// Lock the router state, recovering from a poisoned lock: the maps stay
/// internally consistent, and a panicked holder only ever leaves bookkeeping
/// stale, never corrupt.
fn lock(state: &Mutex<RouterState>) -> std::sync::MutexGuard<'_, RouterState> {
    state
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Revision for a model: a `revisions` entry wins over the global `revision` (a
/// present `None` deliberately masks it), and absence falls back to the global.
fn revision_for<'a>(
    revisions: &'a [(String, Option<String>)],
    key: &str,
    global: Option<&'a str>,
) -> Option<&'a str> {
    match revisions.iter().find(|(k, _)| k == key) {
        Some((_, rev)) => rev.as_deref(),
        None => global,
    }
}

/// A string routing field of a batch request: absent/null is `None`, anything
/// else must be a string.
fn str_field<'a>(value: Option<&'a Value>, field: &str) -> Result<Option<&'a str>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(other) => Err(LuminaError::Invalid(format!(
            "'{field}' must be a string, got {}",
            py_type_name(other)
        ))),
    }
}

/// The per-request routing overrides of a batch request.
fn route_options_from_request(request: &Value) -> Result<RouteOptions<'_>> {
    Ok(RouteOptions {
        model: str_field(request.get("model"), "model")?,
        task: str_field(request.get("task"), "task")?,
        lang: str_field(request.get("lang"), "lang")?,
        lang_guess: str_field(request.get("lang_guess"), "lang_guess")?,
    })
}

/// A per-request token-budget override: absent/null, or a non-negative integer.
fn opt_usize(value: Option<&Value>) -> Result<Option<usize>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(v) => Ok(Some(
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| {
                    LuminaError::Invalid(format!(
                        "token-budget override must be a non-negative integer, got {}",
                        py_type_name(v)
                    ))
                })?,
        )),
    }
}

/// Parse a laya questions dict (`{qid: definition}`) into validated questions in
/// dict order.
///
/// # Errors
/// A non-dict `questions`, and every
/// [`Question::from_def`](crate::Question::from_def) failure, naming the question.
pub fn parse_questions(questions: &Value) -> Result<Vec<(String, Question)>> {
    let map = questions.as_object().ok_or_else(|| {
        LuminaError::Invalid(format!(
            "'questions' must be a dict, got {}",
            py_type_name(questions)
        ))
    })?;
    let mut out = Vec::with_capacity(map.len());
    for (qid, qdef) in map {
        out.push((qid.clone(), Question::from_def(qid, qdef)?));
    }
    Ok(out)
}

/// Append the `routing` key to a prediction payload (after the agent's own keys,
/// as laya does).
fn attach_routing(result: &mut Value, decision: &RouteDecision) -> Result<()> {
    if let Some(obj) = result.as_object_mut() {
        obj.insert("routing".to_owned(), serde_json::to_value(decision)?);
    }
    Ok(())
}

fn internal_index_error() -> LuminaError {
    LuminaError::Invalid("internal error: batch index out of range".to_owned())
}

impl Router {
    /// Build a router over the configured registry; an empty registry routes but
    /// cannot predict.
    ///
    /// # Errors
    /// Unknown model names in `models`/`revisions`/`default`, and any failure of
    /// the preload [`Router::load`] calls.
    pub fn new(opts: &RouterOptions<'_>) -> Result<Self> {
        let mut models: Vec<(String, ModelSpec)> = Vec::with_capacity(opts.models.len());
        for &(name, repo, subfolder) in opts.models {
            let key = normalise_name(name)?;
            let spec = ModelSpec {
                repo: repo.to_owned(),
                subfolder: subfolder.map(str::to_owned),
            };
            match models.iter_mut().find(|(k, _)| *k == key) {
                Some(entry) => entry.1 = spec,
                None => models.push((key, spec)),
            }
        }
        let mut revisions = Vec::with_capacity(opts.revisions.len());
        for &(name, rev) in opts.revisions {
            revisions.push((normalise_name(name)?, rev.map(str::to_owned)));
        }
        let router = Self {
            models,
            device: opts.device.map(str::to_owned),
            token: opts
                .token
                .map(str::to_owned)
                .or_else(|| std::env::var("HF_TOKEN").ok()),
            revision: opts.revision.map(str::to_owned),
            revisions,
            default: normalise_name(opts.default.unwrap_or("english"))?,
            auto_task_detection: opts.auto_task_detection,
            lang_guess: opts.lang_guess.map(str::to_owned),
            state: Mutex::new(RouterState {
                agents: HashMap::new(),
                order: Vec::new(),
                max_loaded: opts.max_loaded.unwrap_or(2).max(1),
            }),
        };
        if opts.preload {
            router.preload(None)?;
        }
        Ok(router)
    }

    /// Return the agent for `name`, downloading and building it on first use.
    ///
    /// The registry entry is resolved and the checkpoint built under the router
    /// lock, so concurrent callers share one agent; the returned [`Arc`] lets
    /// predictions run without the lock.
    ///
    /// # Errors
    /// Unknown model names, and everything [`Agent::load_with`](crate::Agent::load_with) can
    /// fail with.
    // The checkpoint is built under this lock (as laya's `_load` does) so
    // concurrent callers share one agent instead of racing to download and build
    // duplicate ~800 MB models.
    #[allow(clippy::significant_drop_tightening)]
    pub fn load(&self, name: &str) -> Result<Arc<Agent>> {
        let key = normalise_name(name)?;
        let mut state = lock(&self.state);
        if let Some(agent) = state.agents.get(&key) {
            let agent = Arc::clone(agent);
            state.touch(&key);
            return Ok(agent);
        }
        let spec = self
            .models
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, spec)| spec)
            .ok_or_else(|| {
                LuminaError::Invalid(format!(
                    "model '{key}' has no checkpoint source; bind one with \
                     RouterOptions::models or LAYA_MODELS='{key}=<hub repo or local path>'"
                ))
            })?;
        let revision = revision_for(&self.revisions, &key, self.revision.as_deref());
        let agent = Arc::new(Agent::load_with(
            &spec.repo,
            self.device.as_deref(),
            self.token.as_deref(),
            spec.subfolder.as_deref(),
            revision,
        )?);
        state.agents.insert(key.clone(), Arc::clone(&agent));
        state.order.push(key);
        state.evict();
        Ok(agent)
    }

    /// Register an already-built agent under `name` instead of loading a second
    /// copy.
    ///
    /// `max_loaded` is raised so an attached agent is never evicted by a later
    /// load.
    ///
    /// # Errors
    /// Unknown model names.
    pub fn attach(&self, name: &str, agent: Arc<Agent>) -> Result<Arc<Agent>> {
        let key = normalise_name(name)?;
        let mut state = lock(&self.state);
        state.agents.insert(key.clone(), Arc::clone(&agent));
        state.touch(&key);
        state.max_loaded = state.max_loaded.max(state.agents.len());
        drop(state);
        Ok(agent)
    }

    /// Download and build checkpoints up front so no request pays a model load.
    ///
    /// `max_loaded` is raised to fit both the requested checkpoints and everything
    /// already resident, so incremental preloading evicts nothing.
    ///
    /// # Errors
    /// Unknown model names, and everything [`Router::load`] can fail with.
    pub fn preload(&self, names: Option<&[&str]>) -> Result<()> {
        let names: Vec<String> = match names {
            None => self.models.iter().map(|(k, _)| k.clone()).collect(),
            Some(names) => {
                let mut out = Vec::with_capacity(names.len());
                for name in names {
                    out.push(normalise_name(name)?);
                }
                out
            }
        };
        {
            let mut state = lock(&self.state);
            let mut distinct: HashSet<String> = state.agents.keys().cloned().collect();
            for name in &names {
                distinct.insert(name.clone());
            }
            state.max_loaded = state.max_loaded.max(distinct.len());
        }
        for name in &names {
            let already = lock(&self.state).agents.contains_key(name);
            if !already {
                self.load(name)?;
            }
        }
        Ok(())
    }

    /// Free one model, or all of them.
    ///
    /// # Errors
    /// Unknown model names.
    pub fn unload(&self, name: Option<&str>) -> Result<()> {
        let mut state = lock(&self.state);
        match name {
            None => {
                state.agents.clear();
                state.order.clear();
            }
            Some(name) => {
                let key = normalise_name(name)?;
                state.agents.remove(&key);
                if let Some(pos) = state.order.iter().position(|k| *k == key) {
                    state.order.remove(pos);
                }
            }
        }
        drop(state);
        Ok(())
    }

    /// Resident models, least-recently-used first.
    pub fn loaded(&self) -> Vec<String> {
        lock(&self.state).order.clone()
    }

    /// `(name, device)` of every resident checkpoint, least-recently-used first,
    /// without loading anything or touching the LRU order — the read-only view
    /// status reporting needs.
    pub fn loaded_devices(&self) -> Vec<(String, String)> {
        let state = lock(&self.state);
        state
            .order
            .iter()
            .filter_map(|name| {
                state
                    .agents
                    .get(name)
                    .map(|agent| (name.clone(), agent.device()))
            })
            .collect()
    }

    /// The current cap on resident checkpoints.
    pub fn max_loaded(&self) -> usize {
        lock(&self.state).max_loaded
    }

    /// Decide which checkpoint to use for `state`, without loading anything.
    ///
    /// `question_ids` are the ids of the questions that will be asked (used only
    /// to match a typed-decisions workflow).
    ///
    /// # Errors
    /// Unknown model or task names; see [`crate::routing`] for the decision tree.
    pub fn route(
        &self,
        state: &Value,
        question_ids: &[&str],
        opts: &RouteOptions<'_>,
    ) -> Result<RouteDecision> {
        let ctx = RoutingCtx {
            models: &self.models,
            default: &self.default,
            auto_task_detection: self.auto_task_detection,
            lang_guess: self.lang_guess.as_deref(),
        };
        decide(&ctx, state, question_ids, opts)
    }

    /// Route, then answer every question in one forward pass on the chosen
    /// checkpoint — the `system_one` payload plus a `routing` key.
    ///
    /// `min_confidence` (validated first) flags answers below it with
    /// `low_confidence: true`; `max_len`/`head_max_len` override the agent's token
    /// budgets for this call.
    ///
    /// # Errors
    /// Everything [`Router::route`], [`Router::load`], and
    /// [`Agent::predict`](crate::Agent::predict) can fail with, and an invalid
    /// `min_confidence`.
    pub fn predict(
        &self,
        state: &Value,
        questions: &[(String, Question)],
        opts: &PredictOptions<'_>,
    ) -> Result<Value> {
        let mc = opts.min_confidence.map(check_min_confidence).transpose()?;
        let ids: Vec<&str> = questions.iter().map(|(qid, _)| qid.as_str()).collect();
        let decision = self.route(state, &ids, &opts.route_options())?;
        let agent = self.load(&decision.model)?;
        let overrides = PredictOverrides {
            max_len: opts.max_len,
            head_max_len: opts.head_max_len,
        };
        let mut result = agent.predict_overridden(state, questions, &overrides)?;
        attach_routing(&mut result, &decision)?;
        if let Some(mc) = mc {
            flag_low_confidence(std::slice::from_mut(&mut result), mc);
        }
        Ok(result)
    }

    /// [`Router::predict`] under laya's alternate name.
    pub fn system_one(
        &self,
        state: &Value,
        questions: &[(String, Question)],
        opts: &PredictOptions<'_>,
    ) -> Result<Value> {
        self.predict(state, questions, opts)
    }

    /// Route a heterogeneous request batch without loading any checkpoints.
    ///
    /// Each request is a dict with `state` and `questions` plus the same optional
    /// string routing overrides [`Router::route`] takes: `model`, `task`, `lang`,
    /// `lang_guess`. Decisions preserve input order.
    ///
    /// # Errors
    /// Requests that are not dicts, lack `state`/`questions`, or whose `questions`
    /// is not a dict; unknown model names.
    pub fn route_batch(&self, requests: &[Value]) -> Result<Vec<RouteDecision>> {
        let mut decisions = Vec::with_capacity(requests.len());
        for (i, request) in requests.iter().enumerate() {
            let obj = request.as_object().ok_or_else(|| {
                LuminaError::Invalid(format!(
                    "request {i} must be a dict, got {}",
                    py_type_name(request)
                ))
            })?;
            let state = obj.get("state").ok_or_else(|| {
                LuminaError::Invalid(format!("request {i} is missing required key 'state'"))
            })?;
            let questions = obj.get("questions").ok_or_else(|| {
                LuminaError::Invalid(format!("request {i} is missing required key 'questions'"))
            })?;
            let qmap = questions.as_object().ok_or_else(|| {
                LuminaError::Invalid(format!(
                    "request {i} 'questions' must be a dict, got {}",
                    py_type_name(questions)
                ))
            })?;
            let ids: Vec<&str> = qmap.keys().map(String::as_str).collect();
            let opts = route_options_from_request(request)?;
            decisions.push(self.route(state, &ids, &opts)?);
        }
        Ok(decisions)
    }

    /// Route and execute a heterogeneous request batch with minimal model churn.
    ///
    /// Requests are routed first and grouped by checkpoint in first-appearance
    /// order, so each routed model is loaded at most once; results are restored to
    /// input order. Requests may carry per-request `model`/`task`/`lang`/
    /// `lang_guess` and `max_len`/`head_max_len` (non-negative integers). Each
    /// request's forward pass is individual — see the module's deviations note.
    ///
    /// # Errors
    /// Everything [`Router::route_batch`] and the per-request
    /// [`Agent::predict`](crate::Agent::predict) can fail with, and an invalid
    /// `min_confidence`.
    pub fn predict_batch(
        &self,
        requests: &[Value],
        min_confidence: Option<f64>,
    ) -> Result<Vec<Value>> {
        let mc = min_confidence.map(check_min_confidence).transpose()?;
        let decisions = self.route_batch(requests)?;
        if decisions.is_empty() {
            return Ok(Vec::new());
        }
        // Group indices by checkpoint in first-appearance order, so at most one
        // load happens per routed model.
        let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, decision) in decisions.iter().enumerate() {
            let key = normalise_name(&decision.model)?;
            if let Some((_, indices)) = groups.iter_mut().find(|(k, _)| *k == key) {
                indices.push(i);
            } else {
                groups.push((key, Vec::from([i])));
            }
        }
        let mut results: Vec<Option<Value>> = vec![None; requests.len()];
        for (model_name, indices) in &groups {
            // Parse every request of the group before loading, so a bad budget or
            // question definition fails before a checkpoint is downloaded.
            let mut prepared = Vec::with_capacity(indices.len());
            for &i in indices {
                let request = requests.get(i).ok_or_else(internal_index_error)?;
                let state = request.get("state").ok_or_else(|| {
                    LuminaError::Invalid(format!("request {i} is missing required key 'state'"))
                })?;
                let questions = parse_questions(request.get("questions").ok_or_else(|| {
                    LuminaError::Invalid(format!("request {i} is missing required key 'questions'"))
                })?)?;
                let overrides = PredictOverrides {
                    max_len: opt_usize(request.get("max_len"))?,
                    head_max_len: opt_usize(request.get("head_max_len"))?,
                };
                prepared.push((i, state, questions, overrides));
            }
            let agent = self.load(model_name)?;
            for (i, state, questions, overrides) in prepared {
                let mut result = agent.predict_overridden(state, &questions, &overrides)?;
                attach_routing(
                    &mut result,
                    decisions.get(i).ok_or_else(internal_index_error)?,
                )?;
                *results.get_mut(i).ok_or_else(internal_index_error)? = Some(result);
            }
        }
        let mut out = Vec::with_capacity(results.len());
        for result in results {
            out.push(result.ok_or_else(|| {
                LuminaError::Invalid(
                    "internal error: batch execution did not produce every result".to_owned(),
                )
            })?);
        }
        if let Some(mc) = mc {
            flag_low_confidence(&mut out, mc);
        }
        Ok(out)
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
    use serde_json::json;

    use super::*;

    #[test]
    fn routes_english_state_with_a_configured_registry() {
        let router = Router::new(&RouterOptions {
            models: &[("english", "acme/models", None)],
            ..Default::default()
        })
        .unwrap();
        let d = router
            .route(&json!("I was charged twice"), &[], &RouteOptions::default())
            .unwrap();
        assert_eq!(d.model, "english");
        assert_eq!(d.repo.as_deref(), Some("acme/models"));
        assert_eq!(d.reason, "English Latin text");
        assert_eq!(router.loaded(), Vec::<String>::new());
    }

    #[test]
    fn model_entries_bind_registry_sources() {
        let router = Router::new(&RouterOptions {
            models: &[("english", "/data/english-ckpt", None)],
            ..Default::default()
        })
        .unwrap();
        let d = router
            .route(&json!("I was charged twice"), &[], &RouteOptions::default())
            .unwrap();
        assert_eq!(d.model, "english");
        assert_eq!(d.repo.as_deref(), Some("/data/english-ckpt"));
        // routing only: nothing was downloaded
        assert!(router.loaded().is_empty());
    }

    #[test]
    fn an_empty_registry_routes_but_cannot_load() {
        let router = Router::new(&RouterOptions::default()).unwrap();
        let d = router
            .route(&json!("I was charged twice"), &[], &RouteOptions::default())
            .unwrap();
        assert_eq!(d.model, "english");
        assert_eq!(d.repo, None);
        let Err(error) = router.load("english") else {
            panic!("expected loading an unconfigured model to fail");
        };
        let error = error.to_string();
        assert!(
            error.contains("model 'english' has no checkpoint source"),
            "{error}"
        );
        assert!(error.contains("LAYA_MODELS"), "{error}");
    }

    #[test]
    fn model_sources_parse_repos_subfolders_and_paths() {
        let entries = parse_model_sources(
            "english=acme/models, multilingual=acme/models:multilingual, typed=acme/typed",
        )
        .unwrap();
        assert_eq!(
            entries,
            vec![
                ("english".to_owned(), "acme/models".to_owned(), None),
                (
                    "multilingual".to_owned(),
                    "acme/models".to_owned(),
                    Some("multilingual".to_owned())
                ),
                ("typed-decisions".to_owned(), "acme/typed".to_owned(), None),
            ]
        );
        let entries = parse_model_sources("english=/data/english-ckpt").unwrap();
        assert_eq!(
            entries,
            vec![("english".to_owned(), "/data/english-ckpt".to_owned(), None)]
        );
        assert!(parse_model_sources("  ").unwrap().is_empty());
        // a later entry replaces an earlier one for the same name
        let entries = parse_model_sources("en=acme/a, english=acme/b").unwrap();
        assert_eq!(
            entries,
            vec![("english".to_owned(), "acme/b".to_owned(), None)]
        );
    }

    #[test]
    fn model_sources_reject_malformed_entries() {
        for bad in [
            "english",
            "english=",
            "=acme/models",
            "bogus=acme/models",
            "english=acme/models:",
        ] {
            assert!(parse_model_sources(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn router_level_lang_guess_routes_every_request() {
        let router = Router::new(&RouterOptions {
            lang_guess: Some("de"),
            ..Default::default()
        })
        .unwrap();
        let d = router
            .route(&json!("I was charged twice"), &[], &RouteOptions::default())
            .unwrap();
        assert_eq!(d.model, "multilingual");
        assert_eq!(
            d.reason,
            "Router(lang_guess=...): the caller identified this as non-English text"
        );
    }

    #[test]
    fn default_may_be_multilingual() {
        let router = Router::new(&RouterOptions {
            default: Some("ml"),
            ..Default::default()
        })
        .unwrap();
        let d = router
            .route(&json!("12345 !!!"), &[], &RouteOptions::default())
            .unwrap();
        assert_eq!(d.model, "multilingual");
        assert_eq!(
            d.reason,
            "no letters detected in state; using default (multilingual)"
        );
    }

    #[test]
    fn max_loaded_defaults_to_two_and_clamps() {
        assert_eq!(
            Router::new(&RouterOptions::default()).unwrap().max_loaded(),
            2
        );
        assert_eq!(
            Router::new(&RouterOptions {
                max_loaded: Some(0),
                ..Default::default()
            })
            .unwrap()
            .max_loaded(),
            1
        );
    }

    #[test]
    fn revisions_normalise_and_mask_the_global() {
        let router = Router::new(&RouterOptions {
            revision: Some("global"),
            revisions: &[("ml", Some("ml-commit")), ("typed", None)],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            revision_for(&router.revisions, "english", router.revision.as_deref()),
            Some("global")
        );
        assert_eq!(
            revision_for(
                &router.revisions,
                "multilingual",
                router.revision.as_deref()
            ),
            Some("ml-commit")
        );
        // a present None deliberately masks the global revision
        assert_eq!(
            revision_for(
                &router.revisions,
                "typed-decisions",
                router.revision.as_deref()
            ),
            None
        );
        // bad revision names are rejected at construction
        assert!(
            Router::new(&RouterOptions {
                revisions: &[("bogus", None)],
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn route_batch_validates_requests() {
        let router = Router::new(&RouterOptions::default()).unwrap();
        let err = router
            .route_batch(&[json!("not a dict")])
            .unwrap_err()
            .to_string();
        assert_eq!(err, "request 0 must be a dict, got str");
        let err = router
            .route_batch(&[json!({"questions": {}})])
            .unwrap_err()
            .to_string();
        assert_eq!(err, "request 0 is missing required key 'state'");
        let err = router
            .route_batch(&[json!({"state": "hi"})])
            .unwrap_err()
            .to_string();
        assert_eq!(err, "request 0 is missing required key 'questions'");
        let err = router
            .route_batch(&[json!({"state": "hi", "questions": []})])
            .unwrap_err()
            .to_string();
        assert_eq!(err, "request 0 'questions' must be a dict, got list");
        // non-string routing fields are rejected at the boundary
        let err = router
            .route_batch(&[json!({"state": "hi", "questions": {}, "model": 5})])
            .unwrap_err()
            .to_string();
        assert_eq!(err, "'model' must be a string, got int");
        // empty batch: empty decisions
        assert!(router.route_batch(&[]).unwrap().is_empty());
    }

    #[test]
    fn route_batch_routes_in_order_with_overrides() {
        let router = Router::new(&RouterOptions::default()).unwrap();
        let requests = [
            json!({"state": "I was charged twice", "questions": {}}),
            json!({"state": "El pedido llegó roto y quiero que lo cambien", "questions": {}}),
            json!({"state": "I was charged twice", "questions": {}, "model": "ml"}),
        ];
        let decisions = router.route_batch(&requests).unwrap();
        assert_eq!(decisions[0].model, "english");
        assert_eq!(decisions[1].model, "multilingual");
        assert_eq!(decisions[2].model, "multilingual");
        assert_eq!(decisions[2].reason, "explicit model='ml'");
    }

    #[test]
    fn predict_batch_validates_budgets_and_confidence() {
        let router = Router::new(&RouterOptions::default()).unwrap();
        // min_confidence is validated before anything routes
        let err = router
            .predict_batch(&[], Some(1.5))
            .unwrap_err()
            .to_string();
        assert_eq!(err, "min_confidence must be a float in [0.0, 1.0], got 1.5");
        // a bad budget fails before a checkpoint is downloaded
        let bad_budget = [json!({"state": "hi", "questions": {}, "max_len": "big"})];
        let err = router
            .predict_batch(&bad_budget, None)
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "token-budget override must be a non-negative integer, got str"
        );
        // empty batch routes to empty results
        assert!(router.predict_batch(&[], None).unwrap().is_empty());
    }

    #[test]
    fn lru_bookkeeping_touches_and_evicts() {
        let mut state = RouterState {
            agents: HashMap::new(),
            order: vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            max_loaded: 2,
        };
        state.touch("b");
        assert_eq!(state.order, ["a", "c", "b"]);
        state.evict();
        assert_eq!(state.order, ["c", "b"]);
        // touching a fresh key appends, as attach does
        state.touch("d");
        assert_eq!(state.order, ["c", "b", "d"]);
        state.evict();
        assert_eq!(state.order, ["b", "d"]);
        state.max_loaded = 0;
        state.evict();
        assert!(state.order.is_empty());
    }

    #[test]
    fn parse_questions_validates_in_dict_order() {
        let questions = json!({
            "tone": {"type": "noul", "instructions": "Rude?"},
            "team": {"type": "choice", "instructions": "Which team?",
                      "criteria": {"billing": "Payment issues", "tech": null}}
        });
        let parsed = parse_questions(&questions).unwrap();
        let ids: Vec<&str> = parsed.iter().map(|(qid, _)| qid.as_str()).collect();
        assert_eq!(ids, ["tone", "team"]);
        assert!(parse_questions(&json!([])).is_err());
        let err = parse_questions(&json!({"q": {"type": "nope"}}))
            .unwrap_err()
            .to_string();
        assert_eq!(
            err,
            "question \"q\": unknown type 'nope'; use one of ['choice', 'noul', 'score']"
        );
    }
}
