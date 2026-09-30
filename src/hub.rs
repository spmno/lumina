//! Checkpoint resolution: local directories or Hugging Face hub snapshots.

use std::path::{Path, PathBuf};

use hf_hub::{HFClient, HFClientSync};

use crate::{LuminaError, Result};

/// Resolve a laya checkpoint directory to load.
///
/// A local path is used as-is; a hub id downloads the snapshot with laya's
/// allow-pattern filter (weights, configs, and tokenizer only, so sibling
/// checkpoints in the same repo stay untouched). A `subfolder` selects a bundled
/// checkpoint and is appended to either source. `token` authenticates the hub
/// download; `None` keeps hf-hub's own `$HF_TOKEN` / CLI-login fallback.
///
/// # Errors
/// Local-path misses, hub download failures, and subfolders that do not exist.
pub fn resolve(
    source: &str,
    token: Option<&str>,
    subfolder: Option<&str>,
    revision: Option<&str>,
) -> Result<PathBuf> {
    let base = if Path::new(source).is_dir() {
        PathBuf::from(source)
    } else if is_explicit_path(source) {
        return Err(LuminaError::Invalid(format!(
            "local model path not found: {source:?}. Check that the directory exists and that \
             training saved the model successfully."
        )));
    } else {
        download_snapshot(source, token, subfolder, revision)?
    };
    match subfolder {
        Some(s) => {
            let dir = base.join(s);
            if dir.is_dir() {
                Ok(dir)
            } else {
                Err(LuminaError::Invalid(format!(
                    "subfolder {s:?} not found in {source:?}."
                )))
            }
        }
        None => Ok(base),
    }
}

/// Python's local-path test: a source naming an explicit path (`/`, `./`,
/// `../`, or absolute) that does not exist is a missing local directory, never
/// a hub id — anything else falls through to the hub, so repo ids such as
/// `convaiinnovations/laya` keep their slash.
fn is_explicit_path(source: &str) -> bool {
    source.starts_with('/')
        || source.starts_with("./")
        || source.starts_with("../")
        || Path::new(source).is_absolute()
}

fn download_snapshot(
    source: &str,
    token: Option<&str>,
    subfolder: Option<&str>,
    revision: Option<&str>,
) -> Result<PathBuf> {
    let (owner, name) = hf_hub::split_id(source);
    if owner.is_empty() {
        return Err(LuminaError::Invalid(format!(
            "not a hub model id or local directory: {source}"
        )));
    }
    // A token is routed through the async client's builder, which owns the
    // $HF_TOKEN / login-file fallback for the `None` path.
    let client = match token {
        Some(token) => HFClientSync::from_inner(HFClient::builder().token(token).build()?)?,
        None => HFClientSync::new()?,
    };
    let repo = client.model(owner, name);
    let prefix = subfolder.map_or_else(String::new, |s| format!("{s}/"));
    let patterns: Vec<String> = [
        "rl_agent_config.json",
        "model.safetensors",
        "tokenizer/*",
        "encoder/*",
    ]
    .iter()
    .map(|p| format!("{prefix}{p}"))
    .collect();
    Ok(repo
        .snapshot_download()
        .maybe_revision(revision.map(str::to_owned))
        .maybe_allow_patterns(Some(patterns))
        .send()?)
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
    fn repo_ids_with_slashes_stay_hub_sources() {
        assert!(!is_explicit_path("convaiinnovations/laya"));
        assert!(!is_explicit_path("user/model-name"));
        assert!(is_explicit_path("/missing/model"));
        assert!(is_explicit_path("./missing"));
        assert!(is_explicit_path("../missing"));
    }

    #[test]
    fn an_existing_directory_resolves_without_the_hub() {
        let dir = std::env::temp_dir();
        let resolved = resolve(dir.to_str().unwrap(), None, None, None).unwrap();
        assert_eq!(resolved, dir);
    }

    #[test]
    fn missing_explicit_paths_fail_before_any_download() {
        for source in ["/nonexistent/lumina-model", "./nope", "../nope"] {
            let error = resolve(source, None, None, None).unwrap_err().to_string();
            assert!(
                error.starts_with("local model path not found"),
                "{source}: {error}"
            );
        }
    }
}
