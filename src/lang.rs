//! Dependency-free language/script detection used to route between checkpoints, ported
//! from laya's `laya/lang.py` (v0.3.21).
//!
//! Routing needs one decision: *is this English Latin text, or is it something the English
//! checkpoint cannot read?* The English checkpoint does not degrade gently off English, it
//! collapses (0.100 on Hindi at 20 options, where random guessing scores 0.050), so script
//! detection is the primary signal and whether Latin text is English is the secondary one.
//! Script detection is exact; the Latin-script language guess is a stopword/diacritic
//! heuristic and is explicitly best-effort.
//!
//! Two documented deviations from the Python source, neither of which changes a routing
//! outcome where they differ: Python's `str.isalpha` and `[^\W\d_]` are approximated with
//! `char::is_alphabetic` (which additionally counts numeral-letters and combining vowel
//! signs — no stopword contains them, and the runs they join or split only change
//! non-Latin word lists, whose emptiness is what routing reads), and the regexes are
//! hand-scanned.
//!
//! ```
//! # use serde_json::json;
//! let det = lumina::lang::analyse(&json!("Ich habe eine Frage"));
//! assert!(!det.is_english);
//! ```

use std::collections::HashSet;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::lang_data::{
    NON_EN_DIACRITICS, SCRIPT_RANGES, en_only_words, shared_words, stop_set, stop_sets,
};

/// A diacritic rate above this is evidence the text is not English, even when no
/// stopword list matches it (laya constant).
const NON_EN_DIACRITIC_RATE: f64 = 0.02;
/// English function words keep their say only below this rate: a single accented
/// loanword must not alone pull otherwise plain English off the English checkpoint.
const ENGLISH_RESCUE_DIACRITIC_RATE: f64 = 0.06;
/// Non-Latin share that flips a Latin-plurality state: a brand name or order code must
/// not outvote the CJK request around it letter for letter.
const NON_LATIN_FRACTION: f64 = 0.2;
/// A short message needs a large non-Latin share to count; a long payload can dilute it.
const NON_LATIN_MIN_FRACTION: f64 = 0.1;
/// A sentence's worth of non-Latin letters counts too when the share is diluted.
const NON_LATIN_MIN_LETTERS: i64 = 10;
/// laya's detection window: the state, segment and per-line caps are all 4000 characters.
const MAX_CHARS: usize = 4000;

/// Evidence behind the Latin-script language guess (laya's `latin_profile` dict).
#[derive(Debug, Clone, Serialize)]
pub struct LatinProfile {
    /// Best-effort language code; `None` when undecided.
    pub language: Option<String>,
    /// Hits on the English stopword list.
    pub english_hits: usize,
    /// Fraction of characters carrying a non-English diacritic.
    pub diacritic_rate: f64,
    /// Whether the diacritic rate alone says "not English".
    pub looks_non_english: bool,
}

/// Full detection result for a state (laya's `analyse` dict, same key order).
#[derive(Debug, Clone, Serialize)]
pub struct Analysis {
    /// Dominant script: "latin", "han", "devanagari", ... or "unknown" when there are no
    /// letters.
    pub script: String,
    /// Fraction of alphabetic characters per script, insertion-ordered with latin first.
    pub script_profile: Map<String, Value>,
    /// Best-effort language code, `None` when undecided.
    pub language: Option<String>,
    /// Whether the English checkpoint can be expected to read this state.
    pub is_english: bool,
    /// Whether no evidence named the language either way.
    pub language_undecided: bool,
    /// Fraction of characters carrying a non-English diacritic (Latin script only).
    pub diacritic_rate: f64,
    /// Fraction of letters outside the Latin script.
    pub non_latin_fraction: f64,
    /// The line or field that made a mostly-English state non-English, else `None`.
    pub mixed_segment: Option<String>,
}

// --------------------------------------------------------------------- helpers

/// Python's `max()` over key-scored items: the first of equal values wins.
fn first_max_by<T, K: PartialOrd>(
    items: impl Iterator<Item = T>,
    key: impl Fn(&T) -> K,
) -> Option<T> {
    let mut best: Option<(T, K)> = None;
    for item in items {
        let score = key(&item);
        if best.as_ref().is_none_or(|(_, b)| score > *b) {
            best = Some((item, score));
        }
    }
    best.map(|(item, _)| item)
}

/// `chars[i]` for an index the caller keeps in range; `'\0'` classifies as no class.
#[inline]
fn at(chars: &[char], i: usize) -> char {
    chars.get(i).copied().unwrap_or('\0')
}

/// Python's `\w`: alphanumeric or underscore.
#[inline]
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `usize` count to `f64` ratio.
#[allow(clippy::cast_precision_loss)] // counts are bounded by the 4000-char detection window
const fn f(n: usize) -> f64 {
    n as f64
}

/// `usize` count to `i64` for best-line comparisons.
fn i64_of(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// Python's `round(x, 4)`: nearest with ties to even, via decimal formatting.
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

/// Python's `round(x)`: nearest integer with ties to even.
fn round0(x: f64) -> i64 {
    format!("{x:.0}").parse().unwrap_or(0)
}

// ------------------------------------------------------------------ word level

/// Maximal runs of letters (laya's `_WORD` finds `[^\W\d_]+`); the `is_alphabetic`
/// approximation is documented on the module.
fn word_runs(text: &str) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        if ch.is_alphabetic() {
            cur.push(ch);
        } else if !cur.is_empty() {
            words.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    words
}

#[inline]
fn in_identifier_span(c: char) -> bool {
    is_word_char(c) || c == '-' || c == '.' || c == '@'
}

/// Port of `_IDENTIFIER.sub(" ", text)`: `(?<![\w-])[\w-]*(?:[.@][\w-]+)+`. A token whose
/// dot or @ joins word characters is an identifier, not prose (`github.com`, `v1.2.3`,
/// `U.S.A.`); a sentence-final period keeps its word, because the pattern needs word
/// characters on both sides of the dot. Matches never cross a character outside
/// `[\w.@-]` and a leftmost match can only begin at a run start, so each maximal span
/// scans independently — which is also what keeps this linear.
fn replace_identifiers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if !in_identifier_span(at(&chars, i)) {
            out.push(at(&chars, i));
            i += 1;
            continue;
        }
        let start = i;
        while i < chars.len() && in_identifier_span(at(&chars, i)) {
            i += 1;
        }
        out.push_str(&replace_identifiers_in_span(
            chars.get(start..i).unwrap_or(&[]),
        ));
    }
    out
}

/// Scan one `[\w.@-]` span, replacing every identifier match with a space.
fn replace_identifiers_in_span(span: &[char]) -> String {
    let len = span.len();
    let lead = |c: char| is_word_char(c) || c == '-';
    let junction = |c: char| c == '.' || c == '@';
    let mut out: Vec<char> = span.to_vec();
    let mut pos = 0;
    while pos < len {
        // `(?<![\w-])`: a match cannot start right after a word character or hyphen.
        if pos > 0 && lead(at(span, pos - 1)) {
            pos += 1;
            continue;
        }
        let mut run_end = pos;
        while run_end < len && lead(at(span, run_end)) {
            run_end += 1;
        }
        if run_end == len {
            pos += 1;
            continue; // the run ends without a junction
        }
        // span[run_end] is a junction; consume `junction + [\w-]+` groups while they complete.
        let mut next_junction = run_end;
        let mut iterations = 0;
        let mut end = 0;
        while next_junction < len && junction(at(span, next_junction)) {
            let mut group_end = next_junction + 1;
            while group_end < len && lead(at(span, group_end)) {
                group_end += 1;
            }
            if group_end == next_junction + 1 {
                break; // nothing after the junction: the match stops before it
            }
            end = group_end;
            iterations += 1;
            next_junction = group_end;
        }
        if iterations == 0 {
            pos += 1;
            continue;
        }
        for c in out.iter_mut().skip(pos).take(end - pos) {
            *c = ' ';
        }
        pos = end;
    }
    out.into_iter().collect()
}

/// Port of `_CODE_LINE.search`: a line carrying `=`, `;`, braces, brackets or a call
/// `name(` is code, not prose in any language.
fn code_line(segment: &str) -> bool {
    let chars: Vec<char> = segment.chars().collect();
    (0..chars.len()).any(|i| {
        let c = at(&chars, i);
        "=;{}[]".contains(c) || (is_word_char(c) && at(&chars, i + 1) == '(')
    })
}

/// Port of `_JOINED.search` on one whitespace token: a letter or digit, a joiner
/// (`.`, `_`, `/`, `\`) and another letter or digit make an identifier or compound, not a
/// sentence. Fixed length on purpose, as in laya: an open-ended pattern backtracks
/// quadratically on a long run of letters.
fn joined_token(token: &str) -> bool {
    let chars: Vec<char> = token.chars().collect();
    (0..chars.len()).any(|i| {
        at(&chars, i).is_alphanumeric()
            && "._/\\".contains(at(&chars, i + 1))
            && at(&chars, i + 2).is_alphanumeric()
    })
}

fn flush_caps_run(out: &mut String, run: &mut Vec<char>) {
    // An all-caps run inside mixed-case text is an acronym or code, not a word; an
    // entirely capitalised segment keeps its words (Python `isupper()` on the run).
    if run.len() >= 2
        && run.iter().any(|c| c.is_uppercase())
        && !run.iter().any(|c| c.is_lowercase())
    {
        out.push(' ');
    } else {
        out.extend(run.iter().copied());
    }
    run.clear();
}

/// Port of `_LETTER_RUN.sub(...)`: all-caps letter runs of length at least two become
/// spaces in prose that has at least one lowercase character.
fn strip_caps_runs(prose: &str) -> String {
    let mut out = String::with_capacity(prose.len());
    let mut run: Vec<char> = Vec::new();
    for ch in prose.chars() {
        if ch.is_alphabetic() {
            run.push(ch);
        } else {
            flush_caps_run(&mut out, &mut run);
            out.push(ch);
        }
    }
    flush_caps_run(&mut out, &mut run);
    out
}

// ---------------------------------------------------------------- script level

/// Latin, IPA Extensions, Latin Extended-Additional and the fullwidth Latin letters.
#[inline]
fn is_latin_cp(cp: u32) -> bool {
    cp < 0x02B0
        || (0x1E00..=0x1EFF).contains(&cp)
        || (0xFF21..=0xFF3A).contains(&cp)
        || (0xFF41..=0xFF5A).contains(&cp)
}

/// Count the alphabetic characters of `text` by script, in one pass. Latin is inserted
/// last so a named script wins the tie-break of [`script_from_counts`]; an alphabetic
/// character no range claims counts as `other`, which stays non-Latin — the safe
/// direction, because an unreadable script must not reach the English checkpoint.
fn script_counts(text: &str) -> Vec<(&'static str, usize)> {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    let mut latin = 0_usize;
    for ch in text.chars() {
        if !ch.is_alphabetic() {
            continue;
        }
        let cp = u32::from(ch);
        if is_latin_cp(cp) {
            latin += 1;
            continue;
        }
        let name = SCRIPT_RANGES
            .iter()
            .find(|(_, ranges)| ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi))
            .map_or("other", |(name, _)| *name);
        if let Some(entry) = counts.iter_mut().find(|(n, _)| *n == name) {
            entry.1 += 1;
        } else {
            counts.push((name, 1));
        }
    }
    counts.push(("latin", latin));
    counts
}

fn script_from_counts(counts: &[(&'static str, usize)]) -> &'static str {
    if counts.iter().all(|(_, v)| *v == 0) {
        return "unknown";
    }
    first_max_by(counts.iter().copied(), |(_, v)| *v).map_or("unknown", |(name, _)| name)
}

/// Fractions of alphabetic characters per script, latin first when present (the order
/// callers saw in laya's Python dict).
fn profile_from_counts(counts: &[(&'static str, usize)]) -> Vec<(&'static str, f64)> {
    let total: usize = counts.iter().map(|(_, v)| *v).sum();
    if total == 0 {
        return Vec::new();
    }
    let mut ordered: Vec<(&'static str, f64)> = Vec::new();
    if let Some(&(_, latin)) = counts.iter().find(|(n, _)| *n == "latin") {
        if latin > 0 {
            ordered.push(("latin", f(latin) / f(total)));
        }
    }
    for (name, v) in counts {
        if *name != "latin" {
            ordered.push((*name, f(*v) / f(total)));
        }
    }
    ordered
}

fn profile_map(profile: &[(&'static str, f64)]) -> Map<String, Value> {
    let mut map = Map::new();
    for (name, frac) in profile {
        map.insert((*name).to_owned(), Value::from(*frac));
    }
    map
}

/// The named non-Latin script of one letter, or `None` for Latin and unclaimed letters.
fn script_of(ch: char) -> Option<&'static str> {
    let cp = u32::from(ch);
    if is_latin_cp(cp) || cp < 0x0250 {
        return None;
    }
    SCRIPT_RANGES
        .iter()
        .find(|(_, ranges)| ranges.iter().any(|&(lo, hi)| lo <= cp && cp <= hi))
        .map(|(name, _)| *name)
}

/// Canonical combining marks (Python `unicodedata.combining(ch) != 0`) for the ranges
/// where skipping one changes a run: marks inside a script range keep that script's run
/// either way, so only the cross-script combiners need listing.
fn is_combining_mark(c: char) -> bool {
    const RANGES: &[(u32, u32)] = &[
        (0x0300, 0x036F),
        (0x1AB0, 0x1AFF),
        (0x1DC0, 0x1DFF),
        (0x20D0, 0x20F0),
        (0x302A, 0x302F),
        (0xFE20, 0xFE2F),
    ];
    let cp = u32::from(c);
    RANGES.iter().any(|&(lo, hi)| lo <= cp && cp <= hi)
}

/// Non-Latin runs that read as words rather than annotation inside English prose: a
/// symbol (`Set α to 0.05`, one letter), a proper name (capitalised) and a pronunciation
/// (no script range claims it) are all excluded. A combining mark belongs to the letter
/// before it and never splits a word, so `Влади́мир` stays one capitalised name.
fn non_latin_words(text: &str) -> Vec<String> {
    let mut runs: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut script: Option<&'static str> = None;
    for ch in text.chars() {
        if is_combining_mark(ch) {
            continue;
        }
        let s = script_of(ch);
        if s.is_some() && s == script {
            cur.push(ch);
            continue;
        }
        if !cur.is_empty() {
            runs.push(std::mem::take(&mut cur));
        }
        if let Some(s) = s {
            cur.push(ch);
            script = Some(s);
        } else {
            script = None;
        }
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs.into_iter()
        .filter(|w| w.chars().count() >= 2 && !w.chars().next().is_some_and(char::is_uppercase))
        .collect()
}

/// Dominant script of `text`: "latin", "han", "devanagari", ... or "unknown" when there
/// are no letters.
#[must_use]
pub fn detect_script(text: &str) -> String {
    script_from_counts(&script_counts(text)).to_owned()
}

/// Fraction of alphabetic characters belonging to each detected script (latin first).
#[must_use]
pub fn script_profile(text: &str) -> Map<String, Value> {
    profile_map(&profile_from_counts(&script_counts(text)))
}

// ---------------------------------------------------------------- latin guess

/// Whether plain-English function words outvote a marginal diacritic rate (laya #337):
/// a rate still below the rescue ceiling, at least two distinct English-only function
/// words, and at most one word carrying a non-English letter.
fn english_rescued_by_words(words: &[String], diac_rate: f64) -> bool {
    if diac_rate >= ENGLISH_RESCUE_DIACRITIC_RATE {
        return false;
    }
    let distinct: HashSet<&str> = words.iter().map(String::as_str).collect();
    if en_only_words()
        .iter()
        .filter(|w| distinct.contains(*w))
        .count()
        < 2
    {
        return false;
    }
    distinct
        .iter()
        .filter(|w| w.chars().any(|c| NON_EN_DIACRITICS.contains(c)))
        .count()
        <= 1
}

/// Evidence behind the Latin-script language guess (laya's `latin_profile`).
///
/// A non-English language is only named when it matched at least one word no other list
/// claims: shared function words alone (`la`, `e`, `o`) identify no particular language,
/// and a winner on pure overlap is a guess dressed as a detection.
#[must_use]
pub fn latin_profile(text: &str) -> LatinProfile {
    // 'İ'.lower() is 'i' plus a combining dot that matches no word list, so it is
    // replaced before lowering.
    let words = word_runs(&replace_identifiers(text).replace('İ', "i").to_lowercase());
    let lowered = text.to_lowercase();
    let diac = lowered
        .chars()
        .filter(|c| NON_EN_DIACRITICS.contains(*c))
        .count();
    let diac_rate = f(diac) / f(usize::max(1, lowered.chars().count()));
    let non_english = diac_rate >= NON_EN_DIACRITIC_RATE;
    if words.len() < 4 {
        return LatinProfile {
            language: None,
            english_hits: 0,
            diacritic_rate: diac_rate,
            looks_non_english: non_english,
        };
    }
    let distinct: HashSet<&str> = words.iter().map(String::as_str).collect();
    let mut en = 0_usize;
    let mut best_lg: Option<&'static str> = None;
    let mut best_score = 0_usize;
    for (lg, sw) in stop_sets() {
        let hits = words.iter().filter(|w| sw.contains(w.as_str())).count();
        if *lg == "en" {
            en = hits;
            continue;
        }
        let evidenced = sw
            .iter()
            .any(|w| distinct.contains(*w) && !shared_words().contains(*w));
        if evidenced && (best_lg.is_none() || hits > best_score) {
            best_lg = Some(*lg);
            best_score = hits;
        }
    }
    let language = match best_lg {
        // a non-English language needs a clear margin over English function words
        Some(lg) if best_score >= 2.max(en + 2) => Some(lg.to_owned()),
        // ...or two hits here too, on the strength of the diacritics alone
        Some(lg) if non_english && best_score >= 2.max(en) => Some(lg.to_owned()),
        _ if en > 0 && (!non_english || english_rescued_by_words(&words, diac_rate)) => {
            Some("en".to_owned())
        }
        _ => None,
    };
    LatinProfile {
        language,
        english_hits: en,
        diacritic_rate: diac_rate,
        looks_non_english: non_english,
    }
}

/// Best-effort language code for Latin-script text, or `None` when undecided.
#[must_use]
pub fn guess_latin_language(text: &str) -> Option<String> {
    latin_profile(text).language
}

// --------------------------------------------------------------- state level

fn iter_text(state: &Value) -> Vec<String> {
    iter_text_depth(state, 0)
}

/// Collect the string leaves of a state, so detection sees real content. Keys are
/// ignored: they are usually English field names. Non-string scalars contribute
/// nothing, as in laya's `_iter_text`.
fn iter_text_depth(state: &Value, depth: usize) -> Vec<String> {
    if depth > 6 {
        return Vec::new();
    }
    match state {
        Value::String(s) => vec![s.clone()],
        Value::Object(map) => map
            .values()
            .flat_map(|v| iter_text_depth(v, depth + 1))
            .collect(),
        Value::Array(items) => items
            .iter()
            .flat_map(|v| iter_text_depth(v, depth + 1))
            .collect(),
        _ => Vec::new(),
    }
}

/// Flatten a state into the text used for detection, capped at `max_chars` characters
/// (keys are ignored: they are usually English field names).
#[must_use]
pub fn state_text(state: &Value, max_chars: usize) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut budget = max_chars;
    for leaf in iter_text(state) {
        if budget == 0 {
            break;
        }
        let n = leaf.chars().count();
        if n > budget {
            parts.push(leaf.chars().take(budget).collect());
            break;
        }
        parts.push(leaf);
        budget = budget.saturating_sub(n + 1);
    }
    parts.join(" ").chars().take(max_chars).collect()
}

/// Language code for one non-code line, or `None` when it does not name a foreign
/// language. Same evidence bar as the whole state — at least four words and a language
/// `latin_profile` will name — and, because one line carries far less text, two
/// *different* words of that language; acronyms and slash compounds are not words.
fn named_prose_language(segment: &str) -> Option<String> {
    if segment.trim().is_empty() || code_line(segment) {
        return None;
    }
    let prose: String = segment
        .split_whitespace()
        .filter(|tok| !joined_token(tok))
        .collect::<Vec<&str>>()
        .join(" ");
    let prose = if prose.chars().any(char::is_lowercase) {
        strip_caps_runs(&prose)
    } else {
        prose
    };
    let tokens = word_runs(&prose);
    if tokens.len() < 4 {
        return None;
    }
    let lang = match latin_profile(&prose).language {
        Some(lang) if lang != "en" => lang,
        _ => return None,
    };
    let lowered: HashSet<String> = tokens.iter().map(|t| t.to_lowercase()).collect();
    let sw = stop_set(&lang)?;
    if lowered.iter().filter(|w| sw.contains(w.as_str())).count() < 2 {
        return None;
    }
    Some(lang)
}

/// First line or field that, read on its own, is named a non-English language, as
/// `(language, segment)`. Reads at most `max_chars` characters in all, which is what
/// keeps a huge field cheap.
fn non_english_segment(state: &Value, max_chars: usize) -> Option<(String, String)> {
    let mut seen = 0_usize;
    for leaf in iter_text(state) {
        for seg in leaf.split('\n') {
            if seen >= max_chars {
                return None;
            }
            let seg: String = seg.chars().take(max_chars - seen).collect();
            seen += seg.chars().count();
            if let Some(lang) = named_prose_language(&seg) {
                return Some((lang, seg.trim().to_owned()));
            }
        }
    }
    None
}

/// Detection result for one already-flattened string (laya's `_analyse_text`); it does
/// not look for a foreign line inside mostly-English text — [`analyse`] does, because it
/// needs the original state.
fn analyse_text(text: &str) -> Analysis {
    let counts = script_counts(text);
    let profile = profile_from_counts(&counts);
    let profile_json = profile_map(&profile);
    let mut script = script_from_counts(&counts).to_owned();
    let latin_frac = profile
        .iter()
        .find(|(n, _)| *n == "latin")
        .map_or(0.0, |(_, v)| *v);
    let non_latin = if profile.is_empty() {
        0.0
    } else {
        round4(1.0 - latin_frac)
    };
    let n_alpha = text.chars().filter(|c| c.is_alphabetic()).count();
    let n_non_latin = round0(non_latin * f(n_alpha));
    if script == "latin"
        && !non_latin_words(text).is_empty()
        && (non_latin >= NON_LATIN_FRACTION
            || (non_latin >= NON_LATIN_MIN_FRACTION && n_non_latin >= NON_LATIN_MIN_LETTERS))
    {
        // Non-Latin letters are not for the English checkpoint even when Latin letters
        // are the plurality: switch to the strongest other script.
        if let Some((name, _)) = first_max_by(
            profile.iter().filter(|(n, _)| *n != "latin").copied(),
            |(_, v)| *v,
        ) {
            name.clone_into(&mut script);
        }
    }
    if script == "unknown" {
        return Analysis {
            script,
            script_profile: profile_json,
            language: None,
            is_english: true,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: 0.0,
            mixed_segment: None,
        };
    }
    if script != "latin" {
        return Analysis {
            script,
            script_profile: profile_json,
            language: None,
            is_english: false,
            language_undecided: true,
            diacritic_rate: 0.0,
            non_latin_fraction: non_latin,
            mixed_segment: None,
        };
    }
    let prof_lat = latin_profile(text);
    // Undecided is not English. When nothing identifies the language, non-English
    // letters are enough to prefer the multilingual checkpoint; text with no such
    // letters (including short English) still goes to the English one.
    let undecided = prof_lat.language.is_none();
    let english =
        prof_lat.language.as_deref() == Some("en") || (undecided && !prof_lat.looks_non_english);
    Analysis {
        script,
        script_profile: profile_json,
        language: prof_lat.language,
        is_english: english,
        language_undecided: undecided,
        diacritic_rate: round4(prof_lat.diacritic_rate),
        non_latin_fraction: non_latin,
        mixed_segment: None,
    }
}

/// A string value that is itself not safe for the English checkpoint (laya's
/// `_leaf_non_english`): the non-English line with the most alphabetic characters, or
/// `None`. A one-word name ("José") and a capitalised non-Latin name stay out, as do
/// code lines, acronyms and slash compounds, so a pasted traceback is not a message.
/// Each line is capped at 4000 characters, and a long earlier field does not consume
/// the budget of the next one.
fn leaf_non_english(leaf: &str) -> Option<Analysis> {
    let mut best_n: i64 = -1;
    let mut best: Option<Analysis> = None;
    for line in leaf.split('\n') {
        // A line this short cannot carry the four tokens or ten letters the branches
        // below need, so it is skipped before the cap, strip and full analysis pass.
        if line.chars().count() < 7 {
            continue;
        }
        let sample: String = line.chars().take(MAX_CHARS).collect();
        if sample.trim().is_empty() || code_line(&sample) {
            continue;
        }
        let det = analyse_text(&sample);
        if det.is_english {
            continue;
        }
        let n_alpha = i64_of(sample.chars().filter(|c| c.is_alphabetic()).count());
        match det.language.as_deref() {
            Some(lang) if lang != "en" => {
                if named_prose_language(&sample).is_none() {
                    continue;
                }
            }
            _ if det.script != "latin" && det.script != "unknown" => {
                if non_latin_words(&sample).is_empty() || n_alpha < NON_LATIN_MIN_LETTERS {
                    continue;
                }
            }
            _ => {
                if !(det.language_undecided
                    && det.diacritic_rate >= NON_EN_DIACRITIC_RATE
                    && word_runs(&sample).len() >= 4)
                {
                    continue;
                }
            }
        }
        if n_alpha > best_n {
            best_n = n_alpha;
            best = Some(det);
        }
    }
    best
}

/// Full detection result for a state (laya's `analyse`).
///
/// String values are what get read. When a state has several, one non-English value is
/// enough: joining every value into one window let a long English note fill the 4000
/// characters and outvote a short German message, which was then sent to the English
/// checkpoint. The cost is lopsided — English sent to multilingual loses a few points,
/// the reverse loses calibration.
///
/// ```
/// # use serde_json::json;
/// let state = json!({"note": "Please check this.",
///                    "message": "Mein Konto wurde zweimal belastet"});
/// let det = lumina::lang::analyse(&state);
/// assert!(!det.is_english);
/// assert_eq!(det.language.as_deref(), Some("de"));
/// assert!(det.mixed_segment.is_some());
/// ```
#[must_use]
pub fn analyse(state: &Value) -> Analysis {
    let mut result = analyse_text(&state_text(state, MAX_CHARS));
    if result.script == "latin" && result.is_english {
        let leaves = iter_text(state);
        // A single line has no other part to be outvoted by, and was just read whole.
        if leaves.len() > 1 || leaves.iter().any(|leaf| leaf.contains('\n')) {
            if let Some((lang, mixed)) = non_english_segment(state, MAX_CHARS) {
                result.language = Some(lang);
                result.is_english = false;
                result.language_undecided = false;
                result.mixed_segment = Some(mixed);
            }
        }
    }
    // A plain string was just read whole; a structured state can still hide a message
    // past the segment cap, or in a script `latin_profile` does not name.
    if state.is_string() || state.is_null() || !result.is_english {
        return result;
    }
    let mut best_n: i64 = -1;
    let mut best: Option<Analysis> = None;
    for leaf in iter_text(state) {
        let Some(det) = leaf_non_english(&leaf) else {
            continue;
        };
        let n_alpha = i64_of(
            leaf.chars()
                .take(MAX_CHARS)
                .filter(|c| c.is_alphabetic())
                .count(),
        );
        if n_alpha > best_n {
            best_n = n_alpha;
            best = Some(det);
        }
    }
    if let Some(best) = best {
        result.language = best.language;
        result.is_english = false;
        result.language_undecided = best.language_undecided;
    }
    result
}

/// True when the English checkpoint can be expected to read this state.
#[must_use]
pub fn is_english(state: &Value) -> bool {
    analyse(state).is_english
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
    fn detect_script_basic() {
        assert_eq!(detect_script("Hello world"), "latin");
        assert_eq!(detect_script("Привет мир"), "cyrillic");
        assert_eq!(detect_script("こんにちは"), "kana");
        assert_eq!(detect_script("你好"), "han");
        assert_eq!(detect_script("मुझसे मार्च"), "devanagari");
        assert_eq!(detect_script("12345 !!!"), "unknown");
    }

    #[test]
    fn english_text_routes_english() {
        let det = analyse(&json!("I was charged twice"));
        assert_eq!(det.script, "latin");
        assert!(det.is_english);
        assert_eq!(det.language.as_deref(), Some("en"));
    }

    #[test]
    fn readme_quickstart_state_is_english() {
        let state = "Hi, we were billed twice for March. Please refund the duplicate \
                     today or we will cancel our plan.";
        let det = analyse(&json!(state));
        assert!(det.is_english, "det = {det:?}");
        assert_eq!(det.language.as_deref(), Some("en"));
    }

    #[test]
    fn hindi_is_devanagari() {
        let state = "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।";
        let det = analyse(&json!(state));
        assert_eq!(det.script, "devanagari");
        assert!(!det.is_english);
    }

    #[test]
    fn spanish_routes_multilingual_on_diacritics() {
        // Only shared function words match (la, se, que), so no language is named; the
        // two accented ó are what make it unsafe for the English checkpoint.
        let det = analyse(&json!(
            "La aplicación se cierra cada vez que abro la configuración."
        ));
        assert_eq!(det.script, "latin");
        assert!(!det.is_english);
        assert_eq!(det.language, None);
        assert!(det.diacritic_rate > 0.02 && det.diacritic_rate < 0.06);
    }

    #[test]
    fn spanish_named_language() {
        let det = analyse(&json!("El pedido llegó roto y quiero que lo cambien"));
        assert_eq!(det.language.as_deref(), Some("es"));
        assert!(!det.is_english);
    }

    #[test]
    fn german_latin_multilingual() {
        let det = analyse(&json!("Mein Konto wurde zweimal belastet"));
        assert_eq!(det.script, "latin");
        assert_eq!(det.language.as_deref(), Some("de"));
        assert!(!det.is_english);
    }

    #[test]
    fn portuguese_unaccented() {
        // The unaccented Brazilian support words are the only evidence once accents are
        // stripped (laya #172); without them this read as English.
        let det = analyse(&json!("Voce pode me mandar a nota fiscal?"));
        assert_eq!(det.language.as_deref(), Some("pt"));
        assert!(!det.is_english);
    }

    #[test]
    fn english_rescue_with_loanword() {
        // One accented loanword must not pull otherwise plain English off English.
        let det = analyse(&json!("I have a café question"));
        assert!(det.is_english, "det = {det:?}");
        assert_eq!(det.language.as_deref(), Some("en"));
    }

    #[test]
    fn short_english_stays_english() {
        // Too few words to identify anything: undecided, but no non-English letters
        // either, so the default (English checkpoint) keeps it.
        let det = analyse(&json!("Refund please"));
        assert_eq!(det.language, None);
        assert!(det.is_english);
        assert!(det.language_undecided);
    }

    #[test]
    fn identifiers_are_not_prose() {
        // Dotted and @-joined tokens would collide with real function words (`com` is
        // Portuguese "with"), so they are replaced before word extraction.
        let det = analyse(&json!("user@acme.com github.com v1.2.3"));
        assert_eq!(det.language, None);
        assert!(det.is_english);
    }

    #[test]
    fn mixed_segment_flagged() {
        let state = json!({"note": "Please check this.",
                           "message": "Mein Konto wurde zweimal belastet"});
        let det = analyse(&state);
        assert!(!det.is_english);
        assert_eq!(det.language.as_deref(), Some("de"));
        assert_eq!(
            det.mixed_segment.as_deref(),
            Some("Mein Konto wurde zweimal belastet")
        );
        assert!(!det.language_undecided);
    }

    #[test]
    fn code_line_is_not_a_foreign_segment() {
        let state = json!({"msg": "Please check this.\nos.path = round(el, 2)"});
        let det = analyse(&state);
        assert!(det.is_english, "det = {det:?}");
        assert_eq!(det.mixed_segment, None);
    }

    #[test]
    fn non_latin_plurality_override() {
        // Latin letters are the plurality of named scripts, but the non-Latin share is
        // past the flip threshold, so the state routes as han.
        let det = analyse(&json!("ab 请取消订阅 ab"));
        assert_eq!(det.script, "han");
        assert!(!det.is_english);
    }

    #[test]
    fn capitalised_non_latin_name_is_not_a_word() {
        // A combining mark must not split a capitalised name into lowercase runs.
        let words = non_latin_words("Влади́мир Петрович");
        assert!(words.is_empty(), "words = {words:?}");
    }

    #[test]
    fn state_text_respects_char_budget() {
        let long = "x".repeat(5000);
        let text = state_text(&json!(long), MAX_CHARS);
        assert_eq!(text.chars().count(), MAX_CHARS);
        let two = json!(["aaaa", "bbbb"]);
        assert_eq!(state_text(&two, MAX_CHARS), "aaaa bbbb");
    }

    #[test]
    fn python_round_half_even() {
        assert_eq!(round0(0.5), 0);
        assert_eq!(round0(1.5), 2);
        assert_eq!(round0(2.5), 2);
        assert_eq!(round0(9.5), 10);
        assert!((round4(1.0 - 1.0 / 3.0) - 0.6667).abs() < 1e-12);
    }

    #[test]
    fn analysis_serializes_in_laya_key_order() {
        let value = serde_json::to_value(analyse(&json!("I was charged twice"))).unwrap();
        let keys: Vec<&str> = value
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            vec![
                "script",
                "script_profile",
                "language",
                "is_english",
                "language_undecided",
                "diacritic_rate",
                "non_latin_fraction",
                "mixed_segment"
            ]
        );
    }
}
