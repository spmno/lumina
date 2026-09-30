//! Python-compatible JSON serialization (matching `json.dumps` defaults).

use std::io::Write;

use serde::Serialize;
use serde_json::Serializer;
use serde_json::ser::Formatter;

use crate::{LuminaError, Result};

/// Formats values with `", "` between elements and `": "` after keys, matching Python's
/// `json.dumps` default separators when no indent is requested.
#[derive(Clone, Copy, Default)]
struct PythonFormatter;

impl Formatter for PythonFormatter {
    #[inline]
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + Write,
    {
        writer.write_all(b": ")
    }
}

/// Serialize `v` the way Python's `json.dumps(v, ensure_ascii=False)` does, modulo
/// exotic-float spellings (e.g. `1e-05` vs `1e-5`), which do not occur in laya states.
///
/// # Errors
/// Returns [`LuminaError::Json`] when `v` cannot be serialized, or
/// [`LuminaError::Invalid`] on non-UTF-8 output (both are impossible in practice for
/// values that round-tripped through serde_json).
pub fn to_python_json(v: &serde_json::Value) -> Result<String> {
    let mut buf = Vec::new();
    let mut ser = Serializer::with_formatter(&mut buf, PythonFormatter);
    v.serialize(&mut ser)?;
    String::from_utf8(buf).map_err(|e| LuminaError::Invalid(e.to_string()))
}

/// Python's `type(x).__name__` for a JSON value, for error messages.
#[must_use]
pub fn py_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "NoneType",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(n) => {
            if n.is_f64() {
                "float"
            } else {
                "int"
            }
        }
        serde_json::Value::String(_) => "str",
        serde_json::Value::Array(_) => "list",
        serde_json::Value::Object(_) => "dict",
    }
}

/// Python's `str(x)` for a scalar: `"True"`, `"3"`, `"3.5"`, strings as-is, `"None"`.
#[must_use]
pub fn py_str(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Bool(b) => {
            if *b {
                "True".to_owned()
            } else {
                "False".to_owned()
            }
        }
        serde_json::Value::Null => "None".to_owned(),
        other => other.to_string(),
    }
}

/// A Python `repr()` approximation for error messages: `'choice'`, `True`, `None`,
/// `['a', 1]`, `{'k': 1}`. Exotic float spellings (`1e-05` vs `1e-5`) differ.
#[must_use]
pub fn py_repr(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "None".to_owned(),
        serde_json::Value::Bool(b) => {
            if *b {
                "True".to_owned()
            } else {
                "False".to_owned()
            }
        }
        serde_json::Value::String(s) => str_repr(s),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Array(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        serde_json::Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, val)| format!("{}: {}", str_repr(k), py_repr(val)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn str_repr(s: &str) -> String {
    if s.contains('\'') && !s.contains('"') {
        format!("\"{s}\"")
    } else {
        format!("'{s}'")
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

    #[test]
    fn matches_python_dumps_spacing() {
        assert_eq!(
            super::to_python_json(&json!({"a": 1, "b": 2})).unwrap(),
            r#"{"a": 1, "b": 2}"#
        );
        assert_eq!(
            super::to_python_json(&json!([1, 2, 3])).unwrap(),
            "[1, 2, 3]"
        );
        assert_eq!(
            super::to_python_json(&json!({"nested": {"k": "v"}})).unwrap(),
            r#"{"nested": {"k": "v"}}"#
        );
        assert_eq!(super::to_python_json(&json!({})).unwrap(), "{}");
        assert_eq!(super::to_python_json(&json!([])).unwrap(), "[]");
        assert_eq!(
            super::to_python_json(&json!({"text": "héllo wörld"})).unwrap(),
            r#"{"text": "héllo wörld"}"#
        );
    }

    #[test]
    fn python_scalar_conversions() {
        use super::{py_repr, py_str, py_type_name};
        assert_eq!(py_type_name(&json!(null)), "NoneType");
        assert_eq!(py_type_name(&json!(1)), "int");
        assert_eq!(py_type_name(&json!(1.5)), "float");
        assert_eq!(py_type_name(&json!([])), "list");
        assert_eq!(py_str(&json!(true)), "True");
        assert_eq!(py_str(&json!(3.0)), "3.0");
        assert_eq!(py_str(&json!(null)), "None");
        assert_eq!(py_repr(&json!("a'b")), "\"a'b\"");
        assert_eq!(py_repr(&json!(["a", 1])), "['a', 1]");
        assert_eq!(py_repr(&json!({"k": 1})), "{'k': 1}");
    }
}
