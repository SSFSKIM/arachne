//! Just enough JSON writing for the tools' reports.

use crate::{AdapterSummary, Pin};

/// `value` as a JSON string.
pub fn string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 => out.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A finite number, or `null`.
pub fn number(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "null".into()
    }
}

/// An object of already-written values, in the given order.
pub fn object(fields: &[(&str, String)]) -> String {
    let fields: Vec<String> = fields
        .iter()
        .map(|(name, value)| format!("{}: {value}", string(name)))
        .collect();
    format!("{{{}}}", fields.join(", "))
}

/// An array of already-written values, one to a line.
pub fn array(values: &[String]) -> String {
    if values.is_empty() {
        return "[]".into();
    }
    format!("[\n  {}\n]", values.join(",\n  "))
}

pub fn pin(pin: &Pin) -> String {
    object(&[
        ("adapter", string(&pin.adapter)),
        ("backend", string(&pin.backend)),
        ("driver", string(&pin.driver)),
        ("wgpu", string(&pin.wgpu)),
    ])
}

pub fn adapter(adapter: &AdapterSummary) -> String {
    object(&[
        ("name", string(&adapter.name)),
        ("backend", string(&adapter.backend)),
        ("device_type", string(&adapter.device_type)),
        ("driver", string(&adapter.driver)),
        ("driver_info", string(&adapter.driver_info)),
        ("vendor", adapter.vendor.to_string()),
        ("device", adapter.device.to_string()),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_numbers_and_objects_are_written_as_json() {
        assert_eq!(string("a \"b\" \\ c\n\u{1}"), r#""a \"b\" \\ c\n\u0001""#);
        assert_eq!(number(1.5), "1.5");
        assert_eq!(number(f64::NAN), "null");
        assert_eq!(
            object(&[("x", number(2.0)), ("y", string("z"))]),
            r#"{"x": 2, "y": "z"}"#
        );
        assert_eq!(array(&[]), "[]");
        assert_eq!(array(&["1".into(), "2".into()]), "[\n  1,\n  2\n]");
    }
}
