// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! GLM-4.7 / GLM-5.x structural tag, matching xgrammar's builtin `glm_4_7` tag.

use std::sync::LazyLock;

use serde_json::{Value, json};

use super::builder::{ToolCallFormatBuildContext, resolve_tool_schema, resolve_tools_to_include};
use crate::tool_calling::ToolChoice;

const TOOL_CALL_BEGIN: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const THINK_BEGIN: &str = "<think>";
const THINK_END: &str = "</think>";
const ARG_MARKERS: [&str; 4] = ["<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"];

/// Every tool-call control token, banned for `tool_choice="none"`.
pub(crate) static BAN_TOKENS: LazyLock<Vec<String>> = LazyLock::new(|| {
    [TOOL_CALL_BEGIN, TOOL_CALL_END]
        .into_iter()
        .chain(ARG_MARKERS)
        .map(String::from)
        .collect()
});

/// glm_xml quotes values behind a top-level `$ref`, and a dangling `$ref` fails to compile.
fn glm_schema(mut schema: Value) -> Value {
    if has_unresolved_ref(&schema, &schema) {
        return json!(true);
    }
    let root = schema.clone();
    if let Some(props) = schema.get_mut("properties").and_then(Value::as_object_mut) {
        for prop in props.values_mut() {
            // Bounded so a ref cycle stays a `$ref`.
            for _ in 0..8 {
                let Some(target) = prop
                    .get("$ref")
                    .and_then(Value::as_str)
                    .and_then(|r| resolve_ref(&root, r))
                else {
                    break;
                };
                let mut inlined = target.clone();
                if let (Some(dst), Some(src)) = (inlined.as_object_mut(), prop.as_object()) {
                    for (k, v) in src.iter().filter(|(k, _)| *k != "$ref") {
                        dst.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                }
                *prop = inlined;
            }
        }
    }
    schema
}

fn resolve_ref<'a>(root: &'a Value, r: &str) -> Option<&'a Value> {
    root.pointer(r.strip_prefix('#')?)
}

fn has_unresolved_ref(root: &Value, v: &Value) -> bool {
    match v {
        Value::Object(map) => {
            map.get("$ref")
                .and_then(Value::as_str)
                .is_some_and(|r| resolve_ref(root, r).is_none())
                || map.values().any(|v| has_unresolved_ref(root, v))
        }
        Value::Array(items) => items.iter().any(|v| has_unresolved_ref(root, v)),
        _ => false,
    }
}

pub(crate) fn build_glm47(ctx: &ToolCallFormatBuildContext<'_>) -> anyhow::Result<Option<Value>> {
    let (tools, at_least_one) = resolve_tools_to_include(ctx)?;
    let mut tags: Vec<Value> = tools
        .into_iter()
        .map(|tool| {
            let schema = glm_schema(resolve_tool_schema(tool, ctx.strict_schema()));
            json!({
                "type": "tag",
                "begin": format!("{TOOL_CALL_BEGIN}{}", tool.name),
                // Declared order lets an out-of-order key swallow the closing markers.
                "content": {
                    "type": "json_schema",
                    "json_schema": schema,
                    "style": "glm_xml",
                    "any_order": true,
                },
                "end": TOOL_CALL_END,
            })
        })
        .collect();
    if tags.is_empty() {
        return Ok(None);
    }

    // A named call is the whole reply, so the grammar ends with it.
    let mut format = if matches!(ctx.tool_choice, ToolChoice::Named(_)) {
        tags.remove(0)
    } else {
        // `<tool_call>` stays allowed in free text as the trigger.
        let excludes: Vec<&str> = [THINK_BEGIN, THINK_END, TOOL_CALL_END]
            .into_iter()
            .chain(ARG_MARKERS)
            .collect();
        json!({
            "type": "triggered_tags",
            "triggers": [TOOL_CALL_BEGIN],
            "tags": tags,
            "excludes": excludes,
            "at_least_one": at_least_one,
            "stop_after_first": ctx.stop_after_first(),
        })
    };
    // The free text bans `</think>`, so a prompt-opened reasoning block is closed first.
    if ctx.starts_in_reasoning {
        let excludes: Vec<&str> = [THINK_BEGIN, THINK_END, TOOL_CALL_BEGIN, TOOL_CALL_END]
            .into_iter()
            .chain(ARG_MARKERS)
            .collect();
        let reasoning = json!({
            "type": "tag",
            "begin": "",
            "content": {"type": "any_text", "excludes": excludes},
            "end": THINK_END,
        });
        format = json!({"type": "sequence", "elements": [reasoning, format]});
    }
    Ok(Some(json!({"type": "structural_tag", "format": format})))
}
