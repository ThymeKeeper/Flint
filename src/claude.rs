use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::{debug, trace};

use crate::config::ClaudeConfig;
use crate::context::ConversationContext;

// ---------------------------------------------------------------------------
// Tool definition (here because it's a Claude API type)
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone)]
pub struct ToolDefinition {
    pub name:         &'static str,
    pub description:  &'static str,
    pub input_schema: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Message content types
// ---------------------------------------------------------------------------

/// Content of an API message: either a plain string or a list of typed blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// A typed content block used in outgoing messages and tool loop messages.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

// ---------------------------------------------------------------------------
// Trait for testability
// ---------------------------------------------------------------------------

/// Request structure passed to the LLM client.
pub struct ClaudeRequest<'a> {
    pub system: &'a str,
    pub messages: Vec<ApiMessage>,
    pub max_tokens: usize,
    /// Tool definitions to include. Pass `&[]` for tool-free completions.
    pub tools: &'a [ToolDefinition],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMessage {
    pub role: String,
    pub content: MessageContent,
}

/// A parsed tool call from the assistant response.
pub struct ToolUseBlock {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Full response from `complete_with_tools`.
pub struct LlmResponse {
    /// Concatenated text from all text blocks in the response.
    pub text: String,
    /// Tool calls requested by the assistant (empty when stop_reason is "end_turn").
    pub tool_calls: Vec<ToolUseBlock>,
    pub stop_reason: String,
    /// Full assistant content blocks for re-inserting into the message loop.
    pub raw_blocks: Vec<ContentBlock>,
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    /// Full response including tool calls (non-streaming).
    async fn complete_with_tools(&self, req: ClaudeRequest<'_>) -> Result<LlmResponse>;

    /// Convenience wrapper: returns only the text, ignoring tool calls.
    async fn complete(&self, req: ClaudeRequest<'_>) -> Result<String> {
        self.complete_with_tools(req).await.map(|r| r.text)
    }

    /// Streaming variant: calls `on_text` for each text delta as it arrives.
    /// Text that arrives before any tool_use block is streamed; text in rounds
    /// where the model calls tools is suppressed (we don't know which it is until
    /// the tool_use block starts, so text before the first tool_use IS printed).
    /// Default implementation ignores `on_text` and falls back to non-streaming.
    async fn complete_with_tools_streaming(
        &self,
        req: ClaudeRequest<'_>,
        _on_text: Arc<dyn Fn(String) + Send + Sync>,
    ) -> Result<LlmResponse> {
        self.complete_with_tools(req).await
    }
}

// ---------------------------------------------------------------------------
// Claude API client
// ---------------------------------------------------------------------------

pub struct ClaudeClient {
    api_key: String,
    config: ClaudeConfig,
    http: reqwest::Client,
}

impl ClaudeClient {
    pub fn new(api_key: String, config: ClaudeConfig) -> Self {
        Self {
            api_key,
            config,
            http: reqwest::Client::new(),
        }
    }

    /// Build the messages list from a ConversationContext for convenience.
    pub fn messages_from_context(ctx: &ConversationContext) -> Vec<ApiMessage> {
        ctx.messages()
            .iter()
            .map(|m| ApiMessage {
                role: m.role.as_str().to_string(),
                content: MessageContent::Text(m.content.clone()),
            })
            .collect()
    }

    /// Streaming completion. Returns the full assembled response text.
    /// Processes SSE events from the Anthropic streaming API.
    pub async fn complete_streaming(
        &self,
        system: &str,
        messages: Vec<ApiMessage>,
        max_tokens: Option<usize>,
    ) -> Result<String> {
        let max_tok = max_tokens.unwrap_or(self.config.max_tokens);

        let body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": max_tok,
            "system": system,
            "messages": messages,
            "stream": true,
        });

        debug!("Sending streaming request to Claude API (model={})", self.config.model);

        let response = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Claude API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            bail!("Claude API returned {status}: {body_text}");
        }

        let mut stream = response.bytes_stream();
        let mut full_text = String::new();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk: Bytes = chunk.context("Error reading stream chunk")?;
            let chunk_str = String::from_utf8_lossy(&chunk);
            buffer.push_str(&chunk_str);

            // Process complete SSE lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim_end_matches('\r').to_string();
                buffer = buffer[line_end + 1..].to_string();

                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        break;
                    }
                    if let Some(text) = parse_sse_content_delta(data) {
                        full_text.push_str(&text);
                        trace!("SSE delta: {}", truncate(&text, 40));
                    }
                }
            }
        }

        debug!(
            "Streaming complete: {} chars received",
            full_text.len()
        );
        Ok(full_text)
    }
}

#[async_trait]
impl LlmClient for ClaudeClient {
    async fn complete_with_tools(&self, req: ClaudeRequest<'_>) -> Result<LlmResponse> {
        let mut body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": req.messages,
        });
        if !req.tools.is_empty() {
            body["tools"] = serde_json::to_value(req.tools)
                .context("Failed to serialize tool definitions")?;
        }

        debug!("Sending request to Claude API (model={}, tools={})", self.config.model, req.tools.len());

        let response = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Claude API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            bail!("Claude API returned {status}: {body_text}");
        }

        let resp: ApiResponse = response
            .json()
            .await
            .context("Failed to parse Claude API response")?;

        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut raw_blocks = Vec::new();

        for block in &resp.content {
            match block.block_type.as_str() {
                "text" => {
                    if let Some(t) = &block.text {
                        text.push_str(t);
                        raw_blocks.push(ContentBlock::Text { text: t.clone() });
                    }
                }
                "tool_use" => {
                    if let (Some(id), Some(name), Some(input)) =
                        (&block.id, &block.name, &block.input)
                    {
                        raw_blocks.push(ContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });
                        tool_calls.push(ToolUseBlock {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        });
                    }
                }
                _ => {}
            }
        }

        debug!(
            "Claude response: {} chars, {} tool calls, stop_reason={}",
            text.len(),
            tool_calls.len(),
            resp.stop_reason
        );

        Ok(LlmResponse {
            text,
            tool_calls,
            stop_reason: resp.stop_reason,
            raw_blocks,
        })
    }

    async fn complete_with_tools_streaming(
        &self,
        req: ClaudeRequest<'_>,
        on_text: Arc<dyn Fn(String) + Send + Sync>,
    ) -> Result<LlmResponse> {
        let mut body = serde_json::json!({
            "model": self.config.model,
            "max_tokens": req.max_tokens,
            "system": req.system,
            "messages": req.messages,
            "stream": true,
        });
        if !req.tools.is_empty() {
            body["tools"] = serde_json::to_value(req.tools)
                .context("Failed to serialize tool definitions")?;
        }

        debug!(
            "Sending streaming request to Claude API (model={}, tools={})",
            self.config.model,
            req.tools.len()
        );

        let response = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Claude API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            bail!("Claude API returned {status}: {body_text}");
        }

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        // Per-block state (indexed by content_block index from the SSE events).
        let mut block_types:       Vec<String> = Vec::new(); // "text" | "tool_use"
        let mut block_texts:       Vec<String> = Vec::new();
        let mut block_tool_ids:    Vec<String> = Vec::new();
        let mut block_tool_names:  Vec<String> = Vec::new();
        let mut block_tool_inputs: Vec<String> = Vec::new(); // accumulated partial JSON

        // Once we see a tool_use block we stop streaming text to the callback.
        // Text that arrived before the tool_use block was already printed.
        let mut has_tool_use = false;
        let mut stop_reason = "end_turn".to_string();

        while let Some(chunk) = stream.next().await {
            let chunk: Bytes = chunk.context("Error reading stream chunk")?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim_end_matches('\r').to_string();
                buffer = buffer[line_end + 1..].to_string();

                let data = match line.strip_prefix("data: ") {
                    Some(d) if d != "[DONE]" => d.to_string(),
                    _ => continue,
                };

                let v: serde_json::Value = match serde_json::from_str(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                match v["type"].as_str().unwrap_or("") {
                    "content_block_start" => {
                        let idx = v["index"].as_u64().unwrap_or(0) as usize;
                        let block_type = v["content_block"]["type"].as_str().unwrap_or("").to_string();
                        while block_types.len() <= idx {
                            block_types.push(String::new());
                            block_texts.push(String::new());
                            block_tool_ids.push(String::new());
                            block_tool_names.push(String::new());
                            block_tool_inputs.push(String::new());
                        }
                        if block_type == "tool_use" {
                            has_tool_use = true;
                            block_tool_ids[idx] = v["content_block"]["id"]
                                .as_str().unwrap_or("").to_string();
                            block_tool_names[idx] = v["content_block"]["name"]
                                .as_str().unwrap_or("").to_string();
                        }
                        block_types[idx] = block_type;
                    }
                    "content_block_delta" => {
                        let idx = v["index"].as_u64().unwrap_or(0) as usize;
                        match v["delta"]["type"].as_str().unwrap_or("") {
                            "text_delta" => {
                                let text = v["delta"]["text"].as_str().unwrap_or("").to_string();
                                if idx < block_texts.len() {
                                    block_texts[idx].push_str(&text);
                                }
                                // Stream text only while no tool_use block has appeared.
                                if !has_tool_use {
                                    on_text(text);
                                }
                            }
                            "input_json_delta" => {
                                let partial = v["delta"]["partial_json"].as_str().unwrap_or("").to_string();
                                if idx < block_tool_inputs.len() {
                                    block_tool_inputs[idx].push_str(&partial);
                                }
                            }
                            _ => {}
                        }
                    }
                    "message_delta" => {
                        if let Some(sr) = v["delta"]["stop_reason"].as_str() {
                            stop_reason = sr.to_string();
                        }
                    }
                    _ => {}
                }
            }
        }

        // Assemble LlmResponse from accumulated block state.
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        let mut raw_blocks = Vec::new();

        for i in 0..block_types.len() {
            match block_types[i].as_str() {
                "text" => {
                    text.push_str(&block_texts[i]);
                    raw_blocks.push(ContentBlock::Text { text: block_texts[i].clone() });
                }
                "tool_use" => {
                    let input: serde_json::Value =
                        serde_json::from_str(&block_tool_inputs[i])
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                    raw_blocks.push(ContentBlock::ToolUse {
                        id: block_tool_ids[i].clone(),
                        name: block_tool_names[i].clone(),
                        input: input.clone(),
                    });
                    tool_calls.push(ToolUseBlock {
                        id: block_tool_ids[i].clone(),
                        name: block_tool_names[i].clone(),
                        input,
                    });
                }
                _ => {}
            }
        }

        debug!(
            "Streaming complete: {} chars, {} tool calls, stop_reason={stop_reason}",
            text.len(),
            tool_calls.len()
        );

        Ok(LlmResponse { text, tool_calls, stop_reason, raw_blocks })
    }
}

// ---------------------------------------------------------------------------
// API response types (private — only used for deserialization)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ApiRespBlock>,
    stop_reason: String,
}

#[derive(Deserialize)]
struct ApiRespBlock {
    #[serde(rename = "type")]
    block_type: String,
    // text block
    text: Option<String>,
    // tool_use block
    id: Option<String>,
    name: Option<String>,
    input: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

/// Parse a content_block_delta SSE event and extract the text delta.
/// Anthropic streaming format sends events like:
/// ```json
/// {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}
/// ```
fn parse_sse_content_delta(data: &str) -> Option<String> {
    let parsed: serde_json::Value = serde_json::from_str(data).ok()?;
    let event_type = parsed.get("type")?.as_str()?;

    if event_type == "content_block_delta" {
        let delta = parsed.get("delta")?;
        let delta_type = delta.get("type")?.as_str()?;
        if delta_type == "text_delta" {
            return delta.get("text")?.as_str().map(|s| s.to_string());
        }
    }

    None
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

// ---------------------------------------------------------------------------
// Mock for testing
// ---------------------------------------------------------------------------

pub mod mock {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    pub struct MockLlm {
        pub responses: Mutex<VecDeque<String>>,
    }

    impl MockLlm {
        pub fn new(responses: Vec<String>) -> Self {
            Self {
                responses: Mutex::new(VecDeque::from(responses)),
            }
        }
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn complete_with_tools(&self, _req: ClaudeRequest<'_>) -> Result<LlmResponse> {
            let text = {
                let mut queue = self.responses.lock().unwrap();
                queue
                    .pop_front()
                    .context("MockLlm: no more responses queued")?
            };
            Ok(LlmResponse {
                raw_blocks: vec![ContentBlock::Text { text: text.clone() }],
                text,
                tool_calls: vec![],
                stop_reason: "end_turn".to_string(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_sse_content_delta_text() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello world"}}"#;
        let result = parse_sse_content_delta(data);
        assert_eq!(result, Some("Hello world".to_string()));
    }

    #[test]
    fn test_parse_sse_content_delta_non_text() {
        let data = r#"{"type":"message_start","message":{"id":"msg_123"}}"#;
        let result = parse_sse_content_delta(data);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_sse_content_delta_invalid_json() {
        let result = parse_sse_content_delta("not json");
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_sse_content_delta_empty_text() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}"#;
        let result = parse_sse_content_delta(data);
        assert_eq!(result, Some(String::new()));
    }

    #[test]
    fn test_parse_sse_ping() {
        let data = r#"{"type":"ping"}"#;
        let result = parse_sse_content_delta(data);
        assert_eq!(result, None);
    }

    #[test]
    fn test_parse_sse_content_block_start() {
        let data = r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
        let result = parse_sse_content_delta(data);
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_mock_llm() {
        let mock = mock::MockLlm::new(vec!["response1".to_string(), "response2".to_string()]);
        let req = ClaudeRequest {
            system: "test",
            messages: vec![],
            max_tokens: 100,
            tools: &[],
        };
        let r1 = mock.complete(req).await.unwrap();
        assert_eq!(r1, "response1");

        let req2 = ClaudeRequest {
            system: "test",
            messages: vec![],
            max_tokens: 100,
            tools: &[],
        };
        let r2 = mock.complete(req2).await.unwrap();
        assert_eq!(r2, "response2");
    }

    #[test]
    fn test_message_content_text_serializes_as_string() {
        let content = MessageContent::Text("hello".to_string());
        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(json, r#""hello""#);
    }

    #[test]
    fn test_message_content_blocks_serializes_as_array() {
        let content = MessageContent::Blocks(vec![ContentBlock::Text {
            text: "hi".to_string(),
        }]);
        let json = serde_json::to_value(&content).unwrap();
        assert!(json.is_array());
    }

    #[test]
    fn test_content_block_tool_use_serializes_correctly() {
        let block = ContentBlock::ToolUse {
            id: "toolu_1".to_string(),
            name: "shell_exec".to_string(),
            input: serde_json::json!({"command": "ls"}),
        };
        let val = serde_json::to_value(&block).unwrap();
        assert_eq!(val["type"], "tool_use");
        assert_eq!(val["id"], "toolu_1");
        assert_eq!(val["name"], "shell_exec");
    }

    #[test]
    fn test_content_block_tool_result_serializes_correctly() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_1".to_string(),
            content: "Exit: 0\nfoo".to_string(),
        };
        let val = serde_json::to_value(&block).unwrap();
        assert_eq!(val["type"], "tool_result");
        assert_eq!(val["tool_use_id"], "toolu_1");
        assert_eq!(val["content"], "Exit: 0\nfoo");
    }
}
