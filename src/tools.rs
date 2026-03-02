use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

use crate::claude::{
    ApiMessage, ClaudeRequest, ContentBlock, LlmClient, MessageContent, ToolDefinition,
};
use crate::memory::MemoryManager;

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "shell_exec",
            description: "Execute a shell command on the host machine. Returns exit code and output. Ask the user before running destructive commands (rm, rmdir, dd, mkfs, etc.).",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }),
        },
        ToolDefinition {
            name: "file_read",
            description: "Read the contents of a file from the host filesystem.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file"
                    }
                },
                "required": ["path"]
            }),
        },
        ToolDefinition {
            name: "file_write",
            description: "Write content to a file on the host filesystem. Returns PERMISSION_REQUIRED if the file already exists or the path is a system path (/etc, /usr, /bin, etc.), unless force=true is passed. Use force=true only after the user has confirmed.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Set to true to overwrite existing files or write to system paths after user confirmation"
                    }
                },
                "required": ["path", "content"]
            }),
        },
        ToolDefinition {
            name: "web_fetch",
            description: "Fetch the contents of a URL over HTTP/HTTPS. Returns the HTTP status and response body.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch"
                    }
                },
                "required": ["url"]
            }),
        },
    ]
}

// ---------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------

pub async fn execute_tool(name: &str, input: &Value) -> String {
    match name {
        "shell_exec" => shell_exec(input).await,
        "file_read" => file_read(input),
        "file_write" => file_write(input),
        "web_fetch" => web_fetch(input).await,
        _ => format!("Unknown tool: {name}"),
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

async fn shell_exec(input: &Value) -> String {
    let command = match input["command"].as_str() {
        Some(c) => c,
        None => return "Error: missing 'command' field".to_string(),
    };

    let result = timeout(
        Duration::from_secs(30),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output(),
    )
    .await;

    match result {
        Err(_) => "Error: command timed out after 30 seconds".to_string(),
        Ok(Err(e)) => format!("Error: failed to execute command: {e}"),
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);
            if stderr.is_empty() {
                format!("Exit: {exit_code}\n{stdout}")
            } else {
                format!("Exit: {exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}")
            }
        }
    }
}

fn file_read(input: &Value) -> String {
    let path = match input["path"].as_str() {
        Some(p) => p,
        None => return "Error: missing 'path' field".to_string(),
    };

    match std::fs::read_to_string(path) {
        Ok(content) => {
            const MAX: usize = 100_000;
            if content.len() > MAX {
                format!("{}... [truncated at {MAX} chars]", &content[..MAX])
            } else {
                content
            }
        }
        Err(e) => format!("Error reading file: {e}"),
    }
}

fn file_write(input: &Value) -> String {
    let path = match input["path"].as_str() {
        Some(p) => p,
        None => return "Error: missing 'path' field".to_string(),
    };
    let content = match input["content"].as_str() {
        Some(c) => c,
        None => return "Error: missing 'content' field".to_string(),
    };
    let force = input["force"].as_bool().unwrap_or(false);

    if !force {
        if is_system_path(path) {
            return format!(
                "PERMISSION_REQUIRED: '{path}' is a system path (/etc, /usr, /bin, etc.). Please confirm you want to write here, then retry with force=true."
            );
        }
        if Path::new(path).exists() {
            return format!(
                "PERMISSION_REQUIRED: '{path}' already exists. Please confirm you want to overwrite it, then retry with force=true."
            );
        }
    }

    match std::fs::write(path, content) {
        Ok(()) => format!("Successfully wrote {} bytes to '{path}'", content.len()),
        Err(e) => format!("Error writing file: {e}"),
    }
}

async fn web_fetch(input: &Value) -> String {
    let url = match input["url"].as_str() {
        Some(u) => u,
        None => return "Error: missing 'url' field".to_string(),
    };

    let client = reqwest::Client::new();
    match client.get(url).send().await {
        Err(e) => format!("Error fetching URL: {e}"),
        Ok(resp) => {
            let status = resp.status();
            match resp.text().await {
                Err(e) => format!("Error reading response body: {e}"),
                Ok(body) => {
                    const MAX: usize = 50_000;
                    let truncated = if body.len() > MAX {
                        format!("{}... [truncated at {MAX} chars]", &body[..MAX])
                    } else {
                        body
                    };
                    format!("HTTP {status}\n{truncated}")
                }
            }
        }
    }
}

fn is_system_path(path: &str) -> bool {
    const SYSTEM_PREFIXES: &[&str] = &[
        "/etc/", "/usr/", "/bin/", "/sbin/", "/boot/", "/lib/", "/lib64/", "/sys/", "/proc/",
    ];
    SYSTEM_PREFIXES.iter().any(|prefix| path.starts_with(prefix))
}

// ---------------------------------------------------------------------------
// ToolExecutor
// ---------------------------------------------------------------------------

fn memory_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "memory_search",
            description: "Search your long-term memory by semantic query. Returns a JSON array \
                of matching memories with their IDs, content, kind, importance, pinned status, \
                and creation date. Use this to find outdated memories before updating or deleting them.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }),
        },
        ToolDefinition {
            name: "memory_update",
            description: "Update an existing memory by ID. You can change its content \
                (the embedding is automatically recomputed), adjust its importance, or \
                pin/unpin it. Use this when a memory is stale but the topic is still relevant — \
                e.g. a server IP address changed, or a preference has shifted.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The memory ID (from memory_search)"
                    },
                    "content": {
                        "type": "string",
                        "description": "New content to replace the existing content"
                    },
                    "importance": {
                        "type": "number",
                        "description": "New importance score between 0 and 1"
                    },
                    "pinned": {
                        "type": "boolean",
                        "description": "Pin (true) or unpin (false) the memory"
                    }
                },
                "required": ["id"]
            }),
        },
        ToolDefinition {
            name: "memory_delete",
            description: "Permanently delete a memory by ID. Use this when information is \
                fully obsolete and should not be recalled — e.g. a server that no longer exists, \
                a preference that is no longer valid, or a one-off fact that is no longer true.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The memory ID to delete (from memory_search)"
                    }
                },
                "required": ["id"]
            }),
        },
    ]
}

fn spawn_subagent_definition() -> ToolDefinition {
    ToolDefinition {
        name: "spawn_subagent",
        description: "Spawn a focused sub-agent with an isolated context window to handle a \
            self-contained task, preserving the main context window. The sub-agent has \
            shell_exec, file_read, file_write, and web_fetch available but cannot spawn \
            further sub-agents. Use this for context-heavy tasks (e.g. processing large files, \
            multi-step research) where accumulating intermediate tool messages would bloat the \
            main context.",
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The complete self-contained task for the sub-agent to perform"
                },
                "context": {
                    "type": "string",
                    "description": "Optional extra context (file paths, constraints, background)"
                }
            },
            "required": ["task"]
        }),
    }
}

pub struct ToolExecutor {
    llm:        Arc<dyn LlmClient>,
    memory:     Arc<MemoryManager>,
    max_tokens: usize,
}

impl ToolExecutor {
    pub fn new(llm: Arc<dyn LlmClient>, memory: Arc<MemoryManager>, max_tokens: usize) -> Self {
        Self { llm, memory, max_tokens }
    }

    /// Returns the 4 base tools plus memory tools and spawn_subagent (8 total).
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = tool_definitions();
        defs.extend(memory_tool_definitions());
        defs.push(spawn_subagent_definition());
        defs
    }

    pub async fn execute(&self, name: &str, input: &Value) -> String {
        match name {
            "memory_search" => self.memory_search(input).await,
            "memory_update" => self.memory_update(input).await,
            "memory_delete" => self.memory_delete(input).await,
            "spawn_subagent" => self.spawn_subagent(input).await,
            _ => execute_tool(name, input).await,
        }
    }

    // -----------------------------------------------------------------------
    // Memory tool implementations
    // -----------------------------------------------------------------------

    async fn memory_search(&self, input: &Value) -> String {
        let query = match input["query"].as_str() {
            Some(q) => q,
            None => return "Error: missing 'query' field".to_string(),
        };
        let top_k = input["top_k"].as_u64().map(|n| n as usize);

        match self.memory.search(query, top_k).await {
            Err(e) => format!("Error searching memories: {e:#}"),
            Ok(refs) => {
                if refs.is_empty() {
                    return "[]".to_string();
                }
                let items: Vec<serde_json::Value> = refs
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "id": r.id,
                            "content": r.content,
                            "kind": r.kind.as_str(),
                            "importance": r.importance,
                            "pinned": r.pinned,
                            "created_at": r.created_at.to_rfc3339(),
                            "similarity": r.similarity,
                        })
                    })
                    .collect();
                serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string())
            }
        }
    }

    async fn memory_update(&self, input: &Value) -> String {
        let id = match input["id"].as_str() {
            Some(i) => i,
            None => return "Error: missing 'id' field".to_string(),
        };
        let new_content = input["content"].as_str();
        let new_importance = input["importance"].as_f64();
        let new_pinned = input["pinned"].as_bool();

        if new_content.is_none() && new_importance.is_none() && new_pinned.is_none() {
            return "Error: at least one of 'content', 'importance', or 'pinned' must be provided".to_string();
        }

        match self.memory.update(id, new_content, new_importance, new_pinned).await {
            Err(e) => format!("Error updating memory: {e:#}"),
            Ok(false) => format!("Error: no memory found with id '{id}'"),
            Ok(true) => format!("Memory '{id}' updated successfully"),
        }
    }

    async fn memory_delete(&self, input: &Value) -> String {
        let id = match input["id"].as_str() {
            Some(i) => i,
            None => return "Error: missing 'id' field".to_string(),
        };
        match self.memory.delete(&[id.to_string()]).await {
            Err(e) => format!("Error deleting memory: {e:#}"),
            Ok(()) => format!("Memory '{id}' deleted"),
        }
    }

    async fn spawn_subagent(&self, input: &Value) -> String {
        let task = match input["task"].as_str() {
            Some(t) => t,
            None => return "Error: missing 'task' field".to_string(),
        };
        let context = input["context"].as_str().unwrap_or("");

        // Sub-agent gets base 4 tools only — no recursion.
        let sub_defs = tool_definitions();

        let system = "You are a focused sub-agent. Complete the assigned task thoroughly using \
            the available tools (shell_exec, file_read, file_write, web_fetch). \
            Your final response is the ONLY output that reaches the main agent — \
            include all relevant results, findings, and output directly in your reply. \
            Do not spawn further sub-agents.";

        let user_content = if context.is_empty() {
            task.to_string()
        } else {
            format!("{task}\n\nAdditional context:\n{context}")
        };

        let mut messages: Vec<ApiMessage> = vec![ApiMessage {
            role: "user".to_string(),
            content: MessageContent::Text(user_content),
        }];

        loop {
            let resp = match self
                .llm
                .complete_with_tools(ClaudeRequest {
                    system,
                    messages: messages.clone(),
                    max_tokens: self.max_tokens,
                    tools: &sub_defs,
                })
                .await
            {
                Ok(r) => r,
                Err(e) => return format!("Sub-agent error: {e:#}"),
            };

            if resp.tool_calls.is_empty() {
                return resp.text;
            }

            // Assistant turn
            messages.push(ApiMessage {
                role: "assistant".to_string(),
                content: MessageContent::Blocks(resp.raw_blocks),
            });

            // Execute tools and collect results
            let mut results = Vec::new();
            for tc in &resp.tool_calls {
                let output = execute_tool(&tc.name, &tc.input).await;
                results.push(ContentBlock::ToolResult {
                    tool_use_id: tc.id.clone(),
                    content: output,
                });
            }

            // User turn: tool results
            messages.push(ApiMessage {
                role: "user".to_string(),
                content: MessageContent::Blocks(results),
            });
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
    fn test_is_system_path() {
        assert!(is_system_path("/etc/hostname"));
        assert!(is_system_path("/usr/local/bin/foo"));
        assert!(is_system_path("/proc/cpuinfo"));
        assert!(is_system_path("/lib/libc.so"));
        assert!(!is_system_path("/home/user/file.txt"));
        assert!(!is_system_path("/tmp/test.txt"));
        assert!(!is_system_path("/var/log/app.log"));
    }

    #[test]
    fn test_file_write_permission_required_system_path() {
        let input = serde_json::json!({"path": "/etc/hosts", "content": "test"});
        let result = file_write(&input);
        assert!(result.starts_with("PERMISSION_REQUIRED"));
        assert!(result.contains("/etc/hosts"));
    }

    #[test]
    fn test_file_write_force_skips_permission_check() {
        // /tmp/clawd_test_nonexistent should not exist; force=true should attempt write
        let path = "/tmp/clawd_tools_test_write.txt";
        let _ = std::fs::remove_file(path); // clean up if exists
        let input = serde_json::json!({"path": path, "content": "hello", "force": true});
        let result = file_write(&input);
        assert!(result.contains("Successfully wrote"), "got: {result}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_file_write_no_overwrite_without_force() {
        let path = "/tmp/clawd_tools_test_overwrite.txt";
        std::fs::write(path, "existing").unwrap();
        let input = serde_json::json!({"path": path, "content": "new"});
        let result = file_write(&input);
        assert!(result.starts_with("PERMISSION_REQUIRED"));
        assert!(result.contains(path));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_tool_definitions_count() {
        let defs = tool_definitions();
        assert_eq!(defs.len(), 4);
        let names: Vec<_> = defs.iter().map(|d| d.name).collect();
        assert!(names.contains(&"shell_exec"));
        assert!(names.contains(&"file_read"));
        assert!(names.contains(&"file_write"));
        assert!(names.contains(&"web_fetch"));
    }

    #[tokio::test]
    async fn test_execute_unknown_tool() {
        let result = execute_tool("unknown_tool", &serde_json::json!({})).await;
        assert!(result.contains("Unknown tool"));
    }

    #[tokio::test]
    async fn test_shell_exec_basic() {
        let input = serde_json::json!({"command": "echo hello"});
        let result = shell_exec(&input).await;
        assert!(result.contains("Exit: 0"));
        assert!(result.contains("hello"));
    }

    #[test]
    fn test_file_read_nonexistent() {
        let input = serde_json::json!({"path": "/tmp/clawd_does_not_exist_xyz.txt"});
        let result = file_read(&input);
        assert!(result.starts_with("Error reading file"));
    }

    async fn test_executor() -> ToolExecutor {
        use crate::claude::mock::MockLlm;
        use crate::config::MemoryConfig;
        use crate::embeddings::mock::MockEmbeddingClient;
        use std::path::Path;
        let llm = Arc::new(MockLlm::new(vec![]));
        let embedder = Arc::new(MockEmbeddingClient::new(4));
        let memory = Arc::new(
            MemoryManager::new(Path::new(":memory:"), embedder, MemoryConfig {
                max_memories: 100,
                top_k_retrieval: 5,
                importance_decay_days: 30.0,
                min_importance_to_keep: 0.1,
                ttl_days_episodic: 90.0,
            }, 4)
            .await
            .unwrap(),
        );
        ToolExecutor::new(llm, memory, 1000)
    }

    #[tokio::test]
    async fn test_executor_includes_spawn_subagent() {
        let executor = test_executor().await;
        let defs = executor.tool_definitions();
        assert_eq!(defs.len(), 8);
        let names: Vec<_> = defs.iter().map(|d| d.name).collect();
        assert!(names.contains(&"spawn_subagent"));
        assert!(names.contains(&"memory_search"));
        assert!(names.contains(&"memory_update"));
        assert!(names.contains(&"memory_delete"));
    }

    #[tokio::test]
    async fn test_spawn_subagent_returns_mock_text() {
        use crate::claude::mock::MockLlm;
        use crate::config::MemoryConfig;
        use crate::embeddings::mock::MockEmbeddingClient;
        use std::path::Path;
        let llm = Arc::new(MockLlm::new(vec!["sub-agent result".to_string()]));
        let embedder = Arc::new(MockEmbeddingClient::new(4));
        let memory = Arc::new(
            MemoryManager::new(Path::new(":memory:"), embedder, MemoryConfig {
                max_memories: 100,
                top_k_retrieval: 5,
                importance_decay_days: 30.0,
                min_importance_to_keep: 0.1,
                ttl_days_episodic: 90.0,
            }, 4)
            .await
            .unwrap(),
        );
        let executor = ToolExecutor::new(llm, memory, 1000);
        let input = serde_json::json!({"task": "summarise /etc/hostname"});
        let result = executor.execute("spawn_subagent", &input).await;
        assert_eq!(result, "sub-agent result");
    }

    #[tokio::test]
    async fn test_spawn_subagent_missing_task() {
        let executor = test_executor().await;
        let input = serde_json::json!({});
        let result = executor.execute("spawn_subagent", &input).await;
        assert_eq!(result, "Error: missing 'task' field");
    }

    #[tokio::test]
    async fn test_memory_search_empty() {
        let executor = test_executor().await;
        let input = serde_json::json!({"query": "home server"});
        let result = executor.execute("memory_search", &input).await;
        assert_eq!(result, "[]");
    }

    #[tokio::test]
    async fn test_memory_update_missing_id() {
        let executor = test_executor().await;
        let input = serde_json::json!({"content": "new content"});
        let result = executor.execute("memory_update", &input).await;
        assert!(result.contains("missing 'id'"));
    }

    #[tokio::test]
    async fn test_memory_delete_unknown_id() {
        let executor = test_executor().await;
        let input = serde_json::json!({"id": "nonexistent-id"});
        // delete on a non-existent row is a no-op (returns ok)
        let result = executor.execute("memory_delete", &input).await;
        assert!(result.contains("deleted"));
    }
}
