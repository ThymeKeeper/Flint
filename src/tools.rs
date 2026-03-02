use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

use crate::claude::{
    ApiMessage, ClaudeRequest, ContentBlock, LlmClient, MessageContent, ToolDefinition,
};

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
    max_tokens: usize,
}

impl ToolExecutor {
    pub fn new(llm: Arc<dyn LlmClient>, max_tokens: usize) -> Self {
        Self { llm, max_tokens }
    }

    /// Returns the 4 base tools plus spawn_subagent (5 total).
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = tool_definitions();
        defs.push(spawn_subagent_definition());
        defs
    }

    pub async fn execute(&self, name: &str, input: &Value) -> String {
        match name {
            "spawn_subagent" => self.spawn_subagent(input).await,
            _ => execute_tool(name, input).await,
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

    #[test]
    fn test_executor_includes_spawn_subagent() {
        use crate::claude::mock::MockLlm;
        let llm = Arc::new(MockLlm::new(vec![]));
        let executor = ToolExecutor::new(llm, 1000);
        let defs = executor.tool_definitions();
        assert_eq!(defs.len(), 5);
        let names: Vec<_> = defs.iter().map(|d| d.name).collect();
        assert!(names.contains(&"spawn_subagent"));
    }

    #[tokio::test]
    async fn test_spawn_subagent_returns_mock_text() {
        use crate::claude::mock::MockLlm;
        let llm = Arc::new(MockLlm::new(vec!["sub-agent result".to_string()]));
        let executor = ToolExecutor::new(llm, 1000);
        let input = serde_json::json!({"task": "summarise /etc/hostname"});
        let result = executor.execute("spawn_subagent", &input).await;
        assert_eq!(result, "sub-agent result");
    }

    #[tokio::test]
    async fn test_spawn_subagent_missing_task() {
        use crate::claude::mock::MockLlm;
        let llm = Arc::new(MockLlm::new(vec![]));
        let executor = ToolExecutor::new(llm, 1000);
        let input = serde_json::json!({});
        let result = executor.execute("spawn_subagent", &input).await;
        assert_eq!(result, "Error: missing 'task' field");
    }
}
