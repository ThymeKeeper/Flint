use serde_json::Value;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

use crate::claude::{
    ApiMessage, ClaudeRequest, ContentBlock, LlmClient, MessageContent, ToolDefinition,
};
use crate::jobs::BackgroundJobStore;
use crate::memory::MemoryManager;
use crate::observer::AgentObserver;
use crate::signal::SignalClient;
use crate::skills::{SkillManager, ALLOWED_SKILL_TOOLS};
use crate::tasks::TaskManager;

pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "shell_exec",
            description: "Execute a shell command on the host machine. Returns exit code and output. Ask the user before running destructive commands (rm, rmdir, dd, mkfs, etc.).\n\nRULE: Use background=true by default. Only skip it (foreground) for commands that are guaranteed to finish in under ~5 seconds, such as: ls, pwd, echo, cat on a small file, grep, ps, df, uname, date, or similar status/inspection commands.\n\nEverything else — file copies, moves, downloads, installs, builds, scripts, network operations, or anything whose runtime is uncertain — MUST use background=true. When in doubt, use background=true.\n\nWhen background=true the command runs asynchronously; you receive a job ID immediately and are notified when it completes.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Default: true. Set to false ONLY for trivial read-only commands guaranteed to finish in <5 seconds (ls, cat, grep, ps, df, etc.). Everything else must run in the background."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Max seconds to wait (foreground default: 300; background default: 3600). The process is NOT killed on timeout — it continues running."
                    }
                },
                "required": ["command", "background"]
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

    // Default timeout: 5 minutes. Caller can override with "timeout_secs".
    // Note: tokio does NOT kill the child process when the future is dropped,
    // so a timeout means "still running in the background", not "cancelled".
    let timeout_secs = input["timeout_secs"].as_u64().unwrap_or(300);

    let result = timeout(
        Duration::from_secs(timeout_secs),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output(),
    )
    .await;

    match result {
        Err(_) => format!(
            "Command still running after {timeout_secs}s — it was NOT cancelled and continues \
             in the background. Do not assume failure. Verify the result directly (e.g. check \
             whether the expected file or output exists)."
        ),
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
            name: "memory_store",
            description: "Store a new memory in long-term memory. Use this when the user \
                explicitly asks you to remember something, or when you notice an important \
                fact worth preserving across sessions. Choose kind='semantic' for facts and \
                preferences, 'episodic' for events, 'procedural' for how-to knowledge. \
                Set pinned=true for critical facts the user explicitly asked you to remember.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The memory content to store"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural", "reflection"],
                        "description": "semantic for facts/preferences, episodic for events, procedural for how-to"
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score 0–1 (default: 0.8)"
                    },
                    "pinned": {
                        "type": "boolean",
                        "description": "Pin to prevent decay/pruning — use for facts the user explicitly asked you to remember"
                    }
                },
                "required": ["content"]
            }),
        },
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

fn skill_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "create_skill",
            description: "Create (or replace) a named sub-agent skill — a reusable profile \
                with a custom system prompt and tool set. When you spawn a sub-agent with \
                skill='name', it uses that profile instead of the default. \
                All sub-agents automatically inherit the user's principal context regardless \
                of which skill is used. \
                Allowed tools for skills: shell_exec, file_read, file_write, web_fetch, \
                memory_search, memory_store. Any other tool names are silently ignored.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Short identifier for this skill (e.g. 'researcher', 'coder')"
                    },
                    "description": {
                        "type": "string",
                        "description": "What this skill is for (shown in list_skills)"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "The system prompt for sub-agents using this skill"
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Tool names to make available (subset of allowed tools)"
                    }
                },
                "required": ["name", "description", "system_prompt"]
            }),
        },
        ToolDefinition {
            name: "list_skills",
            description: "List all defined skills with their names, descriptions, and tool sets. \
                Use before create_skill to avoid duplicates, and before spawn_subagent to \
                choose the right profile.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "update_skill",
            description: "Update a skill's description, system_prompt, or tools by name. \
                Unspecified fields are left unchanged.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name to update"
                    },
                    "description": {
                        "type": "string",
                        "description": "New description (optional)"
                    },
                    "system_prompt": {
                        "type": "string",
                        "description": "New system prompt (optional)"
                    },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "New tool list (optional)"
                    }
                },
                "required": ["name"]
            }),
        },
        ToolDefinition {
            name: "delete_skill",
            description: "Permanently delete a skill by name.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name to delete"
                    }
                },
                "required": ["name"]
            }),
        },
    ]
}

fn task_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "schedule_task",
            description: "Schedule a recurring or one-time autonomous task. \
                The task runner is a separate LLM call that has access to shell_exec, \
                file_read, file_write, web_fetch, and memory tools. \
                Use trigger_type='interval' with trigger_spec in seconds (e.g. '3600' = hourly). \
                Use trigger_type='cron' with trigger_spec as 'HH:MM' in UTC (e.g. '08:00' = 8am daily). \
                Use trigger_type='once' with trigger_spec as an RFC3339 timestamp. \
                Set max_idle_runs higher (e.g. 100) when monitoring for a specific future event \
                that may take a long time (game release, package delivery, etc.) — \
                the task runner will use 'still_waiting' state to keep the task alive without \
                burning idle budget.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "What the task should do each time it runs (natural language)"
                    },
                    "trigger_type": {
                        "type": "string",
                        "enum": ["interval", "cron", "once"],
                        "description": "How to trigger: interval (recurring by seconds), cron (daily at HH:MM UTC), once (specific time)"
                    },
                    "trigger_spec": {
                        "type": "string",
                        "description": "Interval in seconds, 'HH:MM' for cron, or RFC3339 timestamp for once"
                    },
                    "max_idle_runs": {
                        "type": "integer",
                        "description": "Max consecutive 'nothing_to_do' runs before auto-pausing (default: 10). Use higher for long-wait monitoring."
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "Optional RFC3339 timestamp after which the task expires automatically"
                    }
                },
                "required": ["description", "trigger_type", "trigger_spec"]
            }),
        },
        ToolDefinition {
            name: "list_tasks",
            description: "List all scheduled tasks with their current status, \
                run counts, idle counts, and next scheduled run time. \
                Use this to review what tasks are active before creating new ones \
                or before telling the user what background tasks are running.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "schedule_script_task",
            description: "Schedule a mechanical task that runs a shell command without LLM. \
                Much cheaper than schedule_task — use this when the task is just \
                'run a command and check the output'. The command runs, output is checked \
                against success_pattern (regex), and if matched the message_template is sent. \
                Use {output} in the template to include command output.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Human description of what this task monitors"
                    },
                    "trigger_type": {
                        "type": "string",
                        "enum": ["interval", "cron", "once"],
                        "description": "How to trigger: interval (recurring by seconds), cron (daily at HH:MM UTC), once (specific time)"
                    },
                    "trigger_spec": {
                        "type": "string",
                        "description": "Interval in seconds, 'HH:MM' for cron, or RFC3339 timestamp for once"
                    },
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    },
                    "success_pattern": {
                        "type": "string",
                        "description": "Regex pattern — if output matches, state=acted and message is sent. If omitted, exit code 0 = success."
                    },
                    "message_template": {
                        "type": "string",
                        "description": "Message to send when pattern matches. Use {output} for command output."
                    },
                    "max_idle_runs": {
                        "type": "integer",
                        "description": "Max consecutive non-matching runs before auto-pausing (default: 10)"
                    },
                    "expires_at": {
                        "type": "string",
                        "description": "Optional RFC3339 timestamp after which the task expires"
                    }
                },
                "required": ["description", "trigger_type", "trigger_spec", "command"]
            }),
        },
        ToolDefinition {
            name: "delete_task",
            description: "Permanently delete a scheduled task by ID. \
                Use list_tasks to find the ID. \
                Use this when the user asks to cancel or stop a task.",
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "The task ID from list_tasks"
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

fn signal_send_definition() -> ToolDefinition {
    ToolDefinition {
        name: "signal_send",
        description: "Send a proactive Signal message to the primary contact. \
            Use for task completion notifications, reminders, alerts, or any message \
            that should arrive on the user's phone unprompted. \
            Do NOT use shell_exec to invoke signal-cli directly; that will hang.",
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send"
                }
            },
            "required": ["message"]
        }),
    }
}

// ---------------------------------------------------------------------------
// Tool-event formatting helpers
// ---------------------------------------------------------------------------

fn trunc(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}

/// One-line summary of a tool call, injected as a stream chunk before execution.
fn format_tool_call(name: &str, input: &Value) -> String {
    let detail = match name {
        "shell_exec" => {
            let cmd = trunc(input["command"].as_str().unwrap_or(""), 68);
            if input["background"].as_bool().unwrap_or(false) {
                format!("[bg]  {cmd}")
            } else {
                cmd
            }
        }
        "file_read"      => input["path"].as_str().unwrap_or("").to_string(),
        "file_write"     => input["path"].as_str().unwrap_or("").to_string(),
        "web_fetch"      => trunc(input["url"].as_str().unwrap_or(""), 72),
        "memory_search"  => trunc(input["query"].as_str().unwrap_or(""), 60),
        "memory_store"   => trunc(input["title"].as_str().unwrap_or(""), 60),
        "memory_update"  => input["id"].as_str().unwrap_or("").to_string(),
        "memory_delete"  => input["id"].as_str().unwrap_or("").to_string(),
        "spawn_subagent" => trunc(input["task"].as_str().unwrap_or(""), 60),
        "schedule_task" | "schedule_script_task" => trunc(input["description"].as_str().unwrap_or(""), 60),
        "list_tasks" | "list_skills" => String::new(),
        "delete_task"    => input["id"].as_str().unwrap_or("").to_string(),
        "create_skill"   => trunc(input["name"].as_str().unwrap_or(""), 60),
        "update_skill"   => input["id"].as_str().unwrap_or("").to_string(),
        "delete_skill"   => input["id"].as_str().unwrap_or("").to_string(),
        "signal_send"    => trunc(input["message"].as_str().unwrap_or(""), 60),
        _                => trunc(&input.to_string(), 72),
    };
    if detail.is_empty() {
        format!("\n⚙ {name}\n")
    } else {
        format!("\n⚙ {name}  {detail}\n")
    }
}

/// One-line summary of a tool result, injected as a stream chunk after execution.
fn format_tool_result(result: &str) -> String {
    let first = result
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("(no output)");
    format!("  ↳ {}\n\n", trunc(first, 80))
}

// ---------------------------------------------------------------------------

/// Configuration for constructing a `ToolExecutor`.
/// Using a struct avoids positional-parameter friction when adding new fields.
pub struct ToolExecutorConfig {
    pub llm:             Arc<dyn LlmClient>,
    pub memory:          Arc<MemoryManager>,
    pub max_tokens:      usize,
    pub tasks:           Option<Arc<TaskManager>>,
    pub skills:          Option<Arc<SkillManager>>,
    pub job_store:       Option<Arc<BackgroundJobStore>>,
    pub signal_client:   Option<Arc<dyn SignalClient>>,
    pub primary_contact: String,
    pub soul_context:    String,
    pub is_signal_reply: bool,
    pub observer:        Option<Arc<dyn AgentObserver>>,
}

pub struct ToolExecutor {
    llm:            Arc<dyn LlmClient>,
    memory:         Arc<MemoryManager>,
    tasks:          Option<Arc<TaskManager>>,
    skills:         Option<Arc<SkillManager>>,
    soul_context:   String,
    max_tokens:     usize,
    /// Observer for streaming text and tool events — None for non-TUI transports.
    observer:       Option<Arc<dyn AgentObserver>>,
    /// Background job store — present for the main agent, absent for task runners.
    job_store:      Option<Arc<BackgroundJobStore>>,
    /// Signal client for the `signal_send` tool — None when Signal not configured.
    signal_client:  Option<Arc<dyn SignalClient>>,
    /// Primary contact phone number (recipient for signal_send).
    primary_contact: String,
    /// True when the current conversation arrived via Signal.
    /// signal_send is suppressed in this case — replies are delivered automatically
    /// and offering the tool would cause duplicate messages.
    is_signal_reply: bool,
}

impl ToolExecutor {
    /// Construct from a `ToolExecutorConfig`. Tasks/skills/jobs may be None
    /// to restrict the available tool set.
    pub fn from_config(cfg: ToolExecutorConfig) -> Self {
        Self {
            llm:             cfg.llm,
            memory:          cfg.memory,
            tasks:           cfg.tasks,
            skills:          cfg.skills,
            soul_context:    cfg.soul_context,
            max_tokens:      cfg.max_tokens,
            observer:        cfg.observer,
            job_store:       cfg.job_store,
            signal_client:   cfg.signal_client,
            primary_contact: cfg.primary_contact,
            is_signal_reply: cfg.is_signal_reply,
        }
    }

    /// Convenience constructor for task runners: no task/skill management, no observer.
    pub fn for_task_runner(
        llm: Arc<dyn LlmClient>,
        memory: Arc<MemoryManager>,
        max_tokens: usize,
    ) -> Self {
        Self {
            llm,
            memory,
            tasks: None,
            skills: None,
            soul_context: String::new(),
            max_tokens,
            observer: None,
            job_store: None,
            signal_client: None,
            primary_contact: String::new(),
            is_signal_reply: false,
        }
    }

    /// Full agent tool set: base + memory + spawn_subagent + task + skill management.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = tool_definitions();
        defs.extend(memory_tool_definitions());
        defs.push(spawn_subagent_definition());
        if self.tasks.is_some() {
            defs.extend(task_tool_definitions());
        }
        if self.skills.is_some() {
            defs.extend(skill_tool_definitions());
        }
        if self.signal_client.is_some() && !self.is_signal_reply {
            defs.push(signal_send_definition());
        }
        defs
    }

    /// Restricted task-runner tool set: base + memory_search + memory_store only.
    /// No scheduling tools, no spawn_subagent, no memory mutations.
    pub fn task_runner_tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut defs = tool_definitions();
        defs.extend(
            memory_tool_definitions()
                .into_iter()
                .filter(|d| matches!(d.name, "memory_search" | "memory_store")),
        );
        defs
    }

    pub async fn execute(&self, name: &str, input: &Value) -> String {
        if let Some(obs) = &self.observer {
            obs.on_tool_start(name);
            obs.on_tool_event(format_tool_call(name, input));
        }

        // Background shell execution — spawn and return immediately.
        if name == "shell_exec" && input["background"].as_bool().unwrap_or(false) {
            if let Some(store) = &self.job_store {
                let command = input["command"].as_str().unwrap_or("").to_string();
                let timeout_secs = input["timeout_secs"].as_u64().unwrap_or(3600);
                let id = store.spawn(command, timeout_secs);
                let result = format!(
                    "job:{id} started in background — you will be notified when it completes"
                );
                if let Some(obs) = &self.observer {
                    obs.on_tool_event(format_tool_result(&result));
                }
                return result;
            }
        }

        let result = match name {
            "memory_store"   => self.memory_store(input).await,
            "memory_search"  => self.memory_search(input).await,
            "memory_update"  => self.memory_update(input).await,
            "memory_delete"  => self.memory_delete(input).await,
            "spawn_subagent" => self.spawn_subagent(input).await,
            "schedule_task"        => self.schedule_task(input).await,
            "schedule_script_task" => self.schedule_script_task(input).await,
            "list_tasks"           => self.list_tasks().await,
            "delete_task"          => self.delete_task(input).await,
            "create_skill"   => self.create_skill(input).await,
            "list_skills"    => self.list_skills().await,
            "update_skill"   => self.update_skill(input).await,
            "delete_skill"   => self.delete_skill(input).await,
            "signal_send"    => self.signal_send(input).await,
            _ => execute_tool(name, input).await,
        };
        if let Some(obs) = &self.observer {
            obs.on_tool_event(format_tool_result(&result));
        }
        result
    }

    // -----------------------------------------------------------------------
    // Memory tool implementations
    // -----------------------------------------------------------------------

    async fn memory_store(&self, input: &Value) -> String {
        let content = match input["content"].as_str() {
            Some(c) => c,
            None => return "Error: missing 'content' field".to_string(),
        };
        let kind = crate::memory::MemoryKind::from_str_safe(
            input["kind"].as_str().unwrap_or("semantic"),
        );
        let importance = input["importance"].as_f64().unwrap_or(0.8).clamp(0.0, 1.0);
        let pinned = input["pinned"].as_bool().unwrap_or(false);

        match self.memory.store(content, kind, "user", importance).await {
            Err(e) => format!("Error storing memory: {e:#}"),
            Ok(id) => {
                if pinned {
                    let _ = self.memory.pin(&id).await;
                }
                format!("Memory stored with id '{id}'")
            }
        }
    }

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

    // -----------------------------------------------------------------------
    // Task tool implementations
    // -----------------------------------------------------------------------

    async fn schedule_task(&self, input: &Value) -> String {
        let tasks = match &self.tasks {
            Some(t) => t,
            None => return "Error: task scheduling not available in this context".to_string(),
        };
        let description = match input["description"].as_str() {
            Some(d) => d,
            None => return "Error: missing 'description' field".to_string(),
        };
        let trigger_type = match input["trigger_type"].as_str() {
            Some(t) => t,
            None => return "Error: missing 'trigger_type' field (interval/cron/once)".to_string(),
        };
        let trigger_spec = match input["trigger_spec"].as_str() {
            Some(s) => s,
            None => return "Error: missing 'trigger_spec' field".to_string(),
        };
        let max_idle_runs = input["max_idle_runs"].as_i64().unwrap_or(10).max(1);
        let expires_at = input["expires_at"]
            .as_str()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|t| t.with_timezone(&chrono::Utc));

        match tasks
            .create(description, trigger_type, trigger_spec, max_idle_runs, expires_at)
            .await
        {
            Ok(id) => format!(
                "Task scheduled (id='{id}'). Trigger: {trigger_type}:{trigger_spec}. \
                 Max idle runs: {max_idle_runs}."
            ),
            Err(e) => format!("Error scheduling task: {e:#}"),
        }
    }

    async fn schedule_script_task(&self, input: &Value) -> String {
        let tasks = match &self.tasks {
            Some(t) => t,
            None => return "Error: task scheduling not available in this context".to_string(),
        };
        let description = match input["description"].as_str() {
            Some(d) => d,
            None => return "Error: missing 'description' field".to_string(),
        };
        let trigger_type = match input["trigger_type"].as_str() {
            Some(t) => t,
            None => return "Error: missing 'trigger_type' field".to_string(),
        };
        let trigger_spec = match input["trigger_spec"].as_str() {
            Some(s) => s,
            None => return "Error: missing 'trigger_spec' field".to_string(),
        };
        let command = match input["command"].as_str() {
            Some(c) => c,
            None => return "Error: missing 'command' field".to_string(),
        };
        let success_pattern = input["success_pattern"].as_str();
        let message_template = input["message_template"].as_str();
        let max_idle_runs = input["max_idle_runs"].as_i64().unwrap_or(10).max(1);
        let expires_at = input["expires_at"]
            .as_str()
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
            .map(|t| t.with_timezone(&chrono::Utc));

        match tasks
            .create_script(
                description, trigger_type, trigger_spec, command,
                success_pattern, message_template, max_idle_runs, expires_at,
            )
            .await
        {
            Ok(id) => format!(
                "Script task scheduled (id='{id}'). Trigger: {trigger_type}:{trigger_spec}. \
                 Command: {command}. No LLM cost per run."
            ),
            Err(e) => format!("Error scheduling script task: {e:#}"),
        }
    }

    async fn list_tasks(&self) -> String {
        let tasks = match &self.tasks {
            Some(t) => t,
            None => return "Error: task management not available in this context".to_string(),
        };
        match tasks.list().await {
            Err(e) => format!("Error listing tasks: {e:#}"),
            Ok(entries) if entries.is_empty() => "[]".to_string(),
            Ok(entries) => {
                let items: Vec<serde_json::Value> = entries
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "id": t.id,
                            "type": t.task_type,
                            "description": t.description,
                            "trigger": format!("{}:{}", t.trigger_type, t.trigger_spec),
                            "enabled": t.enabled,
                            "run_count": t.run_count,
                            "idle_count": t.idle_count,
                            "max_idle_runs": t.max_idle_runs,
                            "last_run": t.last_run.map(|t| t.to_rfc3339()),
                            "next_run": t.next_run.to_rfc3339(),
                            "expires_at": t.expires_at.map(|t| t.to_rfc3339()),
                        })
                    })
                    .collect();
                serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string())
            }
        }
    }

    async fn delete_task(&self, input: &Value) -> String {
        let tasks = match &self.tasks {
            Some(t) => t,
            None => return "Error: task management not available in this context".to_string(),
        };
        let id = match input["id"].as_str() {
            Some(i) => i,
            None => return "Error: missing 'id' field".to_string(),
        };
        match tasks.delete(id).await {
            Ok(()) => format!("Task '{id}' deleted"),
            Err(e) => format!("Error deleting task: {e:#}"),
        }
    }

    // -----------------------------------------------------------------------
    // Skill tool implementations
    // -----------------------------------------------------------------------

    async fn create_skill(&self, input: &Value) -> String {
        let skills = match &self.skills {
            Some(s) => s,
            None => return "Error: skill management not available in this context".to_string(),
        };
        let name = match input["name"].as_str() {
            Some(n) => n,
            None => return "Error: missing 'name' field".to_string(),
        };
        let description = match input["description"].as_str() {
            Some(d) => d,
            None => return "Error: missing 'description' field".to_string(),
        };
        let system_prompt = match input["system_prompt"].as_str() {
            Some(p) => p,
            None => return "Error: missing 'system_prompt' field".to_string(),
        };
        let tools: Vec<String> = input["tools"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_else(|| vec!["shell_exec".to_string(), "file_read".to_string(), "file_write".to_string(), "web_fetch".to_string()]);

        match skills.create(name, description, system_prompt, tools).await {
            Ok(id) => format!(
                "Skill '{name}' created (id='{id}'). \
                 Invoke with spawn_subagent using skill='{name}'."
            ),
            Err(e) => format!("Error creating skill: {e:#}"),
        }
    }

    async fn list_skills(&self) -> String {
        let skills = match &self.skills {
            Some(s) => s,
            None => return "Error: skill management not available in this context".to_string(),
        };
        match skills.list().await {
            Err(e) => format!("Error listing skills: {e:#}"),
            Ok(entries) if entries.is_empty() => "[]".to_string(),
            Ok(entries) => {
                let items: Vec<serde_json::Value> = entries
                    .iter()
                    .map(|s| {
                        serde_json::json!({
                            "name": s.name,
                            "description": s.description,
                            "tools": s.tools,
                            "updated_at": s.updated_at.to_rfc3339(),
                        })
                    })
                    .collect();
                serde_json::to_string_pretty(&items).unwrap_or_else(|_| "[]".to_string())
            }
        }
    }

    async fn update_skill(&self, input: &Value) -> String {
        let skills = match &self.skills {
            Some(s) => s,
            None => return "Error: skill management not available in this context".to_string(),
        };
        let name = match input["name"].as_str() {
            Some(n) => n,
            None => return "Error: missing 'name' field".to_string(),
        };
        let new_desc = input["description"].as_str();
        let new_prompt = input["system_prompt"].as_str();
        let new_tools: Option<Vec<String>> = input["tools"].as_array().map(|arr| {
            arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect()
        });

        if new_desc.is_none() && new_prompt.is_none() && new_tools.is_none() {
            return "Error: provide at least one of 'description', 'system_prompt', or 'tools'".to_string();
        }

        match skills.update(name, new_desc, new_prompt, new_tools).await {
            Ok(true) => format!("Skill '{name}' updated"),
            Ok(false) => format!("Error: no skill named '{name}' found"),
            Err(e) => format!("Error updating skill: {e:#}"),
        }
    }

    async fn delete_skill(&self, input: &Value) -> String {
        let skills = match &self.skills {
            Some(s) => s,
            None => return "Error: skill management not available in this context".to_string(),
        };
        let name = match input["name"].as_str() {
            Some(n) => n,
            None => return "Error: missing 'name' field".to_string(),
        };
        match skills.delete(name).await {
            Ok(true) => format!("Skill '{name}' deleted"),
            Ok(false) => format!("Error: no skill named '{name}' found"),
            Err(e) => format!("Error deleting skill: {e:#}"),
        }
    }

    async fn signal_send(&self, input: &Value) -> String {
        let client = match &self.signal_client {
            Some(c) => c,
            None => return "Error: Signal is not configured".to_string(),
        };
        let message = match input["message"].as_str() {
            Some(m) if !m.is_empty() => m,
            _ => return "Error: missing 'message' field".to_string(),
        };
        match client.send(&self.primary_contact, message).await {
            Ok(()) => format!("Sent to {}", self.primary_contact),
            Err(e) => format!("Error sending Signal message: {e:#}"),
        }
    }

    // -----------------------------------------------------------------------
    // spawn_subagent — with soul context injection and optional skill profile
    // -----------------------------------------------------------------------

    async fn spawn_subagent(&self, input: &Value) -> String {
        let task = match input["task"].as_str() {
            Some(t) => t,
            None => return "Error: missing 'task' field".to_string(),
        };
        let context = input["context"].as_str().unwrap_or("");
        let skill_name = input["skill"].as_str();

        // Resolve skill if requested.
        let skill = if let Some(name) = skill_name {
            match &self.skills {
                Some(mgr) => match mgr.get_by_name(name).await {
                    Ok(Some(s)) => Some(s),
                    Ok(None) => {
                        return format!("Error: no skill named '{name}' found. Use list_skills to see available skills.")
                    }
                    Err(e) => return format!("Error loading skill: {e:#}"),
                },
                None => None,
            }
        } else {
            None
        };

        // Build the system prompt: soul context first, then agent identity.
        let base_identity = match &skill {
            Some(s) => s.system_prompt.clone(),
            None => "You are a focused sub-agent. Complete the assigned task thoroughly \
                using the available tools. Do not spawn further sub-agents."
                .to_string(),
        };

        let system_owned = if self.soul_context.is_empty() {
            base_identity
        } else {
            format!("{}\n\n---\n\n{}", self.soul_context, base_identity)
        };

        // Determine allowed tools for this sub-agent.
        let allowed: Vec<&str> = match &skill {
            Some(s) => s.tools.iter().map(|t| t.as_str()).collect(),
            None => ALLOWED_SKILL_TOOLS.to_vec(),
        };
        let sub_defs: Vec<ToolDefinition> = tool_definitions()
            .into_iter()
            .chain(
                memory_tool_definitions()
                    .into_iter()
                    .filter(|d| matches!(d.name, "memory_search" | "memory_store")),
            )
            .filter(|d| allowed.contains(&d.name))
            .collect();

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
                    system: &system_owned,
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

            messages.push(ApiMessage {
                role: "assistant".to_string(),
                content: MessageContent::Blocks(resp.raw_blocks),
            });

            let mut results = Vec::new();
            for tc in &resp.tool_calls {
                let output = execute_tool(&tc.name, &tc.input).await;
                results.push(ContentBlock::ToolResult {
                    tool_use_id: tc.id.clone(),
                    content: output,
                });
            }

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
        // /tmp/flint_test_nonexistent should not exist; force=true should attempt write
        let path = "/tmp/flint_tools_test_write.txt";
        let _ = std::fs::remove_file(path); // clean up if exists
        let input = serde_json::json!({"path": path, "content": "hello", "force": true});
        let result = file_write(&input);
        assert!(result.contains("Successfully wrote"), "got: {result}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_file_write_no_overwrite_without_force() {
        let path = "/tmp/flint_tools_test_overwrite.txt";
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
        let input = serde_json::json!({"path": "/tmp/flint_does_not_exist_xyz.txt"});
        let result = file_read(&input);
        assert!(result.starts_with("Error reading file"));
    }

    async fn test_executor() -> ToolExecutor {
        use crate::claude::mock::MockLlm;
        use crate::config::MemoryConfig;
        use crate::embeddings::mock::MockEmbeddingClient;
        use crate::tasks::TaskManager;
        use std::path::Path;
        let llm = Arc::new(MockLlm::new(vec![]));
        let embedder = Arc::new(MockEmbeddingClient::new(4));
        let memory = Arc::new(
            MemoryManager::new(
                Path::new(":memory:"),
                embedder,
                MemoryConfig {
                    max_memories: 100,
                    top_k_retrieval: 5,
                    importance_decay_days: 30.0,
                    min_importance_to_keep: 0.1,
                    ttl_days_episodic: 90.0,
                },
                4,
            )
            .await
            .unwrap(),
        );
        let tasks = Arc::new(TaskManager::in_memory().await.unwrap());
        let skills = Arc::new(crate::skills::SkillManager::in_memory().await.unwrap());
        ToolExecutor::from_config(ToolExecutorConfig {
            llm,
            memory,
            max_tokens: 1000,
            tasks: Some(tasks),
            skills: Some(skills),
            job_store: None,
            signal_client: None,
            primary_contact: String::new(),
            soul_context: String::new(),
            is_signal_reply: false,
            observer: None,
        })
    }

    #[tokio::test]
    async fn test_executor_tool_definitions() {
        let executor = test_executor().await;
        let defs = executor.tool_definitions();
        // 4 base + 4 memory + spawn_subagent + 4 task tools + 4 skill tools = 17 (no signal_send, signal_client is None)
        assert_eq!(defs.len(), 17);
        let names: Vec<_> = defs.iter().map(|d| d.name).collect();
        assert!(names.contains(&"spawn_subagent"));
        assert!(names.contains(&"memory_store"));
        assert!(names.contains(&"memory_search"));
        assert!(names.contains(&"memory_update"));
        assert!(names.contains(&"memory_delete"));
        assert!(names.contains(&"schedule_task"));
        assert!(names.contains(&"list_tasks"));
        assert!(names.contains(&"delete_task"));
    }

    #[tokio::test]
    async fn test_task_runner_tool_definitions() {
        let executor = test_executor().await;
        let defs = executor.task_runner_tool_definitions();
        // 4 base + memory_search + memory_store = 6
        assert_eq!(defs.len(), 6);
        let names: Vec<_> = defs.iter().map(|d| d.name).collect();
        assert!(names.contains(&"shell_exec"));
        assert!(names.contains(&"web_fetch"));
        assert!(names.contains(&"memory_search"));
        assert!(names.contains(&"memory_store"));
        assert!(!names.contains(&"schedule_task"));
        assert!(!names.contains(&"spawn_subagent"));
        assert!(!names.contains(&"memory_delete"));
    }

    #[tokio::test]
    async fn test_spawn_subagent_returns_mock_text() {
        use crate::claude::mock::MockLlm;
        use crate::config::MemoryConfig;
        use crate::embeddings::mock::MockEmbeddingClient;
        use crate::tasks::TaskManager;
        use std::path::Path;
        let llm = Arc::new(MockLlm::new(vec!["sub-agent result".to_string()]));
        let embedder = Arc::new(MockEmbeddingClient::new(4));
        let memory = Arc::new(
            MemoryManager::new(
                Path::new(":memory:"),
                embedder,
                MemoryConfig {
                    max_memories: 100,
                    top_k_retrieval: 5,
                    importance_decay_days: 30.0,
                    min_importance_to_keep: 0.1,
                    ttl_days_episodic: 90.0,
                },
                4,
            )
            .await
            .unwrap(),
        );
        let tasks = Arc::new(TaskManager::in_memory().await.unwrap());
        let skills = Arc::new(crate::skills::SkillManager::in_memory().await.unwrap());
        let executor = ToolExecutor::from_config(ToolExecutorConfig {
            llm,
            memory,
            max_tokens: 1000,
            tasks: Some(tasks),
            skills: Some(skills),
            job_store: None,
            signal_client: None,
            primary_contact: String::new(),
            soul_context: String::new(),
            is_signal_reply: false,
            observer: None,
        });
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
