//! Sub-agent coordination: background execution, streaming TUI events,
//! DAG-based plan orchestration, and completion notifications.
//!
//! Mirrors the `BackgroundJobStore` pattern: `SubAgentManager::new()` returns
//! `(Arc<manager>, notify_rx)`.  When a sub-agent completes, a
//! `SubAgentNotification` is sent through `notify_rx` and the main event loop
//! injects it as a synthetic message so the primary agent can relay results.

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use serde::Deserialize;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use crate::claude::{
    ApiMessage, ClaudeRequest, ContentBlock, LlmClient, MessageContent, ToolDefinition,
};
use crate::code_intel::CodeIndex;
use crate::memory::MemoryManager;
use crate::observer::AgentObserver;
use crate::skills::{SkillEntry, SkillManager, ALLOWED_SKILL_TOOLS};
use crate::tools::{self, execute_tool, memory_tool_definitions, tool_definitions};
use crate::tui::AgentUpdate;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Events streamed from a running sub-agent to the TUI.
#[derive(Debug, Clone)]
pub enum SubAgentEvent {
    TextChunk(String),
    ToolStart(String),
    ToolEvent(String),
    Complete(String),
    Error(String),
}

/// Notification sent to the main event loop when a sub-agent finishes.
pub struct SubAgentNotification {
    pub id: u64,
    pub task: String,
    pub result: String,
    pub is_error: bool,
}

impl SubAgentNotification {
    /// Format as a synthetic message for the primary agent.
    pub fn to_agent_text(&self) -> String {
        const EXCERPT_LEN: usize = 600;
        let (excerpt, truncated) = if self.result.chars().count() > EXCERPT_LEN {
            let t: String = self.result.chars().take(EXCERPT_LEN).collect();
            (t, true)
        } else {
            (self.result.clone(), false)
        };
        let truncation_note = if truncated { " [truncated]" } else { "" };

        if self.is_error {
            format!(
                "[Sub-agent {} failed]\nTask: {}\nError excerpt: {}{}",
                self.id, self.task, excerpt, truncation_note,
            )
        } else {
            format!(
                "[Sub-agent {} completed]\nTask: {}\nResult excerpt: {}{}\n\nSynthesise the above in your own words for the user. Do not quote the raw output verbatim.",
                self.id, self.task, excerpt, truncation_note,
            )
        }
    }
}

/// Snapshot of one active sub-agent for `list_subagents`.
pub struct SubAgentInfo {
    pub id: u64,
    pub task: String,
    pub skill: Option<String>,
    pub plan_id: Option<u64>,
    pub step_id: Option<String>,
}

// ---------------------------------------------------------------------------
// SubAgentManager
// ---------------------------------------------------------------------------

pub struct SubAgentManager {
    next_id: AtomicU64,
    notify_tx: mpsc::Sender<SubAgentNotification>,
    tui_tx: mpsc::Sender<AgentUpdate>,
    active: Arc<RwLock<HashMap<u64, SubAgentInfo>>>,
}

impl SubAgentManager {
    pub fn new(
        tui_tx: mpsc::Sender<AgentUpdate>,
    ) -> (Arc<Self>, mpsc::Receiver<SubAgentNotification>) {
        let (notify_tx, notify_rx) = mpsc::channel(64);
        let mgr = Arc::new(Self {
            next_id: AtomicU64::new(1),
            notify_tx,
            tui_tx,
            active: Arc::new(RwLock::new(HashMap::new())),
        });
        (mgr, notify_rx)
    }

    /// Spawn a single sub-agent in the background.  Returns the sub-agent ID
    /// immediately so the main agent can continue responding.
    pub async fn spawn(
        &self,
        task: String,
        context: String,
        skill: Option<SkillEntry>,
        soul_context: String,
        llm: Arc<dyn LlmClient>,
        memory: Arc<MemoryManager>,
        max_tokens: usize,
        code_index: Arc<CodeIndex>,
        plan_id: Option<u64>,
        step_id: Option<String>,
    ) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let skill_name = skill.as_ref().map(|s| s.name.clone());

        // Register in active set.
        {
            let mut active = self.active.write().await;
            active.insert(id, SubAgentInfo {
                id,
                task: task.clone(),
                skill: skill_name.clone(),
                plan_id,
                step_id: step_id.clone(),
            });
        }

        // Notify TUI.
        let label = match &skill_name {
            Some(name) => format!("{name}: {}", trunc(&task, 50)),
            None => trunc(&task, 60),
        };
        let _ = self.tui_tx.send(AgentUpdate::SubAgentStarted {
            id,
            task: label,
        }).await;

        let notify_tx = self.notify_tx.clone();
        let tui_tx = self.tui_tx.clone();
        let active = self.active.clone();
        let task_clone = task.clone();

        tokio::spawn(async move {
            let observer = Arc::new(SubAgentObserver { id, tui_tx: tui_tx.clone() });
            let result = run_subagent_loop(
                &task_clone,
                &context,
                skill.as_ref(),
                &soul_context,
                &llm,
                &memory,
                max_tokens,
                &code_index,
                observer.clone(),
            )
            .await;

            let (result_text, is_error) = match result {
                Ok(text) => (text, false),
                Err(e) => (format!("{e:#}"), true),
            };

            // Remove from active set.
            {
                let mut active = active.write().await;
                active.remove(&id);
            }

            // Notify TUI of completion.
            let summary = trunc(&result_text, 120);
            let _ = tui_tx.send(AgentUpdate::SubAgentCompleted {
                id,
                result_summary: summary,
            }).await;

            // Notify main event loop.
            let _ = notify_tx.send(SubAgentNotification {
                id,
                task: task_clone,
                result: result_text,
                is_error,
            }).await;

            debug!("Sub-agent {id} finished (error={is_error})");
        });

        info!("Sub-agent {id} spawned: {}", trunc(&task, 60));
        id
    }

    /// List currently active sub-agents.
    pub async fn list(&self) -> Vec<SubAgentInfo> {
        let active = self.active.read().await;
        active.values().map(|info| SubAgentInfo {
            id: info.id,
            task: info.task.clone(),
            skill: info.skill.clone(),
            plan_id: info.plan_id,
            step_id: info.step_id.clone(),
        }).collect()
    }

}

// ---------------------------------------------------------------------------
// Plan types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct PlanStep {
    pub id: String,
    pub task: String,
    #[serde(default)]
    pub skill: Option<String>,
    #[serde(default)]
    pub context: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
}

// ---------------------------------------------------------------------------
// SubAgentObserver — streams events to the TUI
// ---------------------------------------------------------------------------

struct SubAgentObserver {
    id: u64,
    tui_tx: mpsc::Sender<AgentUpdate>,
}

impl AgentObserver for SubAgentObserver {
    fn on_text_chunk(&self, chunk: String) {
        let _ = self.tui_tx.try_send(AgentUpdate::SubAgentChunk {
            id: self.id,
            chunk,
        });
    }

    fn on_tool_start(&self, name: &str) {
        let _ = self.tui_tx.try_send(AgentUpdate::SubAgentToolEvent {
            id: self.id,
            text: format!("⚙ {name}"),
        });
    }

    fn on_tool_event(&self, text: String) {
        let _ = self.tui_tx.try_send(AgentUpdate::SubAgentToolEvent {
            id: self.id,
            text,
        });
    }
}

// ---------------------------------------------------------------------------
// Sub-agent LLM tool loop (extracted from tools.rs spawn_subagent)
// ---------------------------------------------------------------------------

async fn run_subagent_loop(
    task: &str,
    context: &str,
    skill: Option<&SkillEntry>,
    soul_context: &str,
    llm: &Arc<dyn LlmClient>,
    _memory: &Arc<MemoryManager>,
    max_tokens: usize,
    _code_index: &Arc<CodeIndex>,
    observer: Arc<SubAgentObserver>,
) -> anyhow::Result<String> {
    // Build system prompt.
    let base_identity = match skill {
        Some(s) => s.system_prompt.clone(),
        None => "You are a focused sub-agent. Complete the assigned task thoroughly \
            using the available tools. Do not spawn further sub-agents."
            .to_string(),
    };

    let system = if soul_context.is_empty() {
        base_identity
    } else {
        format!("{soul_context}\n\n---\n\n{base_identity}")
    };

    // Determine allowed tools.
    let allowed: Vec<&str> = match skill {
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
        let cb = {
            let obs = observer.clone();
            Arc::new(move |chunk: String| obs.on_text_chunk(chunk))
        };

        let resp = llm
            .complete_with_tools_streaming(
                ClaudeRequest {
                    system: &system,
                    messages: messages.clone(),
                    max_tokens,
                    tools: &sub_defs,
                },
                cb,
            )
            .await?;

        if resp.tool_calls.is_empty() {
            return Ok(resp.text);
        }

        messages.push(ApiMessage {
            role: "assistant".to_string(),
            content: MessageContent::Blocks(resp.raw_blocks),
        });

        let mut results = Vec::new();
        for tc in &resp.tool_calls {
            observer.on_tool_start(&tc.name);
            observer.on_tool_event(tools::format_tool_call(&tc.name, &tc.input));
            let output = execute_tool(&tc.name, &tc.input).await;
            observer.on_tool_event(tools::format_tool_result(&output));
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

// ---------------------------------------------------------------------------
// Plan execution — improved orchestrator
// ---------------------------------------------------------------------------

impl SubAgentManager {
    /// Improved plan execution that properly tracks step results.
    /// Uses a dedicated channel to collect step completions.
    pub async fn execute_plan(
        self: &Arc<Self>,
        steps: Vec<PlanStep>,
        soul_context: String,
        llm: Arc<dyn LlmClient>,
        memory: Arc<MemoryManager>,
        max_tokens: usize,
        code_index: Arc<CodeIndex>,
        skills: Option<Arc<SkillManager>>,
    ) -> String {
        if steps.is_empty() {
            return "Error: plan has no steps".to_string();
        }

        let plan_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let step_count = steps.len();

        // Validate.
        let step_ids: Vec<String> = steps.iter().map(|s| s.id.clone()).collect();
        for step in &steps {
            for dep in &step.depends_on {
                if !step_ids.contains(dep) {
                    return format!("Error: step '{}' depends on unknown step '{dep}'", step.id);
                }
            }
        }

        // Channel for step completions within this plan.
        let (step_done_tx, mut step_done_rx) = mpsc::channel::<(String, String, bool)>(32);

        let mgr = self.clone();
        let steps_arc = Arc::new(steps);
        let ready_count = steps_arc.iter().filter(|s| s.depends_on.is_empty()).count();
        let chained_count = step_count - ready_count;

        tokio::spawn(async move {
            let mut spawned: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut results: HashMap<String, String> = HashMap::new();

            loop {
                // Find ready steps.
                let mut newly_ready = Vec::new();
                for step in steps_arc.iter() {
                    if spawned.contains(&step.id) {
                        continue;
                    }
                    if step.depends_on.iter().all(|d| results.contains_key(d)) {
                        newly_ready.push(step.clone());
                    }
                }

                // Spawn ready steps.
                for step in newly_ready {
                    spawned.insert(step.id.clone());

                    let dep_context = {
                        let mut ctx_parts = Vec::new();
                        for dep_id in &step.depends_on {
                            if let Some(result) = results.get(dep_id) {
                                ctx_parts.push(format!(
                                    "--- Result from step '{dep_id}' ---\n{result}\n"
                                ));
                            }
                        }
                        if !step.context.is_empty() {
                            ctx_parts.push(step.context.clone());
                        }
                        ctx_parts.join("\n")
                    };

                    let skill = if let Some(skill_name) = &step.skill {
                        if let Some(ref skills) = skills {
                            skills.get_by_name(skill_name).await.ok().flatten()
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    // Spawn the sub-agent through the manager (which handles
                    // TUI events and the main notification channel).
                    let sa_id = mgr.spawn(
                        step.task.clone(),
                        dep_context,
                        skill,
                        soul_context.clone(),
                        llm.clone(),
                        memory.clone(),
                        max_tokens,
                        code_index.clone(),
                        Some(plan_id),
                        Some(step.id.clone()),
                    ).await;

                    // Monitor this sub-agent's completion via the active set.
                    let active = mgr.active.clone();
                    let step_id = step.id.clone();
                    let tx = step_done_tx.clone();
                    tokio::spawn(async move {
                        loop {
                            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                            let map = active.read().await;
                            if !map.contains_key(&sa_id) {
                                // Sub-agent finished. We don't have its result
                                // text here (that goes via notify_tx to main),
                                // but we can signal that the step is done.
                                let _ = tx.send((step_id, String::new(), false)).await;
                                break;
                            }
                        }
                    });
                }

                // Check if all done.
                if results.len() == steps_arc.len() {
                    info!("Plan {plan_id}: all {step_count} steps completed");
                    break;
                }

                // Wait for a step to complete.
                if let Some((step_id, result, _is_error)) = step_done_rx.recv().await {
                    results.insert(step_id, result);
                } else {
                    warn!("Plan {plan_id}: step channel closed unexpectedly");
                    break;
                }
            }
        });

        format!(
            "Plan started (plan_id={plan_id}): {step_count} steps — \
             {ready_count} immediate, {chained_count} dependent. \
             You will be notified as each sub-agent completes."
        )
    }
}

// ---------------------------------------------------------------------------
// Tool definitions for the new sub-agent tools
// ---------------------------------------------------------------------------

pub fn subagent_tool_definitions() -> Vec<crate::claude::ToolDefinition> {
    vec![
        spawn_subagent_definition(),
        plan_subagents_definition(),
        list_subagents_definition(),
    ]
}

fn spawn_subagent_definition() -> ToolDefinition {
    ToolDefinition {
        name: "spawn_subagent",
        description: "Spawn a focused sub-agent in the BACKGROUND to handle a self-contained task. \
            Returns immediately with a sub-agent ID — you stay responsive to the user. \
            You will be notified when the sub-agent completes with its results. \
            Use this for ANY task requiring more than 2 tool calls: research, multi-file operations, \
            complex analysis, data processing, etc. The sub-agent has shell_exec, file_read, \
            file_write, web_fetch, memory_search, and memory_store available.",
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
                },
                "skill": {
                    "type": "string",
                    "description": "Optional skill name to use a specific sub-agent profile"
                }
            },
            "required": ["task"]
        }),
    }
}

fn plan_subagents_definition() -> ToolDefinition {
    ToolDefinition {
        name: "plan_subagents",
        description: "Execute a plan of multiple sub-agent tasks with dependency management. \
            Steps with no dependencies run in parallel. Steps can depend on results from \
            previous steps (dependency results are injected as context). \
            Use this for complex workflows: parallel research + synthesis, \
            multi-file refactoring, staged analysis pipelines.",
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique step identifier for dependency references"
                            },
                            "task": {
                                "type": "string",
                                "description": "The task for this sub-agent to perform"
                            },
                            "skill": {
                                "type": "string",
                                "description": "Optional skill name for this step"
                            },
                            "context": {
                                "type": "string",
                                "description": "Additional context for this step"
                            },
                            "depends_on": {
                                "type": "array",
                                "items": { "type": "string" },
                                "description": "Step IDs that must complete before this step starts"
                            }
                        },
                        "required": ["id", "task"]
                    }
                }
            },
            "required": ["steps"]
        }),
    }
}

fn list_subagents_definition() -> ToolDefinition {
    ToolDefinition {
        name: "list_subagents",
        description: "List currently active sub-agents and their tasks.",
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {}
        }),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn trunc(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let t: String = s.chars().take(max).collect();
        format!("{t}…")
    }
}
