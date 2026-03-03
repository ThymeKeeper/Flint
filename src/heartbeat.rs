use anyhow::{Context, Result};
use std::sync::Arc;
use tracing::{error, info, warn};

use crate::claude::{ApiMessage, ClaudeRequest, ContentBlock, LlmClient, MessageContent};
use crate::config::AppConfig;
use crate::memory::MemoryManager;
use crate::signal::SignalClient;
use crate::tasks::{TaskEntry, TaskLifecycleEvent, TaskManager, TaskRunOutput, TaskRunState};
use crate::tools::ToolExecutor;

// ---------------------------------------------------------------------------
// Heartbeat loop
// ---------------------------------------------------------------------------

/// Run the heartbeat loop forever.
///
/// Each tick:
///   1. Maintenance — decay importance, prune old memories (DB-only, no LLM)
///   2. Tasks       — find due tasks, run each via an isolated LLM call,
///                    handle lifecycle events (auto-pause, expiry, one-shot)
pub async fn run_heartbeat(
    memory: Arc<MemoryManager>,
    tasks: Arc<TaskManager>,
    llm: Arc<dyn LlmClient>,
    signal: Arc<dyn SignalClient>,
    config: AppConfig,
) {
    let interval =
        std::time::Duration::from_secs(config.heartbeat.interval_secs);
    info!("Heartbeat started (interval={}s)", config.heartbeat.interval_secs);

    loop {
        tokio::time::sleep(interval).await;
        info!("Heartbeat firing");

        if let Err(e) = run_maintenance(&memory, config.heartbeat.interval_secs).await {
            error!("Heartbeat maintenance failed: {e:#}");
        }

        if let Err(e) =
            run_due_tasks(&tasks, &llm, &memory, &signal, &config).await
        {
            error!("Heartbeat task execution failed: {e:#}");
        }

        info!("Heartbeat complete");
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Maintenance (DB-only, no LLM)
// ---------------------------------------------------------------------------

async fn run_maintenance(memory: &MemoryManager, interval_secs: u64) -> Result<()> {
    info!("Heartbeat: running maintenance");
    let decayed = memory.decay(interval_secs).await?;
    info!("Decayed {decayed} memories");
    let pruned = memory.prune().await?;
    info!("Pruned {pruned} memories");
    let count = memory.count().await?;
    info!("Total memories after maintenance: {count}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2: Task execution
// ---------------------------------------------------------------------------

async fn run_due_tasks(
    tasks: &TaskManager,
    llm: &Arc<dyn LlmClient>,
    memory: &Arc<MemoryManager>,
    signal: &Arc<dyn SignalClient>,
    config: &AppConfig,
) -> Result<()> {
    let due = tasks.due().await?;
    if due.is_empty() {
        return Ok(());
    }
    info!("Heartbeat: {} task(s) due", due.len());

    for task in &due {
        info!("Running task '{}' ({})", task.id, truncate(&task.description, 60));

        let output =
            match execute_task(task, llm, memory, config.claude.max_tokens).await {
                Ok(o) => o,
                Err(e) => {
                    warn!("Task '{}' execution error: {e:#}", task.id);
                    // Treat errors as nothing_to_do to avoid infinite failure loops.
                    TaskRunOutput {
                        state: TaskRunState::NothingToDo,
                        reason: format!("Execution error: {e:#}"),
                        message: None,
                    }
                }
            };

        info!(
            "Task '{}' → {:?}: {}",
            task.id,
            output.state,
            truncate(&output.reason, 80)
        );

        // Send the message to the user if the task produced one.
        if let Some(msg) = &output.message {
            if let Err(e) = signal.send(&config.primary_contact, msg).await {
                error!("Failed to send task message to {}: {e:#}", config.primary_contact);
            }
        }

        // Update lifecycle state and react to any events.
        match tasks.record_result(task, &output.state).await {
            Ok(TaskLifecycleEvent::Continue) => {}

            Ok(TaskLifecycleEvent::AutoPaused { description, idle_count }) => {
                info!("Task auto-paused after {idle_count} idle runs: {description}");
                let notice = format!(
                    "⏸ I paused a background task after {idle_count} consecutive runs \
                     with nothing to report:\n\n\"{description}\"\n\n\
                     Reply 'resume task {id}' or use list_tasks to manage it.",
                    id = task.id
                );
                if let Err(e) = signal.send(&config.primary_contact, &notice).await {
                    error!("Failed to send auto-pause notice: {e:#}");
                }
            }

            Ok(TaskLifecycleEvent::Expired { description }) => {
                info!("Task expired: {description}");
                let notice = format!(
                    "⏰ A scheduled task has expired and been stopped:\n\n\"{description}\""
                );
                if let Err(e) = signal.send(&config.primary_contact, &notice).await {
                    error!("Failed to send expiry notice: {e:#}");
                }
            }

            Ok(TaskLifecycleEvent::OneShot { description }) => {
                info!("One-shot task completed: {description}");
            }

            Err(e) => error!("Failed to record task result: {e:#}"),
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Task runner — isolated LLM tool loop
// ---------------------------------------------------------------------------

/// System prompt that teaches the task runner about the three states.
const TASK_RUNNER_SYSTEM: &str = "\
You are an autonomous task executor for a personal AI daemon. \
Execute the assigned task using available tools, then return a single JSON object.

## Required output format
Your FINAL message must be ONLY this JSON — no prose around it:
{
  \"state\": \"acted\" | \"still_waiting\" | \"nothing_to_do\",
  \"reason\": \"Brief explanation of what you found or did\",
  \"message\": \"Message to send to the user, or null\"
}

## State definitions — choose carefully

### \"acted\"
You completed meaningful work or found something worth reporting.
- Found new information the user would want to know
- Completed an action (command run, file written, data fetched and processed)
- The specific condition being monitored has now occurred
→ Set `message` to a clear, user-facing summary. Resets the idle counter.

### \"still_waiting\"
Nothing happened yet, but you are actively monitoring for a SPECIFIC expected future event.
- You checked and the event has not occurred yet — but you are confident it will
- Examples: no game release date announced yet, package not yet delivered,
  website not yet updated, PR not yet merged
- You can clearly articulate what you are waiting for
→ Set `message` to null. The idle counter does NOT increment — task stays alive indefinitely.
Only use this state when there is a concrete, identifiable future event you expect to occur.

### \"nothing_to_do\"
No meaningful action was possible and there is no specific future event being awaited.
- The monitored situation seems to have run its course
- No clear condition left to wait for
- Repeated checks have found nothing and there is no reason to expect change
→ Set `message` to null. Idle counter increments.
After enough consecutive `nothing_to_do` runs, the task is automatically paused.

## Decision guide
Ask yourself: \"Is there a specific future event I am confident will eventually occur?\"
  YES and it hasn't happened yet  → \"still_waiting\"
  YES and it just happened        → \"acted\"
  NO but I did something useful   → \"acted\"
  NO and nothing useful to do     → \"nothing_to_do\"

## Available tools
shell_exec, file_read, file_write, web_fetch, memory_search, memory_store
Do NOT schedule new tasks or modify the task list from within a task execution.";

/// Run the LLM tool-loop for a single task and return structured output.
async fn execute_task(
    task: &TaskEntry,
    llm: &Arc<dyn LlmClient>,
    memory: &Arc<MemoryManager>,
    max_tokens: usize,
) -> Result<TaskRunOutput> {
    let executor = ToolExecutor::for_task_runner(llm.clone(), memory.clone(), max_tokens);
    let tool_defs = executor.task_runner_tool_definitions();

    let user_msg = format!(
        "Execute this task: {}\n\nContext:\n- Run #{}\n- Consecutive idle runs: {}\n- Last run: {}",
        task.description,
        task.run_count + 1,
        task.idle_count,
        task.last_run
            .map(|t| t.format("%Y-%m-%d %H:%M UTC").to_string())
            .unwrap_or_else(|| "never".to_string()),
    );

    let mut messages = vec![ApiMessage {
        role: "user".to_string(),
        content: MessageContent::Text(user_msg),
    }];

    let final_text = loop {
        let resp = llm
            .complete_with_tools(ClaudeRequest {
                system: TASK_RUNNER_SYSTEM,
                messages: messages.clone(),
                max_tokens,
                tools: &tool_defs,
            })
            .await
            .context("task runner LLM call failed")?;

        if resp.tool_calls.is_empty() {
            break resp.text;
        }

        messages.push(ApiMessage {
            role: "assistant".to_string(),
            content: MessageContent::Blocks(resp.raw_blocks),
        });

        let mut results = Vec::new();
        for tc in &resp.tool_calls {
            let output = executor.execute(&tc.name, &tc.input).await;
            results.push(ContentBlock::ToolResult {
                tool_use_id: tc.id.clone(),
                content: output,
            });
        }

        messages.push(ApiMessage {
            role: "user".to_string(),
            content: MessageContent::Blocks(results),
        });
    };

    parse_task_output(&final_text)
}

fn parse_task_output(text: &str) -> Result<TaskRunOutput> {
    let trimmed = text.trim();
    let v: serde_json::Value = serde_json::from_str(trimmed)
        .or_else(|_| {
            if let (Some(s), Some(e)) = (trimmed.find('{'), trimmed.rfind('}')) {
                serde_json::from_str(&trimmed[s..=e])
            } else {
                Err(serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "no JSON object",
                )))
            }
        })
        .unwrap_or_else(|_| {
            warn!("Task runner returned non-JSON; treating as nothing_to_do");
            serde_json::json!({
                "state": "nothing_to_do",
                "reason": trimmed,
                "message": null
            })
        });

    Ok(TaskRunOutput {
        state: TaskRunState::from_str(v["state"].as_str().unwrap_or("nothing_to_do")),
        reason: v["reason"].as_str().unwrap_or("").to_string(),
        message: v["message"]
            .as_str()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string()),
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_task_output_acted() {
        let json = r#"{"state":"acted","reason":"Found update.","message":"New version released!"}"#;
        let out = parse_task_output(json).unwrap();
        assert_eq!(out.state, TaskRunState::Acted);
        assert_eq!(out.message, Some("New version released!".to_string()));
    }

    #[test]
    fn test_parse_task_output_still_waiting() {
        let json = r#"{"state":"still_waiting","reason":"No announcement yet.","message":null}"#;
        let out = parse_task_output(json).unwrap();
        assert_eq!(out.state, TaskRunState::StillWaiting);
        assert!(out.message.is_none());
    }

    #[test]
    fn test_parse_task_output_nothing_to_do() {
        let json = r#"{"state":"nothing_to_do","reason":"Nothing changed.","message":null}"#;
        let out = parse_task_output(json).unwrap();
        assert_eq!(out.state, TaskRunState::NothingToDo);
    }

    #[test]
    fn test_parse_task_output_embedded_json() {
        let text = r#"Here is my result: {"state":"acted","reason":"Done.","message":"All good!"}"#;
        let out = parse_task_output(text).unwrap();
        assert_eq!(out.state, TaskRunState::Acted);
    }

    #[test]
    fn test_parse_task_output_non_json_fallback() {
        let out = parse_task_output("I did some stuff but forgot the format").unwrap();
        assert_eq!(out.state, TaskRunState::NothingToDo);
    }
}
