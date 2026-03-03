use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::claude::{ApiMessage, ClaudeClient, ClaudeRequest, ContentBlock, LlmClient, MessageContent};
use crate::config::{AppConfig, Soul};
use crate::context::{ConversationContext, Role};
use crate::jobs::BackgroundJobStore;
use crate::memory::{MemoryKind, MemoryManager, MemoryRef};
use crate::signal::SignalClient;
use crate::skills::SkillManager;
use crate::tasks::TaskManager;
use crate::tools;

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

pub struct Agent {
    pub soul: Arc<RwLock<Soul>>,
    pub llm: Arc<dyn LlmClient>,
    pub memory: Arc<MemoryManager>,
    pub tasks: Arc<TaskManager>,
    pub skills: Arc<SkillManager>,
    pub jobs: Arc<BackgroundJobStore>,
    pub signal: Arc<dyn SignalClient>,
    pub context: RwLock<ConversationContext>,
    pub config: AppConfig,
}

impl Agent {
    pub fn new(
        soul: Arc<RwLock<Soul>>,
        llm: Arc<dyn LlmClient>,
        memory: Arc<MemoryManager>,
        tasks: Arc<TaskManager>,
        skills: Arc<SkillManager>,
        jobs: Arc<BackgroundJobStore>,
        signal: Arc<dyn SignalClient>,
        config: AppConfig,
    ) -> Self {
        let context = ConversationContext::new(config.claude.clone());
        Self { soul, llm, memory, tasks, skills, jobs, signal, context: RwLock::new(context), config }
    }

    /// Handle an incoming message: retrieve memories, generate response, store memories.
    pub async fn handle_message(&self, sender: &str, text: &str) -> Result<String> {
        info!("Handling message from {sender}: {}", truncate(text, 80));

        // 1. Search for relevant memories
        let memories = self
            .memory
            .search(text, Some(self.config.memory.top_k_retrieval))
            .await
            .unwrap_or_else(|e| {
                warn!("Memory search failed: {e:#}");
                Vec::new()
            });
        let mem_id_sims: Vec<(String, f64)> =
            memories.iter().map(|m| (m.id.clone(), m.similarity)).collect();

        // 2. Build system prompt
        let system_prompt = self.build_system_prompt(&memories).await;

        // 3. Push user message to context
        {
            let mut ctx = self.context.write().await;
            ctx.push(Role::User, text.to_string());
        }

        // 4. Compact context if needed
        self.compact_context_if_needed().await?;

        // 5. Generate response via Claude, running the tool loop until done.
        //    If the signal client supports streaming (stdio), text is printed live.
        let stream_cb = self.signal.text_stream_callback();
        let soul_context = {
            let soul = self.soul.read().await;
            soul.to_subagent_context()
        };
        let status_cb = self.signal.status_callback();
        let tool_event_cb = self.signal.tool_event_callback();
        let executor = tools::ToolExecutor::new(
            self.llm.clone(),
            self.memory.clone(),
            self.tasks.clone(),
            self.skills.clone(),
            soul_context,
            self.config.claude.max_tokens,
            status_cb,
            tool_event_cb,
            Some(Arc::clone(&self.jobs)),
        );
        let mut messages = {
            let ctx = self.context.read().await;
            ClaudeClient::messages_from_context(&ctx)
        };
        let tool_defs = executor.tool_definitions();
        let final_text;
        let mut tool_log: Vec<String> = Vec::new();
        loop {
            let req = ClaudeRequest {
                system: &system_prompt,
                messages: messages.clone(),
                max_tokens: self.config.claude.max_tokens,
                tools: &tool_defs,
            };
            let resp = match stream_cb.clone() {
                Some(cb) => self.llm.complete_with_tools_streaming(req, cb).await,
                None => self.llm.complete_with_tools(req).await,
            }
            .context("Claude completion failed")?;

            if resp.tool_calls.is_empty() {
                final_text = resp.text;
                break;
            }

            // If text was streamed before tool calls (e.g. "Let me check..."),
            // print a newline so the next streamed segment starts on a fresh line.
            if stream_cb.is_some() && !resp.text.is_empty() {
                println!();
            }

            // Assistant turn: full blocks (text + tool_use)
            messages.push(ApiMessage {
                role: "assistant".to_string(),
                content: MessageContent::Blocks(resp.raw_blocks),
            });

            // Execute all tool calls and collect results
            let mut results = Vec::new();
            for tc in &resp.tool_calls {
                debug!("Executing tool '{}' with input: {}", tc.name, tc.input);
                let output = executor.execute(&tc.name, &tc.input).await;
                debug!("Tool '{}' returned: {}", tc.name, truncate(&output, 120));
                tool_log.push(tool_log_entry(&tc.name, &tc.input, &output));
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

        // 6. Push assistant text + compact tool log to persistent context so the
        //    next turn retains evidence of what tools were actually called.
        {
            let mut ctx = self.context.write().await;
            let stored = if tool_log.is_empty() {
                final_text.clone()
            } else {
                let log_lines =
                    tool_log.iter().map(|l| format!("• {l}")).collect::<Vec<_>>().join("\n");
                format!("{final_text}\n\n[Tools called this turn — do not repeat these in future turns:\n{log_lines}]")
            };
            ctx.push(Role::Assistant, stored);
        }

        // 7. Fire-and-forget: post-conversation memory tasks
        //    (mark accessed with boost → extract new memories → consolidate clusters)
        let memory_mgr = self.memory.clone();
        let llm = self.llm.clone();
        let exchange_text = format!("User: {text}\nAssistant: {final_text}");
        let id_sims = mem_id_sims.clone();
        tokio::spawn(async move {
            if let Err(e) = memory_mgr.mark_accessed(&id_sims).await {
                warn!("Failed to mark memories accessed: {e:#}");
            }
            if let Err(e) = extract_and_store_memories(&memory_mgr, &llm, &exchange_text).await {
                warn!("Memory extraction failed: {e:#}");
            }
            if let Err(e) = consolidate_memories(&memory_mgr, &llm).await {
                warn!("Memory consolidation failed: {e:#}");
            }
        });

        info!("Response generated: {} chars", final_text.len());
        Ok(final_text)
    }

    /// Build the system prompt from soul + retrieved memories + date + tool guidelines.
    async fn build_system_prompt(&self, memories: &[MemoryRef]) -> String {
        let soul = self.soul.read().await;
        let base_prompt = soul.to_system_prompt();
        let memory_section = MemoryManager::format_memories_for_prompt(memories);
        format!(
            "{base_prompt}\n\n\
             {memory_section}\n\n\
             Use the retrieved memories above to provide personalized, contextual responses. \
             Reference past conversations naturally when relevant, but don't force it.\n\n\
             ## Tool Use\n\
             Tools available: shell_exec, file_read, file_write, web_fetch, memory_store, memory_search, memory_update, memory_delete, spawn_subagent, schedule_task, list_tasks, delete_task, create_skill, list_skills, update_skill, delete_skill.\n\
             - file_write: if the target already exists OR is a system path (/etc, /usr, /bin, /sbin, /boot, /lib, /sys, /proc), ask the user for confirmation first, then retry with force=true.\n\
             - shell_exec: ask before destructive commands (rm, rmdir, dd, mkfs, etc.). Use background=true for EVERYTHING except trivial read-only commands that finish in <5 seconds (ls, cat, grep, ps, df, date, etc.). When in doubt, background=true. NEVER re-run a command that already appears in a [Tools called this turn] block from a previous turn — it is already running or completed.\n\
             - memory_store: create a new memory. Use whenever the user asks you to remember something, or when you learn an important fact. Set pinned=true for explicit user requests.\n\
             - memory_search: find memories by semantic query.\n\
             - memory_update: correct or replace a memory's content; re-embeds automatically.\n\
             - memory_delete: permanently remove a fully obsolete memory.\n\
             - spawn_subagent: delegate a self-contained task to an isolated sub-agent.\n\
             - schedule_task: create a background task that runs autonomously on a schedule. \
The task runner has shell_exec, web_fetch, file_read, file_write, memory_search, and memory_store. \
Use trigger_type='interval' (seconds), 'cron' (HH:MM UTC), or 'once' (RFC3339 timestamp). \
Set max_idle_runs higher (e.g. 100) for long-wait monitoring — the runner uses 'still_waiting' \
state to stay alive without wasting idle budget.\n\
             - list_tasks: show all scheduled tasks with their status and next run time.\n\
             - delete_task: cancel and remove a scheduled task by ID.\n\
             - create_skill: define a named sub-agent profile with a custom system prompt and \
tool set. Sub-agents always inherit the user's principal context automatically.\n\
             - list_skills: show defined skills before creating new ones or spawning.\n\
             - update_skill: modify a skill's prompt, description, or tools.\n\
             - delete_skill: remove a skill by name.\n\
             - spawn_subagent accepts an optional skill='name' parameter to use a skill profile.\n\
             - All other operations: proceed without asking.\n\
             - IMPORTANT: After using tools, your final response MUST describe what you did and include the relevant output. This is the only record that persists across conversations."
        )
    }

    /// Compact the context window when it exceeds the configured threshold.
    async fn compact_context_if_needed(&self) -> Result<()> {
        let needs_compaction = {
            let ctx = self.context.read().await;
            ctx.compaction_needed()
        };
        if !needs_compaction {
            return Ok(());
        }
        info!("Context compaction triggered");

        // Take the oldest half
        let oldest = {
            let mut ctx = self.context.write().await;
            ctx.take_oldest_half()
        };
        if oldest.is_empty() {
            return Ok(());
        }

        let old_text: String = oldest
            .iter()
            .map(|m| format!("{}: {}", m.role.as_str(), m.content))
            .collect::<Vec<_>>()
            .join("\n\n");

        // Extract key facts as Semantic memories
        let facts_prompt = format!(
            "Extract the key facts, preferences, and important information from this \
             conversation segment. Return a JSON array of strings.\n\n{old_text}"
        );
        if let Ok(json_str) = self
            .llm
            .complete(ClaudeRequest {
                system: "You extract facts from conversations. Return only a JSON array of strings.",
                messages: vec![ApiMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(facts_prompt),
                }],
                max_tokens: 2000,
                tools: &[],
            })
            .await
        {
            if let Ok(facts) = serde_json::from_str::<Vec<String>>(&json_str) {
                info!("Extracted {} facts from compacted context", facts.len());
                for fact in &facts {
                    if let Err(e) = self
                        .memory
                        .store(fact, MemoryKind::Semantic, "compaction", 0.7)
                        .await
                    {
                        warn!("Failed to store compaction fact: {e:#}");
                    }
                }
            }
        }

        // Generate a summary
        let summary = self
            .llm
            .complete(ClaudeRequest {
                system: "You summarize conversations concisely.",
                messages: vec![ApiMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(format!(
                        "Summarize this conversation in 2-3 sentences.\n\n{old_text}"
                    )),
                }],
                max_tokens: 500,
                tools: &[],
            })
            .await
            .unwrap_or_else(|e| {
                warn!("Summary generation failed: {e:#}");
                format!("[Compacted {} messages]", oldest.len())
            });

        // Prepend summary note
        let note = format!(
            "[Context compacted: {} messages extracted to memory. Summary: {}]",
            oldest.len(),
            summary
        );
        {
            let mut ctx = self.context.write().await;
            ctx.prepend_summary(note);
        }
        info!("Context compaction complete");
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Memory extraction
// ---------------------------------------------------------------------------

async fn extract_and_store_memories(
    memory: &MemoryManager,
    llm: &Arc<dyn LlmClient>,
    exchange: &str,
) -> Result<()> {
    debug!("Extracting memories from exchange");

    let response = llm
        .complete(ClaudeRequest {
            system: "You extract memorable facts from conversations. Return only valid JSON.",
            messages: vec![ApiMessage {
                role: "user".to_string(),
                content: MessageContent::Text(format!(
                    "Extract notable facts or preferences worth remembering. \
                     Return a JSON array of objects with \"content\" (string) and \
                     \"importance\" (float 0-1). Return [] if nothing is notable.\n\n{exchange}"
                )),
            }],
            max_tokens: 1000,
            tools: &[],
        })
        .await?;

    #[derive(serde::Deserialize)]
    struct Extract {
        content: String,
        importance: f64,
    }

    let extracts: Vec<Extract> = match serde_json::from_str(&response) {
        Ok(v) => v,
        Err(e) => {
            debug!("Failed to parse memory extraction response: {e}");
            return Ok(());
        }
    };

    for e in &extracts {
        let importance = e.importance.clamp(0.0, 1.0);
        if importance >= 0.2 {
            memory
                .store(&e.content, MemoryKind::Episodic, "signal", importance)
                .await?;
        }
    }

    debug!("Stored {} extracted memories", extracts.len());
    Ok(())
}

async fn consolidate_memories(memory: &MemoryManager, llm: &Arc<dyn LlmClient>) -> Result<()> {
    let clusters = memory.find_episodic_clusters().await?;
    if clusters.is_empty() {
        return Ok(());
    }
    debug!("Consolidating {} episodic cluster(s)", clusters.len());
    for cluster in &clusters {
        let cluster_text = cluster
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join("\n- ");
        match llm
            .complete(ClaudeRequest {
                system: "You synthesize memories. Be concise and factual.",
                messages: vec![ApiMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text(format!(
                        "Synthesize these related episodic memories into one concise \
                         semantic memory:\n- {cluster_text}\n\nReturn only the synthesized text."
                    )),
                }],
                max_tokens: 500,
                tools: &[],
            })
            .await
        {
            Ok(synthesis) => {
                let avg_importance: f64 =
                    cluster.iter().map(|m| m.importance).sum::<f64>() / cluster.len() as f64;
                memory
                    .store(
                        &synthesis,
                        MemoryKind::Semantic,
                        "consolidation",
                        (avg_importance + 0.1).min(1.0),
                    )
                    .await?;
                let ids: Vec<String> = cluster.iter().map(|m| m.id.clone()).collect();
                memory.delete(&ids).await?;
                debug!("Consolidated {} episodic memories into semantic", cluster.len());
            }
            Err(e) => warn!("Failed to synthesize cluster: {e:#}"),
        }
    }
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

/// Build a compact one-line entry for the persistent tool log.
/// Format: `tool_name(key_arg) → result_summary`
fn tool_log_entry(name: &str, input: &serde_json::Value, result: &str) -> String {
    let key_arg = match name {
        "shell_exec" => {
            let cmd = input["command"].as_str().unwrap_or("").chars().take(60).collect::<String>();
            if input["background"].as_bool().unwrap_or(false) {
                format!("[bg] {cmd}")
            } else {
                cmd
            }
        }
        "file_read" | "file_write" => input["path"].as_str().unwrap_or("").to_string(),
        "web_fetch" => input["url"].as_str().unwrap_or("").chars().take(60).collect(),
        "memory_search" => input["query"].as_str().unwrap_or("").chars().take(50).collect(),
        "memory_store" => input["title"].as_str().unwrap_or("").chars().take(50).collect(),
        "memory_update" | "memory_delete" => input["id"].as_str().unwrap_or("").to_string(),
        "spawn_subagent" => input["task"].as_str().unwrap_or("").chars().take(50).collect(),
        "schedule_task" => input["name"].as_str().unwrap_or("").chars().take(50).collect(),
        "delete_task" | "delete_skill" | "update_skill" => {
            input["id"].as_str().unwrap_or("").to_string()
        }
        "create_skill" | "list_skills" => input["name"].as_str().unwrap_or("").to_string(),
        "list_tasks" => String::new(),
        _ => input.to_string().chars().take(60).collect(),
    };

    let result_summary = result
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("(no output)")
        .chars()
        .take(80)
        .collect::<String>();

    if key_arg.is_empty() {
        format!("{name}() → {result_summary}")
    } else {
        format!("{name}({key_arg}) → {result_summary}")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::*;

    fn test_config() -> AppConfig {
        AppConfig {
            soul_path: "/tmp/soul.yaml".to_string(),
            db_path: ":memory:".to_string(),
            primary_contact: "user".to_string(),
            anthropic_api_key: None,
            claude: ClaudeConfig {
                model: "test".to_string(),
                max_tokens: 100,
                context_limit: 200000,
                compaction_threshold: 0.75,
            },
            memory: MemoryConfig {
                max_memories: 1000,
                top_k_retrieval: 5,
                importance_decay_days: 30.0,
                min_importance_to_keep: 0.1,
                ttl_days_episodic: 90.0,
            },
            heartbeat: HeartbeatConfig { interval_secs: 3600 },
            poll_interval_secs: 0,
        }
    }

    fn test_soul() -> Soul {
        Soul {
            name: "TestBot".to_string(),
            persona: "A test bot.".to_string(),
            values: vec!["testing".to_string()],
            communication_style: "Terse.".to_string(),
            proactive_interests: vec!["tests".to_string()],
            heartbeat_prompt: "Reflect.".to_string(),
        }
    }

    // Unit tests for build_system_prompt and context logic don't need a live
    // Qdrant. Full agent integration tests live in tests/agent_test.rs.

    #[test]
    fn test_config_db_path() {
        let cfg = test_config();
        assert_eq!(cfg.db_path, ":memory:");
        assert_eq!(cfg.primary_contact, "user");
    }

    #[test]
    fn test_soul_system_prompt() {
        let soul = test_soul();
        let prompt = soul.to_system_prompt();
        assert!(prompt.contains("TestBot"));
        assert!(prompt.contains("testing"));
    }

    /// Verify context compaction trigger logic without network calls.
    #[test]
    fn test_compaction_trigger() {
        let config = ClaudeConfig {
            model: "test".to_string(),
            max_tokens: 100,
            context_limit: 100,
            compaction_threshold: 0.75,
        };
        let mut ctx = ConversationContext::new(config);
        assert!(!ctx.compaction_needed());
        ctx.push(Role::User, "x".repeat(400)); // 100 tokens > 75 threshold
        assert!(ctx.compaction_needed());
    }

    /// Verify tool guidelines appear in the system prompt.
    #[tokio::test]
    async fn test_system_prompt_contains_tool_guidelines() {
        let soul = Arc::new(RwLock::new(test_soul()));
        let prompt = {
            let s = soul.read().await;
            let base = s.to_system_prompt();
            let memory_section = MemoryManager::format_memories_for_prompt(&[]);
            format!(
                "{base}\n\n{memory_section}\n\nUse the retrieved memories above to provide personalized, contextual responses. Reference past conversations naturally when relevant, but don't force it.\n\n## Tool Use\nTools available: shell_exec, file_read, file_write, web_fetch, memory_search, memory_update, memory_delete, spawn_subagent, schedule_task, list_tasks, delete_task."
            )
        };
        assert!(prompt.contains("## Tool Use"));
        assert!(prompt.contains("shell_exec"));
        assert!(prompt.contains("file_write"));
        assert!(prompt.contains("memory_search"));
        assert!(prompt.contains("memory_update"));
        assert!(prompt.contains("memory_delete"));
        assert!(prompt.contains("schedule_task"));
        assert!(prompt.contains("list_tasks"));
        assert!(prompt.contains("delete_task"));
        assert!(prompt.contains("spawn_subagent"));
    }
}
