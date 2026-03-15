use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::claude::{ApiMessage, ClaudeClient, ClaudeRequest, ContentBlock, LlmClient, MessageContent, ToolDefinition};
use crate::config::{AppConfig, Soul};
use crate::context::{ConversationContext, Role};
use crate::conversation_store::{ConversationStore, StoredTurn, ToolLogEntry};
use crate::jobs::BackgroundJobStore;
use crate::memory::{MemoryKind, MemoryManager, MemoryRef};
use crate::observer::AgentObserver;
use crate::skills::SkillManager;
use crate::tasks::TaskManager;
use crate::code_intel::CodeIndex;
use crate::subagents::SubAgentManager;
use crate::tools::{self, ToolExecutorConfig};

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

pub struct Agent {
    pub soul: Arc<RwLock<Soul>>,
    pub llm: Arc<dyn LlmClient>,
    /// Cheap model for background work (memory extraction, consolidation, compaction).
    pub utility_llm: Arc<dyn LlmClient>,
    pub memory: Arc<MemoryManager>,
    pub tasks: Arc<TaskManager>,
    pub skills: Arc<SkillManager>,
    pub jobs: Arc<BackgroundJobStore>,
    pub subagent_mgr: Arc<SubAgentManager>,
    pub conv_store: Arc<ConversationStore>,
    pub context: RwLock<ConversationContext>,
    pub config: AppConfig,
    /// The Signal channel client (if configured). Used by the `signal_send` tool
    /// so the agent can proactively send Signal messages to the primary contact.
    pub signal_client: Option<Arc<dyn crate::signal::SignalClient>>,
    /// Shared tree-sitter code intelligence index.
    pub code_index: Arc<CodeIndex>,
}

impl Agent {
    /// Create the agent, restoring conversation history from `history` into the
    /// context so the first turn has full continuity.
    pub fn new(
        soul: Arc<RwLock<Soul>>,
        llm: Arc<dyn LlmClient>,
        utility_llm: Arc<dyn LlmClient>,
        memory: Arc<MemoryManager>,
        tasks: Arc<TaskManager>,
        skills: Arc<SkillManager>,
        jobs: Arc<BackgroundJobStore>,
        subagent_mgr: Arc<SubAgentManager>,
        conv_store: Arc<ConversationStore>,
        config: AppConfig,
        history: Vec<StoredTurn>,
        signal_client: Option<Arc<dyn crate::signal::SignalClient>>,
        code_index: Arc<CodeIndex>,
    ) -> Self {
        let mut context = ConversationContext::new(config.claude.clone());
        for turn in &history {
            let role = if turn.role == "assistant" { Role::Assistant } else { Role::User };
            // Prefix Signal user turns so the agent knows which channel they came from.
            // For assistant turns, rebuild the tool log block so the LLM retains memory
            // of what tools were called in prior turns.
            let content = match (turn.role.as_str(), turn.channel.as_str()) {
                ("user", "signal") => format!("[Signal] {}", turn.content),
                ("assistant", _)   => rebuild_stored_for_context(&turn.content, &turn.tool_log),
                _                  => turn.content.clone(),
            };
            context.push(role, content);
        }
        Self { soul, llm, utility_llm, memory, tasks, skills, jobs, subagent_mgr, conv_store, context: RwLock::new(context), config, signal_client, code_index }
    }

    /// Handle an incoming message: retrieve memories, generate response, store memories.
    /// `observer` receives streaming text and tool events (TUI only; None for Signal/REST).
    /// `channel` is "tui" or "signal" and is persisted alongside each turn.
    pub async fn handle_message(
        &self,
        sender: &str,
        text: &str,
        observer: Option<Arc<dyn AgentObserver>>,
        channel: &str,
    ) -> Result<AgentResponse> {
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

        // 2. Build executor + tool_defs first so the system prompt reflects live tool set.
        let soul_context = {
            let soul = self.soul.read().await;
            soul.to_subagent_context()
        };
        let executor = tools::ToolExecutor::from_config(ToolExecutorConfig {
            llm:             self.llm.clone(),
            memory:          self.memory.clone(),
            max_tokens:      self.config.claude.max_tokens,
            tasks:           Some(self.tasks.clone()),
            skills:          Some(self.skills.clone()),
            job_store:       Some(Arc::clone(&self.jobs)),
            subagent_mgr:    Some(Arc::clone(&self.subagent_mgr)),
            signal_client:   self.signal_client.clone(),
            primary_contact: self.config.primary_contact.clone(),
            channel:         channel.to_string(),
            soul_context,
            is_signal_reply: channel == "signal",
            user_requested_signal: tools::user_requests_signal(text),
            observer:        observer.clone(),
            code_index:      self.code_index.clone(),
        });
        let tool_defs = executor.tool_definitions();

        // 3. Build system prompt — derives tool list dynamically from live tool_defs.
        let system_prompt = self.build_system_prompt(&memories, channel, Some(&tool_defs)).await;

        // 4. Push user message to context and persist to DB.
        //    Annotate Signal turns so the agent can identify them in the live context,
        //    matching the annotations applied when turns are reloaded from DuckDB.
        {
            let ctx_text = if channel == "signal" {
                format!("[Signal] {text}")
            } else {
                text.to_string()
            };
            let mut ctx = self.context.write().await;
            ctx.push(Role::User, ctx_text);
        }
        if let Err(e) = self.conv_store.push("default", sender, "user", text, channel, &[]) {
            warn!("Failed to persist user turn: {e:#}");
        }

        // 5. Compact context if needed
        self.compact_context_if_needed().await?;

        // 6. Generate response via Claude, running the tool loop until done.
        //    If an observer is provided (TUI), text is streamed live.
        let mut messages = {
            let ctx = self.context.read().await;
            ClaudeClient::messages_from_context(&ctx)
        };
        let mut tool_log: Vec<ToolLogEntry> = Vec::new();
        let final_text = loop {
            let req = ClaudeRequest {
                system: &system_prompt,
                messages: messages.clone(),
                max_tokens: self.config.claude.max_tokens,
                tools: &tool_defs,
            };
            let resp = match observer.clone() {
                Some(obs) => {
                    let cb = Arc::new(move |chunk: String| obs.on_text_chunk(chunk));
                    self.llm.complete_with_tools_streaming(req, cb).await
                }
                None => self.llm.complete_with_tools(req).await,
            };

            // Handle "prompt is too long" errors by truncating the last tool
            // result and retrying, rather than surfacing the error to the user.
            let resp = match resp {
                Ok(r) => r,
                Err(e) => {
                    let msg = format!("{e:#}");
                    if msg.contains("prompt is too long") {
                        warn!("Prompt too long — truncating last tool result and retrying");
                        truncate_last_tool_result(&mut messages);
                        // Compact the persistent context as well.
                        self.compact_context_if_needed().await.ok();
                        continue;
                    }
                    return Err(e).context("Claude completion failed");
                }
            };

            if resp.tool_calls.is_empty() {
                break resp.text;
            }

            // If text was streamed before tool calls (e.g. "Let me check..."),
            // print a newline so the next streamed segment starts on a fresh line.
            if observer.is_some() && !resp.text.is_empty() {
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
                tool_log.push(make_tool_log_entry(&tc.name, &tc.input, &output));
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
        };

        // 6. Push assistant reply + tool log block to in-memory context (for LLM continuity).
        //    Persist reply_text and structured tool_log separately to the DB.
        {
            let mut ctx = self.context.write().await;
            let stored = rebuild_stored_for_context(&final_text, &tool_log);
            ctx.push(Role::Assistant, stored);
            if let Err(e) = self.conv_store.push(
                "default", "assistant", "assistant", &final_text, channel, &tool_log,
            ) {
                warn!("Failed to persist assistant turn: {e:#}");
            }
        }

        // 7. Fire-and-forget: post-conversation memory tasks
        //    (mark accessed with boost → extract new memories → consolidate clusters)
        //    Uses the cheap utility model (Haiku) instead of the main model.
        let memory_mgr = self.memory.clone();
        let llm = self.utility_llm.clone();
        let exchange_text = format!("User: {text}\nAssistant: {final_text}");
        let id_sims = mem_id_sims.clone();
        let channel_owned = channel.to_string();
        tokio::spawn(async move {
            if let Err(e) = memory_mgr.mark_accessed(&id_sims).await {
                warn!("Failed to mark memories accessed: {e:#}");
            }
            if let Err(e) = extract_and_store_memories(&memory_mgr, &llm, &exchange_text, &channel_owned).await {
                warn!("Memory extraction failed: {e:#}");
            }
            if let Err(e) = consolidate_memories(&memory_mgr, &llm).await {
                warn!("Memory consolidation failed: {e:#}");
            }
        });

        info!("Response generated: {} chars", final_text.len());
        Ok(AgentResponse { reply_text: final_text, tool_log })
    }

    /// Build the system prompt from soul + retrieved memories + date + tool guidelines.
    ///
    /// `tool_defs` is `Some(&[ToolDefinition])` in normal operation (executor already built),
    /// or `None` in tests/fallback paths, where a minimal static placeholder is used.
    async fn build_system_prompt(
        &self,
        memories: &[MemoryRef],
        _channel: &str,
        tool_defs: Option<&[ToolDefinition]>,
    ) -> String {
        let soul = self.soul.read().await;
        let base_prompt = soul.to_system_prompt();
        let memory_section = MemoryManager::format_memories_for_prompt(memories);

        // Inject Signal context so the agent accurately knows its channel status.
        let signal_section = if let Some(sc) = &self.config.signal {
            format!(
                "\n## Channels\n\
                 You are reachable via two channels:\n\
                 - **TUI** (local terminal): the primary interactive interface.\n\
                 - **Signal** (phone: {}): messages from authorised senders arrive automatically \
                 and your replies are delivered back through Signal automatically. \
                 You are already linked and the daemon is running.\n\
                 \n\
                 When you receive a Signal message, just return your response — it is \
                 automatically delivered back via Signal. Do NOT call `signal_send` just \
                 because Signal is configured. The `signal_send` tool is ONLY available \
                 (and should ONLY be used) in these three situations:\
                 (a) You are replying to a Signal message and need to send an extra notification.\
                 (b) A background task or scheduler triggered this turn and you have a \
                     result to report to the user's phone.\
                 (c) The user explicitly asked you to notify them on Signal in this TUI \
                     message (e.g. 'send to signal', 'notify me on signal', 'message \
                     me on signal', 'signal me', 'ping me on signal', 'via signal'). \
                 Calling `signal_send` outside these conditions sends an unwanted message \
                 to the user's phone and must be avoided. Do NOT invoke signal-cli via \
                 shell_exec — a daemon owns the data directory and direct invocations will hang.\n",
                sc.phone_number
            )
        } else {
            let config_toml = std::path::Path::new(&self.config.soul_path)
                .parent()
                .map(|p| p.join("config.toml").to_string_lossy().into_owned())
                .unwrap_or_else(|| "~/.flint/config.toml".to_string());
            format!(
                "\n## Channels\n\
                 You currently only have the TUI channel. Signal is not configured.\n\
                 If the user asks to set up Signal, use the `signal-setup` skill to handle \
                 the automated part (download + link + QR code). Once the skill returns the \
                 QR/URI, display it, then guide the user through the rest:\n\
                 - Wait for them to scan and confirm the device appears in Signal\n\
                 - Run: `~/.flint/bin/signal-cli --config ~/.flint/signal-data listAccounts`\n\
                 - If that returns a phone number, ask for their own number (allowed sender)\n\
                 - If listAccounts returns empty or fails, show the raw output and ask the \
                   user to type their phone number manually — do NOT proceed without it\n\
                 - Only once you have a confirmed phone number, append EXACTLY this to \
                   `{config_toml}` (no other fields, no invented field names):\n\
                   poll_interval_secs = 3\n\
                   \n\
                   [signal]\n\
                   phone_number = \"<the number>\"\n\
                   allowed_senders = [\"<their number>\"]\n\
                 - Tell them to restart flint\n"
            )
        };

        let arch_section = "\n## Architecture\n\
             You are a persistent AI daemon. Key facts about your own operation:\n\
             - **Conversation history**: Every turn is persisted to DuckDB. On startup the last \
             200 turns are loaded into your context, so you have full continuity across restarts. \
             Do NOT say \"I only know within this session\" — you have durable memory.\n\
             - **Channel annotations**: Incoming Signal messages are prefixed `[Signal]` in your \
             context. Turns without this prefix are from the TUI. Use these to accurately report \
             which channel a conversation belongs to.\n\
             - **Semantic memory**: Separate from conversation history, important facts are \
             extracted and stored as embeddings in DuckDB after each exchange. Use \
             `memory_search` to query them.\n\
             - **Background tasks**: Scheduled tasks run independently and report via Signal.\n\
             \n\
             ## Sub-Agent Strategy\n\
             You have powerful sub-agent capabilities. Use them aggressively:\n\
             - **Any task requiring more than 2 tool calls**: delegate to a sub-agent.\n\
             - **Research tasks**: always use a sub-agent to preserve your context window.\n\
             - **Multi-file operations**: spawn sub-agents to work on different files in parallel.\n\
             - **Complex analysis**: chain sub-agents where each step builds on the previous using `plan_subagents`.\n\
             - Sub-agents run in the BACKGROUND — you stay responsive to the user.\n\
             - The user can see live sub-agent activity in TUI activity boxes.\n\
             - When a sub-agent finishes, you receive a compact notification with a brief excerpt. Synthesise the result in your own words — never quote the raw sub-agent output verbatim into chat. Report what the user needs to know, concisely, as if you are briefing them on what your colleague found.\n\
             - Use `spawn_subagent` for simple one-off delegations.\n\
             - Use `plan_subagents` for complex workflows with dependencies.\n\
             - Use `list_subagents` to check on running sub-agents.\n\
             - Only handle trivial tasks (quick memory lookups, simple replies) directly.\n";

        // Generate the tool section dynamically from live ToolDefinition structs,
        // so flint always has accurate self-knowledge about its tools.
        // Falls back to a minimal static string when called without live tool defs (e.g. tests).
        let tool_section = if let Some(defs) = tool_defs {
            let auto = crate::tools::tools_to_prompt_section(defs);
            format!(
                "{auto}\
                 - file_write: if the target already exists OR is a system path (/etc, /usr, /bin, /sbin, /boot, /lib, /sys, /proc), ask the user for confirmation first, then retry with force=true.\n\
                 - shell_exec: ask before destructive commands (rm, rmdir, dd, mkfs, etc.). Use background=true for EVERYTHING except trivial read-only commands that finish in <5 seconds (ls, cat, grep, ps, df, date, etc.). When in doubt, background=true. NEVER re-run a command that already appears in a [Tools called this turn] block from a previous turn — it is already running or completed.\n\
                 - memory_store: create a new memory. Use whenever the user asks you to remember something, or when you learn an important fact. Set pinned=true for explicit user requests.\n\
                 - memory_search: find memories by semantic query.\n\
                 - memory_update: correct or replace a memory's content; re-embeds automatically.\n\
                 - memory_delete: permanently remove a fully obsolete memory.\n\
                 - spawn_subagent: runs in the BACKGROUND. Returns immediately with a sub-agent ID. You are notified when it completes. Use for any non-trivial task.\n\
                 - plan_subagents: orchestrate multiple sub-agents with dependencies. Steps without dependencies run in parallel. Use for complex multi-step workflows.\n\
                 - list_subagents: check on currently running sub-agents.\n\
                 - schedule_task: create a background task that runs autonomously on a schedule (uses LLM). The task runner has shell_exec, web_fetch, file_read, file_write, memory_search, and memory_store. Use trigger_type='interval' (seconds), 'cron' (HH:MM UTC), or 'once' (RFC3339 timestamp). Set max_idle_runs higher (e.g. 100) for long-wait monitoring — the runner uses 'still_waiting' state to stay alive without wasting idle budget.\n\
                 - schedule_script_task: create a mechanical background task (NO LLM cost). Runs a shell command and checks output against a regex pattern. Prefer this over schedule_task when the task is just 'run a command and check the result'. Use {{output}} in message_template.\n\
                 - list_tasks: show all scheduled tasks with their status and next run time.\n\
                 - delete_task: cancel and remove a scheduled task by ID.\n\
                 - create_skill: define a named sub-agent profile with a custom system prompt and tool set. Sub-agents always inherit the user's principal context automatically.\n\
                 - list_skills: show defined skills before creating new ones or spawning.\n\
                 - update_skill: modify a skill's prompt, description, or tools.\n\
                 - delete_skill: remove a skill by name.\n\
                 - spawn_subagent accepts an optional skill='name' parameter to use a skill profile.\n\
                 - code_symbols: list function/struct/class/table definitions in a file or directory WITHOUT reading the full file contents. Use this FIRST to orient yourself in unfamiliar code — it is much cheaper than file_read for understanding structure.\n\
                 - code_goto_definition: find where a symbol is defined. Give a file path, line, and column.\n\
                 - code_find_references: find every usage of a symbol across a project directory.\n\
                 - code_diagnostics: check a file for syntax errors and lint issues (Rust via ra_ap_syntax, Python via rustpython-parser, SQL via sqlparser + sqruff). Enhanced beyond basic tree-sitter — catches real parse errors and SQL lint violations natively without external tools.\n\
                 - When doing coding work, prefer code_symbols to scan structure before reading files. Read only the specific sections you need, not entire files.\n\
                 - All other operations: proceed without asking.\n\
                 - IMPORTANT: After using tools, your final response MUST describe what you did and include the relevant output. This is the only record that persists across conversations."
            )
        } else {
            // Fallback: minimal static string used in tests or edge cases without a live executor.
            "Tools available: shell_exec, file_read, file_write, web_fetch, memory_store, \
             memory_search, memory_update, memory_delete, spawn_subagent, schedule_task, \
             schedule_script_task, list_tasks, delete_task, create_skill, list_skills, \
             update_skill, delete_skill, code_symbols, code_goto_definition, \
             code_find_references, code_diagnostics.".to_string()
        };

        format!(
            "{base_prompt}{signal_section}{arch_section}\
             {memory_section}\n\n\
             Use the retrieved memories above to provide personalized, contextual responses. \
             Reference past conversations naturally when relevant, but don't force it.\n\n\
             ## Tool Use\n\
             {tool_section}"
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

        // Extract key facts as Semantic memories (uses cheap utility model)
        let facts_prompt = format!(
            "Extract the key facts, preferences, and important information from this \
             conversation segment. Return a JSON array of strings.\n\n{old_text}"
        );
        if let Ok(json_str) = self
            .utility_llm
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

        // Generate a summary (uses cheap utility model)
        let summary = self
            .utility_llm
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
// AgentResponse — returned by handle_message
// ---------------------------------------------------------------------------

pub struct AgentResponse {
    /// Raw LLM text — may be empty if the agent only called tools.
    pub reply_text: String,
    pub tool_log: Vec<ToolLogEntry>,
}

impl AgentResponse {
    /// What to show the user: reply_text if non-empty, else derived from tool actions.
    pub fn display_text(&self) -> String {
        let has_signal_send = self.tool_log.iter().any(|e| e.name == "signal_send");

        // If signal_send was called, the real content was already delivered.
        // The reply_text is typically filler ("Sent.", "Done.", etc.) — use
        // the signal_send summary instead.
        let text = if has_signal_send {
            if let Some(e) = self.tool_log.iter().find(|e| e.name == "signal_send") {
                e.summary.clone()
            } else {
                self.reply_text.clone()
            }
        } else if !self.reply_text.is_empty() {
            self.reply_text.clone()
        } else if !self.tool_log.is_empty() {
            format!(
                "[{}]",
                self.tool_log.iter().map(|e| e.name.as_str()).collect::<Vec<_>>().join(", ")
            )
        } else {
            String::new()
        };
        // Strip any [Tools called this turn...] block the model may have
        // parroted from conversation history — it's internal bookkeeping,
        // not user-facing content.
        strip_tool_log_block(&text)
    }
}

/// Remove a trailing `[Tools called this turn ...]` block from model output.
/// The model sometimes mimics this pattern from conversation history.
/// If the block contains a fake signal_send() call, extract the message
/// content so the response isn't silently lost.
fn strip_tool_log_block(text: &str) -> String {
    let (before, block) = if let Some(start) = text.find("\n[Tools called this turn") {
        (text[..start].trim_end(), &text[start..])
    } else if text.starts_with("[Tools called this turn") {
        ("", text.as_ref())
    } else {
        return text.to_string();
    };

    // If the model faked a signal_send(), salvage the message content.
    // Pattern: signal_send(actual message text) → ...
    if !before.is_empty() {
        return before.to_string();
    }
    if let Some(msg) = extract_fake_signal_send(block) {
        return msg;
    }
    String::new()
}

/// Extract the message body from a fake `signal_send(...)` in a tool log block.
fn extract_fake_signal_send(block: &str) -> Option<String> {
    let marker = "signal_send(";
    let start = block.find(marker)? + marker.len();
    let rest = &block[start..];
    // The message ends at ") →" or ")\n" or end of the parenthesized content.
    let end = rest.find(") →")
        .or_else(|| rest.find(")\n"))
        .or_else(|| rest.rfind(')'))
        .unwrap_or(rest.len());
    let msg = rest[..end].trim();
    if msg.is_empty() { None } else { Some(msg.to_string()) }
}

// ---------------------------------------------------------------------------
// Memory extraction
// ---------------------------------------------------------------------------

async fn extract_and_store_memories(
    memory: &MemoryManager,
    llm: &Arc<dyn LlmClient>,
    exchange: &str,
    channel: &str,
) -> Result<()> {
    info!("Extracting memories from exchange");

    let response = llm
        .complete(ClaudeRequest {
            system: "You extract useful facts from conversations for long-term memory. \
                     Extract anything that may be helpful to recall in future sessions. \
                     The importance score handles filtering — err on the side of capturing \
                     potentially useful information. Return only valid JSON.",
            messages: vec![ApiMessage {
                role: "user".to_string(),
                content: MessageContent::Text(format!(
                    "Review this exchange and extract facts that could be valuable to \
                     recall in future sessions. Return a JSON array of objects with \
                     \"content\" (string) and \"importance\" (float 0-1).\n\n\
                     ## Importance scale\n\
                     0.9-1.0: Life events, critical preferences, allergies, key relationships\n\
                     0.7-0.8: Stated preferences, ongoing projects, goals, technical setup\n\
                     0.5-0.6: Opinions, interests, recurring topics worth noting\n\
                     Below 0.5: Not worth storing\n\n\
                     ## DO NOT extract\n\
                     - Greetings, small talk, pleasantries\n\
                     - Things the assistant said or did (tool calls, responses)\n\
                     - Anything the user would not expect to be remembered\n\
                     - Transient context (\"I'm tired\", \"just got home\")\n\
                     - Facts already obvious from the conversation flow\n\n\
                     {exchange}"
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
            warn!("Failed to parse memory extraction response: {e}");
            return Ok(());
        }
    };

    let mut stored = 0;
    for e in &extracts {
        let importance = e.importance.clamp(0.0, 1.0);
        if importance >= 0.5 {
            memory
                .store(&e.content, MemoryKind::Episodic, channel, importance)
                .await?;
            stored += 1;
        }
    }

    info!("Memory extraction: {} candidates, {} stored (threshold 0.5)", extracts.len(), stored);
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

/// When the prompt exceeds the model's context limit, find the largest tool
/// result in the local `messages` vec and truncate it so the retry fits.
fn truncate_last_tool_result(messages: &mut [ApiMessage]) {
    // Cap each tool result at roughly 20k chars (~5k tokens).
    const MAX_RESULT_CHARS: usize = 20_000;

    // Walk backwards to find the most recent user message with tool results.
    for msg in messages.iter_mut().rev() {
        if let MessageContent::Blocks(ref mut blocks) = msg.content {
            let mut truncated_any = false;
            for block in blocks.iter_mut() {
                if let ContentBlock::ToolResult { content, .. } = block {
                    if content.len() > MAX_RESULT_CHARS {
                        let note = format!(
                            "\n[TRUNCATED: output was {} chars, only the first {} are shown. \
                             The full content was too large for the context window. \
                             If you need more of this content, request a specific portion \
                             (e.g. a line range or section) rather than the whole thing.]",
                            content.len(),
                            MAX_RESULT_CHARS
                        );
                        content.truncate(MAX_RESULT_CHARS);
                        content.push_str(&note);
                        truncated_any = true;
                    }
                }
            }
            if truncated_any {
                return;
            }
        }
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

/// Build a `ToolLogEntry` for one tool call.
fn make_tool_log_entry(name: &str, input: &serde_json::Value, result: &str) -> ToolLogEntry {
    let summary = match name {
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
        "schedule_task" | "schedule_script_task" => input["description"].as_str().unwrap_or("").chars().take(50).collect(),
        "delete_task" | "delete_skill" | "update_skill" => {
            input["id"].as_str().unwrap_or("").to_string()
        }
        "create_skill" | "list_skills" => input["name"].as_str().unwrap_or("").to_string(),
        "list_tasks" => String::new(),
        "signal_send" => input["message"].as_str().unwrap_or("").to_string(),
        _ => input.to_string().chars().take(60).collect(),
    };

    let result_line = result
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("(no output)")
        .chars()
        .take(80)
        .collect::<String>();

    ToolLogEntry { name: name.to_string(), summary, result: result_line }
}

/// Reconstruct the full assistant context block from clean reply text + structured tool log.
/// This is what goes into the in-memory ConversationContext so the LLM remembers tool calls.
fn rebuild_stored_for_context(content: &str, tool_log: &[ToolLogEntry]) -> String {
    if tool_log.is_empty() {
        return content.to_string();
    }
    let log_lines = tool_log
        .iter()
        .map(|e| {
            if e.summary.is_empty() {
                format!("• {}() → {}", e.name, e.result)
            } else {
                format!("• {}({}) → {}", e.name, e.summary, e.result)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    if content.is_empty() {
        format!("[Tools called this turn — do not repeat these in future turns:\n{log_lines}]")
    } else {
        format!("{content}\n\n[Tools called this turn — do not repeat these in future turns:\n{log_lines}]")
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
                utility_model: "test".to_string(),
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
            signal: None,
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

    // Unit tests for build_system_prompt and context logic use in-memory DuckDB.
    // Full agent integration tests live in tests/agent_test.rs.

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
            utility_model: "test".to_string(),
        };
        let mut ctx = ConversationContext::new(config);
        assert!(!ctx.compaction_needed());
        ctx.push(Role::User, "x".repeat(400)); // 100 tokens > 75 threshold
        assert!(ctx.compaction_needed());
    }

    /// Verify tool guidelines appear in the system prompt, generated dynamically
    /// from live ToolDefinitions via tools_to_prompt_section.
    #[test]
    fn test_system_prompt_contains_tool_guidelines() {
        use crate::claude::ToolDefinition;
        use crate::tools::tools_to_prompt_section;

        // Build a small set of representative ToolDefinitions — mirrors what the
        // real executor returns so the test doesn't depend on a live executor.
        let defs: Vec<ToolDefinition> = vec![
            ToolDefinition { name: "shell_exec",   description: "Execute a shell command.", input_schema: serde_json::json!({}) },
            ToolDefinition { name: "file_write",   description: "Write content to a file.", input_schema: serde_json::json!({}) },
            ToolDefinition { name: "memory_store", description: "Store a memory.",          input_schema: serde_json::json!({}) },
            ToolDefinition { name: "memory_search",description: "Search memories.",         input_schema: serde_json::json!({}) },
            ToolDefinition { name: "memory_update",description: "Update a memory.",         input_schema: serde_json::json!({}) },
            ToolDefinition { name: "memory_delete",description: "Delete a memory.",         input_schema: serde_json::json!({}) },
            ToolDefinition { name: "spawn_subagent",description: "Delegate a task.",        input_schema: serde_json::json!({}) },
            ToolDefinition { name: "schedule_task", description: "Schedule a task.",        input_schema: serde_json::json!({}) },
            ToolDefinition { name: "list_tasks",    description: "List tasks.",             input_schema: serde_json::json!({}) },
            ToolDefinition { name: "delete_task",   description: "Delete a task.",          input_schema: serde_json::json!({}) },
        ];

        let section = tools_to_prompt_section(&defs);
        // The section must contain the "Tools available:" line and per-tool bullets.
        assert!(section.contains("Tools available:"), "missing Tools available line");
        assert!(section.contains("shell_exec"),    "missing shell_exec");
        assert!(section.contains("file_write"),    "missing file_write");
        assert!(section.contains("memory_search"), "missing memory_search");
        assert!(section.contains("memory_update"), "missing memory_update");
        assert!(section.contains("memory_delete"), "missing memory_delete");
        assert!(section.contains("schedule_task"), "missing schedule_task");
        assert!(section.contains("list_tasks"),    "missing list_tasks");
        assert!(section.contains("delete_task"),   "missing delete_task");
        assert!(section.contains("spawn_subagent"),"missing spawn_subagent");

        // Verify the full prompt carries the ## Tool Use header and the dynamic section.
        let soul = test_soul();
        let base = soul.to_system_prompt();
        let memory_section = MemoryManager::format_memories_for_prompt(&[]);
        let prompt = format!(
            "{base}\n{memory_section}\n\n## Tool Use\n{section}"
        );
        assert!(prompt.contains("## Tool Use"));
        assert!(prompt.contains("shell_exec"));
    }

    #[test]
    fn test_strip_tool_log_trailing() {
        let text = "Hello!\n[Tools called this turn — do not repeat:\n• shell_exec(ls) → ok]";
        assert_eq!(strip_tool_log_block(text), "Hello!");
    }

    #[test]
    fn test_strip_tool_log_only_block() {
        let text = "[Tools called this turn — do not repeat:\n• shell_exec(ls) → ok]";
        assert_eq!(strip_tool_log_block(text), "");
    }

    #[test]
    fn test_strip_tool_log_fake_signal_send() {
        let text = "[Tools called this turn — do not repeat these in future turns:\n\
                     • signal_send(Hey, just checking in!) → Sent to +1234]";
        assert_eq!(strip_tool_log_block(text), "Hey, just checking in!");
    }

    #[test]
    fn test_strip_tool_log_no_block() {
        assert_eq!(strip_tool_log_block("Just a normal reply."), "Just a normal reply.");
    }
}
