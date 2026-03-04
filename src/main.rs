use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use clawd::agent::Agent;
use clawd::claude::ClaudeClient;
use clawd::config::{AppConfig, Soul};
use clawd::conversation_store::{ConversationStore, StoredTurn};
use clawd::embeddings::{LocalEmbeddingClient, EMBEDDING_DIM};
use clawd::heartbeat;
use clawd::jobs::BackgroundJobStore;
use clawd::memory::MemoryManager;
use clawd::observer::AgentObserver;
use clawd::setup;
use clawd::signal::{SignalClient, SignalTcpRpcClient, TuiSignalClient};
use clawd::signal_daemon;
use clawd::skills::SkillManager;
use clawd::tasks::TaskManager;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();

    info!("clawd starting up");

    // ── SIGINT handler — Ctrl+C no longer exits; use Ctrl+Q ──────────────────
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigint = signal(SignalKind::interrupt())
            .context("Failed to install SIGINT handler")?;
        tokio::spawn(async move {
            while sigint.recv().await.is_some() {
                println!("  (use ctrl+q to quit)");
            }
        });
    }

    // ── Locate config ────────────────────────────────────────────────────────
    let config_path = resolve_config_path()?;

    info!("Config loaded from {}", config_path.display());
    let config = AppConfig::load(&config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

    // ── API keys ─────────────────────────────────────────────────────────────
    let anthropic_key = config
        .resolve_anthropic_key()
        .context("No Anthropic API key found. Set ANTHROPIC_API_KEY or run clawd --setup.")?;

    // ── Soul ──────────────────────────────────────────────────────────────────
    let soul_path = PathBuf::from(&config.soul_path);
    let soul = Soul::load(&soul_path)
        .with_context(|| format!("Failed to load soul from {}", soul_path.display()))?;
    info!("Soul loaded: {}", soul.name);
    let agent_name = soul.name.clone();
    let soul = Arc::new(RwLock::new(soul));
    let _soul_watcher = clawd::config::spawn_soul_watcher(soul_path, soul.clone())
        .context("Failed to start soul watcher")?;

    // ── LLM client ───────────────────────────────────────────────────────────
    let llm: Arc<dyn clawd::claude::LlmClient> =
        Arc::new(ClaudeClient::new(anthropic_key, config.claude.clone()));

    // ── Embedding client (bundled BGE-small-en-v1.5-Q, no API key needed) ──────
    let embedder: Arc<dyn clawd::embeddings::EmbeddingClient> =
        Arc::new(LocalEmbeddingClient::new().context("Failed to initialize local embeddings")?);

    // ── Memory (DuckDB) ───────────────────────────────────────────────────────
    let db_path = PathBuf::from(&config.db_path);
    let memory = Arc::new(
        MemoryManager::new(&db_path, embedder, config.memory.clone(), EMBEDDING_DIM)
            .await
            .context("Failed to initialize DuckDB memory")?,
    );
    info!("Memory database ready at {}", db_path.display());

    // ── Task + Skill managers (share the same DuckDB connection as memory) ────
    let tasks = Arc::new(
        TaskManager::new(memory.connection())
            .await
            .context("Failed to initialize task manager")?,
    );
    let skills = Arc::new(
        SkillManager::new(memory.connection())
            .await
            .context("Failed to initialize skill manager")?,
    );

    // ── Signal-setup skill: seed when unconfigured, remove when configured ──────
    sync_signal_setup_skill(&skills, &config, &config_path).await;

    // ── Conversation store (shares same DuckDB connection) ────────────────────
    let conv_store = Arc::new(
        ConversationStore::new(memory.connection())
            .context("Failed to initialize conversation store")?,
    );

    // ── Load persisted history ────────────────────────────────────────────────
    let history = conv_store.load_recent("default", 200).unwrap_or_else(|e| {
        warn!("Failed to load conversation history: {e:#}");
        Vec::new()
    });
    info!("Loaded {} conversation turns from history", history.len());

    // ── Background job store ──────────────────────────────────────────────────
    let (job_store, mut job_notify_rx) = BackgroundJobStore::new();

    // ── TUI signal client ─────────────────────────────────────────────────────
    let (tui_channels, user_input_tx, agent_update_rx) = clawd::tui::create_channels();
    let tui_client = Arc::new(TuiSignalClient::new(tui_channels));

    // Push persisted history to TUI before it starts rendering.
    {
        let tui_turns: Vec<(String, String)> = history
            .iter()
            .filter_map(|t| {
                let display_role = if t.role == "assistant" {
                    agent_name.clone()
                } else {
                    "You".to_string()
                };
                let content = display_text_for_turn(t);
                if content.trim().is_empty() { None } else { Some((display_role, content.trim().to_string())) }
            })
            .collect();
        if !tui_turns.is_empty() {
            tui_client.push_history(tui_turns);
        }
    }

    // Spawn the TUI on a dedicated blocking thread.  It calls
    // `std::process::exit(0)` on Ctrl+Q, cleaning up the terminal first.
    {
        let agent_name = agent_name.clone();
        tokio::task::spawn_blocking(move || {
            clawd::tui::run_tui(agent_name, user_input_tx, agent_update_rx);
        });
    }

    let tui_dyn: Arc<dyn SignalClient> = tui_client.clone();

    // ── Signal CLI daemon (started if clawd manages the binary) ──────────────
    // If setup downloaded signal-cli into the data directory, we start it as an
    // HTTP daemon here and keep `_signal_daemon` alive for the process lifetime.
    let data_dir = config_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    if let Some(sc) = &config.signal {
        let bin = signal_daemon::bin_path(&data_dir);
        if bin.exists() {
            info!(
                "Managed signal-cli found at {}; starting daemon for {}",
                bin.display(),
                sc.phone_number
            );
            match signal_daemon::start_daemon(&data_dir, &sc.phone_number).await {
                Ok(daemon) => {
                    info!("signal-cli daemon ready on TCP port {}", signal_daemon::TCP_PORT);
                    // Hand ownership to a watchdog task — it restarts signal-cli if it dies,
                    // and notifies the user via TUI if it gives up after repeated failures.
                    let notify_client = tui_client.clone();
                    tokio::spawn(signal_daemon::run_watchdog(daemon, move |msg| {
                        notify_client.push_notification(msg);
                    }));
                }
                Err(e) => {
                    warn!("Failed to start signal-cli daemon: {e:#}. Signal will be unavailable.");
                }
            }
        }
    }

    // ── Optional Signal TCP JSON-RPC client ───────────────────────────────────
    let signal_rest: Option<Arc<dyn SignalClient>> = config.signal.as_ref().map(|sc| {
        info!(
            "Signal TCP JSON-RPC enabled: {} via {}",
            sc.phone_number,
            signal_daemon::daemon_tcp_addr(),
        );
        Arc::new(SignalTcpRpcClient::new(sc)) as Arc<dyn SignalClient>
    });

    // ── Agent ─────────────────────────────────────────────────────────────────
    let agent = Arc::new(Agent::new(
        soul.clone(),
        llm.clone(),
        memory.clone(),
        tasks.clone(),
        skills.clone(),
        job_store,
        conv_store,
        config.clone(),
        history,
        signal_rest.clone(),
    ));

    info!("All components initialized — ready to chat");

    // ── Heartbeat loop (background) ───────────────────────────────────────────
    {
        let memory = memory.clone();
        let tasks = tasks.clone();
        let llm = llm.clone();
        let signal = tui_dyn.clone();
        let cfg = config.clone();
        tokio::spawn(async move {
            heartbeat::run_heartbeat(memory, tasks, llm, signal, cfg).await;
        });
    }

    // ── Main message loop ─────────────────────────────────────────────────────
    // TUI input and job notifications are handled immediately.
    // Signal REST is polled on its own interval — sleeping inside that branch
    // so TUI responsiveness is never affected.
    let signal_poll_interval = std::time::Duration::from_secs(
        if config.poll_interval_secs > 0 { config.poll_interval_secs } else { 3 },
    );
    loop {
        tokio::select! {
            // TUI input — immediate, no sleep
            result = tui_dyn.receive() => {
                match result {
                    Ok(messages) => {
                        for msg in messages {
                            let obs: Option<Arc<dyn AgentObserver>> =
                                Some(tui_client.clone() as Arc<dyn AgentObserver>);
                            handle_turn(&agent, &tui_dyn, &msg.sender, &msg.sender, &msg.text, "tui", obs).await;
                        }
                    }
                    Err(e) => error!("TUI receive error: {e:#}"),
                }
                drain_jobs(&agent, &tui_dyn, signal_rest.as_ref(), &config.primary_contact, &mut job_notify_rx, &tui_client).await;
            }

            // Signal TCP notification client — blocks until a message arrives,
            // so no polling sleep needed. On error, brief sleep to avoid
            // tight loops on repeated connection failures.
            result = receive_or_pending(&signal_rest) => {
                match result {
                    Ok(messages) => {
                        if let Some(sr) = &signal_rest {
                            for msg in messages {
                                if !sr.is_allowed(&msg.sender) {
                                    warn!("Ignoring Signal message from: {}", msg.sender);
                                    continue;
                                }
                                let resp = handle_turn_ret(
                                    &agent, sr, &msg.sender, &msg.sender, &msg.text, "signal", None,
                                ).await;
                                if let Some(r) = resp {
                                    let text = r.display_text();
                                    if !text.trim().is_empty() {
                                        tui_client.push_history(vec![
                                            ("You".to_string(), msg.text.clone()),
                                            (agent.soul.read().await.name.clone(), text),
                                        ]);
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Signal receive error: {e:#}");
                        tokio::time::sleep(signal_poll_interval).await;
                    }
                }
                drain_jobs(&agent, &tui_dyn, signal_rest.as_ref(), &config.primary_contact, &mut job_notify_rx, &tui_client).await;
            }

            // Background job completion — immediate, no sleep
            Some(notification) = job_notify_rx.recv() => {
                let text = notification.to_agent_text();
                tui_client.push_notification(text.clone());
                let obs: Option<Arc<dyn AgentObserver>> =
                    Some(tui_client.clone() as Arc<dyn AgentObserver>);
                let resp = handle_turn_ret(&agent, &tui_dyn, "system", &config.primary_contact, &text, "tui", obs).await;
                if let (Some(r), Some(sr)) = (resp, &signal_rest) {
                    let _ = sr.send(&config.primary_contact, &r.display_text()).await;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Run handle_turn and return the AgentResponse (for cross-channel forwarding).
async fn handle_turn_ret(
    agent: &Agent,
    signal: &Arc<dyn SignalClient>,
    sender: &str,
    recipient: &str,
    text: &str,
    channel: &str,
    observer: Option<Arc<dyn AgentObserver>>,
) -> Option<clawd::agent::AgentResponse> {
    match agent.handle_message(sender, text, observer, channel).await {
        Ok(response) => {
            let display = response.display_text();
            if let Err(e) = signal.send(recipient, &display).await {
                error!("Failed to send response to {recipient}: {e:#}");
            }
            Some(response)
        }
        Err(e) => {
            error!("Failed to handle message from {sender}: {e:#}");
            let _ = signal.send(recipient, "I encountered an error. Please try again.").await;
            None
        }
    }
}

/// Run handle_turn without returning the response.
async fn handle_turn(
    agent: &Agent,
    signal: &Arc<dyn SignalClient>,
    sender: &str,
    recipient: &str,
    text: &str,
    channel: &str,
    observer: Option<Arc<dyn AgentObserver>>,
) {
    handle_turn_ret(agent, signal, sender, recipient, text, channel, observer).await;
}

/// Drain pending job notifications immediately after a user turn.
async fn drain_jobs(
    agent: &Agent,
    tui: &Arc<dyn SignalClient>,
    signal_rest: Option<&Arc<dyn SignalClient>>,
    primary_contact: &str,
    job_notify_rx: &mut tokio::sync::mpsc::Receiver<clawd::jobs::JobNotification>,
    tui_client: &Arc<TuiSignalClient>,
) {
    while let Ok(notification) = job_notify_rx.try_recv() {
        let text = notification.to_agent_text();
        tui.push_notification(text.clone());
        let obs: Option<Arc<dyn AgentObserver>> =
            Some(tui_client.clone() as Arc<dyn AgentObserver>);
        let resp = handle_turn_ret(agent, tui, "system", primary_contact, &text, "tui", obs).await;
        if let (Some(r), Some(sr)) = (resp, signal_rest) {
            let _ = sr.send(primary_contact, &r.display_text()).await;
        }
    }
}

/// Derive the display text for a stored turn for TUI history.
/// For new rows: `content` is clean reply_text, `tool_log` has structured data.
/// For old rows (pre-migration): `content` has the full stored string including tool log block.
fn display_text_for_turn(t: &StoredTurn) -> String {
    if !t.content.trim().is_empty() {
        return t.content.clone();
    }
    // content is empty — derive from tool_log (new rows where agent only called tools)
    if let Some(e) = t.tool_log.iter().find(|e| e.name == "signal_send") {
        return e.summary.clone();
    }
    if !t.tool_log.is_empty() {
        format!(
            "[{}]",
            t.tool_log.iter().map(|e| e.name.as_str()).collect::<Vec<_>>().join(", ")
        )
    } else {
        String::new()
    }
}

/// Await Signal REST messages, or park forever if Signal REST is not configured.
async fn receive_or_pending(
    client: &Option<Arc<dyn SignalClient>>,
) -> Result<Vec<clawd::signal::IncomingMessage>, anyhow::Error> {
    match client {
        Some(c) => c.receive().await,
        None => std::future::pending().await,
    }
}

// ---------------------------------------------------------------------------
// Signal-setup skill lifecycle
// ---------------------------------------------------------------------------

/// Seed the `signal-setup` skill when Signal is not configured so the agent
/// can guide the user through setup interactively. Remove it once configured.
async fn sync_signal_setup_skill(
    skills: &clawd::skills::SkillManager,
    config: &clawd::config::AppConfig,
    config_path: &std::path::Path,
) {
    const SKILL_NAME: &str = "signal-setup";

    if config.signal.is_some() {
        // Signal is active — remove the setup skill if it exists.
        if let Ok(true) = skills.delete(SKILL_NAME).await {
            info!("Removed {} skill (Signal is now configured)", SKILL_NAME);
        }
        return;
    }

    // Derive paths for the skill's playbook.
    let data_dir = config_path
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "~/.clawd".to_string());
    let bin = format!("{data_dir}/bin/signal-cli");
    let sig_data = format!("{data_dir}/signal-data");
    let config_toml = config_path.to_string_lossy().into_owned();
    let download_url = concat!(
        "https://github.com/AsamK/signal-cli/releases/download/v0.14.0/",
        "signal-cli-0.14.0-Linux-native.tar.gz"
    );

    let prompt = format!(
        "You are a Signal setup helper. Your only job is steps 1-4 below. \
         Stop after step 4 and return your findings to the caller — do NOT \
         write any files or edit config yourself.\n\
         \n\
         1. Check if `{bin}` exists. If not, download and install it:\n\
            ```\n\
            mkdir -p {data_dir}/bin\n\
            curl -L {download_url} | tar xz -C {data_dir}/bin signal-cli\n\
            chmod +x {bin}\n\
            ```\n\
         2. Start the link flow (background=true, redirect stderr to stdout):\n\
            `{bin} --config {sig_data} link -n clawd 2>&1`\n\
         3. Wait ~3 seconds, then read the job log file.\n\
         4. Return ONLY:\n\
            - The exact `sgnl://` URI on its own line\n\
            - The complete QR code block exactly as it appears in the log\n\
            - The job log path (so the caller can monitor it)\n\
         \n\
         STOP HERE. Do not run listAccounts. Do not touch {config_toml}. \
         Do not write any files. The caller will handle everything after the user scans.\n"
    );

    match skills
        .create(
            SKILL_NAME,
            "Downloads signal-cli, starts the device link flow, and returns the QR code/URI.",
            &prompt,
            vec![
                "shell_exec".to_string(),
                "file_read".to_string(),
                "file_write".to_string(),
            ],
        )
        .await
    {
        Ok(_) => info!("Seeded {} skill", SKILL_NAME),
        Err(e) => warn!("Failed to seed {} skill: {e:#}", SKILL_NAME),
    }
}

// ---------------------------------------------------------------------------
// Config resolution
// ---------------------------------------------------------------------------

fn resolve_config_path() -> Result<PathBuf> {
    // Parse --config / -c / --setup from argv.
    let args: Vec<String> = std::env::args().collect();
    let mut explicit_path: Option<PathBuf> = None;
    let mut force_setup = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                i += 1;
                if i >= args.len() {
                    bail!("--config requires a path argument");
                }
                explicit_path = Some(PathBuf::from(&args[i]));
            }
            "--setup" => force_setup = true,
            "--help" | "-h" => {
                println!(
                    "Usage: clawd [--config <path>] [--setup]\n\
                     \n\
                     Options:\n\
                     --config <path>  Path to config.toml (default: ~/.clawd/config.toml)\n\
                     --setup          Re-run the first-time setup wizard\n\
                     --help           Show this help"
                );
                std::process::exit(0);
            }
            other => bail!("Unknown argument: {other}. Use --help for usage."),
        }
        i += 1;
    }

    // If --setup or no config exists → run the wizard.
    if force_setup {
        return setup::run_setup_wizard();
    }

    if let Some(path) = explicit_path {
        return Ok(path);
    }

    // Auto-detect default location.
    if let Some(default_path) = setup::default_config_path() {
        if default_path.exists() {
            return Ok(default_path);
        }
        info!("No config found at {} — running first-time setup", default_path.display());
        return setup::run_setup_wizard();
    }

    bail!("Cannot determine home directory. Use --config <path>.");
}
