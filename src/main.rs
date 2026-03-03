use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use clawd::agent::Agent;
use clawd::claude::ClaudeClient;
use clawd::config::{AppConfig, Soul};
use clawd::embeddings::{LocalEmbeddingClient, EMBEDDING_DIM};
use clawd::heartbeat;
use clawd::jobs::BackgroundJobStore;
use clawd::memory::MemoryManager;
use clawd::setup;
use clawd::signal::{SignalClient, TuiSignalClient};
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

    // ── Background job store ──────────────────────────────────────────────────
    let (job_store, mut job_notify_rx) = BackgroundJobStore::new();

    // ── TUI signal client ─────────────────────────────────────────────────────
    let (tui_channels, user_input_tx, agent_update_rx) = clawd::tui::create_channels();
    let signal_client: Arc<dyn SignalClient> = Arc::new(TuiSignalClient::new(tui_channels));

    // Spawn the TUI on a dedicated blocking thread.  It calls
    // `std::process::exit(0)` on Ctrl+Q, cleaning up the terminal first.
    {
        let agent_name = agent_name.clone();
        tokio::task::spawn_blocking(move || {
            clawd::tui::run_tui(agent_name, user_input_tx, agent_update_rx);
        });
    }

    // ── Agent ─────────────────────────────────────────────────────────────────
    let agent = Arc::new(Agent::new(
        soul.clone(),
        llm.clone(),
        memory.clone(),
        tasks.clone(),
        skills.clone(),
        job_store,
        signal_client.clone(),
        config.clone(),
    ));

    info!("All components initialized — ready to chat");

    // ── Heartbeat loop (background) ───────────────────────────────────────────
    {
        let memory = memory.clone();
        let tasks = tasks.clone();
        let llm = llm.clone();
        let signal = signal_client.clone();
        let cfg = config.clone();
        tokio::spawn(async move {
            heartbeat::run_heartbeat(memory, tasks, llm, signal, cfg).await;
        });
    }

    // ── Main message loop ─────────────────────────────────────────────────────
    let poll_interval = std::time::Duration::from_secs(config.poll_interval_secs);
    loop {
        tokio::select! {
            // User / Signal message
            result = signal_client.receive() => {
                match result {
                    Ok(messages) => {
                        for msg in messages {
                            if !signal_client.is_allowed(&msg.sender) {
                                warn!("Ignoring message from: {}", msg.sender);
                                continue;
                            }
                            handle_turn(&agent, &signal_client, &msg.sender, &msg.text).await;
                        }
                    }
                    Err(e) => error!("Failed to receive messages: {e:#}"),
                }
                // Drain any job notifications that completed during this turn.
                while let Ok(notification) = job_notify_rx.try_recv() {
                    let text = notification.to_agent_text();
                    signal_client.push_notification(text.clone());
                    handle_turn(&agent, &signal_client, "system", &text).await;
                }
            }

            // Background job completion
            Some(notification) = job_notify_rx.recv() => {
                let text = notification.to_agent_text();
                signal_client.push_notification(text.clone());
                handle_turn(&agent, &signal_client, "system", &text).await;
            }
        }

        // poll_interval is 0 for TUI/stdio mode (receive() blocks naturally).
        if poll_interval.as_secs() > 0 {
            tokio::time::sleep(poll_interval).await;
        }
    }
}

async fn handle_turn(
    agent: &Agent,
    signal: &Arc<dyn SignalClient>,
    sender: &str,
    text: &str,
) {
    match agent.handle_message(sender, text).await {
        Ok(response) => {
            if let Err(e) = signal.send(sender, &response).await {
                error!("Failed to send response to {sender}: {e:#}");
            }
        }
        Err(e) => {
            error!("Failed to handle message from {sender}: {e:#}");
            let _ = signal.send(sender, "I encountered an error. Please try again.").await;
        }
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
