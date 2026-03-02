use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use clawd::agent::Agent;
use clawd::claude::ClaudeClient;
use clawd::config::{self, AppConfig, Soul};
use clawd::embeddings::VoyageClient;
use clawd::heartbeat;
use clawd::memory::MemoryManager;
use clawd::signal::{SignalClient, SignalRestClient};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("clawd starting up");

    let config_path = parse_args()?;
    let config = AppConfig::load(&config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;
    info!("Config loaded from {}", config_path.display());

    let anthropic_key = std::env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY environment variable not set")?;
    let voyage_key = std::env::var("VOYAGE_API_KEY")
        .context("VOYAGE_API_KEY environment variable not set")?;

    let soul_path = PathBuf::from(&config.soul_path);
    let soul = Soul::load(&soul_path)
        .with_context(|| format!("Failed to load soul from {}", soul_path.display()))?;
    info!("Soul loaded: {}", soul.name);
    let soul = Arc::new(RwLock::new(soul));
    let _soul_watcher = config::spawn_soul_watcher(soul_path, soul.clone())
        .context("Failed to start soul watcher")?;

    // Initialize clients
    let embedder: Arc<dyn clawd::embeddings::EmbeddingClient> =
        Arc::new(VoyageClient::new(voyage_key, &config.voyage));
    let llm: Arc<dyn clawd::claude::LlmClient> =
        Arc::new(ClaudeClient::new(anthropic_key, config.claude.clone()));
    let signal_client: Arc<dyn SignalClient> =
        Arc::new(SignalRestClient::new(&config.signal));

    // Initialize Qdrant-backed memory (async — creates collection if needed)
    let memory = Arc::new(
        MemoryManager::new(
            config.qdrant.clone(),
            embedder.clone(),
            config.memory.clone(),
        )
        .await
        .context("Failed to initialize Qdrant memory")?,
    );
    info!("Memory manager connected to Qdrant at {}", config.qdrant.url);

    // Build the agent
    let agent = Arc::new(Agent::new(
        soul.clone(),
        llm.clone(),
        memory.clone(),
        signal_client.clone(),
        config.clone(),
    ));

    info!("All components initialized");

    // Spawn the heartbeat loop
    {
        let soul = soul.clone();
        let llm = llm.clone();
        let memory = memory.clone();
        let signal = signal_client.clone();
        let cfg = config.clone();
        tokio::spawn(async move {
            heartbeat::run_heartbeat(soul, llm, memory, signal, cfg).await;
        });
    }

    // Run the signal polling loop
    info!("Starting Signal polling loop (interval={}s)", config.signal.poll_interval_secs);
    run_signal_loop(agent, signal_client, &config).await;

    Ok(())
}

fn parse_args() -> Result<PathBuf> {
    let args: Vec<String> = std::env::args().collect();
    let mut config_path = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" | "-c" => {
                i += 1;
                if i >= args.len() {
                    bail!("--config requires a path argument");
                }
                config_path = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                println!("Usage: clawd --config <path>");
                std::process::exit(0);
            }
            other => bail!("Unknown argument: {other}. Use --help for usage."),
        }
        i += 1;
    }
    config_path.context("Missing required argument: --config <path>")
}

async fn run_signal_loop(
    agent: Arc<Agent>,
    signal: Arc<dyn SignalClient>,
    config: &AppConfig,
) {
    let poll_interval =
        std::time::Duration::from_secs(config.signal.poll_interval_secs);
    loop {
        match signal.receive().await {
            Ok(messages) => {
                for msg in messages {
                    if !signal.is_allowed(&msg.sender) {
                        warn!("Ignoring message from non-allowed sender: {}", msg.sender);
                        continue;
                    }
                    info!("Processing message from {}: {}", msg.sender, truncate(&msg.text, 50));

                    let agent = agent.clone();
                    let signal = signal.clone();
                    let sender = msg.sender.clone();
                    let text = msg.text.clone();
                    tokio::spawn(async move {
                        match agent.handle_message(&sender, &text).await {
                            Ok(response) => {
                                if let Err(e) = signal.send(&sender, &response).await {
                                    error!("Failed to send response to {sender}: {e:#}");
                                }
                            }
                            Err(e) => {
                                error!("Failed to handle message from {sender}: {e:#}");
                                let _ = signal
                                    .send(&sender, "I encountered an error. Please try again.")
                                    .await;
                            }
                        }
                    });
                }
            }
            Err(e) => error!("Failed to receive messages from Signal: {e:#}"),
        }
        tokio::time::sleep(poll_interval).await;
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
