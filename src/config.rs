use anyhow::{Context, Result};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

// ---------------------------------------------------------------------------
// AppConfig — loaded from config.toml
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    pub soul_path: String,
    pub claude: ClaudeConfig,
    pub voyage: VoyageConfig,
    pub signal: SignalConfig,
    pub memory: MemoryConfig,
    pub heartbeat: HeartbeatConfig,
    pub qdrant: QdrantConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    pub collection: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClaudeConfig {
    pub model: String,
    pub max_tokens: usize,
    pub context_limit: usize,
    pub compaction_threshold: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VoyageConfig {
    pub model: String,
    pub dimensions: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SignalConfig {
    pub base_url: String,
    pub phone_number: String,
    pub allowed_senders: Vec<String>,
    pub poll_interval_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MemoryConfig {
    pub max_memories: usize,
    pub top_k_retrieval: usize,
    pub importance_decay_days: f64,
    pub min_importance_to_keep: f64,
    pub ttl_days_episodic: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HeartbeatConfig {
    pub interval_secs: u64,
}

impl AppConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: AppConfig =
            toml::from_str(&text).with_context(|| "Failed to parse config.toml")?;
        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// Soul — loaded from soul.yaml, hot-reloadable
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Soul {
    pub name: String,
    pub persona: String,
    pub values: Vec<String>,
    pub communication_style: String,
    pub proactive_interests: Vec<String>,
    pub heartbeat_prompt: String,
}

impl Soul {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read soul file: {}", path.display()))?;
        let soul: Soul =
            serde_yaml::from_str(&text).with_context(|| "Failed to parse soul.yaml")?;
        Ok(soul)
    }

    pub fn to_system_prompt(&self) -> String {
        let values = self.values.join(", ");
        let interests = self.proactive_interests.join(", ");
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();

        format!(
            "You are {name}.\n\n\
             {persona}\n\n\
             Core values: {values}\n\
             Communication style: {style}\n\
             Proactive interests: {interests}\n\
             Today's date: {today}\n",
            name = self.name,
            persona = self.persona.trim(),
            values = values,
            style = self.communication_style,
            interests = interests,
            today = today,
        )
    }
}

// ---------------------------------------------------------------------------
// Soul hot-reload watcher
// ---------------------------------------------------------------------------

/// Spawns a background task that watches `soul_path` for changes and reloads
/// the soul into the provided `Arc<RwLock<Soul>>`. Returns the watcher handle
/// which must be kept alive for the duration of the program.
pub fn spawn_soul_watcher(
    soul_path: PathBuf,
    soul: Arc<RwLock<Soul>>,
) -> Result<RecommendedWatcher> {
    let reload_path = soul_path.clone();
    let reload_soul = soul.clone();

    let mut watcher =
        notify::recommended_watcher(move |res: std::result::Result<Event, notify::Error>| {
            match res {
                Ok(event) => {
                    let dominated_by_modify = matches!(
                        event.kind,
                        EventKind::Modify(_) | EventKind::Create(_)
                    );
                    if dominated_by_modify {
                        info!("Soul file changed, reloading...");
                        match Soul::load(&reload_path) {
                            Ok(new_soul) => {
                                // We are in a sync callback so we use blocking write.
                                // The RwLock is tokio's, but blocking_write works from
                                // sync contexts.
                                let mut guard = reload_soul.blocking_write();
                                *guard = new_soul;
                                info!("Soul reloaded successfully");
                            }
                            Err(e) => {
                                warn!("Failed to reload soul: {e:#}");
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Soul watcher error: {e}");
                }
            }
        })?;

    // Watch the parent directory (works better across editors that do atomic saves)
    let watch_dir = soul_path
        .parent()
        .unwrap_or_else(|| Path::new("."));
    watcher.watch(watch_dir, RecursiveMode::NonRecursive)?;
    info!("Watching soul file at {}", soul_path.display());

    Ok(watcher)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn sample_soul_yaml() -> &'static str {
        r#"
name: "TestBot"
persona: "You are a test bot."
values: ["honesty", "speed"]
communication_style: "Terse."
proactive_interests: ["testing"]
heartbeat_prompt: "Reflect."
"#
    }

    #[test]
    fn test_soul_load_and_system_prompt() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(sample_soul_yaml().as_bytes()).unwrap();
        let soul = Soul::load(f.path()).unwrap();
        assert_eq!(soul.name, "TestBot");
        let prompt = soul.to_system_prompt();
        assert!(prompt.contains("TestBot"));
        assert!(prompt.contains("honesty, speed"));
        assert!(prompt.contains("Terse."));
    }

    #[test]
    fn test_config_load() {
        let toml_str = r#"
soul_path = "/tmp/soul.yaml"

[claude]
model = "claude-opus-4-6"
max_tokens = 4096
context_limit = 200000
compaction_threshold = 0.75

[voyage]
model = "voyage-3"
dimensions = 1024

[signal]
base_url = "http://localhost:8080"
phone_number = "+15551234567"
allowed_senders = ["+15559876543"]
poll_interval_secs = 5

[memory]
max_memories = 10000
top_k_retrieval = 10
importance_decay_days = 30.0
min_importance_to_keep = 0.1
ttl_days_episodic = 90

[heartbeat]
interval_secs = 3600

[qdrant]
url = "http://localhost:6334"
collection = "memories"
"#;
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(toml_str.as_bytes()).unwrap();
        let config = AppConfig::load(f.path()).unwrap();
        assert_eq!(config.claude.model, "claude-opus-4-6");
        assert_eq!(config.voyage.dimensions, 1024);
        assert_eq!(config.signal.allowed_senders.len(), 1);
    }
}
