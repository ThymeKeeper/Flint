use anyhow::Result;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use crate::claude::{ApiMessage, ClaudeRequest, LlmClient, MessageContent};
use crate::config::{AppConfig, Soul};
use crate::memory::{MemoryKind, MemoryManager, MemoryRef};
use crate::signal::SignalClient;

// ---------------------------------------------------------------------------
// Heartbeat loop
// ---------------------------------------------------------------------------

/// Run the heartbeat loop forever, firing every `config.heartbeat.interval_secs` seconds.
///
/// Each heartbeat:
///   1. Maintenance:    decay importance, prune old memories
///   2. Consolidation:  cluster similar episodics → synthesize → semantic
///   3. Reflection:     Claude reflects on recent memories
///   4. Proactive:      optionally send a message to the primary contact
pub async fn run_heartbeat(
    soul: Arc<RwLock<Soul>>,
    llm: Arc<dyn LlmClient>,
    memory: Arc<MemoryManager>,
    signal: Arc<dyn SignalClient>,
    config: AppConfig,
) {
    let interval =
        std::time::Duration::from_secs(config.heartbeat.interval_secs);
    info!("Heartbeat started (interval={}s)", config.heartbeat.interval_secs);

    loop {
        tokio::time::sleep(interval).await;
        info!("Heartbeat firing");

        // 1. Maintenance
        if let Err(e) = run_maintenance(&memory).await {
            error!("Heartbeat maintenance failed: {e:#}");
        }

        // 2. Consolidation
        if let Err(e) = run_consolidation(&memory, &llm).await {
            error!("Heartbeat consolidation failed: {e:#}");
        }

        // 3. Reflection + 4. Proactive messaging
        match run_reflection(&soul, &llm, &memory, &signal, &config).await {
            Ok(Some(reflection)) => info!("Heartbeat reflection: {}", truncate(&reflection, 120)),
            Ok(None) => {}
            Err(e) => error!("Heartbeat reflection failed: {e:#}"),
        }

        info!("Heartbeat complete");
    }
}

// ---------------------------------------------------------------------------
// Phase 1: Maintenance
// ---------------------------------------------------------------------------

async fn run_maintenance(memory: &MemoryManager) -> Result<()> {
    info!("Heartbeat: running maintenance");
    let decayed = memory.decay().await?;
    info!("Decayed {decayed} memories");
    let pruned = memory.prune().await?;
    info!("Pruned {pruned} memories");
    let count = memory.count().await?;
    info!("Total memories after maintenance: {count}");
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2: Consolidation
// ---------------------------------------------------------------------------

async fn run_consolidation(memory: &MemoryManager, llm: &Arc<dyn LlmClient>) -> Result<()> {
    info!("Heartbeat: running consolidation");
    let clusters = memory.find_episodic_clusters().await?;
    if clusters.is_empty() {
        info!("No episodic clusters found");
        return Ok(());
    }
    info!("Found {} clusters for consolidation", clusters.len());

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
                    .store(&synthesis, MemoryKind::Semantic, "consolidation", (avg_importance + 0.1).min(1.0))
                    .await?;
                let ids: Vec<String> = cluster.iter().map(|m| m.id.clone()).collect();
                memory.delete(&ids).await?;
                info!("Consolidated {} episodic memories into semantic", cluster.len());
            }
            Err(e) => warn!("Failed to synthesize cluster: {e:#}"),
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 3 + 4: Reflection and proactive messaging
// ---------------------------------------------------------------------------

async fn run_reflection(
    soul: &Arc<RwLock<Soul>>,
    llm: &Arc<dyn LlmClient>,
    memory: &MemoryManager,
    signal: &Arc<dyn SignalClient>,
    config: &AppConfig,
) -> Result<Option<String>> {
    info!("Heartbeat: running reflection");

    let recent = memory.recent(20).await?;
    let memories_text = format_memories_for_reflection(&recent);

    let soul_guard = soul.read().await;
    let prompt = format!(
        "{}\n\n## Recent Memories\n{memories_text}",
        soul_guard.heartbeat_prompt
    );

    let response = llm
        .complete(ClaudeRequest {
            system: &format!(
                "You are {}, reflecting during a periodic heartbeat.",
                soul_guard.name
            ),
            messages: vec![ApiMessage {
                role: "user".to_string(),
                content: MessageContent::Text(prompt),
            }],
            max_tokens: 1000,
            tools: &[],
        })
        .await?;

    match parse_reflection_response(&response) {
        Some((reflection, proactive_message)) => {
            memory
                .store(&reflection, MemoryKind::Reflection, "heartbeat", 0.6)
                .await?;
            info!("Stored reflection memory");

            if let Some(msg) = &proactive_message {
                let contact = &config.primary_contact;
                info!("Sending proactive message to {contact}");
                if let Err(e) = signal.send(contact, msg).await {
                    error!("Failed to send proactive message: {e:#}");
                }
            }

            Ok(Some(reflection))
        }
        None => {
            warn!("Could not parse reflection JSON; storing raw response");
            memory
                .store(&response, MemoryKind::Reflection, "heartbeat", 0.5)
                .await?;
            Ok(Some(response))
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_reflection_response(response: &str) -> Option<(String, Option<String>)> {
    let trimmed = response.trim();
    // Try direct parse
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return extract_reflection_fields(&v);
    }
    // Try to find embedded JSON object
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&trimmed[start..=end]) {
            return extract_reflection_fields(&v);
        }
    }
    None
}

fn extract_reflection_fields(v: &serde_json::Value) -> Option<(String, Option<String>)> {
    let reflection = v.get("reflection")?.as_str()?.to_string();
    let proactive = v
        .get("proactive_message")
        .and_then(|p| if p.is_null() { None } else { p.as_str().map(|s| s.to_string()) });
    Some((reflection, proactive))
}

fn format_memories_for_reflection(memories: &[MemoryRef]) -> String {
    if memories.is_empty() {
        return "[No recent memories]".to_string();
    }
    memories
        .iter()
        .enumerate()
        .map(|(i, m)| format!("{}. [{}] {}", i + 1, m.kind.as_str(), m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

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
    fn test_parse_reflection_valid() {
        let json = r#"{"reflection": "Interesting patterns.", "proactive_message": null}"#;
        let (reflection, proactive) = parse_reflection_response(json).unwrap();
        assert_eq!(reflection, "Interesting patterns.");
        assert!(proactive.is_none());
    }

    #[test]
    fn test_parse_reflection_with_message() {
        let json =
            r#"{"reflection": "Good progress.", "proactive_message": "How did the project go?"}"#;
        let (reflection, proactive) = parse_reflection_response(json).unwrap();
        assert_eq!(reflection, "Good progress.");
        assert_eq!(proactive, Some("How did the project go?".to_string()));
    }

    #[test]
    fn test_parse_reflection_embedded_json() {
        let text = r#"Here is my reflection: {"reflection": "All well.", "proactive_message": null}"#;
        let (reflection, _) = parse_reflection_response(text).unwrap();
        assert_eq!(reflection, "All well.");
    }

    #[test]
    fn test_parse_reflection_invalid() {
        assert!(parse_reflection_response("not json").is_none());
    }

    #[test]
    fn test_format_memories_for_reflection_empty() {
        assert_eq!(format_memories_for_reflection(&[]), "[No recent memories]");
    }

    #[test]
    fn test_format_memories_for_reflection() {
        let mems = vec![
            MemoryRef {
                id: "a".to_string(),
                content: "User discussed Rust".to_string(),
                kind: MemoryKind::Episodic,
                importance: 0.8,
                similarity: 1.0,
                pinned: false,
                created_at: chrono::Utc::now(),
            },
            MemoryRef {
                id: "b".to_string(),
                content: "Prefers concise comms".to_string(),
                kind: MemoryKind::Semantic,
                importance: 0.9,
                similarity: 1.0,
                pinned: false,
                created_at: chrono::Utc::now(),
            },
        ];
        let result = format_memories_for_reflection(&mems);
        assert!(result.contains("1. [episodic] User discussed Rust"));
        assert!(result.contains("2. [semantic] Prefers concise comms"));
    }
}
