// Integration tests for Agent backed by Qdrant.
//
// These tests require a running Qdrant instance and are #[ignore] by default.
// Run with: cargo test -- --ignored

use std::sync::Arc;
use tokio::sync::RwLock;

use clawd::agent::Agent;
use clawd::claude::mock::MockLlm;
use clawd::config::*;
use clawd::embeddings::mock::MockEmbeddingClient;
use clawd::memory::MemoryManager;
use clawd::signal::mock::MockSignalClient;

fn test_config() -> AppConfig {
    AppConfig {
        soul_path: "/tmp/soul.yaml".to_string(),
        claude: ClaudeConfig {
            model: "test".to_string(),
            max_tokens: 100,
            context_limit: 200000,
            compaction_threshold: 0.75,
        },
        voyage: VoyageConfig {
            model: "test".to_string(),
            dimensions: 1024,
        },
        signal: SignalConfig {
            base_url: "http://localhost:8080".to_string(),
            phone_number: "+15551234567".to_string(),
            allowed_senders: vec!["+15559876543".to_string()],
            poll_interval_secs: 1,
        },
        memory: MemoryConfig {
            max_memories: 1000,
            top_k_retrieval: 5,
            importance_decay_days: 30.0,
            min_importance_to_keep: 0.1,
            ttl_days_episodic: 90.0,
        },
        heartbeat: HeartbeatConfig { interval_secs: 3600 },
        qdrant: QdrantConfig {
            url: std::env::var("QDRANT_URL")
                .unwrap_or_else(|_| "http://localhost:6334".to_string()),
            collection: format!(
                "test_agent_{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis()
            ),
        },
    }
}

fn test_soul() -> Soul {
    Soul {
        name: "TestBot".to_string(),
        persona: "A test bot for integration testing.".to_string(),
        values: vec!["accuracy".to_string(), "speed".to_string()],
        communication_style: "Concise.".to_string(),
        proactive_interests: vec!["testing".to_string()],
        heartbeat_prompt: "Reflect on tests.".to_string(),
    }
}

async fn build_agent(llm_responses: Vec<String>) -> (Arc<Agent>, Arc<MockSignalClient>) {
    let config = test_config();
    let embedder = Arc::new(MockEmbeddingClient::new(1024));
    let memory = Arc::new(
        MemoryManager::new(config.qdrant.clone(), embedder, config.memory.clone())
            .await
            .unwrap(),
    );
    let llm = Arc::new(MockLlm::new(llm_responses));
    let signal = Arc::new(MockSignalClient::new(vec!["+15559876543".to_string()]));
    let soul = Arc::new(RwLock::new(test_soul()));
    let agent = Arc::new(Agent::new(soul, llm, memory, signal.clone(), config));
    (agent, signal)
}

#[tokio::test]
#[ignore]
async fn test_agent_handle_message() {
    let (agent, _signal) = build_agent(vec![
        "Hello! I'm your assistant.".to_string(),
        "[]".to_string(), // memory extraction
    ])
    .await;

    let response = agent
        .handle_message("+15559876543", "Hi there!")
        .await
        .unwrap();
    assert_eq!(response, "Hello! I'm your assistant.");
}

#[tokio::test]
#[ignore]
async fn test_agent_multiple_messages() {
    let (agent, _signal) = build_agent(vec![
        "First response.".to_string(),
        "[]".to_string(),
        "Second response.".to_string(),
        "[]".to_string(),
    ])
    .await;

    let r1 = agent.handle_message("+15559876543", "Message 1").await.unwrap();
    assert_eq!(r1, "First response.");
    let r2 = agent.handle_message("+15559876543", "Message 2").await.unwrap();
    assert_eq!(r2, "Second response.");

    let ctx = agent.context.read().await;
    assert_eq!(ctx.len(), 4);
}

#[tokio::test]
#[ignore]
async fn test_agent_context_tracking() {
    let (agent, _signal) =
        build_agent(vec!["Response.".to_string(), "[]".to_string()]).await;

    {
        let ctx = agent.context.read().await;
        assert!(ctx.is_empty());
        assert_eq!(ctx.total_tokens(), 0);
    }

    agent.handle_message("+15559876543", "Hello").await.unwrap();

    {
        let ctx = agent.context.read().await;
        assert_eq!(ctx.len(), 2);
        assert!(ctx.total_tokens() > 0);
    }
}
