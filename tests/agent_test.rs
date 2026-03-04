// Integration tests for Agent — run without any external services.
// These use an in-memory DuckDB database and mock LLM/signal clients.

use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

use clawd::agent::Agent;
use clawd::claude::mock::MockLlm;
use clawd::config::*;
use clawd::conversation_store::ConversationStore;
use clawd::embeddings::mock::MockEmbeddingClient;
use clawd::jobs::BackgroundJobStore;
use clawd::memory::MemoryManager;
use clawd::skills::SkillManager;
use clawd::tasks::TaskManager;

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
        signal: None,
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

async fn build_agent(llm_responses: Vec<String>) -> Arc<Agent> {
    let config = test_config();
    let embedder = Arc::new(MockEmbeddingClient::new(4));
    let memory = Arc::new(
        MemoryManager::new(
            Path::new(":memory:"),
            embedder,
            config.memory.clone(),
            4, // tiny dim for tests
        )
        .await
        .unwrap(),
    );
    let llm = Arc::new(MockLlm::new(llm_responses));
    let soul = Arc::new(RwLock::new(test_soul()));
    let tasks = Arc::new(TaskManager::in_memory().await.unwrap());
    let skills = Arc::new(SkillManager::in_memory().await.unwrap());
    let (jobs, _) = BackgroundJobStore::new();
    let conv_store = Arc::new(ConversationStore::new(memory.connection()).unwrap());
    Arc::new(Agent::new(
        soul, llm, memory, tasks, skills, jobs, conv_store, config, vec![], None,
    ))
}

#[tokio::test]
async fn test_agent_handle_message() {
    let agent = build_agent(vec![
        "Hello! I'm your assistant.".to_string(),
        "[]".to_string(), // memory extraction
    ])
    .await;

    let response = agent
        .handle_message("user", "Hi there!", None, "tui")
        .await
        .unwrap();
    assert_eq!(response.reply_text, "Hello! I'm your assistant.");
}

#[tokio::test]
async fn test_agent_multiple_messages() {
    let agent = build_agent(vec![
        "First response.".to_string(),
        "[]".to_string(),
        "Second response.".to_string(),
        "[]".to_string(),
    ])
    .await;

    let r1 = agent.handle_message("user", "Message 1", None, "tui").await.unwrap();
    assert_eq!(r1.reply_text, "First response.");
    let r2 = agent.handle_message("user", "Message 2", None, "tui").await.unwrap();
    assert_eq!(r2.reply_text, "Second response.");

    let ctx = agent.context.read().await;
    assert_eq!(ctx.len(), 4);
}

#[tokio::test]
async fn test_agent_context_tracking() {
    let agent = build_agent(vec!["Response.".to_string(), "[]".to_string()]).await;

    {
        let ctx = agent.context.read().await;
        assert!(ctx.is_empty());
        assert_eq!(ctx.total_tokens(), 0);
    }

    agent.handle_message("user", "Hello", None, "tui").await.unwrap();

    {
        let ctx = agent.context.read().await;
        assert_eq!(ctx.len(), 2);
        assert!(ctx.total_tokens() > 0);
    }
}
