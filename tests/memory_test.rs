// Integration tests for MemoryManager — run against an in-memory DuckDB
// database with no external services required.

use std::path::Path;
use std::sync::Arc;

use flint::config::MemoryConfig;
use flint::embeddings::mock::MockEmbeddingClient;
use flint::memory::{MemoryKind, MemoryManager, MemoryRef};

const DIM: usize = 4; // tiny dimension so tests run fast

fn test_memory_config() -> MemoryConfig {
    MemoryConfig {
        max_memories: 1000,
        top_k_retrieval: 5,
        importance_decay_days: 30.0,
        min_importance_to_keep: 0.1,
        ttl_days_episodic: 90.0,
    }
}

async fn make_manager() -> MemoryManager {
    let embedder = Arc::new(MockEmbeddingClient::new(DIM));
    MemoryManager::new(Path::new(":memory:"), embedder, test_memory_config(), DIM)
        .await
        .unwrap()
}

#[tokio::test]
async fn test_store_and_search_roundtrip() {
    let manager = make_manager().await;

    let id = manager
        .store(
            "The user prefers Rust over Python",
            MemoryKind::Semantic,
            "test",
            0.8,
        )
        .await
        .unwrap();
    assert!(!id.is_empty());

    let results = manager.search("Rust programming", Some(5)).await.unwrap();
    assert!(!results.is_empty());
    assert!(results
        .iter()
        .any(|r| r.content == "The user prefers Rust over Python"));
}

#[tokio::test]
async fn test_count_and_delete() {
    let manager = make_manager().await;

    let id = manager
        .store("To be deleted", MemoryKind::Episodic, "test", 0.5)
        .await
        .unwrap();

    assert_eq!(manager.count().await.unwrap(), 1);
    manager.delete(&[id]).await.unwrap();
    assert_eq!(manager.count().await.unwrap(), 0);
}

#[tokio::test]
async fn test_mark_accessed() {
    let manager = make_manager().await;

    let id = manager
        .store("Test memory", MemoryKind::Episodic, "test", 0.5)
        .await
        .unwrap();
    manager.mark_accessed(&[(id, 0.8)]).await.unwrap();
}

#[tokio::test]
async fn test_recent() {
    let manager = make_manager().await;

    for i in 0..5 {
        manager
            .store(&format!("Memory {i}"), MemoryKind::Episodic, "test", 0.5)
            .await
            .unwrap();
    }

    let recent = manager.recent(3).await.unwrap();
    assert_eq!(recent.len(), 3);
}

#[tokio::test]
async fn test_find_episodic_clusters_empty() {
    let manager = make_manager().await;
    let clusters = manager.find_episodic_clusters().await.unwrap();
    assert!(clusters.is_empty());
}

#[test]
fn test_format_memories_for_prompt() {
    let memories = vec![MemoryRef {
        id: "x".to_string(),
        content: "User likes tea".to_string(),
        kind: MemoryKind::Semantic,
        importance: 0.9,
        similarity: 0.85,
        pinned: false,
        created_at: chrono::Utc::now(),
    }];
    let formatted = MemoryManager::format_memories_for_prompt(&memories);
    assert!(formatted.contains("User likes tea"));
    assert!(formatted.contains("0.85"));
    assert!(formatted.contains("semantic"));
}
