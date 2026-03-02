// Integration tests for MemoryManager backed by Qdrant.
//
// These tests require a running Qdrant instance:
//   docker run -p 6334:6334 qdrant/qdrant
//
// They are marked #[ignore] by default so `cargo test` skips them.
// Run with: cargo test -- --ignored
//
// Or set QDRANT_URL to point at your instance:
//   QDRANT_URL=http://localhost:6334 cargo test -- --ignored

use std::sync::Arc;

use clawd::config::{MemoryConfig, QdrantConfig};
use clawd::embeddings::mock::MockEmbeddingClient;
use clawd::memory::{MemoryKind, MemoryManager, MemoryRef};

fn test_memory_config() -> MemoryConfig {
    MemoryConfig {
        max_memories: 1000,
        top_k_retrieval: 5,
        importance_decay_days: 30.0,
        min_importance_to_keep: 0.1,
        ttl_days_episodic: 90.0,
    }
}

fn qdrant_config() -> QdrantConfig {
    QdrantConfig {
        url: std::env::var("QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6334".to_string()),
        // Use a unique collection per test run to avoid collisions
        collection: format!(
            "test_memories_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ),
    }
}

#[tokio::test]
#[ignore]
async fn test_store_and_search_roundtrip() {
    let embedder = Arc::new(MockEmbeddingClient::new(1024));
    let manager =
        MemoryManager::new(qdrant_config(), embedder, test_memory_config())
            .await
            .unwrap();

    let id = manager
        .store("The user prefers Rust over Python", MemoryKind::Semantic, "test", 0.8)
        .await
        .unwrap();
    assert!(!id.is_empty());

    let results = manager.search("Rust programming", Some(5)).await.unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().any(|r| r.content == "The user prefers Rust over Python"));
}

#[tokio::test]
#[ignore]
async fn test_count_and_delete() {
    let embedder = Arc::new(MockEmbeddingClient::new(1024));
    let manager =
        MemoryManager::new(qdrant_config(), embedder, test_memory_config())
            .await
            .unwrap();

    let id = manager
        .store("To be deleted", MemoryKind::Episodic, "test", 0.5)
        .await
        .unwrap();

    assert_eq!(manager.count().await.unwrap(), 1);
    manager.delete(&[id]).await.unwrap();
    assert_eq!(manager.count().await.unwrap(), 0);
}

#[tokio::test]
#[ignore]
async fn test_mark_accessed() {
    let embedder = Arc::new(MockEmbeddingClient::new(1024));
    let manager =
        MemoryManager::new(qdrant_config(), embedder, test_memory_config())
            .await
            .unwrap();

    let id = manager
        .store("Test memory", MemoryKind::Episodic, "test", 0.5)
        .await
        .unwrap();
    // Should not error
    manager.mark_accessed(&[id]).await.unwrap();
}

#[tokio::test]
#[ignore]
async fn test_recent() {
    let embedder = Arc::new(MockEmbeddingClient::new(1024));
    let manager =
        MemoryManager::new(qdrant_config(), embedder, test_memory_config())
            .await
            .unwrap();

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
#[ignore]
async fn test_find_episodic_clusters_empty() {
    let embedder = Arc::new(MockEmbeddingClient::new(1024));
    let manager =
        MemoryManager::new(qdrant_config(), embedder, test_memory_config())
            .await
            .unwrap();
    let clusters = manager.find_episodic_clusters().await.unwrap();
    assert!(clusters.is_empty());
}

// ---------------------------------------------------------------------------
// Unit tests (no network — testing pure logic)
// ---------------------------------------------------------------------------

#[test]
fn test_format_memories_for_prompt() {
    let memories = vec![MemoryRef {
        id: "x".to_string(),
        content: "User likes tea".to_string(),
        kind: MemoryKind::Semantic,
        importance: 0.9,
        similarity: 0.85,
    }];
    let formatted = MemoryManager::format_memories_for_prompt(&memories);
    assert!(formatted.contains("User likes tea"));
    assert!(formatted.contains("0.85"));
    assert!(formatted.contains("semantic"));
}
