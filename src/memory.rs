use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use tracing::{debug, info};
use uuid::Uuid;

use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    vectors_config::Config as VCVariant,
    Condition, CountPointsBuilder, CreateCollectionBuilder, DeletePointsBuilder, Distance, Filter,
    PointId, PointStruct, PointsIdsList, ScrollPointsBuilder, SearchPointsBuilder,
    SetPayloadPointsBuilder, UpsertPointsBuilder, VectorParams, VectorsConfig, Value,
    point_id::PointIdOptions,
    value::Kind,
    vectors_output::VectorsOptions as OutputVectorsOptions,
};

use crate::config::{MemoryConfig, QdrantConfig};
use crate::embeddings::{cosine_similarity, EmbeddingClient};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum MemoryKind {
    Episodic,
    Semantic,
    Procedural,
    Reflection,
}

impl MemoryKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryKind::Episodic => "episodic",
            MemoryKind::Semantic => "semantic",
            MemoryKind::Procedural => "procedural",
            MemoryKind::Reflection => "reflection",
        }
    }

    pub fn from_str_safe(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "episodic" => MemoryKind::Episodic,
            "semantic" => MemoryKind::Semantic,
            "procedural" => MemoryKind::Procedural,
            "reflection" => MemoryKind::Reflection,
            _ => MemoryKind::Episodic,
        }
    }
}

impl fmt::Display for MemoryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone)]
pub struct Memory {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub kind: MemoryKind,
    pub importance: f64,
    pub source: String,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: i64,
}

#[derive(Debug, Clone)]
pub struct MemoryRef {
    pub id: String,
    pub content: String,
    pub kind: MemoryKind,
    pub importance: f64,
    pub similarity: f64,
}

// ---------------------------------------------------------------------------
// MemoryManager — backed by Qdrant
// ---------------------------------------------------------------------------

pub struct MemoryManager {
    client: Arc<Qdrant>,
    embedder: Arc<dyn EmbeddingClient>,
    config: MemoryConfig,
    collection: String,
}

impl MemoryManager {
    /// Connect to Qdrant and create the collection if it doesn't exist.
    pub async fn new(
        qdrant_config: QdrantConfig,
        embedder: Arc<dyn EmbeddingClient>,
        config: MemoryConfig,
    ) -> Result<Self> {
        let client = Qdrant::from_url(&qdrant_config.url)
            .build()
            .context("Failed to connect to Qdrant")?;
        let client = Arc::new(client);
        let collection = qdrant_config.collection.clone();

        if !client
            .collection_exists(&collection)
            .await
            .context("Failed to check Qdrant collection existence")?
        {
            client
                .create_collection(
                    CreateCollectionBuilder::new(&collection).vectors_config(VectorsConfig {
                        config: Some(VCVariant::Params(VectorParams {
                            size: 1024,
                            distance: Distance::Cosine.into(),
                            ..Default::default()
                        })),
                    }),
                )
                .await
                .context("Failed to create Qdrant collection")?;
            info!("Created Qdrant collection '{collection}'");
        }

        Ok(Self { client, embedder, config, collection })
    }

    /// Embed query text and return top-k most similar memories.
    pub async fn search(&self, query: &str, top_k: Option<usize>) -> Result<Vec<MemoryRef>> {
        let k = top_k.unwrap_or(self.config.top_k_retrieval);
        debug!("Memory search: query={} k={k}", truncate(query, 50));

        let embedding = self.embedder.embed(query).await?;
        let result = self
            .client
            .search_points(
                SearchPointsBuilder::new(&self.collection, embedding, k as u64)
                    .with_payload(true),
            )
            .await
            .context("Qdrant search failed")?;

        let mut refs = Vec::new();
        for scored in result.result {
            let id = point_id_str(&scored.id);
            if id.is_empty() {
                continue;
            }
            refs.push(MemoryRef {
                id,
                content: get_str(&scored.payload, "content")
                    .unwrap_or_default()
                    .to_string(),
                kind: MemoryKind::from_str_safe(
                    get_str(&scored.payload, "kind").unwrap_or("episodic"),
                ),
                importance: get_f64(&scored.payload, "importance").unwrap_or(0.5),
                similarity: scored.score as f64,
            });
        }

        debug!("Memory search returned {} results", refs.len());
        Ok(refs)
    }

    /// Embed content and store as a new memory point.
    pub async fn store(
        &self,
        content: &str,
        kind: MemoryKind,
        source: &str,
        importance: f64,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let embedding = self.embedder.embed(content).await?;
        let now = Utc::now().to_rfc3339();

        let mut payload: HashMap<String, Value> = HashMap::new();
        payload.insert("content".to_string(), content.to_string().into());
        payload.insert("kind".to_string(), kind.as_str().to_string().into());
        payload.insert("importance".to_string(), importance.into());
        payload.insert("source".to_string(), source.to_string().into());
        payload.insert("created_at".to_string(), now.clone().into());
        payload.insert("last_accessed".to_string(), now.into());
        payload.insert("access_count".to_string(), 0i64.into());

        self.client
            .upsert_points(UpsertPointsBuilder::new(
                &self.collection,
                vec![PointStruct::new(id.clone(), embedding, payload)],
            ))
            .await
            .context("Qdrant upsert failed")?;

        debug!("Stored {kind} memory {id}: {}", truncate(content, 80));
        Ok(id)
    }

    /// Update last_accessed for all given IDs (same timestamp — "now").
    pub async fn mark_accessed(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let mut payload: HashMap<String, Value> = HashMap::new();
        payload.insert("last_accessed".to_string(), Utc::now().to_rfc3339().into());

        self.client
            .set_payload(
                SetPayloadPointsBuilder::new(&self.collection, payload)
                    .points_selector(PointsIdsList {
                        ids: ids.iter().map(|s| s.clone().into()).collect(),
                    }),
            )
            .await
            .context("Qdrant set_payload failed")?;

        Ok(())
    }

    /// Decay importance scores: importance *= exp(-age_days / decay_days).
    /// Returns the number of memories whose importance was updated.
    pub async fn decay(&self) -> Result<usize> {
        info!(
            "Running importance decay (half-life={}d)",
            self.config.importance_decay_days
        );
        let now = Utc::now();
        let result = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection)
                    .with_payload(true)
                    .with_vectors(false)
                    .limit(50_000u32),
            )
            .await
            .context("Qdrant scroll failed during decay")?;

        let mut count = 0usize;
        for point in result.result {
            let id = point_id_str(&point.id);
            if id.is_empty() {
                continue;
            }
            let importance = get_f64(&point.payload, "importance").unwrap_or(0.5);
            let age_days = get_str(&point.payload, "created_at")
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| (now - dt.with_timezone(&Utc)).num_seconds() as f64 / 86400.0)
                .unwrap_or(0.0);

            let new_importance =
                importance * (-age_days / self.config.importance_decay_days).exp();
            if (new_importance - importance).abs() > 0.001 {
                let mut p: HashMap<String, Value> = HashMap::new();
                p.insert("importance".to_string(), new_importance.into());
                let _ = self
                    .client
                    .set_payload(
                        SetPayloadPointsBuilder::new(&self.collection, p)
                            .points_selector(PointsIdsList { ids: vec![id.into()] }),
                    )
                    .await;
                count += 1;
            }
        }

        info!("Decayed importance for {count} memories");
        Ok(count)
    }

    /// Delete memories below min_importance or episodic memories past TTL.
    /// Returns the number of memories deleted.
    pub async fn prune(&self) -> Result<usize> {
        info!(
            "Pruning memories (min_importance={}, episodic_ttl={}d)",
            self.config.min_importance_to_keep, self.config.ttl_days_episodic
        );
        let now = Utc::now();
        let result = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection)
                    .with_payload(true)
                    .with_vectors(false)
                    .limit(50_000u32),
            )
            .await
            .context("Qdrant scroll failed during prune")?;

        let mut to_delete: Vec<String> = Vec::new();
        for point in result.result {
            let id = point_id_str(&point.id);
            if id.is_empty() {
                continue;
            }
            let importance = get_f64(&point.payload, "importance").unwrap_or(0.5);
            let kind = get_str(&point.payload, "kind").unwrap_or("episodic");
            let age_days = get_str(&point.payload, "created_at")
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|dt| (now - dt.with_timezone(&Utc)).num_seconds() as f64 / 86400.0)
                .unwrap_or(0.0);
            let expired = kind == "episodic" && age_days > self.config.ttl_days_episodic;
            if importance < self.config.min_importance_to_keep || expired {
                to_delete.push(id);
            }
        }

        let count = to_delete.len();
        if !to_delete.is_empty() {
            self.delete(&to_delete).await?;
        }
        info!("Pruned {count} memories");
        Ok(count)
    }

    /// Find clusters of similar episodic memories (cosine >= 0.85, min size 3).
    /// Fetches embeddings from Qdrant and clusters in-process.
    pub async fn find_episodic_clusters(&self) -> Result<Vec<Vec<Memory>>> {
        let result = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection)
                    .filter(Filter::must(vec![Condition::matches(
                        "kind",
                        "episodic".to_string(),
                    )]))
                    .with_payload(true)
                    .with_vectors(true)
                    .limit(1000u32),
            )
            .await
            .context("Qdrant scroll failed during clustering")?;

        let mut memories: Vec<Memory> = Vec::new();
        for point in result.result {
            let id = point_id_str(&point.id);
            if id.is_empty() {
                continue;
            }
            let embedding = match &point.vectors {
                Some(v) => match &v.vectors_options {
                    #[allow(deprecated)]
                    Some(OutputVectorsOptions::Vector(dense)) => dense.data.clone(),
                    _ => continue,
                },
                None => continue,
            };
            if embedding.is_empty() {
                continue;
            }
            memories.push(Memory {
                id,
                content: get_str(&point.payload, "content")
                    .unwrap_or_default()
                    .to_string(),
                embedding,
                kind: MemoryKind::Episodic,
                importance: get_f64(&point.payload, "importance").unwrap_or(0.5),
                source: get_str(&point.payload, "source").unwrap_or("").to_string(),
                created_at: parse_ts(get_str(&point.payload, "created_at").unwrap_or("")),
                last_accessed: parse_ts(
                    get_str(&point.payload, "last_accessed").unwrap_or(""),
                ),
                access_count: get_i64(&point.payload, "access_count").unwrap_or(0),
            });
        }

        // Greedy clustering
        let n = memories.len();
        let mut used = vec![false; n];
        let mut clusters: Vec<Vec<Memory>> = Vec::new();
        for i in 0..n {
            if used[i] {
                continue;
            }
            let mut cluster = vec![i];
            for j in (i + 1)..n {
                if !used[j]
                    && cosine_similarity(&memories[i].embedding, &memories[j].embedding) >= 0.85
                {
                    cluster.push(j);
                }
            }
            if cluster.len() >= 3 {
                for &idx in &cluster {
                    used[idx] = true;
                }
                clusters.push(cluster.into_iter().map(|idx| memories[idx].clone()).collect());
            }
        }

        Ok(clusters)
    }

    /// Delete memories by ID.
    pub async fn delete(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let point_ids: Vec<PointId> = ids.iter().map(|s| s.clone().into()).collect();
        self.client
            .delete_points(
                DeletePointsBuilder::new(&self.collection)
                    .points(PointsIdsList { ids: point_ids }),
            )
            .await
            .context("Qdrant delete failed")?;
        Ok(())
    }

    /// Return up to N memories (order not guaranteed — used for heartbeat reflection).
    pub async fn recent(&self, n: usize) -> Result<Vec<MemoryRef>> {
        let result = self
            .client
            .scroll(
                ScrollPointsBuilder::new(&self.collection)
                    .with_payload(true)
                    .with_vectors(false)
                    .limit(n as u32),
            )
            .await
            .context("Qdrant scroll failed")?;

        Ok(result
            .result
            .iter()
            .filter_map(|point| {
                let id = point_id_str(&point.id);
                if id.is_empty() {
                    return None;
                }
                Some(MemoryRef {
                    id,
                    content: get_str(&point.payload, "content")
                        .unwrap_or_default()
                        .to_string(),
                    kind: MemoryKind::from_str_safe(
                        get_str(&point.payload, "kind").unwrap_or("episodic"),
                    ),
                    importance: get_f64(&point.payload, "importance").unwrap_or(0.5),
                    similarity: 0.0,
                })
            })
            .collect())
    }

    /// Total number of memory points in the collection.
    pub async fn count(&self) -> Result<usize> {
        let result = self
            .client
            .count(CountPointsBuilder::new(&self.collection))
            .await
            .context("Qdrant count failed")?;
        Ok(result.result.map(|r| r.count as usize).unwrap_or(0))
    }

    /// Format MemoryRef list as a prompt section.
    pub fn format_memories_for_prompt(memories: &[MemoryRef]) -> String {
        if memories.is_empty() {
            return String::from("[No relevant memories found]");
        }
        let mut lines = vec!["## Retrieved Memories".to_string()];
        for (i, mem) in memories.iter().enumerate() {
            lines.push(format!(
                "{}. [{}|imp={:.2}|sim={:.2}] {}",
                i + 1,
                mem.kind.as_str(),
                mem.importance,
                mem.similarity,
                mem.content
            ));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Payload helpers
// ---------------------------------------------------------------------------

fn point_id_str(id: &Option<PointId>) -> String {
    match id {
        Some(PointId { point_id_options: Some(PointIdOptions::Uuid(s)) }) => s.clone(),
        Some(PointId { point_id_options: Some(PointIdOptions::Num(n)) }) => n.to_string(),
        _ => String::new(),
    }
}

fn get_str<'a>(payload: &'a HashMap<String, Value>, key: &str) -> Option<&'a str> {
    match &payload.get(key)?.kind {
        Some(Kind::StringValue(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn get_f64(payload: &HashMap<String, Value>, key: &str) -> Option<f64> {
    match &payload.get(key)?.kind {
        Some(Kind::DoubleValue(d)) => Some(*d),
        Some(Kind::IntegerValue(i)) => Some(*i as f64),
        _ => None,
    }
}

fn get_i64(payload: &HashMap<String, Value>, key: &str) -> Option<i64> {
    match &payload.get(key)?.kind {
        Some(Kind::IntegerValue(i)) => Some(*i),
        _ => None,
    }
}

fn parse_ts(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
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
    use crate::embeddings::cosine_similarity;

    #[test]
    fn test_memory_kind_roundtrip() {
        for kind in &[
            MemoryKind::Episodic,
            MemoryKind::Semantic,
            MemoryKind::Procedural,
            MemoryKind::Reflection,
        ] {
            assert_eq!(*kind, MemoryKind::from_str_safe(kind.as_str()));
        }
    }

    #[test]
    fn test_memory_kind_from_str_unknown() {
        assert_eq!(MemoryKind::from_str_safe("unknown"), MemoryKind::Episodic);
    }

    #[test]
    fn test_format_memories_for_prompt_empty() {
        assert_eq!(
            MemoryManager::format_memories_for_prompt(&[]),
            "[No relevant memories found]"
        );
    }

    #[test]
    fn test_format_memories_for_prompt() {
        let mems = vec![
            MemoryRef {
                id: "a".to_string(),
                content: "The user likes Rust".to_string(),
                kind: MemoryKind::Semantic,
                importance: 0.8,
                similarity: 0.92,
            },
            MemoryRef {
                id: "b".to_string(),
                content: "Discussed Qdrant yesterday".to_string(),
                kind: MemoryKind::Episodic,
                importance: 0.6,
                similarity: 0.75,
            },
        ];
        let result = MemoryManager::format_memories_for_prompt(&mems);
        assert!(result.contains("Retrieved Memories"));
        assert!(result.contains("The user likes Rust"));
        assert!(result.contains("semantic"));
        assert!(result.contains("0.92"));
    }

    #[test]
    fn test_cosine_similarity_basic() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        let c = vec![0.0f32, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
