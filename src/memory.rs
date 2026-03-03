use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use duckdb::Connection;
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};
use uuid::Uuid;

use crate::config::MemoryConfig;
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
    /// Pinned memories are never decayed and never pruned.
    /// Procedural memories are automatically pinned on creation.
    pub pinned: bool,
}

#[derive(Debug, Clone)]
pub struct MemoryRef {
    pub id: String,
    pub content: String,
    pub kind: MemoryKind,
    pub importance: f64,
    pub similarity: f64,
    pub pinned: bool,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// MemoryManager — backed by DuckDB
// ---------------------------------------------------------------------------

pub struct MemoryManager {
    db: Arc<Mutex<Connection>>,
    embedder: Arc<dyn EmbeddingClient>,
    config: MemoryConfig,
    /// Dimension of stored embeddings (e.g. 1024 for Voyage).
    #[allow(dead_code)]
    embedding_dim: usize,
}

impl MemoryManager {
    /// Open (or create) the DuckDB database and ensure the schema exists.
    /// Pass `db_path = Path::new(":memory:")` for an in-memory database (tests).
    pub async fn new(
        db_path: &Path,
        embedder: Arc<dyn EmbeddingClient>,
        config: MemoryConfig,
        embedding_dim: usize,
    ) -> Result<Self> {
        let conn = if db_path == Path::new(":memory:") {
            Connection::open_in_memory()
        } else {
            Connection::open(db_path)
        }
        .context("Failed to open DuckDB database")?;

        // Embedding stored as JSON text (e.g. "[0.1,0.2,...]") — avoids DuckDB
        // array-type binding limitations and works across all DuckDB versions.
        let ddl = "CREATE TABLE IF NOT EXISTS memories (
                id            VARCHAR PRIMARY KEY,
                content       VARCHAR NOT NULL,
                embedding     VARCHAR,
                kind          VARCHAR NOT NULL,
                importance    DOUBLE  NOT NULL,
                source        VARCHAR NOT NULL,
                created_at    VARCHAR NOT NULL,
                last_accessed VARCHAR NOT NULL,
                access_count  BIGINT  NOT NULL DEFAULT 0,
                pinned        BOOLEAN NOT NULL DEFAULT false
            )";
        conn.execute_batch(ddl)
            .context("Failed to create memories table")?;

        info!(
            "Memory database ready at {} (dim={})",
            db_path.display(),
            embedding_dim
        );

        Ok(Self {
            db: Arc::new(Mutex::new(conn)),
            embedder,
            config,
            embedding_dim,
        })
    }

    /// Expose the underlying DuckDB connection so other managers (e.g. TaskManager)
    /// can share the same connection and avoid write-lock conflicts.
    pub fn connection(&self) -> Arc<Mutex<Connection>> {
        self.db.clone()
    }

    // -----------------------------------------------------------------------
    // search
    // -----------------------------------------------------------------------

    pub async fn search(&self, query: &str, top_k: Option<usize>) -> Result<Vec<MemoryRef>> {
        let k = top_k.unwrap_or(self.config.top_k_retrieval);
        debug!("Memory search: query={} k={k}", truncate(query, 50));

        let query_embedding = self.embedder.embed(query).await?;
        let db = self.db.clone();

        let refs = tokio::task::spawn_blocking(move || -> Result<Vec<MemoryRef>> {
            let conn = db.lock().unwrap();

            // Fetch all rows; compute similarity in Rust (works with any dim, no SQL casts).
            let mut stmt = conn
                .prepare(
                    "SELECT id, content, kind, importance, embedding, pinned, created_at
                     FROM memories",
                )
                .context("prepare search failed")?;

            let rows: Vec<(String, String, String, f64, Option<String>, bool, String)> = stmt
                .query_map([], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                    ))
                })
                .context("query_map failed")?
                .collect::<duckdb::Result<Vec<_>>>()
                .context("collect failed")?;

            let mut scored: Vec<MemoryRef> = rows
                .into_iter()
                .map(|(id, content, kind_str, importance, emb_json, pinned, created_at)| {
                    let similarity = if query_embedding.is_empty() {
                        0.0
                    } else {
                        let row_emb = emb_json
                            .as_deref()
                            .and_then(|s| serde_json::from_str::<Vec<f32>>(s).ok())
                            .unwrap_or_default();
                        cosine_similarity(&query_embedding, &row_emb) as f64
                    };
                    MemoryRef {
                        id,
                        content,
                        kind: MemoryKind::from_str_safe(&kind_str),
                        importance,
                        similarity,
                        pinned,
                        created_at: parse_ts(&created_at),
                    }
                })
                .collect();

            // Sort: if we have embeddings, rank by similarity; otherwise by importance.
            if query_embedding.is_empty() {
                scored.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());
            } else {
                scored.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
            }
            scored.truncate(k);
            Ok(scored)
        })
        .await
        .context("spawn_blocking failed")??;

        debug!("Memory search returned {} results", refs.len());
        Ok(refs)
    }

    // -----------------------------------------------------------------------
    // store
    // -----------------------------------------------------------------------

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
        let content = content.to_string();
        // Procedural memories (how to do things on the user's computer) are
        // automatically pinned — they skip decay and pruning entirely.
        let pinned = kind == MemoryKind::Procedural;
        let kind_str = kind.as_str().to_string();
        let source = source.to_string();
        let id2 = id.clone();
        let db = self.db.clone();

        debug!("Storing {kind} memory (pinned={pinned}): {}", truncate(&content, 80));

        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = db.lock().unwrap();
            let emb_json: Option<String> = if embedding.is_empty() {
                None
            } else {
                Some(serde_json::to_string(&embedding).context("Failed to serialize embedding")?)
            };
            conn.execute(
                "INSERT INTO memories
                 (id, content, embedding, kind, importance, source, created_at, last_accessed, access_count, pinned)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                duckdb::params![id2, content, emb_json, kind_str, importance, source, now, now, pinned],
            )
            .context("Failed to insert memory")?;
            Ok(())
        })
        .await
        .context("spawn_blocking failed")??;

        Ok(id)
    }

    // -----------------------------------------------------------------------
    // mark_accessed
    // -----------------------------------------------------------------------

    /// Mark memories as accessed and apply a similarity-weighted importance boost.
    /// `retrieved` is a slice of `(id, similarity)` pairs where `similarity` is
    /// the cosine similarity [0, 1] from the search that retrieved each memory.
    /// Non-pinned memories get `importance += similarity * 0.005`, capped at 0.9.
    pub async fn mark_accessed(&self, retrieved: &[(String, f64)]) -> Result<()> {
        if retrieved.is_empty() {
            return Ok(());
        }
        let now = Utc::now().to_rfc3339();
        let retrieved = retrieved.to_vec();
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = db.lock().unwrap();
            for (id, similarity) in &retrieved {
                let boost = (similarity * 0.005).max(0.0);
                conn.execute(
                    "UPDATE memories
                     SET last_accessed = ?,
                         access_count  = access_count + 1,
                         importance    = CASE WHEN pinned THEN importance
                                              ELSE LEAST(0.9, importance + ?)
                                         END
                     WHERE id = ?",
                    duckdb::params![now, boost, id],
                )
                .context("mark_accessed failed")?;
            }
            Ok(())
        })
        .await
        .context("spawn_blocking failed")??;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // update
    // -----------------------------------------------------------------------

    /// Update an existing memory. Any None field is left unchanged.
    /// If `new_content` is provided the embedding is recomputed.
    /// Returns `true` if the memory was found, `false` if the ID doesn't exist.
    pub async fn update(
        &self,
        id: &str,
        new_content: Option<&str>,
        new_importance: Option<f64>,
        new_pinned: Option<bool>,
    ) -> Result<bool> {
        // Recompute embedding outside spawn_blocking (async embedder).
        let (content_owned, new_embedding) = if let Some(c) = new_content {
            let emb = self.embedder.embed(c).await?;
            (Some(c.to_string()), Some(emb))
        } else {
            (None, None)
        };

        let id = id.to_string();
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<bool> {
            let conn = db.lock().unwrap();

            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE id = ?",
                    duckdb::params![id],
                    |row| row.get(0),
                )
                .context("Failed to check memory existence")?;
            if count == 0 {
                return Ok(false);
            }

            if let (Some(content), Some(embs)) = (&content_owned, &new_embedding) {
                let emb_json: Option<String> = if embs.is_empty() {
                    None
                } else {
                    Some(serde_json::to_string(embs).context("Failed to serialize embedding")?)
                };
                conn.execute(
                    "UPDATE memories SET content = ?, embedding = ? WHERE id = ?",
                    duckdb::params![content, emb_json, id],
                )
                .context("Failed to update content")?;
            }
            if let Some(imp) = new_importance {
                conn.execute(
                    "UPDATE memories SET importance = ? WHERE id = ?",
                    duckdb::params![imp, id],
                )
                .context("Failed to update importance")?;
            }
            if let Some(pin) = new_pinned {
                conn.execute(
                    "UPDATE memories SET pinned = ? WHERE id = ?",
                    duckdb::params![pin, id],
                )
                .context("Failed to update pinned")?;
            }
            Ok(true)
        })
        .await
        .context("spawn_blocking failed")?
    }

    // -----------------------------------------------------------------------
    // pin / unpin
    // -----------------------------------------------------------------------

    /// Pin a memory so it is never decayed or pruned.
    pub async fn pin(&self, id: &str) -> Result<()> {
        self.set_pinned(id, true).await
    }

    /// Unpin a memory, allowing normal decay and pruning again.
    pub async fn unpin(&self, id: &str) -> Result<()> {
        self.set_pinned(id, false).await
    }

    async fn set_pinned(&self, id: &str, pinned: bool) -> Result<()> {
        let id = id.to_string();
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = db.lock().unwrap();
            conn.execute(
                "UPDATE memories SET pinned = ? WHERE id = ?",
                duckdb::params![pinned, id],
            )
            .context("Failed to update pinned flag")?;
            Ok(())
        })
        .await
        .context("spawn_blocking failed")??;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // decay
    // -----------------------------------------------------------------------

    /// Decay importance by a fixed factor derived from `interval_secs` — the
    /// time elapsed since the last decay run.  Using the interval (not age
    /// since creation) means repeated calls compose correctly: applying
    /// `exp(-Δt/h)` N times equals `exp(-NΔt/h)`, matching true exponential
    /// decay regardless of how frequently the function is called.
    pub async fn decay(&self, interval_secs: u64) -> Result<usize> {
        info!(
            "Running importance decay (interval={}s, episodic half-life={}d, semantic 10x, procedural/pinned exempt)",
            interval_secs, self.config.importance_decay_days
        );
        let base_decay_days = self.config.importance_decay_days;
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || -> Result<usize> {
            let conn = db.lock().unwrap();

            #[derive(Debug)]
            struct Row { id: String, kind: String, importance: f64, pinned: bool }

            let mut stmt = conn
                .prepare("SELECT id, kind, importance, pinned FROM memories")
                .context("prepare decay failed")?;
            let rows: Vec<Row> = stmt
                .query_map([], |row| {
                    Ok(Row {
                        id: row.get(0)?,
                        kind: row.get(1)?,
                        importance: row.get(2)?,
                        pinned: row.get(3)?,
                    })
                })
                .context("query_map decay failed")?
                .collect::<duckdb::Result<Vec<_>>>()
                .context("collect decay failed")?;

            let mut count = 0;
            for row in &rows {
                if row.pinned || row.kind == "procedural" {
                    continue;
                }
                let half_life_days = match row.kind.as_str() {
                    "semantic" | "reflection" => base_decay_days * 10.0,
                    _ => base_decay_days,
                };
                // Fixed per-interval decay factor — correct for any call frequency.
                let decay_factor =
                    (-(interval_secs as f64) / (86400.0 * half_life_days)).exp();
                let new_importance = row.importance * decay_factor;
                if (new_importance - row.importance).abs() > 0.001 {
                    conn.execute(
                        "UPDATE memories SET importance = ? WHERE id = ?",
                        duckdb::params![new_importance, row.id],
                    )
                    .context("decay update failed")?;
                    count += 1;
                }
            }
            Ok(count)
        })
        .await
        .context("spawn_blocking failed")?
    }

    // -----------------------------------------------------------------------
    // prune
    // -----------------------------------------------------------------------

    pub async fn prune(&self) -> Result<usize> {
        info!(
            "Pruning memories (min_importance={}, episodic_ttl={}d)",
            self.config.min_importance_to_keep, self.config.ttl_days_episodic
        );
        let min_importance = self.config.min_importance_to_keep;
        let ttl_days = self.config.ttl_days_episodic;
        let db = self.db.clone();
        let now = Utc::now();

        tokio::task::spawn_blocking(move || -> Result<usize> {
            let conn = db.lock().unwrap();

            #[derive(Debug)]
            struct Row { id: String, importance: f64, kind: String, created_at: String, pinned: bool }

            let mut stmt = conn
                .prepare("SELECT id, importance, kind, created_at, pinned FROM memories")
                .context("prepare prune failed")?;
            let rows: Vec<Row> = stmt
                .query_map([], |row| {
                    Ok(Row {
                        id: row.get(0)?,
                        importance: row.get(1)?,
                        kind: row.get(2)?,
                        created_at: row.get(3)?,
                        pinned: row.get(4)?,
                    })
                })
                .context("query_map prune failed")?
                .collect::<duckdb::Result<Vec<_>>>()
                .context("collect prune failed")?;

            let to_delete: Vec<String> = rows
                .into_iter()
                .filter(|row| {
                    // Pinned and procedural memories are never pruned.
                    if row.pinned || row.kind == "procedural" {
                        return false;
                    }
                    let age_days = DateTime::parse_from_rfc3339(&row.created_at)
                        .map(|dt| {
                            (now - dt.with_timezone(&Utc)).num_seconds() as f64 / 86400.0
                        })
                        .unwrap_or(0.0);
                    // Only episodic memories expire by age.
                    // Semantic and reflection are only pruned if importance drops below floor.
                    let expired = row.kind == "episodic" && age_days > ttl_days;
                    row.importance < min_importance || expired
                })
                .map(|r| r.id)
                .collect();

            let count = to_delete.len();
            for id in &to_delete {
                conn.execute("DELETE FROM memories WHERE id = ?", duckdb::params![id])
                    .context("prune delete failed")?;
            }
            Ok(count)
        })
        .await
        .context("spawn_blocking failed")?
    }

    // -----------------------------------------------------------------------
    // find_episodic_clusters — fetches embeddings, clusters in-process
    // -----------------------------------------------------------------------

    pub async fn find_episodic_clusters(&self) -> Result<Vec<Vec<Memory>>> {
        let db = self.db.clone();

        let memories: Vec<Memory> = tokio::task::spawn_blocking(move || -> Result<Vec<Memory>> {
            let conn = db.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT id, content, importance, source, created_at, last_accessed,
                            access_count, embedding, pinned
                     FROM memories
                     WHERE kind = 'episodic' AND embedding IS NOT NULL
                     LIMIT 1000",
                )
                .context("prepare cluster query failed")?;

            let memories = stmt
                .query_map([], |row| {
                    Ok((
                        row.get::<_, String>(0)?,    // id
                        row.get::<_, String>(1)?,    // content
                        row.get::<_, f64>(2)?,       // importance
                        row.get::<_, String>(3)?,    // source
                        row.get::<_, String>(4)?,    // created_at
                        row.get::<_, String>(5)?,    // last_accessed
                        row.get::<_, i64>(6)?,       // access_count
                        row.get::<_, Option<String>>(7)?, // embedding JSON
                        row.get::<_, bool>(8)?,      // pinned
                    ))
                })
                .context("query_map cluster failed")?
                .filter_map(|r| r.ok())
                .filter_map(|(id, content, importance, source, ca, la, ac, emb_json, pinned)| {
                    let embedding = emb_json
                        .as_deref()
                        .and_then(|s| serde_json::from_str::<Vec<f32>>(s).ok())
                        .unwrap_or_default();
                    if embedding.is_empty() {
                        return None;
                    }
                    Some(Memory {
                        id,
                        content,
                        embedding,
                        kind: MemoryKind::Episodic,
                        importance,
                        source,
                        created_at: parse_ts(&ca),
                        last_accessed: parse_ts(&la),
                        access_count: ac,
                        pinned,
                    })
                })
                .collect();
            Ok(memories)
        })
        .await
        .context("spawn_blocking failed")??;

        // Greedy clustering — same algorithm as before
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

    // -----------------------------------------------------------------------
    // delete
    // -----------------------------------------------------------------------

    /// Delete all memories of a given kind. Returns the number deleted.
    pub async fn delete_by_kind(&self, kind: MemoryKind) -> Result<usize> {
        let kind_str = kind.as_str().to_string();
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || -> Result<usize> {
            let conn = db.lock().unwrap();
            let count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE kind = ?",
                    duckdb::params![kind_str],
                    |row| row.get(0),
                )
                .context("count by kind failed")?;
            conn.execute(
                "DELETE FROM memories WHERE kind = ?",
                duckdb::params![kind_str],
            )
            .context("delete by kind failed")?;
            Ok(count as usize)
        })
        .await
        .context("spawn_blocking failed")?
    }

    pub async fn delete(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        let ids = ids.to_vec();
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let conn = db.lock().unwrap();
            for id in &ids {
                conn.execute("DELETE FROM memories WHERE id = ?", duckdb::params![id])
                    .context("delete failed")?;
            }
            Ok(())
        })
        .await
        .context("spawn_blocking failed")??;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // recent
    // -----------------------------------------------------------------------

    pub async fn recent(&self, n: usize) -> Result<Vec<MemoryRef>> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<MemoryRef>> {
            let conn = db.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT id, content, kind, importance, pinned, created_at
                     FROM memories
                     ORDER BY created_at DESC
                     LIMIT ?",
                )
                .context("prepare recent failed")?;
            let refs = stmt
                .query_map([n as i64], |row| {
                    let kind: String = row.get(2)?;
                    let created_at: String = row.get(5)?;
                    Ok(MemoryRef {
                        id: row.get(0)?,
                        content: row.get(1)?,
                        kind: MemoryKind::from_str_safe(&kind),
                        importance: row.get(3)?,
                        similarity: 0.0,
                        pinned: row.get(4)?,
                        created_at: parse_ts(&created_at),
                    })
                })
                .context("query_map recent failed")?
                .collect::<duckdb::Result<Vec<_>>>()
                .context("collect recent failed")?;
            Ok(refs)
        })
        .await
        .context("spawn_blocking failed")?
    }

    // -----------------------------------------------------------------------
    // count
    // -----------------------------------------------------------------------

    pub async fn count(&self) -> Result<usize> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || -> Result<usize> {
            let conn = db.lock().unwrap();
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
                .context("count query failed")?;
            Ok(count as usize)
        })
        .await
        .context("spawn_blocking failed")?
    }

    // -----------------------------------------------------------------------
    // format_memories_for_prompt (unchanged)
    // -----------------------------------------------------------------------

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
// Helpers
// ---------------------------------------------------------------------------

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
    use crate::embeddings::{cosine_similarity, mock::MockEmbeddingClient};

    fn test_config() -> MemoryConfig {
        MemoryConfig {
            max_memories: 1000,
            top_k_retrieval: 5,
            importance_decay_days: 30.0,
            min_importance_to_keep: 0.1,
            ttl_days_episodic: 90.0,
        }
    }

    async fn in_memory_manager() -> MemoryManager {
        let embedder = Arc::new(MockEmbeddingClient::new(4)); // tiny dim for tests
        MemoryManager::new(Path::new(":memory:"), embedder, test_config(), 4)
            .await
            .unwrap()
    }

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
                pinned: false,
                created_at: Utc::now(),
            },
            MemoryRef {
                id: "b".to_string(),
                content: "Discussed DuckDB yesterday".to_string(),
                kind: MemoryKind::Episodic,
                importance: 0.6,
                similarity: 0.75,
                pinned: false,
                created_at: Utc::now(),
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

    #[tokio::test]
    async fn test_store_and_count() {
        let mgr = in_memory_manager().await;
        assert_eq!(mgr.count().await.unwrap(), 0);
        mgr.store("hello world", MemoryKind::Semantic, "test", 0.8)
            .await
            .unwrap();
        assert_eq!(mgr.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_store_and_search() {
        let mgr = in_memory_manager().await;
        mgr.store("The user likes Rust", MemoryKind::Semantic, "test", 0.8)
            .await
            .unwrap();
        let results = mgr.search("Rust programming", Some(5)).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].content, "The user likes Rust");
    }

    #[tokio::test]
    async fn test_delete() {
        let mgr = in_memory_manager().await;
        let id = mgr
            .store("To be deleted", MemoryKind::Episodic, "test", 0.5)
            .await
            .unwrap();
        assert_eq!(mgr.count().await.unwrap(), 1);
        mgr.delete(&[id]).await.unwrap();
        assert_eq!(mgr.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_recent() {
        let mgr = in_memory_manager().await;
        for i in 0..5 {
            mgr.store(&format!("Memory {i}"), MemoryKind::Episodic, "test", 0.5)
                .await
                .unwrap();
        }
        let recent = mgr.recent(3).await.unwrap();
        assert_eq!(recent.len(), 3);
    }

    #[tokio::test]
    async fn test_mark_accessed() {
        let mgr = in_memory_manager().await;
        let id = mgr
            .store("Test memory", MemoryKind::Episodic, "test", 0.5)
            .await
            .unwrap();
        mgr.mark_accessed(&[(id, 0.8)]).await.unwrap(); // should not error
    }

    #[tokio::test]
    async fn test_find_episodic_clusters_empty() {
        let mgr = in_memory_manager().await;
        let clusters = mgr.find_episodic_clusters().await.unwrap();
        assert!(clusters.is_empty());
    }
}
