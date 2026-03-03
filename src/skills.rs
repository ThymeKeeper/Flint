use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use duckdb::Connection;
use std::sync::{Arc, Mutex};
use tracing::info;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A named sub-agent profile with a custom system prompt and tool set.
#[derive(Debug, Clone)]
pub struct SkillEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    /// Tool names allowed for this skill's sub-agent.
    /// Only tools from the permitted set are honoured (no spawn_subagent, no task/skill management).
    pub tools: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Tools a skill sub-agent is ever allowed to use, regardless of what is requested.
pub const ALLOWED_SKILL_TOOLS: &[&str] = &[
    "shell_exec",
    "file_read",
    "file_write",
    "web_fetch",
    "memory_search",
    "memory_store",
];

// ---------------------------------------------------------------------------
// SkillManager
// ---------------------------------------------------------------------------

pub struct SkillManager {
    pub(crate) db: Arc<Mutex<Connection>>,
}

const DDL: &str = "CREATE TABLE IF NOT EXISTS skills (
    id            VARCHAR PRIMARY KEY,
    name          VARCHAR NOT NULL,
    description   VARCHAR NOT NULL,
    system_prompt VARCHAR NOT NULL,
    tools         VARCHAR NOT NULL,
    created_at    VARCHAR NOT NULL,
    updated_at    VARCHAR NOT NULL
)";

// DuckDB doesn't support UNIQUE constraints in all versions; enforce in application layer.

impl SkillManager {
    pub async fn new(db: Arc<Mutex<Connection>>) -> Result<Self> {
        let db2 = db.clone();
        tokio::task::spawn_blocking(move || {
            db2.lock().unwrap().execute_batch(DDL).context("create skills table")
        })
        .await
        .context("spawn_blocking")??;
        info!("Skills table ready");
        Ok(Self { db })
    }

    pub async fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("open in-memory DuckDB for skills")?;
        Self::new(Arc::new(Mutex::new(conn))).await
    }

    // -----------------------------------------------------------------------
    // create
    // -----------------------------------------------------------------------

    pub async fn create(
        &self,
        name: &str,
        description: &str,
        system_prompt: &str,
        tools: Vec<String>,
    ) -> Result<String> {
        // Reject unknown/disallowed tool names up front.
        let filtered: Vec<String> = tools
            .into_iter()
            .filter(|t| ALLOWED_SKILL_TOOLS.contains(&t.as_str()))
            .collect();

        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();
        let tools_json = serde_json::to_string(&filtered).context("serialize tools")?;
        let (id2, name2, desc2, prompt2) = (
            id.clone(),
            name.to_string(),
            description.to_string(),
            system_prompt.to_string(),
        );
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            // Delete any existing skill with the same name (upsert behaviour).
            db.lock()
                .unwrap()
                .execute("DELETE FROM skills WHERE name = ?", duckdb::params![name2])
                .context("delete existing skill")?;
            db.lock()
                .unwrap()
                .execute(
                    "INSERT INTO skills (id, name, description, system_prompt, tools, created_at, updated_at)
                     VALUES (?, ?, ?, ?, ?, ?, ?)",
                    duckdb::params![id2, name2, desc2, prompt2, tools_json, now, now],
                )
                .map(|_| ())
                .context("insert skill")
        })
        .await
        .context("spawn_blocking")??;

        info!("Skill created: {name}");
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // get_by_name
    // -----------------------------------------------------------------------

    pub async fn get_by_name(&self, name: &str) -> Result<Option<SkillEntry>> {
        let name = name.to_string();
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT id, name, description, system_prompt, tools, created_at, updated_at
                     FROM skills WHERE name = ?",
                )
                .context("prepare get_by_name")?;
            let mut iter = stmt
                .query_map([name], row_to_skill)
                .context("query_map get_by_name")?;
            match iter.next() {
                Some(r) => Ok(Some(r.context("row_to_skill")?)),
                None => Ok(None),
            }
        })
        .await
        .context("spawn_blocking")?
    }

    // -----------------------------------------------------------------------
    // list
    // -----------------------------------------------------------------------

    pub async fn list(&self) -> Result<Vec<SkillEntry>> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT id, name, description, system_prompt, tools, created_at, updated_at
                     FROM skills ORDER BY name ASC",
                )
                .context("prepare list skills")?;
            stmt.query_map([], row_to_skill)
                .context("query_map list")?
                .collect::<duckdb::Result<Vec<_>>>()
                .context("collect list")
        })
        .await
        .context("spawn_blocking")?
    }

    // -----------------------------------------------------------------------
    // update
    // -----------------------------------------------------------------------

    /// Update fields on an existing skill. Returns false if the name doesn't exist.
    pub async fn update(
        &self,
        name: &str,
        new_description: Option<&str>,
        new_system_prompt: Option<&str>,
        new_tools: Option<Vec<String>>,
    ) -> Result<bool> {
        let existing = match self.get_by_name(name).await? {
            Some(e) => e,
            None => return Ok(false),
        };

        let desc = new_description
            .map(|s| s.to_string())
            .unwrap_or(existing.description);
        let prompt = new_system_prompt
            .map(|s| s.to_string())
            .unwrap_or(existing.system_prompt);
        let tools: Vec<String> = new_tools
            .map(|t| {
                t.into_iter()
                    .filter(|tool| ALLOWED_SKILL_TOOLS.contains(&tool.as_str()))
                    .collect()
            })
            .unwrap_or(existing.tools);

        let tools_json = serde_json::to_string(&tools).context("serialize tools")?;
        let now = Utc::now().to_rfc3339();
        let name = name.to_string();
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            db.lock()
                .unwrap()
                .execute(
                    "UPDATE skills
                     SET description = ?, system_prompt = ?, tools = ?, updated_at = ?
                     WHERE name = ?",
                    duckdb::params![desc, prompt, tools_json, now, name],
                )
                .map(|_| ())
                .context("update skill")
        })
        .await
        .context("spawn_blocking")??;

        Ok(true)
    }

    // -----------------------------------------------------------------------
    // delete
    // -----------------------------------------------------------------------

    pub async fn delete(&self, name: &str) -> Result<bool> {
        // Check existence first so we can return a meaningful bool.
        let exists = self.get_by_name(name).await?.is_some();
        if !exists {
            return Ok(false);
        }
        let name = name.to_string();
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            db.lock()
                .unwrap()
                .execute("DELETE FROM skills WHERE name = ?", duckdb::params![name])
                .map(|_| ())
                .context("delete skill")
        })
        .await
        .context("spawn_blocking")??;
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// DB helpers
// ---------------------------------------------------------------------------

fn row_to_skill(row: &duckdb::Row<'_>) -> duckdb::Result<SkillEntry> {
    let tools_json: String = row.get(4)?;
    let tools: Vec<String> =
        serde_json::from_str(&tools_json).unwrap_or_default();
    Ok(SkillEntry {
        id: row.get(0)?,
        name: row.get(1)?,
        description: row.get(2)?,
        system_prompt: row.get(3)?,
        tools,
        created_at: parse_ts_col(row, 5),
        updated_at: parse_ts_col(row, 6),
    })
}

fn parse_ts(s: &str) -> DateTime<Utc> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

fn parse_ts_col(row: &duckdb::Row<'_>, idx: usize) -> DateTime<Utc> {
    row.get::<_, String>(idx)
        .ok()
        .as_deref()
        .map(parse_ts)
        .unwrap_or_else(Utc::now)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_and_get() {
        let mgr = SkillManager::in_memory().await.unwrap();
        mgr.create(
            "researcher",
            "Web research specialist",
            "You are a focused research agent.",
            vec!["web_fetch".to_string(), "memory_store".to_string()],
        )
        .await
        .unwrap();

        let skill = mgr.get_by_name("researcher").await.unwrap().unwrap();
        assert_eq!(skill.name, "researcher");
        assert_eq!(skill.tools, vec!["web_fetch", "memory_store"]);
    }

    #[tokio::test]
    async fn test_filters_disallowed_tools() {
        let mgr = SkillManager::in_memory().await.unwrap();
        mgr.create(
            "dangerous",
            "Tries to get bad tools",
            "...",
            vec![
                "shell_exec".to_string(),
                "spawn_subagent".to_string(),   // not allowed
                "schedule_task".to_string(),    // not allowed
                "memory_delete".to_string(),    // not allowed
            ],
        )
        .await
        .unwrap();

        let skill = mgr.get_by_name("dangerous").await.unwrap().unwrap();
        assert_eq!(skill.tools, vec!["shell_exec"]);
    }

    #[tokio::test]
    async fn test_upsert_same_name() {
        let mgr = SkillManager::in_memory().await.unwrap();
        mgr.create("dup", "v1", "prompt v1", vec!["web_fetch".to_string()])
            .await
            .unwrap();
        mgr.create("dup", "v2", "prompt v2", vec!["shell_exec".to_string()])
            .await
            .unwrap();

        let skills = mgr.list().await.unwrap();
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].description, "v2");
    }

    #[tokio::test]
    async fn test_update() {
        let mgr = SkillManager::in_memory().await.unwrap();
        mgr.create("s1", "old desc", "old prompt", vec!["web_fetch".to_string()])
            .await
            .unwrap();

        let updated = mgr
            .update("s1", Some("new desc"), None, None)
            .await
            .unwrap();
        assert!(updated);

        let skill = mgr.get_by_name("s1").await.unwrap().unwrap();
        assert_eq!(skill.description, "new desc");
        assert_eq!(skill.system_prompt, "old prompt"); // unchanged
    }

    #[tokio::test]
    async fn test_delete() {
        let mgr = SkillManager::in_memory().await.unwrap();
        mgr.create("to_delete", "desc", "prompt", vec![])
            .await
            .unwrap();
        assert!(mgr.delete("to_delete").await.unwrap());
        assert!(!mgr.delete("to_delete").await.unwrap()); // already gone
        assert!(mgr.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_list_empty() {
        let mgr = SkillManager::in_memory().await.unwrap();
        assert!(mgr.list().await.unwrap().is_empty());
    }
}
