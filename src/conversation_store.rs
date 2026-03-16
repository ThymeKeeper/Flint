use anyhow::Result;
use duckdb::Connection;
use std::sync::{Arc, Mutex};
use tracing::debug;

use crate::context::ContentBlock;

// ---------------------------------------------------------------------------
// ToolLogEntry — one tool call persisted alongside an assistant turn
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolLogEntry {
    pub name: String,
    /// Key argument (e.g. command, path, query) — 60-char max
    pub summary: String,
    /// First non-empty result line — 80-char max
    pub result: String,
}

// ---------------------------------------------------------------------------
// ConversationStore — DuckDB-backed conversation history
// ---------------------------------------------------------------------------

pub struct ConversationStore {
    db: Arc<Mutex<Connection>>,
}

#[derive(Debug, Clone)]
pub struct StoredTurn {
    pub session_id: String,
    pub sender: String,
    /// "user" or "assistant"
    pub role: String,
    pub content: String,
    pub timestamp_ms: i64,
    /// "tui" or "signal"
    pub channel: String,
    /// Structured tool log for assistant turns (empty for user turns and pre-migration rows)
    pub tool_log: Vec<ToolLogEntry>,
    /// Structured content blocks for tool_use/tool_result messages (None for plain text turns)
    pub content_blocks: Option<Vec<ContentBlock>>,
}

impl ConversationStore {
    /// Create the store, initialising the schema if needed.
    /// Shares the DuckDB connection from `MemoryManager::connection()`.
    pub fn new(db: Arc<Mutex<Connection>>) -> Result<Self> {
        {
            let conn = db.lock().unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS conversation_turns (
                    session_id   VARCHAR NOT NULL DEFAULT 'default',
                    sender       VARCHAR NOT NULL,
                    role         VARCHAR NOT NULL,
                    content      VARCHAR NOT NULL,
                    timestamp_ms BIGINT  NOT NULL,
                    channel      VARCHAR NOT NULL DEFAULT 'tui'
                );",
            )?;
            // Migration: add channel column to databases created before this column existed.
            let has_channel: bool = {
                let mut stmt = conn.prepare(
                    "SELECT COUNT(*) FROM information_schema.columns \
                     WHERE table_name = 'conversation_turns' AND column_name = 'channel'",
                )?;
                let count: i64 = stmt.query_row([], |row| row.get(0))?;
                count > 0
            };
            if !has_channel {
                conn.execute_batch(
                    "ALTER TABLE conversation_turns ADD COLUMN channel VARCHAR; \
                     UPDATE conversation_turns SET channel = 'tui' WHERE channel IS NULL;",
                )?;
            }
            // Migration: add tool_log column (nullable — NULL means no tools used or pre-migration row).
            let has_tool_log: bool = {
                let mut stmt = conn.prepare(
                    "SELECT COUNT(*) FROM information_schema.columns \
                     WHERE table_name = 'conversation_turns' AND column_name = 'tool_log'",
                )?;
                let count: i64 = stmt.query_row([], |row| row.get(0))?;
                count > 0
            };
            if !has_tool_log {
                conn.execute_batch(
                    "ALTER TABLE conversation_turns ADD COLUMN tool_log VARCHAR;",
                )?;
            }
            // Migration: add content_blocks column for structured tool_use/tool_result messages.
            let has_content_blocks: bool = {
                let mut stmt = conn.prepare(
                    "SELECT COUNT(*) FROM information_schema.columns \
                     WHERE table_name = 'conversation_turns' AND column_name = 'content_blocks'",
                )?;
                let count: i64 = stmt.query_row([], |row| row.get(0))?;
                count > 0
            };
            if !has_content_blocks {
                conn.execute_batch(
                    "ALTER TABLE conversation_turns ADD COLUMN content_blocks VARCHAR;",
                )?;
            }
        }
        Ok(Self { db })
    }

    /// Persist a single conversation turn synchronously.
    /// Safe to call from async context as long as the lock is not held across an await.
    pub fn push(
        &self,
        session_id: &str,
        sender: &str,
        role: &str,
        content: &str,
        channel: &str,
        tool_log: &[ToolLogEntry],
        content_blocks: Option<&[ContentBlock]>,
    ) -> Result<()> {
        let ts = chrono::Utc::now().timestamp_millis();
        let tool_log_json = if tool_log.is_empty() {
            None
        } else {
            serde_json::to_string(tool_log).ok()
        };
        let blocks_json = content_blocks.and_then(|b| serde_json::to_string(b).ok());
        let conn = self.db.lock().unwrap();
        conn.execute(
            "INSERT INTO conversation_turns \
             (session_id, sender, role, content, timestamp_ms, channel, tool_log, content_blocks) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![session_id, sender, role, content, ts, channel, tool_log_json, blocks_json],
        )?;
        debug!("Stored {role} turn for session {session_id} via {channel}");
        Ok(())
    }

    /// Load the most recent `limit` turns for a session, oldest-first.
    pub fn load_recent(&self, session_id: &str, limit: usize) -> Result<Vec<StoredTurn>> {
        let conn = self.db.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT session_id, sender, role, content, timestamp_ms, COALESCE(channel, 'tui'), tool_log, content_blocks
             FROM conversation_turns
             WHERE session_id = ?
             ORDER BY timestamp_ms DESC
             LIMIT ?",
        )?;
        let mut turns: Vec<StoredTurn> = stmt
            .query_map(duckdb::params![session_id, limit as i64], |row| {
                let tool_log_json: Option<String> = row.get(6)?;
                let blocks_json: Option<String> = row.get(7)?;
                Ok(StoredTurn {
                    session_id:   row.get(0)?,
                    sender:       row.get(1)?,
                    role:         row.get(2)?,
                    content:      row.get(3)?,
                    timestamp_ms: row.get(4)?,
                    channel:      row.get(5)?,
                    tool_log: tool_log_json
                        .and_then(|j| serde_json::from_str(&j).ok())
                        .unwrap_or_default(),
                    content_blocks: blocks_json
                        .and_then(|j| serde_json::from_str(&j).ok()),
                })
            })?
            .collect::<duckdb::Result<Vec<_>>>()?;
        // Loaded DESC; reverse to chronological order.
        turns.reverse();
        Ok(turns)
    }
}
