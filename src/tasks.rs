use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use duckdb::Connection;
use std::sync::{Arc, Mutex};
use tracing::info;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The three possible outcomes when a task runs.
/// Chosen by the LLM task runner based on what it found.
#[derive(Debug, Clone, PartialEq)]
pub enum TaskRunState {
    /// Took meaningful action or found something worth reporting. Resets idle_count.
    Acted,
    /// Nothing happened yet, but actively monitoring a specific expected future event.
    /// Does NOT increment idle_count — task stays alive indefinitely.
    StillWaiting,
    /// No-op with no specific future condition being waited for. Increments idle_count.
    NothingToDo,
}

impl TaskRunState {
    pub fn from_str(s: &str) -> Self {
        match s {
            "acted" => Self::Acted,
            "still_waiting" => Self::StillWaiting,
            _ => Self::NothingToDo,
        }
    }
}

/// Structured output from a single task execution.
#[derive(Debug)]
pub struct TaskRunOutput {
    pub state: TaskRunState,
    pub reason: String,
    pub message: Option<String>,
}

/// Lifecycle event returned by `record_result` so callers can react.
#[derive(Debug)]
pub enum TaskLifecycleEvent {
    Continue,
    AutoPaused { description: String, idle_count: i64 },
    Expired { description: String },
    OneShot { description: String },
}

/// A row from the `tasks` table.
#[derive(Debug, Clone)]
pub struct TaskEntry {
    pub id: String,
    pub description: String,
    pub trigger_type: String,   // "interval" | "cron" | "once"
    pub trigger_spec: String,   // seconds | "HH:MM" (UTC) | RFC3339
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub last_run: Option<DateTime<Utc>>,
    pub next_run: DateTime<Utc>,
    pub run_count: i64,
    pub idle_count: i64,
    pub max_idle_runs: i64,
    pub expires_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// TaskManager
// ---------------------------------------------------------------------------

pub struct TaskManager {
    pub(crate) db: Arc<Mutex<Connection>>,
}

const DDL: &str = "CREATE TABLE IF NOT EXISTS tasks (
    id            VARCHAR PRIMARY KEY,
    description   VARCHAR NOT NULL,
    trigger_type  VARCHAR NOT NULL,
    trigger_spec  VARCHAR NOT NULL,
    enabled       BOOLEAN NOT NULL DEFAULT true,
    created_at    VARCHAR NOT NULL,
    last_run      VARCHAR,
    next_run      VARCHAR NOT NULL,
    run_count     INTEGER NOT NULL DEFAULT 0,
    idle_count    INTEGER NOT NULL DEFAULT 0,
    max_idle_runs INTEGER NOT NULL DEFAULT 10,
    expires_at    VARCHAR
)";

impl TaskManager {
    /// Share the MemoryManager's DuckDB connection (one connection, no conflicts).
    pub async fn new(db: Arc<Mutex<Connection>>) -> Result<Self> {
        let db2 = db.clone();
        tokio::task::spawn_blocking(move || {
            db2.lock()
                .unwrap()
                .execute_batch(DDL)
                .context("create tasks table")
        })
        .await
        .context("spawn_blocking")??;
        info!("Task table ready");
        Ok(Self { db })
    }

    /// Create a standalone in-memory TaskManager (for tests).
    pub async fn in_memory() -> Result<Self> {
        let conn =
            Connection::open_in_memory().context("open in-memory DuckDB for tasks")?;
        Self::new(Arc::new(Mutex::new(conn))).await
    }

    // -----------------------------------------------------------------------
    // create
    // -----------------------------------------------------------------------

    pub async fn create(
        &self,
        description: &str,
        trigger_type: &str,
        trigger_spec: &str,
        max_idle_runs: i64,
        expires_at: Option<DateTime<Utc>>,
    ) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let next_run =
            compute_next_run(trigger_type, trigger_spec, None).unwrap_or(now);

        let (id2, desc, tt, ts) = (
            id.clone(),
            description.to_string(),
            trigger_type.to_string(),
            trigger_spec.to_string(),
        );
        let now_s = now.to_rfc3339();
        let next_s = next_run.to_rfc3339();
        let exp_s: Option<String> = expires_at.map(|t| t.to_rfc3339());
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            db.lock()
                .unwrap()
                .execute(
                    "INSERT INTO tasks
                     (id, description, trigger_type, trigger_spec, enabled,
                      created_at, last_run, next_run, run_count, idle_count,
                      max_idle_runs, expires_at)
                     VALUES (?, ?, ?, ?, true, ?, NULL, ?, 0, 0, ?, ?)",
                    duckdb::params![id2, desc, tt, ts, now_s, next_s, max_idle_runs, exp_s],
                )
                .map(|_| ())
                .context("insert task")
        })
        .await
        .context("spawn_blocking")??;

        info!("Task created: {id} ({trigger_type}:{trigger_spec})");
        Ok(id)
    }

    // -----------------------------------------------------------------------
    // due — tasks whose next_run has passed and haven't expired
    // -----------------------------------------------------------------------

    pub async fn due(&self) -> Result<Vec<TaskEntry>> {
        let now = Utc::now().to_rfc3339();
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT id, description, trigger_type, trigger_spec, enabled,
                            created_at, last_run, next_run, run_count, idle_count,
                            max_idle_runs, expires_at
                     FROM tasks
                     WHERE enabled = true AND next_run <= ?
                       AND (expires_at IS NULL OR expires_at > ?)",
                )
                .context("prepare due")?;
            stmt.query_map([&now, &now], row_to_entry)
                .context("query_map due")?
                .collect::<duckdb::Result<Vec<_>>>()
                .context("collect due")
        })
        .await
        .context("spawn_blocking")?
    }

    // -----------------------------------------------------------------------
    // list — all tasks (for list_tasks tool)
    // -----------------------------------------------------------------------

    pub async fn list(&self) -> Result<Vec<TaskEntry>> {
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let conn = db.lock().unwrap();
            let mut stmt = conn
                .prepare(
                    "SELECT id, description, trigger_type, trigger_spec, enabled,
                            created_at, last_run, next_run, run_count, idle_count,
                            max_idle_runs, expires_at
                     FROM tasks ORDER BY created_at DESC",
                )
                .context("prepare list")?;
            stmt.query_map([], row_to_entry)
                .context("query_map list")?
                .collect::<duckdb::Result<Vec<_>>>()
                .context("collect list")
        })
        .await
        .context("spawn_blocking")?
    }

    // -----------------------------------------------------------------------
    // record_result — update lifecycle state after a run
    // -----------------------------------------------------------------------

    pub async fn record_result(
        &self,
        task: &TaskEntry,
        state: &TaskRunState,
    ) -> Result<TaskLifecycleEvent> {
        let now = Utc::now();

        let new_idle = match state {
            TaskRunState::Acted => 0,
            TaskRunState::StillWaiting => task.idle_count,
            TaskRunState::NothingToDo => task.idle_count + 1,
        };
        let new_run_count = task.run_count + 1;

        let auto_pause = new_idle >= task.max_idle_runs;
        let expired = task.expires_at.map(|e| now >= e).unwrap_or(false);
        let is_once = task.trigger_type == "once";

        let disable = auto_pause || expired || is_once;
        let next_run = if disable {
            task.next_run
        } else {
            compute_next_run(&task.trigger_type, &task.trigger_spec, Some(now))
                .unwrap_or(now)
        };

        let id = task.id.clone();
        let now_s = now.to_rfc3339();
        let next_s = next_run.to_rfc3339();
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            db.lock()
                .unwrap()
                .execute(
                    "UPDATE tasks
                     SET last_run = ?, next_run = ?, run_count = ?,
                         idle_count = ?, enabled = ?
                     WHERE id = ?",
                    duckdb::params![now_s, next_s, new_run_count, new_idle, !disable, id],
                )
                .map(|_| ())
                .context("record_result update")
        })
        .await
        .context("spawn_blocking")??;

        if expired {
            return Ok(TaskLifecycleEvent::Expired {
                description: task.description.clone(),
            });
        }
        if auto_pause {
            return Ok(TaskLifecycleEvent::AutoPaused {
                description: task.description.clone(),
                idle_count: new_idle,
            });
        }
        if is_once {
            return Ok(TaskLifecycleEvent::OneShot {
                description: task.description.clone(),
            });
        }
        Ok(TaskLifecycleEvent::Continue)
    }

    // -----------------------------------------------------------------------
    // delete / set_enabled
    // -----------------------------------------------------------------------

    pub async fn delete(&self, id: &str) -> Result<()> {
        let id = id.to_string();
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            db.lock()
                .unwrap()
                .execute("DELETE FROM tasks WHERE id = ?", duckdb::params![id])
                .map(|_| ())
                .context("delete task")
        })
        .await
        .context("spawn_blocking")??;
        Ok(())
    }

    pub async fn set_enabled(&self, id: &str, enabled: bool) -> Result<()> {
        let id = id.to_string();
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            db.lock()
                .unwrap()
                .execute(
                    "UPDATE tasks SET enabled = ? WHERE id = ?",
                    duckdb::params![enabled, id],
                )
                .map(|_| ())
                .context("set_enabled")
        })
        .await
        .context("spawn_blocking")??;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Trigger scheduling helpers
// ---------------------------------------------------------------------------

/// Compute when a task should next fire.
/// For `interval` and `cron`, `after` defaults to `Utc::now()`.
/// For `once`, `trigger_spec` is an absolute RFC3339 timestamp and `after` is ignored.
pub fn compute_next_run(
    trigger_type: &str,
    trigger_spec: &str,
    after: Option<DateTime<Utc>>,
) -> Option<DateTime<Utc>> {
    match trigger_type {
        "interval" => {
            let secs: i64 = trigger_spec.parse().ok()?;
            let base = after.unwrap_or_else(Utc::now);
            Some(base + chrono::Duration::seconds(secs))
        }
        "cron" => {
            let base = after.unwrap_or_else(Utc::now);
            next_cron_after(trigger_spec, base)
        }
        "once" => DateTime::parse_from_rfc3339(trigger_spec)
            .ok()
            .map(|t| t.with_timezone(&Utc)),
        _ => None,
    }
}

/// Find the next UTC occurrence of "HH:MM" after `after`.
fn next_cron_after(spec: &str, after: DateTime<Utc>) -> Option<DateTime<Utc>> {
    let mut parts = spec.splitn(2, ':');
    let hour: u32 = parts.next()?.parse().ok()?;
    let minute: u32 = parts.next()?.parse().ok()?;

    for days_ahead in 0i64..=1 {
        let candidate = (after + chrono::Duration::days(days_ahead))
            .date_naive()
            .and_hms_opt(hour, minute, 0)?
            .and_utc();
        if candidate > after {
            return Some(candidate);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// DB helpers
// ---------------------------------------------------------------------------

fn row_to_entry(row: &duckdb::Row<'_>) -> duckdb::Result<TaskEntry> {
    let last_run_s: Option<String> = row.get(6)?;
    let expires_s: Option<String> = row.get(11)?;
    Ok(TaskEntry {
        id: row.get(0)?,
        description: row.get(1)?,
        trigger_type: row.get(2)?,
        trigger_spec: row.get(3)?,
        enabled: row.get(4)?,
        created_at: parse_ts_col(row, 5),
        last_run: last_run_s.as_deref().map(parse_ts),
        next_run: parse_ts_col(row, 7),
        run_count: row.get(8)?,
        idle_count: row.get(9)?,
        max_idle_runs: row.get(10)?,
        expires_at: expires_s.as_deref().map(parse_ts),
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
    async fn test_create_and_list() {
        let mgr = TaskManager::in_memory().await.unwrap();
        let id = mgr
            .create("watch example.com", "interval", "3600", 10, None)
            .await
            .unwrap();
        assert!(!id.is_empty());
        let tasks = mgr.list().await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].description, "watch example.com");
        assert!(tasks[0].enabled);
    }

    #[tokio::test]
    async fn test_due_fires_immediately() {
        let mgr = TaskManager::in_memory().await.unwrap();
        // interval of 0 → next_run = now → should be due immediately
        mgr.create("instant task", "interval", "0", 10, None)
            .await
            .unwrap();
        let due = mgr.due().await.unwrap();
        assert!(!due.is_empty());
    }

    #[tokio::test]
    async fn test_delete() {
        let mgr = TaskManager::in_memory().await.unwrap();
        let id = mgr.create("temp", "interval", "60", 5, None).await.unwrap();
        assert_eq!(mgr.list().await.unwrap().len(), 1);
        mgr.delete(&id).await.unwrap();
        assert_eq!(mgr.list().await.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_record_result_idle_count() {
        let mgr = TaskManager::in_memory().await.unwrap();
        mgr.create("watch folder", "interval", "3600", 3, None)
            .await
            .unwrap();
        let tasks = mgr.list().await.unwrap();
        let task = &tasks[0];

        // nothing_to_do → idle_count increments
        let event = mgr
            .record_result(task, &TaskRunState::NothingToDo)
            .await
            .unwrap();
        assert!(matches!(event, TaskLifecycleEvent::Continue));

        // Simulate 2 more nothing_to_do → should hit max_idle_runs=3
        let tasks = mgr.list().await.unwrap();
        let _ = mgr.record_result(&tasks[0], &TaskRunState::NothingToDo).await.unwrap();
        let tasks = mgr.list().await.unwrap();
        let event = mgr.record_result(&tasks[0], &TaskRunState::NothingToDo).await.unwrap();
        assert!(matches!(event, TaskLifecycleEvent::AutoPaused { .. }));

        // Task should now be disabled
        let tasks = mgr.list().await.unwrap();
        assert!(!tasks[0].enabled);
    }

    #[tokio::test]
    async fn test_still_waiting_does_not_increment_idle() {
        let mgr = TaskManager::in_memory().await.unwrap();
        mgr.create("wait for game release", "interval", "3600", 3, None)
            .await
            .unwrap();

        // still_waiting many times — idle_count stays 0
        for _ in 0..10 {
            let tasks = mgr.list().await.unwrap();
            let event = mgr
                .record_result(&tasks[0], &TaskRunState::StillWaiting)
                .await
                .unwrap();
            assert!(matches!(event, TaskLifecycleEvent::Continue));
        }
        let tasks = mgr.list().await.unwrap();
        assert_eq!(tasks[0].idle_count, 0);
        assert!(tasks[0].enabled);
    }

    #[tokio::test]
    async fn test_acted_resets_idle_count() {
        let mgr = TaskManager::in_memory().await.unwrap();
        mgr.create("check news", "interval", "3600", 5, None)
            .await
            .unwrap();

        // Two nothing_to_do runs
        for _ in 0..2 {
            let tasks = mgr.list().await.unwrap();
            mgr.record_result(&tasks[0], &TaskRunState::NothingToDo).await.unwrap();
        }
        let tasks = mgr.list().await.unwrap();
        assert_eq!(tasks[0].idle_count, 2);

        // acted → resets to 0
        mgr.record_result(&tasks[0], &TaskRunState::Acted).await.unwrap();
        let tasks = mgr.list().await.unwrap();
        assert_eq!(tasks[0].idle_count, 0);
    }

    #[tokio::test]
    async fn test_once_task_disables_after_run() {
        let mgr = TaskManager::in_memory().await.unwrap();
        // Use a past RFC3339 timestamp so it fires immediately
        let spec = "2020-01-01T00:00:00Z";
        mgr.create("send greeting once", "once", spec, 10, None)
            .await
            .unwrap();
        let tasks = mgr.list().await.unwrap();
        let event = mgr
            .record_result(&tasks[0], &TaskRunState::Acted)
            .await
            .unwrap();
        assert!(matches!(event, TaskLifecycleEvent::OneShot { .. }));
        let tasks = mgr.list().await.unwrap();
        assert!(!tasks[0].enabled);
    }

    #[test]
    fn test_compute_next_run_interval() {
        let base = Utc::now();
        let next = compute_next_run("interval", "3600", Some(base)).unwrap();
        let diff = (next - base).num_seconds();
        assert_eq!(diff, 3600);
    }

    #[test]
    fn test_compute_next_run_once() {
        let ts = "2099-12-31T12:00:00Z";
        let next = compute_next_run("once", ts, None).unwrap();
        assert_eq!(next.format("%Y").to_string(), "2099");
    }
}
