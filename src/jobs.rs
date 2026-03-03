//! Lightweight background job store for long-running shell commands.
//!
//! `BackgroundJobStore::new()` returns `(Arc<store>, notify_rx)`.
//! Call `store.spawn(command, timeout_secs)` to launch a shell command
//! asynchronously; it returns a job ID immediately.  When the command
//! finishes (or times out), a `JobNotification` is sent through `notify_rx`.

use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use tokio::sync::mpsc;
use tokio::time::{timeout, Duration};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub struct BackgroundJobStore {
    next_id:   AtomicU64,
    notify_tx: mpsc::Sender<JobNotification>,
}

pub struct JobNotification {
    pub id:      u64,
    pub command: String,
    pub outcome: JobOutcome,
}

pub enum JobOutcome {
    Success { exit_code: i32, output: String },
    Failed(String),
    /// Timeout fired — the OS process may still be running.
    TimedOut,
}

impl JobOutcome {
    pub fn to_message(&self) -> String {
        match self {
            JobOutcome::Success { exit_code, output } => {
                let first: String = output
                    .lines()
                    .find(|l| !l.trim().is_empty())
                    .unwrap_or("(no output)")
                    .chars()
                    .take(120)
                    .collect();
                if *exit_code == 0 {
                    format!("Exit 0 — {first}")
                } else {
                    format!("Exit {exit_code} — {first}")
                }
            }
            JobOutcome::Failed(e) => format!("Failed: {e}"),
            JobOutcome::TimedOut => "Timed out — process may still be running".to_string(),
        }
    }
}

impl JobNotification {
    /// Format the notification as a synthetic message delivered to the agent.
    pub fn to_agent_text(&self) -> String {
        format!(
            "[Background job {} completed]\nCommand: {}\nResult: {}",
            self.id,
            self.command,
            self.outcome.to_message(),
        )
    }
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

impl BackgroundJobStore {
    pub fn new() -> (Arc<Self>, mpsc::Receiver<JobNotification>) {
        let (notify_tx, notify_rx) = mpsc::channel(64);
        let store = Arc::new(Self {
            next_id: AtomicU64::new(1),
            notify_tx,
        });
        (store, notify_rx)
    }

    /// Spawn `command` as a background process and return its job ID immediately.
    /// A `JobNotification` is sent on `notify_rx` when the command completes.
    pub fn spawn(&self, command: String, timeout_secs: u64) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let tx  = self.notify_tx.clone();
        let cmd = command.clone();
        tokio::spawn(async move {
            let outcome = run_command(&cmd, timeout_secs).await;
            let _ = tx.send(JobNotification { id, command: cmd, outcome }).await;
        });
        id
    }
}

// ---------------------------------------------------------------------------
// Internal: run a shell command with a timeout
// ---------------------------------------------------------------------------

async fn run_command(command: &str, timeout_secs: u64) -> JobOutcome {
    let result = timeout(
        Duration::from_secs(timeout_secs),
        tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output(),
    )
    .await;

    match result {
        Err(_) => JobOutcome::TimedOut,
        Ok(Err(e)) => JobOutcome::Failed(e.to_string()),
        Ok(Ok(output)) => {
            let exit_code = output.status.code().unwrap_or(-1);
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let text = if stderr.is_empty() {
                stdout
            } else if stdout.is_empty() {
                stderr
            } else {
                format!("{stdout}\nstderr: {stderr}")
            };
            JobOutcome::Success { exit_code, output: text }
        }
    }
}
