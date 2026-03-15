use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::observer::AgentObserver;

pub use crate::config::SignalConfig;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// An incoming message from Signal.
#[derive(Debug, Clone)]
pub struct IncomingMessage {
    pub sender: String,
    pub text: String,
    pub timestamp: u64,
}

// ---------------------------------------------------------------------------
// Trait for testability
// ---------------------------------------------------------------------------

#[async_trait]
pub trait SignalClient: Send + Sync {
    /// Poll for new messages.
    async fn receive(&self) -> Result<Vec<IncomingMessage>>;
    /// Send a message to a recipient.
    async fn send(&self, recipient: &str, message: &str) -> Result<()>;
    /// Check if a sender is in the allow list.
    fn is_allowed(&self, sender: &str) -> bool;
    /// Push a background-job completion notification to the transport.
    /// For the TUI this adds a System message + agent placeholder to the chat.
    /// Default: no-op (non-TUI transports handle notification differently).
    fn push_notification(&self, _text: String) {}
    /// Show "typing…" indicator to a recipient. Default: no-op.
    async fn send_typing(&self, _recipient: &str) -> Result<()> { Ok(()) }
    /// Clear "typing…" indicator. Default: no-op.
    async fn stop_typing(&self, _recipient: &str) -> Result<()> { Ok(()) }
    /// Mark a message as read by its timestamp. Default: no-op.
    async fn send_read_receipt(&self, _recipient: &str, _timestamp: u64) -> Result<()> { Ok(()) }
}

// ---------------------------------------------------------------------------
// signal-cli REST client
// ---------------------------------------------------------------------------

pub struct SignalRestClient {
    base_url: String,
    phone_number: String,
    allowed_senders: Vec<String>,
    http: reqwest::Client,
}

impl SignalRestClient {
    pub fn new(config: &SignalConfig) -> Self {
        Self {
            base_url: config.base_url.trim_end_matches('/').to_string(),
            phone_number: config.phone_number.clone(),
            allowed_senders: config.allowed_senders.clone(),
            http: reqwest::Client::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// signal-cli REST API types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct Envelope {
    #[serde(rename = "sourceNumber")]
    source_number: Option<String>,
    #[serde(rename = "dataMessage")]
    data_message: Option<DataMessage>,
    timestamp: Option<u64>,
}

#[derive(Deserialize)]
struct DataMessage {
    message: Option<String>,
    timestamp: Option<u64>,
}

#[derive(Serialize)]
struct SendRequest {
    message: String,
    number: String,
    recipients: Vec<String>,
}

#[async_trait]
impl SignalClient for SignalRestClient {
    async fn receive(&self) -> Result<Vec<IncomingMessage>> {
        // The + in E.164 numbers must be percent-encoded in the URL path.
        let encoded_number = self.phone_number.replace('+', "%2B");
        let url = format!("{}/v1/receive/{}", self.base_url, encoded_number);

        debug!("Polling Signal for messages: {url}");

        let response = self
            .http
            .get(&url)
            .send()
            .await
            .context("Failed to poll signal-cli")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("signal-cli receive returned {status}: {body}");
        }

        let envelopes: Vec<Envelope> = response
            .json()
            .await
            .context("Failed to parse signal-cli response")?;

        let mut messages = Vec::new();
        for env in envelopes {
            let sender = match env.source_number {
                Some(s) => s,
                None => continue,
            };
            let data = match env.data_message {
                Some(d) => d,
                None => continue,
            };
            let text = match data.message {
                Some(t) if !t.is_empty() => t,
                _ => continue,
            };
            let timestamp = data.timestamp.or(env.timestamp).unwrap_or(0);

            messages.push(IncomingMessage {
                sender,
                text,
                timestamp,
            });
        }

        debug!("Received {} messages from Signal", messages.len());
        Ok(messages)
    }

    async fn send(&self, recipient: &str, message: &str) -> Result<()> {
        let url = format!("{}/v2/send", self.base_url);

        debug!(
            "Sending Signal message to {}: {}",
            recipient,
            truncate(message, 50)
        );

        let body = SendRequest {
            message: message.to_string(),
            number: self.phone_number.clone(),
            recipients: vec![recipient.to_string()],
        };

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send via signal-cli")?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response.text().await.unwrap_or_default();
            warn!("signal-cli send returned {status}: {body_text}");
            bail!("signal-cli send failed: {status}");
        }

        debug!("Signal message sent successfully to {recipient}");
        Ok(())
    }

    fn is_allowed(&self, sender: &str) -> bool {
        self.allowed_senders.iter().any(|s| s == sender)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

// ---------------------------------------------------------------------------
// signal-cli TCP JSON-RPC client
// ---------------------------------------------------------------------------

/// Communicates with signal-cli via its JSON-RPC TCP socket (`daemon --tcp`).
///
/// signal-cli in `--receive-mode on-start` (the default) actively receives
/// messages in the background and **pushes them as JSON-RPC notifications**
/// to every connected client. We maintain one persistent listener connection
/// and forward notifications to an in-process channel; `receive()` blocks
/// until a message arrives. Outgoing `send()` calls open short-lived
/// connections to issue a JSON-RPC `send` request.
pub struct SignalTcpRpcClient {
    addr: String,
    allowed_senders: Vec<String>,
    message_rx: Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<IncomingMessage>>>,
    next_id: Arc<AtomicU64>,
}

impl SignalTcpRpcClient {
    pub fn new(config: &SignalConfig) -> Self {
        let addr = crate::signal_daemon::daemon_tcp_addr();
        let (tx, rx) = tokio::sync::mpsc::channel::<IncomingMessage>(64);
        // Background task: maintain a persistent connection and forward
        // incoming message notifications to the channel.
        let own_number = config.phone_number.clone();
        tokio::spawn(run_notification_listener(addr.clone(), own_number, tx));
        Self {
            addr,
            allowed_senders: config.allowed_senders.clone(),
            message_rx: Arc::new(tokio::sync::Mutex::new(rx)),
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Open a short-lived connection to send one JSON-RPC request and return
    /// the result. Incoming notifications on this connection are ignored.
    async fn call(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        use tokio::net::TcpStream;

        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let request = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": id,
        });
        let mut req_bytes = serde_json::to_vec(&request)?;
        req_bytes.push(b'\n');

        let mut stream = TcpStream::connect(&self.addr)
            .await
            .with_context(|| format!("Failed to connect to signal-cli at {}", self.addr))?;
        stream.write_all(&req_bytes).await?;

        let mut reader = BufReader::new(stream);
        let mut line = String::new();

        loop {
            line.clear();
            let n = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                reader.read_line(&mut line),
            )
            .await
            .context("Timed out waiting for signal-cli response")?
            .context("signal-cli TCP connection closed")?;

            if n == 0 {
                bail!("signal-cli TCP connection closed before response");
            }

            let response: serde_json::Value = match serde_json::from_str(line.trim()) {
                Ok(v) => v,
                Err(_) => continue, // malformed line, skip
            };

            // Skip notifications (no "id") — only match our response.
            if response.get("id").and_then(|v| v.as_u64()) == Some(id) {
                if let Some(err) = response.get("error") {
                    bail!("signal-cli error: {err}");
                }
                return Ok(response["result"].clone());
            }
        }
    }
}

#[async_trait]
impl SignalClient for SignalTcpRpcClient {
    /// Block until a Signal message arrives via the persistent notification
    /// listener, then drain any additional buffered messages and return them.
    async fn receive(&self) -> Result<Vec<IncomingMessage>> {
        let mut rx = self.message_rx.lock().await;
        match rx.recv().await {
            Some(msg) => {
                let mut messages = vec![msg];
                // Drain any additional messages queued simultaneously.
                while let Ok(m) = rx.try_recv() {
                    messages.push(m);
                }
                debug!("Received {} message(s) from signal-cli", messages.len());
                Ok(messages)
            }
            None => Err(anyhow!("signal-cli notification channel closed")),
        }
    }

    async fn send(&self, recipient: &str, message: &str) -> Result<()> {
        if message.trim().is_empty() {
            debug!("Suppressed empty Signal message to {recipient}");
            return Ok(());
        }
        debug!(
            "Sending Signal message to {}: {}",
            recipient,
            truncate(message, 50)
        );
        self.call(
            "send",
            serde_json::json!({
                "recipient": [recipient],
                "message": message,
            }),
        )
        .await
        .context("signal-cli send failed")?;
        debug!("Signal message sent to {recipient}");
        Ok(())
    }

    fn is_allowed(&self, sender: &str) -> bool {
        self.allowed_senders.iter().any(|s| s == sender)
    }

    async fn send_typing(&self, recipient: &str) -> Result<()> {
        self.call(
            "sendTypingMessage",
            serde_json::json!({
                "recipient": [recipient],
            }),
        )
        .await
        .context("sendTypingMessage failed")?;
        Ok(())
    }

    async fn stop_typing(&self, recipient: &str) -> Result<()> {
        self.call(
            "sendTypingMessage",
            serde_json::json!({
                "recipient": [recipient],
                "stop": true,
            }),
        )
        .await
        .context("sendTypingMessage (stop) failed")?;
        Ok(())
    }

    async fn send_read_receipt(&self, recipient: &str, timestamp: u64) -> Result<()> {
        self.call(
            "sendReceipt",
            serde_json::json!({
                "recipient": recipient,
                "type": "read",
                "targetTimestamp": [timestamp],
            }),
        )
        .await
        .context("sendReceipt failed")?;
        Ok(())
    }
}

/// Background task: connect to signal-cli and forward incoming message
/// notifications to the channel. Reconnects automatically on disconnect.
async fn run_notification_listener(
    addr: String,
    own_number: String,
    tx: tokio::sync::mpsc::Sender<IncomingMessage>,
) {
    // Track recently seen timestamps to deduplicate notifications.
    // signal-cli can emit the same message more than once (e.g. receipt + delivery).
    let mut recent_timestamps: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
    const DEDUP_WINDOW: usize = 64;

    loop {
        debug!("signal-cli notification listener connecting to {addr}");
        match listen_for_notifications(&addr, &own_number, &tx, &mut recent_timestamps, DEDUP_WINDOW).await {
            Ok(()) => {
                debug!("signal-cli notification listener: channel closed, exiting");
                break;
            }
            Err(e) => {
                warn!("signal-cli notification listener disconnected: {e:#}");
                tokio::time::sleep(std::time::Duration::from_secs(3)).await;
            }
        }
    }
}

/// Connect once and forward JSON-RPC notifications until the connection drops
/// or the sender is gone. Returns Ok(()) when the channel receiver is dropped.
async fn listen_for_notifications(
    addr: &str,
    own_number: &str,
    tx: &tokio::sync::mpsc::Sender<IncomingMessage>,
    recent_timestamps: &mut std::collections::VecDeque<u64>,
    dedup_window: usize,
) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::net::TcpStream;

    let stream = TcpStream::connect(addr)
        .await
        .with_context(|| format!("Failed to connect to signal-cli at {addr}"))?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .await
            .context("signal-cli TCP read error")?;
        if n == 0 {
            bail!("signal-cli TCP connection closed");
        }

        let value: serde_json::Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(e) => {
                warn!("Failed to parse signal-cli notification: {e}");
                continue;
            }
        };

        // JSON-RPC notifications have "method" but no "id".
        if value.get("id").is_some() || value.get("method").is_none() {
            continue;
        }

        // Log every notification at info level so we can see what signal-cli sends.
        info!("signal-cli notification: {}", line.trim());

        if let Some(msg) = extract_message_from_notification(&value) {
            // Skip messages from the bot's own number (sync/sent transcripts).
            if msg.sender == own_number {
                debug!("Ignoring self-message from {}", own_number);
                continue;
            }
            // Deduplicate by timestamp — signal-cli can emit the same message
            // more than once (e.g. receipt echo, re-delivery).
            if msg.timestamp > 0 && recent_timestamps.contains(&msg.timestamp) {
                debug!("Ignoring duplicate message (ts={})", msg.timestamp);
                continue;
            }
            if msg.timestamp > 0 {
                recent_timestamps.push_back(msg.timestamp);
                if recent_timestamps.len() > dedup_window {
                    recent_timestamps.pop_front();
                }
            }
            if tx.send(msg).await.is_err() {
                return Ok(()); // receiver dropped
            }
        } else {
            debug!("notification had no extractable data message, skipping");
        }
    }
}

fn extract_message_from_notification(notification: &serde_json::Value) -> Option<IncomingMessage> {
    let params = notification.get("params")?;

    // signal-cli v0.13+: params.envelope.dataMessage
    // Some builds: params.envelope at top level, dataMessage inside envelope
    // Try both layouts.
    let envelope = params.get("envelope")?;

    let sender = envelope
        .get("sourceNumber")
        .or_else(|| envelope.get("source"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())?;

    // dataMessage may be inside envelope or alongside it in params.
    let data = envelope
        .get("dataMessage")
        .or_else(|| params.get("dataMessage"))?;

    let text = data.get("message").and_then(|v| v.as_str())?;
    if text.is_empty() {
        return None;
    }

    let timestamp = data
        .get("timestamp")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    Some(IncomingMessage {
        sender,
        text: text.to_string(),
        timestamp,
    })
}

// ---------------------------------------------------------------------------
// Mock for testing
// ---------------------------------------------------------------------------

pub mod mock {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;

    pub struct MockSignalClient {
        pub incoming: Mutex<VecDeque<Vec<IncomingMessage>>>,
        pub sent: Mutex<Vec<(String, String)>>,
        pub allowed: Vec<String>,
    }

    impl MockSignalClient {
        pub fn new(allowed: Vec<String>) -> Self {
            Self {
                incoming: Mutex::new(VecDeque::new()),
                sent: Mutex::new(Vec::new()),
                allowed,
            }
        }

        pub fn enqueue_messages(&self, messages: Vec<IncomingMessage>) {
            self.incoming.lock().unwrap().push_back(messages);
        }

        pub fn get_sent(&self) -> Vec<(String, String)> {
            self.sent.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl SignalClient for MockSignalClient {
        async fn receive(&self) -> Result<Vec<IncomingMessage>> {
            let mut queue = self.incoming.lock().unwrap();
            Ok(queue.pop_front().unwrap_or_default())
        }

        async fn send(&self, recipient: &str, message: &str) -> Result<()> {
            self.sent
                .lock()
                .unwrap()
                .push((recipient.to_string(), message.to_string()));
            Ok(())
        }

        fn is_allowed(&self, sender: &str) -> bool {
            self.allowed.iter().any(|s| s == sender)
        }
    }
}

// ---------------------------------------------------------------------------
// TuiSignalClient — bridges the async main loop to the TUI thread
// ---------------------------------------------------------------------------

/// A [`SignalClient`] backed by the ratatui TUI.
///
/// `receive()` awaits user input submitted via the TUI input box.
/// `send()` delivers the completed response to the TUI for display.
/// The streaming callback forwards chunks to the TUI in real time.
pub struct TuiSignalClient {
    user_input_rx: Arc<tokio::sync::Mutex<tokio::sync::mpsc::Receiver<String>>>,
    agent_update_tx: tokio::sync::mpsc::Sender<crate::tui::AgentUpdate>,
}

impl TuiSignalClient {
    pub fn new(channels: crate::tui::TuiChannels) -> Self {
        Self {
            user_input_rx: Arc::new(tokio::sync::Mutex::new(channels.user_input_rx)),
            agent_update_tx: channels.agent_update_tx,
        }
    }

    /// Push persisted conversation history to the TUI before the first render.
    /// Each entry is (display_role, content) e.g. ("You", "…") or (agent_name, "…").
    pub fn push_history(&self, turns: Vec<(String, String)>) {
        let _ = self
            .agent_update_tx
            .try_send(crate::tui::AgentUpdate::HistoryLoaded(turns));
    }

    /// Append new turns to the end of the TUI chat (e.g. from Signal exchanges).
    pub fn push_turns(&self, turns: Vec<(String, String)>) {
        let _ = self
            .agent_update_tx
            .try_send(crate::tui::AgentUpdate::NewTurns(turns));
    }
}

#[async_trait]
impl SignalClient for TuiSignalClient {
    async fn receive(&self) -> Result<Vec<IncomingMessage>> {
        let mut rx = self.user_input_rx.lock().await;
        match rx.recv().await {
            Some(text) => Ok(vec![IncomingMessage {
                sender: "user".to_string(),
                text,
                timestamp: chrono::Utc::now().timestamp() as u64,
            }]),
            None => Err(anyhow!("TUI input channel closed")),
        }
    }

    async fn send(&self, _recipient: &str, message: &str) -> Result<()> {
        self.agent_update_tx
            .send(crate::tui::AgentUpdate::Complete(message.to_string()))
            .await
            .ok();
        Ok(())
    }

    fn is_allowed(&self, _sender: &str) -> bool {
        true
    }

    fn push_notification(&self, text: String) {
        let _ = self.agent_update_tx.try_send(crate::tui::AgentUpdate::JobNotification(text));
    }
}

impl AgentObserver for TuiSignalClient {
    fn on_text_chunk(&self, chunk: String) {
        let _ = self.agent_update_tx.try_send(crate::tui::AgentUpdate::StreamChunk(chunk));
    }

    fn on_tool_start(&self, name: &str) {
        let _ = self
            .agent_update_tx
            .try_send(crate::tui::AgentUpdate::StatusUpdate(name.to_string()));
    }

    fn on_tool_event(&self, text: String) {
        let _ = self.agent_update_tx.try_send(crate::tui::AgentUpdate::StreamChunk(text));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_allowed_yes() {
        let client = SignalRestClient {
            base_url: "http://localhost:8080".to_string(),
            phone_number: "+15551234567".to_string(),
            allowed_senders: vec!["+15559876543".to_string(), "+15551111111".to_string()],
            http: reqwest::Client::new(),
        };
        assert!(client.is_allowed("+15559876543"));
        assert!(client.is_allowed("+15551111111"));
    }

    #[test]
    fn test_is_allowed_no() {
        let client = SignalRestClient {
            base_url: "http://localhost:8080".to_string(),
            phone_number: "+15551234567".to_string(),
            allowed_senders: vec!["+15559876543".to_string()],
            http: reqwest::Client::new(),
        };
        assert!(!client.is_allowed("+15550000000"));
        assert!(!client.is_allowed(""));
    }

    #[test]
    fn test_is_allowed_empty_list() {
        let client = SignalRestClient {
            base_url: "http://localhost:8080".to_string(),
            phone_number: "+15551234567".to_string(),
            allowed_senders: vec![],
            http: reqwest::Client::new(),
        };
        assert!(!client.is_allowed("+15559876543"));
    }

    #[tokio::test]
    async fn test_mock_signal_send_receive() {
        let mock = mock::MockSignalClient::new(vec!["+15559876543".to_string()]);

        // Enqueue messages
        mock.enqueue_messages(vec![IncomingMessage {
            sender: "+15559876543".to_string(),
            text: "Hello!".to_string(),
            timestamp: 12345,
        }]);

        let msgs = mock.receive().await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text, "Hello!");

        // Second receive returns empty
        let msgs2 = mock.receive().await.unwrap();
        assert!(msgs2.is_empty());

        // Send
        mock.send("+15559876543", "Reply").await.unwrap();
        let sent = mock.get_sent();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].0, "+15559876543");
        assert_eq!(sent[0].1, "Reply");
    }

    #[test]
    fn test_mock_is_allowed() {
        let mock = mock::MockSignalClient::new(vec!["+15559876543".to_string()]);
        assert!(mock.is_allowed("+15559876543"));
        assert!(!mock.is_allowed("+15550000000"));
    }
}
