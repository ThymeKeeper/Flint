use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};

use crate::config::SignalConfig;

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
        let url = format!(
            "{}/v1/receive/{}",
            self.base_url, self.phone_number
        );

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
