use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::config::VoyageConfig;

// ---------------------------------------------------------------------------
// Trait for testability
// ---------------------------------------------------------------------------

#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

// ---------------------------------------------------------------------------
// Voyage AI client
// ---------------------------------------------------------------------------

pub struct VoyageClient {
    api_key: String,
    model: String,
    dimensions: usize,
    http: reqwest::Client,
}

impl VoyageClient {
    pub fn new(api_key: String, config: &VoyageConfig) -> Self {
        Self {
            api_key,
            model: config.model.clone(),
            dimensions: config.dimensions,
            http: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct VoyageRequest {
    input: Vec<String>,
    model: String,
    output_dimension: usize,
}

#[derive(Deserialize)]
struct VoyageResponse {
    data: Vec<VoyageEmbedding>,
}

#[derive(Deserialize)]
struct VoyageEmbedding {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingClient for VoyageClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .context("Empty response from Voyage AI")
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        debug!("Embedding {} texts via Voyage AI", texts.len());

        let request = VoyageRequest {
            input: texts.to_vec(),
            model: self.model.clone(),
            output_dimension: self.dimensions,
        };

        let response = self
            .http
            .post("https://api.voyageai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Voyage AI")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response
                .text()
                .await
                .unwrap_or_else(|_| "no body".to_string());
            bail!("Voyage AI returned {status}: {body}");
        }

        let voyage_resp: VoyageResponse = response
            .json()
            .await
            .context("Failed to parse Voyage AI response")?;

        let embeddings: Vec<Vec<f32>> = voyage_resp.data.into_iter().map(|e| e.embedding).collect();

        debug!(
            "Received {} embeddings (dim={})",
            embeddings.len(),
            embeddings.first().map(|e| e.len()).unwrap_or(0)
        );

        Ok(embeddings)
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity (used by memory.rs for in-process clustering)
// ---------------------------------------------------------------------------

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// Mock for testing
// ---------------------------------------------------------------------------

pub mod mock {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A mock embedding client that returns deterministic vectors.
    /// Each call returns a unique vector based on a hash of the input text.
    pub struct MockEmbeddingClient {
        pub dimensions: usize,
        pub call_count: AtomicUsize,
    }

    impl MockEmbeddingClient {
        pub fn new(dimensions: usize) -> Self {
            Self {
                dimensions,
                call_count: AtomicUsize::new(0),
            }
        }

        fn text_to_embedding(&self, text: &str) -> Vec<f32> {
            let mut emb = vec![0.0f32; self.dimensions];
            // Simple deterministic hash-based embedding
            for (i, byte) in text.bytes().enumerate() {
                let idx = i % self.dimensions;
                emb[idx] += (byte as f32) / 255.0;
            }
            // Normalize
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut emb {
                    *v /= norm;
                }
            }
            emb
        }
    }

    #[async_trait]
    impl EmbeddingClient for MockEmbeddingClient {
        async fn embed(&self, text: &str) -> Result<Vec<f32>> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            Ok(self.text_to_embedding(text))
        }

        async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            Ok(texts.iter().map(|t| self.text_to_embedding(t)).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mock::MockEmbeddingClient;
    use super::*;

    #[tokio::test]
    async fn test_mock_embedding_deterministic() {
        let client = MockEmbeddingClient::new(1024);
        let e1 = client.embed("hello").await.unwrap();
        let e2 = client.embed("hello").await.unwrap();
        assert_eq!(e1, e2);
        assert_eq!(e1.len(), 1024);
    }

    #[tokio::test]
    async fn test_mock_embedding_different_texts() {
        let client = MockEmbeddingClient::new(1024);
        let e1 = client.embed("hello").await.unwrap();
        let e2 = client.embed("world").await.unwrap();
        assert_ne!(e1, e2);
    }

    #[tokio::test]
    async fn test_mock_embedding_normalized() {
        let client = MockEmbeddingClient::new(1024);
        let e = client.embed("test text").await.unwrap();
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
