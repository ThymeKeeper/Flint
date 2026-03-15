use anyhow::{Context, Result};
use async_trait::async_trait;
use fastembed::{InitOptionsUserDefined, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
use sha2::{Digest, Sha256};
use std::sync::Mutex;
use tracing::debug;

/// Output dimension of the bundled BGE-small-en-v1.5 model.
pub const EMBEDDING_DIM: usize = 384;

// ---------------------------------------------------------------------------
// Trait for testability
// ---------------------------------------------------------------------------

#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    /// A fingerprint that uniquely identifies the embedding model.
    /// If this changes between runs, all stored embeddings must be regenerated.
    fn model_fingerprint(&self) -> String;
}

// ---------------------------------------------------------------------------
// LocalEmbeddingClient — bundled BGE-small-en-v1.5-Q (no API key needed)
// ---------------------------------------------------------------------------

/// Runs the quantized BGE-small-en-v1.5 ONNX model in-process.
/// Model bytes are compiled into the binary at build time via `include_bytes!`.
pub struct LocalEmbeddingClient {
    model: Mutex<TextEmbedding>,
    /// SHA-256 hex digest of model.onnx — changes when the model file changes.
    fingerprint: String,
}

impl LocalEmbeddingClient {
    /// Load the model from the directory baked in at compile time by build.rs.
    pub fn new() -> Result<Self> {
        let dir = std::path::PathBuf::from(env!("FLINT_MODELS_DIR"));
        Self::from_dir(&dir)
    }

    /// Load the model from an explicit directory (useful for tests / custom installs).
    pub fn from_dir(dir: &std::path::Path) -> Result<Self> {
        let read = |name: &str| -> Result<Vec<u8>> {
            std::fs::read(dir.join(name)).with_context(|| {
                format!(
                    "Embedding model file '{}' not found in {}. \
                     Run `cargo build` to download it.",
                    name,
                    dir.display()
                )
            })
        };

        let model_bytes = read("model.onnx")?;
        let fingerprint = format!("{:x}", Sha256::digest(&model_bytes));

        let user_model = UserDefinedEmbeddingModel::new(
            model_bytes,
            TokenizerFiles {
                tokenizer_file: read("tokenizer.json")?,
                config_file: read("config.json")?,
                special_tokens_map_file: read("special_tokens_map.json")?,
                tokenizer_config_file: read("tokenizer_config.json")?,
            },
        );

        let embedding =
            TextEmbedding::try_new_from_user_defined(user_model, InitOptionsUserDefined::new())
                .context("Failed to initialize local embedding model")?;

        Ok(Self {
            model: Mutex::new(embedding),
            fingerprint,
        })
    }
}

#[async_trait]
impl EmbeddingClient for LocalEmbeddingClient {
    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        debug!("Embedding text locally ({} chars)", text.len());
        let mut results = self
            .model
            .lock()
            .unwrap()
            .embed(vec![text.to_string()], None)
            .context("Local embedding failed")?;
        results.pop().context("Empty embedding result")
    }

    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        debug!("Embedding {} texts locally", texts.len());
        self.model
            .lock()
            .unwrap()
            .embed(texts.to_vec(), None)
            .context("Local batch embedding failed")
    }

    fn model_fingerprint(&self) -> String {
        self.fingerprint.clone()
    }
}

// ---------------------------------------------------------------------------
// Cosine similarity (used by memory.rs for in-process search)
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

    /// Returns deterministic vectors based on input text hash.  Used in tests
    /// so the real model is not needed.
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
            for (i, byte) in text.bytes().enumerate() {
                let idx = i % self.dimensions;
                emb[idx] += (byte as f32) / 255.0;
            }
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

        fn model_fingerprint(&self) -> String {
            "mock-embedding-model".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mock::MockEmbeddingClient;
    use super::*;

    #[tokio::test]
    async fn test_mock_embedding_deterministic() {
        let client = MockEmbeddingClient::new(EMBEDDING_DIM);
        let e1 = client.embed("hello").await.unwrap();
        let e2 = client.embed("hello").await.unwrap();
        assert_eq!(e1, e2);
        assert_eq!(e1.len(), EMBEDDING_DIM);
    }

    #[tokio::test]
    async fn test_mock_embedding_different_texts() {
        let client = MockEmbeddingClient::new(EMBEDDING_DIM);
        let e1 = client.embed("hello").await.unwrap();
        let e2 = client.embed("world").await.unwrap();
        assert_ne!(e1, e2);
    }

    #[tokio::test]
    async fn test_mock_embedding_normalized() {
        let client = MockEmbeddingClient::new(EMBEDDING_DIM);
        let e = client.embed("test text").await.unwrap();
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }
}
