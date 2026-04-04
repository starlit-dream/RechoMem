mod config;
mod embedding;
mod error;
mod ingest;
mod mcp;
mod retrieval;
mod storage;
mod types;

use std::sync::Arc;

use config::AppConfig;
use embedding::EmbeddingService;
use error::Result;
use ingest::IngestionService;
use mcp::{AppState, RechoMemMcp};
use storage::jsonl::JsonlStore;
use storage::sqlite::SqliteStore;
use storage::vector::VectorStore;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let config = AppConfig::from_env()?;
    tokio::fs::create_dir_all(&config.data_dir).await?;
    tokio::fs::create_dir_all(&config.lancedb_path).await?;

    let sqlite_url = SqliteStore::database_url(&config.sqlite_path);
    let sqlite = SqliteStore::connect(&sqlite_url).await?;
    let jsonl = JsonlStore::new(config.raw_log_path.to_string_lossy().to_string());
    jsonl.ensure_exists().await?;
    let expected_dimensions = config.embedding_dimensions();
    let vector = VectorStore::new(
        config.lancedb_path.to_string_lossy().to_string(),
        "memory_index",
    );
    vector.init(expected_dimensions as i32).await?;
    let embedding = EmbeddingService::from_config(&config);
    if embedding.dimensions() != expected_dimensions {
        return Err(crate::error::RechoMemError::Config(format!(
            "embedding dimension mismatch: expected {}, got {}",
            expected_dimensions,
            embedding.dimensions()
        )));
    }
    let ingestion = IngestionService::new(
        jsonl.clone(),
        sqlite.clone(),
        vector.clone(),
        embedding.clone(),
    );

    let state = Arc::new(AppState {
        sqlite,
        jsonl,
        vector,
        embedding,
        ingestion,
    });
    let service = RechoMemMcp::new(state);

    mcp::run_stdio_server(service).await
}
