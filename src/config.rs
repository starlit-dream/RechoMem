use std::env;
use std::path::PathBuf;

use crate::error::{RechoMemError, Result};

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub data_dir: PathBuf,
    pub sqlite_path: PathBuf,
    pub lancedb_path: PathBuf,
    pub raw_log_path: PathBuf,
    pub openai_api_key: String,
    pub openai_base_url: Option<String>,
    pub embedding_model: String,
    pub summary_model: String,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        let data_dir =
            PathBuf::from(env::var("RECHOMEM_DATA_DIR").unwrap_or_else(|_| "./data".to_string()));
        let sqlite_path = PathBuf::from(env::var("RECHOMEM_SQLITE_PATH").unwrap_or_else(|_| {
            data_dir
                .join("rechomem.sqlite")
                .to_string_lossy()
                .into_owned()
        }));
        let lancedb_path = PathBuf::from(
            env::var("RECHOMEM_LANCEDB_PATH")
                .unwrap_or_else(|_| data_dir.join("lancedb").to_string_lossy().into_owned()),
        );
        let raw_log_path = PathBuf::from(env::var("RECHOMEM_RAW_LOG_PATH").unwrap_or_else(|_| {
            data_dir
                .join("raw_logs.jsonl")
                .to_string_lossy()
                .into_owned()
        }));

        let openai_api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| RechoMemError::Config("OPENAI_API_KEY is required".to_string()))?;

        let openai_base_url = env::var("OPENAI_BASE_URL").ok();
        let embedding_model = env::var("RECHOMEM_EMBEDDING_MODEL")
            .unwrap_or_else(|_| "text-embedding-3-small".to_string());
        let summary_model =
            env::var("RECHOMEM_SUMMARY_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

        Ok(Self {
            data_dir,
            sqlite_path,
            lancedb_path,
            raw_log_path,
            openai_api_key,
            openai_base_url,
            embedding_model,
            summary_model,
        })
    }

    pub fn embedding_dimensions(&self) -> usize {
        match self.embedding_model.as_str() {
            "text-embedding-3-small" => 1536,
            "text-embedding-3-large" => 3072,
            "text-embedding-ada-002" => 1536,
            _ => 1536,
        }
    }
}
