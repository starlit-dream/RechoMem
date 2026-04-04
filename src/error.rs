use thiserror::Error;

pub type Result<T> = std::result::Result<T, RechoMemError>;

#[derive(Debug, Error)]
pub enum RechoMemError {
    #[error("configuration error: {0}")]
    Config(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("sqlite error: {0}")]
    Sqlx(#[from] sqlx::Error),

    #[error("serde json error: {0}")]
    SerdeJson(#[from] serde_json::Error),

    #[error("openai error: {0}")]
    OpenAi(#[from] async_openai::error::OpenAIError),

    #[error("lancedb error: {0}")]
    LanceDb(#[from] lancedb::error::Error),

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("missing record: {0}")]
    NotFound(String),

    #[error("internal error: {0}")]
    Internal(String),
}
