use chrono::{DateTime, Utc};
use rmcp::schemars;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryBlock {
    pub block_id: String,
    pub topic: String,
    pub summary: String,
    pub entities: Vec<String>,
    pub status: String,
    pub raw_offset: u64,
    pub raw_length: u64,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SummaryRecord {
    pub block_id: String,
    pub topic: String,
    pub summary: String,
    pub entities: Vec<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchResult {
    pub block_id: String,
    pub topic: String,
    pub summary: String,
    pub entities: Vec<String>,
    pub semantic_score: f32,
    pub entity_score: f32,
    pub recency_score: f32,
    pub final_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchResultsResponse {
    pub results: Vec<SearchResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RawLogChunk {
    pub block_id: String,
    pub content: String,
    pub raw_offset: u64,
    pub raw_length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct RawConversationLine {
    pub timestamp: DateTime<Utc>,
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct StructuredRawLogChunk {
    pub block_id: String,
    pub lines: Vec<RawConversationLine>,
    pub raw_offset: u64,
    pub raw_length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IngestMemoryBlockRequest {
    pub topic: String,
    pub summary: String,
    pub entities: Vec<String>,
    pub raw_lines: Vec<RawConversationLine>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IngestConversationRequest {
    pub raw_lines: Vec<RawConversationLine>,
    pub topic_hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct IngestMemoryBlockResult {
    pub block_id: String,
    pub raw_offset: u64,
    pub raw_length: u64,
    pub created_at: DateTime<Utc>,
}
