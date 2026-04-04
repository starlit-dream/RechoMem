use async_openai::config::OpenAIConfig;
use async_openai::types::{
    ChatCompletionRequestSystemMessage, ChatCompletionRequestUserMessage,
    CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs, ResponseFormat,
    ResponseFormatJsonSchema,
};
use async_openai::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::AppConfig;
use crate::error::{RechoMemError, Result};
use crate::types::RawConversationLine;

#[derive(Clone)]
pub struct EmbeddingService {
    backend: EmbeddingBackend,
    model: String,
    dimensions: usize,
    summary_model: String,
}

#[derive(Clone)]
enum EmbeddingBackend {
    OpenAi(Client<OpenAIConfig>),
    #[cfg(test)]
    Fake,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedSummary {
    pub topic: String,
    pub summary: String,
    pub entities: Vec<String>,
}

impl EmbeddingService {
    pub fn from_config(config: &AppConfig) -> Self {
        let mut client_config = OpenAIConfig::new().with_api_key(config.openai_api_key.clone());
        if let Some(base_url) = &config.openai_base_url {
            client_config = client_config.with_api_base(base_url.as_str());
        }

        Self {
            backend: EmbeddingBackend::OpenAi(Client::with_config(client_config)),
            model: config.embedding_model.clone(),
            dimensions: config.embedding_dimensions(),
            summary_model: config.summary_model.clone(),
        }
    }

    #[cfg(test)]
    pub fn fake(dimensions: usize) -> Self {
        Self {
            backend: EmbeddingBackend::Fake,
            model: "fake-embedding".to_string(),
            dimensions,
            summary_model: "fake-summary".to_string(),
        }
    }

    #[cfg(test)]
    pub fn from_openai_config_for_test(
        config: OpenAIConfig,
        embedding_model: &str,
        summary_model: &str,
    ) -> Self {
        Self {
            backend: EmbeddingBackend::OpenAi(Client::with_config(config)),
            model: embedding_model.to_string(),
            dimensions: 1536,
            summary_model: summary_model.to_string(),
        }
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub async fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let embedding = match &self.backend {
            EmbeddingBackend::OpenAi(client) => {
                let request = CreateEmbeddingRequestArgs::default()
                    .model(&self.model)
                    .input(text)
                    .build()
                    .map_err(|e| RechoMemError::Internal(e.to_string()))?;

                let response = client.embeddings().create(request).await?;
                response
                    .data
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        RechoMemError::Internal("embedding response was empty".to_string())
                    })?
                    .embedding
            }
            #[cfg(test)]
            EmbeddingBackend::Fake => fake_embedding(text, self.dimensions),
        };

        if embedding.len() != self.dimensions {
            return Err(RechoMemError::Internal(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        Ok(embedding)
    }

    pub async fn summarize_lines(
        &self,
        raw_lines: &[RawConversationLine],
        topic_hint: Option<&str>,
    ) -> Result<GeneratedSummary> {
        match &self.backend {
            EmbeddingBackend::OpenAi(client) => {
                let transcript = raw_lines
                    .iter()
                    .map(|line| {
                        format!(
                            "[{}] {}: {}",
                            line.timestamp.to_rfc3339(),
                            line.role,
                            line.content
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let hint = topic_hint.unwrap_or("无");
                let response_schema = ResponseFormat::JsonSchema {
                    json_schema: ResponseFormatJsonSchema {
                        description: Some(
                            "Structured conversation summary with topic, summary, and entities"
                                .to_string(),
                        ),
                        name: "generated_summary".into(),
                        schema: Some(json!({
                            "type": "object",
                            "properties": {
                                "topic": { "type": "string" },
                                "summary": { "type": "string" },
                                "entities": {
                                    "type": "array",
                                    "items": { "type": "string" }
                                }
                            },
                            "required": ["topic", "summary", "entities"],
                            "additionalProperties": false
                        })),
                        strict: Some(true),
                    },
                };
                let request = CreateChatCompletionRequestArgs::default()
                    .model(&self.summary_model)
                    .max_tokens(300_u32)
                    .messages([
                        ChatCompletionRequestSystemMessage::from(
                            "You summarize conversation logs into JSON. Return only valid JSON with keys topic, summary, entities. topic and summary must be concise strings. entities must be an array of short strings.",
                        )
                        .into(),
                        ChatCompletionRequestUserMessage::from(format!(
                            "topic_hint: {hint}\n\nconversation:\n{transcript}"
                        ))
                        .into(),
                    ])
                    .response_format(response_schema)
                    .build()
                    .map_err(async_openai::error::OpenAIError::from)?;

                let response = client.chat().create(request).await?;
                let content = response
                    .choices
                    .into_iter()
                    .next()
                    .and_then(|choice| choice.message.content)
                    .ok_or_else(|| {
                        RechoMemError::Internal("summary response was empty".to_string())
                    })?;

                serde_json::from_str(&content).map_err(|error| {
                    RechoMemError::Internal(format!("invalid summary json: {error}"))
                })
            }
            #[cfg(test)]
            EmbeddingBackend::Fake => Ok(fake_summary(raw_lines, topic_hint)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::EmbeddingService;
    use crate::types::RawConversationLine;
    use async_openai::config::OpenAIConfig;
    use chrono::Utc;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn summarize_lines_parses_structured_openai_response() -> anyhow::Result<()> {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer test-key"))
            .and(body_partial_json(serde_json::json!({
                "model": "summary-test-model",
                "response_format": {
                    "type": "json_schema"
                }
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "summary-test-model",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "{\"topic\":\"Release planning\",\"summary\":\"Captured release planning details\",\"entities\":[\"OpenAI\",\"Release\"]}"
                    }
                }],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2
                }
            })))
            .mount(&server)
            .await;

        let config = OpenAIConfig::new()
            .with_api_base(server.uri())
            .with_api_key("test-key");
        let service = EmbeddingService::from_openai_config_for_test(
            config,
            "text-embedding-3-small",
            "summary-test-model",
        );

        let summary = service
            .summarize_lines(
                &[RawConversationLine {
                    timestamp: Utc::now(),
                    role: "user".to_string(),
                    content: "Capture the OpenAI release planning notes".to_string(),
                }],
                Some("Release planning"),
            )
            .await?;

        assert_eq!(summary.topic, "Release planning");
        assert_eq!(summary.summary, "Captured release planning details");
        assert_eq!(summary.entities, vec!["OpenAI", "Release"]);

        Ok(())
    }
}

#[cfg(test)]
fn fake_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; dimensions];
    if dimensions == 0 {
        return vector;
    }

    for token in text.split_whitespace() {
        let normalized = token.to_lowercase();
        let slot = normalized
            .bytes()
            .fold(0_usize, |acc, byte| acc.wrapping_add(byte as usize))
            % dimensions;
        vector[slot] += 1.0;
    }

    if vector.iter().all(|value| *value == 0.0) {
        vector[0] = 1.0;
    }

    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }

    vector
}

#[cfg(test)]
fn fake_summary(raw_lines: &[RawConversationLine], topic_hint: Option<&str>) -> GeneratedSummary {
    let topic = topic_hint
        .filter(|value| !value.trim().is_empty())
        .map(|value| value.trim().to_string())
        .unwrap_or_else(|| {
            raw_lines
                .first()
                .map(|line| line.content.chars().take(32).collect::<String>())
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "Conversation summary".to_string())
        });

    let summary = raw_lines
        .iter()
        .map(|line| line.content.trim())
        .filter(|line| !line.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join(" ");

    let mut entities = Vec::new();
    for token in raw_lines
        .iter()
        .flat_map(|line| line.content.split_whitespace())
        .map(|token| {
            token
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|token| token.len() > 2)
    {
        if !entities.iter().any(|existing| existing == &token) {
            entities.push(token);
        }
        if entities.len() >= 5 {
            break;
        }
    }

    GeneratedSummary {
        topic,
        summary: if summary.is_empty() {
            "Conversation summary".to_string()
        } else {
            summary
        },
        entities,
    }
}
