use chrono::Utc;

use crate::types::{SearchResult, SummaryRecord};

pub fn rerank_results(
    semantic_hits: Vec<(SummaryRecord, f32)>,
    entity_hits: Vec<SummaryRecord>,
    limit: usize,
) -> Vec<SearchResult> {
    let now = Utc::now();
    let mut merged: Vec<SearchResult> = semantic_hits
        .into_iter()
        .map(|(record, semantic_score)| {
            let age_hours = (now - record.created_at).num_hours().max(0) as f32;
            let recency_score = 1.0 / (1.0 + age_hours / 24.0);
            let entity_score = if entity_hits
                .iter()
                .any(|hit| hit.block_id == record.block_id)
            {
                1.0
            } else {
                0.0
            };
            let final_score = semantic_score * 0.65 + entity_score * 0.2 + recency_score * 0.15;

            SearchResult {
                block_id: record.block_id,
                topic: record.topic,
                summary: record.summary,
                entities: record.entities,
                semantic_score,
                entity_score,
                recency_score,
                final_score,
            }
        })
        .collect();

    for record in entity_hits {
        if merged
            .iter()
            .any(|existing| existing.block_id == record.block_id)
        {
            continue;
        }

        let age_hours = (now - record.created_at).num_hours().max(0) as f32;
        let recency_score = 1.0 / (1.0 + age_hours / 24.0);
        let entity_score = 1.0;
        let semantic_score = 0.0;
        let final_score = semantic_score * 0.65 + entity_score * 0.2 + recency_score * 0.15;

        merged.push(SearchResult {
            block_id: record.block_id,
            topic: record.topic,
            summary: record.summary,
            entities: record.entities,
            semantic_score,
            entity_score,
            recency_score,
            final_score,
        });
    }

    merged.sort_by(|a, b| b.final_score.total_cmp(&a.final_score));
    merged.truncate(limit);
    merged
}
