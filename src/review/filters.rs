use crate::review::ContaminationResult;

/// Filter contamination results by various thresholds and criteria
pub fn filter_contamination_results_by_thresholds(results: Vec<ContaminationResult>, min_score: Option<f32>, min_length: Option<usize>, eval_filter: Option<&str>) -> Vec<ContaminationResult> {
    results
        .into_iter()
        .filter(|result| {
            // Check eval dataset filter using eval_key if available
            if let Some(eval_name) = eval_filter {
                // Use eval_key if available (clean dataset identifier)
                // Fall back to eval_dataset for backward compatibility
                let dataset_to_check = result.eval_key.as_ref()
                    .unwrap_or(&result.eval_dataset);

                // Simple exact match
                if dataset_to_check != eval_name {
                    return false;
                }
            }

            // Check contamination score
            if let Some(min_score_threshold) = min_score {
                if let Some(score) = result.contamination_score {
                    if score < min_score_threshold {
                        return false;
                    }
                } else {
                    return false; // If contamination_score is not present, filter out
                }
            }

            // Check n-gram match count
            if let Some(min_len) = min_length {
                if let Some(match_cnt) = result.ngram_match_cnt {
                    if match_cnt < min_len {
                        return false;
                    }
                } else {
                    // If ngram_match_cnt is not present, filter out
                    return false;
                }
            }

            true
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_result(
        score: Option<f32>,
        ngram_count: Option<usize>,
        eval_dataset: &str,
        eval_key: Option<String>,
    ) -> ContaminationResult {
        ContaminationResult {
            training_file: "test.txt".to_string(),
            training_line: 1,
            eval_dataset: eval_dataset.to_string(),
            eval_key,
            eval_line: 1,
            eval_instance_index: None,
            split: None,
            method: None,
            contamination_start_idx: None,
            contamination_end_idx: None,
            question_start_idx: None,
            question_end_idx: None,
            training_overlap_text: None,
            eval_overlap_text: None,
            ngram_match_cnt: ngram_count,
            eval_unique_ngrams: None,
            contamination_score: score,
            length_penalty: None,
            answer_overlap_ratio: None,
            answer_idf_overlap: None,
            answer_start_idx: None,
            answer_end_idx: None,
            passage_start_idx: None,
            passage_end_idx: None,
            idf_overlap: None,
            cluster_token_length: None,
            eval_token_length: None,
            token_length_delta: None,
            ngram_jaccard: None,
            length_adjusted_question_threshold: None,
            passage_overlap_ratio: None,
            passage_idf_overlap: None,
            eval_question_text: None,
            eval_answer_text: None,
            eval_passage_text: None,
            fingerprint: None,
            is_correct: None,
            reference_file: None,
        }
    }

    #[test]
    fn test_filter_by_min_score() {
        let results = vec![
            create_test_result(Some(0.5), None, "dataset1", None),
            create_test_result(Some(0.8), None, "dataset1", None),
            create_test_result(Some(0.9), None, "dataset1", None),
            create_test_result(None, None, "dataset1", None),
        ];

        let filtered = filter_contamination_results_by_thresholds(
            results,
            Some(0.7),
            None,
            None,
        );

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].contamination_score, Some(0.8));
        assert_eq!(filtered[1].contamination_score, Some(0.9));
    }

    #[test]
    fn test_filter_by_min_length() {
        let results = vec![
            create_test_result(None, Some(5), "dataset1", None),
            create_test_result(None, Some(10), "dataset1", None),
            create_test_result(None, Some(15), "dataset1", None),
            create_test_result(None, None, "dataset1", None),
        ];

        let filtered = filter_contamination_results_by_thresholds(
            results,
            None,
            Some(8),
            None,
        );

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].ngram_match_cnt, Some(10));
        assert_eq!(filtered[1].ngram_match_cnt, Some(15));
    }

    #[test]
    fn test_filter_by_eval_dataset() {
        let results = vec![
            create_test_result(None, None, "mmlu", None),
            create_test_result(None, None, "gsm8k", None),
            create_test_result(None, None, "mmlu", Some("mmlu_pro".to_string())),
            create_test_result(None, None, "gsm8k", None),
        ];

        let filtered = filter_contamination_results_by_thresholds(
            results,
            None,
            None,
            Some("mmlu"),
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].eval_dataset, "mmlu");
        assert_eq!(filtered[0].eval_key, None);
    }

    #[test]
    fn test_filter_by_eval_key_when_available() {
        let results = vec![
            create_test_result(None, None, "mmlu_full", Some("mmlu".to_string())),
            create_test_result(None, None, "gsm8k_full", Some("gsm8k".to_string())),
            create_test_result(None, None, "mmlu_full", Some("mmlu".to_string())),
        ];

        let filtered = filter_contamination_results_by_thresholds(
            results,
            None,
            None,
            Some("mmlu"),
        );

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].eval_key, Some("mmlu".to_string()));
        assert_eq!(filtered[1].eval_key, Some("mmlu".to_string()));
    }

    #[test]
    fn test_combined_filters() {
        let results = vec![
            create_test_result(Some(0.9), Some(20), "mmlu", None),
            create_test_result(Some(0.5), Some(30), "mmlu", None),
            create_test_result(Some(0.9), Some(5), "mmlu", None),
            create_test_result(Some(0.9), Some(20), "gsm8k", None),
        ];

        let filtered = filter_contamination_results_by_thresholds(
            results,
            Some(0.8),
            Some(10),
            Some("mmlu"),
        );

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].contamination_score, Some(0.9));
        assert_eq!(filtered[0].ngram_match_cnt, Some(20));
        assert_eq!(filtered[0].eval_dataset, "mmlu");
    }

    #[test]
    fn test_no_filters_returns_all() {
        let results = vec![
            create_test_result(Some(0.1), Some(1), "dataset1", None),
            create_test_result(None, None, "dataset2", None),
            create_test_result(Some(0.9), Some(100), "dataset3", None),
        ];

        let filtered = filter_contamination_results_by_thresholds(
            results.clone(),
            None,
            None,
            None,
        );

        assert_eq!(filtered.len(), results.len());
    }

    #[test]
    fn test_empty_input_returns_empty() {
        let results = vec![];

        let filtered = filter_contamination_results_by_thresholds(
            results,
            Some(0.5),
            Some(10),
            Some("mmlu"),
        );

        assert_eq!(filtered.len(), 0);
    }
}
