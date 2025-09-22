use std::collections::HashSet;
use crate::review::ContaminationResult;

#[derive(Debug)]
pub struct ContaminationMetrics {
    pub training_docs_contaminated: usize,  // Unique (training_file, training_line) tuples
    pub contamination_instances: usize,      // Total number of report lines
    pub contaminated_evals: usize,           // Unique (eval_key, split, eval_instance_index) tuples
}

pub fn compute_contamination_metrics(results: &[ContaminationResult]) -> ContaminationMetrics {
    let mut unique_training_docs = HashSet::new();
    let mut unique_eval_instances = HashSet::new();

    for result in results {
        unique_training_docs.insert((result.training_file.clone(), result.training_line));

        let eval_instance_key = result.eval_instance_index
            .expect("eval_instance_index is required but was not found in contamination result");
        let eval_suite = result.eval_key.as_ref()
            .unwrap_or(&result.eval_dataset)
            .clone();
        let split = result.split.clone()
            .unwrap_or_else(|| "unknown".to_string());
        unique_eval_instances.insert((eval_suite, split, eval_instance_key));
    }

    ContaminationMetrics {
        training_docs_contaminated: unique_training_docs.len(),
        contamination_instances: results.len(),
        contaminated_evals: unique_eval_instances.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::ContaminationResult;

    fn create_test_contamination_result(
        training_file: &str,
        training_line: usize,
        eval_dataset: &str,
        eval_line: usize,
        eval_instance_index: Option<usize>,
        eval_key: Option<String>,
        split: Option<String>,
    ) -> ContaminationResult {
        ContaminationResult {
            training_file: training_file.to_string(),
            training_line,
            eval_dataset: eval_dataset.to_string(),
            eval_key,
            eval_line,
            eval_instance_index,
            split,
            method: None,
            contamination_start_idx: None,
            contamination_end_idx: None,
            question_start_idx: None,
            question_end_idx: None,
            training_overlap_text: None,
            eval_overlap_text: None,
            ngram_match_cnt: None,
            eval_unique_ngrams: None,
            contamination_score: Some(0.5),
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
    fn test_empty_results() {
        let results = vec![];
        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 0);
        assert_eq!(metrics.contamination_instances, 0);
        assert_eq!(metrics.contaminated_evals, 0);
    }

    #[test]
    fn test_single_contamination_result() {
        let results = vec![
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 5, Some(100),
                Some("mmlu_pro".to_string()), Some("test".to_string())
            ),
        ];
        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 1);
        assert_eq!(metrics.contamination_instances, 1);
        assert_eq!(metrics.contaminated_evals, 1);
    }

    #[test]
    fn test_multiple_results_same_training_doc() {
        let results = vec![
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 5, Some(100),
                Some("mmlu_pro".to_string()), Some("test".to_string())
            ),
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 15, Some(101),
                Some("mmlu_pro".to_string()), Some("test".to_string())
            ),
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 25, Some(102),
                Some("mmlu_pro".to_string()), Some("test".to_string())
            ),
        ];
        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 1); // Same training doc
        assert_eq!(metrics.contamination_instances, 3);
        assert_eq!(metrics.contaminated_evals, 3); // Different eval instances
    }

    #[test]
    fn test_multiple_results_different_training_docs() {
        let results = vec![
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 5, Some(100),
                Some("mmlu_pro".to_string()), Some("test".to_string())
            ),
            create_test_contamination_result(
                "train2.jsonl", 20, "mmlu", 5, Some(100),
                Some("mmlu_pro".to_string()), Some("test".to_string())
            ),
            create_test_contamination_result(
                "train3.jsonl", 30, "mmlu", 5, Some(100),
                Some("mmlu_pro".to_string()), Some("test".to_string())
            ),
        ];
        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 3); // Different training docs
        assert_eq!(metrics.contamination_instances, 3);
        assert_eq!(metrics.contaminated_evals, 1); // Same eval instance
    }

    #[test]
    fn test_duplicate_training_docs_counted_once() {
        let results = vec![
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 5, Some(100),
                None, None
            ),
            create_test_contamination_result(
                "train1.jsonl", 10, "gsm8k", 15, Some(200),
                None, None
            ),
        ];
        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 1); // Same file and line
        assert_eq!(metrics.contamination_instances, 2);
        assert_eq!(metrics.contaminated_evals, 2); // Different eval datasets
    }

    #[test]
    fn test_eval_key_fallback_to_dataset() {
        let results = vec![
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 5, Some(100),
                None, // No eval_key, should use eval_dataset
                Some("test".to_string())
            ),
        ];
        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.contaminated_evals, 1);
    }

    #[test]
    fn test_split_defaults_to_unknown() {
        let results = vec![
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 5, Some(100),
                Some("mmlu_pro".to_string()),
                None // No split, should default to "unknown"
            ),
        ];
        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.contaminated_evals, 1);
    }

    #[test]
    fn test_same_eval_instance_different_splits() {
        let results = vec![
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 5, Some(100),
                Some("mmlu_pro".to_string()), Some("train".to_string())
            ),
            create_test_contamination_result(
                "train2.jsonl", 20, "mmlu", 5, Some(100),
                Some("mmlu_pro".to_string()), Some("test".to_string())
            ),
        ];
        let metrics = compute_contamination_metrics(&results);

        assert_eq!(metrics.training_docs_contaminated, 2);
        assert_eq!(metrics.contamination_instances, 2);
        assert_eq!(metrics.contaminated_evals, 2); // Different splits = different eval instances
    }

    #[test]
    #[should_panic(expected = "eval_instance_index is required")]
    fn test_missing_eval_instance_index_panics() {
        let results = vec![
            create_test_contamination_result(
                "train1.jsonl", 10, "mmlu", 5, None, // No eval_instance_index
                Some("mmlu_pro".to_string()), Some("test".to_string())
            ),
        ];
        compute_contamination_metrics(&results);
    }
}
