use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContaminationResult {
    pub training_file: String,
    pub training_line: usize,
    pub eval_dataset: String,
    #[serde(default)]
    pub eval_key: Option<String>,  // The clean dataset identifier (e.g., "mmlu_pro")
    pub eval_line: usize,
    #[serde(default)]
    pub eval_instance_index: Option<usize>,  // The actual dataset instance index
    #[serde(default)]
    pub split: Option<String>,  // The split (train/validation/test) of the eval dataset
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub contamination_start_idx: Option<usize>,
    #[serde(default)]
    pub contamination_end_idx: Option<usize>,
    #[serde(default)]
    pub question_start_idx: Option<usize>,
    #[serde(default)]
    pub question_end_idx: Option<usize>,
    #[serde(default)]
    pub training_overlap_text: Option<String>,
    #[serde(default)]
    pub eval_overlap_text: Option<String>,
    #[serde(default)]
    pub ngram_match_cnt: Option<usize>,
    #[serde(default)]
    pub eval_unique_ngrams: Option<usize>,
    #[serde(default)]
    pub contamination_score: Option<f32>,
    #[serde(default)]
    pub length_penalty: Option<f32>,
    #[serde(default)]
    pub answer_overlap_ratio: Option<f32>,
    #[serde(default)]
    pub answer_idf_overlap: Option<f32>,
    #[serde(default)]
    pub answer_start_idx: Option<usize>,
    #[serde(default)]
    pub answer_end_idx: Option<usize>,
    #[serde(default)]
    pub passage_start_idx: Option<usize>,
    #[serde(default)]
    pub passage_end_idx: Option<usize>,
    #[serde(default)]
    pub idf_overlap: Option<f32>,
    #[serde(default)]
    pub cluster_token_length: Option<usize>,
    #[serde(default)]
    pub eval_token_length: Option<usize>,
    #[serde(default)]
    pub token_length_delta: Option<i32>,
    #[serde(default)]
    pub ngram_jaccard: Option<f32>,
    #[serde(default)]
    pub length_adjusted_question_threshold: Option<f32>,
    #[serde(default)]
    pub passage_overlap_ratio: Option<f32>,
    #[serde(default)]
    pub passage_idf_overlap: Option<f32>,
    #[serde(default)]
    pub eval_question_text: Option<String>,
    #[serde(default)]
    pub eval_answer_text: Option<String>,
    #[serde(default)]
    pub eval_passage_text: Option<String>,
    #[serde(default)]
    pub fingerprint: Option<String>,
    #[serde(default)]
    pub is_correct: Option<bool>,
    #[serde(default)]
    pub reference_file: Option<String>,
}

// Struct to hold contamination result with its source file path
#[derive(Debug)]
pub struct ContaminationResultWithSource {
    pub result: ContaminationResult,
    pub source_file: PathBuf,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    fn create_test_result() -> ContaminationResult {
        ContaminationResult {
            training_file: "train.jsonl".to_string(),
            training_line: 42,
            eval_dataset: "mmlu".to_string(),
            eval_key: Some("mmlu_pro".to_string()),
            eval_line: 10,
            eval_instance_index: Some(5),
            split: Some("test".to_string()),
            method: Some("simple".to_string()),
            contamination_start_idx: Some(100),
            contamination_end_idx: Some(200),
            question_start_idx: Some(110),
            question_end_idx: Some(190),
            training_overlap_text: Some("Example text".to_string()),
            eval_overlap_text: Some("Eval text".to_string()),
            ngram_match_cnt: Some(15),
            eval_unique_ngrams: Some(20),
            contamination_score: Some(0.9),
            length_penalty: None,
            answer_overlap_ratio: Some(0.7),
            answer_idf_overlap: Some(0.6),
            answer_start_idx: Some(150),
            answer_end_idx: Some(180),
            passage_start_idx: None,
            passage_end_idx: None,
            idf_overlap: Some(0.75),
            cluster_token_length: Some(100),
            eval_token_length: Some(95),
            token_length_delta: Some(5),
            ngram_jaccard: None,
            length_adjusted_question_threshold: Some(0.8),
            passage_overlap_ratio: None,
            passage_idf_overlap: None,
            eval_question_text: Some("What is the question?".to_string()),
            eval_answer_text: Some("This is the answer.".to_string()),
            eval_passage_text: None,
            fingerprint: Some("abc123".to_string()),
            is_correct: Some(true),
            reference_file: Some("ref.jsonl".to_string()),
        }
    }

    #[test]
    fn test_contamination_result_serialization() {
        let result = create_test_result();
        
        // Serialize to JSON
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("\"training_file\":\"train.jsonl\""));
        assert!(json.contains("\"eval_dataset\":\"mmlu\""));

        // Deserialize back
        let deserialized: ContaminationResult = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.training_file, "train.jsonl");
        assert_eq!(deserialized.training_line, 42);
        assert_eq!(deserialized.eval_dataset, "mmlu");
    }

    #[test]
    fn test_contamination_result_defaults() {
        // Test that we can deserialize with minimal fields
        let json = r#"{
            "training_file": "train.jsonl",
            "training_line": 10,
            "eval_dataset": "gsm8k",
            "eval_line": 5
        }"#;

        let result: ContaminationResult = serde_json::from_str(json).unwrap();

        // Required fields
        assert_eq!(result.training_file, "train.jsonl");
        assert_eq!(result.training_line, 10);
        assert_eq!(result.eval_dataset, "gsm8k");
        assert_eq!(result.eval_line, 5);
        
        // Optional fields should be None
        assert_eq!(result.eval_key, None);
        assert_eq!(result.eval_instance_index, None);
        assert_eq!(result.split, None);
        assert_eq!(result.method, None);
        assert_eq!(result.contamination_start_idx, None);
        assert_eq!(result.training_overlap_text, None);
        assert_eq!(result.ngram_match_cnt, None);
        assert_eq!(result.contamination_score, None);
        assert_eq!(result.fingerprint, None);
        assert_eq!(result.is_correct, None);
    }

    #[test]
    fn test_contamination_result_with_source() {
        let result = create_test_result();
        let source_path = PathBuf::from("/path/to/report.jsonl");
        
        let with_source = ContaminationResultWithSource {
            result: result.clone(),
            source_file: source_path.clone(),
        };
        
        assert_eq!(with_source.result.training_file, "train.jsonl");
        assert_eq!(with_source.source_file, source_path);
    }

    #[test]
    fn test_serialization_with_all_fields() {
        let result = create_test_result();
        let json = serde_json::to_value(&result).unwrap();
        
        // Check that all populated fields are present
        assert!(json.get("training_file").is_some());
        assert!(json.get("eval_key").is_some());
        assert!(json.get("contamination_score").is_some());
        assert!(json.get("answer_overlap_ratio").is_some());
        assert!(json.get("fingerprint").is_some());
        assert!(json.get("is_correct").is_some());
        
        // Check that None fields are not present (or null)
        let json_str = serde_json::to_string(&result).unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(reparsed.get("length_penalty").is_none() || reparsed.get("length_penalty") == Some(&serde_json::Value::Null));
        assert!(reparsed.get("passage_start_idx").is_none() || reparsed.get("passage_start_idx") == Some(&serde_json::Value::Null));
    }
}