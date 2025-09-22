use crate::review::{ContaminationResult, ContaminationResultWithSource};
use anyhow::{Error, Result};
use mj_io::{expand_dirs, read_pathbuf_to_mem};
use std::io::BufRead;
use std::path::{Path};

/// Load contamination results from a single file
pub fn load_contamination_results(results_path: &Path) -> Result<Vec<ContaminationResult>, Error> {
    let data = read_pathbuf_to_mem(&results_path.to_path_buf())?;
    let mut results = Vec::new();

    for line in data.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            let result: ContaminationResult = serde_json::from_str(&line)?;
            results.push(result);
        }
    }

    Ok(results)
}

pub fn load_contamination_results_from_directory(
    dir_path: &Path,
) -> Result<Vec<ContaminationResult>, Error> {
    let mut all_results = Vec::new();

    // Find all .jsonl files in the directory
    let jsonl_files = expand_dirs(vec![dir_path.to_path_buf()], Some(vec![".jsonl"].as_slice()))?;

    println!(
        "Processing {} JSONL files from directory...",
        jsonl_files.len()
    );

    for file_path in jsonl_files {
        match load_contamination_results(&file_path) {
            Ok(results) => {
                all_results.extend(results);
            }
            Err(e) => {
                // Skip files that can't be parsed as contamination results
                println!("  Skipping file (not a contamination results file): {}", e);
            }
        }
    }

    Ok(all_results)
}

/// Load contamination results from directory with source file tracking
pub fn load_contamination_results_from_directory_with_source(
    dir_path: &Path,
) -> Result<Vec<ContaminationResultWithSource>, Error> {
    let mut all_results = Vec::new();

    // Find all .jsonl files in the directory
    let jsonl_files = expand_dirs(vec![dir_path.to_path_buf()], Some(vec![".jsonl"].as_slice()))?;

    println!(
        "Processing {} JSONL files from directory...",
        jsonl_files.len()
    );

    for file_path in jsonl_files {
        match load_contamination_results(&file_path) {
            Ok(results) => {
                // Add each result with its source file
                for result in results {
                    all_results.push(ContaminationResultWithSource {
                        result,
                        source_file: file_path.clone(),
                    });
                }
            }
            Err(e) => {
                // Skip files that can't be parsed as contamination results
                println!("  Skipping file (not a contamination results file): {}", e);
            }
        }
    }

    Ok(all_results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_test_contamination_result() -> ContaminationResult {
        ContaminationResult {
            training_file: "test.txt".to_string(),
            training_line: 1,
            eval_dataset: "mmlu".to_string(),
            eval_key: None,
            eval_line: 10,
            eval_instance_index: None,
            split: None,
            method: Some("simple".to_string()),
            contamination_start_idx: None,
            contamination_end_idx: None,
            question_start_idx: None,
            question_end_idx: None,
            training_overlap_text: None,
            eval_overlap_text: None,
            ngram_match_cnt: Some(5),
            eval_unique_ngrams: None,
            contamination_score: Some(0.8),
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
    fn test_load_contamination_results_valid_jsonl() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.jsonl");

        let result1 = create_test_contamination_result();
        let mut result2 = create_test_contamination_result();
        result2.training_line = 2;
        result2.eval_line = 20;

        let mut file = fs::File::create(&file_path).unwrap();
        writeln!(file, "{}", serde_json::to_string(&result1).unwrap()).unwrap();
        writeln!(file, "{}", serde_json::to_string(&result2).unwrap()).unwrap();

        let results = load_contamination_results(&file_path).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].training_line, 1);
        assert_eq!(results[0].eval_line, 10);
        assert_eq!(results[1].training_line, 2);
        assert_eq!(results[1].eval_line, 20);
    }

    #[test]
    fn test_load_contamination_results_with_empty_lines() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.jsonl");

        let result = create_test_contamination_result();

        let mut file = fs::File::create(&file_path).unwrap();
        writeln!(file, "{}", serde_json::to_string(&result).unwrap()).unwrap();
        writeln!(file).unwrap();
        writeln!(file, "   ").unwrap();
        writeln!(file, "{}", serde_json::to_string(&result).unwrap()).unwrap();

        let results = load_contamination_results(&file_path).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_load_contamination_results_invalid_json() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.jsonl");

        let mut file = fs::File::create(&file_path).unwrap();
        writeln!(file, "{{invalid json}}").unwrap();

        let result = load_contamination_results(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_contamination_results_from_directory() {
        let temp_dir = TempDir::new().unwrap();

        let result1 = create_test_contamination_result();
        let mut result2 = create_test_contamination_result();
        result2.training_line = 2;

        let file1_path = temp_dir.path().join("file1.jsonl");
        let file2_path = temp_dir.path().join("file2.jsonl");

        let mut file1 = fs::File::create(&file1_path).unwrap();
        writeln!(file1, "{}", serde_json::to_string(&result1).unwrap()).unwrap();

        let mut file2 = fs::File::create(&file2_path).unwrap();
        writeln!(file2, "{}", serde_json::to_string(&result2).unwrap()).unwrap();

        let results = load_contamination_results_from_directory(temp_dir.path()).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_load_contamination_results_from_directory_with_source() {
        let temp_dir = TempDir::new().unwrap();

        let result = create_test_contamination_result();

        let file_path = temp_dir.path().join("file1.jsonl");
        let mut file = fs::File::create(&file_path).unwrap();
        writeln!(file, "{}", serde_json::to_string(&result).unwrap()).unwrap();

        let results = load_contamination_results_from_directory_with_source(temp_dir.path()).unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result.training_line, 1);
        assert_eq!(results[0].source_file.file_name().unwrap(), "file1.jsonl");
    }

    #[test]
    fn test_load_contamination_results_empty_directory() {
        let temp_dir = TempDir::new().unwrap();

        let results = load_contamination_results_from_directory(temp_dir.path()).unwrap();
        assert_eq!(results.len(), 0);
    }
}
