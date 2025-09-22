use crate::review::{
    filter_contamination_results_by_thresholds, load_contamination_results_from_directory,
    ContaminationResult,
};
use anyhow::{Error, Result};
use std::collections::{HashMap, HashSet};

use super::args::CompareArgs;
use super::display::{
    display_common, display_comparison_stats, display_only_in_first, display_only_in_second,
    ComparisonData,
};

// Key type for matching contamination instances across runs
pub type ContaminationKey = (String, usize, String, usize); // (training_file, training_line, eval_dataset, eval_line)

pub fn execute_compare(args: &CompareArgs) -> Result<(), Error> {
    let dir1 = args.dir1.as_path();
    let dir2 = args.dir2.as_path();
    let stats = args.stats;
    let common = args.common;
    let only_in_first = args.only_in_first;
    let only_in_second = args.only_in_second;
    let min_score = args.min_score;
    let eval_filter = args.eval.as_deref();
    let verbose = args.verbose;
    println!("=== CONTAMINATION COMPARISON ===");
    println!();

    if !dir1.exists() {
        return Err(anyhow::anyhow!("Directory not found: {:?}", dir1));
    }
    if !dir2.exists() {
        return Err(anyhow::anyhow!("Directory not found: {:?}", dir2));
    }

    println!("Loading results from first directory: {:?}", dir1);
    let mut results1 = load_contamination_results_from_directory(dir1)?;

    println!("Loading results from second directory: {:?}", dir2);
    let mut results2 = load_contamination_results_from_directory(dir2)?;

    println!();

    // Apply filters if specified
    let original_count1 = results1.len();
    let original_count2 = results2.len();

    results1 = filter_contamination_results_by_thresholds(
        results1,
        min_score,
        None, // min_length not supported in simplified version
        eval_filter,
    );

    results2 = filter_contamination_results_by_thresholds(
        results2,
        min_score,
        None,
        eval_filter,
    );

    // Build lookup maps for efficient comparison
    let map1: HashMap<ContaminationKey, ContaminationResult> = results1
        .into_iter()
        .map(|r| {
            let key = (
                r.training_file.clone(),
                r.training_line,
                r.eval_dataset.clone(),
                r.eval_line,
            );
            (key, r)
        })
        .collect();

    let map2: HashMap<ContaminationKey, ContaminationResult> = results2
        .into_iter()
        .map(|r| {
            let key = (
                r.training_file.clone(),
                r.training_line,
                r.eval_dataset.clone(),
                r.eval_line,
            );
            (key, r)
        })
        .collect();

    // Get key sets for comparison
    let keys1: HashSet<_> = map1.keys().cloned().collect();
    let keys2: HashSet<_> = map2.keys().cloned().collect();

    let only_in_1: HashSet<_> = keys1.difference(&keys2).cloned().collect();
    let only_in_2: HashSet<_> = keys2.difference(&keys1).cloned().collect();
    let in_both: HashSet<_> = keys1.intersection(&keys2).cloned().collect();

    // Default to stats mode if no specific mode is selected
    if !only_in_first && !only_in_second && !common {
        let comparison_data = ComparisonData {
            dir1,
            dir2,
            map1: &map1,
            map2: &map2,
            only_in_1: &only_in_1,
            only_in_2: &only_in_2,
            in_both: &in_both,
            original_count1,
            original_count2,
            min_score,
            eval_filter,
        };
        display_comparison_stats(&comparison_data)?;
    } else {
        if stats {
            let comparison_data = ComparisonData {
                dir1,
                dir2,
                map1: &map1,
                map2: &map2,
                only_in_1: &only_in_1,
                only_in_2: &only_in_2,
                in_both: &in_both,
                original_count1,
                original_count2,
                min_score,
                eval_filter,
            };
            display_comparison_stats(&comparison_data)?;
        }

        if common {
            display_common(&map1, &map2, &in_both, verbose)?;
        }

        if only_in_first {
            display_only_in_first(&map1, &map2)?;
        }

        if only_in_second {
            display_only_in_second(&map1, &map2)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::ContaminationResult;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn create_test_contamination_result(
        training_file: &str,
        training_line: usize,
        eval_dataset: &str,
        eval_line: usize,
        score: f32,
        fingerprint: Option<String>,
    ) -> ContaminationResult {
        ContaminationResult {
            training_file: training_file.to_string(),
            training_line,
            eval_dataset: eval_dataset.to_string(),
            eval_key: Some(eval_dataset.to_string()),
            eval_line,
            eval_instance_index: Some(eval_line * 10), // Simple mapping for tests
            split: Some("test".to_string()),
            method: Some("simple".to_string()),
            contamination_start_idx: None,
            contamination_end_idx: None,
            question_start_idx: None,
            question_end_idx: None,
            training_overlap_text: Some("test overlap text".to_string()),
            eval_overlap_text: Some("test eval text".to_string()),
            ngram_match_cnt: Some(10),
            eval_unique_ngrams: Some(20),
            contamination_score: Some(score),
            length_penalty: None,
            answer_overlap_ratio: None,
            answer_idf_overlap: None,
            answer_start_idx: None,
            answer_end_idx: None,
            passage_start_idx: None,
            passage_end_idx: None,
            idf_overlap: Some(score * 0.8),
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
            fingerprint,
            is_correct: None,
            reference_file: None,
        }
    }

    fn create_test_directory_with_results(results: Vec<ContaminationResult>) -> PathBuf {
        let dir = tempdir().unwrap();
        let dir_path = dir.path().to_path_buf();

        for (i, result) in results.iter().enumerate() {
            let file_path = dir_path.join(format!("result_{}.json", i));
            let mut file = File::create(&file_path).unwrap();
            let json = serde_json::to_string(&result).unwrap();
            writeln!(file, "{}", json).unwrap();
        }

        // Keep the temp directory alive by leaking it
        std::mem::forget(dir);
        dir_path
    }

    #[test]
    fn test_execute_compare_empty_directories() {
        let dir1 = tempdir().unwrap();
        let dir2 = tempdir().unwrap();

        let args = CompareArgs {
            dir1: dir1.path().to_path_buf(),
            dir2: dir2.path().to_path_buf(),
            stats: true,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: None,
            eval: None,
            verbose: false,
        };
        let result = execute_compare(&args);

        // Should succeed even with empty directories
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_compare_nonexistent_directory() {
        let valid_dir = tempdir().unwrap();
        let invalid_dir = PathBuf::from("/nonexistent/directory");

        let args = CompareArgs {
            dir1: invalid_dir,
            dir2: valid_dir.path().to_path_buf(),
            stats: true,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: None,
            eval: None,
            verbose: false,
        };
        let result = execute_compare(&args);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Directory not found"));
    }

    #[test]
    fn test_contamination_key_generation() {
        let results = vec![
            create_test_contamination_result("train1.jsonl", 10, "mmlu", 5, 0.8, None),
            create_test_contamination_result("train2.jsonl", 20, "gsm8k", 15, 0.6, None),
        ];

        let dir1 = create_test_directory_with_results(results.clone());
        let dir2 = create_test_directory_with_results(results);

        let args = CompareArgs {
            dir1,
            dir2,
            stats: false,
            common: true,
            only_in_first: false,
            only_in_second: false,
            min_score: None,
            eval: None,
            verbose: false,
        };
        let result = execute_compare(&args);

        assert!(result.is_ok());
    }

    #[test]
    fn test_min_score_filtering() {
        let results1 = vec![
            create_test_contamination_result("train1.jsonl", 10, "mmlu", 5, 0.3, None),
            create_test_contamination_result("train2.jsonl", 20, "mmlu", 15, 0.8, None),
        ];

        let results2 = vec![
            create_test_contamination_result("train1.jsonl", 10, "mmlu", 5, 0.3, None),
            create_test_contamination_result("train2.jsonl", 20, "mmlu", 15, 0.8, None),
        ];

        let dir1 = create_test_directory_with_results(results1);
        let dir2 = create_test_directory_with_results(results2);

        // With min_score = 0.5, should only keep the 0.8 score result
        let args = CompareArgs {
            dir1,
            dir2,
            stats: true,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: Some(0.5),
            eval: None,
            verbose: false,
        };
        let result = execute_compare(&args);

        assert!(result.is_ok());
    }

    #[test]
    fn test_eval_filter() {
        let results1 = vec![
            create_test_contamination_result("train1.jsonl", 10, "mmlu", 5, 0.8, None),
            create_test_contamination_result("train2.jsonl", 20, "gsm8k", 15, 0.6, None),
            create_test_contamination_result("train3.jsonl", 30, "mmlu", 25, 0.7, None),
        ];

        let dir1 = create_test_directory_with_results(results1);
        let dir2 = create_test_directory_with_results(vec![]);

        // Filter for only "mmlu" results
        let args = CompareArgs {
            dir1,
            dir2,
            stats: true,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: None,
            eval: Some("mmlu".to_string()),
            verbose: false,
        };
        let result = execute_compare(&args);

        assert!(result.is_ok());
    }

    #[test]
    fn test_only_in_first_detection() {
        let results1 = vec![
            create_test_contamination_result("train1.jsonl", 10, "mmlu", 5, 0.8,
                Some("fingerprint1".to_string())),
            create_test_contamination_result("train2.jsonl", 20, "gsm8k", 15, 0.6,
                Some("fingerprint2".to_string())),
        ];

        let results2 = vec![
            create_test_contamination_result("train1.jsonl", 10, "mmlu", 5, 0.8,
                Some("fingerprint1".to_string())),
        ];

        let dir1 = create_test_directory_with_results(results1);
        let dir2 = create_test_directory_with_results(results2);

        // Should find fingerprint2 only in first
        let args = CompareArgs {
            dir1,
            dir2,
            stats: false,
            common: false,
            only_in_first: true,
            only_in_second: false,
            min_score: None,
            eval: None,
            verbose: false,
        };
        let result = execute_compare(&args);

        assert!(result.is_ok());
    }

    #[test]
    fn test_only_in_second_detection() {
        let results1 = vec![
            create_test_contamination_result("train1.jsonl", 10, "mmlu", 5, 0.8,
                Some("fingerprint1".to_string())),
        ];

        let results2 = vec![
            create_test_contamination_result("train1.jsonl", 10, "mmlu", 5, 0.8,
                Some("fingerprint1".to_string())),
            create_test_contamination_result("train3.jsonl", 30, "humaneval", 25, 0.9,
                Some("fingerprint3".to_string())),
        ];

        let dir1 = create_test_directory_with_results(results1);
        let dir2 = create_test_directory_with_results(results2);

        // Should find fingerprint3 only in second
        let args = CompareArgs {
            dir1,
            dir2,
            stats: false,
            common: false,
            only_in_first: false,
            only_in_second: true,
            min_score: None,
            eval: None,
            verbose: false,
        };
        let result = execute_compare(&args);

        assert!(result.is_ok());
    }

    #[test]
    fn test_default_stats_mode() {
        let results = vec![
            create_test_contamination_result("train1.jsonl", 10, "mmlu", 5, 0.8, None),
        ];

        let dir1 = create_test_directory_with_results(results.clone());
        let dir2 = create_test_directory_with_results(results);

        // When no specific mode is selected, should default to stats
        let args = CompareArgs {
            dir1,
            dir2,
            stats: false,
            common: false,
            only_in_first: false,
            only_in_second: false,
            min_score: None,
            eval: None,
            verbose: false,
        };
        let result = execute_compare(&args);

        assert!(result.is_ok());
    }
}
