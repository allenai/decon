use crate::review::ContaminationResultWithSource;
use anyhow::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Display dataset counts showing unique removals per dataset
pub fn display_dataset_counts(
    contamination_results_with_source: &[ContaminationResultWithSource],
    base_dir: &Path,
) -> Result<(), Error> {
    // Track unique removals per dataset
    // Key: dataset name, Value: set of (training_file, training_line) pairs
    let mut dataset_removals: HashMap<String, HashSet<(String, usize)>> = HashMap::new();

    for result_with_source in contamination_results_with_source {
        // Extract dataset name from the report file path
        let dataset_name = extract_dataset_name_from_report_path(&result_with_source.source_file, base_dir);

        // Add this unique removal to the dataset's set
        let removal_key = (
            result_with_source.result.training_file.clone(),
            result_with_source.result.training_line
        );
        dataset_removals
            .entry(dataset_name)
            .or_default()
            .insert(removal_key);
    }

    // Convert to counts and sort by count (descending)
    let mut sorted_counts: Vec<(String, usize)> = dataset_removals
        .into_iter()
        .map(|(dataset, removals)| (dataset, removals.len()))
        .collect();
    sorted_counts.sort_by(|a, b| b.1.cmp(&a.1));

    let total_unique_removals: usize = sorted_counts.iter().map(|(_, count)| count).sum();

    println!("=== TRAINING DATASET COUNTS ===");
    println!();
    println!("Total unique removals: {}", total_unique_removals);
    println!("Unique datasets: {}", sorted_counts.len());
    println!();

    // Display each dataset with count
    for (dataset, count) in &sorted_counts {
        println!("{}\t{}", count, dataset);
    }

    Ok(())
}

/// Helper function to extract dataset name from report file path
fn extract_dataset_name_from_report_path(report_file: &Path, base_dir: &Path) -> String {
    // Try to strip the base directory prefix
    if let Ok(relative_path) = report_file.strip_prefix(base_dir) {
        // Get the first component (top-level directory)
        if let Some(first_component) = relative_path.components().next()
            && let Some(dataset_name) = first_component.as_os_str().to_str() {
                return dataset_name.to_string();
            }
    }

    // Fallback: use "unknown" if we can't determine the dataset
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::ContaminationResult;
    use std::path::PathBuf;

    fn create_test_result_with_source(
        training_file: &str,
        training_line: usize,
        source_file: &str,
    ) -> ContaminationResultWithSource {
        let result = ContaminationResult {
            training_file: training_file.to_string(),
            training_line,
            eval_dataset: "mmlu".to_string(),
            eval_key: None,
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
            ngram_match_cnt: None,
            eval_unique_ngrams: None,
            contamination_score: None,
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
        };

        ContaminationResultWithSource {
            result,
            source_file: PathBuf::from(source_file),
        }
    }

    #[test]
    fn test_extract_dataset_name_simple() {
        let base_dir = PathBuf::from("/reports");
        let report_file = PathBuf::from("/reports/dataset1/results.jsonl");

        let name = extract_dataset_name_from_report_path(&report_file, &base_dir);
        assert_eq!(name, "dataset1");
    }

    #[test]
    fn test_extract_dataset_name_nested() {
        let base_dir = PathBuf::from("/reports");
        let report_file = PathBuf::from("/reports/dataset2/subfolder/results.jsonl");

        let name = extract_dataset_name_from_report_path(&report_file, &base_dir);
        assert_eq!(name, "dataset2");
    }

    #[test]
    fn test_extract_dataset_name_unknown() {
        let base_dir = PathBuf::from("/reports");
        let report_file = PathBuf::from("/other/path/results.jsonl");

        let name = extract_dataset_name_from_report_path(&report_file, &base_dir);
        assert_eq!(name, "unknown");
    }

    #[test]
    fn test_extract_dataset_name_direct_file() {
        let base_dir = PathBuf::from("/reports");
        let report_file = PathBuf::from("/reports/results.jsonl");

        let name = extract_dataset_name_from_report_path(&report_file, &base_dir);
        // A direct file in the base directory will return the filename itself
        assert_eq!(name, "results.jsonl");
    }

    #[test]
    fn test_dataset_counts_unique_removals() {
        let base_dir = PathBuf::from("/reports");

        let results = vec![
            create_test_result_with_source("file1.txt", 10, "/reports/dataset1/results.jsonl"),
            create_test_result_with_source("file1.txt", 10, "/reports/dataset1/results.jsonl"),
            create_test_result_with_source("file2.txt", 20, "/reports/dataset1/results.jsonl"),
            create_test_result_with_source("file3.txt", 30, "/reports/dataset2/results.jsonl"),
        ];

        // Can't easily test the display function itself due to I/O,
        // but we can test the core logic indirectly
        let mut dataset_removals: HashMap<String, HashSet<(String, usize)>> = HashMap::new();

        for result_with_source in &results {
            let dataset_name = extract_dataset_name_from_report_path(&result_with_source.source_file, &base_dir);
            let removal_key = (
                result_with_source.result.training_file.clone(),
                result_with_source.result.training_line
            );
            dataset_removals
                .entry(dataset_name)
                .or_default()
                .insert(removal_key);
        }

        assert_eq!(dataset_removals.len(), 2);
        assert_eq!(dataset_removals.get("dataset1").unwrap().len(), 2);
        assert_eq!(dataset_removals.get("dataset2").unwrap().len(), 1);
    }

    #[test]
    fn test_dataset_counts_empty_results() {
        let base_dir = PathBuf::from("/reports");
        let results: Vec<ContaminationResultWithSource> = vec![];

        let mut dataset_removals: HashMap<String, HashSet<(String, usize)>> = HashMap::new();

        for result_with_source in &results {
            let dataset_name = extract_dataset_name_from_report_path(&result_with_source.source_file, &base_dir);
            let removal_key = (
                result_with_source.result.training_file.clone(),
                result_with_source.result.training_line
            );
            dataset_removals
                .entry(dataset_name)
                .or_default()
                .insert(removal_key);
        }

        assert_eq!(dataset_removals.len(), 0);
    }
}
